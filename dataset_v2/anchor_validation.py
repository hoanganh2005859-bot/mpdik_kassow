"""Independent validator for Dataset v2 anchors (Phase 4).

Reuses ``kinematics/`` (FK, Jacobian, singularity metrics, joint-limit margin, manipulability)
unchanged for every recomputation below -- this module never reimplements kinematics math, and
never calls DLS/any IK solver to decide whether an anchor catalog is valid (spec section 10).
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from dataset_v2.anchor_generation import (
    ANCHOR_CLASS_IDS,
    ANCHOR_CLASS_TOTAL_COUNTS,
    NPZ_NAME,
    SPLIT_IDS,
)
from dataset_v2.config_templates import ANCHOR_CLASS_PRIORITY_HIGHEST_FIRST, SPLITS
from dataset_v2.locator import require_dataset_v2_root
from kinematics.forward_kinematics import forward_kinematics
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import normalized_joint_limit_margin
from kinematics.model_loader import ModelContext, load_model_context
from kinematics.rotation_utils import rotation_geodesic_angle
from kinematics.singularity_metrics import condition_number, numerical_rank, singular_values
from utils.config_loader import load_json_config
from utils.npz_utils import load_npz

FK_POSITION_TOLERANCE_M = 1e-6
FK_ORIENTATION_TOLERANCE_RAD = 1e-6
COVARIATE_TOLERANCE = 1e-6
QUATERNION_NORM_TOLERANCE = 1e-6


@dataclass
class AnchorValidationReport:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    total_anchors: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)
    class_split_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _classify_single(normalized_margin, sigma_min, near_joint_limit_threshold, near_singularity_threshold, moderately_conditioned_upper_bound) -> str:
    is_near_singular = sigma_min <= near_singularity_threshold
    is_near_limit = normalized_margin <= near_joint_limit_threshold
    is_regular = (sigma_min > moderately_conditioned_upper_bound) and (normalized_margin > near_joint_limit_threshold)
    eligibility = {"near_singular": is_near_singular, "near_limit": is_near_limit, "regular": is_regular}
    for name in ANCHOR_CLASS_PRIORITY_HIGHEST_FIRST:
        if eligibility[name]:
            return name
    return "none"


def validate_anchors(
    dataset_root,
    model_context: Optional[ModelContext] = None,
    full_counts: bool = True,
) -> AnchorValidationReport:
    """Validate the on-disk anchor catalog under ``dataset_root``.

    ``full_counts=True`` (default) checks against the locked 12/6-3-3/4-4-4/2-1-1 counts.
    """
    paths = require_dataset_v2_root(dataset_root)
    anchors_dir = paths.anchors_dir
    model_context = model_context if model_context is not None else load_model_context()

    reasons: List[str] = []

    npz_path = anchors_dir / NPZ_NAME
    if not npz_path.is_file():
        raise FileNotFoundError(f"Anchor v2 output not found: {npz_path}")
    arrays = load_npz(npz_path)

    n = arrays["anchor_id"].shape[0]

    # --- shapes/dtypes/finiteness ---
    expectations = {"q": (n, 7), "position": (n, 3), "quaternion_wxyz": (n, 4)}
    for name, expected_shape in expectations.items():
        if arrays[name].shape != expected_shape:
            reasons.append(f"array '{name}' has shape {arrays[name].shape}, expected {expected_shape}")
    for name, arr in arrays.items():
        if arr.dtype == object:
            reasons.append(f"array '{name}' has object dtype (forbidden)")
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            reasons.append(f"array '{name}' contains non-finite values")

    # --- operational limits + quaternion normalization ---
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    if np.any(arrays["q"] < lower) or np.any(arrays["q"] > upper):
        reasons.append("one or more anchor 'q' rows violate operational joint limits")
    norms = np.linalg.norm(arrays["quaternion_wxyz"], axis=1)
    if np.any(np.abs(norms - 1.0) > QUATERNION_NORM_TOLERANCE):
        reasons.append("array 'quaternion_wxyz' contains non-unit quaternion(s)")

    # --- counts ---
    class_counts = {name: int(np.sum(arrays["anchor_class_id"] == ANCHOR_CLASS_IDS[name])) for name in ANCHOR_CLASS_IDS}
    split_counts = {split_name: int(np.sum(arrays["split"] == split_name)) for split_name in SPLITS}
    class_split_counts = {
        name: {
            split_name: int(np.sum((arrays["anchor_class_id"] == ANCHOR_CLASS_IDS[name]) & (arrays["split"] == split_name)))
            for split_name in SPLITS
        }
        for name in ANCHOR_CLASS_IDS
    }

    if full_counts:
        if n != 12:
            reasons.append(f"expected 12 total anchors, found {n}")
        for name, expected in ANCHOR_CLASS_TOTAL_COUNTS.items():
            if class_counts[name] != expected:
                reasons.append(f"anchor class '{name}' has {class_counts[name]} anchor(s), expected {expected}")
        for split_name in SPLITS:
            if split_counts[split_name] != 4:
                reasons.append(f"split '{split_name}' has {split_counts[split_name]} anchor(s), expected 4")
        expected_per_class_split = {"regular": 2, "near_limit": 1, "near_singular": 1}
        for name, expected in expected_per_class_split.items():
            for split_name in SPLITS:
                actual = class_split_counts[name][split_name]
                if actual != expected:
                    reasons.append(f"class '{name}' split '{split_name}' has {actual} anchor(s), expected {expected}")

    # --- uniqueness ---
    anchor_ids = [str(s) for s in arrays["anchor_id"]]
    content_hashes = [str(s) for s in arrays["content_hash"]]
    if len(set(anchor_ids)) != len(anchor_ids):
        reasons.append(f"duplicate anchor_id found ({len(anchor_ids) - len(set(anchor_ids))} collision(s))")
    if len(set(content_hashes)) != len(content_hashes):
        reasons.append(f"duplicate content_hash found ({len(content_hashes) - len(set(content_hashes))} collision(s))")
    q_keys = [arrays["q"][i].tobytes() for i in range(n)]
    if len(set(q_keys)) != len(q_keys):
        reasons.append(f"duplicate exact q found ({len(q_keys) - len(set(q_keys))} collision(s))")

    # --- FK/covariate/classification recomputation ---
    difficulty_thresholds_path = paths.configs_dir / "difficulty_thresholds.json"
    thresholds = load_json_config(difficulty_thresholds_path)
    near_joint_limit_threshold = float(thresholds["near_joint_limit"]["threshold_normalized"])
    near_singularity_threshold = float(thresholds["near_singularity"]["threshold_sigma_min"])
    moderately_conditioned_upper_bound = float(thresholds["moderately_conditioned"]["upper_bound_sigma_min"])

    data = model_context.new_data()
    covariate_mismatches = 0
    classification_mismatches = 0
    flag_mismatches = 0
    fk_position_errors = np.empty(n)
    fk_orientation_errors = np.empty(n)

    for i in range(n):
        q = arrays["q"][i]
        fk = forward_kinematics(model_context, q, data=data)
        J = geometric_jacobian_world(model_context, q, data=data)
        sv = singular_values(J)
        sigma_min = float(sv[-1])
        sigma_max = float(sv[0])
        cond = condition_number(J)
        rank = numerical_rank(J)

        fk_position_errors[i] = np.linalg.norm(fk.position - arrays["position"][i])
        stored_rotation = _quaternion_to_matrix(arrays["quaternion_wxyz"][i])
        fk_orientation_errors[i] = rotation_geodesic_angle(fk.rotation_matrix, stored_rotation)

        per_joint_normalized = normalized_joint_limit_margin(q, lower, upper)
        per_joint_absolute = np.minimum(q - lower, upper - q)
        controlling_joint = int(np.argmin(per_joint_normalized))
        normalized_margin = float(per_joint_normalized[controlling_joint])
        absolute_margin = float(per_joint_absolute[controlling_joint])

        if (
            abs(sigma_min - arrays["sigma_min"][i]) > COVARIATE_TOLERANCE
            or abs(sigma_max - arrays["sigma_max"][i]) > COVARIATE_TOLERANCE
            or abs(normalized_margin - arrays["minimum_normalized_limit_margin"][i]) > COVARIATE_TOLERANCE
            or abs(absolute_margin - arrays["minimum_absolute_limit_margin_rad"][i]) > COVARIATE_TOLERANCE
            or rank != int(arrays["numerical_rank"][i])
            or controlling_joint != int(arrays["controlling_joint_index"][i])
        ):
            covariate_mismatches += 1

        is_near_singular = sigma_min <= near_singularity_threshold
        is_near_limit = normalized_margin <= near_joint_limit_threshold
        is_moderately_conditioned = (sigma_min > near_singularity_threshold) and (sigma_min <= moderately_conditioned_upper_bound)
        is_regular = (sigma_min > moderately_conditioned_upper_bound) and (normalized_margin > near_joint_limit_threshold)

        if (
            is_near_singular != bool(arrays["is_near_singular"][i])
            or is_near_limit != bool(arrays["is_near_limit"][i])
            or is_moderately_conditioned != bool(arrays["is_moderately_conditioned"][i])
            or is_regular != bool(arrays["is_regular"][i])
        ):
            flag_mismatches += 1

        recomputed_class = _classify_single(
            normalized_margin, sigma_min, near_joint_limit_threshold, near_singularity_threshold, moderately_conditioned_upper_bound
        )
        stored_class_name = {v: k for k, v in ANCHOR_CLASS_IDS.items()}[int(arrays["anchor_class_id"][i])]
        # The stored anchor_class is which target list the anchor was *drawn for* (regular/
        # near_limit/near_singular candidate pool), which may legitimately differ from the
        # highest-priority eligible class when an overlap fallback was used (spec section 3);
        # only flag a mismatch when the anchor doesn't even satisfy its own stored class's raw
        # eligibility criterion.
        own_class_eligible = {
            "near_singular": is_near_singular,
            "near_limit": is_near_limit,
            "regular": is_regular,
        }[stored_class_name]
        if not own_class_eligible:
            classification_mismatches += 1

    if np.any(fk_position_errors > FK_POSITION_TOLERANCE_M):
        reasons.append(f"FK(q) does not match stored position for {int(np.sum(fk_position_errors > FK_POSITION_TOLERANCE_M))} anchor(s)")
    if np.any(fk_orientation_errors > FK_ORIENTATION_TOLERANCE_RAD):
        reasons.append(f"FK(q) does not match stored quaternion_wxyz for {int(np.sum(fk_orientation_errors > FK_ORIENTATION_TOLERANCE_RAD))} anchor(s)")
    if covariate_mismatches:
        reasons.append(f"{covariate_mismatches} anchor(s) have a recomputed covariate that does not match the stored value")
    if flag_mismatches:
        reasons.append(f"{flag_mismatches} anchor(s) have a recomputed diagnostic flag that does not match the stored value")
    if classification_mismatches:
        reasons.append(f"{classification_mismatches} anchor(s) do not satisfy their own stored anchor_class's eligibility criterion")

    # --- split anti-leakage / near-duplicate check ---
    anchor_config = load_json_config(paths.configs_dir / "anchor_config.json")
    tol = anchor_config["near_duplicate_tolerance"]
    for i in range(n):
        for j in range(i + 1, n):
            if str(arrays["split"][i]) == str(arrays["split"][j]):
                continue
            joint_dist = float(np.linalg.norm(arrays["q"][i] - arrays["q"][j]))
            pos_dist = float(np.linalg.norm(arrays["position"][i] - arrays["position"][j]))
            orient_dist = rotation_geodesic_angle(
                _quaternion_to_matrix(arrays["quaternion_wxyz"][i]), _quaternion_to_matrix(arrays["quaternion_wxyz"][j])
            )
            if joint_dist <= tol["joint_space_rad"] and pos_dist <= tol["position_m"] and orient_dist <= tol["orientation_rad"]:
                reasons.append(f"near-duplicate anchors {anchor_ids[i]}/{anchor_ids[j]} found in different splits")

    return AnchorValidationReport(
        passed=len(reasons) == 0,
        reasons=reasons,
        total_anchors=n,
        class_counts=class_counts,
        split_counts=split_counts,
        class_split_counts=class_split_counts,
    )


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    from kinematics.quaternion_utils import quaternion_wxyz_to_matrix

    return quaternion_wxyz_to_matrix(q)
