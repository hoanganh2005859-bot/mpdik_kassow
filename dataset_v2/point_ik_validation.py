"""Independent validator for Dataset v2 Tier 1 Point-IK samples.

Reuses ``kinematics/`` (FK, Jacobian, singularity metrics, joint-limit margin, SO(3) geodesic
angle) unchanged for every recomputation below -- this module never reimplements kinematics math,
and never calls DLS/any IK solver to decide whether data is valid (spec section 11).
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from dataset_v2.config_templates import CLASSIFICATION_PRIORITY_HIGHEST_FIRST, DIFFICULTY_GROUPS, SPLITS
from dataset_v2.locator import require_dataset_v2_root
from dataset_v2.point_ik_generation import (
    DIFFICULTY_GROUP_IDS,
    MIN_JOINT_DISTANCE_RAD,
    NPZ_NAMES,
    SPLIT_IDS,
)
from kinematics.forward_kinematics import forward_kinematics
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import minimum_joint_limit_margin
from kinematics.model_loader import ModelContext, load_model_context
from kinematics.rotation_utils import rotation_geodesic_angle
from kinematics.singularity_metrics import condition_number, singular_values
from utils.npz_utils import load_npz

# Numerical-reproducibility tolerances (recomputing a pure function from the exact same stored
# input must reproduce it almost exactly) -- not DLS/evaluation acceptance thresholds.
FK_POSITION_TOLERANCE_M = 1e-6
FK_ORIENTATION_TOLERANCE_RAD = 1e-6
COVARIATE_TOLERANCE = 1e-6
QUATERNION_NORM_TOLERANCE = 1e-6


@dataclass
class PointIKValidationReport:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    total_samples: int = 0
    group_counts: Dict[str, int] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)
    group_split_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _check_shapes_and_dtypes(arrays: Dict[str, np.ndarray], split_name: str, reasons: List[str]) -> None:
    n = arrays["sample_id"].shape[0]
    expectations = {
        "q_initial": (n, 7),
        "q_target_reference": (n, 7),
        "initial_position": (n, 3),
        "target_position": (n, 3),
        "initial_quaternion_wxyz": (n, 4),
        "target_quaternion_wxyz": (n, 4),
    }
    for name, expected_shape in expectations.items():
        if arrays[name].shape != expected_shape:
            reasons.append(f"{split_name}: array '{name}' has shape {arrays[name].shape}, expected {expected_shape}")
    for name, arr in arrays.items():
        if arr.dtype == object:
            reasons.append(f"{split_name}: array '{name}' has object dtype (forbidden)")
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            reasons.append(f"{split_name}: array '{name}' contains non-finite values")


def _check_limits_and_quaternions(arrays: Dict[str, np.ndarray], model_context: ModelContext, split_name: str, reasons: List[str]) -> None:
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    for name in ("q_initial", "q_target_reference"):
        q = arrays[name]
        if np.any(q < lower) or np.any(q > upper):
            reasons.append(f"{split_name}: one or more '{name}' rows violate operational joint limits")
    for name in ("initial_quaternion_wxyz", "target_quaternion_wxyz"):
        norms = np.linalg.norm(arrays[name], axis=1)
        if np.any(np.abs(norms - 1.0) > QUATERNION_NORM_TOLERANCE):
            reasons.append(f"{split_name}: array '{name}' contains non-unit quaternion(s)")


def _check_counts(all_arrays: Dict[str, Dict[str, np.ndarray]], full_counts: bool, expected_samples_per_group: Optional[int], reasons: List[str]):
    group_counts: Dict[str, int] = {name: 0 for name in DIFFICULTY_GROUPS}
    split_counts: Dict[str, int] = {}
    group_split_counts: Dict[str, Dict[str, int]] = {name: {} for name in DIFFICULTY_GROUPS}

    for split_name in SPLITS:
        arrays = all_arrays[split_name]
        split_counts[split_name] = int(arrays["sample_id"].shape[0])
        if int(np.any(arrays["split_id"] != SPLIT_IDS[split_name])):
            reasons.append(f"{split_name}: array 'split_id' contains a value other than {SPLIT_IDS[split_name]}")
        for name in DIFFICULTY_GROUPS:
            group_id = DIFFICULTY_GROUP_IDS[name]
            count = int(np.sum(arrays["difficulty_id"] == group_id))
            group_counts[name] += count
            group_split_counts[name][split_name] = count

    total_samples = sum(split_counts.values())

    if full_counts:
        if total_samples != 6000:
            reasons.append(f"expected 6000 total Point-IK samples, found {total_samples}")
        for name in DIFFICULTY_GROUPS:
            if group_counts[name] != 1000:
                reasons.append(f"difficulty group '{name}' has {group_counts[name]} sample(s), expected 1000")
        expected_split_totals = {"development": 1200, "validation": 1200, "frozen_test": 3600}
        for split_name, expected in expected_split_totals.items():
            if split_counts[split_name] != expected:
                reasons.append(f"split '{split_name}' has {split_counts[split_name]} sample(s), expected {expected}")
        expected_group_split = {"development": 200, "validation": 200, "frozen_test": 600}
        for name in DIFFICULTY_GROUPS:
            for split_name, expected in expected_group_split.items():
                if group_split_counts[name][split_name] != expected:
                    reasons.append(
                        f"group '{name}' split '{split_name}' has {group_split_counts[name][split_name]} sample(s), expected {expected}"
                    )
    elif expected_samples_per_group is not None:
        for name in DIFFICULTY_GROUPS:
            if group_counts[name] != expected_samples_per_group:
                reasons.append(f"difficulty group '{name}' has {group_counts[name]} sample(s), expected {expected_samples_per_group}")

    return total_samples, group_counts, split_counts, group_split_counts


def _check_uniqueness_and_leakage(all_arrays: Dict[str, Dict[str, np.ndarray]], reasons: List[str]) -> None:
    all_sample_ids: List[str] = []
    all_content_hashes: List[str] = []
    all_pair_keys: List[bytes] = []

    for split_name in SPLITS:
        arrays = all_arrays[split_name]
        all_sample_ids.extend(str(s) for s in arrays["sample_id"])
        all_content_hashes.extend(str(s) for s in arrays["content_hash"])
        q_i = arrays["q_initial"]
        q_t = arrays["q_target_reference"]
        for i in range(q_i.shape[0]):
            all_pair_keys.append(q_i[i].tobytes() + q_t[i].tobytes())

    if len(set(all_sample_ids)) != len(all_sample_ids):
        reasons.append(f"duplicate sample_id found ({len(all_sample_ids) - len(set(all_sample_ids))} collision(s))")
    if len(set(all_content_hashes)) != len(all_content_hashes):
        reasons.append(f"duplicate content_hash found ({len(all_content_hashes) - len(set(all_content_hashes))} collision(s))")
    if len(set(all_pair_keys)) != len(all_pair_keys):
        reasons.append(f"duplicate (q_initial, q_target_reference) pair found ({len(all_pair_keys) - len(set(all_pair_keys))} collision(s))")


def _recompute_and_check_fk_and_covariates(
    model_context: ModelContext, arrays: Dict[str, np.ndarray], split_name: str, thresholds: dict, reasons: List[str]
) -> None:
    data = model_context.new_data()
    n = arrays["sample_id"].shape[0]

    fk_position_errors = np.empty(n)
    fk_orientation_errors = np.empty(n)
    fk_target_position_errors = np.empty(n)
    fk_target_orientation_errors = np.empty(n)
    covariate_mismatches = 0
    classification_mismatches = 0

    for i in range(n):
        q_i = arrays["q_initial"][i]
        q_t = arrays["q_target_reference"][i]

        fk_i = forward_kinematics(model_context, q_i, data=data)
        fk_t = forward_kinematics(model_context, q_t, data=data)

        fk_position_errors[i] = np.linalg.norm(fk_i.position - arrays["initial_position"][i])
        fk_target_position_errors[i] = np.linalg.norm(fk_t.position - arrays["target_position"][i])

        stored_initial_rot = _quaternion_wxyz_to_matrix_local(arrays["initial_quaternion_wxyz"][i])
        stored_target_rot = _quaternion_wxyz_to_matrix_local(arrays["target_quaternion_wxyz"][i])
        fk_orientation_errors[i] = rotation_geodesic_angle(fk_i.rotation_matrix, stored_initial_rot)
        fk_target_orientation_errors[i] = rotation_geodesic_angle(fk_t.rotation_matrix, stored_target_rot)

        position_distance = float(np.linalg.norm(fk_t.position - fk_i.position))
        orientation_distance = rotation_geodesic_angle(fk_i.rotation_matrix, fk_t.rotation_matrix)
        joint_distance = float(np.linalg.norm(q_t - q_i))

        J_i = geometric_jacobian_world(model_context, q_i, data=data)
        J_t = geometric_jacobian_world(model_context, q_t, data=data)
        sv_i = singular_values(J_i)
        sv_t = singular_values(J_t)
        margin_i = minimum_joint_limit_margin(q_i, model_context.operational_lower_rad, model_context.operational_upper_rad)
        margin_t = minimum_joint_limit_margin(q_t, model_context.operational_lower_rad, model_context.operational_upper_rad)

        if (
            abs(position_distance - arrays["position_distance_m"][i]) > COVARIATE_TOLERANCE
            or abs(orientation_distance - arrays["orientation_distance_rad"][i]) > COVARIATE_TOLERANCE
            or abs(joint_distance - arrays["joint_distance_rad"][i]) > COVARIATE_TOLERANCE
            or abs(float(sv_i[-1]) - arrays["initial_sigma_min"][i]) > COVARIATE_TOLERANCE
            or abs(float(sv_t[-1]) - arrays["target_sigma_min"][i]) > COVARIATE_TOLERANCE
            or abs(margin_i - arrays["minimum_initial_limit_margin_normalized"][i]) > COVARIATE_TOLERANCE
            or abs(margin_t - arrays["minimum_target_limit_margin_normalized"][i]) > COVARIATE_TOLERANCE
        ):
            covariate_mismatches += 1

        pair_margin = min(margin_i, margin_t)
        pair_sigma_min = min(float(sv_i[-1]), float(sv_t[-1]))
        recomputed_group = _classify_single(
            position_distance, orientation_distance, joint_distance, pair_margin, pair_sigma_min, thresholds
        )
        if DIFFICULTY_GROUP_IDS[recomputed_group] != int(arrays["difficulty_id"][i]):
            classification_mismatches += 1

    if np.any(fk_position_errors > FK_POSITION_TOLERANCE_M):
        reasons.append(f"{split_name}: FK(q_initial) does not match stored initial_position for {int(np.sum(fk_position_errors > FK_POSITION_TOLERANCE_M))} sample(s)")
    if np.any(fk_orientation_errors > FK_ORIENTATION_TOLERANCE_RAD):
        reasons.append(f"{split_name}: FK(q_initial) does not match stored initial_quaternion_wxyz for {int(np.sum(fk_orientation_errors > FK_ORIENTATION_TOLERANCE_RAD))} sample(s)")
    if np.any(fk_target_position_errors > FK_POSITION_TOLERANCE_M):
        reasons.append(f"{split_name}: FK(q_target_reference) does not match stored target_position for {int(np.sum(fk_target_position_errors > FK_POSITION_TOLERANCE_M))} sample(s)")
    if np.any(fk_target_orientation_errors > FK_ORIENTATION_TOLERANCE_RAD):
        reasons.append(f"{split_name}: FK(q_target_reference) does not match stored target_quaternion_wxyz for {int(np.sum(fk_target_orientation_errors > FK_ORIENTATION_TOLERANCE_RAD))} sample(s)")
    if covariate_mismatches:
        reasons.append(f"{split_name}: {covariate_mismatches} sample(s) have a recomputed covariate that does not match the stored value")
    if classification_mismatches:
        reasons.append(f"{split_name}: {classification_mismatches} sample(s) have a recomputed difficulty classification that does not match the stored difficulty_id")


def _quaternion_wxyz_to_matrix_local(q: np.ndarray) -> np.ndarray:
    from kinematics.quaternion_utils import quaternion_wxyz_to_matrix

    return quaternion_wxyz_to_matrix(q)


def _classify_single(position_distance, orientation_distance, joint_distance, pair_margin, pair_sigma_min, thresholds) -> str:
    eligibility = {
        "near_singularity": pair_sigma_min <= thresholds["near_singularity_threshold"],
        "near_joint_limit": pair_margin <= thresholds["near_joint_limit_threshold"],
        "large_orientation_change": orientation_distance >= thresholds["quantile_thresholds"]["orientation_distance_rad_top_quantile"],
        "far_target": position_distance >= thresholds["quantile_thresholds"]["position_distance_m_high_quantile"],
        "medium_target": (position_distance > thresholds["quantile_thresholds"]["position_distance_m_low_quantile"])
        and (position_distance < thresholds["quantile_thresholds"]["position_distance_m_high_quantile"]),
        "near_target": (position_distance <= thresholds["quantile_thresholds"]["position_distance_m_low_quantile"])
        and (joint_distance > MIN_JOINT_DISTANCE_RAD),
    }
    for name in CLASSIFICATION_PRIORITY_HIGHEST_FIRST:
        if eligibility[name]:
            return name
    raise ValueError("candidate did not qualify for any difficulty group -- classification eligibility is not exhaustive")


def validate_point_ik(
    dataset_root,
    model_context: Optional[ModelContext] = None,
    full_counts: bool = True,
    expected_samples_per_group: Optional[int] = None,
) -> PointIKValidationReport:
    """Validate the on-disk Point-IK v2 NPZ files under ``dataset_root``.

    ``full_counts=True`` (default) checks against the locked 6000/1000-per-group/1200-1200-3600
    counts. Pass ``full_counts=False`` with ``expected_samples_per_group`` for reduced fixtures.
    """
    paths = require_dataset_v2_root(dataset_root)
    tier1_dir = paths.tier1_point_ik_dir
    model_context = model_context if model_context is not None else load_model_context()

    reasons: List[str] = []

    all_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name in SPLITS:
        path = tier1_dir / NPZ_NAMES[split_name]
        if not path.is_file():
            raise FileNotFoundError(f"Point-IK v2 output not found: {path}")
        all_arrays[split_name] = load_npz(path)

    for split_name in SPLITS:
        _check_shapes_and_dtypes(all_arrays[split_name], split_name, reasons)
        _check_limits_and_quaternions(all_arrays[split_name], model_context, split_name, reasons)

    total_samples, group_counts, split_counts, group_split_counts = _check_counts(
        all_arrays, full_counts, expected_samples_per_group, reasons
    )
    _check_uniqueness_and_leakage(all_arrays, reasons)

    difficulty_definition_path = tier1_dir / "difficulty_definition.json"
    if difficulty_definition_path.is_file():
        difficulty_definition = json.loads(difficulty_definition_path.read_text(encoding="utf-8"))
        thresholds = {
            "near_joint_limit_threshold": difficulty_definition["near_joint_limit_threshold"],
            "near_singularity_threshold": difficulty_definition["near_singularity_threshold"],
            "quantile_thresholds": difficulty_definition["quantile_thresholds"],
        }
        for split_name in SPLITS:
            _recompute_and_check_fk_and_covariates(model_context, all_arrays[split_name], split_name, thresholds, reasons)
    else:
        reasons.append(f"missing {difficulty_definition_path}; cannot recompute/verify difficulty classification")

    return PointIKValidationReport(
        passed=len(reasons) == 0,
        reasons=reasons,
        total_samples=total_samples,
        group_counts=group_counts,
        split_counts=split_counts,
        group_split_counts=group_split_counts,
    )
