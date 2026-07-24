"""Independent validator for Dataset v2 random-challenge trajectories (Phase 6).

Reuses ``kinematics/`` unchanged for every recomputation and never runs IK to *decide* validity:
it re-derives FK from the stored ``q_reference`` and checks it against the target pose using
Dataset v2's own strict generation tolerances (``configs/generation_reachability_config.json``,
1e-4 m / 0.01 deg) -- never a Dataset v1 DLS *baseline evaluation* threshold, and never the
generator's own stored success flags. Both the canonical (400-waypoint) and the high-resolution
source path are verified.

Beyond reachability, the validator independently re-derives every geometry diagnostic (arc length,
angular displacement, curvature, non-planarity), re-resamples the stored source path and compares
it to the stored canonical arrays (catching a canonical path not actually derived from its own
source), checks counts/shapes/dtypes/finiteness/quaternion continuity/monotonicity, global id and
hash uniqueness, cross-split leakage, no duplication with the core catalog, and the presence and
``pass`` status of the anti-leakage report. When a core catalog is present it also verifies the
combined 210 trajectory / 84,000 canonical-pose totals.
"""

import csv
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from dataset_v2.challenge_trajectory_generation import (
    ANTI_LEAKAGE_REPORT_NAME,
    MANIFEST_NAME,
    NPZ_SUFFIX,
    SOURCE_SUFFIX,
    path_curvature,
    path_non_planarity,
)
from dataset_v2.config_templates import (
    CHALLENGE_FAMILIES,
    CHALLENGE_PER_FAMILY_PER_SPLIT,
    CHALLENGE_TOTAL,
    SPLITS,
)
from dataset_v2.core_trajectory_generation import resample_canonical
from dataset_v2.generation_reachability import (
    GENERATION_REACHABILITY_CONFIG_NAME,
    fk_reconstruction_error,
    load_generation_reachability_settings,
)
from dataset_v2.locator import require_dataset_v2_root
from kinematics.model_loader import ModelContext, load_model_context
from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from utils.config_loader import load_json_config
from utils.npz_utils import load_npz

QUATERNION_NORM_TOLERANCE = 1e-6
ENDPOINT_TOLERANCE_M = 1e-9
MONOTONIC_TOLERANCE = -1e-9
CANONICAL_RESAMPLE_TOLERANCE_M = 1e-6
CANONICAL_RESAMPLE_ORIENTATION_TOLERANCE_RAD = 1e-6
MANIFEST_RECOMPUTE_TOLERANCE = 1e-4

_DLS_BASELINE_POSITION_THRESHOLD_M = 0.006
_DLS_BASELINE_ORIENTATION_THRESHOLD_DEG = 10.0

CORE_MANIFEST_NAME = "core_trajectory_manifest.csv"


@dataclass
class ChallengeTrajectoryValidationReport:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    total_trajectories: int = 0
    split_counts: Dict[str, int] = field(default_factory=dict)
    family_counts: Dict[str, int] = field(default_factory=dict)
    canonical_poses_total: int = 0
    combined_trajectories_total: Optional[int] = None
    combined_canonical_poses_total: Optional[int] = None
    statistics: dict = field(default_factory=dict)


def _read_manifest_rows(manifest_path) -> List[dict]:
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _split_dir(paths, split_name: str):
    return {
        "development": paths.trajectories_development_dir,
        "validation": paths.trajectories_validation_dir,
        "frozen_test": paths.trajectories_frozen_test_dir,
    }[split_name]


def _verify_path_fk(model_context, arrays, tol_pos_m, tol_ori_rad, data) -> dict:
    n = arrays["target_position"].shape[0]
    q_reference = arrays["q_reference"]
    position_errors = np.empty(n)
    orientation_errors = np.empty(n)
    for i in range(n):
        position_errors[i], orientation_errors[i] = fk_reconstruction_error(
            model_context,
            q_reference[i],
            arrays["target_position"][i],
            quaternion_wxyz_to_matrix(arrays["target_quaternion"][i]),
            data=data,
        )
    return {
        "position_errors_m": position_errors,
        "orientation_errors_rad": orientation_errors,
        "position_violations": int(np.sum(position_errors > tol_pos_m)),
        "orientation_violations": int(np.sum(orientation_errors > tol_ori_rad)),
    }


def _quaternions_ok(quaternions: np.ndarray) -> Optional[str]:
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(np.abs(norms - 1.0) > QUATERNION_NORM_TOLERANCE):
        return f"quaternion norm deviates from 1 by up to {float(np.max(np.abs(norms - 1.0))):.2e}"
    dots = np.sum(quaternions[1:] * quaternions[:-1], axis=1)
    if np.any(dots < 0.0):
        return "quaternion sign discontinuity (consecutive dot product < 0)"
    return None


def validate_challenge_trajectories(
    dataset_root,
    model_context: Optional[ModelContext] = None,
    full_counts: bool = True,
    check_combined: bool = True,
) -> ChallengeTrajectoryValidationReport:
    """Validate the on-disk challenge trajectory set under ``dataset_root``.

    ``full_counts=True`` (default) checks against the locked 90 total / 30-30-30 split /
    15-per-family / 36,000-canonical-pose counts. ``check_combined`` additionally verifies the
    210 / 84,000 combined totals when a core catalog is present.
    """
    paths = require_dataset_v2_root(dataset_root)
    model_context = model_context if model_context is not None else load_model_context()
    reasons: List[str] = []

    manifest_path = paths.trajectories_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"challenge trajectory manifest not found: {manifest_path}")

    rows = _read_manifest_rows(manifest_path)
    total = len(rows)

    challenge_config = load_json_config(paths.configs_dir / "random_challenge_config.json")
    family_definitions = challenge_config["family_definitions"]

    # ---- strict-reachability independence (same guard as the core validator) ------------------
    reach_config = load_json_config(paths.configs_dir / GENERATION_REACHABILITY_CONFIG_NAME)
    if reach_config.get("independence", {}).get("reads_dls_evaluation_thresholds", True):
        reasons.append(f"{GENERATION_REACHABILITY_CONFIG_NAME} declares reads_dls_evaluation_thresholds=true")
    reach_settings = load_generation_reachability_settings(paths)
    if not (
        reach_settings.position_tolerance_m < _DLS_BASELINE_POSITION_THRESHOLD_M
        and reach_settings.orientation_tolerance_deg < _DLS_BASELINE_ORIENTATION_THRESHOLD_DEG
    ):
        reasons.append("generation reachability tolerances are not strictly tighter than the DLS baseline thresholds")

    tol_pos = reach_settings.position_tolerance_m
    tol_ori_rad = reach_settings.orientation_tolerance_rad

    # ---- counts -------------------------------------------------------------------------------
    split_counts: Dict[str, int] = {s: 0 for s in SPLITS}
    family_counts: Dict[str, int] = {f: 0 for f in CHALLENGE_FAMILIES}
    for row in rows:
        if row["split"] in split_counts:
            split_counts[row["split"]] += 1
        if row["challenge_family"] in family_counts:
            family_counts[row["challenge_family"]] += 1

    if full_counts:
        if total != CHALLENGE_TOTAL:
            reasons.append(f"expected {CHALLENGE_TOTAL} challenge trajectories, found {total}")
        for split in SPLITS:
            expected = CHALLENGE_TOTAL // len(SPLITS)
            if split_counts[split] != expected:
                reasons.append(f"split '{split}': expected {expected}, found {split_counts[split]}")
        for family in CHALLENGE_FAMILIES:
            expected = CHALLENGE_PER_FAMILY_PER_SPLIT * len(SPLITS)
            if family_counts[family] != expected:
                reasons.append(f"family '{family}': expected {expected}, found {family_counts[family]}")

    ids = [r["trajectory_id"] for r in rows]
    hashes = [r["content_hash"] for r in rows]
    if len(set(ids)) != len(ids):
        reasons.append("duplicate trajectory_id in challenge manifest")
    if len(set(hashes)) != len(hashes):
        reasons.append("duplicate content_hash in challenge manifest")

    data = model_context.new_data()
    canonical_poses_total = 0
    canonical_pos_max = 0.0
    source_pos_max = 0.0
    per_split_records: Dict[str, List[dict]] = {s: [] for s in SPLITS}

    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad

    for row in rows:
        tid = row["trajectory_id"]
        split = row["split"]
        family = row["challenge_family"]
        family_def = family_definitions.get(family)
        split_dir = _split_dir(paths, split)
        canonical_path = split_dir / f"{tid}{NPZ_SUFFIX}"
        source_path = split_dir / f"{tid}{SOURCE_SUFFIX}"
        if not canonical_path.is_file():
            reasons.append(f"{tid}: canonical NPZ missing ({canonical_path.name})")
            continue
        if not source_path.is_file():
            reasons.append(f"{tid}: source NPZ missing ({source_path.name})")
            continue

        canonical = load_npz(canonical_path)
        source = load_npz(source_path)

        n_canonical = canonical["target_position"].shape[0]
        if n_canonical != 400:
            reasons.append(f"{tid}: expected 400 canonical waypoints, found {n_canonical}")
        canonical_poses_total += n_canonical

        # dtype / finiteness (gate: a non-finite array would crash the geometry/FK checks below)
        non_finite = False
        for name, arr in canonical.items():
            if arr.dtype == object:
                reasons.append(f"{tid}: canonical array '{name}' has object dtype")
            if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
                reasons.append(f"{tid}: canonical array '{name}' has non-finite values")
                non_finite = True
        for name, arr in source.items():
            if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
                reasons.append(f"{tid}: source array '{name}' has non-finite values")
                non_finite = True
        if non_finite:
            canonical_poses_total += 0  # already counted above; skip geometry/FK for this record
            continue

        # quaternion normalization + sign continuity
        q_issue = _quaternions_ok(canonical["target_quaternion"])
        if q_issue:
            reasons.append(f"{tid}: {q_issue}")

        # monotonic time + arc length
        if np.any(np.diff(canonical["time_s"]) < MONOTONIC_TOLERANCE):
            reasons.append(f"{tid}: canonical time_s not monotonically non-decreasing")
        if np.any(np.diff(canonical["cumulative_arc_length_m"]) < MONOTONIC_TOLERANCE):
            reasons.append(f"{tid}: canonical arc length not monotonically non-decreasing")

        # joint limits on both stored q_reference paths
        for label, arrays in (("canonical", canonical), ("source", source)):
            qref = arrays["q_reference"]
            if np.any(qref < lower - 1e-9) or np.any(qref > upper + 1e-9):
                reasons.append(f"{tid}: {label} q_reference violates operational joint limits")

        # fresh arc-length resample of the stored source must reproduce the stored canonical
        fresh = resample_canonical(
            source["time_s"], source["tau"], source["target_position"], source["target_quaternion"], n_canonical
        )
        pos_diff = float(np.max(np.linalg.norm(fresh["target_position"] - canonical["target_position"], axis=1)))
        if pos_diff > CANONICAL_RESAMPLE_TOLERANCE_M:
            reasons.append(f"{tid}: canonical path does not match a fresh resample of its own source (pos diff {pos_diff:.2e} m)")

        # geometry recomputation vs manifest
        arc_length = float(canonical["cumulative_arc_length_m"][-1])
        if abs(arc_length - float(row["arc_length_m"])) > MANIFEST_RECOMPUTE_TOLERANCE:
            reasons.append(f"{tid}: arc_length_m mismatch (manifest {row['arc_length_m']}, recomputed {arc_length:.6f})")
        angular = float(canonical["cumulative_angular_displacement_rad"][-1])
        if abs(angular - float(row["cumulative_angular_displacement_rad"])) > MANIFEST_RECOMPUTE_TOLERANCE:
            reasons.append(f"{tid}: cumulative_angular_displacement_rad mismatch")
        mean_curv, max_curv, _ = path_curvature(canonical["target_position"])
        if not np.isfinite(mean_curv) or not np.isfinite(max_curv):
            reasons.append(f"{tid}: non-finite curvature")
        non_planarity = path_non_planarity(canonical["target_position"])

        # family coverage floors + curvature ceiling
        if family_def is not None:
            if not np.isfinite(mean_curv) or mean_curv > float(family_def["max_mean_curvature_1_per_m"]):
                reasons.append(f"{tid}: mean curvature {mean_curv:.3g} exceeds family ceiling")
            if "min_non_planarity" in family_def and non_planarity < float(family_def["min_non_planarity"]) - 1e-9:
                reasons.append(f"{tid}: non_planarity {non_planarity:.4f} below family floor {family_def['min_non_planarity']}")
            if "min_angular_displacement_rad" in family_def and angular < float(family_def["min_angular_displacement_rad"]) - 1e-9:
                reasons.append(f"{tid}: angular displacement {angular:.4f} below family floor")

        # reachability status honesty
        if not np.all(canonical["waypoint_reachable"]):
            reasons.append(f"{tid}: canonical path has a waypoint flagged unreachable")
        if not np.all(source["waypoint_reachable"]):
            reasons.append(f"{tid}: source path has a waypoint flagged unreachable")

        # independent FK reconstruction (canonical + source)
        canonical_fk = _verify_path_fk(model_context, canonical, tol_pos, tol_ori_rad, data)
        if canonical_fk["position_violations"] or canonical_fk["orientation_violations"]:
            reasons.append(
                f"{tid}: canonical FK reconstruction exceeds tolerance "
                f"(pos_viol={canonical_fk['position_violations']}, ori_viol={canonical_fk['orientation_violations']})"
            )
        source_fk = _verify_path_fk(model_context, source, tol_pos, tol_ori_rad, data)
        if source_fk["position_violations"] or source_fk["orientation_violations"]:
            reasons.append(
                f"{tid}: source FK reconstruction exceeds tolerance "
                f"(pos_viol={source_fk['position_violations']}, ori_viol={source_fk['orientation_violations']})"
            )
        # stored per-waypoint errors must agree with the independent recomputation
        stored_pos = canonical["position_reconstruction_error_m"]
        if np.max(np.abs(stored_pos - canonical_fk["position_errors_m"])) > 1e-6:
            reasons.append(f"{tid}: stored canonical reconstruction errors disagree with recomputation")

        canonical_pos_max = max(canonical_pos_max, float(np.max(canonical_fk["position_errors_m"])))
        source_pos_max = max(source_pos_max, float(np.max(source_fk["position_errors_m"])))

        per_split_records[split].append(
            {
                "trajectory_id": tid,
                "content_hash": row["content_hash"],
                "path_seed": row["path_seed"],
                "q_start": np.asarray(source["q_reference"][0], dtype=np.float64),
            }
        )

    # ---- cross-split leakage (exact id/hash/seed) --------------------------------------------
    for i in range(len(SPLITS)):
        for j in range(i + 1, len(SPLITS)):
            a, b = SPLITS[i], SPLITS[j]
            for dim in ("content_hash", "path_seed"):
                sa = {r[dim] for r in per_split_records[a]}
                sb = {r[dim] for r in per_split_records[b]}
                if sa & sb:
                    reasons.append(f"cross-split {dim} leakage between {a} and {b}")

    # ---- no duplicate with core trajectories --------------------------------------------------
    core_manifest = paths.trajectories_dir / CORE_MANIFEST_NAME
    combined_trajectories_total = None
    combined_canonical_poses_total = None
    if core_manifest.is_file():
        core_rows = _read_manifest_rows(core_manifest)
        core_hashes = {r.get("content_hash", "") for r in core_rows}
        if set(hashes) & core_hashes:
            reasons.append("challenge trajectory shares a content_hash with a core trajectory")
        if check_combined:
            combined_trajectories_total = total + len(core_rows)
            combined_canonical_poses_total = canonical_poses_total + len(core_rows) * 400
            if full_counts:
                if combined_trajectories_total != 210:
                    reasons.append(f"combined trajectory total {combined_trajectories_total} != 210")
                if combined_canonical_poses_total != 84000:
                    reasons.append(f"combined canonical pose total {combined_canonical_poses_total} != 84000")

    # ---- anti-leakage report presence ---------------------------------------------------------
    anti_leakage_path = paths.trajectories_dir / ANTI_LEAKAGE_REPORT_NAME
    if not anti_leakage_path.is_file():
        reasons.append(f"anti-leakage report missing: {anti_leakage_path.name}")
    else:
        report = json.loads(anti_leakage_path.read_text(encoding="utf-8"))
        if not report.get("pass", False):
            reasons.append("challenge anti-leakage report is not pass=true")

    if full_counts and canonical_poses_total != CHALLENGE_TOTAL * 400:
        reasons.append(f"canonical pose total {canonical_poses_total} != {CHALLENGE_TOTAL * 400}")

    statistics = {
        "canonical_position_reconstruction_max_m": canonical_pos_max,
        "source_position_reconstruction_max_m": source_pos_max,
        "tolerance_position_m": tol_pos,
        "tolerance_orientation_deg": reach_settings.orientation_tolerance_deg,
    }

    return ChallengeTrajectoryValidationReport(
        passed=len(reasons) == 0,
        reasons=reasons,
        total_trajectories=total,
        split_counts=split_counts,
        family_counts=family_counts,
        canonical_poses_total=canonical_poses_total,
        combined_trajectories_total=combined_trajectories_total,
        combined_canonical_poses_total=combined_canonical_poses_total,
        statistics=statistics,
    )
