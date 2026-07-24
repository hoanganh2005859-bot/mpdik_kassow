"""Independent validator for Dataset v2 core trajectories (Phase 5, hardened in Phase 5.1).

Reuses ``kinematics/`` (FK, quaternion/rotation utilities) unchanged for every recomputation. The
validator never runs IK: it re-derives FK from the stored ``q_reference`` and checks it against
the target pose using Dataset v2's **own strict generation tolerances**
(``configs/generation_reachability_config.json``) -- never a Dataset v1 DLS *baseline evaluation*
threshold, and never the generator's own stored success flags.

Both the canonical (400-waypoint) and the high-resolution source path are verified.
"""

import csv
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from dataset_v2.config_templates import (
    CORE_SHAPES,
    FROZEN_CORE_SEED_REVISION as EXPECTED_FROZEN_CORE_SEED_REVISION,
    ORIENTATION_MODES,
    SPLITS,
)
from dataset_v2.seeds import SEED_ALGORITHM_ID
from dataset_v2.core_trajectory_generation import (
    ANTI_LEAKAGE_REPORT_NAME,
    GEOMETRY_SEARCH_REPORT_NAME,
    MANIFEST_NAME,
    SHAPE_CLOSED_PATH,
    resample_canonical,
    scale_band_of,
)
from dataset_v2.generation_reachability import (
    GENERATION_REACHABILITY_CONFIG_NAME,
    fk_reconstruction_error,
    load_generation_reachability_settings,
)
from dataset_v2.locator import require_dataset_v2_root
from kinematics.model_loader import ModelContext, load_model_context
from kinematics.quaternion_utils import quaternion_geodesic_angle, quaternion_wxyz_to_matrix
from utils.config_loader import load_json_config
from utils.npz_utils import load_npz

QUATERNION_NORM_TOLERANCE = 1e-6
ENDPOINT_TOLERANCE_M = 1e-6
MONOTONIC_TOLERANCE = -1e-9
FIXED_ORIENTATION_VARIATION_TOLERANCE_RAD = 1e-6
VARIABLE_ORIENTATION_MIN_VARIATION_RAD = 1e-3
CANONICAL_RESAMPLE_TOLERANCE_M = 1e-6
CANONICAL_RESAMPLE_ORIENTATION_TOLERANCE_RAD = 1e-6

# Dataset v1's DLS baseline evaluation thresholds. Referenced ONLY to assert that the generation
# tolerances are strictly tighter (i.e. that reachability was not defined by the solver the
# dataset will later benchmark). Never used as an acceptance tolerance here.
_DLS_BASELINE_POSITION_THRESHOLD_M = 0.006
_DLS_BASELINE_ORIENTATION_THRESHOLD_DEG = 10.0


@dataclass
class CoreTrajectoryValidationReport:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    total_trajectories: int = 0
    split_counts: Dict[str, int] = field(default_factory=dict)
    shape_counts: Dict[str, int] = field(default_factory=dict)
    orientation_counts: Dict[str, int] = field(default_factory=dict)
    canonical_poses_total: int = 0
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


def _verify_path_fk(model_context, arrays, tolerance_position_m, tolerance_orientation_rad, data) -> dict:
    """Independently recompute FK(q_reference) for a whole stored path."""
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
        "position_violations": int(np.sum(position_errors > tolerance_position_m)),
        "orientation_violations": int(np.sum(orientation_errors > tolerance_orientation_rad)),
    }


def validate_core_trajectories(
    dataset_root,
    model_context: Optional[ModelContext] = None,
    full_counts: bool = True,
) -> CoreTrajectoryValidationReport:
    """Validate the on-disk core trajectory set under ``dataset_root``.

    ``full_counts=True`` (default) checks against the locked 120/40-40-40/24-per-shape/60-60
    counts and 48,000 total canonical poses.
    """
    paths = require_dataset_v2_root(dataset_root)
    model_context = model_context if model_context is not None else load_model_context()
    reasons: List[str] = []

    # --- generation tolerance must be independent of, and tighter than, the DLS baseline --------
    reach_config = load_json_config(paths.configs_dir / GENERATION_REACHABILITY_CONFIG_NAME)
    if reach_config.get("independence", {}).get("reads_dls_evaluation_thresholds", True):
        # Fatal for the whole report: without an independent tolerance there is nothing trustworthy
        # left to validate against, and load_generation_reachability_settings would refuse anyway.
        return CoreTrajectoryValidationReport(
            passed=False,
            reasons=[
                "generation reachability config declares reads_dls_evaluation_thresholds=true; "
                "generation acceptance must not be defined by the DLS baseline evaluation thresholds "
                "that this dataset will later be used to measure"
            ],
        )
    reach_settings = load_generation_reachability_settings(paths)
    if reach_settings.position_tolerance_m >= _DLS_BASELINE_POSITION_THRESHOLD_M:
        reasons.append(
            f"generation position tolerance {reach_settings.position_tolerance_m} m is not strictly tighter "
            f"than the DLS baseline success threshold {_DLS_BASELINE_POSITION_THRESHOLD_M} m"
        )
    if reach_settings.orientation_tolerance_deg >= _DLS_BASELINE_ORIENTATION_THRESHOLD_DEG:
        reasons.append(
            f"generation orientation tolerance {reach_settings.orientation_tolerance_deg} deg is not strictly "
            f"tighter than the DLS baseline success threshold {_DLS_BASELINE_ORIENTATION_THRESHOLD_DEG} deg"
        )

    trajectory_config = load_json_config(paths.configs_dir / "trajectory_config.json")
    gate = trajectory_config["minimum_scale_gate"]
    minimum_scale = gate.get("minimum_core_accepted_scale")
    gate_enforced = bool(gate.get("enforced", False))
    diagnostic_bands = [float(b) for b in gate["diagnostic_bands"]]

    # --- locked alternative set + frozen revision (Phase 5.3) ------------------------------------
    from dataset_v2.core_trajectory_generation import enumerate_geometry_alternatives

    alternatives_cfg = trajectory_config["geometry_alternatives"]
    locked_alternative_ids: Dict[str, set] = {}
    for shape_name in CORE_SHAPES:
        locked_alternative_ids[shape_name] = {
            a["alternative_id"] for a in enumerate_geometry_alternatives(shape_name, alternatives_cfg)
        }

    seed_policy = load_json_config(paths.configs_dir / "seed_policy.json")
    frozen_revision = int(seed_policy.get("frozen_core_seed_revision", 0))
    if frozen_revision != EXPECTED_FROZEN_CORE_SEED_REVISION:
        reasons.append(
            f"frozen_core_seed_revision is {frozen_revision}, expected {EXPECTED_FROZEN_CORE_SEED_REVISION}"
        )
    history = {int(e["revision"]): e for e in seed_policy.get("frozen_core_seed_revision_history", [])}
    # every superseded revision must stay recorded as burned, and exactly one may be active
    for burned in range(1, EXPECTED_FROZEN_CORE_SEED_REVISION):
        if history.get(burned, {}).get("status") != "burned_not_shippable":
            reasons.append(f"frozen seed revision {burned} is not recorded as burned_not_shippable")
        elif not (history.get(burned, {}).get("reason") or "").strip():
            reasons.append(f"frozen seed revision {burned} is burned but records no reason")
    if history.get(EXPECTED_FROZEN_CORE_SEED_REVISION, {}).get("status") != "active":
        reasons.append(f"frozen seed revision {EXPECTED_FROZEN_CORE_SEED_REVISION} is not recorded as active")
    active = [r for r, e in history.items() if e.get("status") == "active"]
    if active != [EXPECTED_FROZEN_CORE_SEED_REVISION]:
        reasons.append(f"exactly one active frozen seed revision expected, found {sorted(active)}")

    # the dataset root must declare Dataset v2's own NumPy-independent seed algorithm
    if seed_policy.get("seed_algorithm_id") != SEED_ALGORITHM_ID:
        reasons.append(
            f"seed_algorithm_id is {seed_policy.get('seed_algorithm_id')!r}, expected {SEED_ALGORITHM_ID!r} "
            "(Dataset v2 must not depend on Dataset v1's NumPy-version-dependent seed derivation)"
        )

    manifest_path = paths.trajectories_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"core trajectory manifest not found: {manifest_path}")
    rows = _read_manifest_rows(manifest_path)
    n = len(rows)

    trajectory_ids = [r["trajectory_id"] for r in rows]
    content_hashes = [r["content_hash"] for r in rows]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        reasons.append(f"duplicate trajectory_id found ({len(trajectory_ids) - len(set(trajectory_ids))} collision(s))")
    if len(set(content_hashes)) != len(content_hashes):
        reasons.append(f"duplicate content_hash found ({len(content_hashes) - len(set(content_hashes))} collision(s))")

    split_counts = {name: sum(1 for r in rows if r["split"] == name) for name in SPLITS}
    shape_counts = {name: sum(1 for r in rows if r["shape"] == name) for name in CORE_SHAPES}
    orientation_counts = {name: sum(1 for r in rows if r["orientation_mode"] == name) for name in ORIENTATION_MODES}
    anchor_counts: Dict[str, int] = {}
    for r in rows:
        anchor_counts[r["anchor_id"]] = anchor_counts.get(r["anchor_id"], 0) + 1

    if full_counts:
        if n != 120:
            reasons.append(f"expected 120 total core trajectories, found {n}")
        for name in SPLITS:
            if split_counts[name] != 40:
                reasons.append(f"split '{name}' has {split_counts[name]} trajectories, expected 40")
        for name in CORE_SHAPES:
            if shape_counts[name] != 24:
                reasons.append(f"shape '{name}' has {shape_counts[name]} trajectories, expected 24")
        for name in ORIENTATION_MODES:
            if orientation_counts[name] != 60:
                reasons.append(f"orientation_mode '{name}' has {orientation_counts[name]} trajectories, expected 60")
        for anchor_id, count in anchor_counts.items():
            if count != 10:
                reasons.append(f"anchor '{anchor_id}' has {count} trajectories, expected 10")

    anchor_to_splits: Dict[str, set] = {}
    for r in rows:
        anchor_to_splits.setdefault(r["anchor_id"], set()).add(r["split"])
    for anchor_id, splits_seen in anchor_to_splits.items():
        if len(splits_seen) > 1:
            reasons.append(f"anchor '{anchor_id}' is used by trajectories in more than one split: {sorted(splits_seen)}")

    # --- geometry-search metadata must be present ------------------------------------------------
    geometry_search_path = paths.trajectories_dir / GEOMETRY_SEARCH_REPORT_NAME
    geometry_search_by_id: Dict[str, dict] = {}
    if not geometry_search_path.is_file():
        reasons.append(f"geometry-search report not found: {geometry_search_path}")
    else:
        payload = json.loads(geometry_search_path.read_text(encoding="utf-8"))
        geometry_search_by_id = {rec["trajectory_id"]: rec for rec in payload.get("trajectories", [])}

    tolerance_orientation_rad = reach_settings.orientation_tolerance_rad
    data = model_context.new_data()

    canonical_position_max: List[float] = []
    canonical_position_p95: List[float] = []
    canonical_orientation_max: List[float] = []
    canonical_orientation_p95: List[float] = []
    source_position_max: List[float] = []
    source_orientation_max: List[float] = []
    accepted_scales: List[float] = []
    per_shape_position_max: Dict[str, List[float]] = {shape: [] for shape in CORE_SHAPES}
    worst_records: List[dict] = []
    canonical_poses_total = 0

    for r in rows:
        trajectory_id = r["trajectory_id"]
        split_dir = _split_dir(paths, r["split"])
        canonical_npz = split_dir / f"{trajectory_id}.npz"
        source_npz = split_dir / f"{trajectory_id}_source.npz"

        if not canonical_npz.is_file():
            reasons.append(f"missing canonical NPZ for '{trajectory_id}': {canonical_npz}")
            continue
        if not source_npz.is_file():
            reasons.append(f"missing source NPZ for '{trajectory_id}': {source_npz}")
            continue

        canonical = load_npz(canonical_npz)
        source = load_npz(source_npz)
        canonical_poses_total += canonical["target_position"].shape[0]

        expected_count = int(r["canonical_waypoint_count"])
        expected_source_count = int(r["source_waypoint_count"])
        if canonical["target_position"].shape != (expected_count, 3):
            reasons.append(f"'{trajectory_id}' canonical target_position has shape {canonical['target_position'].shape}, expected ({expected_count}, 3)")
        if canonical["target_quaternion"].shape != (expected_count, 4):
            reasons.append(f"'{trajectory_id}' canonical target_quaternion has shape {canonical['target_quaternion'].shape}, expected ({expected_count}, 4)")
        if source["target_position"].shape[0] != expected_source_count:
            reasons.append(
                f"'{trajectory_id}' source waypoint count {source['target_position'].shape[0]} does not match manifest {expected_source_count}"
            )
        if expected_source_count <= expected_count:
            reasons.append(f"'{trajectory_id}' source waypoint count {expected_source_count} must exceed the canonical count {expected_count}")

        for name, arr in list(canonical.items()) + list(source.items()):
            if arr.dtype == object:
                reasons.append(f"'{trajectory_id}' array '{name}' has object dtype (forbidden)")
            if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
                reasons.append(f"'{trajectory_id}' array '{name}' contains non-finite values")

        # --- q_reference must exist, be complete, and lie within operational limits --------------
        lower = model_context.operational_lower_rad
        upper = model_context.operational_upper_rad
        missing_reference = False
        for label, arrays, expected in (("canonical", canonical, expected_count), ("source", source, expected_source_count)):
            if "q_reference" not in arrays:
                reasons.append(f"'{trajectory_id}' {label} path has no q_reference (strict reachability evidence is mandatory)")
                missing_reference = True
                continue
            if arrays["q_reference"].shape != (expected, model_context.nq):
                reasons.append(
                    f"'{trajectory_id}' {label} q_reference has shape {arrays['q_reference'].shape}, expected ({expected}, {model_context.nq}) "
                    "-- every waypoint must carry reachability evidence, none may be skipped"
                )
                missing_reference = True
                continue
            if np.any(arrays["q_reference"] < lower - 1e-9) or np.any(arrays["q_reference"] > upper + 1e-9):
                reasons.append(f"'{trajectory_id}' {label} q_reference violates operational joint limits")
            if "waypoint_reachable" in arrays and not np.all(arrays["waypoint_reachable"]):
                reasons.append(f"'{trajectory_id}' {label} path has waypoint(s) marked unreachable")
        if missing_reference:
            continue

        # --- INDEPENDENT FK reconstruction against Dataset v2's own strict tolerances ------------
        canonical_fk = _verify_path_fk(model_context, canonical, reach_settings.position_tolerance_m, tolerance_orientation_rad, data)
        source_fk = _verify_path_fk(model_context, source, reach_settings.position_tolerance_m, tolerance_orientation_rad, data)

        if canonical_fk["position_violations"]:
            reasons.append(
                f"'{trajectory_id}' {canonical_fk['position_violations']} canonical waypoint(s) exceed the strict position "
                f"tolerance {reach_settings.position_tolerance_m} m (max {float(np.max(canonical_fk['position_errors_m'])):.3e} m)"
            )
        if canonical_fk["orientation_violations"]:
            reasons.append(
                f"'{trajectory_id}' {canonical_fk['orientation_violations']} canonical waypoint(s) exceed the strict orientation "
                f"tolerance {reach_settings.orientation_tolerance_deg} deg (max {float(np.degrees(np.max(canonical_fk['orientation_errors_rad']))):.3e} deg)"
            )
        if source_fk["position_violations"]:
            reasons.append(
                f"'{trajectory_id}' {source_fk['position_violations']} source waypoint(s) exceed the strict position "
                f"tolerance {reach_settings.position_tolerance_m} m (max {float(np.max(source_fk['position_errors_m'])):.3e} m)"
            )
        if source_fk["orientation_violations"]:
            reasons.append(
                f"'{trajectory_id}' {source_fk['orientation_violations']} source waypoint(s) exceed the strict orientation "
                f"tolerance {reach_settings.orientation_tolerance_deg} deg (max {float(np.degrees(np.max(source_fk['orientation_errors_rad']))):.3e} deg)"
            )

        # --- stored per-waypoint error metadata must match the independent recomputation ---------
        if "position_reconstruction_error_m" in canonical:
            stored = canonical["position_reconstruction_error_m"]
            if stored.shape == canonical_fk["position_errors_m"].shape:
                if np.max(np.abs(stored - canonical_fk["position_errors_m"])) > 1e-9:
                    reasons.append(f"'{trajectory_id}' stored canonical position_reconstruction_error_m does not match the independent recomputation")

        canonical_position_max.append(float(np.max(canonical_fk["position_errors_m"])))
        canonical_position_p95.append(float(np.percentile(canonical_fk["position_errors_m"], 95)))
        canonical_orientation_max.append(float(np.degrees(np.max(canonical_fk["orientation_errors_rad"]))))
        canonical_orientation_p95.append(float(np.degrees(np.percentile(canonical_fk["orientation_errors_rad"], 95))))
        source_position_max.append(float(np.max(source_fk["position_errors_m"])))
        source_orientation_max.append(float(np.degrees(np.max(source_fk["orientation_errors_rad"]))))
        per_shape_position_max[r["shape"]].append(float(np.max(canonical_fk["position_errors_m"])))
        worst_records.append(
            {
                "trajectory_id": trajectory_id,
                "canonical_position_max_m": float(np.max(canonical_fk["position_errors_m"])),
                "source_position_max_m": float(np.max(source_fk["position_errors_m"])),
                "accepted_scale": float(r["accepted_scale"]),
            }
        )

        # --- scale metadata + minimum-scale gate --------------------------------------------------
        accepted_scale = float(r["accepted_scale"])
        accepted_scales.append(accepted_scale)
        expected_band = scale_band_of(accepted_scale, diagnostic_bands)
        if r.get("scale_band") != expected_band:
            reasons.append(f"'{trajectory_id}' scale_band '{r.get('scale_band')}' does not match the accepted scale {accepted_scale} (expected '{expected_band}')")
        if accepted_scale < 1.0 and not (r.get("scale_reduction_reason") or "").strip():
            reasons.append(f"'{trajectory_id}' has accepted_scale {accepted_scale} < 1.0 but no scale_reduction_reason recorded")
        if gate_enforced and minimum_scale is not None and accepted_scale < float(minimum_scale):
            reasons.append(f"'{trajectory_id}' accepted_scale {accepted_scale} is below the locked minimum_core_accepted_scale {minimum_scale}")
        selected_alternative_id = (r.get("geometry_alternative_id") or "").strip()
        if not selected_alternative_id:
            reasons.append(f"'{trajectory_id}' has no geometry_alternative_id recorded")
        elif selected_alternative_id not in locked_alternative_ids.get(r["shape"], set()):
            reasons.append(
                f"'{trajectory_id}' selected geometry_alternative_id '{selected_alternative_id}' is not in the "
                f"locked alternative set for shape '{r['shape']}'"
            )
        if not (r.get("geometry_alternative_family") or "").strip():
            reasons.append(f"'{trajectory_id}' has no geometry_alternative_family recorded")
        try:
            alternative_metadata = json.loads(r.get("geometry_alternative_metadata_json") or "{}")
        except json.JSONDecodeError:
            alternative_metadata = {}
            reasons.append(f"'{trajectory_id}' geometry_alternative_metadata_json is not valid JSON")
        if not alternative_metadata:
            reasons.append(f"'{trajectory_id}' has empty geometry_alternative_metadata_json")

        # traversal / handedness metadata must agree with the stored path
        if r["shape"] == "circle" and alternative_metadata:
            if alternative_metadata.get("traversal_direction") not in ("ccw", "cw"):
                reasons.append(f"'{trajectory_id}' circle alternative has no valid traversal_direction")
        if r["shape"] == "figure8" and alternative_metadata:
            if alternative_metadata.get("handedness") not in ("left", "right"):
                reasons.append(f"'{trajectory_id}' figure8 alternative has no valid handedness")
            if "axis_swap" not in alternative_metadata:
                reasons.append(f"'{trajectory_id}' figure8 alternative has no axis_swap flag")

        if geometry_search_by_id and trajectory_id not in geometry_search_by_id:
            reasons.append(f"'{trajectory_id}' has no entry in the geometry-search report")
        elif geometry_search_by_id:
            record = geometry_search_by_id[trajectory_id]
            if record.get("accepted_alternative_id") != selected_alternative_id:
                reasons.append(f"'{trajectory_id}' manifest geometry_alternative_id does not match the geometry-search report")
            # the selected alternative must actually be the best scale the search observed
            best_observed = record.get("max_reachable_scale_observed")
            if best_observed is not None and accepted_scale + 1e-12 < float(best_observed):
                reasons.append(
                    f"'{trajectory_id}' accepted scale {accepted_scale} is below the best reachable scale "
                    f"{best_observed} observed during the geometry search"
                )
            available = record.get("alternatives_available") or []
            if selected_alternative_id and available and selected_alternative_id not in available:
                reasons.append(f"'{trajectory_id}' selected alternative is not among the alternatives the search enumerated")
            if not record.get("attempts"):
                reasons.append(f"'{trajectory_id}' geometry-search report records no attempts")
            for attempt in record.get("attempts", []):
                if not attempt.get("reachable") and not attempt.get("rejection_reason"):
                    reasons.append(f"'{trajectory_id}' has a rejected geometry-search attempt with no rejection_reason")
                    break

        # --- quaternion normalization + sign continuity -------------------------------------------
        norms = np.linalg.norm(canonical["target_quaternion"], axis=1)
        if np.any(np.abs(norms - 1.0) > QUATERNION_NORM_TOLERANCE):
            reasons.append(f"'{trajectory_id}' canonical target_quaternion contains non-unit quaternion(s)")
        dots = np.sum(canonical["target_quaternion"][:-1] * canonical["target_quaternion"][1:], axis=1)
        if np.any(dots < -1e-9):
            reasons.append(f"'{trajectory_id}' canonical target_quaternion is not sign-continuous")

        # --- monotonicity ---------------------------------------------------------------------------
        if np.any(np.diff(canonical["time_s"]) < MONOTONIC_TOLERANCE):
            reasons.append(f"'{trajectory_id}' canonical time_s is not monotonically non-decreasing")
        if np.any(np.diff(canonical["source_parameter_u"]) < MONOTONIC_TOLERANCE):
            reasons.append(f"'{trajectory_id}' canonical source_parameter_u is not monotonically non-decreasing")
        if np.any(np.diff(source["tau"]) < MONOTONIC_TOLERANCE):
            reasons.append(f"'{trajectory_id}' source tau is not monotonically non-decreasing")
        if np.any(np.diff(source["time_s"]) < MONOTONIC_TOLERANCE):
            reasons.append(f"'{trajectory_id}' source time_s is not monotonically non-decreasing")

        # --- endpoint preservation -----------------------------------------------------------------
        if np.linalg.norm(canonical["target_position"][0] - source["target_position"][0]) > ENDPOINT_TOLERANCE_M:
            reasons.append(f"'{trajectory_id}' canonical first waypoint does not match source first waypoint")
        if np.linalg.norm(canonical["target_position"][-1] - source["target_position"][-1]) > ENDPOINT_TOLERANCE_M:
            reasons.append(f"'{trajectory_id}' canonical last waypoint does not match source last waypoint")

        # --- arc length / angular displacement recomputation ---------------------------------------
        recomputed_arc_length = float(np.sum(np.linalg.norm(np.diff(canonical["target_position"], axis=0), axis=1)))
        stored_arc_length = float(r["arc_length_m"])
        if abs(recomputed_arc_length - stored_arc_length) > 1e-6 + 1e-6 * abs(stored_arc_length):
            reasons.append(f"'{trajectory_id}' recomputed arc_length_m={recomputed_arc_length:.8f} does not match manifest {stored_arc_length:.8f}")
        if abs(float(canonical["cumulative_arc_length_m"][-1]) - recomputed_arc_length) > 1e-6:
            reasons.append(f"'{trajectory_id}' stored cumulative_arc_length_m[-1] does not match recomputed total")

        recomputed_angular = 0.0
        for i in range(1, canonical["target_quaternion"].shape[0]):
            recomputed_angular += quaternion_geodesic_angle(canonical["target_quaternion"][i - 1], canonical["target_quaternion"][i])
        stored_angular = float(r["cumulative_angular_displacement_rad"])
        if abs(recomputed_angular - stored_angular) > 1e-6 + 1e-6 * abs(stored_angular):
            reasons.append(f"'{trajectory_id}' recomputed angular displacement={recomputed_angular:.8f} does not match manifest {stored_angular:.8f}")

        # --- canonical resample-from-source consistency ---------------------------------------------
        recomputed_canonical = resample_canonical(
            source["time_s"], source["tau"], source["target_position"], source["target_quaternion"], expected_count
        )
        pos_diff = np.max(np.linalg.norm(recomputed_canonical["target_position"] - canonical["target_position"], axis=1))
        if pos_diff > CANONICAL_RESAMPLE_TOLERANCE_M:
            reasons.append(f"'{trajectory_id}' canonical target_position does not match a fresh resample of the stored source path (max diff {pos_diff:.8f} m)")
        orient_diff = max(
            quaternion_geodesic_angle(recomputed_canonical["target_quaternion"][i], canonical["target_quaternion"][i])
            for i in range(expected_count)
        )
        if orient_diff > CANONICAL_RESAMPLE_ORIENTATION_TOLERANCE_RAD:
            reasons.append(f"'{trajectory_id}' canonical target_quaternion does not match a fresh resample of the stored source path (max diff {orient_diff:.8f} rad)")

        # --- closure -------------------------------------------------------------------------------
        shape = r["shape"]
        closed_path_expected = SHAPE_CLOSED_PATH.get(shape)
        stored_closed = r["closed_path"] in ("True", "true", "1")
        if closed_path_expected is not None and stored_closed != closed_path_expected:
            reasons.append(f"'{trajectory_id}' closed_path={stored_closed} does not match expected {closed_path_expected} for shape '{shape}'")
        if closed_path_expected:
            closure_pos_err = float(np.linalg.norm(canonical["target_position"][-1] - canonical["target_position"][0]))
            if closure_pos_err > 1e-3:
                reasons.append(f"'{trajectory_id}' closed-path position closure error {closure_pos_err:.8f} m exceeds tolerance")
            if r["orientation_mode"] == "variable":
                closure_orient_err = quaternion_geodesic_angle(canonical["target_quaternion"][0], canonical["target_quaternion"][-1])
                if closure_orient_err > 1e-3:
                    reasons.append(f"'{trajectory_id}' closed-path variable-orientation closure error {closure_orient_err:.8f} rad exceeds tolerance")

        # --- orientation mode behavior --------------------------------------------------------------
        quats = canonical["target_quaternion"]
        max_variation = max(quaternion_geodesic_angle(quats[0], quats[i]) for i in range(quats.shape[0]))
        if r["orientation_mode"] == "fixed" and max_variation > FIXED_ORIENTATION_VARIATION_TOLERANCE_RAD:
            reasons.append(f"'{trajectory_id}' orientation_mode=fixed but orientation varies by {max_variation:.8f} rad")
        if r["orientation_mode"] == "variable" and max_variation < VARIABLE_ORIENTATION_MIN_VARIATION_RAD:
            reasons.append(f"'{trajectory_id}' orientation_mode=variable but orientation barely varies ({max_variation:.8f} rad)")

    if full_counts and canonical_poses_total != 48000:
        reasons.append(f"expected 48000 total canonical poses, found {canonical_poses_total}")

    anti_leakage_path = paths.trajectories_dir / ANTI_LEAKAGE_REPORT_NAME
    if anti_leakage_path.is_file():
        anti_leakage_report = json.loads(anti_leakage_path.read_text(encoding="utf-8"))
        if not anti_leakage_report.get("pass", False):
            reasons.append("anti-leakage report on disk does not report pass=True")
    else:
        reasons.append(f"anti-leakage report not found: {anti_leakage_path}")

    statistics = {}
    if canonical_position_max:
        band_counts: Dict[str, int] = {}
        for scale in accepted_scales:
            band = scale_band_of(scale, diagnostic_bands)
            band_counts[band] = band_counts.get(band, 0) + 1
        statistics = {
            "generation_tolerance_position_m": reach_settings.position_tolerance_m,
            "generation_tolerance_orientation_deg": reach_settings.orientation_tolerance_deg,
            "canonical_position_error_m": {
                "max": max(canonical_position_max),
                "p95": float(np.percentile(canonical_position_max, 95)),
                "median": float(np.median(canonical_position_max)),
            },
            "canonical_orientation_error_deg": {
                "max": max(canonical_orientation_max),
                "p95": float(np.percentile(canonical_orientation_max, 95)),
                "median": float(np.median(canonical_orientation_max)),
            },
            "source_position_error_m": {
                "max": max(source_position_max),
                "p95": float(np.percentile(source_position_max, 95)),
                "median": float(np.median(source_position_max)),
            },
            "source_orientation_error_deg": {
                "max": max(source_orientation_max),
                "p95": float(np.percentile(source_orientation_max, 95)),
                "median": float(np.median(source_orientation_max)),
            },
            "accepted_scale": {
                "min": min(accepted_scales),
                "p05": float(np.percentile(accepted_scales, 5)),
                "median": float(np.median(accepted_scales)),
                "mean": float(np.mean(accepted_scales)),
                "max": max(accepted_scales),
            },
            "scale_band_counts": band_counts,
            "below_0.75": sum(1 for v in accepted_scales if v < 0.75),
            "below_0.50": sum(1 for v in accepted_scales if v < 0.50),
            "below_0.25": sum(1 for v in accepted_scales if v < 0.25),
            "per_shape_canonical_position_max_m": {
                shape: (max(values) if values else None) for shape, values in per_shape_position_max.items()
            },
            "worst_10_trajectories": sorted(worst_records, key=lambda x: -x["canonical_position_max_m"])[:10],
        }

    return CoreTrajectoryValidationReport(
        passed=len(reasons) == 0,
        reasons=reasons,
        total_trajectories=n,
        split_counts=split_counts,
        shape_counts=shape_counts,
        orientation_counts=orientation_counts,
        canonical_poses_total=canonical_poses_total,
        statistics=statistics,
    )
