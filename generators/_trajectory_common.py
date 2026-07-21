"""Shared trajectory generation infrastructure: time scaling, anchor selection, geometry,
sequential DLS reachability validation, and manifest/trial CSV I/O.

Internal to the generators/ package; used by generate_line_trajectory.py,
generate_circle_trajectory.py, generate_figure8_trajectory.py, and generate_helix_trajectory.py.
"""

import csv
import json
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from generators._common import (
    REPO_ROOT,
    derive_seed,
    load_trajectory_config,
    relative_to_repo,
    rng_from,
    save_npz,
)
from generators.generate_orientation_profile import (
    bounded_orientation_target,
    fixed_orientation_profile,
    variable_orientation_profile,
)
from kinematics.dls_solver import solve_dls_until_converged
from kinematics.forward_kinematics import forward_kinematics
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import minimum_joint_limit_margin
from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from kinematics.singularity_metrics import minimum_singular_value
from utils.file_checksum import sha256_file

TRAJECTORIES_DIR = REPO_ROOT / "trajectories"
MANIFEST_PATH = TRAJECTORIES_DIR / "trajectory_manifest.csv"
TRIALS_PATH = TRAJECTORIES_DIR / "trajectory_trials.csv"

MANIFEST_COLUMNS = [
    "trajectory_id",
    "type",
    "file_path",
    "num_waypoints",
    "duration_s",
    "control_period_s",
    "orientation_mode",
    "closed_path",
    "anchor_q_json",
    "center_x_m",
    "center_y_m",
    "center_z_m",
    "path_scale_1_m",
    "path_scale_2_m",
    "path_scale_3_m",
    "generation_seed",
    "validation_waypoint_success_rate",
    "validation_position_max_mm",
    "validation_orientation_max_deg",
    "generation_status",
    "sha256",
]

TRIAL_COLUMNS = [
    "trial_id",
    "trajectory_id",
    "trial_category",
    "repeat_id",
    "seed",
    "speed_scale",
    "control_period_s",
    "q1_init",
    "q2_init",
    "q3_init",
    "q4_init",
    "q5_init",
    "q6_init",
    "q7_init",
]

REPEATABILITY_REPEATS = 10
ROBUSTNESS_INITIAL_CONFIGS = 5

ANCHOR_SEARCH_MARGIN_RAD = 0.35
ANCHOR_SEARCH_POOL_SIZE = 3000
ANCHOR_SIGMA_RATIO = 3.0

SHRINK_FACTOR = 0.85
MAX_SHRINK_ATTEMPTS = 15
MIN_SCALE = 0.05

ANCHOR_TAG = 90


def quintic_time_scaling(tau: np.ndarray):
    """Quintic scaling s(tau)=10tau^3-15tau^4+6tau^5 and its tau-derivatives.

    s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0.
    """
    tau = np.asarray(tau, dtype=np.float64)
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    s_dot = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    s_ddot = 60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3
    return s, s_dot, s_ddot


def build_time_and_path_parameter(num_waypoints: int, duration_s: float):
    if num_waypoints < 2:
        raise ValueError("num_waypoints must be >= 2")
    tau = np.linspace(0.0, 1.0, num_waypoints)
    time_s = tau * duration_s
    control_period_s = float(time_s[1] - time_s[0])
    s, s_dot_dtau, s_ddot_dtau = quintic_time_scaling(tau)
    return time_s, tau, s, s_dot_dtau, s_ddot_dtau, control_period_s


def cartesian_velocity_acceleration(dp_ds, d2p_ds2, s_dot_dtau, s_ddot_dtau, duration_s):
    """Analytic Cartesian velocity/acceleration via the chain rule through the quintic time scaling."""
    s_dot_dt = s_dot_dtau / duration_s
    s_ddot_dt = s_ddot_dtau / (duration_s**2)
    velocity = dp_ds * s_dot_dt[:, None]
    acceleration = d2p_ds2 * (s_dot_dt[:, None] ** 2) + dp_ds * s_ddot_dt[:, None]
    return velocity, acceleration


def orientation_phase_for_shape(s: np.ndarray, closed_path: bool) -> np.ndarray:
    """Path-parameter-to-orientation-phase map: for closed paths, go out (0->1) and back (1->0)
    so the orientation profile returns to its start alongside the closed position path.
    """
    if closed_path:
        return 1.0 - np.abs(1.0 - 2.0 * s)
    return s


def orientation_arrays(mode, R_anchor, rotation_vector_rad, s, closed_path, num_waypoints):
    """Build the wxyz quaternion array for a trajectory's orientation_mode ('fixed' or 'variable')."""
    if mode == "fixed":
        return fixed_orientation_profile(R_anchor, num_waypoints)
    if mode == "variable":
        R_end = bounded_orientation_target(R_anchor, rotation_vector_rad)
        phase = orientation_phase_for_shape(s, closed_path)
        return variable_orientation_profile(R_anchor, R_end, phase)
    raise ValueError(f"unknown orientation_mode '{mode}'")


def select_anchor(model_context, dls_config, seed: int, tag: int):
    """Search for a well-conditioned, joint-limit-interior anchor configuration.

    Picks the candidate with the largest normalized joint-limit margin among candidates whose
    sigma_min is comfortably above the singularity threshold (never assumed, always computed).
    """
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    threshold = float(dls_config["singularity_sigma_threshold"])
    rng = rng_from(seed, tag)

    candidates = rng.uniform(
        lower + ANCHOR_SEARCH_MARGIN_RAD, upper - ANCHOR_SEARCH_MARGIN_RAD, size=(ANCHOR_SEARCH_POOL_SIZE, model_context.nq)
    )
    best_q = None
    best_margin = -np.inf
    for q in candidates:
        J = geometric_jacobian_world(model_context, q)
        sigma_min = minimum_singular_value(J)
        if sigma_min < threshold * ANCHOR_SIGMA_RATIO:
            continue
        margin = minimum_joint_limit_margin(q, lower, upper)
        if margin > best_margin:
            best_margin = margin
            best_q = q

    if best_q is None:
        raise RuntimeError("failed to find a well-conditioned, joint-limit-interior anchor configuration")

    fk = forward_kinematics(model_context, best_q)
    return best_q, fk


def validate_sequential_reachability(model_context, dls_config, q_start, positions, quaternions):
    """Sequential warm-start DLS across an ordered waypoint chain; each solve reuses the previous solution."""
    n = positions.shape[0]
    q = np.asarray(q_start, dtype=np.float64).copy()
    successes = np.zeros(n, dtype=bool)
    position_errors_m = np.zeros(n, dtype=np.float64)
    orientation_errors_deg = np.zeros(n, dtype=np.float64)

    for i in range(n):
        R_target = quaternion_wxyz_to_matrix(quaternions[i])
        result = solve_dls_until_converged(model_context, q, positions[i], R_target, config=dls_config)
        successes[i] = bool(result.success)
        position_errors_m[i] = result.position_error_m
        orientation_errors_deg[i] = result.orientation_error_deg
        q = result.q_solution

    return successes, position_errors_m, orientation_errors_deg


def generate_validated_geometry(
    model_context,
    dls_config,
    q_anchor,
    build_fn: Callable[[float], Tuple[np.ndarray, np.ndarray, Dict]],
):
    """Try build_fn(scale) at decreasing scales until every waypoint validates via sequential DLS.

    build_fn(scale) must return (positions [N,3], quaternions [N,4], extra_metadata dict).
    Raises RuntimeError if no scale down to MIN_SCALE achieves 100% waypoint success.
    """
    scale = 1.0
    last_result = None
    for _ in range(MAX_SHRINK_ATTEMPTS):
        positions, quaternions, extra = build_fn(scale)
        successes, pos_err, orient_err = validate_sequential_reachability(model_context, dls_config, q_anchor, positions, quaternions)
        success_rate = float(np.mean(successes))
        last_result = {
            "positions": positions,
            "quaternions": quaternions,
            "extra": extra,
            "scale": scale,
            "success_rate": success_rate,
            "position_error_max_m": float(np.max(pos_err)),
            "orientation_error_max_deg": float(np.max(orient_err)),
        }
        if success_rate >= 1.0:
            return last_result
        scale *= SHRINK_FACTOR
        if scale < MIN_SCALE:
            break

    raise RuntimeError(
        f"could not reach 100% sequential-DLS waypoint success down to scale={last_result['scale']:.4f} "
        f"(best success_rate={last_result['success_rate']:.4f}); reduce the nominal path size further"
    )


def build_trajectory_arrays(time_s, s, positions, quaternions, dp_ds, d2p_ds2, s_dot_dtau, s_ddot_dtau, duration_s):
    velocity, acceleration = cartesian_velocity_acceleration(dp_ds, d2p_ds2, s_dot_dtau, s_ddot_dtau, duration_s)
    n = time_s.shape[0]
    arrays = {
        "waypoint_id": np.arange(n, dtype=np.int64),
        "time_s": time_s.astype(np.float64),
        "path_parameter_s": s.astype(np.float64),
        "target_position": positions.astype(np.float64),
        "target_quaternion": quaternions.astype(np.float64),
        "target_linear_velocity": velocity.astype(np.float64),
        "target_linear_acceleration": acceleration.astype(np.float64),
    }
    for key, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            raise ValueError(f"array '{key}' contains non-finite values")
    return arrays


def _format_float(value) -> str:
    return f"{float(value):.10f}"


def upsert_manifest_row(row: dict) -> Path:
    """Insert/replace a single trajectory_manifest.csv row keyed by trajectory_id."""
    rows = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as handle:
            for existing in csv.DictReader(handle):
                rows[existing["trajectory_id"]] = existing
    rows[row["trajectory_id"]] = {col: row.get(col, "") for col in MANIFEST_COLUMNS}
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for key in sorted(rows.keys()):
            writer.writerow(rows[key])
    return MANIFEST_PATH


def replace_trial_rows(trajectory_id: str, new_rows: List[dict]) -> Path:
    """Replace all trajectory_trials.csv rows for one trajectory_id with a freshly generated set."""
    existing_rows = []
    if TRIALS_PATH.exists():
        with open(TRIALS_PATH, newline="", encoding="utf-8") as handle:
            for existing in csv.DictReader(handle):
                if existing.get("trajectory_id") != trajectory_id:
                    existing_rows.append(existing)
    formatted_new = [{col: row.get(col, "") for col in TRIAL_COLUMNS} for row in new_rows]
    all_rows = existing_rows + formatted_new
    all_rows.sort(key=lambda r: r["trial_id"])
    with open(TRIALS_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    return TRIALS_PATH


def build_trials(model_context, dls_config, trajectory_id, q_anchor, first_position, first_quaternion, seed, control_period_s):
    """Build repeatability and robustness trial rows for one trajectory.

    Repeatability: same trajectory, same q_initial (the anchor), same conditions, repeat_id 0..9,
    no random perturbation, at each speed scale (deterministic kinematics means ideal repeatability
    is expected here; this only records the trial definitions, not Tier 4 results).

    Robustness: several independently valid q_initial configurations (each verified able to solve
    the trajectory's first waypoint), at each speed scale, with distinct seeds.
    """
    trajectory_config = load_trajectory_config()
    speed_scales = trajectory_config["speed_scales"]
    traj_tag = zlib.crc32(trajectory_id.encode("utf-8"))

    rows = []
    for speed_scale in speed_scales:
        for repeat_id in range(REPEATABILITY_REPEATS):
            trial_id = f"{trajectory_id}_repeatability_speed{speed_scale}_r{repeat_id}"
            rows.append(
                {
                    "trial_id": trial_id,
                    "trajectory_id": trajectory_id,
                    "trial_category": "repeatability",
                    "repeat_id": repeat_id,
                    "seed": seed,
                    "speed_scale": speed_scale,
                    "control_period_s": _format_float(control_period_s),
                    **{f"q{i + 1}_init": _format_float(q_anchor[i]) for i in range(7)},
                }
            )

    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    R_first = quaternion_wxyz_to_matrix(first_quaternion)

    found_q, found_seeds = [], []
    tries = 0
    while len(found_q) < ROBUSTNESS_INITIAL_CONFIGS and tries < 10 * ROBUSTNESS_INITIAL_CONFIGS + 50:
        tries += 1
        candidate_seed = derive_seed(seed, 91, traj_tag, tries)
        candidate_rng = np.random.default_rng(candidate_seed)
        q_candidate = candidate_rng.uniform(lower + 0.2, upper - 0.2)
        result = solve_dls_until_converged(model_context, q_candidate, first_position, R_first, config=dls_config)
        if result.success:
            found_q.append(q_candidate)
            found_seeds.append(candidate_seed)

    if len(found_q) < ROBUSTNESS_INITIAL_CONFIGS:
        raise RuntimeError(f"could not find {ROBUSTNESS_INITIAL_CONFIGS} robustness initial configurations for {trajectory_id}")

    for speed_scale in speed_scales:
        for idx, (q_init, candidate_seed) in enumerate(zip(found_q, found_seeds)):
            trial_id = f"{trajectory_id}_robustness_speed{speed_scale}_c{idx}"
            rows.append(
                {
                    "trial_id": trial_id,
                    "trajectory_id": trajectory_id,
                    "trial_category": "robustness",
                    "repeat_id": 0,
                    "seed": candidate_seed,
                    "speed_scale": speed_scale,
                    "control_period_s": _format_float(control_period_s),
                    **{f"q{i + 1}_init": _format_float(q_init[i]) for i in range(7)},
                }
            )

    return rows


def write_trajectory_outputs(
    trajectory_id: str,
    trajectory_type: str,
    orientation_mode: str,
    closed_path: bool,
    npz_path: Path,
    arrays: dict,
    q_anchor,
    center,
    path_scales,
    seed: int,
    validation_result: dict,
    overwrite: bool,
):
    saved_path = save_npz(npz_path, overwrite, arrays)
    sha = sha256_file(saved_path)

    manifest_row = {
        "trajectory_id": trajectory_id,
        "type": trajectory_type,
        "file_path": relative_to_repo(saved_path),
        "num_waypoints": int(arrays["waypoint_id"].shape[0]),
        "duration_s": _format_float(arrays["time_s"][-1]),
        "control_period_s": _format_float(arrays["time_s"][1] - arrays["time_s"][0]),
        "orientation_mode": orientation_mode,
        "closed_path": bool(closed_path),
        "anchor_q_json": json.dumps([float(v) for v in q_anchor]),
        "center_x_m": _format_float(center[0]),
        "center_y_m": _format_float(center[1]),
        "center_z_m": _format_float(center[2]),
        "path_scale_1_m": _format_float(path_scales[0]) if len(path_scales) > 0 else "",
        "path_scale_2_m": _format_float(path_scales[1]) if len(path_scales) > 1 else "",
        "path_scale_3_m": _format_float(path_scales[2]) if len(path_scales) > 2 else "",
        "generation_seed": seed,
        "validation_waypoint_success_rate": _format_float(validation_result["success_rate"]),
        "validation_position_max_mm": _format_float(validation_result["position_error_max_m"] * 1000.0),
        "validation_orientation_max_deg": _format_float(validation_result["orientation_error_max_deg"]),
        "generation_status": "validated" if validation_result["success_rate"] >= 1.0 else "incomplete",
        "sha256": sha,
    }
    upsert_manifest_row(manifest_row)
    return saved_path, sha, manifest_row
