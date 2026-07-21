"""Tier 2: sequential DLS IK solve across an ordered waypoint chain.

Loads one generated trajectory (trajectories/<type>/<trajectory_id>.npz plus its
trajectory_manifest.csv row) and one trial definition (a row of
trajectories/trajectory_trials.csv), validates them against each other and against the robot's
operational limits, then dispatches to either algorithms.warm_start_dls or
algorithms.cold_start_dls to produce a WaypointResult per waypoint.

Speed scaling only rescales *time* (``duration_scaled = duration_original / speed_scale``,
carried through to each waypoint's reported ``time_s`` and to the realized control period used
for deadline metrics elsewhere); it never changes the Cartesian path geometry -- the target
positions/quaternions loaded from the trajectory NPZ are used unmodified regardless of
``speed_scale``.

This module only evaluates reachability at runtime; it never re-invokes or substitutes for the
generation-time reachability validation performed by generators/_trajectory_common.py.
"""

import csv
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np

from algorithms.cold_start_dls import run_cold_start_dls
from algorithms.result_types import WaypointResult, waypoint_results_to_dataframe
from algorithms.warm_start_dls import RawWaypointSolve, run_warm_start_dls
from kinematics.dls_solver import load_dls_config
from kinematics.forward_kinematics import forward_kinematics
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import minimum_joint_limit_margin
from kinematics.manipulability import yoshikawa_manipulability
from kinematics.model_loader import ModelContext, load_model_context
from kinematics.singularity_metrics import condition_number, minimum_singular_value
from utils.npz_utils import load_npz

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAJECTORIES_DIR = REPO_ROOT / "trajectories"
DEFAULT_MANIFEST_PATH = TRAJECTORIES_DIR / "trajectory_manifest.csv"
DEFAULT_TRIALS_PATH = TRAJECTORIES_DIR / "trajectory_trials.csv"

Method = Literal["warm_start", "cold_start"]

__all__ = [
    "waypoint_results_to_dataframe",
    "load_trajectory_manifest_row",
    "load_trial_row",
    "load_trajectory_waypoints",
    "validate_trial_against_trajectory",
    "run_sequential_trial",
]


def load_trajectory_manifest_row(trajectory_id: str, manifest_path: Optional[Path] = None) -> Dict[str, str]:
    """Read one trajectory_manifest.csv row by trajectory_id."""
    manifest_path = manifest_path or DEFAULT_MANIFEST_PATH
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["trajectory_id"] == trajectory_id:
                return row
    raise KeyError(f"trajectory_id '{trajectory_id}' not found in {manifest_path}")


def load_trial_row(trial_id: str, trials_path: Optional[Path] = None) -> Dict[str, str]:
    """Read one trajectory_trials.csv row by trial_id."""
    trials_path = trials_path or DEFAULT_TRIALS_PATH
    with open(trials_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["trial_id"] == trial_id:
                return row
    raise KeyError(f"trial_id '{trial_id}' not found in {trials_path}")


def load_trajectory_waypoints(trajectory_row: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Load the waypoint NPZ referenced by a trajectory_manifest.csv row."""
    return load_npz(REPO_ROOT / trajectory_row["file_path"])


def validate_trial_against_trajectory(
    trial_row: Dict[str, str],
    trajectory_row: Dict[str, str],
    model_context: ModelContext,
) -> np.ndarray:
    """Cross-validate a trial row against its trajectory row and the robot's operational limits.

    Returns the trial's parsed q_initial (7,) on success; raises ValueError on any mismatch.
    """
    if trial_row["trajectory_id"] != trajectory_row["trajectory_id"]:
        raise ValueError(
            f"trial trajectory_id '{trial_row['trajectory_id']}' does not match "
            f"trajectory row '{trajectory_row['trajectory_id']}'"
        )

    q_initial = np.array([float(trial_row[f"q{i}_init"]) for i in range(1, 8)], dtype=np.float64)
    if q_initial.shape != (model_context.nq,):
        raise ValueError(f"trial q_initial has shape {q_initial.shape}, expected ({model_context.nq},)")
    if not np.all(np.isfinite(q_initial)):
        raise ValueError("trial q_initial contains non-finite values")
    lower, upper = model_context.operational_lower_rad, model_context.operational_upper_rad
    if np.any(q_initial < lower - 1e-9) or np.any(q_initial > upper + 1e-9):
        raise ValueError("trial q_initial violates operational joint limits")

    trial_control_period_s = float(trial_row["control_period_s"])
    trajectory_control_period_s = float(trajectory_row["control_period_s"])
    if trial_control_period_s <= 0.0:
        raise ValueError("trial control_period_s must be positive")
    if not np.isclose(trial_control_period_s, trajectory_control_period_s, rtol=1e-6, atol=1e-9):
        raise ValueError(
            f"trial control_period_s ({trial_control_period_s}) does not match "
            f"trajectory control_period_s ({trajectory_control_period_s})"
        )

    speed_scale = float(trial_row["speed_scale"])
    if speed_scale <= 0.0:
        raise ValueError("trial speed_scale must be positive")

    return q_initial


def _build_waypoint_results(
    raw_solves: List[RawWaypointSolve],
    model_context: ModelContext,
    data,
    trial_id: str,
    trajectory_id: str,
    trial_category: str,
    method: str,
    time_s_scaled: np.ndarray,
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
) -> List[WaypointResult]:
    results = []
    for raw in raw_solves:
        k = raw.waypoint_index
        q_solution = raw.dls_result.q_solution
        fk = forward_kinematics(model_context, q_solution, data=data)
        J = geometric_jacobian_world(model_context, q_solution, data=data)
        sigma_min = minimum_singular_value(J)
        cond = condition_number(J)
        manipulability = yoshikawa_manipulability(J)
        margin = minimum_joint_limit_margin(
            q_solution, model_context.operational_lower_rad, model_context.operational_upper_rad
        )

        results.append(
            WaypointResult(
                trial_id=trial_id,
                trajectory_id=trajectory_id,
                trial_category=trial_category,
                method=method,
                waypoint_id=int(k),
                time_s=float(time_s_scaled[k]),
                q_initial_used=raw.q_initial_used.copy(),
                q_solution=q_solution.copy(),
                target_position=target_positions[k].copy(),
                actual_position=fk.position.copy(),
                target_quaternion=target_quaternions[k].copy(),
                actual_quaternion=fk.quaternion_wxyz.copy(),
                position_error_m=float(raw.dls_result.position_error_m),
                orientation_error_rad=float(raw.dls_result.orientation_error_rad),
                orientation_error_deg=float(raw.dls_result.orientation_error_deg),
                success=bool(raw.dls_result.success),
                iterations=int(raw.dls_result.iterations),
                solve_time_ms=float(raw.dls_result.solve_time_ms),
                sigma_min=float(sigma_min),
                condition_number=float(cond),
                manipulability=float(manipulability),
                minimum_joint_limit_margin=float(margin),
                recovered_after_previous_failure=bool(raw.recovered_after_previous_failure),
                failure_reason=raw.dls_result.failure_reason,
            )
        )
    return results


def run_sequential_trial(
    trajectory_id: str,
    trial_id: str,
    method: Method,
    model_context: Optional[ModelContext] = None,
    dls_config: Optional[dict] = None,
    waypoint_limit: Optional[int] = None,
    fail_fast: bool = False,
    show_progress: bool = True,
) -> List[WaypointResult]:
    """Run one (trajectory, trial, method) sequential-DLS combination end to end.

    Args:
        trajectory_id: Key into trajectory_manifest.csv.
        trial_id: Key into trajectory_trials.csv.
        method: 'warm_start' or 'cold_start'.
        waypoint_limit: If given, only the first N waypoints of the trajectory are solved
            (for smoke tests); this truncates the waypoint chain but does not change the
            control period or path geometry of the waypoints that are kept.
        fail_fast: Passed through to the warm/cold-start runner (see their docstrings).

    Returns:
        A list of WaypointResult, one per waypoint actually processed, in waypoint order.
        Deterministic for a given (trajectory_id, trial_id, method, dls_config, waypoint_limit).
    """
    if method not in ("warm_start", "cold_start"):
        raise ValueError(f"unknown method '{method}', expected 'warm_start' or 'cold_start'")

    model_context = model_context if model_context is not None else load_model_context()
    dls_config = dls_config if dls_config is not None else load_dls_config()

    trajectory_row = load_trajectory_manifest_row(trajectory_id)
    trial_row = load_trial_row(trial_id)
    q_trial_initial = validate_trial_against_trajectory(trial_row, trajectory_row, model_context)

    waypoints = load_trajectory_waypoints(trajectory_row)
    target_positions = waypoints["target_position"]
    target_quaternions = waypoints["target_quaternion"]
    time_s = waypoints["time_s"]

    if waypoint_limit is not None:
        target_positions = target_positions[:waypoint_limit]
        target_quaternions = target_quaternions[:waypoint_limit]
        time_s = time_s[:waypoint_limit]

    speed_scale = float(trial_row["speed_scale"])
    time_s_scaled = time_s / speed_scale

    if method == "warm_start":
        raw_solves = run_warm_start_dls(
            model_context, q_trial_initial, target_positions, target_quaternions, dls_config,
            fail_fast=fail_fast, show_progress=show_progress,
        )
    else:
        raw_solves = run_cold_start_dls(
            model_context, q_trial_initial, target_positions, target_quaternions, dls_config,
            fail_fast=fail_fast, show_progress=show_progress,
        )

    data = model_context.new_data()
    return _build_waypoint_results(
        raw_solves,
        model_context,
        data,
        trial_id=trial_id,
        trajectory_id=trajectory_id,
        trial_category=trial_row["trial_category"],
        method=method,
        time_s_scaled=time_s_scaled,
        target_positions=target_positions,
        target_quaternions=target_quaternions,
    )
