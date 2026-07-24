"""Tier 2 public trajectory (sequential DLS) evaluator: warm-start and cold-start.

Input per trial is exactly the trial's ``q_initial`` + the public canonical target position /
quaternion sequence + public timing (task section 6). ``q_reference`` is absent from the public
trajectory NPZ and never consulted.

Warm-start / cold-start semantics (reused unchanged from ``algorithms/``):

* **warm_start** -- waypoint 0 solves from the trial ``q_initial``; waypoint ``i>0`` solves from
  the previously *accepted* solution (with the documented finite-state recovery policy on a
  failed waypoint). Ordinary waypoint failures never stop the chain.
* **cold_start** -- every waypoint solves from the same fixed trial ``q_initial``; no
  cross-waypoint continuity.

Both methods use the SAME target sequence, trial, DLS candidate config, iteration cap, and
convergence tolerance -- the only difference is the seed configuration per waypoint.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from algorithms.cold_start_dls import run_cold_start_dls
from algorithms.warm_start_dls import run_warm_start_dls
from evaluation_v2.candidate_configs import CandidateConfig
from evaluation_v2.protected_guard import assert_no_protected_fields
from evaluation_v2.reporting import success_columns
from kinematics.forward_kinematics import forward_kinematics
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import minimum_joint_limit_margin
from kinematics.manipulability import yoshikawa_manipulability
from kinematics.model_loader import ModelContext, load_model_context
from kinematics.singularity_metrics import condition_number, minimum_singular_value
from utils.npz_utils import load_npz

_Q_COLS = [f"q{i}" for i in range(1, 8)]
_XYZ = ["x", "y", "z"]
_QUAT = ["qw", "qx", "qy", "qz"]


def load_public_trajectory_arrays(public_canonical_file) -> Dict[str, np.ndarray]:
    """Load a public canonical trajectory NPZ and assert it carries no protected array."""
    arrays = load_npz(public_canonical_file)
    assert_no_protected_fields(arrays, f"public trajectory NPZ {public_canonical_file}")
    return arrays


def evaluate_trajectory_trial(
    public_canonical_file,
    *,
    trial_id: str,
    trajectory_id: str,
    trajectory_family: str,
    difficulty: str,
    split: str,
    q_initial: np.ndarray,
    method: str,
    candidate: CandidateConfig,
    model_context: Optional[ModelContext] = None,
    waypoint_limit: Optional[int] = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Run one (trial, method) over its trajectory and return a per-waypoint DataFrame."""
    if method not in ("warm_start", "cold_start"):
        raise ValueError(f"unknown method '{method}'")

    arrays = load_public_trajectory_arrays(public_canonical_file)
    target_positions = np.asarray(arrays["target_position"], dtype=np.float64)
    target_quaternions = np.asarray(arrays["target_quaternion"], dtype=np.float64)
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    if waypoint_limit is not None:
        target_positions = target_positions[:waypoint_limit]
        target_quaternions = target_quaternions[:waypoint_limit]
        time_s = time_s[:waypoint_limit]

    model_context = model_context or load_model_context()
    cfg = candidate.solver_config()
    q_initial = np.asarray(q_initial, dtype=np.float64)

    runner = run_warm_start_dls if method == "warm_start" else run_cold_start_dls
    raw_solves = runner(
        model_context, q_initial, target_positions, target_quaternions, cfg,
        fail_fast=False, show_progress=show_progress,
    )

    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    data = model_context.new_data()

    rows = []
    for raw in raw_solves:
        k = raw.waypoint_index
        q_sol = raw.dls_result.q_solution
        fk = forward_kinematics(model_context, q_sol, data=data)
        J = geometric_jacobian_world(model_context, q_sol, data=data)
        row = {
            "trial_id": trial_id,
            "trajectory_id": trajectory_id,
            "trajectory_family": trajectory_family,
            "difficulty": difficulty,
            "split": split,
            "method": method,
            "waypoint_id": int(k),
            "time_s": float(time_s[k]),
            "position_error_m": float(raw.dls_result.position_error_m),
            "orientation_error_rad": float(raw.dls_result.orientation_error_rad),
            "orientation_error_deg": float(raw.dls_result.orientation_error_deg),
            "converged": bool(raw.dls_result.success),
            "iterations": int(raw.dls_result.iterations),
            "solve_time_ms": float(raw.dls_result.solve_time_ms),
            "sigma_min": float(minimum_singular_value(J)),
            "condition_number": float(condition_number(J)),
            "manipulability": float(yoshikawa_manipulability(J)),
            "minimum_joint_limit_margin": float(minimum_joint_limit_margin(q_sol, lower, upper)),
            "recovered_after_previous_failure": bool(raw.recovered_after_previous_failure),
            "failure_reason": raw.dls_result.failure_reason or "",
        }
        row.update(success_columns(row["position_error_m"], row["orientation_error_deg"]))
        row.update({f"target_position_{a}": float(v) for a, v in zip(_XYZ, target_positions[k])})
        row.update({f"actual_position_{a}": float(v) for a, v in zip(_XYZ, fk.position)})
        row.update({f"target_quaternion_{a}": float(v) for a, v in zip(_QUAT, target_quaternions[k])})
        row.update({f"actual_quaternion_{a}": float(v) for a, v in zip(_QUAT, fk.quaternion_wxyz)})
        row.update({f"q_solution_{c}": float(v) for c, v in zip(_Q_COLS, np.asarray(q_sol, float))})
        rows.append(row)

    return pd.DataFrame(rows)
