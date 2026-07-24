"""Tier 1 public Point-IK DLS evaluator.

Input per sample is exactly ``q_initial`` + ``target_position`` + ``target_quaternion`` (task
section 6). No reference joint configuration is read or recorded: ``q_target_reference`` is not in
the public export, and this module never stores any reference solution -- only the solver's own
achieved ``q_solution``.

Produces a per-sample DataFrame (one row per sample) with convergence + coarse/standard/strict
reporting success, achieved errors, iteration/runtime, and public solver diagnostics.
"""

from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluation_v2.candidate_configs import CandidateConfig
from evaluation_v2.protected_guard import assert_no_protected_fields
from evaluation_v2.reporting import success_columns
from kinematics.dls_solver import solve_dls_until_converged
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import (
    minimum_joint_limit_margin,
    operational_limit_violation_mask,
)
from kinematics.model_loader import ModelContext, load_model_context
from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from kinematics.singularity_metrics import condition_number, minimum_singular_value
from utils.npz_utils import load_npz

_Q_COLS = [f"q{i}" for i in range(1, 8)]


def evaluate_point_ik_split(
    public_point_ik_file,
    candidate: CandidateConfig,
    *,
    model_context: Optional[ModelContext] = None,
    sample_limit: Optional[int] = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Solve every (selected) sample of a public Point-IK split and return a per-sample DataFrame."""
    raw = load_npz(public_point_ik_file)
    assert_no_protected_fields(raw, f"public point-IK NPZ {public_point_ik_file}")

    model_context = model_context or load_model_context()
    cfg = candidate.solver_config()
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    data = model_context.new_data()

    n = int(raw["sample_id"].shape[0])
    indices = range(n if sample_limit is None else min(int(sample_limit), n))

    rows = []
    for idx in tqdm(indices, desc=f"point_ik/{candidate.candidate_id}", disable=not show_progress):
        q_initial = np.asarray(raw["q_initial"][idx], dtype=np.float64)
        target_position = np.asarray(raw["target_position"][idx], dtype=np.float64)
        target_quaternion = np.asarray(raw["target_quaternion_wxyz"][idx], dtype=np.float64)
        target_rotation = quaternion_wxyz_to_matrix(target_quaternion)

        J_init = geometric_jacobian_world(model_context, q_initial, data=data)
        initial_sigma_min = float(minimum_singular_value(J_init))

        result = solve_dls_until_converged(
            model_context, q_initial, target_position, target_rotation, config=cfg
        )
        q_sol = result.q_solution
        J_final = geometric_jacobian_world(model_context, q_sol, data=data)
        final_sigma_min = float(minimum_singular_value(J_final))
        final_condition_number = float(condition_number(J_final))
        margin = float(minimum_joint_limit_margin(q_sol, lower, upper))
        violation = bool(np.any(operational_limit_violation_mask(q_sol, lower, upper)))

        row = {
            "sample_id": str(raw["sample_id"][idx]),
            "difficulty_id": int(raw["difficulty_id"][idx]),
            "converged": bool(result.success),
            "position_error_m": float(result.position_error_m),
            "orientation_error_rad": float(result.orientation_error_rad),
            "orientation_error_deg": float(result.orientation_error_deg),
            "iterations": int(result.iterations),
            "solve_time_ms": float(result.solve_time_ms),
            "initial_sigma_min": initial_sigma_min,
            "final_sigma_min": final_sigma_min,
            "final_condition_number": final_condition_number,
            "minimum_joint_limit_margin": margin,
            "joint_limit_violation": violation,
            "failure_reason": result.failure_reason or "",
        }
        row.update(success_columns(row["position_error_m"], row["orientation_error_deg"]))
        row.update({f"q_solution_{c}": float(v) for c, v in zip(_Q_COLS, np.asarray(q_sol, float))})
        rows.append(row)

    return pd.DataFrame(rows)
