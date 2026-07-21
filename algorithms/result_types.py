"""Result dataclasses shared by algorithms/ (Tier 1 point DLS, Tier 2 sequential DLS) and the
evaluation modules that consume their output.

Vector-valued fields (q, position, quaternion) are kept as numpy arrays on the dataclasses
themselves (for in-process use), but every ``*_to_dataframe`` function here splits them into
separate scalar columns (``q1``..``q7``, ``x``/``y``/``z``, ``qw``/``qx``/``qy``/``qz``) before
writing to a DataFrame/CSV, so no cell ever holds a non-standard Python repr of an array.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from utils.csv_utils import json_safe_scalar

_Q_COLUMNS = [f"q{i}" for i in range(1, 8)]
_XYZ_COLUMNS = ["x", "y", "z"]
_QUAT_COLUMNS = ["qw", "qx", "qy", "qz"]


def _vector_columns(prefix: str, value: np.ndarray, names: List[str]) -> dict:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != len(names):
        raise ValueError(f"expected a {len(names)}-vector for '{prefix}', got shape {arr.shape}")
    return {f"{prefix}_{name}": float(v) for name, v in zip(names, arr)}


@dataclass
class PointIKResult:
    """Result of one independent Tier 1 point-IK solve (see algorithms.point_dls)."""

    sample_id: int
    difficulty_id: int
    success: bool
    q_initial: np.ndarray
    q_target_reference: np.ndarray
    q_solution: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    orientation_error_deg: float
    iterations: int
    solve_time_ms: float
    initial_sigma_min: float
    final_sigma_min: float
    initial_condition_number: float
    final_condition_number: float
    minimum_joint_limit_margin: float
    joint_limit_violation: bool
    failure_reason: Optional[str]


@dataclass
class WaypointResult:
    """Result of one waypoint solve within a Tier 2 sequential (warm/cold-start) DLS run."""

    trial_id: str
    trajectory_id: str
    trial_category: str
    method: str
    waypoint_id: int
    time_s: float
    q_initial_used: np.ndarray
    q_solution: np.ndarray
    target_position: np.ndarray
    actual_position: np.ndarray
    target_quaternion: np.ndarray
    actual_quaternion: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    orientation_error_deg: float
    success: bool
    iterations: int
    solve_time_ms: float
    sigma_min: float
    condition_number: float
    manipulability: float
    minimum_joint_limit_margin: float
    recovered_after_previous_failure: bool
    failure_reason: Optional[str]


def point_ik_results_to_dataframe(results: List[PointIKResult]) -> pd.DataFrame:
    """Flatten a list of PointIKResult into a DataFrame with a stable column schema."""
    rows = []
    for r in results:
        row = {
            "sample_id": int(r.sample_id),
            "difficulty_id": int(r.difficulty_id),
            "success": bool(r.success),
        }
        row.update(_vector_columns("q_initial", r.q_initial, _Q_COLUMNS))
        row.update(_vector_columns("q_target_reference", r.q_target_reference, _Q_COLUMNS))
        row.update(_vector_columns("q_solution", r.q_solution, _Q_COLUMNS))
        row.update(
            {
                "position_error_m": float(r.position_error_m),
                "orientation_error_rad": float(r.orientation_error_rad),
                "orientation_error_deg": float(r.orientation_error_deg),
                "iterations": int(r.iterations),
                "solve_time_ms": float(r.solve_time_ms),
                "initial_sigma_min": float(r.initial_sigma_min),
                "final_sigma_min": float(r.final_sigma_min),
                "initial_condition_number": float(r.initial_condition_number),
                "final_condition_number": float(r.final_condition_number),
                "minimum_joint_limit_margin": float(r.minimum_joint_limit_margin),
                "joint_limit_violation": bool(r.joint_limit_violation),
                "failure_reason": json_safe_scalar(r.failure_reason),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def waypoint_results_to_dataframe(results: List[WaypointResult]) -> pd.DataFrame:
    """Flatten a list of WaypointResult into a DataFrame with a stable column schema."""
    rows = []
    for r in results:
        row = {
            "trial_id": r.trial_id,
            "trajectory_id": r.trajectory_id,
            "trial_category": r.trial_category,
            "method": r.method,
            "waypoint_id": int(r.waypoint_id),
            "time_s": float(r.time_s),
        }
        row.update(_vector_columns("q_initial_used", r.q_initial_used, _Q_COLUMNS))
        row.update(_vector_columns("q_solution", r.q_solution, _Q_COLUMNS))
        row.update(_vector_columns("target_position", r.target_position, _XYZ_COLUMNS))
        row.update(_vector_columns("actual_position", r.actual_position, _XYZ_COLUMNS))
        row.update(_vector_columns("target_quaternion", r.target_quaternion, _QUAT_COLUMNS))
        row.update(_vector_columns("actual_quaternion", r.actual_quaternion, _QUAT_COLUMNS))
        row.update(
            {
                "position_error_m": float(r.position_error_m),
                "orientation_error_rad": float(r.orientation_error_rad),
                "orientation_error_deg": float(r.orientation_error_deg),
                "success": bool(r.success),
                "iterations": int(r.iterations),
                "solve_time_ms": float(r.solve_time_ms),
                "sigma_min": float(r.sigma_min),
                "condition_number": float(r.condition_number),
                "manipulability": float(r.manipulability),
                "minimum_joint_limit_margin": float(r.minimum_joint_limit_margin),
                "recovered_after_previous_failure": bool(r.recovered_after_previous_failure),
                "failure_reason": json_safe_scalar(r.failure_reason),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
