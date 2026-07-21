"""Evaluation-level aggregation of singularity proximity metrics across a trajectory.

Wraps kinematics.singularity_metrics / kinematics.manipulability (the core, tested singular
value / manipulability computations) rather than reimplementing them; this module only adds the
per-waypoint iteration and trajectory-level aggregation on top.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from kinematics.jacobian import geometric_jacobian_world
from kinematics.manipulability import positional_manipulability, yoshikawa_manipulability
from kinematics.model_loader import ModelContext
from kinematics.singularity_metrics import (
    condition_number,
    is_near_singular,
    numerical_rank,
    singular_values,
)


@dataclass
class WaypointSingularityMetrics:
    """Singularity proximity metrics at one joint configuration along a trajectory."""

    waypoint_index: int
    sigma_min: float
    sigma_max: float
    condition_number: float
    numerical_rank: int
    yoshikawa_manipulability: float
    positional_manipulability: float
    near_singular: bool


@dataclass
class TrajectorySingularitySummary:
    """Aggregate singularity proximity summary over a full trajectory."""

    waypoint_count: int
    minimum_sigma_min: float
    p05_sigma_min: float
    maximum_condition_number: float
    near_singular_count: int
    near_singular_fraction: float
    worst_waypoint_index: int


def compute_singularity_metrics_for_trajectory(
    model_context: ModelContext,
    q_trajectory: np.ndarray,
    near_singular_threshold: float,
):
    """Compute per-waypoint and aggregate singularity metrics for a joint trajectory.

    Returns (per_waypoint: List[WaypointSingularityMetrics], summary: TrajectorySingularitySummary).
    """
    q_trajectory = np.asarray(q_trajectory, dtype=np.float64)
    if q_trajectory.ndim != 2:
        raise ValueError(f"expected q_trajectory of shape (N, J), got {q_trajectory.shape}")
    n = q_trajectory.shape[0]
    if n == 0:
        raise ValueError("q_trajectory must have at least one waypoint")

    data = model_context.new_data()
    per_waypoint: List[WaypointSingularityMetrics] = []
    sigma_mins = np.empty(n, dtype=np.float64)

    for k in range(n):
        J = geometric_jacobian_world(model_context, q_trajectory[k], data=data)
        sv = singular_values(J)
        sigma_min = float(sv[-1])
        sigma_mins[k] = sigma_min
        per_waypoint.append(
            WaypointSingularityMetrics(
                waypoint_index=k,
                sigma_min=sigma_min,
                sigma_max=float(sv[0]),
                condition_number=condition_number(J),
                numerical_rank=numerical_rank(J),
                yoshikawa_manipulability=yoshikawa_manipulability(J),
                positional_manipulability=positional_manipulability(J),
                near_singular=is_near_singular(J, near_singular_threshold),
            )
        )

    condition_numbers = np.array([w.condition_number for w in per_waypoint], dtype=np.float64)
    near_singular_count = int(sum(1 for w in per_waypoint if w.near_singular))
    worst_idx = int(np.argmin(sigma_mins))

    summary = TrajectorySingularitySummary(
        waypoint_count=n,
        minimum_sigma_min=float(np.min(sigma_mins)),
        p05_sigma_min=float(np.percentile(sigma_mins, 5)),
        maximum_condition_number=float(np.max(condition_numbers)),
        near_singular_count=near_singular_count,
        near_singular_fraction=near_singular_count / n,
        worst_waypoint_index=worst_idx,
    )
    return per_waypoint, summary
