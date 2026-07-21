"""Aggregated per-trial result dataclass, combining waypoint/trajectory/runtime/singularity metrics.

Unlike algorithms.result_types (raw per-sample/per-waypoint solver output), TrajectoryTrialSummary
is a Tier 2-4 evaluation-level rollup: one row per (trial_id, method), built by combining the
outputs of evaluation.waypoint_metrics, evaluation.trajectory_metrics, evaluation.runtime_metrics,
and evaluation.singularity_metrics for that trial.
"""

from dataclasses import dataclass
from typing import List

import pandas as pd

from utils.csv_utils import json_safe_scalar


@dataclass
class TrajectoryTrialSummary:
    """One row per (trial_id, method): the Tier 2-4 aggregate result for a sequential-DLS run."""

    trial_id: str
    trajectory_id: str
    trial_category: str
    method: str
    repeat_id: int
    seed: int
    speed_scale: float
    control_period_s: float
    waypoint_count: int
    successful_waypoints: int
    failed_waypoints: int
    waypoint_success_rate: float
    full_trajectory_completed: bool
    maximum_failure_streak: int
    recovery_rate: float
    position_rmse_m: float
    position_mae_m: float
    position_median_m: float
    position_p95_m: float
    position_max_m: float
    orientation_rmse_deg: float
    orientation_p95_deg: float
    orientation_max_deg: float
    mean_iterations: float
    p95_iterations: float
    mean_solve_time_ms: float
    p95_solve_time_ms: float
    deadline_miss_rate: float
    minimum_sigma_min: float
    maximum_condition_number: float
    minimum_joint_limit_margin: float


def trajectory_trial_summaries_to_dataframe(summaries: List[TrajectoryTrialSummary]) -> pd.DataFrame:
    """Flatten a list of TrajectoryTrialSummary into a DataFrame with a stable column schema."""
    rows = []
    for s in summaries:
        rows.append({field: json_safe_scalar(getattr(s, field)) for field in TrajectoryTrialSummary.__dataclass_fields__})
    return pd.DataFrame(rows, columns=list(TrajectoryTrialSummary.__dataclass_fields__))
