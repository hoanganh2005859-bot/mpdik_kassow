"""Tier 2: per-waypoint sequential IK success rate and error metrics along a trajectory.

Deliberately keeps two distinct concepts separate:

- ``full_trajectory_completed``: the runner (algorithms.sequential_dls) processed every waypoint
  it was asked to and never hit a fatal (non-finite) numerical failure. Compares the number of
  results actually produced against ``expected_waypoint_count``.
- ``waypoint_success_rate``: the fraction of *processed* waypoints whose individual DLS solve met
  the configured position/orientation success thresholds.

A trajectory can be fully processed (``full_trajectory_completed=True``) while still having a
waypoint_success_rate below 1.0 -- these are not the same thing and must not be conflated.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from algorithms.result_types import WaypointResult
from evaluation.confidence_intervals import WilsonInterval, wilson_confidence_interval


@dataclass
class WaypointMetricsSummary:
    """Aggregate Tier 2 metrics for one (trial_id, method) sequential-DLS run."""

    waypoint_count: int
    successful_waypoints: int
    failed_waypoints: int
    waypoint_success_rate: float
    waypoint_success_rate_wilson_ci: WilsonInterval
    full_trajectory_completed: bool
    maximum_failure_streak: int
    number_of_failure_streaks: int
    recovery_attempts: int
    successful_recoveries: int
    recovery_rate: float
    mean_iterations: float
    p95_iterations: float
    mean_runtime_ms: float
    p95_runtime_ms: float
    deadline_miss_count: Optional[int]
    deadline_miss_rate: Optional[float]


def _failure_streaks(success_mask: np.ndarray):
    max_streak = 0
    n_streaks = 0
    current = 0
    for success in success_mask:
        if success:
            current = 0
        else:
            if current == 0:
                n_streaks += 1
            current += 1
            max_streak = max(max_streak, current)
    return max_streak, n_streaks


def _recovery_counts(success_mask: np.ndarray):
    attempts = 0
    successful = 0
    for i in range(1, len(success_mask)):
        if not success_mask[i - 1]:
            attempts += 1
            if success_mask[i]:
                successful += 1
    return attempts, successful


def compute_waypoint_metrics(
    results: List[WaypointResult],
    expected_waypoint_count: int,
    confidence_level: float = 0.95,
    deadline_s: Optional[float] = None,
) -> WaypointMetricsSummary:
    """Compute Tier 2 waypoint-level metrics for one sequential-DLS run's results.

    Args:
        results: WaypointResult list for a single (trial_id, method) run, in waypoint order.
        expected_waypoint_count: Number of waypoints the runner was asked to process; used to
            decide ``full_trajectory_completed`` (see module docstring).
        deadline_s: If given, per-waypoint solve_time_ms is compared against
            ``deadline_s * 1000`` to compute deadline-miss statistics; if omitted, the deadline
            fields are None (not silently zero).
    """
    if len(results) == 0:
        raise ValueError("cannot compute waypoint metrics for an empty result set")
    if expected_waypoint_count <= 0:
        raise ValueError("expected_waypoint_count must be positive")

    success_mask = np.array([r.success for r in results], dtype=bool)
    iterations = np.array([r.iterations for r in results], dtype=np.float64)
    runtimes_ms = np.array([r.solve_time_ms for r in results], dtype=np.float64)

    successful = int(np.sum(success_mask))
    total = len(results)
    max_streak, n_streaks = _failure_streaks(success_mask)
    recovery_attempts, successful_recoveries = _recovery_counts(success_mask)
    recovery_rate = float(successful_recoveries / recovery_attempts) if recovery_attempts > 0 else float("nan")

    if deadline_s is not None:
        deadline_ms = deadline_s * 1000.0
        deadline_miss_count = int(np.sum(runtimes_ms > deadline_ms))
        deadline_miss_rate = deadline_miss_count / total
    else:
        deadline_miss_count = None
        deadline_miss_rate = None

    return WaypointMetricsSummary(
        waypoint_count=total,
        successful_waypoints=successful,
        failed_waypoints=total - successful,
        waypoint_success_rate=successful / total,
        waypoint_success_rate_wilson_ci=wilson_confidence_interval(successful, total, confidence_level),
        full_trajectory_completed=bool(total == expected_waypoint_count),
        maximum_failure_streak=max_streak,
        number_of_failure_streaks=n_streaks,
        recovery_attempts=recovery_attempts,
        successful_recoveries=successful_recoveries,
        recovery_rate=recovery_rate,
        mean_iterations=float(np.mean(iterations)),
        p95_iterations=float(np.percentile(iterations, 95)),
        mean_runtime_ms=float(np.mean(runtimes_ms)),
        p95_runtime_ms=float(np.percentile(runtimes_ms, 95)),
        deadline_miss_count=deadline_miss_count,
        deadline_miss_rate=deadline_miss_rate,
    )
