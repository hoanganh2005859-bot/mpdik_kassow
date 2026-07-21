"""Simulation-adapted / ISO 9283-inspired path accuracy and repeatability metrics.

These are project-defined, simulation-adapted metrics *inspired by* ISO 9283's pose accuracy
and repeatability definitions, adapted from single-pose to a full commanded path. They are
**not** an ISO 9283 certification and no such certification is claimed (see
configs/evaluation_config.json and DATASET_MANIFEST.json for the same caveat).

Only repeatability trials (trajectories/trajectory_trials.csv trial_category="repeatability") --
the same trajectory, same q_initial, same speed_scale, same method, distinguished only by
repeat_id -- may be used here. Robustness trials (different q_initial per trial) are rejected,
since ISO 9283-style repeatability is specifically about repeated attempts under *identical*
commanded conditions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from algorithms.result_types import WaypointResult

MIN_REPEATS_FOR_STD = 2
RECOMMENDED_MIN_REPEATS = 10


@dataclass
class PathAccuracyResult:
    """ISO 9283-inspired path accuracy (ATp): worst-case deviation of the mean attained path
    from the commanded path, across repeated runs of the same commanded trajectory."""

    atp_m: float
    atp_waypoint_index: int
    mean_deviation_m: float
    rmse_deviation_m: float
    p95_deviation_m: float
    waypoint_count: int


def compute_path_accuracy(commanded_positions: np.ndarray, repeated_actual_positions: np.ndarray) -> PathAccuracyResult:
    """Compute ATp from a commanded path and a stack of repeated actual position sequences.

    Args:
        commanded_positions: (M, 3) commanded/target positions.
        repeated_actual_positions: (R, M, 3) achieved positions, one (M, 3) sequence per repeat.
    """
    commanded_positions = np.asarray(commanded_positions, dtype=np.float64)
    repeated_actual_positions = np.asarray(repeated_actual_positions, dtype=np.float64)
    if repeated_actual_positions.ndim != 3 or repeated_actual_positions.shape[2] != 3:
        raise ValueError(f"expected repeated_actual_positions of shape (R, M, 3), got {repeated_actual_positions.shape}")
    if commanded_positions.shape != repeated_actual_positions.shape[1:]:
        raise ValueError(
            f"commanded_positions shape {commanded_positions.shape} does not match "
            f"repeated_actual_positions waypoint shape {repeated_actual_positions.shape[1:]}"
        )

    mean_attained = np.mean(repeated_actual_positions, axis=0)  # (M, 3)
    deviation = np.linalg.norm(commanded_positions - mean_attained, axis=1)  # (M,)
    atp_idx = int(np.argmax(deviation))

    return PathAccuracyResult(
        atp_m=float(deviation[atp_idx]),
        atp_waypoint_index=atp_idx,
        mean_deviation_m=float(np.mean(deviation)),
        rmse_deviation_m=float(np.sqrt(np.mean(deviation**2))),
        p95_deviation_m=float(np.percentile(deviation, 95)),
        waypoint_count=int(commanded_positions.shape[0]),
    )


@dataclass
class PathRepeatabilityResult:
    """ISO 9283-inspired path repeatability (RTp): worst-case (r_bar + 3*s) radial spread of
    repeated runs around their own mean attained path."""

    rtp_m: float
    rtp_waypoint_index: int
    maximum_radial_spread_m: float
    n_repeats: int
    warning: Optional[str]


def compute_path_repeatability(repeated_actual_positions: np.ndarray) -> PathRepeatabilityResult:
    """Compute RTp = max_j(r_bar_j + 3 * s_r,j) from a stack of repeated actual position sequences.

    Args:
        repeated_actual_positions: (R, M, 3) achieved positions, one (M, 3) sequence per repeat.
            Requires R >= 2 (sample standard deviation is undefined for a single repeat); a
            deterministic kinematic simulation with no injected noise is expected to give an
            RTp near 0 -- that is the expected result here, not a sign of a broken metric.
    """
    repeated_actual_positions = np.asarray(repeated_actual_positions, dtype=np.float64)
    if repeated_actual_positions.ndim != 3 or repeated_actual_positions.shape[2] != 3:
        raise ValueError(f"expected repeated_actual_positions of shape (R, M, 3), got {repeated_actual_positions.shape}")

    n_repeats = repeated_actual_positions.shape[0]
    if n_repeats < MIN_REPEATS_FOR_STD:
        raise ValueError(f"need at least {MIN_REPEATS_FOR_STD} repeats to compute a sample standard deviation, got {n_repeats}")

    mean_attained = np.mean(repeated_actual_positions, axis=0)  # (M, 3)
    radial = np.linalg.norm(repeated_actual_positions - mean_attained[None, :, :], axis=2)  # (R, M)
    r_bar = np.mean(radial, axis=0)  # (M,)
    s_r = np.std(radial, axis=0, ddof=1)  # (M,)
    rtp_per_waypoint = r_bar + 3.0 * s_r
    idx = int(np.argmax(rtp_per_waypoint))

    warning = None
    if n_repeats < RECOMMENDED_MIN_REPEATS:
        warning = (
            f"only {n_repeats} repeats available; ISO 9283-inspired repeatability is conventionally "
            f"reported from >= {RECOMMENDED_MIN_REPEATS} repeats (metric is still computed, not withheld)"
        )

    return PathRepeatabilityResult(
        rtp_m=float(rtp_per_waypoint[idx]),
        rtp_waypoint_index=idx,
        maximum_radial_spread_m=float(np.max(radial)),
        n_repeats=int(n_repeats),
        warning=warning,
    )


@dataclass
class ISO9283Result:
    accuracy: PathAccuracyResult
    repeatability: PathRepeatabilityResult


def build_repeatability_group(results_by_repeat: Dict[int, List[WaypointResult]]) -> Tuple[np.ndarray, np.ndarray]:
    """Validate a {repeat_id: WaypointResult list} group and extract ISO input arrays.

    Enforces: every group member is a repeatability trial (never robustness), every repeat has
    the same waypoint count, and every repeat commands the same target path (no silent
    resampling/realignment). Returns (commanded_positions [M,3], repeated_actual_positions [R,M,3]).
    """
    if len(results_by_repeat) < MIN_REPEATS_FOR_STD:
        raise ValueError(f"need at least {MIN_REPEATS_FOR_STD} repeat_id groups, got {len(results_by_repeat)}")

    repeat_ids = sorted(results_by_repeat.keys())
    reference_positions = None
    waypoint_count = None
    actual_stack = []

    for repeat_id in repeat_ids:
        results = results_by_repeat[repeat_id]
        if len(results) == 0:
            raise ValueError(f"repeat_id {repeat_id} has no waypoint results")
        if any(r.trial_category != "repeatability" for r in results):
            raise ValueError(
                f"repeat_id {repeat_id} contains non-repeatability (robustness) trial results; "
                "robustness trials cannot be used for ISO 9283-inspired repeatability calculation"
            )

        if waypoint_count is None:
            waypoint_count = len(results)
        elif len(results) != waypoint_count:
            raise ValueError(
                f"mismatched waypoint count across repeats: repeat_id {repeat_id} has {len(results)}, expected {waypoint_count}"
            )

        target = np.array([r.target_position for r in results], dtype=np.float64)
        if reference_positions is None:
            reference_positions = target
        elif not np.allclose(reference_positions, target, atol=1e-9):
            raise ValueError(f"mismatched commanded target path across repeats at repeat_id {repeat_id}")

        actual_stack.append(np.array([r.actual_position for r in results], dtype=np.float64))

    return reference_positions, np.stack(actual_stack, axis=0)


def compute_iso9283_metrics(results_by_repeat: Dict[int, List[WaypointResult]]) -> ISO9283Result:
    """Compute simulation-adapted ISO 9283-inspired ATp/RTp from a repeatability trial group."""
    commanded_positions, repeated_actual_positions = build_repeatability_group(results_by_repeat)
    accuracy = compute_path_accuracy(commanded_positions, repeated_actual_positions)
    repeatability = compute_path_repeatability(repeated_actual_positions)
    return ISO9283Result(accuracy=accuracy, repeatability=repeatability)
