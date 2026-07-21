"""Tier 1: per-point IK success rate, position/orientation error, and iteration-count metrics.

Success is never recomputed here against a different threshold: it is taken verbatim from each
PointIKResult.success, which was already decided by kinematics.dls_solver against
configs/dls_config.json's position_success_threshold_m / orientation_success_threshold_deg.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from algorithms.result_types import PointIKResult
from evaluation.confidence_intervals import WilsonInterval, wilson_confidence_interval

DEFAULT_SIGMA_MIN_BINS = 5


@dataclass
class SigmaMinBin:
    """Success rate and mean position error within one bin of final_sigma_min."""

    bin_index: int
    sigma_min_low: float
    sigma_min_high: float
    count: int
    success_rate: float
    mean_position_error_m: float


@dataclass
class PointIKGroupMetrics:
    """Aggregate Tier 1 point-IK metrics for one difficulty group (or the whole benchmark)."""

    sample_count: int
    success_count: int
    success_rate: float
    success_rate_wilson_ci: WilsonInterval
    position_rmse_m: float
    position_mae_m: float
    position_median_m: float
    position_p95_m: float
    position_max_m: float
    orientation_rmse_deg: float
    orientation_mae_deg: float
    orientation_median_deg: float
    orientation_p95_deg: float
    orientation_max_deg: float
    mean_iterations: float
    median_iterations: float
    p95_iterations: float
    mean_solve_time_ms: float
    median_solve_time_ms: float
    p95_solve_time_ms: float
    failure_reason_counts: Dict[str, int]
    joint_limit_violation_rate: float
    success_by_sigma_min_bin: List[SigmaMinBin] = field(default_factory=list)


def _summarize_group(
    results: List[PointIKResult], confidence_level: float, n_sigma_min_bins: int
) -> PointIKGroupMetrics:
    if len(results) == 0:
        raise ValueError("cannot summarize an empty group of PointIKResult")

    success_mask = np.array([r.success for r in results], dtype=bool)
    position_errors = np.array([r.position_error_m for r in results], dtype=np.float64)
    orientation_errors_deg = np.array([r.orientation_error_deg for r in results], dtype=np.float64)
    iterations = np.array([r.iterations for r in results], dtype=np.float64)
    solve_times = np.array([r.solve_time_ms for r in results], dtype=np.float64)
    joint_limit_violations = np.array([r.joint_limit_violation for r in results], dtype=bool)
    final_sigma_min = np.array([r.final_sigma_min for r in results], dtype=np.float64)

    success_count = int(np.sum(success_mask))
    sample_count = len(results)

    failure_reason_counts = Counter(r.failure_reason for r in results if r.failure_reason is not None)

    return PointIKGroupMetrics(
        sample_count=sample_count,
        success_count=success_count,
        success_rate=success_count / sample_count,
        success_rate_wilson_ci=wilson_confidence_interval(success_count, sample_count, confidence_level),
        position_rmse_m=float(np.sqrt(np.mean(position_errors**2))),
        position_mae_m=float(np.mean(position_errors)),
        position_median_m=float(np.median(position_errors)),
        position_p95_m=float(np.percentile(position_errors, 95)),
        position_max_m=float(np.max(position_errors)),
        orientation_rmse_deg=float(np.sqrt(np.mean(orientation_errors_deg**2))),
        orientation_mae_deg=float(np.mean(np.abs(orientation_errors_deg))),
        orientation_median_deg=float(np.median(orientation_errors_deg)),
        orientation_p95_deg=float(np.percentile(orientation_errors_deg, 95)),
        orientation_max_deg=float(np.max(orientation_errors_deg)),
        mean_iterations=float(np.mean(iterations)),
        median_iterations=float(np.median(iterations)),
        p95_iterations=float(np.percentile(iterations, 95)),
        mean_solve_time_ms=float(np.mean(solve_times)),
        median_solve_time_ms=float(np.median(solve_times)),
        p95_solve_time_ms=float(np.percentile(solve_times, 95)),
        failure_reason_counts=dict(failure_reason_counts),
        joint_limit_violation_rate=float(np.mean(joint_limit_violations)),
        success_by_sigma_min_bin=_bin_by_sigma_min(final_sigma_min, success_mask, position_errors, n_sigma_min_bins),
    )


def _bin_by_sigma_min(
    sigma_min: np.ndarray, success_mask: np.ndarray, position_errors: np.ndarray, n_bins: int
) -> List[SigmaMinBin]:
    if n_bins <= 0 or sigma_min.shape[0] == 0:
        return []

    edges = np.quantile(sigma_min, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.shape[0] < 2:
        edges = np.array([float(sigma_min.min()), float(sigma_min.max()) + 1e-12])

    bins = []
    for i in range(edges.shape[0] - 1):
        low, high = edges[i], edges[i + 1]
        if i == edges.shape[0] - 2:
            mask = (sigma_min >= low) & (sigma_min <= high)
        else:
            mask = (sigma_min >= low) & (sigma_min < high)
        count = int(np.sum(mask))
        if count == 0:
            continue
        bins.append(
            SigmaMinBin(
                bin_index=i,
                sigma_min_low=float(low),
                sigma_min_high=float(high),
                count=count,
                success_rate=float(np.mean(success_mask[mask])),
                mean_position_error_m=float(np.mean(position_errors[mask])),
            )
        )
    return bins


def compute_point_ik_metrics(
    results: List[PointIKResult],
    confidence_level: float = 0.95,
    n_sigma_min_bins: int = DEFAULT_SIGMA_MIN_BINS,
) -> Dict[str, PointIKGroupMetrics]:
    """Compute overall and per-difficulty-group Tier 1 point-IK metrics.

    Returns a dict with key "overall" plus one key per observed difficulty_id (as a string,
    e.g. "0"), each mapping to a PointIKGroupMetrics.
    """
    if len(results) == 0:
        raise ValueError("cannot compute point-IK metrics for an empty result set")

    metrics = {"overall": _summarize_group(results, confidence_level, n_sigma_min_bins)}

    by_group: Dict[int, List[PointIKResult]] = {}
    for r in results:
        by_group.setdefault(r.difficulty_id, []).append(r)

    for difficulty_id in sorted(by_group):
        metrics[str(difficulty_id)] = _summarize_group(by_group[difficulty_id], confidence_level, n_sigma_min_bins)

    return metrics
