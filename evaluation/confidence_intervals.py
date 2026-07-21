"""Wilson score and bootstrap confidence intervals used across Tier 1-4 evaluation metrics.

Wilson intervals are used for binomial success-rate proportions (e.g. point-IK success rate,
waypoint success rate); this module implements the Wilson score interval specifically (never
the simpler but poorly-calibrated Wald/normal-approximation interval).

Bootstrap intervals are used for trial-level scalar metrics (e.g. position RMSE per trajectory
trial). Per configs/evaluation_config.json, resampling always happens at the trial level -- this
module never resamples individual waypoints independently within a single trial, since waypoint
errors within one trial are not independent observations.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class WilsonInterval:
    """Wilson score confidence interval for a binomial success proportion."""

    successes: int
    trials: int
    proportion: float
    lower: float
    upper: float
    confidence_level: float


def wilson_confidence_interval(successes: int, trials: int, confidence_level: float = 0.95) -> WilsonInterval:
    """Wilson score interval for ``successes`` out of ``trials`` at ``confidence_level``.

    Handles trials=0 explicitly (proportion/lower/upper are NaN) rather than dividing by zero.
    """
    successes = int(successes)
    trials = int(trials)
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"invalid successes/trials: {successes}/{trials}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    if trials == 0:
        return WilsonInterval(0, 0, float("nan"), float("nan"), float("nan"), confidence_level)

    z = float(stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))
    n = float(trials)
    p = successes / n
    z2 = z * z

    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half_width = (z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom

    lower = float(np.clip(center - half_width, 0.0, 1.0))
    upper = float(np.clip(center + half_width, 0.0, 1.0))
    return WilsonInterval(successes, trials, p, lower, upper, confidence_level)


@dataclass
class BootstrapInterval:
    """Percentile bootstrap confidence interval for a mean or median statistic."""

    estimate: float
    lower: float
    upper: float
    n_resamples: int
    statistic: str
    confidence_level: float
    sample_size: int


_STATISTIC_FNS = {"mean": np.mean, "median": np.median}


def bootstrap_confidence_interval(
    data: Sequence[float],
    statistic: str = "mean",
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile bootstrap CI for the mean/median of trial-level ``data``.

    ``data`` must already be one scalar per independent unit (e.g. one RMSE value per
    trajectory trial), not raw per-waypoint samples from a single trial.
    """
    if statistic not in _STATISTIC_FNS:
        raise ValueError(f"unsupported statistic '{statistic}', expected one of {sorted(_STATISTIC_FNS)}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    arr = np.asarray(data, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.shape[0]
    stat_fn = _STATISTIC_FNS[statistic]

    if n == 0:
        return BootstrapInterval(float("nan"), float("nan"), float("nan"), n_resamples, statistic, confidence_level, 0)

    point_estimate = float(stat_fn(arr))
    if n == 1:
        return BootstrapInterval(point_estimate, point_estimate, point_estimate, n_resamples, statistic, confidence_level, 1)

    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n, size=(n_resamples, n))
    resample_stats = stat_fn(arr[resample_indices], axis=1)

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(resample_stats, alpha / 2.0))
    upper = float(np.quantile(resample_stats, 1.0 - alpha / 2.0))
    return BootstrapInterval(point_estimate, lower, upper, n_resamples, statistic, confidence_level, n)
