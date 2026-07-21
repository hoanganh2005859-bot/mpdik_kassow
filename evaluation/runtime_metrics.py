"""Records solver runtime statistics per Tier 0-4 run.

``solve_time_ms`` must already be sourced from ``kinematics.dls_solver.DLSResult.solve_time_ms``
(measured with ``time.perf_counter()`` around the solve itself) -- never a duration that
includes file I/O, NPZ loading, or other bookkeeping.

Deadline comparisons always use whatever real-time period the caller resolved and stored on the
run (e.g. ``control_period_s / speed_scale`` from algorithms.sequential_dls), passed explicitly
as ``deadline_ms``; this module never re-derives a deadline on its own.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RuntimeMetrics:
    """Distributional summary of a set of solve_time_ms samples, plus optional deadline stats."""

    count: int
    total_runtime_ms: float
    mean_ms: float
    median_ms: float
    std_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    deadline_miss_count: Optional[int]
    deadline_miss_rate: Optional[float]


def compute_runtime_metrics(solve_times_ms, deadline_ms: Optional[float] = None) -> RuntimeMetrics:
    """Compute runtime distribution statistics for a 1D array of solve_time_ms samples."""
    arr = np.asarray(solve_times_ms, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] == 0:
        raise ValueError("need at least one solve_time_ms sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError("solve_times_ms contains non-finite values")

    if deadline_ms is not None:
        deadline_miss_count = int(np.sum(arr > deadline_ms))
        deadline_miss_rate = deadline_miss_count / arr.shape[0]
    else:
        deadline_miss_count = None
        deadline_miss_rate = None

    return RuntimeMetrics(
        count=int(arr.shape[0]),
        total_runtime_ms=float(np.sum(arr)),
        mean_ms=float(np.mean(arr)),
        median_ms=float(np.median(arr)),
        std_ms=float(np.std(arr, ddof=1)) if arr.shape[0] > 1 else 0.0,
        p90_ms=float(np.percentile(arr, 90)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        max_ms=float(np.max(arr)),
        deadline_miss_count=deadline_miss_count,
        deadline_miss_rate=deadline_miss_rate,
    )
