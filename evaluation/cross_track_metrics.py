"""Tier 3: cross-track (perpendicular deviation from the intended path) error metrics.

Never uses "nearest waypoint" as a proxy for cross-track distance. Every actual point is
projected onto the *nearest segment* of the reference polyline (with the projection parameter
clamped to [0, 1] so the projection never falls outside the segment), and the true perpendicular
(or endpoint, if the nearest point on the segment is an endpoint) Euclidean distance to that
projection is reported.

For closed reference paths (``closed_path=True``, matching trajectories/trajectory_manifest.csv
for circle/figure8 trajectories), the final segment connecting the last reference point back to
the first is included, so a point near the seam is not incorrectly penalized.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

_BACKWARD_PROGRESS_TOL_M = 1e-9


@dataclass
class CrossTrackPointResult:
    """Cross-track projection result for a single actual point against the reference polyline."""

    point_index: int
    cross_track_distance_m: float
    nearest_segment_index: int
    segment_parameter: float
    along_path_coordinate_m: float


@dataclass
class CrossTrackSummary:
    """Distributional summary of cross-track and along-track behavior over a trajectory run."""

    point_count: int
    cross_track_rmse_m: float
    cross_track_mae_m: float
    cross_track_p95_m: float
    cross_track_max_m: float
    total_path_length_m: float
    final_progress_ratio: float
    backward_progress_count: int
    synchronized_along_track_rmse_m: Optional[float]


@dataclass
class CrossTrackResult:
    per_point: List[CrossTrackPointResult]
    summary: CrossTrackSummary


def _segment_geometry(reference_path: np.ndarray, closed_path: bool):
    if closed_path:
        segment_starts = reference_path
        segment_ends = np.roll(reference_path, -1, axis=0)
    else:
        segment_starts = reference_path[:-1]
        segment_ends = reference_path[1:]

    segment_vectors = segment_ends - segment_starts
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative_start = np.concatenate([[0.0], np.cumsum(segment_lengths)[:-1]])
    return segment_starts, segment_vectors, segment_lengths, cumulative_start


def project_point_to_polyline(
    point: np.ndarray,
    segment_starts: np.ndarray,
    segment_vectors: np.ndarray,
    segment_lengths: np.ndarray,
    cumulative_start: np.ndarray,
) -> Tuple[int, float, float]:
    """Project ``point`` onto the nearest segment of a polyline (vectorized over all segments).

    Returns (nearest_segment_index, clamped_segment_parameter, cross_track_distance_m).
    """
    diffs = point[None, :] - segment_starts  # (S, 3)
    seg_len_sq = np.maximum(segment_lengths**2, 1e-18)
    t = np.sum(diffs * segment_vectors, axis=1) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    projections = segment_starts + t_clamped[:, None] * segment_vectors
    distances = np.linalg.norm(point[None, :] - projections, axis=1)
    best = int(np.argmin(distances))
    return best, float(t_clamped[best]), float(distances[best])


def _cumulative_arc_length(path: np.ndarray) -> np.ndarray:
    """Cumulative arc length up to (and including) each point of an ordered point sequence."""
    if path.shape[0] < 2:
        return np.zeros(path.shape[0], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def compute_cross_track_metrics(
    actual_positions: np.ndarray,
    reference_path: np.ndarray,
    closed_path: bool = False,
) -> CrossTrackResult:
    """Compute cross-track and along-track metrics of ``actual_positions`` against ``reference_path``.

    Args:
        actual_positions: (N, 3) achieved end-effector positions, in time order.
        reference_path: (M, 3) reference/commanded polyline. If M == N, ``reference_path`` is
            also treated as the time-synchronized commanded sequence and a synchronized
            along-track error is additionally computed; otherwise that field is None.
        closed_path: Whether the reference polyline wraps from its last point back to its first
            (matching trajectory_manifest.csv's closed_path for circle/figure8 trajectories).
    """
    actual_positions = np.asarray(actual_positions, dtype=np.float64)
    reference_path = np.asarray(reference_path, dtype=np.float64)
    if actual_positions.ndim != 2 or actual_positions.shape[1] != 3:
        raise ValueError(f"expected actual_positions of shape (N, 3), got {actual_positions.shape}")
    if reference_path.ndim != 2 or reference_path.shape[1] != 3:
        raise ValueError(f"expected reference_path of shape (M, 3), got {reference_path.shape}")
    if reference_path.shape[0] < 2:
        raise ValueError("reference_path needs at least 2 points to define a segment")

    segment_starts, segment_vectors, segment_lengths, cumulative_start = _segment_geometry(
        reference_path, closed_path
    )
    total_path_length = float(np.sum(segment_lengths))

    per_point: List[CrossTrackPointResult] = []
    cross_track = np.empty(actual_positions.shape[0], dtype=np.float64)
    along = np.empty(actual_positions.shape[0], dtype=np.float64)

    for i, point in enumerate(actual_positions):
        seg_idx, t, dist = project_point_to_polyline(
            point, segment_starts, segment_vectors, segment_lengths, cumulative_start
        )
        along_coord = cumulative_start[seg_idx] + t * segment_lengths[seg_idx]
        cross_track[i] = dist
        along[i] = along_coord
        per_point.append(
            CrossTrackPointResult(
                point_index=i,
                cross_track_distance_m=dist,
                nearest_segment_index=seg_idx,
                segment_parameter=t,
                along_path_coordinate_m=along_coord,
            )
        )

    backward_progress_count = int(np.sum(np.diff(along) < -_BACKWARD_PROGRESS_TOL_M))
    final_progress_ratio = float(along[-1] / total_path_length) if total_path_length > 0.0 else float("nan")

    synchronized_rmse = None
    if reference_path.shape[0] == actual_positions.shape[0]:
        commanded_along = _cumulative_arc_length(reference_path)
        synchronized_error = along - commanded_along
        synchronized_rmse = float(np.sqrt(np.mean(synchronized_error**2)))

    summary = CrossTrackSummary(
        point_count=int(actual_positions.shape[0]),
        cross_track_rmse_m=float(np.sqrt(np.mean(cross_track**2))),
        cross_track_mae_m=float(np.mean(cross_track)),
        cross_track_p95_m=float(np.percentile(cross_track, 95)),
        cross_track_max_m=float(np.max(cross_track)),
        total_path_length_m=total_path_length,
        final_progress_ratio=final_progress_ratio,
        backward_progress_count=backward_progress_count,
        synchronized_along_track_rmse_m=synchronized_rmse,
    )
    return CrossTrackResult(per_point=per_point, summary=summary)
