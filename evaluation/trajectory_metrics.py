"""Tier 3: aggregate trajectory tracking metrics (RMSE, P95, max position/orientation error).

Target and actual arrays must already be synchronized (same length, same waypoint order) --
this module never resamples or interpolates to reconcile mismatched lengths; a length mismatch
is always a caller error and raises ValueError. Accepts plain numpy arrays; a pandas DataFrame
with numeric x/y/z (or qw/qx/qy/qz) columns in that order also works transparently since
``np.asarray(df)`` on a numeric DataFrame already yields the equivalent ndarray.
"""

from dataclasses import dataclass

import numpy as np

from evaluation.orientation_metrics import (
    OrientationErrorSummary,
    geodesic_angles_from_quaternions,
    summarize_orientation_errors,
)


@dataclass
class PositionTrackingMetrics:
    """Cartesian position tracking error summary between a synchronized target/actual pair."""

    count: int
    rmse_m: float
    mae_m: float
    median_m: float
    p95_m: float
    max_m: float
    endpoint_error_m: float
    start_point_error_m: float
    rmse_x_m: float
    rmse_y_m: float
    rmse_z_m: float
    centroid_offset_m: float
    target_path_length_m: float
    actual_path_length_m: float
    path_length_ratio: float
    path_length_abs_diff_m: float


def _as_position_array(positions) -> np.ndarray:
    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected an (N, 3) position array, got shape {arr.shape}")
    return arr


def _path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def compute_position_tracking_metrics(target_positions, actual_positions) -> PositionTrackingMetrics:
    """Compute Tier 3 Cartesian position tracking metrics for one synchronized waypoint sequence."""
    target = _as_position_array(target_positions)
    actual = _as_position_array(actual_positions)
    if target.shape != actual.shape:
        raise ValueError(
            f"target/actual position shape mismatch: {target.shape} vs {actual.shape}; "
            "align the sequences before calling (no silent resampling is performed here)"
        )
    if target.shape[0] < 1:
        raise ValueError("need at least one synchronized waypoint")

    diff = target - actual
    errors = np.linalg.norm(diff, axis=1)

    target_path_length = _path_length(target)
    actual_path_length = _path_length(actual)
    path_length_ratio = float(actual_path_length / target_path_length) if target_path_length > 0.0 else float("nan")

    return PositionTrackingMetrics(
        count=int(target.shape[0]),
        rmse_m=float(np.sqrt(np.mean(errors**2))),
        mae_m=float(np.mean(errors)),
        median_m=float(np.median(errors)),
        p95_m=float(np.percentile(errors, 95)),
        max_m=float(np.max(errors)),
        endpoint_error_m=float(errors[-1]),
        start_point_error_m=float(errors[0]),
        rmse_x_m=float(np.sqrt(np.mean(diff[:, 0] ** 2))),
        rmse_y_m=float(np.sqrt(np.mean(diff[:, 1] ** 2))),
        rmse_z_m=float(np.sqrt(np.mean(diff[:, 2] ** 2))),
        centroid_offset_m=float(np.linalg.norm(np.mean(target, axis=0) - np.mean(actual, axis=0))),
        target_path_length_m=target_path_length,
        actual_path_length_m=actual_path_length,
        path_length_ratio=path_length_ratio,
        path_length_abs_diff_m=abs(actual_path_length - target_path_length),
    )


@dataclass
class TrajectoryTrackingMetrics:
    """Combined Tier 3 position + orientation tracking metrics for one trajectory run."""

    position: PositionTrackingMetrics
    orientation: OrientationErrorSummary


def compute_trajectory_tracking_metrics(
    target_positions, actual_positions, target_quaternions, actual_quaternions
) -> TrajectoryTrackingMetrics:
    """Compute combined Tier 3 position + orientation tracking metrics.

    All four inputs must be synchronized (same waypoint order/length); position and orientation
    lengths may differ from each other only if the caller has a reason to evaluate them
    separately, but each pair (target/actual) must match in length internally.
    """
    position = compute_position_tracking_metrics(target_positions, actual_positions)
    angles_rad = geodesic_angles_from_quaternions(target_quaternions, actual_quaternions)
    orientation = summarize_orientation_errors(angles_rad)
    return TrajectoryTrackingMetrics(position=position, orientation=orientation)
