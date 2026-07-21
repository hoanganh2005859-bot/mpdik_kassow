"""Tier 4: joint trajectory smoothness metrics (velocity, acceleration, jerk statistics).

Never assumes uniform sampling: derivatives are always computed against the caller-supplied
``time_s`` coordinate array via ``numpy.gradient`` (non-uniform-spacing aware), using
``edge_order=2`` whenever there are enough points for it. ``edge_order=2`` matters here because
it is exact (not just close) at the boundary samples for polynomials up to degree 2 -- e.g. a
perfectly linear or quadratic joint trajectory yields an exactly-zero-noise constant
acceleration/near-zero jerk, matching what a synthetic (noise-free) test trajectory should show,
rather than an artificial boundary bias from a coarser one-sided estimate.

Reports "unavailable" (rather than fabricating a number) when there are too few samples to
support a derivative order: velocity needs >= 2 points, acceleration >= 3, jerk >= 4.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

MIN_POINTS_VELOCITY = 2
MIN_POINTS_ACCELERATION = 3
MIN_POINTS_JERK = 4


def _validate_time(time_s: np.ndarray) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=np.float64)
    if time_s.ndim != 1:
        raise ValueError(f"expected a 1D time_s array, got shape {time_s.shape}")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("time_s contains non-finite values")
    if time_s.shape[0] < 2:
        raise ValueError("time_s needs at least 2 samples")
    diffs = np.diff(time_s)
    if np.any(diffs <= 0.0):
        raise ValueError("time_s must be strictly increasing (no duplicate or non-monotonic timestamps)")
    return time_s


def _gradient(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    edge_order = 2 if time_s.shape[0] >= 3 else 1
    return np.gradient(values, time_s, axis=0, edge_order=edge_order)


@dataclass
class SmoothnessMetrics:
    """Joint-space smoothness metrics for one q(t) trajectory. Fields are None when the
    trajectory has too few samples to support that derivative order (see module docstring)."""

    joint_count: int
    sample_count: int

    velocity_available: bool
    velocity: Optional[np.ndarray]
    max_abs_velocity_per_joint: Optional[np.ndarray]
    rms_velocity_per_joint: Optional[np.ndarray]

    acceleration_available: bool
    acceleration: Optional[np.ndarray]
    max_abs_acceleration_per_joint: Optional[np.ndarray]
    rms_acceleration_per_joint: Optional[np.ndarray]

    jerk_available: bool
    jerk: Optional[np.ndarray]
    max_abs_jerk_per_joint: Optional[np.ndarray]
    rms_jerk_per_joint: Optional[np.ndarray]
    global_rms_jerk: Optional[float]

    max_joint_jump_rad: float
    max_joint_jump_joint_index: int
    max_joint_jump_timestep_index: int
    max_joint_jump_per_joint: np.ndarray
    total_joint_variation_per_joint: np.ndarray
    second_difference_norm_rad: Optional[float]


def compute_smoothness_metrics(q_trajectory: np.ndarray, time_s: np.ndarray) -> SmoothnessMetrics:
    """Compute Tier 4 joint smoothness metrics for ``q_trajectory`` (N, J) sampled at ``time_s`` (N,)."""
    q = np.asarray(q_trajectory, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"expected q_trajectory of shape (N, J), got shape {q.shape}")
    time_s = _validate_time(time_s)
    if q.shape[0] != time_s.shape[0]:
        raise ValueError(f"q_trajectory has {q.shape[0]} samples but time_s has {time_s.shape[0]}")

    n, joint_count = q.shape

    velocity_available = n >= MIN_POINTS_VELOCITY
    velocity = _gradient(q, time_s) if velocity_available else None
    max_abs_velocity = np.max(np.abs(velocity), axis=0) if velocity_available else None
    rms_velocity = np.sqrt(np.mean(velocity**2, axis=0)) if velocity_available else None

    acceleration_available = n >= MIN_POINTS_ACCELERATION
    acceleration = _gradient(velocity, time_s) if acceleration_available else None
    max_abs_acceleration = np.max(np.abs(acceleration), axis=0) if acceleration_available else None
    rms_acceleration = np.sqrt(np.mean(acceleration**2, axis=0)) if acceleration_available else None

    jerk_available = n >= MIN_POINTS_JERK
    jerk = _gradient(acceleration, time_s) if jerk_available else None
    max_abs_jerk = np.max(np.abs(jerk), axis=0) if jerk_available else None
    rms_jerk = np.sqrt(np.mean(jerk**2, axis=0)) if jerk_available else None
    global_rms_jerk = float(np.sqrt(np.mean(jerk**2))) if jerk_available else None

    consecutive_diff = np.diff(q, axis=0)  # (N-1, J)
    abs_diff = np.abs(consecutive_diff)
    max_joint_jump_per_joint = np.max(abs_diff, axis=0)
    flat_argmax = int(np.argmax(abs_diff))
    max_timestep_index, max_joint_index = np.unravel_index(flat_argmax, abs_diff.shape)
    total_joint_variation = np.sum(abs_diff, axis=0)

    second_difference_norm = None
    if n >= 3:
        second_diff = q[2:] - 2.0 * q[1:-1] + q[:-2]
        second_difference_norm = float(np.sqrt(np.mean(second_diff**2)))

    return SmoothnessMetrics(
        joint_count=joint_count,
        sample_count=n,
        velocity_available=velocity_available,
        velocity=velocity,
        max_abs_velocity_per_joint=max_abs_velocity,
        rms_velocity_per_joint=rms_velocity,
        acceleration_available=acceleration_available,
        acceleration=acceleration,
        max_abs_acceleration_per_joint=max_abs_acceleration,
        rms_acceleration_per_joint=rms_acceleration,
        jerk_available=jerk_available,
        jerk=jerk,
        max_abs_jerk_per_joint=max_abs_jerk,
        rms_jerk_per_joint=rms_jerk,
        global_rms_jerk=global_rms_jerk,
        max_joint_jump_rad=float(abs_diff[max_timestep_index, max_joint_index]),
        max_joint_jump_joint_index=int(max_joint_index),
        max_joint_jump_timestep_index=int(max_timestep_index),
        max_joint_jump_per_joint=max_joint_jump_per_joint,
        total_joint_variation_per_joint=total_joint_variation,
        second_difference_norm_rad=second_difference_norm,
    )
