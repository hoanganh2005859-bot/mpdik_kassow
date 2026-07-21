"""Orientation tracking error metrics computed via SO(3) logarithm (geodesic angle).

Always uses the geodesic angle between rotation matrices (equivalently, the SO(3) logarithm
convention used throughout kinematics/pose_error.py). Never subtracts Euler angles and never
takes the raw norm of a quaternion difference (both are invalid distance measures on SO(3)
without further correction). Never adds orientation degrees to position meters.
"""

from dataclasses import dataclass

import numpy as np

from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from kinematics.rotation_utils import rotation_geodesic_angle


def geodesic_angle_rad(R_target: np.ndarray, R_actual: np.ndarray) -> float:
    """Geodesic (shortest-arc) angle in radians between two rotation matrices."""
    return rotation_geodesic_angle(R_target, R_actual)


def geodesic_angles_from_quaternions(target_quaternions: np.ndarray, actual_quaternions: np.ndarray) -> np.ndarray:
    """Per-sample geodesic angle (radians) between synchronized wxyz quaternion sequences.

    ``target_quaternions``/``actual_quaternions`` must have identical shape (N, 4); q and -q are
    treated as the same rotation (see kinematics.rotation_utils.rotation_geodesic_angle, which
    operates on rotation matrices and is therefore already sign-invariant).
    """
    target_quaternions = np.asarray(target_quaternions, dtype=np.float64)
    actual_quaternions = np.asarray(actual_quaternions, dtype=np.float64)
    if target_quaternions.shape != actual_quaternions.shape:
        raise ValueError(
            f"target/actual quaternion shape mismatch: {target_quaternions.shape} vs {actual_quaternions.shape}"
        )
    if target_quaternions.ndim != 2 or target_quaternions.shape[1] != 4:
        raise ValueError(f"expected quaternion arrays of shape (N, 4), got {target_quaternions.shape}")

    n = target_quaternions.shape[0]
    angles = np.empty(n, dtype=np.float64)
    for i in range(n):
        R_target = quaternion_wxyz_to_matrix(target_quaternions[i])
        R_actual = quaternion_wxyz_to_matrix(actual_quaternions[i])
        angles[i] = rotation_geodesic_angle(R_target, R_actual)
    return angles


@dataclass
class OrientationErrorSummary:
    """Distributional summary of a set of geodesic orientation errors, reported in degrees."""

    count: int
    rmse_deg: float
    mae_deg: float
    median_deg: float
    p95_deg: float
    max_deg: float


def summarize_orientation_errors(angles_rad: np.ndarray) -> OrientationErrorSummary:
    """Summarize a 1D array of geodesic angle errors (radians) as an OrientationErrorSummary (degrees)."""
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    if angles_rad.ndim != 1:
        raise ValueError(f"expected a 1D array of geodesic angles, got shape {angles_rad.shape}")
    if angles_rad.shape[0] == 0:
        raise ValueError("no orientation error samples provided")
    if not np.all(np.isfinite(angles_rad)):
        raise ValueError("angles_rad contains non-finite values")

    deg = np.degrees(angles_rad)
    return OrientationErrorSummary(
        count=int(deg.shape[0]),
        rmse_deg=float(np.sqrt(np.mean(deg**2))),
        mae_deg=float(np.mean(np.abs(deg))),
        median_deg=float(np.median(deg)),
        p95_deg=float(np.percentile(deg, 95)),
        max_deg=float(np.max(deg)),
    )
