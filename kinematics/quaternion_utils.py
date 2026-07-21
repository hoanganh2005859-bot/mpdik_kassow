"""Quaternion utilities (wxyz convention): normalization, canonicalization, and conversions.

All quaternions produced by this module use scalar-first ordering: q = [w, x, y, z].
"""

import numpy as np

from utils.exceptions import NumericalKinematicsError

_MIN_QUATERNION_NORM = 1e-9


def normalize_quaternion_wxyz(q: np.ndarray) -> np.ndarray:
    """Normalize a wxyz quaternion to unit norm."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise NumericalKinematicsError(f"expected quaternion shape (4,), got {q.shape}")
    norm = np.linalg.norm(q)
    if norm < _MIN_QUATERNION_NORM:
        raise NumericalKinematicsError("quaternion norm too close to zero to normalize")
    return q / norm


def canonicalize_quaternion_wxyz(q: np.ndarray) -> np.ndarray:
    """Normalize and pick a canonical sign so that q and -q map to the same representative.

    The sign is chosen so that the first component with |component| above tolerance is
    positive (q and -q represent the same rotation, so this removes that ambiguity).
    """
    q = normalize_quaternion_wxyz(q)
    for component in q:
        if component > 1e-12:
            return q
        if component < -1e-12:
            return -q
    return q


def quaternion_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = normalize_quaternion_wxyz(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a wxyz quaternion (Shepperd's method).

    Numerically stable across all rotation angles, including near theta=pi where the
    naive trace-based formula loses precision.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise NumericalKinematicsError(f"expected 3x3 matrix, got shape {R.shape}")

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return normalize_quaternion_wxyz(np.array([w, x, y, z], dtype=np.float64))


def quaternion_geodesic_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle in radians between two wxyz quaternions, in [0, pi].

    Uses |dot(q1, q2)| so that q and -q (the same rotation) yield an identical angle.
    """
    q1n = normalize_quaternion_wxyz(q1)
    q2n = normalize_quaternion_wxyz(q2)
    dot = np.clip(abs(float(np.dot(q1n, q2n))), -1.0, 1.0)
    return float(2.0 * np.arccos(dot))
