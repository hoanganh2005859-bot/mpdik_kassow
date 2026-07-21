"""Rotation utilities: conversions between rotation matrices, quaternions, and SO(3) log/exp maps.

All logarithm/exponential maps are implemented directly (no scipy dependency) so that
Stage 3 orientation error and Jacobian code has one internal, testable source of truth.
scipy.spatial.transform.Rotation may be used in tests for cross-checking only.
"""

import numpy as np

from utils.exceptions import NumericalKinematicsError

_NEAR_ZERO_ANGLE_RAD = 1e-8
_NEAR_PI_MARGIN_RAD = 1e-6


def skew(v: np.ndarray) -> np.ndarray:
    """Map a 3-vector to its 3x3 skew-symmetric (cross-product) matrix."""
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (3,):
        raise NumericalKinematicsError(f"expected 3-vector, got shape {v.shape}")
    x, y, z = v
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def vee(matrix: np.ndarray) -> np.ndarray:
    """Inverse of ``skew``: extract the 3-vector from a (nearly) skew-symmetric matrix.

    Uses the antisymmetric part of ``matrix``, so small symmetric numerical noise is
    ignored rather than propagated.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise NumericalKinematicsError(f"expected 3x3 matrix, got shape {matrix.shape}")
    antisym = (matrix - matrix.T) / 2.0
    return np.array([antisym[2, 1], antisym[0, 2], antisym[1, 0]], dtype=np.float64)


def validate_rotation_matrix(R: np.ndarray, tol: float = 1e-6) -> bool:
    """Check whether ``R`` is a valid SO(3) rotation matrix within tolerance.

    Verifies shape, finiteness, orthogonality (R.T @ R ~= I), and det(R) ~= 1.
    Returns a bool; does not raise and does not modify ``R``.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return False
    if not np.all(np.isfinite(R)):
        return False
    orth_err = np.linalg.norm(R.T @ R - np.eye(3))
    det = np.linalg.det(R)
    return bool(orth_err < tol and abs(det - 1.0) < tol)


def project_to_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Project a near-orthogonal matrix onto the closest proper rotation matrix (SVD method).

    Intended for numerical cleanup only (e.g. after repeated composition), not for
    correcting grossly invalid input.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise NumericalKinematicsError(f"expected 3x3 matrix, got shape {R.shape}")
    u, _, vt = np.linalg.svd(R)
    d = np.sign(np.linalg.det(u @ vt))
    if d == 0.0:
        d = 1.0
    correction = np.diag([1.0, 1.0, d])
    return u @ correction @ vt


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential map: rotation vector (axis * angle, radians) -> rotation matrix.

    Uses the closed-form Rodrigues formula with a small-angle Taylor fallback so the
    result stays finite and accurate as ||phi|| -> 0.
    """
    phi = np.asarray(phi, dtype=np.float64)
    if phi.shape != (3,):
        raise NumericalKinematicsError(f"expected 3-vector, got shape {phi.shape}")
    theta = np.linalg.norm(phi)
    k = skew(phi)
    if theta < _NEAR_ZERO_ANGLE_RAD:
        a = 1.0 - (theta ** 2) / 6.0
        b = 0.5 - (theta ** 2) / 24.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta ** 2)
    return np.eye(3) + a * k + b * (k @ k)


def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map: rotation matrix -> rotation vector (axis * angle, radians).

    Stable near theta=0 (first-order antisymmetric-part approximation) and near
    theta=pi (symmetric-part axis extraction, avoiding division by sin(theta) ~= 0).
    The arccos argument is clamped to [-1, 1] to absorb floating-point drift.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise NumericalKinematicsError(f"expected 3x3 matrix, got shape {R.shape}")

    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < _NEAR_ZERO_ANGLE_RAD:
        return vee((R - R.T) / 2.0)

    if theta > (np.pi - _NEAR_PI_MARGIN_RAD):
        symmetric_part = (R + np.eye(3)) / 2.0
        axis_sq = np.clip(np.diag(symmetric_part), 0.0, None)
        axis = np.sqrt(axis_sq)
        pivot = int(np.argmax(axis))
        if axis[pivot] < 1e-12:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            for j in range(3):
                if j != pivot and symmetric_part[pivot, j] < 0.0:
                    axis[j] = -axis[j]
            axis = axis / np.linalg.norm(axis)
        return axis * theta

    return vee((R - R.T) / (2.0 * np.sin(theta))) * theta


def rotation_geodesic_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic (shortest-arc) angle in radians between two rotation matrices, in [0, pi]."""
    R1 = np.asarray(R1, dtype=np.float64)
    R2 = np.asarray(R2, dtype=np.float64)
    if R1.shape != (3, 3) or R2.shape != (3, 3):
        raise NumericalKinematicsError("expected two 3x3 rotation matrices")
    relative = R1.T @ R2
    cos_theta = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_theta))
