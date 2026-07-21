"""Singularity proximity metrics (e.g. smallest Jacobian singular value) for the KR810 arm."""

import numpy as np

from utils.exceptions import NumericalKinematicsError

_DEFAULT_RANK_TOL = 1e-6


def singular_values(J: np.ndarray) -> np.ndarray:
    """Singular values of J, descending order, shape (min(J.shape),)."""
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2:
        raise NumericalKinematicsError(f"expected a 2D Jacobian, got shape {J.shape}")
    if not np.all(np.isfinite(J)):
        raise NumericalKinematicsError("Jacobian contains non-finite values")
    return np.linalg.svd(J, compute_uv=False)


def minimum_singular_value(J: np.ndarray) -> float:
    """Smallest singular value of J (proximity to rank deficiency / singularity)."""
    return float(singular_values(J)[-1])


def maximum_singular_value(J: np.ndarray) -> float:
    """Largest singular value of J."""
    return float(singular_values(J)[0])


def numerical_rank(J: np.ndarray, tol: float = _DEFAULT_RANK_TOL) -> int:
    """Numerical rank of J: count of singular values above ``tol`` times the largest one."""
    sv = singular_values(J)
    if sv[0] <= 0.0:
        return 0
    threshold = tol * sv[0]
    return int(np.sum(sv > threshold))


def condition_number(J: np.ndarray, safe_floor: float = 1e-12) -> float:
    """Condition number sigma_max / sigma_min.

    Returns np.inf (never NaN) when sigma_min is at or below ``safe_floor`` instead
    of dividing by (near) zero.
    """
    sv = singular_values(J)
    sigma_max = float(sv[0])
    sigma_min = float(sv[-1])
    if sigma_min <= safe_floor:
        return float("inf")
    return sigma_max / sigma_min


def is_near_singular(J: np.ndarray, threshold: float) -> bool:
    """True if the smallest singular value of J is at or below ``threshold``."""
    if threshold < 0.0:
        raise NumericalKinematicsError("threshold must be non-negative")
    return minimum_singular_value(J) <= threshold
