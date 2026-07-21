"""Adaptive damping factor schedule for the DLS solver based on manipulability/singularity proximity.

Formula (smooth quadratic interpolation, continuous at the threshold):

    lambda(sigma_min) = lambda_min                                                     if sigma_min >= threshold
    lambda(sigma_min) = lambda_max - (lambda_max - lambda_min) * (sigma_min / threshold)^2   otherwise

At sigma_min == threshold both branches evaluate to lambda_min, so the schedule is
continuous there. As sigma_min -> 0 (approaching a singularity), lambda -> lambda_max.
As sigma_min grows beyond threshold, lambda stays pinned at lambda_min. The function is
monotonically non-increasing in sigma_min, so damping never decreases as the arm gets
closer to a singularity.
"""

import numpy as np

from utils.exceptions import DLSSolverError


def compute_adaptive_damping(
    sigma_min: float,
    threshold: float,
    lambda_min: float,
    lambda_max: float,
) -> float:
    """Compute the adaptive damping factor lambda from the smallest Jacobian singular value.

    Args:
        sigma_min: Smallest singular value of the current Jacobian (>= 0).
        threshold: Singularity proximity threshold (singularity_sigma_threshold), > 0.
        lambda_min: Damping floor used away from singularities.
        lambda_max: Damping ceiling used at/near a singularity (sigma_min -> 0).

    Returns:
        A deterministic damping factor in [lambda_min, lambda_max].
    """
    if threshold <= 0.0:
        raise DLSSolverError("adaptive damping threshold must be positive")
    if sigma_min < 0.0:
        raise DLSSolverError("sigma_min must be non-negative")
    if lambda_max < lambda_min:
        raise DLSSolverError("lambda_max must be >= lambda_min")

    if sigma_min >= threshold:
        return float(lambda_min)

    ratio = sigma_min / threshold
    lam = lambda_max - (lambda_max - lambda_min) * (ratio ** 2)
    return float(np.clip(lam, lambda_min, lambda_max))
