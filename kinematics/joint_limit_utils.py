"""Joint limit clamping and limit-avoidance gradient utilities used by the DLS solver.

Operational limits are the ``operational_lower_rad`` / ``operational_upper_rad`` bounds
from configs/robot_config.json (sourced from assets/model_metadata.json). For
joint_1, joint_3, joint_5, joint_6, joint_7 those bounds encode a +/-2*pi continuous-joint
convention, not a verified mechanical hard stop; this module treats all bounds uniformly
as operational sampling/solver limits and makes no hard-stop claim.

Validation functions in this module never clip silently. Only ``clip_to_operational_limits``
performs clipping, and only when a caller (e.g. the DLS solver, per its config flag)
explicitly invokes it.
"""

import numpy as np

from utils.exceptions import InvalidJointVectorError


def _as_limit_arrays(q: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    q = np.asarray(q, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if q.shape != lower.shape or q.shape != upper.shape:
        raise InvalidJointVectorError(
            f"shape mismatch: q={q.shape}, lower={lower.shape}, upper={upper.shape}"
        )
    return q, lower, upper


def validate_joint_vector(q: np.ndarray, lower: np.ndarray = None, upper: np.ndarray = None) -> np.ndarray:
    """Validate a joint vector's shape and finiteness; optionally check operational limits.

    Never clips. If ``lower``/``upper`` are given and ``q`` violates them, raises
    InvalidJointVectorError rather than silently correcting the value.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1:
        raise InvalidJointVectorError(f"expected a 1D joint vector, got shape {q.shape}")
    if not np.all(np.isfinite(q)):
        raise InvalidJointVectorError("joint vector contains non-finite values (NaN or Inf)")

    if lower is not None and upper is not None:
        q, lower, upper = _as_limit_arrays(q, lower, upper)
        if np.any(q < lower) or np.any(q > upper):
            raise InvalidJointVectorError("joint vector violates operational limits")

    return q


def clip_to_operational_limits(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Explicitly clip ``q`` elementwise into [lower, upper]. Never called implicitly."""
    q, lower, upper = _as_limit_arrays(q, lower, upper)
    return np.clip(q, lower, upper)


def operational_limit_violation_mask(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Boolean mask, True where ``q`` is outside [lower, upper]."""
    q, lower, upper = _as_limit_arrays(q, lower, upper)
    return (q < lower) | (q > upper)


def normalized_joint_limit_margin(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Per-joint margin normalized by half the joint range.

    margin_i = distance from q_i to the nearer bound, divided by half_range_i:
    > 0 strictly inside the limits, == 0 exactly at a bound, < 0 when violated.
    A value of 1.0 corresponds to being exactly at the joint's center.
    """
    q, lower, upper = _as_limit_arrays(q, lower, upper)
    half_range = (upper - lower) / 2.0
    if np.any(half_range <= 0.0):
        raise InvalidJointVectorError("joint range (upper - lower) must be strictly positive")
    distance_to_nearer_bound = np.minimum(q - lower, upper - q)
    return distance_to_nearer_bound / half_range


def minimum_joint_limit_margin(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Worst-case (smallest) normalized joint limit margin across all joints."""
    return float(np.min(normalized_joint_limit_margin(q, lower, upper)))


def joint_center(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Midpoint of each joint's operational range."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return (lower + upper) / 2.0


def joint_centering_gradient(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Direction (per joint) that reduces a quadratic center-seeking cost.

    For cost H(q) = 0.5 * sum(((q_i - center_i) / half_range_i)^2), this returns
    -dH/dq_i = -(q_i - center_i) / half_range_i^2, i.e. points toward the joint center,
    scaled down for joints with a wider operational range. Intended as the secondary
    task vector ``z`` for null-space joint-centering in the DLS solver.
    """
    q, lower, upper = _as_limit_arrays(q, lower, upper)
    center = joint_center(lower, upper)
    half_range = (upper - lower) / 2.0
    if np.any(half_range <= 0.0):
        raise InvalidJointVectorError("joint range (upper - lower) must be strictly positive")
    return -(q - center) / (half_range ** 2)
