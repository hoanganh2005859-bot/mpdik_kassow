"""Tier 4: checks joint trajectories against operational position/velocity limits.

Position limits come from configs/robot_config.json (``operational_lower_rad``/
``operational_upper_rad``); velocity limits from ``velocity_limits_rad_s``. Acceleration limits
are never fabricated: configs/robot_config.json has no acceleration limits, so acceleration
utilization is only computed when the caller explicitly supplies ``acceleration_limits``: with no
limits given, the corresponding fields are reported with ``status="unavailable"`` rather than a
guessed pass/fail.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from kinematics.joint_limit_utils import (
    normalized_joint_limit_margin,
    operational_limit_violation_mask,
)

DEFAULT_VELOCITY_TOLERANCE = 1e-6


@dataclass
class JointFeasibilityMetrics:
    """Tier 4 joint-limit / velocity-limit / acceleration-limit feasibility summary."""

    sample_count: int
    joint_count: int

    operational_limit_violation_count: int
    operational_limit_violation_rate: float
    minimum_normalized_joint_limit_margin: float
    per_joint_minimum_margin: np.ndarray

    maximum_velocity_utilization: float
    per_joint_velocity_utilization: np.ndarray
    velocity_violation_count: int

    acceleration_status: str  # "available" or "unavailable"
    maximum_acceleration_utilization: Optional[float]
    per_joint_acceleration_utilization: Optional[np.ndarray]
    acceleration_violation_count: Optional[int]


def compute_joint_feasibility_metrics(
    q_trajectory: np.ndarray,
    joint_velocity: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    velocity_limits: np.ndarray,
    joint_acceleration: Optional[np.ndarray] = None,
    acceleration_limits: Optional[np.ndarray] = None,
    velocity_tolerance: float = DEFAULT_VELOCITY_TOLERANCE,
) -> JointFeasibilityMetrics:
    """Compute Tier 4 joint-limit/velocity/acceleration feasibility metrics.

    Args:
        q_trajectory: (N, J) joint positions, radians.
        joint_velocity: (N, J) joint velocities, radians/second (e.g. from
            evaluation.smoothness_metrics.compute_smoothness_metrics).
        lower/upper: (J,) operational joint limits, radians.
        velocity_limits: (J,) per-joint velocity limits, radians/second.
        joint_acceleration: Optional (N, J) joint accelerations, radians/second^2.
        acceleration_limits: Optional (J,) per-joint acceleration limits. If either
            ``joint_acceleration`` or ``acceleration_limits`` is None, acceleration utilization
            is reported as unavailable rather than fabricated.
    """
    q = np.asarray(q_trajectory, dtype=np.float64)
    velocity = np.asarray(joint_velocity, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    velocity_limits = np.asarray(velocity_limits, dtype=np.float64)

    if q.ndim != 2:
        raise ValueError(f"expected q_trajectory of shape (N, J), got {q.shape}")
    if velocity.shape != q.shape:
        raise ValueError(f"joint_velocity shape {velocity.shape} does not match q_trajectory shape {q.shape}")

    n, joint_count = q.shape

    violation_mask = np.array([np.any(operational_limit_violation_mask(q[k], lower, upper)) for k in range(n)])
    margins = np.array([normalized_joint_limit_margin(q[k], lower, upper) for k in range(n)])  # (N, J)

    velocity_utilization = np.abs(velocity) / velocity_limits[None, :]  # (N, J)
    velocity_violations = velocity_utilization > (1.0 + velocity_tolerance)

    if joint_acceleration is not None and acceleration_limits is not None:
        acceleration = np.asarray(joint_acceleration, dtype=np.float64)
        acceleration_limits = np.asarray(acceleration_limits, dtype=np.float64)
        if acceleration.shape != q.shape:
            raise ValueError(f"joint_acceleration shape {acceleration.shape} does not match q_trajectory shape {q.shape}")
        acceleration_utilization = np.abs(acceleration) / acceleration_limits[None, :]
        acceleration_violations = acceleration_utilization > (1.0 + velocity_tolerance)
        acceleration_status = "available"
        max_acceleration_utilization = float(np.max(acceleration_utilization))
        per_joint_acceleration_utilization = np.max(acceleration_utilization, axis=0)
        acceleration_violation_count = int(np.sum(acceleration_violations))
    else:
        acceleration_status = "unavailable"
        max_acceleration_utilization = None
        per_joint_acceleration_utilization = None
        acceleration_violation_count = None

    return JointFeasibilityMetrics(
        sample_count=n,
        joint_count=joint_count,
        operational_limit_violation_count=int(np.sum(violation_mask)),
        operational_limit_violation_rate=float(np.mean(violation_mask)),
        minimum_normalized_joint_limit_margin=float(np.min(margins)),
        per_joint_minimum_margin=np.min(margins, axis=0),
        maximum_velocity_utilization=float(np.max(velocity_utilization)),
        per_joint_velocity_utilization=np.max(velocity_utilization, axis=0),
        velocity_violation_count=int(np.sum(velocity_violations)),
        acceleration_status=acceleration_status,
        maximum_acceleration_utilization=max_acceleration_utilization,
        per_joint_acceleration_utilization=per_joint_acceleration_utilization,
        acceleration_violation_count=acceleration_violation_count,
    )
