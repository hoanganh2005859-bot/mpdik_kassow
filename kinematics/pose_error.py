"""Computes position and SO(3) logarithm orientation error between a target pose and a current pose.

Orientation error convention (world/space frame, matching the world-frame geometric
Jacobian used elsewhere in this package):

    orientation_error_world = Log(R_target @ R_current.T)^vee

This is the *world*-frame (left) error. The body-frame (right) error
``Log(R_current.T @ R_target)^vee`` is numerically different (though it has the same
norm) and must not be mixed with a world-frame Jacobian without an explicit R @ (.)
frame conversion. This module always uses the world-frame convention above.

Full pose error ordering:
    [position_x, position_y, position_z, rotation_x, rotation_y, rotation_z]
"""

import numpy as np

from kinematics.rotation_utils import rotation_geodesic_angle, so3_log
from utils.exceptions import NumericalKinematicsError


def position_error_vector(target_position: np.ndarray, current_position: np.ndarray) -> np.ndarray:
    """Position error e_p = target - current, shape (3,), meters."""
    target_position = np.asarray(target_position, dtype=np.float64)
    current_position = np.asarray(current_position, dtype=np.float64)
    if target_position.shape != (3,) or current_position.shape != (3,):
        raise NumericalKinematicsError("expected position vectors of shape (3,)")
    return target_position - current_position


def position_error_norm(target_position: np.ndarray, current_position: np.ndarray) -> float:
    """Euclidean norm of the position error, meters."""
    return float(np.linalg.norm(position_error_vector(target_position, current_position)))


def orientation_error_vector_world(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """World-frame orientation error vector: Log(R_target @ R_current.T)^vee, shape (3,), radians."""
    R_target = np.asarray(R_target, dtype=np.float64)
    R_current = np.asarray(R_current, dtype=np.float64)
    if R_target.shape != (3, 3) or R_current.shape != (3, 3):
        raise NumericalKinematicsError("expected rotation matrices of shape (3, 3)")
    return so3_log(R_target @ R_current.T)


def orientation_geodesic_angle(R_target: np.ndarray, R_current: np.ndarray) -> float:
    """Geodesic angle (radians, in [0, pi]) between target and current orientation."""
    return rotation_geodesic_angle(R_target, R_current)


def full_pose_error(
    target_position: np.ndarray,
    R_target: np.ndarray,
    current_position: np.ndarray,
    R_current: np.ndarray,
) -> np.ndarray:
    """Full 6-vector pose error: [position error (3,), world-frame orientation error (3,)]."""
    e_p = position_error_vector(target_position, current_position)
    e_o = orientation_error_vector_world(R_target, R_current)
    return np.concatenate([e_p, e_o])


def weighted_pose_error(
    pose_error_vector: np.ndarray,
    position_weight: float,
    orientation_weight: float,
) -> np.ndarray:
    """Apply explicit per-block weights to a 6-vector pose error (no implicit unit mixing).

    Returns weights * pose_error_vector elementwise, where the first 3 entries use
    ``position_weight`` and the last 3 use ``orientation_weight``.
    """
    pose_error_vector = np.asarray(pose_error_vector, dtype=np.float64)
    if pose_error_vector.shape != (6,):
        raise NumericalKinematicsError("expected full pose error of shape (6,)")
    weights = np.array([position_weight] * 3 + [orientation_weight] * 3, dtype=np.float64)
    return weights * pose_error_vector
