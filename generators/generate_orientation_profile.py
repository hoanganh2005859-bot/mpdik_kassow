"""Generates fixed/variable end-effector orientation profiles to accompany Cartesian trajectories.

Fixed mode holds every waypoint's orientation equal to a validated anchor rotation. Variable
mode uses SO(3) geodesic interpolation

    R(s) = R0 @ Exp(s * Log(R0.T @ R1))

never independent Euler-angle interpolation. Output quaternions are wxyz and canonicalized
so consecutive samples have a non-negative dot product (no needless sign flips along the path).

This module is a library used by the trajectory generators; it does not write files itself.
"""

import numpy as np

from kinematics.quaternion_utils import canonicalize_quaternion_wxyz, rotation_matrix_to_quaternion_wxyz
from kinematics.rotation_utils import so3_exp, so3_log, validate_rotation_matrix


def _canonicalize_sequence(quaternions: np.ndarray) -> np.ndarray:
    """Canonicalize each quaternion, then flip sign where needed so consecutive dot >= 0."""
    out = np.empty_like(quaternions)
    out[0] = canonicalize_quaternion_wxyz(quaternions[0])
    for i in range(1, quaternions.shape[0]):
        q = canonicalize_quaternion_wxyz(quaternions[i])
        if np.dot(q, out[i - 1]) < 0.0:
            q = -q
        out[i] = q
    return out


def fixed_orientation_profile(R_anchor: np.ndarray, num_waypoints: int) -> np.ndarray:
    """Constant orientation profile: every waypoint uses R_anchor. Returns wxyz quats, shape (N, 4)."""
    if not validate_rotation_matrix(R_anchor):
        raise ValueError("R_anchor is not a valid SO(3) rotation matrix")
    if num_waypoints < 1:
        raise ValueError("num_waypoints must be >= 1")
    q_anchor = rotation_matrix_to_quaternion_wxyz(R_anchor)
    quats = np.tile(q_anchor, (num_waypoints, 1))
    return _canonicalize_sequence(quats)


def variable_orientation_profile(R_start: np.ndarray, R_end: np.ndarray, s: np.ndarray) -> np.ndarray:
    """SO(3) geodesic interpolation from R_start to R_end over path parameter s in [0, 1].

    R(s) = R_start @ Exp(s * Log(R_start.T @ R_end)). Returns wxyz quaternions, shape (len(s), 4).
    """
    if not validate_rotation_matrix(R_start) or not validate_rotation_matrix(R_end):
        raise ValueError("R_start/R_end must be valid SO(3) rotation matrices")
    s = np.asarray(s, dtype=np.float64)
    if s.ndim != 1 or s.shape[0] < 1:
        raise ValueError("s must be a non-empty 1D array of path parameters")

    relative_log = so3_log(R_start.T @ R_end)
    quats = np.empty((s.shape[0], 4), dtype=np.float64)
    for i, si in enumerate(s):
        R_s = R_start @ so3_exp(si * relative_log)
        quats[i] = rotation_matrix_to_quaternion_wxyz(R_s)
    return _canonicalize_sequence(quats)


def bounded_orientation_target(R_anchor: np.ndarray, rotation_vector_rad: np.ndarray) -> np.ndarray:
    """Build a bounded target rotation R1 = R_anchor @ Exp(rotation_vector_rad) from a local delta.

    Intended for generating a "moderate" (well below pi) variable-orientation endpoint that a
    warm-started DLS solve can track continuously from the fixed-orientation anchor pose.
    """
    if not validate_rotation_matrix(R_anchor):
        raise ValueError("R_anchor is not a valid SO(3) rotation matrix")
    rotation_vector_rad = np.asarray(rotation_vector_rad, dtype=np.float64)
    if rotation_vector_rad.shape != (3,):
        raise ValueError("rotation_vector_rad must have shape (3,)")
    return R_anchor @ so3_exp(rotation_vector_rad)
