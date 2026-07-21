"""Computes the geometric end-effector Jacobian at a given joint configuration.

Jacobian frame: world (space) frame, consistent with kinematics.pose_error's world-frame
orientation error convention. J = [J_position; J_rotation], shape (6, 7), columns ordered
by the resolved dof addresses of joint_1 ... joint_7 (not assumed to be a fixed index range).
"""

from typing import Optional

import mujoco
import numpy as np

from kinematics.forward_kinematics import forward_kinematics
from kinematics.model_loader import ModelContext
from kinematics.rotation_utils import so3_log
from utils.exceptions import NumericalKinematicsError

_DEFAULT_FD_EPSILON_RAD = 1e-6
_DEFAULT_RELATIVE_ERROR_FLOOR = 1e-12


def geometric_jacobian_world(
    model_context: ModelContext,
    q: np.ndarray,
    data: Optional[mujoco.MjData] = None,
) -> np.ndarray:
    """Analytic geometric Jacobian at ``q``, shape (6, 7), world frame.

    Computed via mujoco.mj_jacSite on the ee_site, then restricted to the resolved
    dof addresses of joint_1 ... joint_7.
    """
    q_arr = model_context.validate_q(q)
    working_data = data if data is not None else model_context.new_data()

    model_context.set_qpos(working_data, q_arr)
    model_context.forward(working_data)

    jacp = np.zeros((3, model_context.model.nv), dtype=np.float64)
    jacr = np.zeros((3, model_context.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model_context.model, working_data, jacp, jacr, model_context.ee_site_id)

    dof_idx = list(model_context.dof_addresses)
    jacobian = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])

    if not np.all(np.isfinite(jacobian)):
        raise NumericalKinematicsError("geometric Jacobian contains non-finite values")
    if jacobian.shape != (6, len(dof_idx)):
        raise NumericalKinematicsError(f"unexpected Jacobian shape {jacobian.shape}")

    return jacobian


def finite_difference_jacobian_world(
    model_context: ModelContext,
    q: np.ndarray,
    epsilon: float = _DEFAULT_FD_EPSILON_RAD,
) -> np.ndarray:
    """Central finite-difference Jacobian at ``q``, shape (6, nq), world frame.

    Position columns use central difference on ee_site position. Orientation columns
    use the world-frame rotation increment Log(R_plus @ R_minus.T) / (2*epsilon) rather
    than finite-differencing Euler angles or quaternion components directly.

    Callers must choose ``q`` and ``epsilon`` such that q +/- epsilon stays within the
    robot's operational limits; this function does not clip or resample.
    """
    q_arr = model_context.validate_q(q)
    if epsilon <= 0.0:
        raise NumericalKinematicsError("epsilon must be positive")

    nq = model_context.nq
    data = model_context.new_data()

    position_columns = []
    rotation_columns = []
    for i in range(nq):
        perturbation = np.zeros(nq, dtype=np.float64)
        perturbation[i] = epsilon

        fk_plus = forward_kinematics(model_context, q_arr + perturbation, data=data)
        fk_minus = forward_kinematics(model_context, q_arr - perturbation, data=data)

        position_columns.append((fk_plus.position - fk_minus.position) / (2.0 * epsilon))
        rotation_increment = so3_log(fk_plus.rotation_matrix @ fk_minus.rotation_matrix.T)
        rotation_columns.append(rotation_increment / (2.0 * epsilon))

    jacobian = np.vstack(
        [
            np.column_stack(position_columns),
            np.column_stack(rotation_columns),
        ]
    )

    if not np.all(np.isfinite(jacobian)):
        raise NumericalKinematicsError("finite-difference Jacobian contains non-finite values")

    return jacobian


def jacobian_relative_error(
    J_analytic: np.ndarray,
    J_fd: np.ndarray,
    numerical_epsilon: float = _DEFAULT_RELATIVE_ERROR_FLOOR,
) -> float:
    """Frobenius-norm relative error between an analytic and finite-difference Jacobian.

    ||J_analytic - J_fd||_F / max(||J_fd||_F, numerical_epsilon)
    """
    J_analytic = np.asarray(J_analytic, dtype=np.float64)
    J_fd = np.asarray(J_fd, dtype=np.float64)
    if J_analytic.shape != J_fd.shape:
        raise NumericalKinematicsError(
            f"Jacobian shape mismatch: {J_analytic.shape} vs {J_fd.shape}"
        )
    numerator = np.linalg.norm(J_analytic - J_fd, ord="fro")
    denominator = max(np.linalg.norm(J_fd, ord="fro"), numerical_epsilon)
    return float(numerator / denominator)
