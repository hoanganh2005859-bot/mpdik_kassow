"""Computes end-effector position/orientation from joint configuration via the MuJoCo model.

Pose is read directly from the compiled model's ee_site (data.site_xpos / data.site_xmat)
after mj_forward — never from mesh geometry, and never with an added TCP offset.
"""

from dataclasses import dataclass
from typing import List, Optional

import mujoco
import numpy as np

from kinematics.model_loader import ModelContext
from kinematics.quaternion_utils import rotation_matrix_to_quaternion_wxyz


@dataclass(frozen=True)
class FKResult:
    """End-effector pose at a given joint configuration."""

    position: np.ndarray  # (3,), meters, world frame
    rotation_matrix: np.ndarray  # (3, 3), world frame
    quaternion_wxyz: np.ndarray  # (4,), wxyz order, world frame


def forward_kinematics(
    model_context: ModelContext,
    q: np.ndarray,
    data: Optional[mujoco.MjData] = None,
) -> FKResult:
    """Compute the ee_site pose for joint configuration ``q``.

    Args:
        model_context: Loaded ModelContext (see kinematics.model_loader).
        q: Joint vector, shape (nq,), radians. Validated for shape/finiteness;
            never silently clipped.
        data: Optional caller-owned MjData to reuse (e.g. to avoid repeated
            allocation in a hot loop). If omitted, a fresh MjData is created via
            ``model_context.new_data()`` so no state is shared across calls.

    Returns:
        FKResult with position, rotation matrix, and wxyz quaternion, all in world frame.
    """
    q_arr = model_context.validate_q(q)
    working_data = data if data is not None else model_context.new_data()

    model_context.set_qpos(working_data, q_arr)
    model_context.forward(working_data)

    site_id = model_context.ee_site_id
    position = np.array(working_data.site_xpos[site_id], dtype=np.float64, copy=True)
    rotation_matrix = np.array(
        working_data.site_xmat[site_id], dtype=np.float64, copy=True
    ).reshape(3, 3)
    quaternion = rotation_matrix_to_quaternion_wxyz(rotation_matrix)

    return FKResult(position=position, rotation_matrix=rotation_matrix, quaternion_wxyz=quaternion)


def forward_kinematics_batch(
    model_context: ModelContext,
    q_batch: List[np.ndarray],
) -> List[FKResult]:
    """Convenience helper: compute FK for a list/array of joint vectors, one MjData reused."""
    data = model_context.new_data()
    return [forward_kinematics(model_context, q, data=data) for q in q_batch]
