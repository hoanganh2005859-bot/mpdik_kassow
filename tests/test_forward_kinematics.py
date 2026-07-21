"""Tests for kinematics.forward_kinematics correctness; implemented in the Tier 0 kinematics stage."""

import mujoco
import numpy as np
import pytest

from kinematics.forward_kinematics import forward_kinematics
from kinematics.model_loader import load_model_context
from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from kinematics.rotation_utils import validate_rotation_matrix
from utils.exceptions import InvalidJointVectorError

CONTEXT = load_model_context()


def test_fk_runs_at_zero_configuration():
    q = np.zeros(CONTEXT.nq)
    result = forward_kinematics(CONTEXT, q)
    assert result is not None


def test_fk_output_shapes():
    q = np.zeros(CONTEXT.nq)
    result = forward_kinematics(CONTEXT, q)
    assert result.position.shape == (3,)
    assert result.rotation_matrix.shape == (3, 3)
    assert result.quaternion_wxyz.shape == (4,)


def test_fk_output_is_finite():
    q = np.array([0.2, -0.3, 0.5, -0.6, 0.1, 0.4, -0.2])
    result = forward_kinematics(CONTEXT, q)
    assert np.all(np.isfinite(result.position))
    assert np.all(np.isfinite(result.rotation_matrix))
    assert np.all(np.isfinite(result.quaternion_wxyz))


def test_fk_rotation_matrix_is_orthogonal_with_unit_determinant():
    q = np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.1])
    result = forward_kinematics(CONTEXT, q)
    assert validate_rotation_matrix(result.rotation_matrix, tol=1e-6)
    assert np.isclose(np.linalg.det(result.rotation_matrix), 1.0, atol=1e-6)


def test_fk_quaternion_is_normalized():
    q = np.array([0.5, -0.1, 0.2, 0.3, -0.4, 0.1, 0.2])
    result = forward_kinematics(CONTEXT, q)
    assert np.isclose(np.linalg.norm(result.quaternion_wxyz), 1.0, atol=1e-8)


def test_fk_quaternion_is_wxyz_and_matches_rotation_matrix():
    q = np.array([0.3, 0.1, -0.2, 0.25, 0.15, -0.05, 0.4])
    result = forward_kinematics(CONTEXT, q)
    # wxyz convention: quaternion_wxyz_to_matrix must reproduce the same rotation matrix.
    reconstructed = quaternion_wxyz_to_matrix(result.quaternion_wxyz)
    assert np.allclose(reconstructed, result.rotation_matrix, atol=1e-6)
    # scalar component first (w), within [-1, 1]
    assert -1.0 <= result.quaternion_wxyz[0] <= 1.0


def test_fk_is_deterministic_for_same_q():
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.1, 0.2])
    result_a = forward_kinematics(CONTEXT, q)
    result_b = forward_kinematics(CONTEXT, q)
    assert np.allclose(result_a.position, result_b.position)
    assert np.allclose(result_a.rotation_matrix, result_b.rotation_matrix)
    assert np.allclose(result_a.quaternion_wxyz, result_b.quaternion_wxyz)


def test_fk_rejects_wrong_shape_q():
    with pytest.raises(InvalidJointVectorError):
        forward_kinematics(CONTEXT, np.zeros(6))


def test_fk_rejects_nan_q():
    q = np.zeros(CONTEXT.nq)
    q[3] = np.nan
    with pytest.raises(InvalidJointVectorError):
        forward_kinematics(CONTEXT, q)


def test_fk_does_not_silently_clip_out_of_range_q():
    q = np.full(CONTEXT.nq, 100.0)  # clearly out of any operational range
    # Should not raise for being out of range (no silent clip either); shape/finite is all that's checked.
    result = forward_kinematics(CONTEXT, q)
    assert np.all(np.isfinite(result.position))


def test_fk_matches_direct_mujoco_site_pose():
    q = np.array([0.2, 0.1, -0.15, 0.3, -0.25, 0.05, -0.1])
    data = mujoco.MjData(CONTEXT.model)
    for addr, value in zip(CONTEXT.qpos_addresses, q):
        data.qpos[addr] = value
    mujoco.mj_forward(CONTEXT.model, data)

    expected_position = np.array(data.site_xpos[CONTEXT.ee_site_id])
    expected_rotation = np.array(data.site_xmat[CONTEXT.ee_site_id]).reshape(3, 3)

    result = forward_kinematics(CONTEXT, q)
    assert np.allclose(result.position, expected_position)
    assert np.allclose(result.rotation_matrix, expected_rotation)
