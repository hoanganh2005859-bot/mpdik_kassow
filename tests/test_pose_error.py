"""Tests for kinematics.pose_error position and SO(3) logarithm orientation error; implemented in the Tier 0 kinematics stage."""

import numpy as np
import pytest

from kinematics.pose_error import (
    full_pose_error,
    orientation_error_vector_world,
    orientation_geodesic_angle,
    position_error_norm,
    position_error_vector,
    weighted_pose_error,
)
from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from kinematics.rotation_utils import so3_exp
from utils.exceptions import NumericalKinematicsError


def test_same_pose_gives_zero_error():
    R = so3_exp(np.array([0.3, -0.2, 0.5]))
    p = np.array([0.1, 0.2, 0.3])
    e = full_pose_error(p, R, p, R)
    assert np.allclose(e, np.zeros(6), atol=1e-10)


def test_pure_translation_error():
    R = np.eye(3)
    p_current = np.array([0.0, 0.0, 0.0])
    p_target = np.array([0.01, -0.02, 0.03])
    e = full_pose_error(p_target, R, p_current, R)
    assert np.allclose(e[:3], p_target - p_current)
    assert np.allclose(e[3:], np.zeros(3), atol=1e-10)


def test_pure_rotation_error():
    p = np.array([0.1, 0.1, 0.1])
    R_current = np.eye(3)
    axis_angle = np.array([0.0, 0.0, 0.4])
    R_target = so3_exp(axis_angle)
    e = full_pose_error(p, R_target, p, R_current)
    assert np.allclose(e[:3], np.zeros(3), atol=1e-10)
    assert np.allclose(e[3:], axis_angle, atol=1e-8)


def test_orientation_error_direction_matches_simple_rotation():
    # A target rotated by +angle about world z relative to current (identity) should
    # produce a world-frame orientation error vector pointing along +z.
    R_current = np.eye(3)
    R_target = so3_exp(np.array([0.0, 0.0, 0.6]))
    e_o = orientation_error_vector_world(R_target, R_current)
    assert e_o[2] > 0.0
    assert np.isclose(e_o[2], 0.6, atol=1e-8)
    assert np.allclose(e_o[:2], 0.0, atol=1e-10)


def test_orientation_geodesic_angle_within_valid_range():
    rng = np.random.default_rng(42)
    for _ in range(20):
        R_target = so3_exp(rng.uniform(-np.pi, np.pi, size=3) * 0.5)
        R_current = so3_exp(rng.uniform(-np.pi, np.pi, size=3) * 0.5)
        angle = orientation_geodesic_angle(R_target, R_current)
        assert 0.0 <= angle <= np.pi + 1e-9


def test_orientation_error_norm_equals_geodesic_angle():
    R_current = so3_exp(np.array([0.1, -0.2, 0.05]))
    R_target = so3_exp(np.array([0.4, 0.1, -0.3]))
    e_o = orientation_error_vector_world(R_target, R_current)
    angle = orientation_geodesic_angle(R_target, R_current)
    assert np.isclose(np.linalg.norm(e_o), angle, atol=1e-8)


def test_quaternion_sign_ambiguity_does_not_change_error_angle():
    rng = np.random.default_rng(7)
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    R_target_pos = quaternion_wxyz_to_matrix(q)
    R_target_neg = quaternion_wxyz_to_matrix(-q)
    R_current = np.eye(3)

    angle_pos = orientation_geodesic_angle(R_target_pos, R_current)
    angle_neg = orientation_geodesic_angle(R_target_neg, R_current)
    assert np.isclose(angle_pos, angle_neg, atol=1e-10)

    e_pos = orientation_error_vector_world(R_target_pos, R_current)
    e_neg = orientation_error_vector_world(R_target_neg, R_current)
    assert np.isclose(np.linalg.norm(e_pos), np.linalg.norm(e_neg), atol=1e-10)


def test_full_pose_error_shape():
    R = so3_exp(np.array([0.1, 0.2, 0.3]))
    p = np.array([0.5, -0.5, 0.2])
    e = full_pose_error(p + 0.01, so3_exp(np.array([0.1, 0.2, 0.35])), p, R)
    assert e.shape == (6,)
    assert np.all(np.isfinite(e))


def test_position_error_vector_and_norm_consistent():
    target = np.array([1.0, 2.0, 3.0])
    current = np.array([1.1, 1.9, 3.2])
    vec = position_error_vector(target, current)
    norm = position_error_norm(target, current)
    assert np.isclose(np.linalg.norm(vec), norm)


def test_weighted_pose_error_does_not_mix_units_implicitly():
    e = np.array([0.01, 0.02, 0.03, 0.1, 0.2, 0.3])
    weighted = weighted_pose_error(e, position_weight=2.0, orientation_weight=0.5)
    assert np.allclose(weighted[:3], e[:3] * 2.0)
    assert np.allclose(weighted[3:], e[3:] * 0.5)


def test_pose_error_rejects_wrong_shapes():
    with pytest.raises(NumericalKinematicsError):
        position_error_vector(np.zeros(2), np.zeros(3))
    with pytest.raises(NumericalKinematicsError):
        orientation_error_vector_world(np.eye(4), np.eye(3))
