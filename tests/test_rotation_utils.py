"""Tests for kinematics.rotation_utils and kinematics.quaternion_utils conversions; implemented in the Tier 0 kinematics stage."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyRotation

from kinematics.quaternion_utils import (
    canonicalize_quaternion_wxyz,
    normalize_quaternion_wxyz,
    quaternion_geodesic_angle,
    quaternion_wxyz_to_matrix,
    rotation_matrix_to_quaternion_wxyz,
)
from kinematics.rotation_utils import (
    project_to_rotation_matrix,
    rotation_geodesic_angle,
    skew,
    so3_exp,
    so3_log,
    validate_rotation_matrix,
    vee,
)
from utils.exceptions import NumericalKinematicsError


def _rotvec_to_matrix(rotvec):
    return ScipyRotation.from_rotvec(rotvec).as_matrix()


def _wxyz_to_scipy_xyzw(q_wxyz):
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


# ---------------------------------------------------------------------------
# skew / vee
# ---------------------------------------------------------------------------


def test_skew_vee_round_trip():
    v = np.array([0.3, -0.7, 1.2])
    assert np.allclose(vee(skew(v)), v)


def test_skew_is_antisymmetric():
    v = np.array([1.0, 2.0, 3.0])
    K = skew(v)
    assert np.allclose(K, -K.T)


# ---------------------------------------------------------------------------
# validate_rotation_matrix / project_to_rotation_matrix
# ---------------------------------------------------------------------------


def test_validate_rotation_matrix_identity_is_valid():
    assert validate_rotation_matrix(np.eye(3)) is True


def test_validate_rotation_matrix_detects_non_orthogonal():
    bad = np.eye(3)
    bad[0, 1] = 0.5
    assert validate_rotation_matrix(bad) is False


def test_validate_rotation_matrix_detects_reflection():
    reflection = np.diag([1.0, 1.0, -1.0])
    assert validate_rotation_matrix(reflection) is False


def test_validate_rotation_matrix_detects_wrong_shape():
    assert validate_rotation_matrix(np.eye(4)) is False


def test_project_to_rotation_matrix_recovers_valid_rotation():
    R = _rotvec_to_matrix([0.4, -0.2, 0.9])
    noisy = R + 1e-3 * np.array([[0.1, -0.2, 0.05], [0.0, 0.1, -0.1], [0.2, 0.0, 0.1]])
    projected = project_to_rotation_matrix(noisy)
    assert validate_rotation_matrix(projected, tol=1e-8)
    assert np.allclose(projected, R, atol=5e-3)


# ---------------------------------------------------------------------------
# so3_exp / so3_log stability
# ---------------------------------------------------------------------------


def test_so3_log_identity_is_zero():
    phi = so3_log(np.eye(3))
    assert np.allclose(phi, np.zeros(3), atol=1e-10)


def test_so3_log_small_angle_is_stable():
    phi_true = np.array([1e-7, -2e-7, 5e-8])
    R = _rotvec_to_matrix(phi_true)
    phi = so3_log(R)
    assert np.all(np.isfinite(phi))
    assert np.allclose(phi, phi_true, atol=1e-9)


def test_so3_log_90_degrees_matches_axis_angle():
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 2.0
    R = _rotvec_to_matrix(axis * angle)
    phi = so3_log(R)
    assert np.isclose(np.linalg.norm(phi), angle, atol=1e-8)
    assert np.allclose(phi / np.linalg.norm(phi), axis, atol=1e-8)


def test_so3_log_near_pi_is_stable():
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    angle = np.pi - 1e-4
    R = _rotvec_to_matrix(axis * angle)
    phi = so3_log(R)
    assert np.all(np.isfinite(phi))
    assert np.isclose(np.linalg.norm(phi), angle, atol=1e-5)


def test_so3_exp_of_so3_log_recovers_rotation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        rotvec = rng.uniform(-np.pi, np.pi, size=3)
        R = _rotvec_to_matrix(rotvec)
        R_recovered = so3_exp(so3_log(R))
        assert np.allclose(R_recovered, R, atol=1e-8)


def test_so3_log_of_so3_exp_in_principal_range():
    rng = np.random.default_rng(1)
    for _ in range(20):
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle = rng.uniform(0.0, np.pi - 1e-3)
        phi_in = axis * angle
        R = so3_exp(phi_in)
        phi_out = so3_log(R)
        assert np.linalg.norm(phi_out) <= np.pi + 1e-9
        assert np.allclose(phi_out, phi_in, atol=1e-6)


def test_so3_exp_zero_is_identity():
    assert np.allclose(so3_exp(np.zeros(3)), np.eye(3))


# ---------------------------------------------------------------------------
# quaternion conversions
# ---------------------------------------------------------------------------


def test_quaternion_round_trip_matches_scipy():
    rng = np.random.default_rng(2)
    for _ in range(20):
        rotvec = rng.uniform(-np.pi, np.pi, size=3)
        R = _rotvec_to_matrix(rotvec)
        q_wxyz = rotation_matrix_to_quaternion_wxyz(R)
        R_back = quaternion_wxyz_to_matrix(q_wxyz)
        assert np.allclose(R_back, R, atol=1e-8)

        scipy_quat_xyzw = ScipyRotation.from_matrix(R).as_quat()
        scipy_quat_wxyz = np.array(
            [scipy_quat_xyzw[3], scipy_quat_xyzw[0], scipy_quat_xyzw[1], scipy_quat_xyzw[2]]
        )
        angle_between = quaternion_geodesic_angle(q_wxyz, scipy_quat_wxyz)
        assert angle_between < 1e-6


def test_quaternion_output_is_wxyz_order_for_identity():
    q = rotation_matrix_to_quaternion_wxyz(np.eye(3))
    assert np.allclose(q, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-8)


def test_quaternion_and_negated_quaternion_are_equivalent():
    rng = np.random.default_rng(3)
    q = rng.normal(size=4)
    q = normalize_quaternion_wxyz(q)
    R_pos = quaternion_wxyz_to_matrix(q)
    R_neg = quaternion_wxyz_to_matrix(-q)
    assert np.allclose(R_pos, R_neg, atol=1e-10)
    assert quaternion_geodesic_angle(q, -q) < 1e-10


def test_canonicalize_quaternion_produces_consistent_sign():
    rng = np.random.default_rng(4)
    q = rng.normal(size=4)
    q = normalize_quaternion_wxyz(q)
    canon_pos = canonicalize_quaternion_wxyz(q)
    canon_neg = canonicalize_quaternion_wxyz(-q)
    assert np.allclose(canon_pos, canon_neg, atol=1e-10)


def test_quaternion_geodesic_angle_zero_for_identical_quaternion():
    rng = np.random.default_rng(5)
    q = normalize_quaternion_wxyz(rng.normal(size=4))
    assert quaternion_geodesic_angle(q, q) < 1e-10


def test_quaternion_geodesic_angle_in_valid_range():
    rng = np.random.default_rng(6)
    for _ in range(20):
        q1 = normalize_quaternion_wxyz(rng.normal(size=4))
        q2 = normalize_quaternion_wxyz(rng.normal(size=4))
        angle = quaternion_geodesic_angle(q1, q2)
        assert 0.0 <= angle <= np.pi + 1e-9


# ---------------------------------------------------------------------------
# rotation_geodesic_angle / arccos clamping
# ---------------------------------------------------------------------------


def test_rotation_geodesic_angle_zero_for_identical_rotation():
    # arccos(x) near x=1 has an unbounded derivative, so even a ~1e-16 floating point
    # perturbation in trace(R.T @ R) can surface as a ~1e-8 rad angle; tolerance reflects that.
    R = _rotvec_to_matrix([0.3, 0.1, -0.2])
    assert rotation_geodesic_angle(R, R) < 1e-6


def test_rotation_geodesic_angle_handles_numerical_drift_beyond_one():
    # Slightly perturbed identity-like matrix whose trace-based cosine can drift
    # a hair outside [-1, 1] due to floating point noise; must not raise/NaN.
    R1 = np.eye(3)
    R2 = np.eye(3) + 1e-16 * np.ones((3, 3))
    angle = rotation_geodesic_angle(R1, R2)
    assert np.isfinite(angle)
    assert angle >= 0.0


def test_invalid_rotation_matrix_rejected_by_shape_functions():
    with pytest.raises(NumericalKinematicsError):
        so3_log(np.eye(4))
    with pytest.raises(NumericalKinematicsError):
        so3_exp(np.array([1.0, 2.0]))
