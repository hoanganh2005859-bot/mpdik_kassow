"""Tests for evaluation.smoothness_metrics and joint_feasibility_metrics; implemented in the Tier 4 evaluation stage."""

import numpy as np
import pytest

from evaluation.joint_feasibility_metrics import compute_joint_feasibility_metrics
from evaluation.smoothness_metrics import compute_smoothness_metrics


def _time(n, dt=0.1):
    return np.arange(n) * dt


# --- 1. Constant q --------------------------------------------------------------------------


def test_constant_q_gives_zero_velocity_acceleration_jerk_and_jump():
    n = 6
    q = np.tile(np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2]), (n, 1))
    time_s = _time(n)

    metrics = compute_smoothness_metrics(q, time_s)

    assert np.allclose(metrics.velocity, 0.0, atol=1e-10)
    assert np.allclose(metrics.acceleration, 0.0, atol=1e-10)
    assert np.allclose(metrics.jerk, 0.0, atol=1e-10)
    assert metrics.max_joint_jump_rad == pytest.approx(0.0, abs=1e-12)


# --- 2. Linear q(t) --------------------------------------------------------------------------


def test_linear_q_gives_constant_velocity_and_near_zero_acceleration_jerk():
    n = 10
    time_s = _time(n)
    slope = np.array([1.0, -0.5, 0.2, 0.0, 0.3, -0.8, 0.4])
    q0 = np.array([0.1, 0.2, -0.1, 0.0, 0.05, 0.15, -0.2])
    q = q0[None, :] + slope[None, :] * time_s[:, None]

    metrics = compute_smoothness_metrics(q, time_s)

    assert np.allclose(metrics.velocity, slope[None, :], atol=1e-10)
    assert np.allclose(metrics.acceleration, 0.0, atol=1e-8)
    assert np.allclose(metrics.jerk, 0.0, atol=1e-6)


# --- 3. Quadratic q(t) -----------------------------------------------------------------------


def test_quadratic_q_gives_constant_acceleration_and_near_zero_jerk():
    n = 12
    time_s = _time(n)
    a = np.array([0.5, -0.3, 0.2, 0.1, -0.1, 0.05, 0.3])
    q = a[None, :] * (time_s[:, None] ** 2)
    expected_acceleration = 2.0 * a

    metrics = compute_smoothness_metrics(q, time_s)

    assert np.allclose(metrics.acceleration, expected_acceleration[None, :], atol=1e-8)
    assert np.allclose(metrics.jerk, 0.0, atol=1e-5)


# --- 4. Cubic q(t): jerk approximately constant per the analytic formula ---------------------


def test_cubic_q_gives_approximately_constant_jerk():
    n = 200
    time_s = np.linspace(0.0, 1.0, n)
    d = np.array([0.2, -0.1, 0.05, 0.3, -0.2, 0.1, 0.15])
    q = d[None, :] * (time_s[:, None] ** 3)
    expected_jerk = 6.0 * d

    metrics = compute_smoothness_metrics(q, time_s)

    interior = metrics.jerk[5:-5]
    assert np.allclose(interior, expected_jerk[None, :], rtol=5e-2, atol=1e-4)


# --- 5. Known discontinuity: detected at the correct joint and timestep ----------------------


def test_known_discontinuity_detected_at_correct_joint_and_timestep():
    n = 8
    q = np.zeros((n, 7))
    time_s = _time(n)
    jump_timestep = 3  # discontinuity between sample 3 and sample 4
    jump_joint = 5
    jump_size = 0.75
    q[jump_timestep + 1 :, jump_joint] += jump_size

    metrics = compute_smoothness_metrics(q, time_s)

    assert metrics.max_joint_jump_rad == pytest.approx(jump_size)
    assert metrics.max_joint_jump_joint_index == jump_joint
    assert metrics.max_joint_jump_timestep_index == jump_timestep


# --- 6. Non-uniform time: derivatives remain accurate -----------------------------------------


def test_non_uniform_time_gives_reasonable_derivatives():
    time_s = np.array([0.0, 0.05, 0.2, 0.25, 0.5, 0.9, 1.0])
    slope = np.array([1.0, 0.5, -0.3, 0.2, 0.0, 0.1, -0.5])
    q = slope[None, :] * time_s[:, None]

    metrics = compute_smoothness_metrics(q, time_s)

    assert np.allclose(metrics.velocity, slope[None, :], atol=1e-8)


# --- 7. Duplicate/non-monotonic time rejected --------------------------------------------------


def test_duplicate_time_is_rejected():
    q = np.zeros((5, 7))
    time_s = np.array([0.0, 0.1, 0.1, 0.3, 0.4])
    with pytest.raises(ValueError):
        compute_smoothness_metrics(q, time_s)


def test_non_monotonic_time_is_rejected():
    q = np.zeros((5, 7))
    time_s = np.array([0.0, 0.2, 0.1, 0.3, 0.4])
    with pytest.raises(ValueError):
        compute_smoothness_metrics(q, time_s)


def test_too_few_points_is_rejected():
    q = np.zeros((1, 7))
    time_s = np.array([0.0])
    with pytest.raises(ValueError):
        compute_smoothness_metrics(q, time_s)


# --- 8. Velocity utilization: below / equal / above limit --------------------------------------


def test_velocity_utilization_below_equal_above_limit():
    q = np.zeros((3, 2))
    velocity_limits = np.array([1.0, 2.0])
    lower = np.array([-3.0, -3.0])
    upper = np.array([3.0, 3.0])
    # joint 0 stays well below its limit (0.5/1.0); joint 1 sits exactly at its limit (2.0/2.0)
    joint_velocity = np.array([[0.5, 2.0], [0.5, 2.0], [0.5, 2.0]])

    metrics = compute_joint_feasibility_metrics(q, joint_velocity, lower, upper, velocity_limits)
    assert metrics.maximum_velocity_utilization == pytest.approx(1.0, rel=1e-6)
    assert metrics.velocity_violation_count == 0  # exactly at the limit is not a violation

    joint_velocity_over = joint_velocity.copy()
    joint_velocity_over[:, 1] = 2.5  # 2.5 / 2.0 = 1.25, clearly above the limit
    metrics_over = compute_joint_feasibility_metrics(q, joint_velocity_over, lower, upper, velocity_limits)
    assert metrics_over.velocity_violation_count > 0
    assert metrics_over.maximum_velocity_utilization == pytest.approx(1.25, rel=1e-6)


# --- 9. Missing acceleration limits: reported unavailable, never fabricated --------------------


def test_missing_acceleration_limits_are_reported_unavailable():
    q = np.zeros((3, 2))
    velocity_limits = np.array([1.0, 1.0])
    lower = np.array([-3.0, -3.0])
    upper = np.array([3.0, 3.0])
    joint_velocity = np.zeros((3, 2))

    metrics = compute_joint_feasibility_metrics(q, joint_velocity, lower, upper, velocity_limits)

    assert metrics.acceleration_status == "unavailable"
    assert metrics.maximum_acceleration_utilization is None
    assert metrics.per_joint_acceleration_utilization is None
    assert metrics.acceleration_violation_count is None


def test_acceleration_limits_supplied_are_used_when_available():
    q = np.zeros((3, 2))
    velocity_limits = np.array([1.0, 1.0])
    lower = np.array([-3.0, -3.0])
    upper = np.array([3.0, 3.0])
    joint_velocity = np.zeros((3, 2))
    joint_acceleration = np.array([[0.5, 2.0], [0.5, 2.0], [0.5, 2.0]])
    acceleration_limits = np.array([1.0, 1.0])

    metrics = compute_joint_feasibility_metrics(
        q,
        joint_velocity,
        lower,
        upper,
        velocity_limits,
        joint_acceleration=joint_acceleration,
        acceleration_limits=acceleration_limits,
    )

    assert metrics.acceleration_status == "available"
    assert metrics.acceleration_violation_count > 0
    assert metrics.maximum_acceleration_utilization == pytest.approx(2.0, rel=1e-6)
