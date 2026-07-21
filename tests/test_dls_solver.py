"""Tests for kinematics.dls_solver convergence and limit handling; implemented in the Tier 1 point-DLS stage."""

import numpy as np
import pytest

from kinematics.adaptive_damping import compute_adaptive_damping
from kinematics.dls_solver import (
    dls_single_update,
    load_dls_config,
    solve_dls_until_converged,
)
from kinematics.forward_kinematics import forward_kinematics
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import joint_center
from kinematics.model_loader import load_model_context
from kinematics.singularity_metrics import minimum_singular_value
from utils.exceptions import InvalidJointVectorError

CONTEXT = load_model_context()
BASE_CONFIG = load_dls_config()

REGULAR_Q_SAMPLES = [
    np.array([0.2, 0.3, -0.1, 0.4, 0.15, -0.25, 0.1]),
    np.array([-0.3, 0.6, 0.2, 0.8, -0.4, 0.3, -0.2]),
    np.array([0.5, -0.5, 0.5, 0.2, 0.6, -0.6, 0.4]),
]

_ALLOWED_FAILURE_REASONS = {
    "max_iterations",
    "non_finite_input",
    "non_finite_jacobian",
    "linear_solve_failure",
    "joint_limit_failure",
    "invalid_target",
    "stagnation",
}


def test_target_equals_current_converges_with_zero_iterations():
    q_initial = REGULAR_Q_SAMPLES[0]
    fk = forward_kinematics(CONTEXT, q_initial)
    result = solve_dls_until_converged(CONTEXT, q_initial, fk.position, fk.rotation_matrix, config=BASE_CONFIG)
    assert result.success is True
    assert result.iterations == 0
    assert np.allclose(result.q_solution, q_initial)
    assert result.failure_reason is None


def test_solver_converges_to_nearby_target():
    q_initial = REGULAR_Q_SAMPLES[1]
    rng = np.random.default_rng(11)
    q_target_true = q_initial + rng.uniform(-0.05, 0.05, size=CONTEXT.nq)
    target_pose = forward_kinematics(CONTEXT, q_target_true)

    result = solve_dls_until_converged(
        CONTEXT, q_initial, target_pose.position, target_pose.rotation_matrix, config=BASE_CONFIG
    )

    assert result.success is True
    assert result.position_error_m <= BASE_CONFIG["position_success_threshold_m"]
    assert result.orientation_error_deg <= BASE_CONFIG["orientation_success_threshold_deg"]
    assert not result.joint_limit_violation
    assert np.all(np.isfinite(result.q_solution))


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_solver_handles_random_reachable_targets_without_crashing(seed):
    rng = np.random.default_rng(seed)
    q_initial = REGULAR_Q_SAMPLES[seed % len(REGULAR_Q_SAMPLES)]
    q_target_true = rng.uniform(-1.0, 1.0, size=CONTEXT.nq)
    q_target_true = np.clip(q_target_true, CONTEXT.operational_lower_rad, CONTEXT.operational_upper_rad)
    target_pose = forward_kinematics(CONTEXT, q_target_true)

    result = solve_dls_until_converged(
        CONTEXT, q_initial, target_pose.position, target_pose.rotation_matrix, config=BASE_CONFIG
    )

    assert np.all(np.isfinite(result.q_solution))
    assert np.isfinite(result.position_error_m)
    assert np.isfinite(result.orientation_error_rad)
    if not result.success:
        assert result.failure_reason in _ALLOWED_FAILURE_REASONS


def test_near_singular_configuration_increases_damping_and_stays_finite():
    rng = np.random.default_rng(123)
    candidates = []
    for _ in range(200):
        q = np.empty(CONTEXT.nq)
        q[0] = rng.uniform(-1.5, 1.5)
        q[1] = rng.uniform(-1.0, 2.5)
        q[2] = rng.uniform(-1.5, 1.5)
        q[3] = rng.uniform(-1.0, 2.5)
        q[4] = rng.uniform(-1.5, 1.5)
        q[5] = rng.uniform(-1.5, 1.5)
        q[6] = rng.uniform(-1.5, 1.5)
        sigma_min = minimum_singular_value(geometric_jacobian_world(CONTEXT, q))
        candidates.append((sigma_min, q))
    candidates.sort(key=lambda c: c[0])

    near_singular_q = candidates[0][1]
    regular_q = candidates[-1][1]

    sigma_near = candidates[0][0]
    sigma_regular = candidates[-1][0]

    damping_near = compute_adaptive_damping(
        sigma_near, BASE_CONFIG["singularity_sigma_threshold"], BASE_CONFIG["lambda_min"], BASE_CONFIG["lambda_max"]
    )
    damping_regular = compute_adaptive_damping(
        sigma_regular, BASE_CONFIG["singularity_sigma_threshold"], BASE_CONFIG["lambda_min"], BASE_CONFIG["lambda_max"]
    )
    assert damping_near >= damping_regular

    # Drive a single update step at the near-singular pose toward an arbitrary nearby target.
    target_q = near_singular_q + 0.05
    target_q = np.clip(target_q, CONTEXT.operational_lower_rad, CONTEXT.operational_upper_rad)
    target_pose = forward_kinematics(CONTEXT, target_q)

    step = dls_single_update(
        CONTEXT, near_singular_q, target_pose.position, target_pose.rotation_matrix, BASE_CONFIG
    )
    assert step.failure_reason is None
    assert np.all(np.isfinite(step.q_next))
    assert np.all(np.isfinite(step.delta_q))
    assert np.max(np.abs(step.delta_q)) <= BASE_CONFIG["max_joint_step_rad"] + 1e-9


def test_max_joint_step_is_respected_for_large_pose_error():
    q_initial = REGULAR_Q_SAMPLES[0]
    far_target_q = REGULAR_Q_SAMPLES[0] + 1.5  # deliberately large joint-space error
    far_target_q = np.clip(far_target_q, CONTEXT.operational_lower_rad, CONTEXT.operational_upper_rad)
    target_pose = forward_kinematics(CONTEXT, far_target_q)

    small_step_config = dict(BASE_CONFIG)
    small_step_config["max_joint_step_rad"] = 0.02

    step = dls_single_update(
        CONTEXT, q_initial, target_pose.position, target_pose.rotation_matrix, small_step_config
    )
    assert step.failure_reason is None
    assert np.all(np.abs(step.delta_q) <= small_step_config["max_joint_step_rad"] + 1e-9)


def test_clip_to_operational_limits_engages_and_can_be_disabled():
    lower = CONTEXT.operational_lower_rad
    upper = CONTEXT.operational_upper_rad

    q_near_limit = joint_center(lower, upper)
    q_near_limit[0] = upper[0] - 0.02

    q_beyond_limit = q_near_limit.copy()
    q_beyond_limit[0] = upper[0] + 0.05
    target_pose = forward_kinematics(CONTEXT, q_beyond_limit)

    clip_enabled_config = dict(BASE_CONFIG)
    clip_enabled_config["clip_to_operational_limits"] = True
    clip_enabled_config["joint_limit_avoidance"] = False
    clip_enabled_config["max_joint_step_rad"] = 0.2

    step_clipped = dls_single_update(
        CONTEXT, q_near_limit, target_pose.position, target_pose.rotation_matrix, clip_enabled_config
    )
    assert step_clipped.failure_reason is None
    assert step_clipped.joint_limit_violation_before_clip is True
    assert step_clipped.q_next[0] <= upper[0] + 1e-9

    clip_disabled_config = dict(clip_enabled_config)
    clip_disabled_config["clip_to_operational_limits"] = False

    step_unclipped = dls_single_update(
        CONTEXT, q_near_limit, target_pose.position, target_pose.rotation_matrix, clip_disabled_config
    )
    assert step_unclipped.failure_reason == "joint_limit_failure"


def test_invalid_joint_vector_shape_is_rejected():
    with pytest.raises(InvalidJointVectorError):
        solve_dls_until_converged(CONTEXT, np.zeros(3), np.zeros(3), np.eye(3), config=BASE_CONFIG)


def test_invalid_target_rotation_is_rejected():
    q_initial = REGULAR_Q_SAMPLES[0]
    invalid_rotation = np.eye(3) * 2.0  # not orthogonal, not a valid rotation matrix
    result = solve_dls_until_converged(
        CONTEXT, q_initial, np.array([0.3, 0.2, 0.4]), invalid_rotation, config=BASE_CONFIG
    )
    assert result.success is False
    assert result.failure_reason == "invalid_target"
    assert np.allclose(result.q_solution, q_initial)


def test_dls_result_dataclass_has_all_required_fields():
    q_initial = REGULAR_Q_SAMPLES[2]
    rng = np.random.default_rng(99)
    q_target_true = q_initial + rng.uniform(-0.05, 0.05, size=CONTEXT.nq)
    target_pose = forward_kinematics(CONTEXT, q_target_true)

    result = solve_dls_until_converged(
        CONTEXT, q_initial, target_pose.position, target_pose.rotation_matrix, config=BASE_CONFIG
    )

    required_fields = [
        "q_solution",
        "success",
        "iterations",
        "position_error_m",
        "orientation_error_rad",
        "orientation_error_deg",
        "solve_time_ms",
        "sigma_min",
        "condition_number",
        "damping",
        "joint_limit_violation",
        "minimum_joint_limit_margin",
        "failure_reason",
        "error_history",
    ]
    for field_name in required_fields:
        assert hasattr(result, field_name)
    assert result.solve_time_ms >= 0.0
    assert result.q_solution.shape == (CONTEXT.nq,)
