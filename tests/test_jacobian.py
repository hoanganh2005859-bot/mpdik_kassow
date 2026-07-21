"""Tests for kinematics.jacobian against finite-difference references; implemented in the Tier 0 kinematics stage."""

import numpy as np
import pytest

from kinematics.jacobian import (
    finite_difference_jacobian_world,
    geometric_jacobian_world,
    jacobian_relative_error,
)
from kinematics.model_loader import load_model_context
from kinematics.singularity_metrics import numerical_rank

CONTEXT = load_model_context()

# Sample configurations that stay well inside every joint's operational range,
# including the narrower joint_2/joint_4 range of [-1.22173, 3.14159].
SAMPLE_CONFIGS = [
    np.zeros(CONTEXT.nq),
    np.array([0.2, 0.3, -0.1, 0.4, 0.15, -0.25, 0.1]),
    np.array([-0.3, 0.6, 0.2, 0.8, -0.4, 0.3, -0.2]),
    np.array([0.5, -0.5, 0.5, 0.2, 0.6, -0.6, 0.4]),
    np.array([-0.15, 0.1, -0.35, 0.9, 0.25, 0.45, -0.3]),
]

FD_EPSILON = 1e-6
RELATIVE_ERROR_TOLERANCE = 1e-4


def test_geometric_jacobian_shape():
    J = geometric_jacobian_world(CONTEXT, SAMPLE_CONFIGS[1])
    assert J.shape == (6, CONTEXT.nq)


def test_geometric_jacobian_is_finite():
    for q in SAMPLE_CONFIGS:
        J = geometric_jacobian_world(CONTEXT, q)
        assert np.all(np.isfinite(J))


def test_geometric_jacobian_has_full_rank_at_regular_pose():
    J = geometric_jacobian_world(CONTEXT, SAMPLE_CONFIGS[1])
    assert numerical_rank(J) == min(J.shape)


def test_finite_difference_jacobian_shape_and_finite():
    J_fd = finite_difference_jacobian_world(CONTEXT, SAMPLE_CONFIGS[2], epsilon=FD_EPSILON)
    assert J_fd.shape == (6, CONTEXT.nq)
    assert np.all(np.isfinite(J_fd))


@pytest.mark.parametrize("q", SAMPLE_CONFIGS)
def test_geometric_jacobian_matches_finite_difference(q):
    J_analytic = geometric_jacobian_world(CONTEXT, q)
    J_fd = finite_difference_jacobian_world(CONTEXT, q, epsilon=FD_EPSILON)
    relative_error = jacobian_relative_error(J_analytic, J_fd)
    assert relative_error <= RELATIVE_ERROR_TOLERANCE, (
        f"relative error {relative_error} exceeds tolerance {RELATIVE_ERROR_TOLERANCE} at q={q}"
    )


def test_jacobian_relative_error_is_zero_for_identical_jacobians():
    J = geometric_jacobian_world(CONTEXT, SAMPLE_CONFIGS[0])
    assert jacobian_relative_error(J, J) == 0.0


def test_jacobian_relative_error_rejects_shape_mismatch():
    J = geometric_jacobian_world(CONTEXT, SAMPLE_CONFIGS[0])
    with pytest.raises(Exception):
        jacobian_relative_error(J, J[:, :5])
