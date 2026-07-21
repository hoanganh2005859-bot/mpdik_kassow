"""Tier 0: validates FK/Jacobian outputs against finite-difference references and MuJoCo ground truth.

This is the mandatory Tier 0 gate: Tier 1-4 must not run against a model/config that fails these
checks. Three independent validation passes are provided, each over one of the generated
benchmarks/validation/*.npz state sets:

- ``validate_fk_states``: per-sample forward-kinematics sanity (finite pose, valid SO(3) rotation,
  unit quaternion, and repeat-call determinism). ``fk_test_states.npz`` carries no independent
  ground-truth reference pose to diff against, so no "numerical discrepancy vs. reference" field is
  fabricated here -- that column is reported as unavailable rather than invented.
- ``validate_jacobian_states``: analytic vs. central finite-difference Jacobian relative error,
  numerical rank, sigma_min, and condition number.
- ``validate_singularity_states``: sigma_min/condition-number distribution at states the generator
  deliberately placed near singularities. Near-singular states are valid, expected test data --
  never treated as a failure on their own (see ``compute_gate_result``).

``compute_gate_result`` combines the above into the single pass/fail gate that
``pipelines.run_tier0_kinematics`` enforces before Tier 1 is allowed to run.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from kinematics.jacobian import (
    finite_difference_jacobian_world,
    geometric_jacobian_world,
    jacobian_relative_error,
)
from kinematics.model_loader import ModelContext
from kinematics.quaternion_utils import rotation_matrix_to_quaternion_wxyz
from kinematics.rotation_utils import validate_rotation_matrix
from kinematics.singularity_metrics import condition_number, is_near_singular, numerical_rank, singular_values
from utils.csv_utils import json_safe_scalar

DEFAULT_JACOBIAN_RELATIVE_ERROR_THRESHOLD = 1e-4
DEFAULT_ROTATION_TOLERANCE = 1e-6


@dataclass
class FKValidationResult:
    """Per-sample Tier 0 forward-kinematics validation outcome."""

    sample_id: int
    group_id: int
    position_finite: bool
    rotation_finite: bool
    rotation_orthogonality_error: float
    rotation_determinant: float
    quaternion_norm_error: float
    rotation_valid: bool
    deterministic: bool
    passed: bool


@dataclass
class JacobianValidationResult:
    """Per-sample Tier 0 analytic-vs-finite-difference Jacobian validation outcome."""

    sample_id: int
    group_id: int
    relative_error: float
    sigma_min: float
    sigma_max: float
    condition_number: float
    numerical_rank: int
    finite: bool
    passed: bool


@dataclass
class SingularityValidationResult:
    """Per-sample Tier 0 singularity-proximity record (not a pass/fail check)."""

    sample_id: int
    group_id: int
    sigma_min: float
    condition_number: float
    near_singular: bool


def _fk_validation_result_to_dataframe(results: List[FKValidationResult]) -> pd.DataFrame:
    return pd.DataFrame([{k: json_safe_scalar(v) for k, v in r.__dict__.items()} for r in results])


def _jacobian_validation_result_to_dataframe(results: List[JacobianValidationResult]) -> pd.DataFrame:
    return pd.DataFrame([{k: json_safe_scalar(v) for k, v in r.__dict__.items()} for r in results])


def _singularity_validation_result_to_dataframe(results: List[SingularityValidationResult]) -> pd.DataFrame:
    return pd.DataFrame([{k: json_safe_scalar(v) for k, v in r.__dict__.items()} for r in results])


fk_validation_results_to_dataframe = _fk_validation_result_to_dataframe
jacobian_validation_results_to_dataframe = _jacobian_validation_result_to_dataframe
singularity_validation_results_to_dataframe = _singularity_validation_result_to_dataframe


def validate_fk_states(
    model_context: ModelContext,
    q_samples: np.ndarray,
    sample_ids: np.ndarray,
    group_ids: np.ndarray,
    rotation_tolerance: float = DEFAULT_ROTATION_TOLERANCE,
) -> List[FKValidationResult]:
    """Run Tier 0 FK sanity checks for each row of ``q_samples``.

    Checks (per sample): finite position/rotation, SO(3) validity (orthogonality + determinant)
    of the rotation matrix, unit-norm quaternion, and that calling FK twice from the same q
    yields the identical pose (determinism).
    """
    from kinematics.forward_kinematics import forward_kinematics

    data = model_context.new_data()
    results: List[FKValidationResult] = []
    for i in range(q_samples.shape[0]):
        q = q_samples[i]
        fk1 = forward_kinematics(model_context, q, data=data)
        fk2 = forward_kinematics(model_context, q, data=data)

        position_finite = bool(np.all(np.isfinite(fk1.position)))
        rotation_finite = bool(np.all(np.isfinite(fk1.rotation_matrix)))

        if rotation_finite:
            orth_err = float(np.linalg.norm(fk1.rotation_matrix.T @ fk1.rotation_matrix - np.eye(3)))
            det = float(np.linalg.det(fk1.rotation_matrix))
            rotation_valid = validate_rotation_matrix(fk1.rotation_matrix, tol=rotation_tolerance)
        else:
            orth_err = float("nan")
            det = float("nan")
            rotation_valid = False

        quaternion = rotation_matrix_to_quaternion_wxyz(fk1.rotation_matrix) if rotation_finite else None
        quaternion_norm_error = (
            float(abs(np.linalg.norm(quaternion) - 1.0)) if quaternion is not None else float("nan")
        )

        deterministic = bool(
            position_finite
            and rotation_finite
            and np.array_equal(fk1.position, fk2.position)
            and np.array_equal(fk1.rotation_matrix, fk2.rotation_matrix)
        )

        passed = bool(position_finite and rotation_finite and rotation_valid and deterministic)

        results.append(
            FKValidationResult(
                sample_id=int(sample_ids[i]),
                group_id=int(group_ids[i]),
                position_finite=position_finite,
                rotation_finite=rotation_finite,
                rotation_orthogonality_error=orth_err,
                rotation_determinant=det,
                quaternion_norm_error=quaternion_norm_error,
                rotation_valid=rotation_valid,
                deterministic=deterministic,
                passed=passed,
            )
        )
    return results


def validate_jacobian_states(
    model_context: ModelContext,
    q_samples: np.ndarray,
    sample_ids: np.ndarray,
    group_ids: np.ndarray,
    finite_difference_epsilon: float = 1e-6,
    relative_error_threshold: float = DEFAULT_JACOBIAN_RELATIVE_ERROR_THRESHOLD,
) -> List[JacobianValidationResult]:
    """Compare the analytic geometric Jacobian against a central finite-difference Jacobian.

    ``finite_difference_epsilon`` may be a scalar or a per-sample array (matching
    ``jacobian_test_states.npz``'s ``finite_difference_epsilon`` field).
    """
    epsilons = np.broadcast_to(np.asarray(finite_difference_epsilon, dtype=np.float64), (q_samples.shape[0],))
    data = model_context.new_data()
    results: List[JacobianValidationResult] = []
    for i in range(q_samples.shape[0]):
        q = q_samples[i]
        J_analytic = geometric_jacobian_world(model_context, q, data=data)
        J_fd = finite_difference_jacobian_world(model_context, q, epsilon=float(epsilons[i]))
        rel_err = jacobian_relative_error(J_analytic, J_fd)

        sv = singular_values(J_analytic)
        sigma_min = float(sv[-1])
        sigma_max = float(sv[0])
        cond = condition_number(J_analytic)
        rank = numerical_rank(J_analytic)
        finite = bool(np.all(np.isfinite(J_analytic)) and np.all(np.isfinite(J_fd)))
        passed = bool(finite and rel_err <= relative_error_threshold)

        results.append(
            JacobianValidationResult(
                sample_id=int(sample_ids[i]),
                group_id=int(group_ids[i]),
                relative_error=float(rel_err),
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                condition_number=cond,
                numerical_rank=rank,
                finite=finite,
                passed=passed,
            )
        )
    return results


def validate_singularity_states(
    model_context: ModelContext,
    q_samples: np.ndarray,
    sample_ids: np.ndarray,
    group_ids: np.ndarray,
    near_singular_threshold: float,
) -> List[SingularityValidationResult]:
    """Record sigma_min/condition-number at deliberately near-singular test configurations.

    Never a pass/fail check: being near-singular is the expected property of this data, not a
    validation failure (see module docstring).
    """
    data = model_context.new_data()
    results: List[SingularityValidationResult] = []
    for i in range(q_samples.shape[0]):
        q = q_samples[i]
        J = geometric_jacobian_world(model_context, q, data=data)
        sigma_min = float(singular_values(J)[-1])
        cond = condition_number(J)
        results.append(
            SingularityValidationResult(
                sample_id=int(sample_ids[i]),
                group_id=int(group_ids[i]),
                sigma_min=sigma_min,
                condition_number=cond,
                near_singular=is_near_singular(J, near_singular_threshold),
            )
        )
    return results


@dataclass
class Tier0GateResult:
    """Combined Tier 0 pass/fail gate over FK/Jacobian/singularity validation results."""

    gate_pass: bool
    reasons: List[str]
    fk_sample_count: int
    fk_rotation_failures: int
    fk_determinism_failures: int
    jacobian_sample_count: int
    max_jacobian_relative_error: float
    mean_jacobian_relative_error: float
    p95_jacobian_relative_error: float
    jacobian_relative_error_threshold: float
    jacobian_over_threshold_count: int
    minimum_sigma_min: float
    maximum_condition_number: float
    singularity_sample_count: int
    singularity_minimum_sigma_min: float
    singularity_maximum_condition_number: float
    fk_reference_discrepancy_status: str


def compute_gate_result(
    fk_results: List[FKValidationResult],
    jacobian_results: List[JacobianValidationResult],
    singularity_results: List[SingularityValidationResult],
    relative_error_threshold: float = DEFAULT_JACOBIAN_RELATIVE_ERROR_THRESHOLD,
) -> Tier0GateResult:
    """Compute the Tier 0 gate: asset load (implicit, by having reached this point) + no NaN/Inf
    + valid rotations + max Jacobian relative error <= ``relative_error_threshold``.

    Near-singular states are never used as a failure reason (see module docstring).
    """
    if len(fk_results) == 0:
        raise ValueError("cannot compute the Tier 0 gate from an empty FK validation result set")
    if len(jacobian_results) == 0:
        raise ValueError("cannot compute the Tier 0 gate from an empty Jacobian validation result set")

    reasons: List[str] = []

    rotation_failures = sum(1 for r in fk_results if not r.rotation_valid)
    determinism_failures = sum(1 for r in fk_results if not r.deterministic)
    non_finite_fk = sum(1 for r in fk_results if not (r.position_finite and r.rotation_finite))

    if non_finite_fk > 0:
        reasons.append(f"{non_finite_fk} FK sample(s) produced non-finite pose values")
    if rotation_failures > 0:
        reasons.append(f"{rotation_failures} FK sample(s) produced an invalid SO(3) rotation")
    if determinism_failures > 0:
        reasons.append(f"{determinism_failures} FK sample(s) were non-deterministic across repeat calls")

    jacobian_rel_errors = np.array([r.relative_error for r in jacobian_results], dtype=np.float64)
    jacobian_non_finite = sum(1 for r in jacobian_results if not r.finite)
    max_rel_error = float(np.max(jacobian_rel_errors))
    over_threshold_count = int(np.sum(jacobian_rel_errors > relative_error_threshold))

    if jacobian_non_finite > 0:
        reasons.append(f"{jacobian_non_finite} Jacobian sample(s) produced non-finite values")
    if max_rel_error > relative_error_threshold:
        reasons.append(
            f"max Jacobian relative error {max_rel_error:.3e} exceeds threshold {relative_error_threshold:.3e} "
            f"({over_threshold_count} sample(s) over threshold)"
        )

    jacobian_sigma_min = np.array([r.sigma_min for r in jacobian_results], dtype=np.float64)
    jacobian_condition = np.array([r.condition_number for r in jacobian_results], dtype=np.float64)
    finite_condition = jacobian_condition[np.isfinite(jacobian_condition)]

    if singularity_results:
        singularity_sigma_min = np.array([r.sigma_min for r in singularity_results], dtype=np.float64)
        singularity_condition = np.array([r.condition_number for r in singularity_results], dtype=np.float64)
        finite_singularity_condition = singularity_condition[np.isfinite(singularity_condition)]
        singularity_min_sigma = float(np.min(singularity_sigma_min))
        singularity_max_condition = (
            float(np.max(finite_singularity_condition)) if finite_singularity_condition.size else float("inf")
        )
    else:
        singularity_min_sigma = float("nan")
        singularity_max_condition = float("nan")

    gate_pass = bool(non_finite_fk == 0 and rotation_failures == 0 and jacobian_non_finite == 0 and max_rel_error <= relative_error_threshold)

    return Tier0GateResult(
        gate_pass=gate_pass,
        reasons=reasons,
        fk_sample_count=len(fk_results),
        fk_rotation_failures=rotation_failures,
        fk_determinism_failures=determinism_failures,
        jacobian_sample_count=len(jacobian_results),
        max_jacobian_relative_error=max_rel_error,
        mean_jacobian_relative_error=float(np.mean(jacobian_rel_errors)),
        p95_jacobian_relative_error=float(np.percentile(jacobian_rel_errors, 95)),
        jacobian_relative_error_threshold=float(relative_error_threshold),
        jacobian_over_threshold_count=over_threshold_count,
        minimum_sigma_min=float(np.min(jacobian_sigma_min)),
        maximum_condition_number=float(np.max(finite_condition)) if finite_condition.size else float("inf"),
        singularity_sample_count=len(singularity_results),
        singularity_minimum_sigma_min=singularity_min_sigma,
        singularity_maximum_condition_number=singularity_max_condition,
        fk_reference_discrepancy_status="unavailable",
    )
