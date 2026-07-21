"""Damped Least Squares inverse kinematics solver core, configured via configs/dls_config.json.

Solves, per update, the weighted-damped normal equations equivalent to
    minimize ||W^(1/2) (J @ delta_q - e)||^2 + lambda^2 ||delta_q||^2
via
    (J.T @ W @ J + lambda^2 * I) @ delta_q = J.T @ W @ e
using np.linalg.solve (never a direct np.linalg.inv). Damping lambda is adaptive in
sigma_min (see kinematics.adaptive_damping). An optional null-space joint-centering
term uses the Moore-Penrose pseudo-inverse (np.linalg.pinv, SVD-based) projector
(I - J^+ J); if joint_limit_avoidance is disabled in config, only the exact weighted
DLS term is used.

This module is self-contained within kinematics/: it does not import from
algorithms/, evaluation/, or pipelines/.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from kinematics.adaptive_damping import compute_adaptive_damping
from kinematics.forward_kinematics import forward_kinematics
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import (
    clip_to_operational_limits,
    joint_centering_gradient,
    minimum_joint_limit_margin,
    operational_limit_violation_mask,
)
from kinematics.model_loader import ModelContext
from kinematics.pose_error import full_pose_error, weighted_pose_error
from kinematics.rotation_utils import validate_rotation_matrix
from kinematics.singularity_metrics import condition_number as safe_condition_number
from kinematics.singularity_metrics import minimum_singular_value
from utils.config_loader import load_json_config
from utils.exceptions import NumericalKinematicsError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DLS_CONFIG_PATH = REPO_ROOT / "configs" / "dls_config.json"

_STAGNATION_WINDOW = 5
_STAGNATION_MIN_RELATIVE_IMPROVEMENT = 1e-3


def load_dls_config(path: Optional[Path] = None) -> dict:
    """Load the DLS solver configuration (defaults to configs/dls_config.json)."""
    return load_json_config(path or DEFAULT_DLS_CONFIG_PATH)


@dataclass
class DLSStepResult:
    """Result of a single dls_single_update call."""

    q_next: np.ndarray
    delta_q: np.ndarray
    sigma_min: float
    condition_number: float
    damping: float
    joint_limit_violation_before_clip: bool
    joint_limit_violation_after_clip: bool
    failure_reason: Optional[str]


@dataclass
class DLSResult:
    """Result of a full solve_dls_until_converged run."""

    q_solution: np.ndarray
    success: bool
    iterations: int
    position_error_m: float
    orientation_error_rad: float
    orientation_error_deg: float
    solve_time_ms: float
    sigma_min: float
    condition_number: float
    damping: float
    joint_limit_violation: bool
    minimum_joint_limit_margin: float
    failure_reason: Optional[str]
    error_history: Optional[list] = field(default=None)


def _pose_error_components(target_position, target_rotation, current_position, current_rotation):
    e = full_pose_error(target_position, target_rotation, current_position, current_rotation)
    position_error_m = float(np.linalg.norm(e[:3]))
    orientation_error_rad = float(np.linalg.norm(e[3:]))
    return e, position_error_m, orientation_error_rad


def dls_single_update(
    model_context: ModelContext,
    q: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    config: dict,
    data: Optional[mujoco.MjData] = None,
) -> DLSStepResult:
    """Perform one weighted, adaptively-damped DLS update step from ``q`` toward the target.

    Assumes ``q``, ``target_position``, and ``target_rotation`` have already been
    validated by the caller (see solve_dls_until_converged). Never raises for
    anticipated numerical failure modes; instead returns a DLSStepResult with
    ``failure_reason`` set and ``q_next`` left equal to the input ``q``.
    """
    nq = model_context.nq
    working_data = data if data is not None else model_context.new_data()
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad

    try:
        J = geometric_jacobian_world(model_context, q, data=working_data)
    except NumericalKinematicsError:
        return DLSStepResult(
            q_next=q.copy(),
            delta_q=np.zeros(nq),
            sigma_min=0.0,
            condition_number=float("inf"),
            damping=0.0,
            joint_limit_violation_before_clip=bool(np.any(operational_limit_violation_mask(q, lower, upper))),
            joint_limit_violation_after_clip=bool(np.any(operational_limit_violation_mask(q, lower, upper))),
            failure_reason="non_finite_jacobian",
        )

    current_pose = forward_kinematics(model_context, q, data=working_data)
    if not (
        np.all(np.isfinite(current_pose.position)) and np.all(np.isfinite(current_pose.rotation_matrix))
    ):
        return DLSStepResult(
            q_next=q.copy(),
            delta_q=np.zeros(nq),
            sigma_min=0.0,
            condition_number=float("inf"),
            damping=0.0,
            joint_limit_violation_before_clip=bool(np.any(operational_limit_violation_mask(q, lower, upper))),
            joint_limit_violation_after_clip=bool(np.any(operational_limit_violation_mask(q, lower, upper))),
            failure_reason="non_finite_input",
        )

    e, _, _ = _pose_error_components(
        target_position, target_rotation, current_pose.position, current_pose.rotation_matrix
    )

    sigma_min = minimum_singular_value(J)
    cond = safe_condition_number(J)

    if config.get("damping_mode", "adaptive") == "adaptive":
        damping = compute_adaptive_damping(
            sigma_min,
            config["singularity_sigma_threshold"],
            config["lambda_min"],
            config["lambda_max"],
        )
    else:
        damping = float(config.get("lambda_default", config["lambda_min"]))

    position_weight = float(config.get("position_weight", 1.0))
    orientation_weight = float(config.get("orientation_weight", 1.0))
    W = np.diag(np.array([position_weight] * 3 + [orientation_weight] * 3, dtype=np.float64))

    A = J.T @ W @ J + (damping ** 2) * np.eye(nq)
    b = J.T @ W @ e

    try:
        delta_q = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return DLSStepResult(
            q_next=q.copy(),
            delta_q=np.zeros(nq),
            sigma_min=sigma_min,
            condition_number=cond,
            damping=damping,
            joint_limit_violation_before_clip=bool(np.any(operational_limit_violation_mask(q, lower, upper))),
            joint_limit_violation_after_clip=bool(np.any(operational_limit_violation_mask(q, lower, upper))),
            failure_reason="linear_solve_failure",
        )

    if config.get("joint_limit_avoidance", False):
        null_space_gain = float(config.get("null_space_gain", 0.0))
        if null_space_gain != 0.0:
            J_pinv = np.linalg.pinv(J)
            null_projector = np.eye(nq) - J_pinv @ J
            z = joint_centering_gradient(q, lower, upper)
            delta_q = delta_q + null_space_gain * (null_projector @ z)

    step_scale = float(config.get("step_scale", 1.0))
    delta_q = delta_q * step_scale

    max_joint_step = config.get("max_joint_step_rad")
    if max_joint_step is not None:
        delta_q = np.clip(delta_q, -float(max_joint_step), float(max_joint_step))

    q_candidate = q + delta_q

    if not np.all(np.isfinite(q_candidate)):
        return DLSStepResult(
            q_next=q.copy(),
            delta_q=np.zeros(nq),
            sigma_min=sigma_min,
            condition_number=cond,
            damping=damping,
            joint_limit_violation_before_clip=bool(np.any(operational_limit_violation_mask(q, lower, upper))),
            joint_limit_violation_after_clip=bool(np.any(operational_limit_violation_mask(q, lower, upper))),
            failure_reason="non_finite_input",
        )

    violation_before_clip = bool(np.any(operational_limit_violation_mask(q_candidate, lower, upper)))
    clip_enabled = bool(config.get("clip_to_operational_limits", False))

    if violation_before_clip and not clip_enabled:
        return DLSStepResult(
            q_next=q.copy(),
            delta_q=np.zeros(nq),
            sigma_min=sigma_min,
            condition_number=cond,
            damping=damping,
            joint_limit_violation_before_clip=True,
            joint_limit_violation_after_clip=True,
            failure_reason="joint_limit_failure",
        )

    q_final = clip_to_operational_limits(q_candidate, lower, upper) if clip_enabled else q_candidate
    violation_after_clip = bool(np.any(operational_limit_violation_mask(q_final, lower, upper)))

    return DLSStepResult(
        q_next=q_final,
        delta_q=delta_q,
        sigma_min=sigma_min,
        condition_number=cond,
        damping=damping,
        joint_limit_violation_before_clip=violation_before_clip,
        joint_limit_violation_after_clip=violation_after_clip,
        failure_reason=None,
    )


def solve_dls_until_converged(
    model_context: ModelContext,
    q_init: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    config: Optional[dict] = None,
    record_history: bool = False,
) -> DLSResult:
    """Iteratively solve IK for ``target_position``/``target_rotation`` starting from ``q_init``.

    Checks the success condition at the initial state before performing any update
    (target == current returns success=True, iterations=0, q_solution unchanged).
    """
    start_time = time.perf_counter()
    cfg = config if config is not None else load_dls_config()

    q = model_context.validate_q(q_init).copy()
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad

    target_position_arr = np.asarray(target_position, dtype=np.float64)
    target_rotation_arr = np.asarray(target_rotation, dtype=np.float64)

    target_valid = (
        target_position_arr.shape == (3,)
        and np.all(np.isfinite(target_position_arr))
        and target_rotation_arr.shape == (3, 3)
        and np.all(np.isfinite(target_rotation_arr))
        and validate_rotation_matrix(target_rotation_arr)
    )
    if not target_valid:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return DLSResult(
            q_solution=q.copy(),
            success=False,
            iterations=0,
            position_error_m=float("inf"),
            orientation_error_rad=float("inf"),
            orientation_error_deg=float("inf"),
            solve_time_ms=elapsed_ms,
            sigma_min=0.0,
            condition_number=float("inf"),
            damping=0.0,
            joint_limit_violation=bool(np.any(operational_limit_violation_mask(q, lower, upper))),
            minimum_joint_limit_margin=minimum_joint_limit_margin(q, lower, upper),
            failure_reason="invalid_target",
            error_history=[] if record_history else None,
        )

    max_iterations = int(cfg.get("max_iterations", 100))
    position_threshold_m = float(cfg["position_success_threshold_m"])
    orientation_threshold_deg = float(cfg["orientation_success_threshold_deg"])

    data = model_context.new_data()

    current_pose = forward_kinematics(model_context, q, data=data)
    e0, pos_err, orient_err_rad = _pose_error_components(
        target_position_arr, target_rotation_arr, current_pose.position, current_pose.rotation_matrix
    )
    orient_err_deg = float(np.degrees(orient_err_rad))
    position_weight = cfg.get("position_weight", 1.0)
    orientation_weight = cfg.get("orientation_weight", 1.0)
    combined_history = [
        float(np.linalg.norm(weighted_pose_error(e0, position_weight, orientation_weight)))
    ]
    history = list(combined_history) if record_history else None

    if pos_err <= position_threshold_m and orient_err_deg <= orientation_threshold_deg:
        J0 = geometric_jacobian_world(model_context, q, data=data)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return DLSResult(
            q_solution=q.copy(),
            success=True,
            iterations=0,
            position_error_m=pos_err,
            orientation_error_rad=orient_err_rad,
            orientation_error_deg=orient_err_deg,
            solve_time_ms=elapsed_ms,
            sigma_min=minimum_singular_value(J0),
            condition_number=safe_condition_number(J0),
            damping=0.0,
            joint_limit_violation=bool(np.any(operational_limit_violation_mask(q, lower, upper))),
            minimum_joint_limit_margin=minimum_joint_limit_margin(q, lower, upper),
            failure_reason=None,
            error_history=history,
        )

    success = False
    failure_reason = None
    last_sigma_min = 0.0
    last_condition_number = float("inf")
    last_damping = 0.0
    last_violation = bool(np.any(operational_limit_violation_mask(q, lower, upper)))
    iterations_completed = 0

    for iteration in range(1, max_iterations + 1):
        step = dls_single_update(model_context, q, target_position_arr, target_rotation_arr, cfg, data=data)

        last_sigma_min = step.sigma_min
        last_condition_number = step.condition_number
        last_damping = step.damping
        last_violation = step.joint_limit_violation_after_clip

        if step.failure_reason is not None:
            failure_reason = step.failure_reason
            break

        q = step.q_next
        iterations_completed = iteration

        current_pose = forward_kinematics(model_context, q, data=data)
        e, pos_err, orient_err_rad = _pose_error_components(
            target_position_arr, target_rotation_arr, current_pose.position, current_pose.rotation_matrix
        )
        orient_err_deg = float(np.degrees(orient_err_rad))

        combined = float(
            np.linalg.norm(weighted_pose_error(e, position_weight, orientation_weight))
        )
        combined_history.append(combined)
        if record_history:
            history.append(combined)

        if pos_err <= position_threshold_m and orient_err_deg <= orientation_threshold_deg:
            success = True
            break

        if len(combined_history) > _STAGNATION_WINDOW:
            old = combined_history[-_STAGNATION_WINDOW - 1]
            new = combined_history[-1]
            improvement = old - new
            if improvement < _STAGNATION_MIN_RELATIVE_IMPROVEMENT * max(old, 1e-9):
                failure_reason = "stagnation"
                break
    else:
        if failure_reason is None:
            failure_reason = "max_iterations"

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return DLSResult(
        q_solution=q.copy(),
        success=success,
        iterations=iterations_completed,
        position_error_m=pos_err,
        orientation_error_rad=orient_err_rad,
        orientation_error_deg=orient_err_deg,
        solve_time_ms=elapsed_ms,
        sigma_min=last_sigma_min,
        condition_number=last_condition_number,
        damping=last_damping,
        joint_limit_violation=last_violation,
        minimum_joint_limit_margin=minimum_joint_limit_margin(q, lower, upper),
        failure_reason=None if success else failure_reason,
        error_history=history,
    )
