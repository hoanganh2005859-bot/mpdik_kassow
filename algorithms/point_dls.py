"""Tier 1: single-target DLS IK solve wrapper around kinematics.dls_solver.

Runs each sample of the Tier 1 point-IK benchmark (benchmarks/point_ik/point_ik_v1.npz)
independently: ``q_initial`` (from the benchmark) is the solver's starting configuration,
``target_position``/``target_quaternion`` (the forward-kinematics image of the benchmark's
``q_target``) is the goal pose. ``q_target`` itself is never used as an initial guess -- it is
only carried through as ``q_target_reference`` on the result, for optional joint-space
comparison against the achieved solution.

Every sample produces a PointIKResult, including failed samples (with ``failure_reason`` set);
none are silently dropped, and the original benchmark ordering is preserved unless the caller
explicitly asks for a subset by ``sample_ids``.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from tqdm import tqdm

from algorithms.result_types import PointIKResult, point_ik_results_to_dataframe
from kinematics.dls_solver import load_dls_config, solve_dls_until_converged
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import (
    minimum_joint_limit_margin,
    operational_limit_violation_mask,
)
from kinematics.model_loader import ModelContext, load_model_context
from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from kinematics.singularity_metrics import condition_number, minimum_singular_value
from utils.npz_utils import load_npz

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "point_ik" / "point_ik_v1.npz"

logger = logging.getLogger(__name__)

__all__ = [
    "point_ik_results_to_dataframe",
    "load_point_ik_benchmark",
    "run_point_dls",
]


def load_point_ik_benchmark(path: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
    """Load the Tier 1 point-IK benchmark NPZ (allow_pickle=False) as a plain dict of arrays."""
    return load_npz(path or DEFAULT_BENCHMARK_PATH)


def _select_indices(
    sample_id_array: np.ndarray,
    sample_ids: Optional[Sequence[int]],
    sample_limit: Optional[int],
) -> np.ndarray:
    if sample_ids is not None:
        wanted = set(int(s) for s in sample_ids)
        mask = np.isin(sample_id_array, list(wanted))
        indices = np.flatnonzero(mask)
        found = set(int(s) for s in sample_id_array[indices])
        missing = wanted - found
        if missing:
            raise ValueError(f"sample_ids not found in benchmark: {sorted(missing)}")
    else:
        indices = np.arange(sample_id_array.shape[0])

    if sample_limit is not None:
        indices = indices[: int(sample_limit)]
    return indices


def _solve_one_sample(
    model_context: ModelContext,
    data,
    sample_id: int,
    difficulty_id: int,
    q_initial: np.ndarray,
    q_target: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    dls_config: dict,
) -> PointIKResult:
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad

    J_initial = geometric_jacobian_world(model_context, q_initial, data=data)
    initial_sigma_min = minimum_singular_value(J_initial)
    initial_condition_number = condition_number(J_initial)

    target_rotation = quaternion_wxyz_to_matrix(target_quaternion)
    dls_result = solve_dls_until_converged(
        model_context, q_initial, target_position, target_rotation, config=dls_config
    )

    J_final = geometric_jacobian_world(model_context, dls_result.q_solution, data=data)
    final_sigma_min = minimum_singular_value(J_final)
    final_condition_number = condition_number(J_final)

    joint_limit_violation = bool(
        np.any(operational_limit_violation_mask(dls_result.q_solution, lower, upper))
    )
    margin = minimum_joint_limit_margin(dls_result.q_solution, lower, upper)

    return PointIKResult(
        sample_id=int(sample_id),
        difficulty_id=int(difficulty_id),
        success=bool(dls_result.success),
        q_initial=np.asarray(q_initial, dtype=np.float64).copy(),
        q_target_reference=np.asarray(q_target, dtype=np.float64).copy(),
        q_solution=dls_result.q_solution.copy(),
        position_error_m=float(dls_result.position_error_m),
        orientation_error_rad=float(dls_result.orientation_error_rad),
        orientation_error_deg=float(dls_result.orientation_error_deg),
        iterations=int(dls_result.iterations),
        solve_time_ms=float(dls_result.solve_time_ms),
        initial_sigma_min=float(initial_sigma_min),
        final_sigma_min=float(final_sigma_min),
        initial_condition_number=float(initial_condition_number),
        final_condition_number=float(final_condition_number),
        minimum_joint_limit_margin=float(margin),
        joint_limit_violation=joint_limit_violation,
        failure_reason=dls_result.failure_reason,
    )


def run_point_dls(
    benchmark: Optional[Dict[str, np.ndarray]] = None,
    model_context: Optional[ModelContext] = None,
    dls_config: Optional[dict] = None,
    sample_ids: Optional[Sequence[int]] = None,
    sample_limit: Optional[int] = None,
    show_progress: bool = True,
) -> List[PointIKResult]:
    """Solve DLS independently for each selected sample of the Tier 1 point-IK benchmark.

    Args:
        benchmark: Pre-loaded benchmark dict (see load_point_ik_benchmark). Loaded from the
            default path if omitted.
        model_context: Pre-loaded ModelContext. Loaded via load_model_context() if omitted.
        dls_config: DLS solver configuration. Loaded via load_dls_config() if omitted.
        sample_ids: If given, restrict to these sample IDs (original benchmark order preserved).
        sample_limit: If given, keep only the first N selected samples (for smoke tests).
        show_progress: Whether to show a tqdm progress bar (never per-sample print statements).

    Returns:
        A list of PointIKResult, one per selected sample, in benchmark order. Failed samples
        are included (never skipped), with ``failure_reason`` set.
    """
    benchmark = benchmark if benchmark is not None else load_point_ik_benchmark()
    model_context = model_context if model_context is not None else load_model_context()
    dls_config = dls_config if dls_config is not None else load_dls_config()

    indices = _select_indices(benchmark["sample_id"], sample_ids, sample_limit)

    data = model_context.new_data()
    results: List[PointIKResult] = []
    for idx in tqdm(indices, desc="point_dls", disable=not show_progress):
        idx = int(idx)
        result = _solve_one_sample(
            model_context,
            data,
            sample_id=benchmark["sample_id"][idx],
            difficulty_id=benchmark["difficulty_id"][idx],
            q_initial=benchmark["q_initial"][idx],
            q_target=benchmark["q_target"][idx],
            target_position=benchmark["target_position"][idx],
            target_quaternion=benchmark["target_quaternion"][idx],
            dls_config=dls_config,
        )
        results.append(result)

    n_failed = sum(1 for r in results if not r.success)
    logger.info("point_dls: %d/%d samples solved, %d failed", len(results) - n_failed, len(results), n_failed)

    return results
