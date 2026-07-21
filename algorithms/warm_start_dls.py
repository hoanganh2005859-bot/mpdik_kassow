"""DLS variant that initializes each solve from the previous waypoint's converged joint configuration.

Recovery policy when waypoint k-1 failed (deterministic, in priority order):

1. Use the failed solve's own ``q_solution``, if it is finite. ``solve_dls_until_converged``
   never advances its internal ``q`` past a non-finite step (see kinematics.dls_solver:
   ``dls_single_update`` returns ``q_next=q.copy()`` unchanged on every anticipated failure
   mode), so in practice this is always finite and simply reuses the best joint state the
   solver reached before it gave up.
2. Otherwise, fall back to the last waypoint that *did* converge successfully.
3. Otherwise (no successful waypoint yet in this trial), fall back to the trial's own
   ``q_initial``.

Waypoint failures never stop the sequence early. The only condition that stops processing
remaining waypoints is the solver producing a non-finite joint state, which
``solve_dls_until_converged`` is designed never to do; the check here is defensive.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from kinematics.dls_solver import DLSResult, solve_dls_until_converged
from kinematics.model_loader import ModelContext
from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from utils.exceptions import DLSSolverError


@dataclass
class RawWaypointSolve:
    """One waypoint's solve outcome plus the bookkeeping needed to build a WaypointResult."""

    waypoint_index: int
    q_initial_used: np.ndarray
    dls_result: DLSResult
    recovered_after_previous_failure: bool


def _recover_q_initial(
    previous_dls_result: DLSResult,
    last_successful_q: Optional[np.ndarray],
    trial_q_initial: np.ndarray,
) -> np.ndarray:
    candidate = previous_dls_result.q_solution
    if candidate is not None and np.all(np.isfinite(candidate)):
        return candidate
    if last_successful_q is not None:
        return last_successful_q
    return trial_q_initial


def run_warm_start_dls(
    model_context: ModelContext,
    q_trial_initial: np.ndarray,
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
    dls_config: dict,
    fail_fast: bool = False,
    show_progress: bool = True,
) -> List[RawWaypointSolve]:
    """Sequentially warm-start DLS across an ordered waypoint chain.

    Args:
        q_trial_initial: Starting joint configuration for waypoint 0, shape (7,).
        target_positions: Shape (N, 3), meters.
        target_quaternions: Shape (N, 4), wxyz.
        dls_config: DLS solver configuration.
        fail_fast: If a solver step ever produces a non-finite joint state, raise
            DLSSolverError immediately instead of stopping the sequence gracefully. Ordinary
            (finite) waypoint failures always continue the sequence regardless of this flag.
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        A list of RawWaypointSolve, one per waypoint actually processed. Shorter than N only
        if a non-finite solver state was encountered (fail_fast=False) and processing stopped.
    """
    target_positions = np.asarray(target_positions, dtype=np.float64)
    target_quaternions = np.asarray(target_quaternions, dtype=np.float64)
    n = target_positions.shape[0]

    q_current = np.asarray(q_trial_initial, dtype=np.float64).copy()
    last_successful_q: Optional[np.ndarray] = None
    previous_failed = False

    results: List[RawWaypointSolve] = []
    for k in tqdm(range(n), desc="warm_start_dls", disable=not show_progress):
        q_initial_used = q_current
        R_target = quaternion_wxyz_to_matrix(target_quaternions[k])
        dls_result = solve_dls_until_converged(
            model_context, q_initial_used, target_positions[k], R_target, config=dls_config
        )

        if not np.all(np.isfinite(dls_result.q_solution)):
            if fail_fast:
                raise DLSSolverError(f"non-finite joint state at waypoint {k}")
            break

        recovered = bool(previous_failed and dls_result.success)
        results.append(RawWaypointSolve(k, q_initial_used, dls_result, recovered))

        if dls_result.success:
            last_successful_q = dls_result.q_solution
            q_current = dls_result.q_solution
            previous_failed = False
        else:
            previous_failed = True
            q_current = _recover_q_initial(dls_result, last_successful_q, q_trial_initial)

    return results
