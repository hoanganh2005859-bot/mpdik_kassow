"""DLS variant that initializes every solve from the trial's fixed initial joint configuration.

Every waypoint starts from ``q_trial_initial`` -- never from the previous waypoint's solution.
Cold-start exists purely as a baseline to quantify the benefit of sequential warm-starting
(see algorithms.warm_start_dls); it carries no cross-waypoint continuity, so there is no
recovery-policy question here.
"""

from typing import List

import numpy as np
from tqdm import tqdm

from algorithms.warm_start_dls import RawWaypointSolve
from kinematics.dls_solver import solve_dls_until_converged
from kinematics.model_loader import ModelContext
from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from utils.exceptions import DLSSolverError


def run_cold_start_dls(
    model_context: ModelContext,
    q_trial_initial: np.ndarray,
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
    dls_config: dict,
    fail_fast: bool = False,
    show_progress: bool = True,
) -> List[RawWaypointSolve]:
    """Independently solve DLS for each waypoint, always starting from ``q_trial_initial``.

    Args:
        q_trial_initial: Fixed starting joint configuration reused for every waypoint, shape (7,).
        target_positions: Shape (N, 3), meters.
        target_quaternions: Shape (N, 4), wxyz.
        dls_config: DLS solver configuration.
        fail_fast: If a solver step ever produces a non-finite joint state, raise
            DLSSolverError immediately instead of stopping the sequence gracefully.
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        A list of RawWaypointSolve, one per waypoint actually processed (``recovered_after_
        previous_failure`` is always False: cold-start has no cross-waypoint continuity to
        recover from).
    """
    q_trial_initial = np.asarray(q_trial_initial, dtype=np.float64)
    target_positions = np.asarray(target_positions, dtype=np.float64)
    target_quaternions = np.asarray(target_quaternions, dtype=np.float64)
    n = target_positions.shape[0]

    results: List[RawWaypointSolve] = []
    for k in tqdm(range(n), desc="cold_start_dls", disable=not show_progress):
        R_target = quaternion_wxyz_to_matrix(target_quaternions[k])
        dls_result = solve_dls_until_converged(
            model_context, q_trial_initial, target_positions[k], R_target, config=dls_config
        )

        if not np.all(np.isfinite(dls_result.q_solution)):
            if fail_fast:
                raise DLSSolverError(f"non-finite joint state at waypoint {k}")
            break

        results.append(RawWaypointSolve(k, q_trial_initial.copy(), dls_result, False))

    return results
