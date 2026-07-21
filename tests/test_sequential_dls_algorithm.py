"""Tests for algorithms.sequential_dls (warm-start/cold-start runners); implemented in the Tier 2 stage.

Uses one real trajectory (line_fixed_orientation) with a small waypoint subset so these tests
stay fast; never runs the full 360-trial trajectory_trials.csv set here.
"""

import numpy as np
import pytest

from algorithms.result_types import WaypointResult
from algorithms.sequential_dls import run_sequential_trial, waypoint_results_to_dataframe
from algorithms.warm_start_dls import RawWaypointSolve, _recover_q_initial
from kinematics.dls_solver import DLSResult

TRAJECTORY_ID = "line_fixed_orientation"
TRIAL_ID = "line_fixed_orientation_repeatability_speed1.0_r0"
WAYPOINT_LIMIT = 15


def test_warm_start_runs_the_full_requested_waypoint_subset():
    results = run_sequential_trial(TRAJECTORY_ID, TRIAL_ID, "warm_start", waypoint_limit=WAYPOINT_LIMIT, show_progress=False)
    assert len(results) == WAYPOINT_LIMIT
    assert [r.waypoint_id for r in results] == list(range(WAYPOINT_LIMIT))
    assert all(r.method == "warm_start" for r in results)
    assert all(r.trial_id == TRIAL_ID for r in results)
    assert all(r.trajectory_id == TRAJECTORY_ID for r in results)


def test_cold_start_runs_the_full_requested_waypoint_subset():
    results = run_sequential_trial(TRAJECTORY_ID, TRIAL_ID, "cold_start", waypoint_limit=WAYPOINT_LIMIT, show_progress=False)
    assert len(results) == WAYPOINT_LIMIT
    assert all(r.method == "cold_start" for r in results)


def test_cold_start_always_reuses_the_trial_initial_configuration():
    results = run_sequential_trial(TRAJECTORY_ID, TRIAL_ID, "cold_start", waypoint_limit=WAYPOINT_LIMIT, show_progress=False)
    q_initial_used = np.array([r.q_initial_used for r in results])
    assert np.allclose(q_initial_used, q_initial_used[0])  # identical every waypoint
    assert all(r.recovered_after_previous_failure is False for r in results)


def test_warm_start_q_initial_used_follows_previous_solution_when_successful():
    results = run_sequential_trial(TRAJECTORY_ID, TRIAL_ID, "warm_start", waypoint_limit=WAYPOINT_LIMIT, show_progress=False)
    for k in range(1, len(results)):
        if results[k - 1].success:
            assert np.allclose(results[k].q_initial_used, results[k - 1].q_solution)


def test_speed_scale_only_changes_timing_not_target_geometry():
    slow = run_sequential_trial(
        TRAJECTORY_ID, "line_fixed_orientation_repeatability_speed0.5_r0", "warm_start",
        waypoint_limit=WAYPOINT_LIMIT, show_progress=False,
    )
    fast = run_sequential_trial(
        TRAJECTORY_ID, "line_fixed_orientation_repeatability_speed1.5_r0", "warm_start",
        waypoint_limit=WAYPOINT_LIMIT, show_progress=False,
    )
    slow_targets = np.array([r.target_position for r in slow])
    fast_targets = np.array([r.target_position for r in fast])
    assert np.allclose(slow_targets, fast_targets)  # same path geometry regardless of speed_scale

    slow_times = np.array([r.time_s for r in slow])
    fast_times = np.array([r.time_s for r in fast])
    # speed_scale=1.5 covers the same waypoints in less time than speed_scale=0.5
    assert fast_times[-1] < slow_times[-1]
    assert np.allclose(fast_times, slow_times * (0.5 / 1.5))


def test_output_schema_and_dataframe_conversion():
    results = run_sequential_trial(TRAJECTORY_ID, TRIAL_ID, "warm_start", waypoint_limit=5, show_progress=False)
    assert all(isinstance(r, WaypointResult) for r in results)
    df = waypoint_results_to_dataframe(results)
    assert len(df) == 5
    for col in [
        "trial_id", "trajectory_id", "trial_category", "method", "waypoint_id", "time_s",
        "q_solution_q1", "target_position_x", "actual_quaternion_qw", "success", "failure_reason",
    ]:
        assert col in df.columns


def test_unknown_trajectory_id_raises_key_error():
    with pytest.raises(KeyError):
        run_sequential_trial("does_not_exist", TRIAL_ID, "warm_start", waypoint_limit=2, show_progress=False)


def test_mismatched_trial_and_trajectory_raises_value_error():
    with pytest.raises(ValueError):
        run_sequential_trial(TRAJECTORY_ID, "circle_fixed_orientation_repeatability_speed1.0_r0", "warm_start")


# --- recovery policy determinism (white-box on algorithms.warm_start_dls._recover_q_initial) ---


def _dls_result(q_solution, success):
    return DLSResult(
        q_solution=np.asarray(q_solution, dtype=np.float64),
        success=success,
        iterations=1,
        position_error_m=0.0,
        orientation_error_rad=0.0,
        orientation_error_deg=0.0,
        solve_time_ms=0.1,
        sigma_min=1.0,
        condition_number=1.0,
        damping=0.01,
        joint_limit_violation=False,
        minimum_joint_limit_margin=0.5,
        failure_reason=None if success else "max_iterations",
    )


def test_recovery_prefers_failed_solves_own_finite_q_solution():
    trial_q_initial = np.zeros(7)
    last_successful_q = np.ones(7)
    failed = _dls_result(np.full(7, 0.3), success=False)
    recovered = _recover_q_initial(failed, last_successful_q, trial_q_initial)
    assert np.allclose(recovered, failed.q_solution)


def test_recovery_falls_back_to_last_successful_q_when_solver_state_is_non_finite():
    trial_q_initial = np.zeros(7)
    last_successful_q = np.full(7, 0.7)
    failed = _dls_result(np.full(7, np.nan), success=False)
    recovered = _recover_q_initial(failed, last_successful_q, trial_q_initial)
    assert np.allclose(recovered, last_successful_q)


def test_recovery_falls_back_to_trial_initial_when_no_successful_waypoint_yet():
    trial_q_initial = np.full(7, 0.2)
    failed = _dls_result(np.full(7, np.inf), success=False)
    recovered = _recover_q_initial(failed, None, trial_q_initial)
    assert np.allclose(recovered, trial_q_initial)
