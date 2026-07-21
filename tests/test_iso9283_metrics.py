"""Tests for evaluation.iso9283_metrics against known reference statistics; implemented in the Tier 3 evaluation stage."""

import numpy as np
import pytest

from algorithms.result_types import WaypointResult
from evaluation.iso9283_metrics import (
    build_repeatability_group,
    compute_iso9283_metrics,
    compute_path_accuracy,
    compute_path_repeatability,
)

_Q_ZERO = np.zeros(7)
_QUAT_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _waypoint_result(waypoint_id, target_position, actual_position, trial_category="repeatability"):
    return WaypointResult(
        trial_id="trial",
        trajectory_id="traj",
        trial_category=trial_category,
        method="warm_start",
        waypoint_id=waypoint_id,
        time_s=float(waypoint_id) * 0.1,
        q_initial_used=_Q_ZERO,
        q_solution=_Q_ZERO,
        target_position=np.asarray(target_position, dtype=np.float64),
        actual_position=np.asarray(actual_position, dtype=np.float64),
        target_quaternion=_QUAT_IDENTITY,
        actual_quaternion=_QUAT_IDENTITY,
        position_error_m=float(np.linalg.norm(np.asarray(target_position) - np.asarray(actual_position))),
        orientation_error_rad=0.0,
        orientation_error_deg=0.0,
        success=True,
        iterations=1,
        solve_time_ms=0.1,
        sigma_min=1.0,
        condition_number=1.0,
        manipulability=1.0,
        minimum_joint_limit_margin=1.0,
        recovered_after_previous_failure=False,
        failure_reason=None,
    )


def _commanded_path(n=5):
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([t, np.zeros(n), np.zeros(n)])


# --- 1. Perfect repeated paths --------------------------------------------------------------


def test_perfect_repeated_paths_give_zero_atp_and_rtp():
    commanded = _commanded_path()
    repeated = np.tile(commanded, (5, 1, 1))  # 5 identical, exact repeats

    accuracy = compute_path_accuracy(commanded, repeated)
    repeatability = compute_path_repeatability(repeated)

    assert accuracy.atp_m == pytest.approx(0.0, abs=1e-12)
    assert repeatability.rtp_m == pytest.approx(0.0, abs=1e-12)


# --- 2. Constant bias, identical repeats -----------------------------------------------------


def test_constant_bias_identical_repeats_gives_known_atp_and_zero_rtp():
    commanded = _commanded_path()
    bias = np.array([0.003, -0.001, 0.0])
    biased = commanded + bias
    repeated = np.tile(biased, (6, 1, 1))  # identical across repeats, same bias every time

    accuracy = compute_path_accuracy(commanded, repeated)
    repeatability = compute_path_repeatability(repeated)

    assert accuracy.atp_m == pytest.approx(float(np.linalg.norm(bias)), rel=1e-9)
    assert repeatability.rtp_m == pytest.approx(0.0, abs=1e-12)


# --- 3. Zero-mean repeated spread -------------------------------------------------------------


def test_zero_mean_spread_gives_small_atp_and_positive_rtp():
    commanded = _commanded_path()
    half = 10
    rng = np.random.default_rng(7)
    # Symmetric +/-delta pairs at each waypoint so the mean attained path is exactly commanded.
    delta = rng.normal(scale=0.001, size=(half, commanded.shape[0], 3))
    positive = commanded[None, :, :] + delta
    negative = commanded[None, :, :] - delta
    repeated = np.concatenate([positive, negative], axis=0)  # (2*half, M, 3)

    accuracy = compute_path_accuracy(commanded, repeated)
    repeatability = compute_path_repeatability(repeated)

    assert accuracy.atp_m == pytest.approx(0.0, abs=1e-9)
    assert repeatability.rtp_m > 0.0


# --- 4. Known radial deviations: verify r_bar + 3*s (ddof=1) formula ---------------------------


def test_known_radial_deviation_matches_manual_formula():
    commanded = _commanded_path(3)
    # Construct 4 repeats with hand-computable radial deviations at each waypoint: 0, 1, 2, 3 (mm-scale)
    offsets_m = np.array([0.000, 0.001, 0.002, 0.003])
    repeated = np.stack([commanded + np.array([o, 0.0, 0.0]) for o in offsets_m], axis=0)

    repeatability = compute_path_repeatability(repeated)

    mean_attained = np.mean(repeated, axis=0)
    radial = np.linalg.norm(repeated - mean_attained[None, :, :], axis=2)
    r_bar = np.mean(radial, axis=0)
    s_r = np.std(radial, axis=0, ddof=1)
    expected_rtp = float(np.max(r_bar + 3.0 * s_r))

    assert repeatability.rtp_m == pytest.approx(expected_rtp, rel=1e-9)
    assert repeatability.n_repeats == 4


# --- 5. Mismatched waypoint count --------------------------------------------------------------


def test_mismatched_waypoint_count_raises_value_error():
    group = {
        0: [_waypoint_result(0, [0, 0, 0], [0, 0, 0]), _waypoint_result(1, [1, 0, 0], [1, 0, 0])],
        1: [_waypoint_result(0, [0, 0, 0], [0, 0, 0])],
    }
    with pytest.raises(ValueError):
        build_repeatability_group(group)


# --- 6. Mismatched target path -----------------------------------------------------------------


def test_mismatched_target_path_raises_value_error():
    group = {
        0: [_waypoint_result(0, [0, 0, 0], [0, 0, 0])],
        1: [_waypoint_result(0, [0, 0, 1], [0, 0, 1])],  # different commanded position
    }
    with pytest.raises(ValueError):
        build_repeatability_group(group)


# --- 7. Robustness trials rejected ---------------------------------------------------------------


def test_robustness_trials_rejected_from_repeatability_group():
    group = {
        0: [_waypoint_result(0, [0, 0, 0], [0, 0, 0], trial_category="repeatability")],
        1: [_waypoint_result(0, [0, 0, 0], [0, 0, 0], trial_category="robustness")],
    }
    with pytest.raises(ValueError):
        build_repeatability_group(group)


# --- 8. Under 2 repeats -----------------------------------------------------------------------


def test_single_repeat_raises_value_error():
    group = {0: [_waypoint_result(0, [0, 0, 0], [0, 0, 0])]}
    with pytest.raises(ValueError):
        build_repeatability_group(group)

    with pytest.raises(ValueError):
        compute_path_repeatability(np.zeros((1, 3, 3)))


# --- 9. Under 10 repeats: computed, but flagged with a warning ---------------------------------


def test_under_ten_repeats_still_computes_but_warns():
    commanded = _commanded_path()
    repeated = np.tile(commanded, (4, 1, 1))
    repeatability = compute_path_repeatability(repeated)
    assert repeatability.n_repeats == 4
    assert repeatability.warning is not None
    assert "4" in repeatability.warning


def test_ten_or_more_repeats_has_no_warning():
    commanded = _commanded_path()
    repeated = np.tile(commanded, (10, 1, 1))
    repeatability = compute_path_repeatability(repeated)
    assert repeatability.warning is None


# --- end-to-end: build_repeatability_group + compute_iso9283_metrics ---------------------------


def test_compute_iso9283_metrics_end_to_end_from_waypoint_results():
    commanded = _commanded_path()
    group = {}
    for repeat_id in range(3):
        group[repeat_id] = [
            _waypoint_result(i, commanded[i], commanded[i]) for i in range(commanded.shape[0])
        ]
    result = compute_iso9283_metrics(group)
    assert result.accuracy.atp_m == pytest.approx(0.0, abs=1e-12)
    assert result.repeatability.rtp_m == pytest.approx(0.0, abs=1e-12)
