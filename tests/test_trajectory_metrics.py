"""Tests for evaluation.trajectory_metrics and cross_track_metrics computations; implemented in the Tier 3 evaluation stage."""

import numpy as np
import pytest

from evaluation.cross_track_metrics import compute_cross_track_metrics, project_point_to_polyline
from evaluation.orientation_metrics import geodesic_angles_from_quaternions, summarize_orientation_errors
from evaluation.trajectory_metrics import (
    compute_position_tracking_metrics,
    compute_trajectory_tracking_metrics,
)
from kinematics.quaternion_utils import rotation_matrix_to_quaternion_wxyz
from kinematics.rotation_utils import so3_exp


def _line_positions(n=10):
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([t, np.zeros(n), np.zeros(n)])


# --- 1. Perfect tracking -----------------------------------------------------------------


def test_perfect_tracking_gives_zero_error_and_unit_path_length_ratio():
    target = _line_positions()
    actual = target.copy()
    metrics = compute_position_tracking_metrics(target, actual)
    assert metrics.rmse_m == pytest.approx(0.0, abs=1e-12)
    assert metrics.mae_m == pytest.approx(0.0, abs=1e-12)
    assert metrics.p95_m == pytest.approx(0.0, abs=1e-12)
    assert metrics.max_m == pytest.approx(0.0, abs=1e-12)
    assert metrics.path_length_ratio == pytest.approx(1.0)
    assert metrics.path_length_abs_diff_m == pytest.approx(0.0, abs=1e-12)


# --- 2. Constant Cartesian offset --------------------------------------------------------


def test_constant_offset_gives_known_error_and_centroid_offset():
    target = _line_positions()
    offset = np.array([0.01, -0.02, 0.005])
    actual = target - offset
    metrics = compute_position_tracking_metrics(target, actual)
    expected_norm = float(np.linalg.norm(offset))

    assert metrics.rmse_m == pytest.approx(expected_norm, rel=1e-9)
    assert metrics.mae_m == pytest.approx(expected_norm, rel=1e-9)
    assert metrics.max_m == pytest.approx(expected_norm, rel=1e-9)
    assert metrics.centroid_offset_m == pytest.approx(expected_norm, rel=1e-9)
    # a rigid translation of the whole path does not change its length
    assert metrics.path_length_ratio == pytest.approx(1.0, rel=1e-9)


# --- 3. One large outlier -----------------------------------------------------------------


def test_single_outlier_dominates_max_but_not_mean():
    n = 20
    target = _line_positions(n)
    rng = np.random.default_rng(0)
    actual = target + rng.normal(scale=1e-4, size=target.shape)
    actual[10] += np.array([0.05, 0.0, 0.0])

    metrics = compute_position_tracking_metrics(target, actual)
    errors = np.linalg.norm(target - actual, axis=1)

    assert metrics.max_m == pytest.approx(float(np.max(errors)))
    assert metrics.max_m > metrics.mae_m
    assert metrics.rmse_m > metrics.mae_m
    assert metrics.p95_m <= metrics.max_m + 1e-12


# --- 4. Orientation ------------------------------------------------------------------------


def test_orientation_identity_gives_zero_error():
    q_identity = rotation_matrix_to_quaternion_wxyz(np.eye(3))
    target_q = np.tile(q_identity, (5, 1))
    actual_q = target_q.copy()
    summary = summarize_orientation_errors(geodesic_angles_from_quaternions(target_q, actual_q))
    assert summary.rmse_deg == pytest.approx(0.0, abs=1e-8)
    assert summary.max_deg == pytest.approx(0.0, abs=1e-8)


def test_orientation_constant_known_rotation_matches_analytic_angle():
    angle_rad = 0.3
    R_offset = so3_exp(np.array([0.0, 0.0, angle_rad]))
    q_offset = rotation_matrix_to_quaternion_wxyz(R_offset)
    q_identity = rotation_matrix_to_quaternion_wxyz(np.eye(3))

    target_q = np.tile(q_offset, (4, 1))
    actual_q = np.tile(q_identity, (4, 1))
    summary = summarize_orientation_errors(geodesic_angles_from_quaternions(target_q, actual_q))

    assert summary.rmse_deg == pytest.approx(np.degrees(angle_rad), rel=1e-6)
    assert summary.max_deg == pytest.approx(np.degrees(angle_rad), rel=1e-6)


def test_orientation_quaternion_sign_ambiguity_is_equivalent():
    rng = np.random.default_rng(3)
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    target_q = np.array([q, q])
    actual_q = np.array([q, -q])
    angles = geodesic_angles_from_quaternions(target_q, actual_q)
    assert np.allclose(angles, 0.0, atol=1e-8)


def test_compute_trajectory_tracking_metrics_combines_position_and_orientation():
    target = _line_positions(6)
    actual = target.copy()
    q_identity = rotation_matrix_to_quaternion_wxyz(np.eye(3))
    target_q = np.tile(q_identity, (6, 1))
    actual_q = target_q.copy()

    result = compute_trajectory_tracking_metrics(target, actual, target_q, actual_q)
    assert result.position.rmse_m == pytest.approx(0.0, abs=1e-12)
    assert result.orientation.rmse_deg == pytest.approx(0.0, abs=1e-8)


# --- 5. Cross-track on a straight line -----------------------------------------------------


def test_cross_track_straight_line_known_perpendicular_offset():
    reference = _line_positions(11)
    perpendicular_offset = 0.02
    actual = reference.copy()
    actual[:, 1] += perpendicular_offset

    result = compute_cross_track_metrics(actual, reference, closed_path=False)
    assert result.summary.cross_track_rmse_m == pytest.approx(perpendicular_offset, rel=1e-6)
    assert result.summary.cross_track_max_m == pytest.approx(perpendicular_offset, rel=1e-6)


# --- 6. Projection on a segment: interior and clamped endpoints ----------------------------


def test_projection_within_segment_and_clamped_at_endpoints():
    segment_starts = np.array([[0.0, 0.0, 0.0]])
    segment_vectors = np.array([[1.0, 0.0, 0.0]])
    segment_lengths = np.array([1.0])
    cumulative_start = np.array([0.0])

    idx, t, dist = project_point_to_polyline(
        np.array([0.5, 0.3, 0.0]), segment_starts, segment_vectors, segment_lengths, cumulative_start
    )
    assert idx == 0
    assert t == pytest.approx(0.5)
    assert dist == pytest.approx(0.3)

    idx, t, dist = project_point_to_polyline(
        np.array([1.5, 0.0, 0.0]), segment_starts, segment_vectors, segment_lengths, cumulative_start
    )
    assert t == pytest.approx(1.0)
    assert dist == pytest.approx(0.5)

    idx, t, dist = project_point_to_polyline(
        np.array([-0.5, 0.0, 0.0]), segment_starts, segment_vectors, segment_lengths, cumulative_start
    )
    assert t == pytest.approx(0.0)
    assert dist == pytest.approx(0.5)


# --- 7. Closed path: final segment wraps back to the first point ---------------------------


def test_closed_path_wraps_final_segment_to_first_point():
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    seam_point = np.array([[0.0, 0.5, 0.1]])  # near the segment connecting point3 -> point0

    closed_result = compute_cross_track_metrics(seam_point, reference, closed_path=True)
    assert closed_result.per_point[0].nearest_segment_index == 3
    assert closed_result.per_point[0].cross_track_distance_m == pytest.approx(0.1, abs=1e-9)

    open_result = compute_cross_track_metrics(seam_point, reference, closed_path=False)
    assert open_result.summary.cross_track_max_m >= closed_result.summary.cross_track_max_m


# --- 8. Along-track progress ----------------------------------------------------------------


def test_along_track_progress_is_monotonic_for_forward_motion():
    reference = _line_positions(11)
    result = compute_cross_track_metrics(reference.copy(), reference, closed_path=False)
    along = np.array([p.along_path_coordinate_m for p in result.per_point])
    assert np.all(np.diff(along) >= -1e-9)
    assert result.summary.backward_progress_count == 0
    assert result.summary.final_progress_ratio == pytest.approx(1.0)


def test_along_track_backward_progress_is_detected():
    reference = _line_positions(11)
    backward_actual = reference[::-1].copy()
    result = compute_cross_track_metrics(backward_actual, reference, closed_path=False)
    assert result.summary.backward_progress_count > 0


# --- 9. Invalid shape/time --------------------------------------------------------------------


def test_mismatched_position_shapes_raise_value_error():
    target = _line_positions(5)
    actual = _line_positions(6)
    with pytest.raises(ValueError):
        compute_position_tracking_metrics(target, actual)


def test_degenerate_reference_path_raises_value_error():
    with pytest.raises(ValueError):
        compute_cross_track_metrics(_line_positions(5), np.zeros((1, 3)), closed_path=False)


def test_wrong_dimensional_position_array_raises_value_error():
    with pytest.raises(ValueError):
        compute_position_tracking_metrics(np.zeros((5, 2)), np.zeros((5, 2)))
