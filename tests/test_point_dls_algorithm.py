"""Tests for algorithms.point_dls; implemented in the Tier 1 point-DLS stage."""

import numpy as np
import pytest

from algorithms.point_dls import load_point_ik_benchmark, point_ik_results_to_dataframe, run_point_dls
from kinematics.dls_solver import load_dls_config
from kinematics.model_loader import load_model_context

CONTEXT = load_model_context()
DLS_CONFIG = load_dls_config()
BENCHMARK = load_point_ik_benchmark()


def test_sample_limit_returns_requested_count_in_original_order():
    results = run_point_dls(BENCHMARK, CONTEXT, DLS_CONFIG, sample_limit=6, show_progress=False)
    assert len(results) == 6
    assert [r.sample_id for r in results] == list(BENCHMARK["sample_id"][:6])


def test_sample_ids_subset_preserves_benchmark_order():
    ids = [500, 10, 250]  # deliberately out of order
    results = run_point_dls(BENCHMARK, CONTEXT, DLS_CONFIG, sample_ids=ids, show_progress=False)
    assert sorted(r.sample_id for r in results) == sorted(ids)
    assert [r.sample_id for r in results] == sorted(ids)  # benchmark's own sample_id order is ascending


def test_unknown_sample_id_raises_value_error():
    with pytest.raises(ValueError):
        run_point_dls(BENCHMARK, CONTEXT, DLS_CONFIG, sample_ids=[999999], show_progress=False)


def test_near_target_sample_converges():
    near_target_idx = int(np.flatnonzero(BENCHMARK["difficulty_id"] == 0)[0])
    sample_id = int(BENCHMARK["sample_id"][near_target_idx])
    results = run_point_dls(BENCHMARK, CONTEXT, DLS_CONFIG, sample_ids=[sample_id], show_progress=False)
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].failure_reason is None


def test_failures_are_never_dropped_for_hard_samples():
    near_singular_ids = BENCHMARK["sample_id"][BENCHMARK["difficulty_id"] == 5][:8].tolist()
    results = run_point_dls(BENCHMARK, CONTEXT, DLS_CONFIG, sample_ids=near_singular_ids, show_progress=False)
    assert len(results) == len(near_singular_ids)
    for r in results:
        assert r.failure_reason is None or isinstance(r.failure_reason, str)
        assert np.all(np.isfinite(r.q_solution))


def test_q_target_is_not_used_as_initial_guess():
    sample_id = int(BENCHMARK["sample_id"][0])
    result = run_point_dls(BENCHMARK, CONTEXT, DLS_CONFIG, sample_ids=[sample_id], show_progress=False)[0]
    assert np.allclose(result.q_initial, BENCHMARK["q_initial"][0])
    assert np.allclose(result.q_target_reference, BENCHMARK["q_target"][0])


def test_results_to_dataframe_has_stable_schema():
    results = run_point_dls(BENCHMARK, CONTEXT, DLS_CONFIG, sample_limit=3, show_progress=False)
    df = point_ik_results_to_dataframe(results)
    assert len(df) == 3
    for col in ["sample_id", "difficulty_id", "success", "q_initial_q1", "q_solution_q7", "position_error_m", "failure_reason"]:
        assert col in df.columns
