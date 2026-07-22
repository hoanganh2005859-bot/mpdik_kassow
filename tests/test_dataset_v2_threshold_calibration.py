"""Tests for Phase 2.5 Dataset v2 threshold calibration (near_joint_limit / near_singularity /
moderately_conditioned / regular).

Everything here runs against small, test-scale candidate pools (never the locked-default
50,000/20,000 pools) for speed; a full-scale run is exercised manually/in the phase report, not in
the regular suite. Nothing here writes a candidate pool to disk or touches Dataset v1.
"""

import json

import numpy as np
import pytest

from dataset_v2.config_templates import all_configs
from dataset_v2.threshold_calibration import (
    CLASSIFICATION_PRIORITY,
    build_generic_candidate_pool,
    build_singularity_biased_pool,
    calibrate,
    classify_single,
    compute_candidate_metrics,
)
from generators._common import get_model_context
from kinematics.joint_limit_utils import normalized_joint_limit_margin
from kinematics.singularity_metrics import minimum_singular_value
from kinematics.jacobian import geometric_jacobian_world

MODEL_CONTEXT = get_model_context()

SMALL_POOL = 400
SMALL_SINGULARITY_POOL = 400


def test_calibration_deterministic_same_seed():
    result_a = calibrate(42, generic_pool_size=SMALL_POOL, singularity_pool_size=SMALL_SINGULARITY_POOL, model_context=MODEL_CONTEXT)
    result_b = calibrate(42, generic_pool_size=SMALL_POOL, singularity_pool_size=SMALL_SINGULARITY_POOL, model_context=MODEL_CONTEXT)

    np.testing.assert_array_equal(result_a.generic_metrics.q_samples, result_b.generic_metrics.q_samples)
    np.testing.assert_array_equal(result_a.singularity_metrics.q_samples, result_b.singularity_metrics.q_samples)
    assert result_a.near_joint_limit_normalized_threshold == result_b.near_joint_limit_normalized_threshold
    assert result_a.generic_pool_seed == result_b.generic_pool_seed


def test_different_seed_produces_different_pool_and_thresholds():
    result_a = calibrate(42, generic_pool_size=SMALL_POOL, singularity_pool_size=SMALL_SINGULARITY_POOL, model_context=MODEL_CONTEXT)
    result_b = calibrate(43, generic_pool_size=SMALL_POOL, singularity_pool_size=SMALL_SINGULARITY_POOL, model_context=MODEL_CONTEXT)

    assert result_a.generic_pool_seed != result_b.generic_pool_seed
    assert not np.array_equal(result_a.generic_metrics.q_samples, result_b.generic_metrics.q_samples)
    assert result_a.near_joint_limit_normalized_threshold != result_b.near_joint_limit_normalized_threshold


def test_candidate_pools_are_duplicate_free():
    rng = np.random.default_rng(123)
    q = build_generic_candidate_pool(rng, MODEL_CONTEXT, SMALL_POOL)
    _, first_idx = np.unique(q, axis=0, return_index=True)
    assert first_idx.shape[0] == q.shape[0]

    rng2 = np.random.default_rng(456)
    q2 = build_singularity_biased_pool(rng2, MODEL_CONTEXT, SMALL_SINGULARITY_POOL)
    _, first_idx2 = np.unique(q2, axis=0, return_index=True)
    assert first_idx2.shape[0] == q2.shape[0]


def test_candidate_pools_stay_within_operational_limits():
    rng = np.random.default_rng(7)
    q = build_generic_candidate_pool(rng, MODEL_CONTEXT, SMALL_POOL)
    lower = MODEL_CONTEXT.operational_lower_rad
    upper = MODEL_CONTEXT.operational_upper_rad
    assert np.all(q >= lower)
    assert np.all(q <= upper)


def test_normalized_margin_formula_matches_kinematics_joint_limit_utils():
    rng = np.random.default_rng(11)
    q = build_generic_candidate_pool(rng, MODEL_CONTEXT, 20)
    metrics = compute_candidate_metrics(MODEL_CONTEXT, q)

    lower = MODEL_CONTEXT.operational_lower_rad
    upper = MODEL_CONTEXT.operational_upper_rad
    for i in range(q.shape[0]):
        expected = float(np.min(normalized_joint_limit_margin(q[i], lower, upper)))
        assert metrics.normalized_min_margin[i] == pytest.approx(expected)


def test_sigma_min_matches_kinematics_singularity_metrics():
    rng = np.random.default_rng(21)
    q = build_generic_candidate_pool(rng, MODEL_CONTEXT, 20)
    metrics = compute_candidate_metrics(MODEL_CONTEXT, q)

    data = MODEL_CONTEXT.new_data()
    for i in range(q.shape[0]):
        J = geometric_jacobian_world(MODEL_CONTEXT, q[i], data=data)
        expected = minimum_singular_value(J)
        assert metrics.sigma_min[i] == pytest.approx(expected)


def test_classification_boundary_behavior():
    near_joint_limit_threshold = 0.05
    near_singularity_threshold = 0.03
    moderate_upper = 0.09

    at_boundary = classify_single(0.05, 0.03, near_joint_limit_threshold, near_singularity_threshold, moderate_upper)
    assert at_boundary["near_joint_limit"] is True  # <= is near-limit
    assert at_boundary["near_singular"] is True  # <= is near-singular

    just_above = classify_single(0.0500001, 0.0300001, near_joint_limit_threshold, near_singularity_threshold, moderate_upper)
    assert just_above["near_joint_limit"] is False
    assert just_above["near_singular"] is False
    assert just_above["moderately_conditioned"] is True

    at_moderate_boundary = classify_single(0.5, moderate_upper, near_joint_limit_threshold, near_singularity_threshold, moderate_upper)
    assert at_moderate_boundary["moderately_conditioned"] is True
    assert at_moderate_boundary["regular_sigma"] is False

    just_above_moderate = classify_single(0.5, moderate_upper + 1e-9, near_joint_limit_threshold, near_singularity_threshold, moderate_upper)
    assert just_above_moderate["regular_sigma"] is True
    assert just_above_moderate["moderately_conditioned"] is False


def test_classification_groups_are_non_overlapping():
    rng = np.random.default_rng(99)
    n = 300
    normalized_margins = rng.uniform(0.0, 1.0, size=n)
    sigma_mins = rng.uniform(0.0, 0.3, size=n)

    for margin, sigma in zip(normalized_margins, sigma_mins):
        result = classify_single(margin, sigma, 0.025, 0.03, 0.09)
        # sigma-axis tri-state must be mutually exclusive
        assert sum([result["near_singular"], result["moderately_conditioned"], result["regular_sigma"]]) == 1
        # "regular" (spec section 6) implies not-near-limit and regular-sigma simultaneously
        if result["regular"]:
            assert not result["near_joint_limit"]
            assert result["regular_sigma"]


def test_regular_state_is_not_near_limit_and_not_near_singular():
    # Note: q=0 (the arm's "zero/home" pose) is itself an exact singularity for this model
    # (sigma_min ~1e-18, confirmed by direct computation) -- consistent with Tier 0's
    # "zero_or_home" FK group perturbing away from zero rather than assuming it's regular. Use a
    # generic interior configuration instead.
    interior_q = np.array([0.3, 0.5, -0.4, 0.8, 0.2, -0.3, 0.6])
    metrics = compute_candidate_metrics(MODEL_CONTEXT, interior_q.reshape(1, -1))
    result = classify_single(
        float(metrics.normalized_min_margin[0]),
        float(metrics.sigma_min[0]),
        near_joint_limit_threshold=0.025,
        near_singularity_threshold=0.03,
        moderately_conditioned_upper_bound=0.09,
    )
    assert result["near_joint_limit"] is False
    assert result["near_singular"] is False
    assert result["regular"] is True


def test_near_singular_uses_computed_sigma_min_not_a_label():
    rng = np.random.default_rng(31)
    q = build_singularity_biased_pool(rng, MODEL_CONTEXT, SMALL_SINGULARITY_POOL)
    metrics = compute_candidate_metrics(MODEL_CONTEXT, q)
    threshold = 0.03

    classified_near_singular = metrics.sigma_min <= threshold
    data = MODEL_CONTEXT.new_data()
    for i in np.flatnonzero(classified_near_singular)[:10]:
        J = geometric_jacobian_world(MODEL_CONTEXT, q[i], data=data)
        assert minimum_singular_value(J) <= threshold


def test_expected_candidate_counts_sufficient_for_point_ik_and_anchors():
    pool_size = 5000
    result = calibrate(42, generic_pool_size=pool_size, singularity_pool_size=pool_size, model_context=MODEL_CONTEXT)

    # A P10-derived threshold guarantees ~10% of the generic pool qualifies, by definition of a
    # quantile; at the locked default pool size (50,000) that is ~5,000, comfortably >= the 1,000
    # Point-IK needs per difficulty group. Verify the proportion holds at this smaller test scale.
    assert result.candidate_counts["near_joint_limit_in_generic_pool"] >= 0.08 * pool_size
    # Anchors need only 3 near_limit / 3 near_singular candidates with diversity -- comfortably
    # covered by the biased pool's near-singular yield.
    assert result.candidate_counts["near_singularity_in_biased_pool"] >= 3
    assert result.candidate_counts["moderately_conditioned_in_biased_pool"] >= 3
    assert result.candidate_counts["regular_sigma_in_biased_pool"] >= 6
    assert result.candidate_counts["regular_combined_in_generic_pool"] >= 6


def test_difficulty_threshold_config_parses_and_matches_calibration_module():
    from dataset_v2 import config_templates as ct

    config = ct.difficulty_threshold_config()
    json.dumps(config)  # must be JSON-serializable
    assert config["near_joint_limit"]["threshold_normalized"] == ct.NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD
    assert config["near_singularity"]["threshold_sigma_min"] == ct.NEAR_SINGULARITY_SIGMA_MIN_THRESHOLD
    assert config["classification_priority_highest_first"] == CLASSIFICATION_PRIORITY


def test_all_configs_include_difficulty_thresholds():
    configs = all_configs(master_seed=42)
    assert "difficulty_thresholds.json" in configs
    assert configs["difficulty_thresholds.json"]["status"] == "locked"


def test_dataset_v1_files_not_touched_by_calibration_import():
    # Importing/running calibration must never read/write anything under Dataset v1's paths;
    # regression-covered here by asserting the v1 dls_config.json used as the singularity
    # threshold source is read, not written, and its content is unchanged by calibrate().
    from utils.dataset_locator import CONFIGS_DIR as V1_CONFIGS_DIR

    dls_config_path = V1_CONFIGS_DIR / "dls_config.json"
    before = dls_config_path.read_text(encoding="utf-8")
    calibrate(42, generic_pool_size=50, singularity_pool_size=50, model_context=MODEL_CONTEXT)
    after = dls_config_path.read_text(encoding="utf-8")
    assert before == after
