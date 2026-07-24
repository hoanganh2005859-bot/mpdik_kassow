"""Unit tests for the Dataset v2 evaluation harness that need no dataset root (pure logic):
candidate configs, the protected-field runtime guard, reporting-tier classification, the
checkpoint/resume manager, and the lexicographic selection objective.
"""

import numpy as np
import pandas as pd
import pytest

from evaluation_v2 import candidate_configs as cc
from evaluation_v2.checkpoint import CheckpointManager, ResumeMismatchError, RunFingerprints
from evaluation_v2.protected_guard import (
    PROTECTED_FIELD_NAMES,
    assert_no_protected_fields,
    find_protected_fields,
    is_protected_field,
)
from evaluation_v2.reporting import reporting_success
from evaluation_v2.selection import select_candidate
from utils.exceptions import ModelConfigurationError


# ---- candidate configs --------------------------------------------------------------------
def test_candidate_set_is_finite_and_unique():
    cands = cc.candidate_set()
    assert len(cands) >= 2
    ids = [c.candidate_id for c in cands]
    assert len(ids) == len(set(ids))


def test_reporting_thresholds_are_the_locked_tiers():
    assert cc.REPORTING_THRESHOLDS["coarse"] == {"position_m": 0.006, "orientation_deg": 5.0}
    assert cc.REPORTING_THRESHOLDS["standard"] == {"position_m": 0.003, "orientation_deg": 2.0}
    assert cc.REPORTING_THRESHOLDS["strict"] == {"position_m": 0.001, "orientation_deg": 1.0}


def test_generation_tolerance_is_not_a_reporting_threshold():
    gen = (cc.GENERATION_TOLERANCE_POSITION_M, cc.GENERATION_TOLERANCE_ORIENTATION_DEG)
    for tier in cc.REPORTING_THRESHOLDS.values():
        assert (tier["position_m"], tier["orientation_deg"]) != gen


def test_solver_config_has_all_keys_the_solver_reads():
    required = {
        "max_iterations", "position_success_threshold_m", "orientation_success_threshold_deg",
        "position_weight", "orientation_weight", "damping_mode", "lambda_min", "lambda_max",
        "lambda_default", "step_scale", "max_joint_step_rad", "clip_to_operational_limits",
        "joint_limit_avoidance", "null_space_gain", "singularity_sigma_threshold",
    }
    for cand in cc.candidate_set():
        cfg = cand.solver_config()
        assert required <= set(cfg)
        # Convergence tolerance equals the strict reporting tier.
        assert cfg["position_success_threshold_m"] == cc.REPORTING_THRESHOLDS["strict"]["position_m"]
        assert cfg["orientation_success_threshold_deg"] == cc.REPORTING_THRESHOLDS["strict"]["orientation_deg"]


def test_pre_registration_record_lists_all_candidates():
    rec = cc.pre_registration_record()
    assert len(rec["candidates"]) == len(cc.candidate_set())
    assert rec["primary_reporting_tier"] == "standard"


# ---- protected guard ----------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(PROTECTED_FIELD_NAMES))
def test_protected_names_are_flagged(name):
    assert is_protected_field(name)


def test_guard_raises_on_protected_field():
    with pytest.raises(ModelConfigurationError):
        assert_no_protected_fields({"q_initial": 1, "q_reference": 2}, "test")


def test_guard_passes_clean_mapping():
    assert_no_protected_fields({"q_initial": 1, "target_position": 2, "target_quaternion_wxyz": 3}, "test")


def test_guard_catches_renamed_variants():
    assert find_protected_fields(["canonical_q_reference", "q_target_reference_v2", "x"]) == [
        "canonical_q_reference",
        "q_target_reference_v2",
    ]


# ---- reporting ----------------------------------------------------------------------------
def test_reporting_tiers_are_nested():
    # 2mm/1.5deg: within coarse and standard, not strict.
    s = reporting_success(0.002, 1.5)
    assert s["coarse"] and s["standard"] and not s["strict"]
    # 0.5mm/0.5deg: strict.
    assert all(reporting_success(0.0005, 0.5).values())
    # 8mm: nothing.
    assert not any(reporting_success(0.008, 0.1).values())


# ---- checkpoint / resume ------------------------------------------------------------------
def _fps(config="c0"):
    return RunFingerprints(config=config, code="code0", dataset="ds0", public_bundle="pb0")


def test_checkpoint_writes_and_reuses_shard(tmp_path):
    ckpt = CheckpointManager(tmp_path / "run")
    ckpt.begin(_fps(), resume=False, overwrite=False)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert not ckpt.has_shard("tier1/development")
    ckpt.write_shard("tier1/development", df)
    assert ckpt.has_shard("tier1/development")
    pd.testing.assert_frame_equal(ckpt.read_shard("tier1/development"), df)


def test_checkpoint_resume_reuses_completed_shards(tmp_path):
    run = tmp_path / "run"
    a = CheckpointManager(run)
    a.begin(_fps(), resume=False, overwrite=False)
    a.write_shard("s1", pd.DataFrame({"x": [1]}))
    # New manager, resume: shard already present and valid.
    b = CheckpointManager(run)
    b.begin(_fps(), resume=True, overwrite=False)
    assert b.has_shard("s1")
    assert b.completed_shard_keys() == ["s1"]


def test_checkpoint_resume_refuses_on_fingerprint_mismatch(tmp_path):
    run = tmp_path / "run"
    a = CheckpointManager(run)
    a.begin(_fps(config="c0"), resume=False, overwrite=False)
    b = CheckpointManager(run)
    with pytest.raises(ResumeMismatchError):
        b.begin(_fps(config="DIFFERENT"), resume=True, overwrite=False)


def test_checkpoint_detects_corrupted_shard(tmp_path):
    ckpt = CheckpointManager(tmp_path / "run")
    ckpt.begin(_fps(), resume=False, overwrite=False)
    ckpt.write_shard("s1", pd.DataFrame({"x": [1, 2]}))
    # Corrupt the shard file on disk -> has_shard must return False (hash mismatch).
    ckpt.shard_file("s1").write_text("garbage", encoding="utf-8")
    assert not ckpt.has_shard("s1")


def test_checkpoint_assemble_has_no_duplicate_rows(tmp_path):
    ckpt = CheckpointManager(tmp_path / "run")
    ckpt.begin(_fps(), resume=False, overwrite=False)
    ckpt.write_shard("s1", pd.DataFrame({"x": [1, 2]}))
    ckpt.write_shard("s2", pd.DataFrame({"x": [3, 4]}))
    assembled = ckpt.assemble(["s1", "s2"])
    assert list(assembled["x"]) == [1, 2, 3, 4]


# ---- selection objective ------------------------------------------------------------------
def _sel_metrics(**overrides):
    base = {
        "safe_execution": 1.0, "pointik_standard_macro": 0.5, "trajectory_standard_warm": 0.5,
        "trajectory_standard_cold": 0.5, "completion_macro": 1.0, "coverage_macro": 1.0,
        "neg_pose_error_p95": -0.01, "warm_continuity": 0.5, "neg_runtime_p99": -5.0,
    }
    base.update(overrides)
    return base


def test_selection_objective_spec_is_recorded_and_ordered():
    from evaluation_v2.selection import SELECTION_LEVELS, selection_objective_spec

    spec = selection_objective_spec()
    assert spec["aggregation"] == "task_level_macros_not_pooled_rows"
    orders = [lvl["order"] for lvl in spec["levels"]]
    assert orders == sorted(orders)
    # Point-IK and trajectory success are separate levels (never pooled).
    assert "pointik_standard_macro" in SELECTION_LEVELS
    assert "trajectory_standard_warm" in SELECTION_LEVELS
    assert "trajectory_standard_cold" in SELECTION_LEVELS


def test_selection_prefers_higher_pointik_macro_when_safe_ties():
    metrics = {
        "cand_A_adaptive_baseline": _sel_metrics(pointik_standard_macro=0.80),
        "cand_B_adaptive_deep": _sel_metrics(pointik_standard_macro=0.90),
    }
    assert select_candidate(metrics)["selected_candidate_id"] == "cand_B_adaptive_deep"


def test_selection_warm_and_cold_are_separate_levels():
    # Equal point macro; A better warm, B better cold. Warm (L3) outranks cold (L4) -> A wins.
    metrics = {
        "cand_A_adaptive_baseline": _sel_metrics(trajectory_standard_warm=0.9, trajectory_standard_cold=0.1),
        "cand_B_adaptive_deep": _sel_metrics(trajectory_standard_warm=0.8, trajectory_standard_cold=0.99),
    }
    assert select_candidate(metrics)["selected_candidate_id"] == "cand_A_adaptive_baseline"


def test_selection_safe_execution_dominates():
    metrics = {
        "cand_A_adaptive_baseline": _sel_metrics(
            safe_execution=1.0, pointik_standard_macro=0.10, trajectory_standard_warm=0.1,
            trajectory_standard_cold=0.1, completion_macro=0.0, coverage_macro=0.0,
        ),
        "cand_B_adaptive_deep": _sel_metrics(
            safe_execution=0.99, pointik_standard_macro=0.99, trajectory_standard_warm=0.99,
            trajectory_standard_cold=0.99, neg_pose_error_p95=-0.001, warm_continuity=1.0,
            neg_runtime_p99=-1.0,
        ),
    }
    assert select_candidate(metrics)["selected_candidate_id"] == "cand_A_adaptive_baseline"
