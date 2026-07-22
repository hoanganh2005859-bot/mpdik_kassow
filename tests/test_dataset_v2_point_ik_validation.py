"""Tests for the independent Dataset v2 Point-IK validator (Phase 3, spec section 11).

Confirms the validator passes on a clean fixture and separately detects each corruption class:
wrong counts, duplicate sample/content-hash, out-of-limit joint state, FK/target mismatch, and a
wrong difficulty classification. Never calls DLS. Never touches Dataset v1.
"""

import numpy as np
import pytest

from dataset_v2.config_templates import DIFFICULTY_GROUPS
from dataset_v2.point_ik_generation import run_point_ik_generation
from dataset_v2.point_ik_validation import validate_point_ik
from dataset_v2.scaffold import create_dataset_v2_scaffold
from kinematics.model_loader import load_model_context
from pipelines.run_dataset_v2_tier1_generation import main as point_ik_cli_main
from utils.npz_utils import load_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()
SAMPLES_PER_GROUP_SMALL = 20
SPLIT_SIZES_SMALL = {"development": 4, "validation": 4, "frozen_test": 12}
POOL_SIZE_SMALL = 1500


@pytest.fixture
def generated_root(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    result = run_point_ik_generation(
        root,
        master_seed=MASTER_SEED,
        samples_per_group=SAMPLES_PER_GROUP_SMALL,
        pool_size=POOL_SIZE_SMALL,
        split_sizes_per_group=dict(SPLIT_SIZES_SMALL),
        model_context=MODEL_CONTEXT,
    )
    return root, result


def _tamper(path, mutate_fn):
    arrays = dict(load_npz(path))
    mutate_fn(arrays)
    np.savez(path, **arrays)


def test_validator_passes_on_clean_fixture(generated_root):
    root, result = generated_root
    report = validate_point_ik(root, full_counts=False, expected_samples_per_group=SAMPLES_PER_GROUP_SMALL)
    assert report.passed, report.reasons
    assert report.total_samples == SAMPLES_PER_GROUP_SMALL * len(DIFFICULTY_GROUPS)
    assert report.group_counts == {name: SAMPLES_PER_GROUP_SMALL for name in DIFFICULTY_GROUPS}


def test_validator_detects_wrong_total_count(generated_root):
    root, result = generated_root
    report = validate_point_ik(root, full_counts=False, expected_samples_per_group=SAMPLES_PER_GROUP_SMALL + 1)
    assert not report.passed
    assert any("expected" in r for r in report.reasons)


def test_validator_detects_wrong_group_count_under_full_counts(generated_root):
    root, result = generated_root
    # full_counts=True expects the locked 6000/1000-per-group/1200-1200-3600 shape; this fixture
    # is intentionally smaller, so the full-counts check must fail loudly, not silently pass.
    report = validate_point_ik(root, full_counts=True)
    assert not report.passed
    assert any("expected 6000" in r for r in report.reasons)


def test_validator_detects_duplicate_sample_id(generated_root):
    root, result = generated_root
    dev_path = result.tier1_point_ik_dir / "development.npz"

    def mutate(arrays):
        arrays["sample_id"][1] = arrays["sample_id"][0]

    _tamper(dev_path, mutate)
    report = validate_point_ik(root, full_counts=False, expected_samples_per_group=SAMPLES_PER_GROUP_SMALL)
    assert not report.passed
    assert any("duplicate sample_id" in r for r in report.reasons)


def test_validator_detects_duplicate_content_hash(generated_root):
    root, result = generated_root
    dev_path = result.tier1_point_ik_dir / "development.npz"

    def mutate(arrays):
        arrays["content_hash"][1] = arrays["content_hash"][0]

    _tamper(dev_path, mutate)
    report = validate_point_ik(root, full_counts=False, expected_samples_per_group=SAMPLES_PER_GROUP_SMALL)
    assert not report.passed
    assert any("duplicate content_hash" in r for r in report.reasons)


def test_validator_detects_out_of_limit_joint_state(generated_root):
    root, result = generated_root
    dev_path = result.tier1_point_ik_dir / "development.npz"

    def mutate(arrays):
        arrays["q_initial"][0, 0] += 100.0

    _tamper(dev_path, mutate)
    report = validate_point_ik(root, full_counts=False, expected_samples_per_group=SAMPLES_PER_GROUP_SMALL)
    assert not report.passed
    assert any("operational joint limits" in r for r in report.reasons)


def test_validator_detects_fk_pose_mismatch(generated_root):
    root, result = generated_root
    dev_path = result.tier1_point_ik_dir / "development.npz"

    def mutate(arrays):
        arrays["target_position"][0] += 0.5  # far beyond FK_POSITION_TOLERANCE_M

    _tamper(dev_path, mutate)
    report = validate_point_ik(root, full_counts=False, expected_samples_per_group=SAMPLES_PER_GROUP_SMALL)
    assert not report.passed
    assert any("target_position" in r for r in report.reasons)


def test_validator_detects_wrong_difficulty_classification(generated_root):
    root, result = generated_root
    dev_path = result.tier1_point_ik_dir / "development.npz"

    def mutate(arrays):
        current = int(arrays["difficulty_id"][0])
        arrays["difficulty_id"][0] = (current + 1) % len(DIFFICULTY_GROUPS)

    _tamper(dev_path, mutate)
    report = validate_point_ik(root, full_counts=False, expected_samples_per_group=SAMPLES_PER_GROUP_SMALL)
    assert not report.passed
    assert any("difficulty classification" in r for r in report.reasons)


def test_cli_validate_exit_code_0_on_pass(generated_root):
    root, result = generated_root
    exit_code = point_ik_cli_main(
        ["--dataset-root", str(root), "--validate-only", "--sample-limit-per-group", str(SAMPLES_PER_GROUP_SMALL)]
    )
    assert exit_code == 0


def test_cli_validate_exit_code_1_on_corruption(generated_root):
    root, result = generated_root
    dev_path = result.tier1_point_ik_dir / "development.npz"

    def mutate(arrays):
        arrays["q_initial"][0, 0] += 100.0

    _tamper(dev_path, mutate)
    exit_code = point_ik_cli_main(
        ["--dataset-root", str(root), "--validate-only", "--sample-limit-per-group", str(SAMPLES_PER_GROUP_SMALL)]
    )
    assert exit_code == 1
