"""Tests for the independent Dataset v2 anchor validator (Phase 4, spec section 10).

Confirms the validator passes on a clean fixture and separately detects each corruption class:
wrong counts, wrong class, wrong split assignment, duplicate ID/hash, out-of-limit joint state,
and FK/pose mismatch. Never calls DLS. Never touches Dataset v1.
"""

import numpy as np
import pytest

from dataset_v2.anchor_generation import ANCHOR_CLASS_IDS, run_anchor_generation
from dataset_v2.anchor_validation import validate_anchors
from dataset_v2.scaffold import create_dataset_v2_scaffold
from kinematics.model_loader import load_model_context
from pipelines.run_dataset_v2_anchor_generation import main as anchor_cli_main
from utils.npz_utils import load_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()
POOL_SIZE_SMALL = 200


@pytest.fixture
def generated_root(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    result = run_anchor_generation(
        root,
        master_seed=MASTER_SEED,
        regular_pool_size=POOL_SIZE_SMALL,
        near_limit_biased_pool_size=POOL_SIZE_SMALL,
        singularity_biased_pool_size=POOL_SIZE_SMALL,
        model_context=MODEL_CONTEXT,
    )
    return root, result


def _tamper(path, mutate_fn):
    arrays = dict(load_npz(path))
    mutate_fn(arrays)
    np.savez(path, **arrays)


def test_validator_passes_on_clean_fixture(generated_root):
    root, result = generated_root
    report = validate_anchors(root, full_counts=True)
    assert report.passed, report.reasons
    assert report.total_anchors == 12
    assert report.class_counts == {"regular": 6, "near_limit": 3, "near_singular": 3}
    assert report.split_counts == {"development": 4, "validation": 4, "frozen_test": 4}


def test_validator_detects_wrong_count(generated_root):
    root, result = generated_root
    npz_path = result.anchors_dir / "anchors.npz"

    def mutate(arrays):
        for key in arrays:
            arrays[key] = arrays[key][:-1]  # drop the last anchor -> 11 total

    _tamper(npz_path, mutate)
    report = validate_anchors(root, full_counts=True)
    assert not report.passed
    assert any("expected 12" in r for r in report.reasons)


def test_validator_detects_wrong_class_count(generated_root):
    root, result = generated_root
    npz_path = result.anchors_dir / "anchors.npz"

    def mutate(arrays):
        # flip one regular anchor's class_id to near_limit -> wrong 6/3/3 class split
        idx = int(np.flatnonzero(arrays["anchor_class_id"] == ANCHOR_CLASS_IDS["regular"])[0])
        arrays["anchor_class_id"][idx] = ANCHOR_CLASS_IDS["near_limit"]

    _tamper(npz_path, mutate)
    report = validate_anchors(root, full_counts=True)
    assert not report.passed
    assert any("anchor class" in r for r in report.reasons)


def test_validator_detects_wrong_split_assignment(generated_root):
    root, result = generated_root
    npz_path = result.anchors_dir / "anchors.npz"

    def mutate(arrays):
        # move one development anchor to validation -> wrong per-split counts
        idx = int(np.flatnonzero(arrays["split"] == "development")[0])
        arrays["split"][idx] = "validation"

    _tamper(npz_path, mutate)
    report = validate_anchors(root, full_counts=True)
    assert not report.passed
    assert any("split" in r for r in report.reasons)


def test_validator_detects_duplicate_anchor_id(generated_root):
    root, result = generated_root
    npz_path = result.anchors_dir / "anchors.npz"

    def mutate(arrays):
        arrays["anchor_id"][1] = arrays["anchor_id"][0]

    _tamper(npz_path, mutate)
    report = validate_anchors(root, full_counts=True)
    assert not report.passed
    assert any("duplicate anchor_id" in r for r in report.reasons)


def test_validator_detects_duplicate_content_hash(generated_root):
    root, result = generated_root
    npz_path = result.anchors_dir / "anchors.npz"

    def mutate(arrays):
        arrays["content_hash"][1] = arrays["content_hash"][0]

    _tamper(npz_path, mutate)
    report = validate_anchors(root, full_counts=True)
    assert not report.passed
    assert any("duplicate content_hash" in r for r in report.reasons)


def test_validator_detects_out_of_limit_joint_state(generated_root):
    root, result = generated_root
    npz_path = result.anchors_dir / "anchors.npz"

    def mutate(arrays):
        arrays["q"][0, 0] += 100.0

    _tamper(npz_path, mutate)
    report = validate_anchors(root, full_counts=True)
    assert not report.passed
    assert any("operational joint limits" in r for r in report.reasons)


def test_validator_detects_fk_pose_mismatch(generated_root):
    root, result = generated_root
    npz_path = result.anchors_dir / "anchors.npz"

    def mutate(arrays):
        arrays["position"][0] += 0.5

    _tamper(npz_path, mutate)
    report = validate_anchors(root, full_counts=True)
    assert not report.passed
    assert any("position" in r for r in report.reasons)


def test_validator_detects_corrupted_metadata_via_covariate_mismatch(generated_root):
    root, result = generated_root
    npz_path = result.anchors_dir / "anchors.npz"

    def mutate(arrays):
        arrays["sigma_min"][0] += 10.0

    _tamper(npz_path, mutate)
    report = validate_anchors(root, full_counts=True)
    assert not report.passed
    assert any("recomputed covariate" in r for r in report.reasons)


def test_cli_validate_exit_code_0_on_pass(generated_root):
    root, result = generated_root
    exit_code = anchor_cli_main(
        [
            "--dataset-root", str(root), "--validate-only",
            "--regular-pool-size", str(POOL_SIZE_SMALL),
            "--near-limit-pool-size", str(POOL_SIZE_SMALL),
            "--singularity-pool-size", str(POOL_SIZE_SMALL),
        ]
    )
    assert exit_code == 0


def test_cli_validate_exit_code_1_on_corruption(generated_root):
    root, result = generated_root
    npz_path = result.anchors_dir / "anchors.npz"

    def mutate(arrays):
        arrays["q"][0, 0] += 100.0

    _tamper(npz_path, mutate)
    exit_code = anchor_cli_main(
        [
            "--dataset-root", str(root), "--validate-only",
            "--regular-pool-size", str(POOL_SIZE_SMALL),
            "--near-limit-pool-size", str(POOL_SIZE_SMALL),
            "--singularity-pool-size", str(POOL_SIZE_SMALL),
        ]
    )
    assert exit_code == 1
