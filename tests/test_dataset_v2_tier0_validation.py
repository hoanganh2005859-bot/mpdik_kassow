"""Tests for the Dataset v2 Tier 0 validator (Phase 2).

Confirms the validator passes on freshly generated fixtures and rejects deliberately corrupted
NPZ/metadata (wrong counts, duplicates, out-of-limit states, oversized FD epsilon, misclassified
singularity groups).
"""

import json

import numpy as np
import pytest

from dataset_v2.locator import dataset_v2_paths
from dataset_v2.scaffold import create_dataset_v2_scaffold
from dataset_v2.tier0_generation import JACOBIAN_GROUPS, SINGULARITY_GROUPS, run_tier0_generation
from dataset_v2.tier0_validation import validate_tier0
from kinematics.model_loader import load_model_context
from pipelines.run_dataset_v2_tier0_generation import main as tier0_cli_main
from utils.npz_utils import load_npz, save_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()

FK_SMALL = 10
JACOBIAN_SMALL = 10
SINGULARITY_SMALL = 9
SINGULARITY_POOL_SMALL = 4000
JACOBIAN_POOL_SMALL = 1000

EXPECTED_GROUP_COUNTS = {
    "fk": {name: FK_SMALL // 5 for name in ("zero_or_home", "random_interior", "near_operational_lower_limit", "near_operational_upper_limit", "mixed_near_limits")},
    "jacobian": {name: JACOBIAN_SMALL // 5 for name in JACOBIAN_GROUPS.values()},
    "singularity": {name: SINGULARITY_SMALL // 3 for name in SINGULARITY_GROUPS.values()},
}


@pytest.fixture
def generated_v2_root(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    run_tier0_generation(
        root,
        master_seed=MASTER_SEED,
        fk_total=FK_SMALL,
        jacobian_total=JACOBIAN_SMALL,
        singularity_total=SINGULARITY_SMALL,
        jacobian_candidate_pool_size=JACOBIAN_POOL_SMALL,
        singularity_candidate_pool_size=SINGULARITY_POOL_SMALL,
        model_context=MODEL_CONTEXT,
    )
    return root


def test_validator_passes_on_freshly_generated_fixture(generated_v2_root):
    report = validate_tier0(
        generated_v2_root, model_context=MODEL_CONTEXT, expected_group_counts=EXPECTED_GROUP_COUNTS, full_counts=False
    )
    assert report.passed, report.reasons
    assert report.fk_count == FK_SMALL
    assert report.jacobian_count == JACOBIAN_SMALL
    assert report.singularity_count == SINGULARITY_SMALL
    assert report.max_jacobian_relative_error <= 1e-4


def test_validator_detects_wrong_total_count(generated_v2_root):
    paths = dataset_v2_paths(generated_v2_root)
    fk_path = paths.tier0_validation_dir / "fk_test_states_v2.npz"
    arrays = load_npz(fk_path)
    truncated = {name: arr[:-1] for name, arr in arrays.items()}
    save_npz(fk_path, truncated, overwrite=True)

    report = validate_tier0(
        generated_v2_root, model_context=MODEL_CONTEXT, expected_group_counts=EXPECTED_GROUP_COUNTS, full_counts=False
    )
    assert not report.passed
    assert any("expected" in r and "FK" in r for r in report.reasons)


def test_validator_detects_wrong_group_counts(generated_v2_root):
    report = validate_tier0(
        generated_v2_root,
        model_context=MODEL_CONTEXT,
        expected_group_counts={
            "fk": {**EXPECTED_GROUP_COUNTS["fk"], "zero_or_home": 999},
            "jacobian": EXPECTED_GROUP_COUNTS["jacobian"],
            "singularity": EXPECTED_GROUP_COUNTS["singularity"],
        },
        full_counts=False,
    )
    assert not report.passed
    assert any("zero_or_home" in r for r in report.reasons)


def test_validator_detects_duplicate_states(generated_v2_root):
    paths = dataset_v2_paths(generated_v2_root)
    fk_path = paths.tier0_validation_dir / "fk_test_states_v2.npz"
    arrays = load_npz(fk_path)
    arrays = dict(arrays)
    arrays["q_samples"] = np.array(arrays["q_samples"], copy=True)
    arrays["q_samples"][1] = arrays["q_samples"][0]
    save_npz(fk_path, arrays, overwrite=True)

    report = validate_tier0(
        generated_v2_root, model_context=MODEL_CONTEXT, expected_group_counts=EXPECTED_GROUP_COUNTS, full_counts=False
    )
    assert not report.passed
    assert any("duplicate" in r.lower() for r in report.reasons)


def test_validator_detects_out_of_limit_states(generated_v2_root):
    paths = dataset_v2_paths(generated_v2_root)
    fk_path = paths.tier0_validation_dir / "fk_test_states_v2.npz"
    arrays = dict(load_npz(fk_path))
    arrays["q_samples"] = np.array(arrays["q_samples"], copy=True)
    arrays["q_samples"][0, 0] = MODEL_CONTEXT.operational_upper_rad[0] + 1.0
    save_npz(fk_path, arrays, overwrite=True)

    report = validate_tier0(
        generated_v2_root, model_context=MODEL_CONTEXT, expected_group_counts=EXPECTED_GROUP_COUNTS, full_counts=False
    )
    assert not report.passed
    assert any("operational" in r.lower() for r in report.reasons)


def test_validator_detects_jacobian_relative_error_over_threshold(generated_v2_root):
    paths = dataset_v2_paths(generated_v2_root)
    jacobian_path = paths.tier0_validation_dir / "jacobian_test_states_v2.npz"
    arrays = dict(load_npz(jacobian_path))
    arrays["finite_difference_epsilon"] = np.full_like(arrays["finite_difference_epsilon"], 0.5)
    save_npz(jacobian_path, arrays, overwrite=True)

    report = validate_tier0(
        generated_v2_root, model_context=MODEL_CONTEXT, expected_group_counts=EXPECTED_GROUP_COUNTS, full_counts=False
    )
    assert not report.passed
    assert any("relative-error" in r for r in report.reasons)


def test_validator_detects_singularity_misclassification(generated_v2_root):
    paths = dataset_v2_paths(generated_v2_root)
    singularity_path = paths.tier0_validation_dir / "singularity_test_states_v2.npz"
    arrays = dict(load_npz(singularity_path))
    name_to_id = {name: gid for gid, name in SINGULARITY_GROUPS.items()}
    arrays["group_id"] = np.array(arrays["group_id"], copy=True)
    near_idx = np.flatnonzero(arrays["group_id"] == name_to_id["near_singular"])[0]
    arrays["group_id"][near_idx] = name_to_id["regular"]
    save_npz(singularity_path, arrays, overwrite=True)

    report = validate_tier0(
        generated_v2_root, model_context=MODEL_CONTEXT, expected_group_counts=EXPECTED_GROUP_COUNTS, full_counts=False
    )
    assert not report.passed
    assert any("regular" in r and "margin" in r for r in report.reasons)


def test_validator_cli_exit_code_nonzero_on_failure(generated_v2_root):
    paths = dataset_v2_paths(generated_v2_root)
    fk_path = paths.tier0_validation_dir / "fk_test_states_v2.npz"
    arrays = load_npz(fk_path)
    truncated = {name: arr[:-1] for name, arr in arrays.items()}
    save_npz(fk_path, truncated, overwrite=True)

    exit_code = tier0_cli_main(
        [
            "--dataset-root", str(generated_v2_root),
            "--validate-only",
            "--fk-count", str(FK_SMALL),
            "--jacobian-count", str(JACOBIAN_SMALL),
            "--singularity-count", str(SINGULARITY_SMALL),
        ]
    )
    assert exit_code == 1


def test_validator_cli_exit_code_zero_on_success(generated_v2_root):
    exit_code = tier0_cli_main(
        [
            "--dataset-root", str(generated_v2_root),
            "--validate-only",
            "--fk-count", str(FK_SMALL),
            "--jacobian-count", str(JACOBIAN_SMALL),
            "--singularity-count", str(SINGULARITY_SMALL),
        ]
    )
    assert exit_code == 0
