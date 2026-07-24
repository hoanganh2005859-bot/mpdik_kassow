"""Tests for the independent Dataset v2 core trajectory validator (Phase 5, spec section 13).

Generates one small fixture set (module-scoped, one anchor x all shapes/modes) once, then copies
it into a fresh directory per corruption test before tampering -- avoids re-running expensive
sequential-DLS reachability validation for every corruption case. Never calls DLS itself. Never
touches Dataset v1.
"""

import csv
import shutil

import numpy as np
import pytest

from dataset_v2.anchor_generation import run_anchor_generation
from dataset_v2.core_trajectory_generation import MANIFEST_NAME, run_core_trajectory_generation
from dataset_v2.core_trajectory_validation import validate_core_trajectories
from dataset_v2.locator import dataset_v2_paths
from dataset_v2.scaffold import create_dataset_v2_scaffold
from kinematics.model_loader import load_model_context
from pipelines.run_dataset_v2_core_trajectory_generation import main as core_trajectory_cli_main
from utils.npz_utils import load_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()
ANCHOR_POOL_SMALL = 200
SOURCE_WAYPOINTS_SMALL = 450


def _read_manifest(root):
    paths = dataset_v2_paths(root)
    with open(paths.trajectories_dir / MANIFEST_NAME, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(root, rows, fieldnames):
    paths = dataset_v2_paths(root)
    with open(paths.trajectories_dir / MANIFEST_NAME, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tamper_npz(path, mutate_fn):
    arrays = dict(load_npz(path))
    mutate_fn(arrays)
    np.savez(path, **arrays)


@pytest.fixture(scope="module")
def golden_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("core_traj_val") / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    run_anchor_generation(
        root,
        master_seed=MASTER_SEED,
        regular_pool_size=ANCHOR_POOL_SMALL,
        near_limit_biased_pool_size=ANCHOR_POOL_SMALL,
        singularity_biased_pool_size=ANCHOR_POOL_SMALL,
        model_context=MODEL_CONTEXT,
    )
    run_core_trajectory_generation(
        root,
        master_seed=MASTER_SEED,
        anchor_ids=["anchor_regular_00"],
        source_waypoint_count=SOURCE_WAYPOINTS_SMALL,
        model_context=MODEL_CONTEXT,
    )
    return root


@pytest.fixture
def working_root(golden_root, tmp_path):
    dest = tmp_path / "kr810_dataset_v2"
    shutil.copytree(golden_root, dest)
    return dest


def test_validator_passes_on_clean_fixture(working_root):
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert report.passed, report.reasons
    assert report.total_trajectories == 10
    assert report.canonical_poses_total == 4000


def test_validator_detects_duplicate_trajectory_id(working_root):
    rows = _read_manifest(working_root)
    fieldnames = list(rows[0].keys())
    rows[1] = dict(rows[1])
    rows[1]["trajectory_id"] = rows[0]["trajectory_id"]
    _write_manifest(working_root, rows, fieldnames)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("duplicate trajectory_id" in r for r in report.reasons)


def test_validator_detects_duplicate_content_hash(working_root):
    rows = _read_manifest(working_root)
    fieldnames = list(rows[0].keys())
    rows[1] = dict(rows[1])
    rows[1]["content_hash"] = rows[0]["content_hash"]
    _write_manifest(working_root, rows, fieldnames)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("duplicate content_hash" in r for r in report.reasons)


def test_validator_detects_wrong_total_count(working_root):
    rows = _read_manifest(working_root)
    fieldnames = list(rows[0].keys())
    _write_manifest(working_root, rows[:-1], fieldnames)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=True)
    assert not report.passed
    assert any("expected 120" in r for r in report.reasons)


def test_validator_detects_missing_npz_file(working_root):
    rows = _read_manifest(working_root)
    paths = dataset_v2_paths(working_root)
    row = rows[0]
    npz_path = paths.trajectories_development_dir / f"{row['trajectory_id']}.npz"
    npz_path.unlink()
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("missing canonical NPZ" in r for r in report.reasons)


def test_validator_detects_non_finite_values(working_root):
    rows = _read_manifest(working_root)
    paths = dataset_v2_paths(working_root)
    row = rows[0]
    npz_path = paths.trajectories_development_dir / f"{row['trajectory_id']}.npz"

    def mutate(arrays):
        arrays["target_position"][5, 0] = np.nan

    _tamper_npz(npz_path, mutate)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("non-finite" in r for r in report.reasons)


def test_validator_detects_quaternion_not_normalized(working_root):
    rows = _read_manifest(working_root)
    paths = dataset_v2_paths(working_root)
    row = rows[0]
    npz_path = paths.trajectories_development_dir / f"{row['trajectory_id']}.npz"

    def mutate(arrays):
        arrays["target_quaternion"][3] = arrays["target_quaternion"][3] * 2.0

    _tamper_npz(npz_path, mutate)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("non-unit quaternion" in r for r in report.reasons)


def test_validator_detects_wrong_arc_length_in_manifest(working_root):
    rows = _read_manifest(working_root)
    fieldnames = list(rows[0].keys())
    rows[0] = dict(rows[0])
    rows[0]["arc_length_m"] = str(float(rows[0]["arc_length_m"]) + 1.0)
    _write_manifest(working_root, rows, fieldnames)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("recomputed arc_length_m" in r for r in report.reasons)


def test_validator_detects_wrong_angular_displacement_in_manifest(working_root):
    rows = _read_manifest(working_root)
    fieldnames = list(rows[0].keys())
    rows[0] = dict(rows[0])
    rows[0]["cumulative_angular_displacement_rad"] = str(float(rows[0]["cumulative_angular_displacement_rad"]) + 1.0)
    _write_manifest(working_root, rows, fieldnames)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("recomputed angular displacement" in r for r in report.reasons)


def test_validator_detects_unreachable_waypoint_flag(working_root):
    rows = _read_manifest(working_root)
    paths = dataset_v2_paths(working_root)
    row = rows[0]
    npz_path = paths.trajectories_development_dir / f"{row['trajectory_id']}.npz"

    def mutate(arrays):
        arrays["waypoint_reachable"][7] = False

    _tamper_npz(npz_path, mutate)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("unreachable" in r for r in report.reasons)


def test_validator_detects_fk_mismatch_in_q_reference(working_root):
    rows = _read_manifest(working_root)
    paths = dataset_v2_paths(working_root)
    row = rows[0]
    npz_path = paths.trajectories_development_dir / f"{row['trajectory_id']}.npz"

    def mutate(arrays):
        arrays["q_reference"][10] = arrays["q_reference"][10] + 1.0

    _tamper_npz(npz_path, mutate)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("exceed the strict position tolerance" in r for r in report.reasons)


def test_validator_detects_canonical_not_resampled_from_source(working_root):
    rows = _read_manifest(working_root)
    paths = dataset_v2_paths(working_root)
    row = rows[0]
    npz_path = paths.trajectories_development_dir / f"{row['trajectory_id']}.npz"

    def mutate(arrays):
        arrays["target_position"][200] = arrays["target_position"][200] + np.array([0.05, 0.0, 0.0])

    _tamper_npz(npz_path, mutate)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("fresh resample" in r for r in report.reasons)


def test_validator_detects_fixed_orientation_that_actually_varies(working_root):
    rows = _read_manifest(working_root)
    paths = dataset_v2_paths(working_root)
    row = next(r for r in rows if r["orientation_mode"] == "fixed")
    npz_path = paths.trajectories_development_dir / f"{row['trajectory_id']}.npz"

    def mutate(arrays):
        from kinematics.rotation_utils import so3_exp
        from kinematics.quaternion_utils import quaternion_wxyz_to_matrix, rotation_matrix_to_quaternion_wxyz

        R = quaternion_wxyz_to_matrix(arrays["target_quaternion"][300])
        R_perturbed = R @ so3_exp(np.array([0.5, 0.0, 0.0]))
        arrays["target_quaternion"][300] = rotation_matrix_to_quaternion_wxyz(R_perturbed)

    _tamper_npz(npz_path, mutate)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("orientation_mode=fixed but orientation varies" in r for r in report.reasons)


def test_validator_detects_split_leakage(working_root):
    rows = _read_manifest(working_root)
    fieldnames = list(rows[0].keys())
    rows[0] = dict(rows[0])
    rows[0]["split"] = "validation"
    _write_manifest(working_root, rows, fieldnames)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("more than one split" in r for r in report.reasons)


def test_validator_detects_missing_anti_leakage_report(working_root):
    paths = dataset_v2_paths(working_root)
    (paths.trajectories_dir / "core_trajectory_anti_leakage_report.json").unlink()
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("anti-leakage report not found" in r for r in report.reasons)


def test_cli_validate_exit_codes(working_root):
    assert core_trajectory_cli_main(["--dataset-root", str(working_root), "--validate-only", "--anchor-id", "anchor_regular_00"]) == 0

    rows = _read_manifest(working_root)
    fieldnames = list(rows[0].keys())
    rows[1] = dict(rows[1])
    rows[1]["trajectory_id"] = rows[0]["trajectory_id"]
    _write_manifest(working_root, rows, fieldnames)
    assert core_trajectory_cli_main(["--dataset-root", str(working_root), "--validate-only", "--anchor-id", "anchor_regular_00"]) == 1
