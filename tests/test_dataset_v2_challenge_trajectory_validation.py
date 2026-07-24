"""Tests for the Dataset v2 random-challenge trajectory validator (Phase 6).

Generates one small clean challenge set (module fixture), then copies-and-tampers it per test so
each corruption is checked independently without re-running expensive generation. Never touches
Dataset v1.
"""

import csv
import json
import shutil

import numpy as np
import pytest

from dataset_v2.challenge_trajectory_generation import MANIFEST_NAME, NPZ_SUFFIX, run_challenge_trajectory_generation
from dataset_v2.challenge_trajectory_validation import validate_challenge_trajectories
from dataset_v2.locator import dataset_v2_paths
from dataset_v2.scaffold import create_dataset_v2_scaffold
from kinematics.model_loader import load_model_context
from pipelines.run_dataset_v2_challenge_trajectory_generation import main as challenge_cli_main
from utils.npz_utils import load_npz, save_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()
SOURCE_SMALL = 151
POOL_SMALL = 6
FAMILIES = ["smooth_random", "non_planar"]
SPLITS_USED = ["development", "validation"]
PER = 2


@pytest.fixture(scope="module")
def clean_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("challenge_val") / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    run_challenge_trajectory_generation(
        root, master_seed=MASTER_SEED, families=FAMILIES, splits=SPLITS_USED,
        per_family_per_split=PER, source_waypoint_count=SOURCE_SMALL, candidate_pool_size=POOL_SMALL, model_context=MODEL_CONTEXT,
    )
    return root


def _copy(clean_root, tmp_path):
    dest = tmp_path / "ds"
    shutil.copytree(clean_root, dest)
    return dest


def _validate(root, **kwargs):
    kwargs.setdefault("full_counts", False)
    kwargs.setdefault("check_combined", False)
    return validate_challenge_trajectories(root, model_context=MODEL_CONTEXT, **kwargs)


def _manifest_path(root):
    return dataset_v2_paths(root).trajectories_dir / MANIFEST_NAME


def _read_rows(root):
    with open(_manifest_path(root), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle)), None


def _write_rows(root, rows, fieldnames):
    with open(_manifest_path(root), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _first_canonical_npz(root, row):
    return dataset_v2_paths(root).trajectories_development_dir / f"{row['trajectory_id']}{NPZ_SUFFIX}"


def test_clean_set_passes(clean_root):
    report = _validate(clean_root)
    assert report.passed, report.reasons
    assert report.total_trajectories == len(FAMILIES) * len(SPLITS_USED) * PER


def test_wrong_total_count_detected(clean_root):
    # full_counts=True expects 90; a small set must fail the count checks
    report = _validate(clean_root, full_counts=True)
    assert not report.passed
    assert any("challenge trajectories" in r for r in report.reasons)


def test_duplicate_trajectory_id_detected(clean_root, tmp_path):
    root = _copy(clean_root, tmp_path)
    with open(_manifest_path(root), newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[1]["trajectory_id"] = rows[0]["trajectory_id"]
    _write_rows(root, rows, fieldnames)
    report = _validate(root)
    assert not report.passed
    assert any("duplicate trajectory_id" in r for r in report.reasons)


def test_missing_npz_detected(clean_root, tmp_path):
    root = _copy(clean_root, tmp_path)
    rows, _ = _read_rows(root)
    dev_row = next(r for r in rows if r["split"] == "development")
    _first_canonical_npz(root, dev_row).unlink()
    report = _validate(root)
    assert not report.passed
    assert any("canonical NPZ missing" in r for r in report.reasons)


def test_non_finite_value_detected(clean_root, tmp_path):
    root = _copy(clean_root, tmp_path)
    rows, _ = _read_rows(root)
    dev_row = next(r for r in rows if r["split"] == "development")
    path = _first_canonical_npz(root, dev_row)
    arrays = dict(load_npz(path))
    arrays["target_position"][5, 0] = np.inf
    # write directly (save_npz rejects non-finite values); exercises the validator's own check
    np.savez(path, **arrays)
    report = _validate(root)
    assert not report.passed
    assert any("non-finite" in r for r in report.reasons)


def test_non_unit_quaternion_detected(clean_root, tmp_path):
    root = _copy(clean_root, tmp_path)
    rows, _ = _read_rows(root)
    dev_row = next(r for r in rows if r["split"] == "development")
    path = _first_canonical_npz(root, dev_row)
    arrays = dict(load_npz(path))
    arrays["target_quaternion"][3] = arrays["target_quaternion"][3] * 1.5
    save_npz(path, arrays, overwrite=True)
    report = _validate(root)
    assert not report.passed
    assert any("quaternion" in r for r in report.reasons)


def test_wrong_arc_length_in_manifest_detected(clean_root, tmp_path):
    root = _copy(clean_root, tmp_path)
    with open(_manifest_path(root), newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["arc_length_m"] = "999.0"
    _write_rows(root, rows, fieldnames)
    report = _validate(root)
    assert not report.passed
    assert any("arc_length_m mismatch" in r for r in report.reasons)


def test_unreachable_flag_detected(clean_root, tmp_path):
    root = _copy(clean_root, tmp_path)
    rows, _ = _read_rows(root)
    dev_row = next(r for r in rows if r["split"] == "development")
    path = _first_canonical_npz(root, dev_row)
    arrays = dict(load_npz(path))
    arrays["waypoint_reachable"][10] = False
    save_npz(path, arrays, overwrite=True)
    report = _validate(root)
    assert not report.passed
    assert any("unreachable" in r for r in report.reasons)


def test_fk_mismatch_detected(clean_root, tmp_path):
    root = _copy(clean_root, tmp_path)
    rows, _ = _read_rows(root)
    dev_row = next(r for r in rows if r["split"] == "development")
    path = _first_canonical_npz(root, dev_row)
    arrays = dict(load_npz(path))
    arrays["q_reference"][7] = arrays["q_reference"][7] + 0.2  # break FK reconstruction
    save_npz(path, arrays, overwrite=True)
    report = _validate(root)
    assert not report.passed
    assert any("FK reconstruction" in r or "disagree" in r or "joint limits" in r for r in report.reasons)


def test_canonical_not_from_source_detected(clean_root, tmp_path):
    root = _copy(clean_root, tmp_path)
    rows, _ = _read_rows(root)
    dev_row = next(r for r in rows if r["split"] == "development")
    path = _first_canonical_npz(root, dev_row)
    arrays = dict(load_npz(path))
    arrays["target_position"][20] = arrays["target_position"][20] + np.array([0.05, 0.0, 0.0])
    save_npz(path, arrays, overwrite=True)
    report = _validate(root)
    assert not report.passed
    assert any("fresh resample" in r for r in report.reasons)


def test_cross_split_seed_leakage_detected(clean_root, tmp_path):
    root = _copy(clean_root, tmp_path)
    with open(_manifest_path(root), newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    dev_seed = next(r["path_seed"] for r in rows if r["split"] == "development")
    for r in rows:
        if r["split"] == "validation":
            r["path_seed"] = dev_seed
            break
    _write_rows(root, rows, fieldnames)
    report = _validate(root)
    assert not report.passed
    assert any("path_seed leakage" in r for r in report.reasons)


def test_missing_anti_leakage_report_detected(clean_root, tmp_path):
    root = _copy(clean_root, tmp_path)
    (dataset_v2_paths(root).trajectories_dir / "challenge_trajectory_anti_leakage_report.json").unlink()
    report = _validate(root)
    assert not report.passed
    assert any("anti-leakage report missing" in r for r in report.reasons)


def test_duplicate_with_core_detected(clean_root, tmp_path):
    root = _copy(clean_root, tmp_path)
    rows, _ = _read_rows(root)
    core_manifest = dataset_v2_paths(root).trajectories_dir / "core_trajectory_manifest.csv"
    with open(core_manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory_id", "content_hash"])
        writer.writerow(["core_line_fixed_anchor_regular_00", rows[0]["content_hash"]])
    report = _validate(root)
    assert not report.passed
    assert any("core trajectory" in r for r in report.reasons)


def test_combined_totals_checked_when_core_present(clean_root, tmp_path):
    # a core manifest with mismatched size makes the combined 210 check fail under full_counts
    root = _copy(clean_root, tmp_path)
    core_manifest = dataset_v2_paths(root).trajectories_dir / "core_trajectory_manifest.csv"
    with open(core_manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory_id", "content_hash"])
        writer.writerow(["core_line_fixed_anchor_regular_00", "deadbeef"])
    report = validate_challenge_trajectories(root, model_context=MODEL_CONTEXT, full_counts=True, check_combined=True)
    assert not report.passed
    assert any("combined trajectory total" in r for r in report.reasons)


def test_cli_validate_exit_codes(clean_root, tmp_path, capsys):
    rc_ok = challenge_cli_main(
        ["--dataset-root", str(clean_root), "--validate-only", "--family", "smooth_random", "--family", "non_planar", "--split", "development", "--split", "validation", "--per-family-per-split", str(PER)]
    )
    assert rc_ok == 0
    root = _copy(clean_root, tmp_path)
    rows, _ = _read_rows(root)
    dev_row = next(r for r in rows if r["split"] == "development")
    _first_canonical_npz(root, dev_row).unlink()
    rc_fail = challenge_cli_main(
        ["--dataset-root", str(root), "--validate-only", "--family", "smooth_random", "--family", "non_planar", "--split", "development", "--split", "validation", "--per-family-per-split", str(PER)]
    )
    assert rc_fail == 1
