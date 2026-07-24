"""Tests for the Dataset v2 combined trajectory catalog and public/protected loaders (Phase 7).

Uses the fast FK-only fixture (``tests/_dataset_v2_trial_helpers.py``); the real 210-row / 84,000
canonical-pose combined catalog is verified against the persistent working dataset and reported in
docs/V2_IMPLEMENTATION_LOG.md, not inside this suite. Never touches Dataset v1.
"""

import csv

import pytest

from _dataset_v2_trial_helpers import build_fixture
from dataset_v2.locator import dataset_v2_paths
from dataset_v2.trajectory_catalog import (
    COMBINED_MANIFEST_NAME,
    build_combined_catalog,
    load_combined_catalog,
    validate_combined_catalog,
)
from dataset_v2.trajectory_loading import (
    PROTECTED_ARRAY_KEYS,
    load_protected_trajectory,
    load_public_trajectory,
)
from kinematics.model_loader import load_model_context

MODEL_CONTEXT = load_model_context()


@pytest.fixture(scope="module")
def catalog_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2_catalog") / "ds"
    build_fixture(root, MODEL_CONTEXT, n_core=6, n_challenge=6)
    return root


def test_combined_catalog_is_union_of_both_manifests(catalog_root):
    rows = load_combined_catalog(catalog_root)
    assert len(rows) == 12
    assert sum(1 for r in rows if r["family"] == "core") == 6
    assert sum(1 for r in rows if r["family"] == "random_challenge") == 6
    # rows are sorted by trajectory_id
    assert [r["trajectory_id"] for r in rows] == sorted(r["trajectory_id"] for r in rows)


def test_combined_catalog_validation_passes(catalog_root):
    report = validate_combined_catalog(catalog_root, full_counts=False)
    assert report.passed, report.reasons


def test_combined_catalog_paths_are_relative_and_exist(catalog_root):
    paths = dataset_v2_paths(catalog_root)
    rows = load_combined_catalog(catalog_root)
    for row in rows:
        for col in ("canonical_path", "source_path"):
            rel = row[col]
            assert not rel.startswith("/") and ":" not in rel, f"{col} must be relative: {rel}"
            assert (paths.root / rel).is_file()


def test_combined_catalog_detects_duplicate_row(catalog_root, tmp_path):
    paths = dataset_v2_paths(catalog_root)
    manifest_path = paths.trajectories_dir / COMBINED_MANIFEST_NAME
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    corrupt = tmp_path / "ds2"
    build_fixture(corrupt, MODEL_CONTEXT, n_core=6, n_challenge=6)
    corrupt_manifest = dataset_v2_paths(corrupt).trajectories_dir / COMBINED_MANIFEST_NAME
    with open(corrupt_manifest, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(rows[1])  # duplicate an existing data row
    report = validate_combined_catalog(corrupt, full_counts=False)
    assert not report.passed
    assert any("duplicate" in r for r in report.reasons)


def test_build_requires_both_per_family_manifests(tmp_path):
    from dataset_v2.scaffold import create_dataset_v2_scaffold

    root = tmp_path / "ds"
    create_dataset_v2_scaffold(root, master_seed=42)
    with pytest.raises(FileNotFoundError):
        build_combined_catalog(root, full_counts=False)


def test_public_loader_hides_q_reference(catalog_root):
    rows = load_combined_catalog(catalog_root)
    public = load_public_trajectory(catalog_root, rows[0]["trajectory_id"])
    for protected_key in PROTECTED_ARRAY_KEYS:
        assert protected_key not in public.arrays, f"public loader leaked {protected_key}"
    # public geometry is present
    assert "target_position" in public.arrays
    assert public.first_target_position.shape == (3,)
    assert public.first_target_quaternion_wxyz.shape == (4,)


def test_protected_loader_reads_q_reference(catalog_root):
    rows = load_combined_catalog(catalog_root)
    protected = load_protected_trajectory(catalog_root, rows[0]["trajectory_id"])
    assert "q_reference" in protected.canonical
    assert protected.q_reference_start.shape == (7,)


def test_loaders_are_cwd_independent(catalog_root, monkeypatch, tmp_path):
    rows = load_combined_catalog(catalog_root)
    monkeypatch.chdir(tmp_path)
    public = load_public_trajectory(catalog_root, rows[0]["trajectory_id"])
    assert "q_reference" not in public.arrays
    protected = load_protected_trajectory(catalog_root, rows[0]["trajectory_id"])
    assert protected.q_reference_start.shape == (7,)
