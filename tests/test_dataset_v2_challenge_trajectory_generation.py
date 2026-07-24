"""Tests for the Dataset v2 random-challenge trajectory generator (Phase 6).

Uses small shared fixtures (a few families x splits, reduced source count / candidate pool) for
most assertions; the locked full 90-trajectory / 30-30-30 / 36,000-canonical-pose counts are
checked arithmetically against the resolved config, with the real full run performed manually and
reported separately (docs/V2_IMPLEMENTATION_LOG.md). Never touches Dataset v1.
"""

import csv
import json

import jsonschema
import numpy as np
import pytest

from dataset_v2.challenge_trajectory_generation import (
    ANTI_LEAKAGE_REPORT_NAME,
    DIVERSITY_REPORT_NAME,
    FEASIBILITY_REPORT_NAME,
    MANIFEST_NAME,
    REACHABILITY_REPORT_NAME,
    REPORT_NAME,
    run_challenge_trajectory_generation,
)
from dataset_v2.challenge_trajectory_validation import validate_challenge_trajectories
from dataset_v2.checksums import verify_checksum_manifest
from dataset_v2.config_templates import (
    CHALLENGE_FAMILIES,
    CHALLENGE_PER_FAMILY_PER_SPLIT,
    CHALLENGE_TOTAL,
    SPLITS,
    random_challenge_config,
)
from dataset_v2.locator import dataset_v2_paths
from dataset_v2.manifest import build_dataset_manifest
from dataset_v2.schemas import challenge_trajectory_schema
from dataset_v2.scaffold import create_dataset_v2_scaffold
from kinematics.model_loader import load_model_context
from pipelines.run_dataset_v2_challenge_trajectory_generation import main as challenge_cli_main
from utils.dataset_locator import ASSETS_DIR, BENCHMARKS_DIR, CONFIGS_DIR, REPO_ROOT, SCHEMAS_DIR, TRAJECTORIES_DIR
from utils.npz_utils import load_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()
SOURCE_SMALL = 451  # > 400, as the locked dataset requires (schema minimum is 401)
POOL_SMALL = 8
FLOOR_FAMILIES = ["smooth_random", "non_planar", "large_orientation"]

_WATCHED_V1_DIRS = [ASSETS_DIR, BENCHMARKS_DIR, TRAJECTORIES_DIR, CONFIGS_DIR, SCHEMAS_DIR]


def _snapshot_v1():
    snapshot = {}
    for directory in _WATCHED_V1_DIRS:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path.relative_to(REPO_ROOT))] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _generate_small(root, master_seed=MASTER_SEED, overwrite=False, **kwargs):
    kwargs.setdefault("families", FLOOR_FAMILIES)
    kwargs.setdefault("splits", ["development", "validation"])
    kwargs.setdefault("per_family_per_split", 2)
    kwargs.setdefault("source_waypoint_count", SOURCE_SMALL)
    kwargs.setdefault("candidate_pool_size", POOL_SMALL)
    return run_challenge_trajectory_generation(root, master_seed=master_seed, overwrite=overwrite, model_context=MODEL_CONTEXT, **kwargs)


def _read_manifest(root):
    paths = dataset_v2_paths(root)
    with open(paths.trajectories_dir / MANIFEST_NAME, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def shared_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("challenge_gen") / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    _generate_small(root)
    return root


# --- locked-count arithmetic (independent of any run) -----------------------------------------


def test_locked_full_configuration_resolves_to_90():
    cfg = random_challenge_config()
    total = len(cfg["families"]) * cfg["per_family_per_split"] * len(SPLITS)
    assert total == CHALLENGE_TOTAL == 90
    assert cfg["split_sizes"] == {"development": 30, "validation": 30, "frozen_test": 30}
    assert cfg["canonical_waypoints_per_trajectory"] == 400
    assert total * 400 == 36000


def test_locked_family_allocation_is_5_per_split():
    cfg = random_challenge_config()
    assert len(cfg["families"]) == 6
    assert cfg["per_family_per_split"] == CHALLENGE_PER_FAMILY_PER_SPLIT == 5
    # 6 families x 5 = 30 per split
    assert len(cfg["families"]) * cfg["per_family_per_split"] == 30


def test_combined_totals_resolve_to_210_and_84000():
    manifest = build_dataset_manifest()
    counts = manifest["counts"]
    assert counts["core_trajectories"]["total"] == 120
    assert counts["random_challenge_trajectories"]["total"] == 90
    assert counts["trajectories_total"] == 210
    assert counts["canonical_poses_total"] == 84000


# --- files, counts, reachability of a real small run -------------------------------------------


def test_generation_writes_expected_files(shared_root):
    paths = dataset_v2_paths(shared_root)
    for name in (MANIFEST_NAME, REPORT_NAME, REACHABILITY_REPORT_NAME, DIVERSITY_REPORT_NAME, FEASIBILITY_REPORT_NAME, ANTI_LEAKAGE_REPORT_NAME):
        assert (paths.trajectories_dir / name).is_file(), name


def test_small_run_counts_and_family_allocation(shared_root):
    rows = _read_manifest(shared_root)
    assert len(rows) == len(FLOOR_FAMILIES) * 2 * 2  # families x splits x per_family_per_split
    per_family_split = {}
    for r in rows:
        per_family_split.setdefault((r["challenge_family"], r["split"]), 0)
        per_family_split[(r["challenge_family"], r["split"])] += 1
    for family in FLOOR_FAMILIES:
        for split in ("development", "validation"):
            assert per_family_split[(family, split)] == 2


def test_all_six_families_generate():
    # separate quick run: all 6 families, one split, one each
    import tempfile
    import pathlib

    root = pathlib.Path(tempfile.mkdtemp()) / "ds"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    res = run_challenge_trajectory_generation(
        root, master_seed=MASTER_SEED, families=CHALLENGE_FAMILIES, splits=["development"],
        per_family_per_split=1, source_waypoint_count=SOURCE_SMALL, candidate_pool_size=POOL_SMALL, model_context=MODEL_CONTEXT,
    )
    assert res.total_trajectories == 6
    assert set(res.family_counts) == set(CHALLENGE_FAMILIES)
    assert all(v == 1 for v in res.family_counts.values())


def test_canonical_count_is_400(shared_root):
    paths = dataset_v2_paths(shared_root)
    for npz in paths.trajectories_development_dir.glob("challenge_*[0-9].npz"):
        arrays = load_npz(npz)
        assert arrays["target_position"].shape == (400, 3)
        assert arrays["target_quaternion"].shape == (400, 4)


def test_strict_source_and_canonical_reachability(shared_root):
    rows = _read_manifest(shared_root)
    for r in rows:
        assert r["reachability_status"] == "validated"
        assert int(r["canonical_waypoints_reachable"]) == 400
        assert int(r["source_waypoints_reachable"]) == SOURCE_SMALL
        assert float(r["canonical_position_reconstruction_max_m"]) <= 1e-4
        assert float(r["source_position_reconstruction_max_m"]) <= 1e-4
        assert float(r["canonical_orientation_reconstruction_max_deg"]) <= 0.01


def test_no_waypoint_skipping(shared_root):
    paths = dataset_v2_paths(shared_root)
    for split_dir in (paths.trajectories_development_dir, paths.trajectories_validation_dir):
        for npz in split_dir.glob("challenge_*[0-9].npz"):
            arrays = load_npz(npz)
            assert bool(np.all(arrays["waypoint_reachable"]))
            source = load_npz(npz.with_name(npz.stem + "_source.npz"))
            assert bool(np.all(source["waypoint_reachable"]))


# --- geometry / smoothness / coverage floors --------------------------------------------------


def test_start_pose_is_first_waypoint(shared_root):
    paths = dataset_v2_paths(shared_root)
    npz = sorted(paths.trajectories_development_dir.glob("challenge_*[0-9].npz"))[0]
    canonical = load_npz(npz)
    source = load_npz(npz.with_name(npz.stem + "_source.npz"))
    assert np.allclose(canonical["target_position"][0], source["target_position"][0], atol=1e-9)


def test_paths_are_smooth(shared_root):
    # a smooth path has small, gradually-changing consecutive deltas (no white-noise jumps)
    paths = dataset_v2_paths(shared_root)
    for npz in paths.trajectories_development_dir.glob("challenge_*[0-9].npz"):
        p = load_npz(npz)["target_position"]
        deltas = np.linalg.norm(np.diff(p, axis=0), axis=1)
        # arc-length-uniform resample -> nearly constant step; smoothness => low relative variation
        assert np.all(np.isfinite(deltas))
        assert deltas.std() / (deltas.mean() + 1e-12) < 0.5


def test_curvature_is_finite_and_bounded(shared_root):
    rows = _read_manifest(shared_root)
    cfg = random_challenge_config()["family_definitions"]
    for r in rows:
        mean_curv = float(r["mean_curvature_1_per_m"])
        max_curv = float(r["max_curvature_1_per_m"])
        assert np.isfinite(mean_curv) and np.isfinite(max_curv)
        assert mean_curv >= 0 and max_curv >= 0
        assert mean_curv <= cfg[r["challenge_family"]]["max_mean_curvature_1_per_m"]


def test_non_planar_family_meets_floor(shared_root):
    rows = _read_manifest(shared_root)
    floor = random_challenge_config()["family_definitions"]["non_planar"]["min_non_planarity"]
    non_planar_rows = [r for r in rows if r["challenge_family"] == "non_planar"]
    assert non_planar_rows
    for r in non_planar_rows:
        assert float(r["non_planarity"]) >= floor


def test_large_orientation_family_meets_floor(shared_root):
    rows = _read_manifest(shared_root)
    floor = random_challenge_config()["family_definitions"]["large_orientation"]["min_angular_displacement_rad"]
    lo_rows = [r for r in rows if r["challenge_family"] == "large_orientation"]
    assert lo_rows
    for r in lo_rows:
        assert float(r["cumulative_angular_displacement_rad"]) >= floor


def test_quaternions_normalized_and_sign_continuous(shared_root):
    paths = dataset_v2_paths(shared_root)
    for npz in paths.trajectories_development_dir.glob("challenge_*[0-9].npz"):
        q = load_npz(npz)["target_quaternion"]
        assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-6)
        assert np.all(np.sum(q[1:] * q[:-1], axis=1) >= 0.0)


# --- determinism / seed policy ----------------------------------------------------------------


def test_same_seed_byte_identical(tmp_path):
    r1 = tmp_path / "a"
    r2 = tmp_path / "b"
    for r in (r1, r2):
        create_dataset_v2_scaffold(r, master_seed=MASTER_SEED)
        run_challenge_trajectory_generation(r, master_seed=MASTER_SEED, families=["smooth_random"], splits=["development"], per_family_per_split=2, source_waypoint_count=SOURCE_SMALL, candidate_pool_size=POOL_SMALL, model_context=MODEL_CONTEXT)
    p1 = dataset_v2_paths(r1).trajectories_development_dir
    p2 = dataset_v2_paths(r2).trajectories_development_dir
    for f1 in sorted(p1.glob("challenge_*[0-9].npz")):
        f2 = p2 / f1.name
        assert load_npz(f1)["target_position"].tobytes() == load_npz(f2)["target_position"].tobytes()


def test_different_seed_changes_content(tmp_path):
    r1 = tmp_path / "a"
    r2 = tmp_path / "b"
    create_dataset_v2_scaffold(r1, master_seed=MASTER_SEED)
    create_dataset_v2_scaffold(r2, master_seed=MASTER_SEED + 1)
    run_challenge_trajectory_generation(r1, master_seed=MASTER_SEED, families=["smooth_random"], splits=["development"], per_family_per_split=2, source_waypoint_count=SOURCE_SMALL, candidate_pool_size=POOL_SMALL, model_context=MODEL_CONTEXT)
    run_challenge_trajectory_generation(r2, master_seed=MASTER_SEED + 1, families=["smooth_random"], splits=["development"], per_family_per_split=2, source_waypoint_count=SOURCE_SMALL, candidate_pool_size=POOL_SMALL, model_context=MODEL_CONTEXT)
    p1 = dataset_v2_paths(r1).trajectories_development_dir
    p2 = dataset_v2_paths(r2).trajectories_development_dir
    a = load_npz(sorted(p1.glob("challenge_*[0-9].npz"))[0])["target_position"]
    b = load_npz(sorted(p2.glob("challenge_*[0-9].npz"))[0])["target_position"]
    assert not np.array_equal(a, b)


def test_report_records_stable_seed_algorithm_and_frozen_revision(shared_root):
    paths = dataset_v2_paths(shared_root)
    report = json.loads((paths.trajectories_dir / REPORT_NAME).read_text(encoding="utf-8"))
    assert report["seed_derivation"]["frozen_challenge_seed_revision"] == 1
    seed_policy = json.loads((paths.configs_dir / "seed_policy.json").read_text(encoding="utf-8"))
    assert seed_policy["seed_algorithm_id"] == "dataset_v2/seed/sha256/v1"
    assert seed_policy["frozen_core_seed_revision"] == 4  # unchanged


def test_frozen_split_uses_separate_namespace(tmp_path):
    # dev and frozen_test path seeds for the same family/slot differ (frozen mixes in the revision)
    root = tmp_path / "ds"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    run_challenge_trajectory_generation(root, master_seed=MASTER_SEED, families=["smooth_random"], splits=["development", "frozen_test"], per_family_per_split=2, source_waypoint_count=SOURCE_SMALL, candidate_pool_size=POOL_SMALL, model_context=MODEL_CONTEXT)
    rows = _read_manifest(root)
    dev_seeds = {r["path_seed"] for r in rows if r["split"] == "development"}
    frozen_seeds = {r["path_seed"] for r in rows if r["split"] == "frozen_test"}
    assert dev_seeds and frozen_seeds
    assert dev_seeds.isdisjoint(frozen_seeds)


# --- feasibility / diversity ------------------------------------------------------------------


def test_feasibility_report_structure(shared_root):
    paths = dataset_v2_paths(shared_root)
    report = json.loads((paths.trajectories_dir / FEASIBILITY_REPORT_NAME).read_text(encoding="utf-8"))
    fams = report["families"]
    assert fams
    for fam in fams:
        assert fam["candidates_feasible"] >= fam["quota"]
        assert "full_validation_attempts" in fam


def test_diversity_selection_not_first_n(shared_root):
    # greedy farthest-point over a feasible pool larger than the selection count must reorder
    # candidates -- the ranked order is not the pool's natural 0..k-1 order.
    paths = dataset_v2_paths(shared_root)
    report = json.loads((paths.trajectories_dir / DIVERSITY_REPORT_NAME).read_text(encoding="utf-8"))
    ranked = [f["diversity_ranked_indices"] for f in report["families"]]
    assert any(order != list(range(len(order))) for order in ranked)


# --- anti-leakage / no-core-duplicate ---------------------------------------------------------


def test_anti_leakage_report_passes(shared_root):
    paths = dataset_v2_paths(shared_root)
    report = json.loads((paths.trajectories_dir / ANTI_LEAKAGE_REPORT_NAME).read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert report["collisions_found"] == 0


def test_no_cross_split_hash_or_seed_leakage(shared_root):
    rows = _read_manifest(shared_root)
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)
    dev_hashes = {r["content_hash"] for r in by_split["development"]}
    val_hashes = {r["content_hash"] for r in by_split["validation"]}
    assert dev_hashes.isdisjoint(val_hashes)
    dev_seeds = {r["path_seed"] for r in by_split["development"]}
    val_seeds = {r["path_seed"] for r in by_split["validation"]}
    assert dev_seeds.isdisjoint(val_seeds)


def test_generation_detects_duplicate_with_core(tmp_path):
    # inject a fake core manifest whose content_hash equals a challenge hash and re-run generation:
    # generation must fail loudly on the core-duplicate anti-leakage dimension.
    root = tmp_path / "ds"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    res = run_challenge_trajectory_generation(root, master_seed=MASTER_SEED, families=["smooth_random"], splits=["development"], per_family_per_split=2, source_waypoint_count=SOURCE_SMALL, candidate_pool_size=POOL_SMALL, model_context=MODEL_CONTEXT)
    rows = _read_manifest(root)
    stolen_hash = rows[0]["content_hash"]
    core_manifest = dataset_v2_paths(root).trajectories_dir / "core_trajectory_manifest.csv"
    with open(core_manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory_id", "content_hash"])
        writer.writerow(["core_line_fixed_anchor_regular_00", stolen_hash])
    with pytest.raises(ValueError, match="anti-leakage"):
        run_challenge_trajectory_generation(root, master_seed=MASTER_SEED, families=["smooth_random"], splits=["development"], per_family_per_split=2, source_waypoint_count=SOURCE_SMALL, candidate_pool_size=POOL_SMALL, model_context=MODEL_CONTEXT, overwrite=True)


# --- NPZ / schema / checksum ------------------------------------------------------------------


def test_npz_allow_pickle_false_and_no_object_dtype(shared_root):
    paths = dataset_v2_paths(shared_root)
    for npz in paths.trajectories_development_dir.glob("challenge_*.npz"):
        arrays = load_npz(npz)  # load_npz uses allow_pickle=False
        for arr in arrays.values():
            assert arr.dtype != object


def _coerce_record(row):
    return {
        "trajectory_id": row["trajectory_id"],
        "family": row["family"],
        "challenge_family": row["challenge_family"],
        "split": row["split"],
        "family_candidate_index": int(row["family_candidate_index"]),
        "source_seed": int(row["source_seed"]),
        "path_seed": int(row["path_seed"]),
        "frozen_challenge_seed_revision": int(row["frozen_challenge_seed_revision"]) if int(row["frozen_challenge_seed_revision"]) >= 1 else 1,
        "source_waypoint_count": int(row["source_waypoint_count"]),
        "canonical_waypoint_count": int(row["canonical_waypoint_count"]),
        "quaternion_convention": row["quaternion_convention"],
        "duration_s": float(row["duration_s"]),
        "canonical_control_period_s": float(row["canonical_control_period_s"]),
        "start_sigma_min": float(row["start_sigma_min"]),
        "start_normalized_limit_margin": float(row["start_normalized_limit_margin"]),
        "start_controlling_joint_index": int(row["start_controlling_joint_index"]),
        "start_content_hash": row["start_content_hash"],
        "envelope_margin_fraction": float(row["envelope_margin_fraction"]),
        "arc_length_m": float(row["arc_length_m"]),
        "cumulative_angular_displacement_rad": float(row["cumulative_angular_displacement_rad"]),
        "mean_curvature_1_per_m": float(row["mean_curvature_1_per_m"]),
        "max_curvature_1_per_m": float(row["max_curvature_1_per_m"]),
        "non_planarity": float(row["non_planarity"]),
        "reachability_status": row["reachability_status"],
        "reachability_tolerance_position_m": float(row["reachability_tolerance_position_m"]),
        "reachability_tolerance_orientation_deg": float(row["reachability_tolerance_orientation_deg"]),
        "canonical_waypoints_reachable": int(row["canonical_waypoints_reachable"]),
        "source_waypoints_reachable": int(row["source_waypoints_reachable"]),
        "generation_status": row["generation_status"],
        "content_hash": row["content_hash"],
        "sha256": row["sha256"],
        "source_sha256": row["source_sha256"],
    }


def test_schema_validates_representative_record(shared_root):
    schema = challenge_trajectory_schema()
    row = _read_manifest(shared_root)[0]
    jsonschema.Draft202012Validator(schema).validate(_coerce_record(row))


def test_checksum_manifest_includes_challenge_and_verifies(shared_root):
    mismatches = verify_checksum_manifest(shared_root)
    assert mismatches == []
    paths = dataset_v2_paths(shared_root)
    manifest = json.loads(paths.checksum_manifest_file.read_text(encoding="utf-8"))
    generated = [e["filename"] for e in manifest["categories"]["generated_data_checksum"]]
    # NPZ data files (in the split dirs) are checksummed; the manifest CSV / reports live at the
    # trajectories/ top level which the generated-data fingerprint does not scan (matches core).
    assert any("challenge_" in f and f.endswith(".npz") for f in generated)


def test_manifest_updated_with_challenge_counts(shared_root):
    paths = dataset_v2_paths(shared_root)
    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    challenge = manifest["counts"]["random_challenge_trajectories"]
    assert challenge["generated"] is True
    assert challenge["total"] == len(FLOOR_FAMILIES) * 2 * 2


# --- protection / independence / v1 ------------------------------------------------------------


def test_overwrite_protection(tmp_path):
    root = tmp_path / "ds"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    _generate_small(root)
    with pytest.raises(FileExistsError):
        _generate_small(root, overwrite=False)
    _generate_small(root, overwrite=True)


def test_dry_run_writes_nothing(tmp_path):
    root = tmp_path / "ds"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    res = run_challenge_trajectory_generation(root, master_seed=MASTER_SEED, dry_run=True, model_context=MODEL_CONTEXT)
    assert res.dry_run is True
    assert res.total_trajectories == 90
    paths = dataset_v2_paths(root)
    assert not (paths.trajectories_dir / MANIFEST_NAME).is_file()
    assert not any(paths.trajectories_development_dir.glob("challenge_*.npz"))


def test_cwd_independence(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    res = _generate_small(root, families=["smooth_random"], splits=["development"], per_family_per_split=2)
    assert res.total_trajectories == 2


def test_generation_never_touches_dataset_v1(tmp_path):
    root = tmp_path / "ds"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    before = _snapshot_v1()
    _generate_small(root, families=["smooth_random"], splits=["development"], per_family_per_split=2)
    after = _snapshot_v1()
    assert before == after


def test_unknown_family_rejected(tmp_path):
    root = tmp_path / "ds"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    with pytest.raises(ValueError, match="unknown challenge family"):
        run_challenge_trajectory_generation(root, master_seed=MASTER_SEED, families=["nope"], model_context=MODEL_CONTEXT)


def test_cli_generate_and_validate_round_trip(tmp_path, capsys):
    root = tmp_path / "ds"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    rc = challenge_cli_main(
        ["--dataset-root", str(root), "--master-seed", str(MASTER_SEED), "--family", "smooth_random", "--split", "development", "--per-family-per-split", "2", "--source-waypoints", str(SOURCE_SMALL), "--candidate-pool-size", str(POOL_SMALL)]
    )
    assert rc == 0
    rc_v = challenge_cli_main(
        ["--dataset-root", str(root), "--validate-only", "--family", "smooth_random", "--split", "development", "--per-family-per-split", "2"]
    )
    assert rc_v == 0
