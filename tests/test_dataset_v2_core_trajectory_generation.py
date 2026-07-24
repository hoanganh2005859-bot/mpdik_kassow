"""Tests for the Dataset v2 core trajectory generator (Phase 5).

Uses a small shared fixture (one anchor, all 5 shapes x 2 orientation modes = 10 trajectories,
reduced high-resolution source count) for most assertions, since the canonical count (400) -- not
the source count -- dominates reachability-validation runtime; the locked full 120-trajectory
counts are checked arithmetically against the resolved config (see
``test_locked_full_configuration_resolves_to_120``), with the real full run performed manually and
reported separately (docs/V2_IMPLEMENTATION_LOG.md), not inside this suite. Never touches Dataset
v1.
"""

import csv
import json

import jsonschema
import numpy as np
import pytest

from dataset_v2.anchor_generation import run_anchor_generation
from dataset_v2.config_templates import CORE_SHAPES, ORIENTATION_MODES
from dataset_v2.core_trajectory_generation import (
    ANTI_LEAKAGE_REPORT_NAME,
    MANIFEST_NAME,
    REACHABILITY_REPORT_NAME,
    REPORT_NAME,
    run_core_trajectory_generation,
)
from dataset_v2.locator import dataset_v2_paths
from dataset_v2.schemas import trajectory_schema
from dataset_v2.scaffold import create_dataset_v2_scaffold
from kinematics.forward_kinematics import forward_kinematics
from kinematics.model_loader import load_model_context
from pipelines.run_dataset_v2_core_trajectory_generation import main as core_trajectory_cli_main
from utils.config_loader import load_json_config
from utils.dataset_locator import ASSETS_DIR, BENCHMARKS_DIR, CONFIGS_DIR, REPO_ROOT, SCHEMAS_DIR, TRAJECTORIES_DIR
from utils.npz_utils import load_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()
ANCHOR_POOL_SMALL = 200
SOURCE_WAYPOINTS_SMALL = 450

_WATCHED_V1_DIRS = [ASSETS_DIR, BENCHMARKS_DIR, TRAJECTORIES_DIR, CONFIGS_DIR, SCHEMAS_DIR]


def _snapshot_v1():
    snapshot = {}
    for directory in _WATCHED_V1_DIRS:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path.relative_to(REPO_ROOT))] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _scaffold_with_anchors(root, master_seed=MASTER_SEED):
    create_dataset_v2_scaffold(root, master_seed=master_seed)
    run_anchor_generation(
        root,
        master_seed=master_seed,
        regular_pool_size=ANCHOR_POOL_SMALL,
        near_limit_biased_pool_size=ANCHOR_POOL_SMALL,
        singularity_biased_pool_size=ANCHOR_POOL_SMALL,
        model_context=MODEL_CONTEXT,
    )
    return root


def _generate_small(root, master_seed=MASTER_SEED, overwrite=False, **kwargs):
    kwargs.setdefault("anchor_ids", ["anchor_regular_00"])
    kwargs.setdefault("source_waypoint_count", SOURCE_WAYPOINTS_SMALL)
    return run_core_trajectory_generation(root, master_seed=master_seed, overwrite=overwrite, model_context=MODEL_CONTEXT, **kwargs)


def _read_manifest(root):
    paths = dataset_v2_paths(root)
    with open(paths.trajectories_dir / MANIFEST_NAME, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def shared_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("core_traj_gen") / "kr810_dataset_v2"
    _scaffold_with_anchors(root)
    _generate_small(root)
    return root


def test_generation_never_touches_dataset_v1(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    _scaffold_with_anchors(root)
    before = _snapshot_v1()
    _generate_small(root)
    after = _snapshot_v1()
    assert before == after, "Core trajectory v2 generation must never touch Dataset v1"


def test_generation_requires_anchor_catalog(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    with pytest.raises(RuntimeError, match="anchor catalog"):
        run_core_trajectory_generation(root, master_seed=MASTER_SEED, model_context=MODEL_CONTEXT, source_waypoint_count=SOURCE_WAYPOINTS_SMALL)


def test_generation_writes_expected_files(shared_root):
    paths = dataset_v2_paths(shared_root)
    assert (paths.trajectories_dir / MANIFEST_NAME).is_file()
    assert (paths.trajectories_dir / REPORT_NAME).is_file()
    assert (paths.trajectories_dir / REACHABILITY_REPORT_NAME).is_file()
    assert (paths.trajectories_dir / ANTI_LEAKAGE_REPORT_NAME).is_file()
    rows = _read_manifest(shared_root)
    assert len(rows) == 10
    for row in rows:
        split_dir = paths.trajectories_development_dir if row["split"] == "development" else None
        assert split_dir is not None, "smoke fixture's anchor is in the development split"
        assert (split_dir / f"{row['trajectory_id']}.npz").is_file()
        assert (split_dir / f"{row['trajectory_id']}_source.npz").is_file()


def test_ten_trajectories_per_anchor_five_shapes_two_orientations(shared_root):
    rows = _read_manifest(shared_root)
    assert len(rows) == 10
    assert {r["shape"] for r in rows} == set(CORE_SHAPES)
    assert {r["orientation_mode"] for r in rows} == set(ORIENTATION_MODES)
    assert sum(1 for r in rows if r["orientation_mode"] == "fixed") == 5
    assert sum(1 for r in rows if r["orientation_mode"] == "variable") == 5
    for shape in CORE_SHAPES:
        assert sum(1 for r in rows if r["shape"] == shape) == 2


def test_canonical_waypoint_count_is_400_regardless_of_source_override(shared_root):
    paths = dataset_v2_paths(shared_root)
    rows = _read_manifest(shared_root)
    row = rows[0]
    canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
    assert canonical["target_position"].shape == (400, 3)
    assert canonical["target_quaternion"].shape == (400, 4)
    assert int(row["canonical_waypoint_count"]) == 400


def test_source_waypoint_count_exceeds_canonical(shared_root):
    paths = dataset_v2_paths(shared_root)
    rows = _read_manifest(shared_root)
    row = rows[0]
    source = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}_source.npz")
    assert source["target_position"].shape[0] == SOURCE_WAYPOINTS_SMALL
    assert SOURCE_WAYPOINTS_SMALL > 400


def test_same_seed_is_deterministic(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _scaffold_with_anchors(root_a)
    _scaffold_with_anchors(root_b)
    _generate_small(root_a, anchor_ids=["anchor_near_singular_00"])
    _generate_small(root_b, anchor_ids=["anchor_near_singular_00"])
    rows_a = {r["trajectory_id"]: r["content_hash"] for r in _read_manifest(root_a)}
    rows_b = {r["trajectory_id"]: r["content_hash"] for r in _read_manifest(root_b)}
    assert rows_a == rows_b

    paths_a, paths_b = dataset_v2_paths(root_a), dataset_v2_paths(root_b)
    trajectory_id = next(iter(rows_a))
    npz_a = (paths_a.trajectories_development_dir / f"{trajectory_id}.npz").read_bytes()
    npz_b = (paths_b.trajectories_development_dir / f"{trajectory_id}.npz").read_bytes()
    assert npz_a == npz_b


def test_different_seed_changes_content(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _scaffold_with_anchors(root_a, master_seed=42)
    _scaffold_with_anchors(root_b, master_seed=43)
    _generate_small(root_a, master_seed=42, anchor_ids=["anchor_regular_00"], shapes=["free_form"])
    _generate_small(root_b, master_seed=43, anchor_ids=["anchor_regular_00"], shapes=["free_form"])
    rows_a = {r["trajectory_id"]: r["content_hash"] for r in _read_manifest(root_a)}
    rows_b = {r["trajectory_id"]: r["content_hash"] for r in _read_manifest(root_b)}
    assert set(rows_a) == set(rows_b)
    assert rows_a != rows_b


def test_line_starts_at_anchor_position(shared_root):
    paths = dataset_v2_paths(shared_root)
    rows = _read_manifest(shared_root)
    row = next(r for r in rows if r["shape"] == "line" and r["orientation_mode"] == "fixed")
    canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
    anchors = load_npz(paths.anchors_dir / "anchors.npz")
    anchor_idx = int(np.flatnonzero(anchors["anchor_id"] == row["anchor_id"])[0])
    fk_anchor = forward_kinematics(MODEL_CONTEXT, anchors["q"][anchor_idx])
    assert np.linalg.norm(canonical["target_position"][0] - fk_anchor.position) < 1e-6


def test_circle_is_closed(shared_root):
    paths = dataset_v2_paths(shared_root)
    rows = _read_manifest(shared_root)
    row = next(r for r in rows if r["shape"] == "circle")
    canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
    closure_err = np.linalg.norm(canonical["target_position"][-1] - canonical["target_position"][0])
    assert closure_err < 1e-3
    assert row["closed_path"] == "True"


def test_figure8_is_closed(shared_root):
    paths = dataset_v2_paths(shared_root)
    rows = _read_manifest(shared_root)
    row = next(r for r in rows if r["shape"] == "figure8")
    canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
    closure_err = np.linalg.norm(canonical["target_position"][-1] - canonical["target_position"][0])
    assert closure_err < 1e-3
    assert row["closed_path"] == "True"


def test_helix_is_open(shared_root):
    rows = _read_manifest(shared_root)
    row = next(r for r in rows if r["shape"] == "helix")
    assert row["closed_path"] == "False"


def test_free_form_control_points_and_smoothness(shared_root):
    paths = dataset_v2_paths(shared_root)
    rows = _read_manifest(shared_root)
    row = next(r for r in rows if r["shape"] == "free_form")
    source = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}_source.npz")
    assert "control_points_m" in source
    control_points = source["control_points_m"]
    assert control_points.shape == (5, 3)
    # finite curvature: no pair of consecutive control points coincide (a degenerate spline)
    diffs = np.linalg.norm(np.diff(control_points, axis=0), axis=1)
    assert np.all(diffs > 1e-6)
    # smooth: second differences of the source path stay bounded (no sharp kinks)
    second_diff = np.diff(source["target_position"], n=2, axis=0)
    assert np.all(np.linalg.norm(second_diff, axis=1) < 0.05)


def test_fixed_orientation_is_constant(shared_root):
    paths = dataset_v2_paths(shared_root)
    rows = _read_manifest(shared_root)
    row = next(r for r in rows if r["orientation_mode"] == "fixed")
    canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
    quats = canonical["target_quaternion"]
    dots = np.abs(quats @ quats[0])
    assert np.all(dots > 1.0 - 1e-6)


def test_variable_orientation_actually_varies(shared_root):
    paths = dataset_v2_paths(shared_root)
    rows = _read_manifest(shared_root)
    row = next(r for r in rows if r["orientation_mode"] == "variable")
    canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
    quats = canonical["target_quaternion"]
    dots = np.clip(np.abs(quats @ quats[0]), -1.0, 1.0)
    max_angle = float(np.max(2.0 * np.arccos(dots)))
    assert max_angle > 1e-3


def test_quaternion_wxyz_normalized(shared_root):
    paths = dataset_v2_paths(shared_root)
    for row in _read_manifest(shared_root):
        canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
        norms = np.linalg.norm(canonical["target_quaternion"], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)


def test_quaternion_sign_continuity(shared_root):
    paths = dataset_v2_paths(shared_root)
    for row in _read_manifest(shared_root):
        canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
        quats = canonical["target_quaternion"]
        dots = np.sum(quats[:-1] * quats[1:], axis=1)
        assert np.all(dots >= -1e-9)


def test_arc_length_recomputation_matches_manifest(shared_root):
    paths = dataset_v2_paths(shared_root)
    for row in _read_manifest(shared_root):
        canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
        recomputed = float(np.sum(np.linalg.norm(np.diff(canonical["target_position"], axis=0), axis=1)))
        assert abs(recomputed - float(row["arc_length_m"])) < 1e-6


def test_angular_displacement_recomputation_matches_manifest(shared_root):
    from kinematics.quaternion_utils import quaternion_geodesic_angle

    paths = dataset_v2_paths(shared_root)
    for row in _read_manifest(shared_root):
        canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
        quats = canonical["target_quaternion"]
        recomputed = sum(quaternion_geodesic_angle(quats[i - 1], quats[i]) for i in range(1, quats.shape[0]))
        assert abs(recomputed - float(row["cumulative_angular_displacement_rad"])) < 1e-6


def test_canonical_endpoints_match_source_endpoints(shared_root):
    paths = dataset_v2_paths(shared_root)
    for row in _read_manifest(shared_root):
        canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
        source = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}_source.npz")
        assert np.linalg.norm(canonical["target_position"][0] - source["target_position"][0]) < 1e-9
        assert np.linalg.norm(canonical["target_position"][-1] - source["target_position"][-1]) < 1e-9


def test_q_reference_fk_consistency(shared_root):
    """FK(q_reference) must reconstruct the target within Dataset v2's OWN strict generation
    tolerance (Phase 5.1) -- deliberately not the DLS baseline's success threshold."""
    from dataset_v2.generation_reachability import load_generation_reachability_settings

    paths = dataset_v2_paths(shared_root)
    settings = load_generation_reachability_settings(paths)
    row = _read_manifest(shared_root)[0]
    canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
    data = MODEL_CONTEXT.new_data()
    for i in range(0, 400, 40):
        fk = forward_kinematics(MODEL_CONTEXT, canonical["q_reference"][i], data=data)
        assert np.linalg.norm(fk.position - canonical["target_position"][i]) <= settings.position_tolerance_m


def test_no_waypoint_marked_unreachable(shared_root):
    paths = dataset_v2_paths(shared_root)
    for row in _read_manifest(shared_root):
        canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
        assert np.all(canonical["waypoint_reachable"])
        assert row["reachability_status"] == "validated"


def test_no_duplicate_ids_or_hashes_across_anchors(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    _scaffold_with_anchors(root)
    _generate_small(root, anchor_ids=["anchor_regular_00", "anchor_regular_01"], shapes=["line"], orientation_modes=["fixed"])
    rows = _read_manifest(root)
    ids = [r["trajectory_id"] for r in rows]
    hashes = [r["content_hash"] for r in rows]
    assert len(ids) == len(set(ids)) == 2
    assert len(hashes) == len(set(hashes)) == 2


def test_anchor_inheritance_fields(shared_root):
    paths = dataset_v2_paths(shared_root)
    anchor_manifest_row = None
    with open(paths.anchors_dir / "anchor_manifest.csv", newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["anchor_id"] == "anchor_regular_00":
                anchor_manifest_row = r
    assert anchor_manifest_row is not None
    for row in _read_manifest(shared_root):
        assert row["anchor_id"] == "anchor_regular_00"
        assert row["anchor_class"] == anchor_manifest_row["anchor_class"]
        assert row["anchor_content_hash"] == anchor_manifest_row["content_hash"]
        assert row["split"] == "development"


def test_npz_loads_with_allow_pickle_false(shared_root):
    paths = dataset_v2_paths(shared_root)
    row = _read_manifest(shared_root)[0]
    # load_npz already enforces allow_pickle=False internally; a successful load is the assertion.
    arrays = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
    assert arrays["target_position"].dtype != object


def test_schema_validates_representative_record(shared_root):
    schema = trajectory_schema()
    row = _read_manifest(shared_root)[0]
    record = {
        "trajectory_id": row["trajectory_id"],
        "family": row["family"],
        "split": row["split"],
        "shape": row["shape"],
        "orientation_mode": row["orientation_mode"],
        "anchor_id": row["anchor_id"],
        "anchor_class": row["anchor_class"],
        "source_seed": int(row["source_seed"]),
        "source_waypoint_count": int(row["source_waypoint_count"]),
        "canonical_waypoint_count": int(row["canonical_waypoint_count"]),
        "quaternion_convention": row["quaternion_convention"],
        "duration_s": float(row["duration_s"]),
        "canonical_control_period_s": float(row["canonical_control_period_s"]),
        "arc_length_m": float(row["arc_length_m"]),
        "cumulative_angular_displacement_rad": float(row["cumulative_angular_displacement_rad"]),
        "closed_path": row["closed_path"] == "True",
        "reachability_status": row["reachability_status"],
        "reachability_tolerance_position_m": float(row["reachability_tolerance_position_m"]),
        "reachability_tolerance_orientation_deg": float(row["reachability_tolerance_orientation_deg"]),
        "geometry_alternative_id": row["geometry_alternative_id"],
        "accepted_scale": float(row["accepted_scale"]),
        "scale_band": row["scale_band"],
        "generation_status": row["generation_status"],
        "content_hash": row["content_hash"],
        "sha256": row["sha256"],
    }
    jsonschema.Draft202012Validator(schema).validate(record)


def test_overwrite_protection(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    _scaffold_with_anchors(root)
    _generate_small(root)
    with pytest.raises(FileExistsError):
        _generate_small(root, overwrite=False)
    _generate_small(root, overwrite=True)  # should not raise


def test_dry_run_writes_nothing(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    _scaffold_with_anchors(root)
    result = run_core_trajectory_generation(
        root, master_seed=MASTER_SEED, anchor_ids=["anchor_regular_00"], source_waypoint_count=SOURCE_WAYPOINTS_SMALL, dry_run=True
    )
    assert result.dry_run is True
    assert result.total_trajectories == 10
    paths = dataset_v2_paths(root)
    assert not (paths.trajectories_dir / MANIFEST_NAME).exists()
    assert not any(paths.trajectories_development_dir.glob("*.npz"))


def test_cwd_independence(tmp_path, monkeypatch):
    root = tmp_path / "kr810_dataset_v2"
    _scaffold_with_anchors(root)
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)
    result = _generate_small(root)
    assert result.total_trajectories == 10


def test_unknown_shape_filter_raises(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    _scaffold_with_anchors(root)
    with pytest.raises(ValueError, match="unknown shape"):
        run_core_trajectory_generation(root, master_seed=MASTER_SEED, shapes=["not_a_shape"], source_waypoint_count=SOURCE_WAYPOINTS_SMALL)


def test_unknown_anchor_id_filter_raises(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    _scaffold_with_anchors(root)
    with pytest.raises(ValueError, match="not found"):
        run_core_trajectory_generation(root, master_seed=MASTER_SEED, anchor_ids=["not_an_anchor"], source_waypoint_count=SOURCE_WAYPOINTS_SMALL)


def test_locked_full_configuration_resolves_to_120(shared_root):
    trajectory_config = load_json_config(dataset_v2_paths(shared_root).configs_dir / "trajectory_config.json")
    assert len(CORE_SHAPES) == 5
    assert len(ORIENTATION_MODES) == 2
    anchor_count = trajectory_config["anchor_count"]
    assert anchor_count == 12
    total = len(CORE_SHAPES) * len(ORIENTATION_MODES) * anchor_count
    assert total == 120
    assert trajectory_config["total_core_trajectories"] == 120
    assert trajectory_config["canonical_waypoints_per_trajectory"] == 400
    assert trajectory_config["total_canonical_poses"] == 48000
    # split sizes: 4 anchors/split x 10 trajectories/anchor = 40/40/40
    assert 4 * (len(CORE_SHAPES) * len(ORIENTATION_MODES)) == 40
    # shape counts: 12 anchors x 2 orientation modes = 24 per shape
    assert anchor_count * len(ORIENTATION_MODES) == 24
    # orientation counts: 12 anchors x 5 shapes = 60 per orientation mode
    assert anchor_count * len(CORE_SHAPES) == 60


def test_cli_generate_and_validate_round_trip(tmp_path, capsys):
    root = tmp_path / "kr810_dataset_v2"
    _scaffold_with_anchors(root)
    exit_code = core_trajectory_cli_main(
        [
            "--dataset-root", str(root),
            "--master-seed", str(MASTER_SEED),
            "--anchor-id", "anchor_regular_00",
            "--source-waypoints", str(SOURCE_WAYPOINTS_SMALL),
        ]
    )
    assert exit_code == 0
    exit_code = core_trajectory_cli_main(
        ["--dataset-root", str(root), "--validate-only", "--anchor-id", "anchor_regular_00", "--source-waypoints", str(SOURCE_WAYPOINTS_SMALL)]
    )
    out = capsys.readouterr().out
    assert exit_code == 0, out
    assert "total=10" in out


def test_cli_overwrite_rejection(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    _scaffold_with_anchors(root)
    args = ["--dataset-root", str(root), "--master-seed", str(MASTER_SEED), "--anchor-id", "anchor_regular_00", "--source-waypoints", str(SOURCE_WAYPOINTS_SMALL)]
    assert core_trajectory_cli_main(args) == 0
    assert core_trajectory_cli_main(args) == 2
