"""Tests for Phase 5.1 reachability/scale hardening of Dataset v2 core trajectories.

The property under test throughout: **generation acceptance is defined by Dataset v2's own strict
independent-FK tolerances, not by the DLS baseline evaluation thresholds the dataset will later be
used to measure.** Also covers the geometry-alternative search, the scale-band/minimum-scale gate,
and the validator's new rejection paths.

Uses a small shared fixture (one anchor x all shapes/modes) generated once per module and copied
before tampering. Never touches Dataset v1.
"""

import csv
import json
import shutil

import numpy as np
import pytest

from dataset_v2.anchor_generation import run_anchor_generation
from dataset_v2.config_templates import (
    GENERATION_ORIENTATION_TOLERANCE_DEG,
    GENERATION_POSITION_TOLERANCE_M,
    generation_reachability_config,
)
from dataset_v2.core_trajectory_generation import (
    GEOMETRY_SEARCH_REPORT_NAME,
    MANIFEST_NAME,
    enumerate_geometry_alternatives,
    run_core_trajectory_generation,
    scale_band_of,
    search_geometry_and_scale,
)
from dataset_v2.core_trajectory_validation import validate_core_trajectories
from dataset_v2.generation_reachability import (
    GENERATION_REACHABILITY_CONFIG_NAME,
    fk_reconstruction_error,
    load_generation_reachability_settings,
    probe_settings_from,
)
from dataset_v2.locator import dataset_v2_paths
from dataset_v2.scaffold import create_dataset_v2_scaffold
from kinematics.model_loader import load_model_context
from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from utils.config_loader import load_json_config
from utils.dataset_locator import ASSETS_DIR, BENCHMARKS_DIR, CONFIGS_DIR, REPO_ROOT, SCHEMAS_DIR, TRAJECTORIES_DIR
from utils.npz_utils import load_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()
ANCHOR_POOL_SMALL = 200
SOURCE_WAYPOINTS_SMALL = 450
SMOKE_ANCHOR = "anchor_regular_00"

_WATCHED_V1_DIRS = [ASSETS_DIR, BENCHMARKS_DIR, TRAJECTORIES_DIR, CONFIGS_DIR, SCHEMAS_DIR]


def _snapshot_v1():
    snapshot = {}
    for directory in _WATCHED_V1_DIRS:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path.relative_to(REPO_ROOT))] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


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
    root = tmp_path_factory.mktemp("p51") / "kr810_dataset_v2"
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
        anchor_ids=[SMOKE_ANCHOR],
        source_waypoint_count=SOURCE_WAYPOINTS_SMALL,
        model_context=MODEL_CONTEXT,
    )
    return root


@pytest.fixture
def working_root(golden_root, tmp_path):
    dest = tmp_path / "kr810_dataset_v2"
    shutil.copytree(golden_root, dest)
    return dest


# --- 1/2: generation tolerance is independent of the DLS baseline evaluation config -------------


def test_generation_tolerance_is_not_read_from_dls_evaluation_config():
    config = generation_reachability_config()
    assert config["independence"]["reads_dls_evaluation_thresholds"] is False
    dls_config = load_json_config(REPO_ROOT / "configs" / "dls_config.json")
    assert config["position_reconstruction_tolerance_m"] < dls_config["position_success_threshold_m"]
    assert config["orientation_reconstruction_tolerance_deg"] < dls_config["orientation_success_threshold_deg"]
    # and dramatically so -- not a token difference
    assert config["position_reconstruction_tolerance_m"] <= dls_config["position_success_threshold_m"] / 50
    assert config["orientation_reconstruction_tolerance_deg"] <= dls_config["orientation_success_threshold_deg"] / 100


def _executable_string_constants(path):
    """String literals that are actually evaluated, excluding module/class/function docstrings."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                docstring_nodes.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstring_nodes
    ]


def test_generation_reachability_code_never_reads_dls_config():
    """Prose may *mention* dls_config (to explain the independence); executable code may not read it."""
    for module_name in ("generation_reachability.py", "core_trajectory_generation.py"):
        path = REPO_ROOT / "dataset_v2" / module_name
        offending = [s for s in _executable_string_constants(path) if "dls_config" in s]
        assert not offending, f"{module_name} references Dataset v1's DLS baseline config in executable code: {offending}"


def test_changing_dls_baseline_threshold_does_not_change_generation_acceptance(working_root, monkeypatch):
    """The decisive test: perturbing the DLS baseline's success thresholds must not move the
    generation acceptance boundary at all."""
    paths = dataset_v2_paths(working_root)
    settings_before = load_generation_reachability_settings(paths)

    real_loader = load_json_config

    def fake_loader(path):
        config = real_loader(path)
        if str(path).endswith("dls_config.json"):
            config = dict(config)
            config["position_success_threshold_m"] = 999.0
            config["orientation_success_threshold_deg"] = 999.0
        return config

    monkeypatch.setattr("utils.config_loader.load_json_config", fake_loader)
    settings_after = load_generation_reachability_settings(paths)
    assert settings_before.position_tolerance_m == settings_after.position_tolerance_m
    assert settings_before.orientation_tolerance_deg == settings_after.orientation_tolerance_deg

    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert report.passed, report.reasons
    assert report.statistics["generation_tolerance_position_m"] == GENERATION_POSITION_TOLERANCE_M


def test_validator_rejects_config_declaring_dls_dependence(working_root):
    paths = dataset_v2_paths(working_root)
    config_path = paths.configs_dir / GENERATION_REACHABILITY_CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["independence"]["reads_dls_evaluation_thresholds"] = True
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("reads_dls_evaluation_thresholds" in r for r in report.reasons)


def test_validator_rejects_loose_generation_tolerance(working_root):
    paths = dataset_v2_paths(working_root)
    config_path = paths.configs_dir / GENERATION_REACHABILITY_CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["position_reconstruction_tolerance_m"] = 0.01  # looser than the DLS baseline's 0.006
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("not strictly tighter" in r for r in report.reasons)


# --- 3/4/5/6: strict FK reconstruction is what decides reachability ------------------------------


def test_strict_fk_reconstruction_passes_on_generated_fixture(golden_root):
    paths = dataset_v2_paths(golden_root)
    settings = load_generation_reachability_settings(paths)
    row = _read_manifest(golden_root)[0]
    canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
    data = MODEL_CONTEXT.new_data()
    for i in range(0, canonical["target_position"].shape[0], 25):
        pos_err, orient_err = fk_reconstruction_error(
            MODEL_CONTEXT,
            canonical["q_reference"][i],
            canonical["target_position"][i],
            quaternion_wxyz_to_matrix(canonical["target_quaternion"][i]),
            data=data,
        )
        assert pos_err <= settings.position_tolerance_m
        assert orient_err <= settings.orientation_tolerance_rad


def test_validator_rejects_solver_success_flag_without_fk_consistency(working_root):
    """A waypoint flagged reachable whose q_reference does not actually reconstruct the pose must
    be rejected -- the stored success flag is never sufficient evidence."""
    paths = dataset_v2_paths(working_root)
    row = _read_manifest(working_root)[0]
    npz_path = paths.trajectories_development_dir / f"{row['trajectory_id']}.npz"

    def mutate(arrays):
        arrays["q_reference"][12] = arrays["q_reference"][12] + 0.02  # still in limits, wrong pose
        arrays["waypoint_reachable"][12] = True  # solver still claims success

    _tamper_npz(npz_path, mutate)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("exceed the strict position tolerance" in r for r in report.reasons)


def test_validator_rejects_q_reference_outside_limits(working_root):
    paths = dataset_v2_paths(working_root)
    row = _read_manifest(working_root)[0]
    npz_path = paths.trajectories_development_dir / f"{row['trajectory_id']}.npz"

    def mutate(arrays):
        arrays["q_reference"][5] = MODEL_CONTEXT.operational_upper_rad + 0.5

    _tamper_npz(npz_path, mutate)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("violates operational joint limits" in r for r in report.reasons)


def test_validator_rejects_missing_q_reference(working_root):
    paths = dataset_v2_paths(working_root)
    row = _read_manifest(working_root)[0]
    npz_path = paths.trajectories_development_dir / f"{row['trajectory_id']}.npz"

    def mutate(arrays):
        arrays.pop("q_reference")

    _tamper_npz(npz_path, mutate)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("no q_reference" in r for r in report.reasons)


def test_validator_rejects_skipped_waypoints_in_reference(working_root):
    """A truncated q_reference means some waypoint carries no reachability evidence."""
    paths = dataset_v2_paths(working_root)
    row = _read_manifest(working_root)[0]
    npz_path = paths.trajectories_development_dir / f"{row['trajectory_id']}.npz"

    def mutate(arrays):
        arrays["q_reference"] = arrays["q_reference"][:-10]

    _tamper_npz(npz_path, mutate)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("none may be skipped" in r for r in report.reasons)


# --- 7/8: both source and canonical paths carry the guarantee -----------------------------------


def test_source_path_has_full_reachability_evidence(golden_root):
    paths = dataset_v2_paths(golden_root)
    for row in _read_manifest(golden_root):
        source = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}_source.npz")
        expected = int(row["source_waypoint_count"])
        assert source["q_reference"].shape == (expected, 7)
        assert bool(np.all(source["waypoint_reachable"]))
        assert int(row["source_waypoints_reachable"]) == expected


def test_canonical_path_has_full_reachability_evidence(golden_root):
    paths = dataset_v2_paths(golden_root)
    for row in _read_manifest(golden_root):
        canonical = load_npz(paths.trajectories_development_dir / f"{row['trajectory_id']}.npz")
        expected = int(row["canonical_waypoint_count"])
        assert canonical["q_reference"].shape == (expected, 7)
        assert bool(np.all(canonical["waypoint_reachable"]))
        assert int(row["canonical_waypoints_reachable"]) == expected


def test_validator_detects_source_path_violation(working_root):
    paths = dataset_v2_paths(working_root)
    row = _read_manifest(working_root)[0]
    npz_path = paths.trajectories_development_dir / f"{row['trajectory_id']}_source.npz"

    def mutate(arrays):
        arrays["q_reference"][100] = arrays["q_reference"][100] + 0.02

    _tamper_npz(npz_path, mutate)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("source waypoint(s) exceed the strict position tolerance" in r for r in report.reasons)


# --- 10/11: deterministic geometry-alternative search --------------------------------------------


def test_geometry_alternatives_are_enumerated_deterministically(golden_root):
    paths = dataset_v2_paths(golden_root)
    config = load_json_config(paths.configs_dir / "trajectory_config.json")
    alternatives_cfg = config["geometry_alternatives"]
    # Counts expanded by Phase 5.3 (spec section H.4): circle 3->12, figure8 3->24, free_form 4->8.
    for shape, expected_count in (("line", 6), ("circle", 12), ("figure8", 24), ("helix", 6), ("free_form", 8)):
        first = enumerate_geometry_alternatives(shape, alternatives_cfg)
        second = enumerate_geometry_alternatives(shape, alternatives_cfg)
        assert [a["alternative_id"] for a in first] == [a["alternative_id"] for a in second]
        assert len(first) == expected_count
        assert len({a["alternative_id"] for a in first}) == expected_count


def test_search_prefers_largest_scale_across_alternatives(golden_root):
    """A stub where only the second alternative is reachable, and only below scale 1.0, must yield
    that alternative at the largest scale it actually achieves.

    Phase 5.3 changed the loop order to alternative-outer / scale-inner (spec section H.4), so the
    call trace now walks alternative A down its own schedule before starting B. Selection of the
    maximum accepted scale and the full tie-break chain are covered in
    ``tests/test_dataset_v2_geometry_alternatives.py``.
    """
    paths = dataset_v2_paths(golden_root)
    settings_full = load_generation_reachability_settings(paths)
    probe = probe_settings_from(paths, settings_full)
    from dataset_v2.core_trajectory_generation import load_core_trajectory_generation_settings

    core_settings = load_core_trajectory_generation_settings(paths)
    alternatives = [{"alternative_id": "A"}, {"alternative_id": "B"}]
    reachable_at = {("B", 2): True}  # B reachable only at the 3rd scale step

    calls = []

    def fake_validate(model_context, settings, q_anchor, positions, quaternions, path_seed, stop_on_first_failure=False):
        alt_id, step = positions  # stub encodes (alternative_id, scale_step)
        calls.append((alt_id, step))
        ok = reachable_at.get((alt_id, step), False)
        return {
            "all_reachable": ok,
            "first_failure_index": -1 if ok else 0,
            "waypoints_checked": 1,
            "position_errors_m": np.array([1e-7]),
            "restarts_used": np.array([0]),
        }

    import dataset_v2.core_trajectory_generation as gen_module

    original = gen_module.validate_path_strict
    gen_module.validate_path_strict = fake_validate
    try:

        def build_fn(scale, alternative):
            step = int(round(np.log(scale) / np.log(core_settings.shrink_factor))) if scale < 1.0 else 0
            return {"canonical": {"target_position": (alternative["alternative_id"], step), "target_quaternion": None}}

        result = search_geometry_and_scale(
            MODEL_CONTEXT, probe, np.zeros(7), alternatives, build_fn, core_settings, path_seed=1, label="stub"
        )
    finally:
        gen_module.validate_path_strict = original

    assert result["accepted"] is True
    assert result["alternative"]["alternative_id"] == "B"
    # alternative-outer ordering: A is walked down its own schedule first
    assert calls[0] == ("A", 0) and calls[1] == ("A", 1)
    assert result["scale"] < 1.0
    assert result["scale_reduction_reason"]


def test_generated_trajectories_record_geometry_alternative(golden_root):
    paths = dataset_v2_paths(golden_root)
    search_report = json.loads((paths.trajectories_dir / GEOMETRY_SEARCH_REPORT_NAME).read_text(encoding="utf-8"))
    by_id = {rec["trajectory_id"]: rec for rec in search_report["trajectories"]}
    for row in _read_manifest(golden_root):
        assert row["geometry_alternative_id"]
        assert row["trajectory_id"] in by_id
        record = by_id[row["trajectory_id"]]
        assert record["accepted_alternative_id"] == row["geometry_alternative_id"]
        assert record["attempts"], "every trajectory must record the alternatives it attempted"
        for attempt in record["attempts"]:
            if not attempt["reachable"]:
                assert attempt["rejection_reason"], "every rejected attempt must record why"


def test_validator_rejects_missing_geometry_alternative_metadata(working_root):
    rows = _read_manifest(working_root)
    fieldnames = list(rows[0].keys())
    rows[0] = dict(rows[0])
    rows[0]["geometry_alternative_id"] = ""
    _write_manifest(working_root, rows, fieldnames)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("no geometry_alternative_id" in r for r in report.reasons)


# --- 12/13/14: scale bands, minimum-scale gate, scale metadata ----------------------------------


def test_scale_band_labels():
    bands = [1.0, 0.75, 0.5, 0.25]
    assert scale_band_of(1.0, bands) == ">=1"
    assert scale_band_of(0.9, bands) == "[0.75,1)"
    assert scale_band_of(0.6, bands) == "[0.5,0.75)"
    assert scale_band_of(0.3, bands) == "[0.25,0.5)"
    assert scale_band_of(0.1, bands) == "<0.25"


def test_validator_reports_scale_distribution(golden_root):
    report = validate_core_trajectories(golden_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert report.passed, report.reasons
    stats = report.statistics
    for key in ("accepted_scale", "scale_band_counts", "below_0.75", "below_0.50", "below_0.25"):
        assert key in stats
    for key in ("max", "p95", "median"):
        assert key in stats["canonical_position_error_m"]
        assert key in stats["source_position_error_m"]
    assert len(stats["worst_10_trajectories"]) <= 10


def test_minimum_scale_gate_rejects_when_enforced(working_root):
    paths = dataset_v2_paths(working_root)
    config_path = paths.configs_dir / "trajectory_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["minimum_scale_gate"]["minimum_core_accepted_scale"] = 1.5  # above anything achievable
    config["minimum_scale_gate"]["enforced"] = True
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("below the locked minimum_core_accepted_scale" in r for r in report.reasons)


def test_minimum_scale_gate_is_locked_and_enforced(golden_root):
    """Phase 5.2 locked the gate at 0.50 and turned enforcement on (was null/disabled in 5.1)."""
    paths = dataset_v2_paths(golden_root)
    config = load_json_config(paths.configs_dir / "trajectory_config.json")
    gate = config["minimum_scale_gate"]
    assert gate["enforced"] is True
    assert gate["minimum_core_accepted_scale"] == 0.50
    assert gate["minimum_scale_status"] == "locked"


def test_every_generated_trajectory_meets_the_locked_gate(golden_root):
    for row in _read_manifest(golden_root):
        assert float(row["accepted_scale"]) >= 0.50, f"{row['trajectory_id']} is below the locked minimum scale"


def test_validator_rejects_trajectory_below_locked_gate(working_root):
    """A trajectory recorded below 0.50 must be a hard validation failure, not a warning."""
    rows = _read_manifest(working_root)
    fieldnames = list(rows[0].keys())
    rows[0] = dict(rows[0])
    rows[0]["accepted_scale"] = "0.3000000000"
    rows[0]["scale_band"] = "[0.25,0.5)"
    rows[0]["scale_reduction_reason"] = "synthetic below-gate value for this test"
    _write_manifest(working_root, rows, fieldnames)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("below the locked minimum_core_accepted_scale" in r for r in report.reasons)


def test_validator_detects_wrong_scale_band_metadata(working_root):
    rows = _read_manifest(working_root)
    fieldnames = list(rows[0].keys())
    rows[0] = dict(rows[0])
    rows[0]["scale_band"] = "<0.25"
    _write_manifest(working_root, rows, fieldnames)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("scale_band" in r for r in report.reasons)


def test_validator_detects_missing_scale_reduction_reason(working_root):
    rows = _read_manifest(working_root)
    fieldnames = list(rows[0].keys())
    mutated = False
    for row in rows:
        if float(row["accepted_scale"]) < 1.0:
            row["scale_reduction_reason"] = ""
            mutated = True
            break
    if not mutated:
        # the smoke fixture accepted everything at scale 1.0; synthesize the condition instead
        rows[0] = dict(rows[0])
        rows[0]["accepted_scale"] = "0.5000000000"
        rows[0]["scale_band"] = "[0.5,0.75)"
        rows[0]["scale_reduction_reason"] = ""
    _write_manifest(working_root, rows, fieldnames)
    report = validate_core_trajectories(working_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("no scale_reduction_reason" in r for r in report.reasons)


# --- 15/16: locked counts and Dataset v1 both untouched ------------------------------------------


def test_locked_count_logic_unchanged(golden_root):
    config = load_json_config(dataset_v2_paths(golden_root).configs_dir / "trajectory_config.json")
    assert config["total_core_trajectories"] == 120
    assert config["canonical_waypoints_per_trajectory"] == 400
    assert config["total_canonical_poses"] == 48000
    assert config["anchor_count"] == 12


def test_hardened_generation_never_touches_dataset_v1(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    run_anchor_generation(
        root,
        master_seed=MASTER_SEED,
        regular_pool_size=ANCHOR_POOL_SMALL,
        near_limit_biased_pool_size=ANCHOR_POOL_SMALL,
        singularity_biased_pool_size=ANCHOR_POOL_SMALL,
        model_context=MODEL_CONTEXT,
    )
    before = _snapshot_v1()
    run_core_trajectory_generation(
        root,
        master_seed=MASTER_SEED,
        anchor_ids=[SMOKE_ANCHOR],
        shapes=["line"],
        source_waypoint_count=SOURCE_WAYPOINTS_SMALL,
        model_context=MODEL_CONTEXT,
    )
    after = _snapshot_v1()
    assert before == after, "Phase 5.1 generation must never touch Dataset v1"
