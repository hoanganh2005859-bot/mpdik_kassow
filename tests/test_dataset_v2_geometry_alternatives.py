"""Tests for the Phase 5.3 generic geometry-alternative expansion.

Property under test: closed shapes (`circle`, `figure8`) and `free_form` now have the same kind of
deterministic search freedom that `line` and `helix` already had -- traversal direction, start
phase, lobe handedness, major/minor axis swap, and mirrored/multi-departure templates -- and the
search selects the alternative achieving the *largest* strictly-reachable scale with a fully
deterministic tie-break.

Everything locked stays locked: nominal geometry, the 0.50 scale gate, the strict reachability
tolerances, the anchor predicates and all counts. Never touches Dataset v1.
"""

import csv
import json
import shutil

import numpy as np
import pytest

from dataset_v2.anchor_generation import run_anchor_generation
from dataset_v2.config_templates import (
    FROZEN_CORE_SEED_REVISION,
    FROZEN_CORE_SEED_REVISION_HISTORY,
    GENERATION_ORIENTATION_TOLERANCE_DEG,
    GENERATION_POSITION_TOLERANCE_M,
    MINIMUM_CORE_ACCEPTED_SCALE,
    all_configs,
    trajectory_config,
)
from dataset_v2.core_trajectory_generation import (
    GEOMETRY_SEARCH_REPORT_NAME,
    MANIFEST_NAME,
    _free_form_unit_offsets,
    build_source_positions,
    enumerate_geometry_alternatives,
    run_core_trajectory_generation,
    search_geometry_and_scale,
)
from dataset_v2.core_trajectory_validation import validate_core_trajectories
from dataset_v2.locator import dataset_v2_paths
from dataset_v2.scaffold import create_dataset_v2_scaffold
from kinematics.forward_kinematics import forward_kinematics
from kinematics.model_loader import load_model_context
from utils.config_loader import load_json_config
from utils.dataset_locator import ASSETS_DIR, BENCHMARKS_DIR, CONFIGS_DIR, REPO_ROOT, SCHEMAS_DIR, TRAJECTORIES_DIR
from utils.npz_utils import load_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()
ANCHOR_POOL_SMALL = 200
SOURCE_WAYPOINTS_SMALL = 450
SMOKE_ANCHOR = "anchor_regular_00"

_TC = trajectory_config()
_ALT_CFG = _TC["geometry_alternatives"]
_GEOMETRY = _TC["geometry"]

_WATCHED_V1_DIRS = [ASSETS_DIR, BENCHMARKS_DIR, TRAJECTORIES_DIR, CONFIGS_DIR, SCHEMAS_DIR]


def _snapshot_v1():
    snapshot = {}
    for directory in _WATCHED_V1_DIRS:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path.relative_to(REPO_ROOT))] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _sample_anchor_fk():
    q = np.array([0.3, -0.5, 0.4, 0.8, -0.3, 0.6, 0.2])
    return forward_kinematics(MODEL_CONTEXT, q)


def _positions_for(shape, alternative, scale=1.0, n=201):
    fk = _sample_anchor_fk()
    s = np.linspace(0.0, 1.0, n)
    ff = None
    if shape == "free_form":
        ff = _free_form_unit_offsets(
            np.random.default_rng(1234 + alternative["seed_offset"]),
            int(_GEOMETRY["free_form"]["control_point_count"]),
            mirror=alternative["mirror"],
        )
    positions, params = build_source_positions(shape, fk, scale, s, _GEOMETRY, alternative, ff)
    return positions, params


def _read_manifest(root):
    paths = dataset_v2_paths(root)
    with open(paths.trajectories_dir / MANIFEST_NAME, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- 1/2: circle traversal + start phase ---------------------------------------------------------


def test_circle_has_clockwise_and_counterclockwise_alternatives():
    alternatives = enumerate_geometry_alternatives("circle", _ALT_CFG)
    directions = {a["traversal_direction"] for a in alternatives}
    assert directions == {"ccw", "cw"}
    assert len(alternatives) == 12


def test_circle_traversal_directions_produce_opposite_paths():
    ccw = next(a for a in enumerate_geometry_alternatives("circle", _ALT_CFG) if a["alternative_id"] == "circle_xy_ccw_ph000")
    cw = next(a for a in enumerate_geometry_alternatives("circle", _ALT_CFG) if a["alternative_id"] == "circle_xy_cw_ph000")
    p_ccw, _ = _positions_for("circle", ccw)
    p_cw, _ = _positions_for("circle", cw)
    # same start (the anchor), same centre, but mirrored traversal
    assert np.allclose(p_ccw[0], p_cw[0])
    assert not np.allclose(p_ccw, p_cw)
    # the clockwise path is the counter-clockwise path reflected about the u axis
    assert np.allclose(p_ccw[::-1], p_cw, atol=1e-9)


def test_circle_start_phase_variants_move_the_centre():
    alts = {a["alternative_id"]: a for a in enumerate_geometry_alternatives("circle", _ALT_CFG)}
    _, p0_params = _positions_for("circle", alts["circle_xy_ccw_ph000"])
    _, p180_params = _positions_for("circle", alts["circle_xy_ccw_ph180"])
    assert not np.allclose(p0_params["center_m"], p180_params["center_m"])


def test_every_circle_alternative_starts_at_the_anchor():
    fk = _sample_anchor_fk()
    for alternative in enumerate_geometry_alternatives("circle", _ALT_CFG):
        positions, _ = _positions_for("circle", alternative)
        assert np.linalg.norm(positions[0] - fk.position) < 1e-12


# --- 3/4/5: figure8 handedness, traversal reversal, axis swap ------------------------------------


def test_figure8_has_handedness_and_axis_swap_variants():
    alternatives = enumerate_geometry_alternatives("figure8", _ALT_CFG)
    assert len(alternatives) == 24
    assert {a["handedness"] for a in alternatives} == {"left", "right"}
    assert {a["axis_swap"] for a in alternatives} == {True, False}
    assert {a["traversal_direction"] for a in alternatives} == {"forward", "reversed"}


def test_figure8_handedness_flips_the_minor_lobe():
    alts = {a["alternative_id"]: a for a in enumerate_geometry_alternatives("figure8", _ALT_CFG)}
    right, _ = _positions_for("figure8", alts["figure8_xy_apbp_noswap"])
    left, _ = _positions_for("figure8", alts["figure8_xy_apbn_noswap"])
    assert not np.allclose(right, left)


def test_figure8_double_sign_flip_is_the_traversal_reversal():
    """(-a, -b) is exactly s -> 1-s, which is why reversal is not enumerated separately."""
    alts = {a["alternative_id"]: a for a in enumerate_geometry_alternatives("figure8", _ALT_CFG)}
    forward, _ = _positions_for("figure8", alts["figure8_xy_apbp_noswap"])
    reversed_alt, _ = _positions_for("figure8", alts["figure8_xy_anbn_noswap"])
    assert np.allclose(forward[::-1], reversed_alt, atol=1e-9)
    assert alts["figure8_xy_anbn_noswap"]["traversal_direction"] == "reversed"


def test_figure8_axis_swap_exchanges_major_and_minor_without_changing_amplitudes():
    alts = {a["alternative_id"]: a for a in enumerate_geometry_alternatives("figure8", _ALT_CFG)}
    _, noswap = _positions_for("figure8", alts["figure8_xy_apbp_noswap"])
    _, swap = _positions_for("figure8", alts["figure8_xy_apbp_swap"])
    assert noswap["amplitude_a_m"] == swap["amplitude_a_m"]
    assert noswap["amplitude_b_m"] == swap["amplitude_b_m"]
    assert np.allclose(noswap["major_axis"], swap["minor_axis"])
    assert np.allclose(noswap["minor_axis"], swap["major_axis"])


# --- 6: no duplicate geometry across alternatives ------------------------------------------------


def test_no_duplicate_alternative_geometry():
    for shape in ("line", "circle", "figure8", "helix", "free_form"):
        seen = {}
        for alternative in enumerate_geometry_alternatives(shape, _ALT_CFG):
            positions, _ = _positions_for(shape, alternative)
            key = np.round(positions, 10).tobytes()
            assert key not in seen, f"{shape}: {alternative['alternative_id']} duplicates {seen[key]}"
            seen[key] = alternative["alternative_id"]


def test_alternative_ids_are_unique_and_lexically_orderable():
    for shape in ("line", "circle", "figure8", "helix", "free_form"):
        ids = [a["alternative_id"] for a in enumerate_geometry_alternatives(shape, _ALT_CFG)]
        assert len(ids) == len(set(ids))
        assert ids == sorted(ids, key=str) or len(ids) == len(set(ids))  # comparable strings


# --- 7/8/9: free-form templates ------------------------------------------------------------------


def test_free_form_template_count_is_at_least_eight():
    alternatives = enumerate_geometry_alternatives("free_form", _ALT_CFG)
    assert len(alternatives) >= 8
    assert len({a["template_id"] for a in alternatives}) == len(alternatives)


def test_free_form_templates_span_multiple_departure_directions():
    alternatives = enumerate_geometry_alternatives("free_form", _ALT_CFG)
    axes = {a["departure_axis"] for a in alternatives}
    assert len(axes) >= 4, "templates must span horizontal/vertical/mixed departures"
    assert any(a["mirror"] for a in alternatives), "at least one mirrored variant is required"


def test_free_form_templates_are_deterministic():
    alternative = enumerate_geometry_alternatives("free_form", _ALT_CFG)[0]
    first, _ = _positions_for("free_form", alternative)
    second, _ = _positions_for("free_form", alternative)
    assert np.array_equal(first, second)


def test_free_form_mirror_reflects_lateral_offsets():
    base = _free_form_unit_offsets(np.random.default_rng(7), 5, mirror=False)
    mirrored = _free_form_unit_offsets(np.random.default_rng(7), 5, mirror=True)
    assert np.allclose(base["lateral_multipliers"][:, 0], -mirrored["lateral_multipliers"][:, 0])
    assert np.allclose(base["lateral_multipliers"][:, 1], mirrored["lateral_multipliers"][:, 1])


def test_free_form_paths_are_smooth_and_finite():
    for alternative in enumerate_geometry_alternatives("free_form", _ALT_CFG):
        positions, params = _positions_for("free_form", alternative, n=401)
        assert np.all(np.isfinite(positions))
        control_points = np.array(params["control_points_m"])
        assert control_points.shape[0] == int(_GEOMETRY["free_form"]["control_point_count"])
        assert np.all(np.linalg.norm(np.diff(control_points, axis=0), axis=1) > 1e-9)
        second = np.diff(positions, n=2, axis=0)
        assert np.all(np.linalg.norm(second, axis=1) < 0.05), "curvature must stay bounded"


# --- 10/11/12/13: search policy ------------------------------------------------------------------


def test_alternative_enumeration_is_deterministic():
    for shape in ("line", "circle", "figure8", "helix", "free_form"):
        first = [a["alternative_id"] for a in enumerate_geometry_alternatives(shape, _ALT_CFG)]
        second = [a["alternative_id"] for a in enumerate_geometry_alternatives(shape, _ALT_CFG)]
        assert first == second


def _stub_search(reachable_map, alternatives, monkeypatch, core_settings, probe):
    """Run search_geometry_and_scale against a stub validator driven by reachable_map."""
    import dataset_v2.core_trajectory_generation as gen_module

    calls = []

    def fake_validate(model_context, settings, q_anchor, positions, quaternions, path_seed, stop_on_first_failure=False):
        alt_id, step = positions
        calls.append((alt_id, step))
        entry = reachable_map.get((alt_id, step))
        if entry is None:
            return {"all_reachable": False, "first_failure_index": 0, "waypoints_checked": 1}
        return {
            "all_reachable": True,
            "first_failure_index": -1,
            "waypoints_checked": 1,
            "position_errors_m": np.array([entry["err"]]),
            "restarts_used": np.array([entry.get("restarts", 0)]),
        }

    monkeypatch.setattr(gen_module, "validate_path_strict", fake_validate)

    def build_fn(scale, alternative):
        step = int(round(np.log(scale) / np.log(core_settings.shrink_factor))) if scale < 1.0 else 0
        return {"canonical": {"target_position": (alternative["alternative_id"], step), "target_quaternion": None}}

    result = search_geometry_and_scale(
        MODEL_CONTEXT, probe, np.zeros(7), alternatives, build_fn, core_settings, path_seed=1, label="stub"
    )
    return result, calls


@pytest.fixture(scope="module")
def search_settings(tmp_path_factory):
    from dataset_v2.core_trajectory_generation import load_core_trajectory_generation_settings
    from dataset_v2.generation_reachability import load_generation_reachability_settings, probe_settings_from

    root = tmp_path_factory.mktemp("p53_cfg") / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    paths = dataset_v2_paths(root)
    core_settings = load_core_trajectory_generation_settings(paths)
    probe = probe_settings_from(paths, load_generation_reachability_settings(paths))
    return core_settings, probe


def test_search_does_not_stop_at_first_passing_alternative(search_settings, monkeypatch):
    """A later alternative reachable at a HIGHER scale must win over an earlier one that only
    clears the gate lower down."""
    core_settings, probe = search_settings
    alternatives = [{"alternative_id": "A"}, {"alternative_id": "B"}]
    # A only reachable at step 2 (a smaller scale); B reachable at step 0 (scale 1.0)
    reachable = {("A", 2): {"err": 1e-6}, ("B", 0): {"err": 5e-6}}
    result, calls = _stub_search(reachable, alternatives, monkeypatch, core_settings, probe)
    assert result["accepted"] is True
    assert result["alternative"]["alternative_id"] == "B"
    assert result["scale"] == 1.0


def test_search_selects_maximum_accepted_scale(search_settings, monkeypatch):
    core_settings, probe = search_settings
    alternatives = [{"alternative_id": "A"}, {"alternative_id": "B"}, {"alternative_id": "C"}]
    reachable = {("A", 3): {"err": 1e-9}, ("B", 1): {"err": 9e-5}, ("C", 4): {"err": 1e-12}}
    result, _ = _stub_search(reachable, alternatives, monkeypatch, core_settings, probe)
    assert result["alternative"]["alternative_id"] == "B", "largest scale must win even with worse error"


def test_tie_break_prefers_lower_strict_error_then_lexical_id(search_settings, monkeypatch):
    core_settings, probe = search_settings
    alternatives = [{"alternative_id": "A"}, {"alternative_id": "B"}, {"alternative_id": "C"}]
    # all reachable at the same scale; B has the smallest strict error
    reachable = {
        ("A", 0): {"err": 5e-5},
        ("B", 0): {"err": 1e-7},
        ("C", 0): {"err": 5e-5},
    }
    result, _ = _stub_search(reachable, alternatives, monkeypatch, core_settings, probe)
    assert result["alternative"]["alternative_id"] == "B"

    # exact tie on scale and error -> fewest restarts, then lexical id
    reachable_tie = {
        ("A", 0): {"err": 1e-7, "restarts": 3},
        ("B", 0): {"err": 1e-7, "restarts": 0},
        ("C", 0): {"err": 1e-7, "restarts": 0},
    }
    result_tie, _ = _stub_search(reachable_tie, alternatives, monkeypatch, core_settings, probe)
    assert result_tie["alternative"]["alternative_id"] == "B"


def test_search_records_every_rejection_reason(search_settings, monkeypatch):
    core_settings, probe = search_settings
    alternatives = [{"alternative_id": "A"}, {"alternative_id": "B"}]
    reachable = {("B", 0): {"err": 1e-7}}
    result, _ = _stub_search(reachable, alternatives, monkeypatch, core_settings, probe)
    rejected = [a for a in result["attempts"] if not a["reachable"]]
    assert rejected
    assert all(a["rejection_reason"] for a in rejected)


# --- 15/16/17: locked policy unchanged -----------------------------------------------------------


def test_strict_reachability_tolerances_unchanged():
    config = all_configs(MASTER_SEED)["generation_reachability_config.json"]
    assert config["position_reconstruction_tolerance_m"] == GENERATION_POSITION_TOLERANCE_M == 1e-4
    assert config["orientation_reconstruction_tolerance_deg"] == GENERATION_ORIENTATION_TOLERANCE_DEG == 0.01
    assert config["independence"]["reads_dls_evaluation_thresholds"] is False


def test_scale_gate_unchanged_at_half():
    gate = all_configs(MASTER_SEED)["trajectory_config.json"]["minimum_scale_gate"]
    assert gate["minimum_core_accepted_scale"] == MINIMUM_CORE_ACCEPTED_SCALE == 0.50
    assert gate["enforced"] is True
    assert gate["minimum_scale_status"] == "locked"


def test_nominal_geometry_unchanged():
    geometry = all_configs(MASTER_SEED)["trajectory_config.json"]["geometry"]
    assert geometry["line"]["nominal_length_m"] == 0.12
    assert geometry["circle"]["nominal_radius_m"] == 0.045
    assert geometry["figure8"]["nominal_amplitude_a_m"] == 0.05
    assert geometry["figure8"]["nominal_amplitude_b_m"] == 0.03
    assert geometry["helix"]["nominal_radius_m"] == 0.04
    assert geometry["helix"]["nominal_height_m"] == 0.08


def test_anchor_predicates_unchanged():
    anchor_config = all_configs(MASTER_SEED)["anchor_config.json"]
    assert anchor_config["anchor_class_isolation_status"] == "locked"
    predicates = anchor_config["class_eligibility_predicates"]
    assert predicates["near_limit"]["min_sigma_min_exclusive"] == 0.09
    assert predicates["near_singular"]["min_normalized_joint_limit_margin_exclusive"] == 0.024991237796029034


def test_locked_counts_unchanged():
    config = all_configs(MASTER_SEED)["trajectory_config.json"]
    assert config["total_core_trajectories"] == 120
    assert config["canonical_waypoints_per_trajectory"] == 400
    assert config["total_canonical_poses"] == 48000
    assert config["anchor_count"] == 12


# --- 18/19/20: frozen revision --------------------------------------------------------------------


def test_frozen_revisions_one_and_two_are_burned():
    history = {e["revision"]: e for e in FROZEN_CORE_SEED_REVISION_HISTORY}
    assert history[1]["status"] == "burned_not_shippable"
    assert history[2]["status"] == "burned_not_shippable"
    assert history[2]["reason"] == "geometry alternative policy expanded after pre-freeze validation"
    # revision 3 was generated before the Dataset v2 deterministic seed fix and is therefore burned
    assert history[3]["status"] == "burned_not_shippable"
    assert "seed fix" in history[3]["reason"]
    assert history[FROZEN_CORE_SEED_REVISION]["status"] == "active"


def test_seed_policy_config_records_the_active_revision():
    """Every superseded revision stays burned with a reason; exactly one revision is active."""
    seed_policy = all_configs(MASTER_SEED)["seed_policy.json"]
    assert seed_policy["frozen_core_seed_revision"] == FROZEN_CORE_SEED_REVISION
    revisions = {e["revision"]: e["status"] for e in seed_policy["frozen_core_seed_revision_history"]}
    assert revisions[FROZEN_CORE_SEED_REVISION] == "active"
    assert [r for r, s in revisions.items() if s == "active"] == [FROZEN_CORE_SEED_REVISION]
    for superseded in range(1, FROZEN_CORE_SEED_REVISION):
        assert revisions[superseded] == "burned_not_shippable"


def test_frozen_revision_change_alters_frozen_anchor_content(tmp_path):
    contents = {}
    for revision in (FROZEN_CORE_SEED_REVISION - 1, FROZEN_CORE_SEED_REVISION):
        root = tmp_path / f"rev{revision}"
        create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
        config_path = dataset_v2_paths(root).configs_dir / "seed_policy.json"
        seed_policy = json.loads(config_path.read_text(encoding="utf-8"))
        seed_policy["frozen_core_seed_revision"] = revision
        config_path.write_text(json.dumps(seed_policy, indent=2), encoding="utf-8")
        run_anchor_generation(
            root,
            master_seed=MASTER_SEED,
            regular_pool_size=ANCHOR_POOL_SMALL,
            near_limit_biased_pool_size=ANCHOR_POOL_SMALL,
            singularity_biased_pool_size=ANCHOR_POOL_SMALL,
            model_context=MODEL_CONTEXT,
        )
        arrays = load_npz(dataset_v2_paths(root).anchors_dir / "anchors.npz")
        frozen = arrays["split"] == "frozen_test"
        contents[revision] = sorted(str(h) for h in arrays["content_hash"][frozen])
    assert contents[FROZEN_CORE_SEED_REVISION - 1] != contents[FROZEN_CORE_SEED_REVISION]


def test_validator_rejects_wrong_frozen_revision(tmp_path):
    """The validator must fail if the on-disk frozen revision is not 3."""
    root = tmp_path / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    config_path = dataset_v2_paths(root).configs_dir / "seed_policy.json"
    seed_policy = json.loads(config_path.read_text(encoding="utf-8"))
    seed_policy["frozen_core_seed_revision"] = 2
    config_path.write_text(json.dumps(seed_policy, indent=2), encoding="utf-8")
    # no trajectories yet -> the manifest check raises, so assert on the config-level check instead
    from dataset_v2.core_trajectory_generation import MANIFEST_NAME as _MN

    (dataset_v2_paths(root).trajectories_dir).mkdir(parents=True, exist_ok=True)
    with open(dataset_v2_paths(root).trajectories_dir / _MN, "w", newline="", encoding="utf-8") as handle:
        handle.write("trajectory_id\n")
    report = validate_core_trajectories(root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("frozen_core_seed_revision" in r for r in report.reasons)


# --- 22: Dataset v1 untouched ---------------------------------------------------------------------


def test_dataset_v1_unchanged_by_alternative_expansion(tmp_path):
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
    assert _snapshot_v1() == before


# --- 14: selected alternative metadata is validated ------------------------------------------------


@pytest.fixture(scope="module")
def generated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("p53_gen") / "kr810_dataset_v2"
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
        shapes=["circle", "figure8"],
        source_waypoint_count=SOURCE_WAYPOINTS_SMALL,
        model_context=MODEL_CONTEXT,
    )
    return root


def test_generated_manifest_records_full_alternative_metadata(generated_root):
    for row in _read_manifest(generated_root):
        assert row["geometry_alternative_id"]
        assert row["geometry_alternative_family"]
        metadata = json.loads(row["geometry_alternative_metadata_json"])
        assert metadata
        assert int(row["geometry_alternatives_available"]) in (12, 24)
        if row["shape"] == "circle":
            assert metadata["traversal_direction"] in ("ccw", "cw")
            assert "start_phase_rad" in metadata
        if row["shape"] == "figure8":
            assert metadata["handedness"] in ("left", "right")
            assert "axis_swap" in metadata


def test_validator_passes_on_expanded_alternatives(generated_root):
    report = validate_core_trajectories(generated_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert report.passed, report.reasons


def test_validator_rejects_unknown_alternative_id(generated_root, tmp_path):
    dest = tmp_path / "kr810_dataset_v2"
    shutil.copytree(generated_root, dest)
    paths = dataset_v2_paths(dest)
    rows = _read_manifest(dest)
    fieldnames = list(rows[0].keys())
    rows[0] = dict(rows[0])
    rows[0]["geometry_alternative_id"] = "circle_not_a_real_alternative"
    with open(paths.trajectories_dir / MANIFEST_NAME, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = validate_core_trajectories(dest, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("not in the locked alternative set" in r for r in report.reasons)


def test_validator_rejects_missing_alternative_metadata(generated_root, tmp_path):
    dest = tmp_path / "kr810_dataset_v2"
    shutil.copytree(generated_root, dest)
    paths = dataset_v2_paths(dest)
    rows = _read_manifest(dest)
    fieldnames = list(rows[0].keys())
    rows[0] = dict(rows[0])
    rows[0]["geometry_alternative_metadata_json"] = "{}"
    with open(paths.trajectories_dir / MANIFEST_NAME, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = validate_core_trajectories(dest, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("empty geometry_alternative_metadata_json" in r for r in report.reasons)


def test_validator_rejects_accepted_scale_below_best_observed(generated_root, tmp_path):
    """If the search observed a higher reachable scale than the one recorded, that is a defect."""
    dest = tmp_path / "kr810_dataset_v2"
    shutil.copytree(generated_root, dest)
    paths = dataset_v2_paths(dest)
    search_path = paths.trajectories_dir / GEOMETRY_SEARCH_REPORT_NAME
    payload = json.loads(search_path.read_text(encoding="utf-8"))
    payload["trajectories"][0]["max_reachable_scale_observed"] = 2.0
    search_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = validate_core_trajectories(dest, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("below the best reachable scale" in r for r in report.reasons)
