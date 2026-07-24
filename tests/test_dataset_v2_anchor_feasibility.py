"""Tests for Phase 5.4 feasibility-aware anchor selection.

Property under test: an anchor may only enter the 12-anchor catalog if **all ten** locked
(shape, orientation_mode) combinations can be certified at ``accepted_scale >= 0.50``. Partial
feasibility is never accepted, a rejected candidate is replaced from the same class pool, and
nothing locked is weakened -- class predicates, thresholds, the scale gate, strict reachability
tolerances, nominal geometry, counts and frozen revision 4 all stay exactly as they were.

Never touches Dataset v1.
"""

import json

import numpy as np
import pytest

from dataset_v2.anchor_feasibility import (
    FeasibilitySettings,
    FeasibilityStats,
    config_fingerprints,
    feasibility_cache_key,
    load_feasibility_settings,
    locked_combinations,
    probe_anchor_feasibility,
)
from dataset_v2.anchor_generation import ANCHOR_CLASS_IDS, run_anchor_generation
from dataset_v2.anchor_validation import validate_anchors
from dataset_v2.config_templates import (
    CORE_SHAPES,
    FROZEN_CORE_SEED_REVISION,
    GENERATION_ORIENTATION_TOLERANCE_DEG,
    GENERATION_POSITION_TOLERANCE_M,
    MINIMUM_CORE_ACCEPTED_SCALE,
    ORIENTATION_MODES,
    all_configs,
)
from dataset_v2.locator import dataset_v2_paths
from dataset_v2.scaffold import create_dataset_v2_scaffold
from kinematics.model_loader import load_model_context
from utils.dataset_locator import ASSETS_DIR, BENCHMARKS_DIR, CONFIGS_DIR, REPO_ROOT, SCHEMAS_DIR, TRAJECTORIES_DIR
from utils.npz_utils import load_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()
POOL_SIZE = 300

_WATCHED_V1_DIRS = [ASSETS_DIR, BENCHMARKS_DIR, TRAJECTORIES_DIR, CONFIGS_DIR, SCHEMAS_DIR]


def _snapshot_v1():
    snapshot = {}
    for directory in _WATCHED_V1_DIRS:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path.relative_to(REPO_ROOT))] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


@pytest.fixture(scope="module")
def scaffold_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("p54_cfg") / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    return root


@pytest.fixture(scope="module")
def generated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("p54_gen") / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    result = run_anchor_generation(
        root,
        master_seed=MASTER_SEED,
        regular_pool_size=POOL_SIZE,
        near_limit_biased_pool_size=POOL_SIZE,
        singularity_biased_pool_size=POOL_SIZE,
        model_context=MODEL_CONTEXT,
    )
    return root, result


def _anchors(root):
    return load_npz(dataset_v2_paths(root).anchors_dir / "anchors.npz")


# --- policy shape ---------------------------------------------------------------------------------


def test_ten_locked_combinations():
    combos = locked_combinations()
    assert len(combos) == 10
    assert combos == [(s, m) for s in CORE_SHAPES for m in ORIENTATION_MODES]


def test_feasibility_config_is_locked_and_requires_all_ten(scaffold_root):
    settings = load_feasibility_settings(dataset_v2_paths(scaffold_root))
    assert settings.enabled is True
    assert settings.required_combinations == 10
    assert settings.minimum_passing_combinations == 10
    assert settings.minimum_scale == MINIMUM_CORE_ACCEPTED_SCALE == 0.50
    screening = all_configs(MASTER_SEED)["anchor_config.json"]["feasibility_screening"]
    assert screening["status"] == "locked"
    assert screening["partial_feasibility_accepted"] is False
    assert screening["selection_order"][:3] == ["class_predicate", "feasibility_screen", "diversity_selection"]


def test_partial_feasibility_config_is_rejected(scaffold_root, tmp_path):
    import shutil

    dest = tmp_path / "kr810_dataset_v2"
    shutil.copytree(scaffold_root, dest)
    path = dataset_v2_paths(dest).configs_dir / "anchor_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["feasibility_screening"]["partial_feasibility_accepted"] = True
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="[Pp]artial"):
        load_feasibility_settings(dataset_v2_paths(dest))


# --- 1/2/3: all-ten requirement, rejection, no partial acceptance ---------------------------------


def _settings_for(scaffold_root, **overrides):
    base = load_feasibility_settings(dataset_v2_paths(scaffold_root))
    fields = {
        "enabled": base.enabled,
        "required_combinations": base.required_combinations,
        "minimum_passing_combinations": base.minimum_passing_combinations,
        "minimum_scale": base.minimum_scale,
        "coarse_canonical_waypoints": base.coarse_canonical_waypoints,
        "coarse_source_waypoints": base.coarse_source_waypoints,
        "max_attempts_per_combination": base.max_attempts_per_combination,
        "screening_budget_per_class": base.screening_budget_per_class,
        "cache_enabled": base.cache_enabled,
    }
    fields.update(overrides)
    return FeasibilitySettings(**fields)


def _probe_with_stub(scaffold_root, monkeypatch, feasible_map, settings=None):
    """Drive probe_anchor_feasibility against a stubbed per-combination outcome map."""
    import dataset_v2.anchor_feasibility as mod

    paths = dataset_v2_paths(scaffold_root)
    settings = settings or load_feasibility_settings(paths)

    def fake_combo(model_context, q, shape, mode, s, core_settings, probe_reach, path_seed):
        label = f"{shape}_{mode}"
        ok = feasible_map.get(label, True)
        return {
            "feasible": ok,
            "attempts": 1,
            "reason": None if ok else "stubbed failure",
            "best_scale": 0.52 if ok else None,
        }

    monkeypatch.setattr(mod, "probe_combination_feasible", fake_combo)
    return mod.probe_anchor_feasibility(
        MODEL_CONTEXT, np.zeros(7), settings, object(), object(), MASTER_SEED, cache=None, cache_key=None, stats=None
    )


def test_candidate_passing_all_ten_is_feasible(scaffold_root, monkeypatch):
    verdict = _probe_with_stub(scaffold_root, monkeypatch, {})
    assert verdict["feasible"] is True
    assert verdict["combinations_passed"] == 10
    assert verdict["combinations_tested"] == 10
    assert verdict["first_failing_combination"] is None


@pytest.mark.parametrize("failing", ["line_fixed", "circle_variable", "figure8_fixed", "helix_variable", "free_form_variable"])
def test_candidate_failing_one_combination_is_rejected(scaffold_root, monkeypatch, failing):
    verdict = _probe_with_stub(scaffold_root, monkeypatch, {failing: False})
    assert verdict["feasible"] is False
    assert verdict["first_failing_combination"] == failing
    assert verdict["combinations_passed"] < 10


def test_no_partial_feasibility_acceptance(scaffold_root, monkeypatch):
    """Nine of ten passing is still a rejection."""
    verdict = _probe_with_stub(scaffold_root, monkeypatch, {"free_form_variable": False})
    assert verdict["feasible"] is False


# --- 5/6: determinism -----------------------------------------------------------------------------


def test_feasibility_results_are_deterministic(scaffold_root, monkeypatch):
    a = _probe_with_stub(scaffold_root, monkeypatch, {"helix_fixed": False})
    b = _probe_with_stub(scaffold_root, monkeypatch, {"helix_fixed": False})
    assert a["feasible"] == b["feasible"]
    assert a["first_failing_combination"] == b["first_failing_combination"]


def test_anchor_selection_is_deterministic(tmp_path):
    roots = []
    for name in ("a", "b"):
        root = tmp_path / name
        create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
        run_anchor_generation(
            root,
            master_seed=MASTER_SEED,
            regular_pool_size=POOL_SIZE,
            near_limit_biased_pool_size=POOL_SIZE,
            singularity_biased_pool_size=POOL_SIZE,
            model_context=MODEL_CONTEXT,
        )
        roots.append(root)
    a, b = _anchors(roots[0]), _anchors(roots[1])
    assert np.array_equal(a["q"], b["q"])
    assert list(a["content_hash"]) == list(b["content_hash"])
    assert list(a["split"]) == list(b["split"])


# --- 7/8/9: cache key stability and invalidation ---------------------------------------------------


def test_cache_key_is_stable_and_content_only(scaffold_root):
    paths = dataset_v2_paths(scaffold_root)
    settings = load_feasibility_settings(paths)
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    geometry_fp, reach_fp = config_fingerprints(paths)
    k1 = feasibility_cache_key(q, "modelfp", geometry_fp, reach_fp, settings)
    k2 = feasibility_cache_key(q, "modelfp", geometry_fp, reach_fp, settings)
    assert k1 == k2
    assert len(k1) == 64
    # no absolute path or timestamp can influence it: a different root with identical configs
    # yields the same fingerprints
    assert feasibility_cache_key(q.copy(), "modelfp", geometry_fp, reach_fp, settings) == k1
    # different q -> different key
    q2 = q.copy()
    q2[0] += 1e-9
    assert feasibility_cache_key(q2, "modelfp", geometry_fp, reach_fp, settings) != k1
    # different model fingerprint -> different key
    assert feasibility_cache_key(q, "otherfp", geometry_fp, reach_fp, settings) != k1


def test_cache_key_invalidates_when_geometry_config_changes(scaffold_root, tmp_path):
    import shutil

    dest = tmp_path / "kr810_dataset_v2"
    shutil.copytree(scaffold_root, dest)
    paths_a = dataset_v2_paths(scaffold_root)
    paths_b = dataset_v2_paths(dest)
    settings = load_feasibility_settings(paths_a)
    q = np.zeros(7)

    geo_a, reach_a = config_fingerprints(paths_a)
    path = paths_b.configs_dir / "trajectory_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["geometry"]["circle"]["nominal_radius_m"] = 0.099  # test-local mutation only
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    geo_b, reach_b = config_fingerprints(paths_b)

    assert geo_a != geo_b
    assert feasibility_cache_key(q, "m", geo_a, reach_a, settings) != feasibility_cache_key(q, "m", geo_b, reach_b, settings)


def test_cache_key_invalidates_when_reachability_config_changes(scaffold_root, tmp_path):
    import shutil

    dest = tmp_path / "kr810_dataset_v2"
    shutil.copytree(scaffold_root, dest)
    paths_a = dataset_v2_paths(scaffold_root)
    paths_b = dataset_v2_paths(dest)
    settings = load_feasibility_settings(paths_a)
    q = np.zeros(7)

    geo_a, reach_a = config_fingerprints(paths_a)
    path = paths_b.configs_dir / "generation_reachability_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["refinement_policy"]["max_refinement_rounds"] = 99
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    geo_b, reach_b = config_fingerprints(paths_b)

    assert reach_a != reach_b
    assert feasibility_cache_key(q, "m", geo_a, reach_a, settings) != feasibility_cache_key(q, "m", geo_b, reach_b, settings)


def test_cache_hit_avoids_recomputation(scaffold_root, monkeypatch):
    import dataset_v2.anchor_feasibility as mod

    settings = load_feasibility_settings(dataset_v2_paths(scaffold_root))
    calls = {"n": 0}

    def fake_combo(*args, **kwargs):
        calls["n"] += 1
        return {"feasible": True, "attempts": 1, "reason": None, "best_scale": 0.52}

    monkeypatch.setattr(mod, "probe_combination_feasible", fake_combo)
    cache, stats = {}, FeasibilityStats()
    for _ in range(3):
        mod.probe_anchor_feasibility(
            MODEL_CONTEXT, np.zeros(7), settings, object(), object(), MASTER_SEED, cache=cache, cache_key="k", stats=stats
        )
    assert calls["n"] == 10, "only the first call may probe; the rest must hit the cache"
    assert stats.cache_hits == 2 and stats.cache_misses == 1


# --- 10/11/12: selection operates on the feasible subset, counts preserved -------------------------


def test_selected_anchors_all_carry_full_feasibility_evidence(generated_root):
    root, _ = generated_root
    arrays = _anchors(root)
    assert np.all(arrays["feasibility_passed"])
    assert np.all(arrays["feasibility_combinations_passed"] == 10)
    assert np.all(arrays["feasibility_worst_accepted_scale"] >= MINIMUM_CORE_ACCEPTED_SCALE)


def test_diversity_selection_used_only_feasible_candidates(generated_root):
    root, result = generated_root
    screening = result.report["feasibility_screening"]
    for class_name in ("regular", "near_limit", "near_singular"):
        report = result.report["selection_report_by_class"][class_name]
        assert report["feasibility_screened"] is True
        assert report["selected_source"] == "feasible_subset_of_isolated_eligible_pool"
        assert report["feasible_count"] >= report["selected_count"]
        assert screening["candidates_screened_per_class"][class_name] >= report["feasible_count"]


def test_class_counts_remain_6_3_3(generated_root):
    _, result = generated_root
    assert result.class_counts == {"regular": 6, "near_limit": 3, "near_singular": 3}


def test_split_counts_remain_2_1_1(generated_root):
    _, result = generated_root
    assert result.split_counts == {"development": 4, "validation": 4, "frozen_test": 4}
    for class_name, per_split in (("regular", 2), ("near_limit", 1), ("near_singular", 1)):
        for split_name in ("development", "validation", "frozen_test"):
            assert result.class_split_counts[class_name][split_name] == per_split


def test_replacement_never_crosses_class_boundaries(generated_root):
    """Every selected anchor must still satisfy its own class's locked predicate -- a rejected
    candidate is replaced from the same class, never borrowed from another."""
    root, _ = generated_root
    report = validate_anchors(root, model_context=MODEL_CONTEXT, full_counts=True)
    assert report.passed, report.reasons


def test_report_records_screening_bookkeeping(generated_root):
    _, result = generated_root
    screening = result.report["feasibility_screening"]
    for key in (
        "candidates_screened_per_class",
        "candidates_passed_per_class",
        "candidates_rejected_per_class",
        "failure_histogram_by_combination",
        "cache_hits",
        "cache_misses",
        "screening_runtime_seconds",
        "selected_anchor_feasibility",
        "selected_anchor_worst_combination",
    ):
        assert key in screening
    assert len(screening["selected_anchor_feasibility"]) == 12


# --- validator rejections ---------------------------------------------------------------------------


def _copy_and_tamper(generated_root, tmp_path, mutate):
    import shutil

    root, _ = generated_root
    dest = tmp_path / "kr810_dataset_v2"
    shutil.copytree(root, dest)
    npz_path = dataset_v2_paths(dest).anchors_dir / "anchors.npz"
    arrays = dict(load_npz(npz_path))
    mutate(arrays)
    np.savez(npz_path, **arrays)
    return dest


def test_validator_rejects_missing_feasibility_evidence(generated_root, tmp_path):
    dest = _copy_and_tamper(generated_root, tmp_path, lambda a: a.pop("feasibility_passed"))
    report = validate_anchors(dest, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("feasibility" in r for r in report.reasons)


def test_validator_rejects_fewer_than_ten_combinations(generated_root, tmp_path):
    def mutate(arrays):
        arrays["feasibility_combinations_passed"][0] = 9

    dest = _copy_and_tamper(generated_root, tmp_path, mutate)
    report = validate_anchors(dest, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("passed only 9" in r for r in report.reasons)


def test_validator_rejects_below_gate_feasibility_scale(generated_root, tmp_path):
    def mutate(arrays):
        arrays["feasibility_worst_accepted_scale"][0] = 0.3

    dest = _copy_and_tamper(generated_root, tmp_path, mutate)
    report = validate_anchors(dest, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("below the locked minimum" in r for r in report.reasons)


def test_validator_rejects_mismatched_geometry_fingerprint(generated_root, tmp_path):
    import shutil

    root, _ = generated_root
    dest = tmp_path / "kr810_dataset_v2"
    shutil.copytree(root, dest)
    path = dataset_v2_paths(dest).configs_dir / "trajectory_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["geometry"]["line"]["nominal_length_m"] = 0.2
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = validate_anchors(dest, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("geometry_config_fingerprint" in r for r in report.reasons)


def test_validator_rejects_evidence_belonging_to_another_anchor(generated_root, tmp_path):
    import shutil

    root, _ = generated_root
    dest = tmp_path / "kr810_dataset_v2"
    shutil.copytree(root, dest)
    report_path = dataset_v2_paths(dest).anchors_dir / "anchor_generation_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    matrix = payload["feasibility_screening"]["selected_anchor_feasibility"]
    first = sorted(matrix)[0]
    matrix[first]["combinations_passed"] = 7
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = validate_anchors(dest, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("different anchor" in r for r in report.reasons)


# --- 13-18: nothing locked was weakened -------------------------------------------------------------


def test_scale_gate_strict_tolerances_and_geometry_unchanged():
    configs = all_configs(MASTER_SEED)
    gate = configs["trajectory_config.json"]["minimum_scale_gate"]
    assert gate["minimum_core_accepted_scale"] == 0.50 and gate["enforced"] is True
    reach = configs["generation_reachability_config.json"]
    assert reach["position_reconstruction_tolerance_m"] == GENERATION_POSITION_TOLERANCE_M == 1e-4
    assert reach["orientation_reconstruction_tolerance_deg"] == GENERATION_ORIENTATION_TOLERANCE_DEG == 0.01
    geometry = configs["trajectory_config.json"]["geometry"]
    assert geometry["line"]["nominal_length_m"] == 0.12
    assert geometry["circle"]["nominal_radius_m"] == 0.045
    assert geometry["figure8"]["nominal_amplitude_a_m"] == 0.05
    assert geometry["figure8"]["nominal_amplitude_b_m"] == 0.03
    assert geometry["helix"]["nominal_radius_m"] == 0.04
    assert geometry["helix"]["nominal_height_m"] == 0.08


def test_anchor_predicates_and_frozen_revision_unchanged():
    configs = all_configs(MASTER_SEED)
    predicates = configs["anchor_config.json"]["class_eligibility_predicates"]
    assert predicates["near_limit"]["min_sigma_min_exclusive"] == 0.09
    assert predicates["near_singular"]["min_normalized_joint_limit_margin_exclusive"] == 0.024991237796029034
    assert configs["seed_policy.json"]["frozen_core_seed_revision"] == FROZEN_CORE_SEED_REVISION == 4


def test_no_legacy_seed_derivation_returns():
    import ast
    import pathlib

    package = pathlib.Path(__file__).resolve().parent.parent / "dataset_v2"
    offenders = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "generators._common":
                if any(alias.name == "derive_seed" for alias in node.names):
                    offenders.append(path.name)
    assert not offenders, offenders


def test_dataset_v1_unchanged_by_feasibility_screening(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    before = _snapshot_v1()
    run_anchor_generation(
        root,
        master_seed=MASTER_SEED,
        regular_pool_size=POOL_SIZE,
        near_limit_biased_pool_size=POOL_SIZE,
        singularity_biased_pool_size=POOL_SIZE,
        model_context=MODEL_CONTEXT,
    )
    assert _snapshot_v1() == before
