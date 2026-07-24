"""Tests for Phase 5.2 anchor class isolation, the locked minimum-scale gate, and the
frozen-test seed reset.

Property under test: anchor classes are mutually exclusive by construction -- a near_limit anchor
is well-conditioned and a near_singular anchor is joint-space-interior -- so no anchor is
simultaneously near-limit and near-singular. Phase 5.1 measured the cost of allowing that compound
case (closed-shape trajectories forced to scale 0.12/0.20).

Never touches Dataset v1. Never changes the global difficulty definitions or the Point-IK
classification priority.
"""

import json

import numpy as np
import pytest

from dataset_v2.anchor_generation import (
    ANCHOR_CLASS_IDS,
    ANCHOR_CLASS_TOTAL_COUNTS,
    candidate_availability_report,
    class_eligibility_masks,
    run_anchor_generation,
)
from dataset_v2.anchor_validation import validate_anchors
from dataset_v2.config_templates import (
    ANCHOR_NEAR_LIMIT_MIN_SIGMA_MIN,
    CLASSIFICATION_PRIORITY_HIGHEST_FIRST,
    FROZEN_CORE_SEED_REVISION,
    MINIMUM_CORE_ACCEPTED_SCALE,
    NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD,
    NEAR_SINGULARITY_SIGMA_MIN_THRESHOLD,
    SPLITS,
    all_configs,
)
from dataset_v2.locator import dataset_v2_paths
from dataset_v2.scaffold import create_dataset_v2_scaffold
from kinematics.model_loader import load_model_context
from utils.config_loader import load_json_config
from utils.dataset_locator import ASSETS_DIR, BENCHMARKS_DIR, CONFIGS_DIR, REPO_ROOT, SCHEMAS_DIR, TRAJECTORIES_DIR
from utils.npz_utils import load_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()
POOL_SIZE = 1200

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
def generated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("p52") / "kr810_dataset_v2"
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


def _anchor_arrays(root):
    return load_npz(dataset_v2_paths(root).anchors_dir / "anchors.npz")


# --- 1/2/3: the three isolated eligibility predicates -------------------------------------------


def test_near_limit_anchors_require_sigma_min_above_floor(generated_root):
    root, _ = generated_root
    arrays = _anchor_arrays(root)
    mask = arrays["anchor_class_id"] == ANCHOR_CLASS_IDS["near_limit"]
    assert np.any(mask)
    assert np.all(arrays["sigma_min"][mask] > ANCHOR_NEAR_LIMIT_MIN_SIGMA_MIN)
    assert np.all(arrays["minimum_normalized_limit_margin"][mask] <= NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD)
    assert not np.any(arrays["is_near_singular"][mask]), "a near_limit anchor must never also be near-singular"


def test_near_singular_anchors_require_margin_above_threshold(generated_root):
    root, _ = generated_root
    arrays = _anchor_arrays(root)
    mask = arrays["anchor_class_id"] == ANCHOR_CLASS_IDS["near_singular"]
    assert np.any(mask)
    assert np.all(arrays["sigma_min"][mask] <= NEAR_SINGULARITY_SIGMA_MIN_THRESHOLD)
    assert np.all(arrays["minimum_normalized_limit_margin"][mask] > NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD)
    assert not np.any(arrays["is_near_limit"][mask]), "a near_singular anchor must never also be near-limit"


def test_regular_anchors_satisfy_regular_predicate(generated_root):
    root, _ = generated_root
    arrays = _anchor_arrays(root)
    mask = arrays["anchor_class_id"] == ANCHOR_CLASS_IDS["regular"]
    assert np.any(mask)
    assert np.all(arrays["sigma_min"][mask] > ANCHOR_NEAR_LIMIT_MIN_SIGMA_MIN)
    assert np.all(arrays["minimum_normalized_limit_margin"][mask] > NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD)


def test_class_isolation_no_anchor_is_both(generated_root):
    root, _ = generated_root
    arrays = _anchor_arrays(root)
    both = arrays["is_near_limit"] & arrays["is_near_singular"]
    assert not np.any(both), "no anchor may be simultaneously near-limit and near-singular"


def test_eligibility_masks_are_mutually_exclusive():
    """Synthetic candidates spanning the whole (sigma_min, margin) plane must never land in two
    classes at once."""
    sigma = np.array([0.001, 0.02, 0.03, 0.05, 0.09, 0.091, 0.2, 0.5])
    margin = np.array([0.0, 0.01, 0.024991237796029034, 0.025, 0.1, 0.3, 0.6, 0.9])
    sigma_grid, margin_grid = np.meshgrid(sigma, margin)
    metrics = {"sigma_min": sigma_grid.ravel(), "normalized_margin": margin_grid.ravel()}
    classification = {
        "is_near_singular": metrics["sigma_min"] <= NEAR_SINGULARITY_SIGMA_MIN_THRESHOLD,
        "is_near_limit": metrics["normalized_margin"] <= NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD,
    }
    masks = class_eligibility_masks(
        metrics,
        classification,
        NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD,
        NEAR_SINGULARITY_SIGMA_MIN_THRESHOLD,
        ANCHOR_NEAR_LIMIT_MIN_SIGMA_MIN,
    )
    stacked = np.vstack([masks["regular"], masks["near_limit"], masks["near_singular"]]).astype(int)
    assert np.all(stacked.sum(axis=0) <= 1), "eligibility predicates must be mutually exclusive"


def test_global_difficulty_definitions_and_point_ik_priority_unchanged():
    configs = all_configs(MASTER_SEED)
    thresholds = configs["difficulty_thresholds.json"]
    assert thresholds["near_joint_limit"]["threshold_normalized"] == NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD
    assert thresholds["near_singularity"]["threshold_sigma_min"] == NEAR_SINGULARITY_SIGMA_MIN_THRESHOLD
    assert thresholds["classification_priority_highest_first"] == [
        "near_singularity",
        "near_joint_limit",
        "large_orientation_change",
        "far_target",
        "medium_target",
        "near_target",
    ]
    assert CLASSIFICATION_PRIORITY_HIGHEST_FIRST[0] == "near_singularity"


def test_isolation_status_is_locked_in_config():
    anchor_config = all_configs(MASTER_SEED)["anchor_config.json"]
    assert anchor_config["anchor_class_isolation_status"] == "locked"
    predicates = anchor_config["class_eligibility_predicates"]
    assert predicates["near_limit"]["min_sigma_min_exclusive"] == ANCHOR_NEAR_LIMIT_MIN_SIGMA_MIN
    assert predicates["near_limit"]["require_not_near_singular"] is True
    assert predicates["near_singular"]["require_not_near_limit"] is True
    assert "no_fallback" in predicates


# --- 5: candidate availability ------------------------------------------------------------------


def test_candidate_availability_reports_all_required_breakdowns(generated_root):
    root, result = generated_root
    availability = result.report["candidate_availability"]
    for key in (
        "near_limit_well_conditioned_count",
        "near_limit_moderately_conditioned_count",
        "near_limit_near_singular_overlap_count",
        "near_singular_clean_count",
        "eligible_regular_count",
        "eligible_near_limit_count",
        "eligible_near_singular_count",
    ):
        assert key in availability
    assert availability["eligible_regular_count"] >= ANCHOR_CLASS_TOTAL_COUNTS["regular"]
    assert availability["eligible_near_limit_count"] >= ANCHOR_CLASS_TOTAL_COUNTS["near_limit"]
    assert availability["eligible_near_singular_count"] >= ANCHOR_CLASS_TOTAL_COUNTS["near_singular"]


def test_insufficient_candidates_fails_loudly_without_relaxing(generated_root):
    """A pool too small to satisfy the isolated predicate must raise with the availability
    breakdown -- never silently fall back to an overlapping candidate."""
    root, _ = generated_root
    with pytest.raises(ValueError, match="class-isolation predicate|eligible candidate"):
        run_anchor_generation(
            root,
            master_seed=MASTER_SEED,
            overwrite=True,
            regular_pool_size=4,
            near_limit_biased_pool_size=4,
            singularity_biased_pool_size=4,
            model_context=MODEL_CONTEXT,
        )


# --- 6/7/8: counts, splits, determinism ---------------------------------------------------------


def test_class_counts_are_6_3_3(generated_root):
    _, result = generated_root
    assert result.class_counts == {"regular": 6, "near_limit": 3, "near_singular": 3}


def test_split_counts_are_2_1_1_per_class(generated_root):
    _, result = generated_root
    assert result.split_counts == {"development": 4, "validation": 4, "frozen_test": 4}
    expected = {"regular": 2, "near_limit": 1, "near_singular": 1}
    for class_name, per_split in expected.items():
        for split_name in SPLITS:
            assert result.class_split_counts[class_name][split_name] == per_split


def test_anchor_regeneration_is_deterministic(tmp_path):
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
    a = _anchor_arrays(roots[0])
    b = _anchor_arrays(roots[1])
    assert np.array_equal(a["q"], b["q"])
    assert list(a["content_hash"]) == list(b["content_hash"])
    assert list(a["split"]) == list(b["split"])


def test_near_limit_controlling_joints_are_diverse(generated_root):
    root, _ = generated_root
    arrays = _anchor_arrays(root)
    mask = arrays["anchor_class_id"] == ANCHOR_CLASS_IDS["near_limit"]
    joints = arrays["controlling_joint_index"][mask]
    assert len(set(int(j) for j in joints)) == len(joints), "near_limit anchors should use distinct controlling joints"


# --- validator enforcement ----------------------------------------------------------------------


def test_validator_passes_on_isolated_catalog(generated_root):
    root, _ = generated_root
    report = validate_anchors(root, model_context=MODEL_CONTEXT, full_counts=True)
    assert report.passed, report.reasons


def test_validator_rejects_near_limit_anchor_that_is_near_singular(generated_root, tmp_path):
    """Hand-corrupt a near_singular anchor into the near_limit class: it violates the isolation
    predicate (sigma_min <= 0.09) and must be rejected."""
    import shutil

    root, _ = generated_root
    dest = tmp_path / "kr810_dataset_v2"
    shutil.copytree(root, dest)
    npz_path = dataset_v2_paths(dest).anchors_dir / "anchors.npz"
    arrays = dict(load_npz(npz_path))
    idx = int(np.flatnonzero(arrays["anchor_class_id"] == ANCHOR_CLASS_IDS["near_singular"])[0])
    arrays["anchor_class_id"][idx] = ANCHOR_CLASS_IDS["near_limit"]
    np.savez(npz_path, **arrays)
    report = validate_anchors(dest, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("class-isolation predicate" in r for r in report.reasons)


def test_validator_rejects_unlocked_isolation_status(generated_root, tmp_path):
    import shutil

    root, _ = generated_root
    dest = tmp_path / "kr810_dataset_v2"
    shutil.copytree(root, dest)
    config_path = dataset_v2_paths(dest).configs_dir / "anchor_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["anchor_class_isolation_status"] = "unlocked"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = validate_anchors(dest, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("anchor_class_isolation_status" in r for r in report.reasons)


# --- 9: minimum scale gate is locked at exactly 0.50 --------------------------------------------


def test_minimum_core_accepted_scale_is_locked_at_half():
    assert MINIMUM_CORE_ACCEPTED_SCALE == 0.50
    gate = all_configs(MASTER_SEED)["trajectory_config.json"]["minimum_scale_gate"]
    assert gate["minimum_core_accepted_scale"] == 0.50
    assert gate["minimum_scale_status"] == "locked"
    assert gate["enforced"] is True
    assert gate["minimum_scale_rationale"] == "Preserve at least half of nominal core trajectory geometry"


def test_nominal_geometry_unchanged():
    geometry = all_configs(MASTER_SEED)["trajectory_config.json"]["geometry"]
    assert geometry["line"]["nominal_length_m"] == 0.12
    assert geometry["circle"]["nominal_radius_m"] == 0.045
    assert geometry["figure8"]["nominal_amplitude_a_m"] == 0.05
    assert geometry["figure8"]["nominal_amplitude_b_m"] == 0.03
    assert geometry["helix"]["nominal_radius_m"] == 0.04
    assert geometry["helix"]["nominal_height_m"] == 0.08


# --- 15/16: frozen-test seed reset ---------------------------------------------------------------


def test_frozen_core_seed_revision_is_current_and_documented():
    """Phase 5.2 introduced revision 2; Phase 5.3 burned it and moved to revision 3. Every
    superseded revision must stay recorded as burned, and exactly one revision may be active."""
    seed_policy = all_configs(MASTER_SEED)["seed_policy.json"]
    assert seed_policy["frozen_core_seed_revision"] == FROZEN_CORE_SEED_REVISION
    history = seed_policy["frozen_core_seed_revision_history"]
    revisions = {entry["revision"]: entry for entry in history}
    assert revisions[1]["status"] == "burned_not_shippable"
    assert "tuned" in revisions[1]["reason"] or "observed" in revisions[1]["reason"]
    active = [r for r, e in revisions.items() if e["status"] == "active"]
    assert active == [FROZEN_CORE_SEED_REVISION]
    for superseded in range(1, FROZEN_CORE_SEED_REVISION):
        assert revisions[superseded]["status"] == "burned_not_shippable"
        assert revisions[superseded]["reason"]


def test_frozen_seed_revision_changes_frozen_content(tmp_path):
    """Bumping the frozen revision must change what frozen_test contains; the development and
    validation seed namespace is not itself revised."""
    from dataset_v2.core_trajectory_generation import run_core_trajectory_generation

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
            regular_pool_size=POOL_SIZE,
            near_limit_biased_pool_size=POOL_SIZE,
            singularity_biased_pool_size=POOL_SIZE,
            model_context=MODEL_CONTEXT,
        )
        arrays = _anchor_arrays(root)
        frozen_mask = arrays["split"] == "frozen_test"
        contents[revision] = sorted(str(h) for h in arrays["content_hash"][frozen_mask])

    assert contents[FROZEN_CORE_SEED_REVISION - 1] != contents[FROZEN_CORE_SEED_REVISION], "a new frozen seed revision must change frozen_test content"


def test_dataset_v1_unchanged_by_anchor_regeneration(tmp_path):
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
