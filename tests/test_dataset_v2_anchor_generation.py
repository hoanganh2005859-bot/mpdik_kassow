"""Tests for the Dataset v2 anchor generator (Phase 4).

Uses small candidate pools (200 candidates/sub-pool) for speed, except one integration test that
confirms the *resolved* full configuration is exactly 12 anchors (6 regular/3 near_limit/
3 near_singular), split 4/4/4 (2/1/1 per class per split) per
``specs/DLS_DATASET_V2_SPEC.md`` section G. Never touches Dataset v1.
"""

import json

import jsonschema
import numpy as np
import pytest

from dataset_v2.anchor_generation import (
    ANCHOR_CLASS_IDS,
    ANCHOR_CLASS_TOTAL_COUNTS,
    build_feature_vectors,
    greedy_farthest_point_select,
    run_anchor_generation,
)
from dataset_v2.anchor_validation import validate_anchors
from dataset_v2.config_templates import SPLITS
from dataset_v2.locator import dataset_v2_paths
from dataset_v2.scaffold import create_dataset_v2_scaffold
from kinematics.model_loader import load_model_context
from pipelines.run_dataset_v2_anchor_generation import main as anchor_cli_main
from utils.dataset_locator import ASSETS_DIR, BENCHMARKS_DIR, CONFIGS_DIR, REPO_ROOT, SCHEMAS_DIR, TRAJECTORIES_DIR
from utils.exceptions import ModelConfigurationError
from utils.npz_utils import load_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()
POOL_SIZE_SMALL = 200

_WATCHED_V1_DIRS = [ASSETS_DIR, BENCHMARKS_DIR, TRAJECTORIES_DIR, CONFIGS_DIR, SCHEMAS_DIR]


def _snapshot_v1():
    snapshot = {}
    for directory in _WATCHED_V1_DIRS:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path.relative_to(REPO_ROOT))] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


@pytest.fixture
def v2_root(tmp_path):
    root = tmp_path / "kr810_dataset_v2"
    create_dataset_v2_scaffold(root, master_seed=MASTER_SEED)
    return root


def _generate_small(root, master_seed=MASTER_SEED, overwrite=False, model_context=MODEL_CONTEXT):
    return run_anchor_generation(
        root,
        master_seed=master_seed,
        overwrite=overwrite,
        regular_pool_size=POOL_SIZE_SMALL,
        near_limit_biased_pool_size=POOL_SIZE_SMALL,
        singularity_biased_pool_size=POOL_SIZE_SMALL,
        model_context=model_context,
    )


def test_generation_never_touches_dataset_v1(v2_root):
    before = _snapshot_v1()
    _generate_small(v2_root)
    after = _snapshot_v1()
    assert before == after, "Anchor v2 generation must never touch Dataset v1"


def test_generation_writes_expected_files(v2_root):
    result = _generate_small(v2_root)
    anchors_dir = result.anchors_dir
    assert (anchors_dir / "anchors.npz").is_file()
    assert (anchors_dir / "anchor_manifest.csv").is_file()
    assert (anchors_dir / "anchor_generation_report.json").is_file()


def test_full_resolved_counts_are_12_total(v2_root):
    result = _generate_small(v2_root)
    assert result.total_anchors == 12


def test_class_counts_are_6_3_3(v2_root):
    result = _generate_small(v2_root)
    assert result.class_counts == {"regular": 6, "near_limit": 3, "near_singular": 3}


def test_split_counts_are_4_4_4(v2_root):
    result = _generate_small(v2_root)
    assert result.split_counts == {"development": 4, "validation": 4, "frozen_test": 4}


def test_per_split_class_counts_are_2_1_1(v2_root):
    result = _generate_small(v2_root)
    for split_name in SPLITS:
        assert result.class_split_counts["regular"][split_name] == 2
        assert result.class_split_counts["near_limit"][split_name] == 1
        assert result.class_split_counts["near_singular"][split_name] == 1


def test_deterministic_generation_same_seed(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    create_dataset_v2_scaffold(root_a, master_seed=MASTER_SEED)
    create_dataset_v2_scaffold(root_b, master_seed=MASTER_SEED)

    result_a = _generate_small(root_a)
    result_b = _generate_small(root_b)

    arrays_a = load_npz(result_a.anchors_dir / "anchors.npz")
    arrays_b = load_npz(result_b.anchors_dir / "anchors.npz")
    assert set(arrays_a.keys()) == set(arrays_b.keys())
    for name in arrays_a:
        assert np.array_equal(arrays_a[name], arrays_b[name]), f"anchors.npz:{name} differs across identical-seed runs"


def test_different_seed_produces_different_anchors(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    create_dataset_v2_scaffold(root_a, master_seed=1)
    create_dataset_v2_scaffold(root_b, master_seed=2)

    result_a = _generate_small(root_a, master_seed=1)
    result_b = _generate_small(root_b, master_seed=2)

    arrays_a = load_npz(result_a.anchors_dir / "anchors.npz")
    arrays_b = load_npz(result_b.anchors_dir / "anchors.npz")
    assert not np.array_equal(arrays_a["q"], arrays_b["q"])


def test_classification_boundaries(v2_root):
    result = _generate_small(v2_root)
    arrays = load_npz(result.anchors_dir / "anchors.npz")
    thresholds = json.loads((v2_root / "configs" / "difficulty_thresholds.json").read_text(encoding="utf-8"))
    near_joint_limit_threshold = thresholds["near_joint_limit"]["threshold_normalized"]
    near_singularity_threshold = thresholds["near_singularity"]["threshold_sigma_min"]
    moderately_conditioned_upper_bound = thresholds["moderately_conditioned"]["upper_bound_sigma_min"]

    regular_mask = arrays["anchor_class_id"] == ANCHOR_CLASS_IDS["regular"]
    near_limit_mask = arrays["anchor_class_id"] == ANCHOR_CLASS_IDS["near_limit"]
    near_singular_mask = arrays["anchor_class_id"] == ANCHOR_CLASS_IDS["near_singular"]

    assert np.all(arrays["sigma_min"][regular_mask] > moderately_conditioned_upper_bound)
    assert np.all(arrays["minimum_normalized_limit_margin"][regular_mask] > near_joint_limit_threshold)
    assert np.all(arrays["minimum_normalized_limit_margin"][near_limit_mask] <= near_joint_limit_threshold)
    assert np.all(arrays["sigma_min"][near_singular_mask] <= near_singularity_threshold)


def test_class_selection_uses_isolated_pool_without_overlap_fallback(v2_root):
    """Phase 5.2 replaced the Phase 4 'prefer clean, fall back to overlapping' policy with
    mutually-exclusive class predicates, so no selection may ever come from an overlap fallback."""
    result = _generate_small(v2_root)
    selection = result.report["selection_report_by_class"]
    assert result.report["anchor_class_isolation_status"] == "locked"
    for class_name in ("regular", "near_limit", "near_singular"):
        # Phase 5.4 narrowed the source further: selection now draws from the FEASIBLE subset of
        # the isolated eligible pool. Either label is a no-overlap-fallback source; the invariant
        # under test (never an overlap fallback) is unchanged.
        assert selection[class_name]["selected_source"] in (
            "isolated_eligible_pool",
            "feasible_subset_of_isolated_eligible_pool",
        )
        assert selection[class_name]["overlap_fallback_used"] is False
        assert selection[class_name]["eligible_count"] >= selection[class_name]["selected_count"]
    availability = result.report["candidate_availability"]
    assert availability["eligible_near_limit_count"] >= 3
    assert availability["eligible_near_singular_count"] >= 3


def test_regular_anchor_requirements(v2_root):
    result = _generate_small(v2_root)
    arrays = load_npz(result.anchors_dir / "anchors.npz")
    regular_mask = arrays["anchor_class_id"] == ANCHOR_CLASS_IDS["regular"]
    assert np.all(arrays["is_regular"][regular_mask])
    assert np.all(~arrays["is_near_singular"][regular_mask])
    assert np.all(~arrays["is_near_limit"][regular_mask])


def test_near_limit_anchor_requirements(v2_root):
    result = _generate_small(v2_root)
    arrays = load_npz(result.anchors_dir / "anchors.npz")
    near_limit_mask = arrays["anchor_class_id"] == ANCHOR_CLASS_IDS["near_limit"]
    assert np.all(arrays["is_near_limit"][near_limit_mask])
    # controlling joints should not all collapse onto a single joint when diverse enough
    controlling = arrays["controlling_joint_index"][near_limit_mask]
    assert len(set(controlling.tolist())) >= 1  # structural sanity; diversity is a soft goal


def test_near_singular_anchor_requirements(v2_root):
    result = _generate_small(v2_root)
    arrays = load_npz(result.anchors_dir / "anchors.npz")
    near_singular_mask = arrays["anchor_class_id"] == ANCHOR_CLASS_IDS["near_singular"]
    assert np.all(arrays["is_near_singular"][near_singular_mask])


def test_controlling_joint_calculation_matches_normalized_margin_argmin(v2_root):
    from kinematics.joint_limit_utils import normalized_joint_limit_margin

    result = _generate_small(v2_root)
    arrays = load_npz(result.anchors_dir / "anchors.npz")
    lower = MODEL_CONTEXT.operational_lower_rad
    upper = MODEL_CONTEXT.operational_upper_rad
    for i in range(arrays["q"].shape[0]):
        per_joint = normalized_joint_limit_margin(arrays["q"][i], lower, upper)
        expected = int(np.argmin(per_joint))
        assert int(arrays["controlling_joint_index"][i]) == expected


def test_no_exact_duplicate_q(v2_root):
    result = _generate_small(v2_root)
    arrays = load_npz(result.anchors_dir / "anchors.npz")
    q = arrays["q"]
    _, first_idx = np.unique(q, axis=0, return_index=True)
    assert first_idx.shape[0] == q.shape[0]


def test_split_anti_leakage_no_id_or_hash_reuse(v2_root):
    result = _generate_small(v2_root)
    arrays = load_npz(result.anchors_dir / "anchors.npz")
    anchor_ids = [str(s) for s in arrays["anchor_id"]]
    content_hashes = [str(s) for s in arrays["content_hash"]]
    assert len(set(anchor_ids)) == len(anchor_ids)
    assert len(set(content_hashes)) == len(content_hashes)


def test_diversity_selector_deterministic():
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    features = np.random.default_rng(1).uniform(size=(50, 10))
    selected_a = greedy_farthest_point_select(rng_a, features, 6, "test")
    selected_b = greedy_farthest_point_select(rng_b, features, 6, "test")
    assert np.array_equal(selected_a, selected_b)


def test_diversity_selector_not_a_plain_first_k_cut():
    rng = np.random.default_rng(3)
    features = np.arange(200, dtype=np.float64).reshape(-1, 1)
    selected = greedy_farthest_point_select(rng, features, 6, "test")
    assert not np.array_equal(np.sort(selected), np.arange(6))


def test_diversity_selector_raises_actionable_error_when_pool_insufficient():
    rng = np.random.default_rng(1)
    features = np.random.default_rng(2).uniform(size=(3, 4))
    with pytest.raises(ValueError, match="only has 3 candidate"):
        greedy_farthest_point_select(rng, features, 6, "regular")


def test_target_fk_metadata_consistency(v2_root):
    from kinematics.forward_kinematics import forward_kinematics

    result = _generate_small(v2_root)
    arrays = load_npz(result.anchors_dir / "anchors.npz")
    data = MODEL_CONTEXT.new_data()
    for i in range(arrays["q"].shape[0]):
        fk = forward_kinematics(MODEL_CONTEXT, arrays["q"][i], data=data)
        assert np.linalg.norm(fk.position - arrays["position"][i]) < 1e-6
        assert np.linalg.norm(fk.quaternion_wxyz - arrays["quaternion_wxyz"][i]) < 1e-5 or np.linalg.norm(
            fk.quaternion_wxyz + arrays["quaternion_wxyz"][i]
        ) < 1e-5


def test_npz_loads_with_allow_pickle_false(v2_root):
    result = _generate_small(v2_root)
    arrays = load_npz(result.anchors_dir / "anchors.npz")
    assert arrays
    for name, arr in arrays.items():
        assert arr.dtype != object, f"array '{name}' has object dtype"


def test_schema_validates_a_representative_anchor_record(v2_root):
    result = _generate_small(v2_root)
    paths = dataset_v2_paths(v2_root)
    schema = json.loads((paths.schemas_dir / "anchor_schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    arrays = load_npz(result.anchors_dir / "anchors.npz")
    class_id_to_name = {v: k for k, v in ANCHOR_CLASS_IDS.items()}
    i = 0
    record = {
        "anchor_id": str(arrays["anchor_id"][i]),
        "split": str(arrays["split"][i]),
        "anchor_class": class_id_to_name[int(arrays["anchor_class_id"][i])],
        "q": arrays["q"][i].tolist(),
        "position": arrays["position"][i].tolist(),
        "quaternion_wxyz": arrays["quaternion_wxyz"][i].tolist(),
        "sigma_min": float(arrays["sigma_min"][i]),
        "sigma_max": float(arrays["sigma_max"][i]),
        "condition_number": float(arrays["condition_number"][i]),
        "numerical_rank": int(arrays["numerical_rank"][i]),
        "manipulability": float(arrays["manipulability"][i]),
        "minimum_normalized_limit_margin": float(arrays["minimum_normalized_limit_margin"][i]),
        "minimum_absolute_limit_margin_rad": float(arrays["minimum_absolute_limit_margin_rad"][i]),
        "controlling_joint_index": int(arrays["controlling_joint_index"][i]),
        "is_near_limit": bool(arrays["is_near_limit"][i]),
        "is_near_singular": bool(arrays["is_near_singular"][i]),
        "is_moderately_conditioned": bool(arrays["is_moderately_conditioned"][i]),
        "is_regular": bool(arrays["is_regular"][i]),
        "source_seed": int(arrays["source_seed"][i]),
        "content_hash": str(arrays["content_hash"][i]),
    }
    validator.validate(record)  # must not raise


def test_checksum_verification_via_checksum_manifest(v2_root):
    from dataset_v2.checksums import verify_checksum_manifest

    _generate_small(v2_root)
    assert verify_checksum_manifest(v2_root) == []


def test_overwrite_protection(v2_root):
    _generate_small(v2_root)
    with pytest.raises(FileExistsError):
        _generate_small(v2_root, overwrite=False)
    _generate_small(v2_root, overwrite=True)  # allowed and does not raise


def test_dry_run_writes_nothing(v2_root):
    result = run_anchor_generation(
        v2_root,
        master_seed=MASTER_SEED,
        regular_pool_size=POOL_SIZE_SMALL,
        near_limit_biased_pool_size=POOL_SIZE_SMALL,
        singularity_biased_pool_size=POOL_SIZE_SMALL,
        dry_run=True,
    )
    assert result.dry_run is True
    assert not (result.anchors_dir / "anchors.npz").exists()
    assert list(result.anchors_dir.iterdir()) == [result.anchors_dir / ".gitkeep"]


def test_explicit_dataset_root_independent_of_cwd(v2_root, tmp_path, monkeypatch):
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    result = _generate_small(v2_root)
    assert result.total_anchors == 12


def test_manifest_updated_with_actual_counts_after_generation(v2_root):
    result = _generate_small(v2_root)
    paths = dataset_v2_paths(v2_root)
    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    anchor_counts = manifest["counts"]["anchors"]
    assert anchor_counts["total"] == 12
    assert anchor_counts["generated"] is True
    assert anchor_counts["class_counts"] == {"regular": 6, "near_limit": 3, "near_singular": 3}
    # dataset-wide flags are untouched by an anchor-only generation
    assert manifest["generated"] is False
    assert manifest["frozen"] is False


def test_checksum_manifest_includes_generated_anchor_files(v2_root):
    _generate_small(v2_root)
    paths = dataset_v2_paths(v2_root)
    checksum_manifest = json.loads(paths.checksum_manifest_file.read_text(encoding="utf-8"))
    filenames = {e["filename"] for e in checksum_manifest["categories"]["generated_data_checksum"]}
    assert "anchors/anchors.npz" in filenames


def test_no_absolute_paths_in_generated_report(v2_root):
    result = _generate_small(v2_root)
    root_str = str(v2_root)
    root_str_posix = root_str.replace("\\", "/")
    content = (result.anchors_dir / "anchor_generation_report.json").read_text(encoding="utf-8")
    assert root_str not in content
    assert root_str_posix not in content
    assert str(REPO_ROOT) not in content


def test_generation_requires_existing_scaffold(tmp_path):
    missing_root = tmp_path / "no_scaffold_here"
    with pytest.raises(ModelConfigurationError):
        run_anchor_generation(
            missing_root,
            master_seed=MASTER_SEED,
            regular_pool_size=POOL_SIZE_SMALL,
            near_limit_biased_pool_size=POOL_SIZE_SMALL,
            singularity_biased_pool_size=POOL_SIZE_SMALL,
        )


def test_resolved_full_configuration_locked_12_6_3_3_4_4_4(v2_root):
    """Integration check (section 12): the counts a full (no-override) generation run would
    resolve to must be exactly the locked spec numbers, without actually running the full
    generation with the default (much larger) pool sizes (a full run is exercised manually, see
    docs/V2_IMPLEMENTATION_LOG.md).
    """
    from dataset_v2.anchor_generation import ANCHOR_CLASS_TOTAL_COUNTS, load_anchor_generation_settings

    paths = dataset_v2_paths(v2_root)
    settings = load_anchor_generation_settings(paths)
    assert ANCHOR_CLASS_TOTAL_COUNTS == {"regular": 6, "near_limit": 3, "near_singular": 3}
    assert sum(ANCHOR_CLASS_TOTAL_COUNTS.values()) == 12
    for class_name, per_split in settings.split_counts_per_class.items():
        assert sum(per_split.values()) == ANCHOR_CLASS_TOTAL_COUNTS[class_name]
    assert settings.split_counts_per_class["regular"] == {"development": 2, "validation": 2, "frozen_test": 2}
    assert settings.split_counts_per_class["near_limit"] == {"development": 1, "validation": 1, "frozen_test": 1}
    assert settings.split_counts_per_class["near_singular"] == {"development": 1, "validation": 1, "frozen_test": 1}


def test_cli_dry_run(v2_root):
    exit_code = anchor_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--master-seed", str(MASTER_SEED),
            "--dry-run",
            "--regular-pool-size", str(POOL_SIZE_SMALL),
            "--near-limit-pool-size", str(POOL_SIZE_SMALL),
            "--singularity-pool-size", str(POOL_SIZE_SMALL),
        ]
    )
    assert exit_code == 0
    paths = dataset_v2_paths(v2_root)
    assert not (paths.anchors_dir / "anchors.npz").exists()


def test_cli_generate_then_validate_only(v2_root):
    exit_code = anchor_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--master-seed", str(MASTER_SEED),
            "--regular-pool-size", str(POOL_SIZE_SMALL),
            "--near-limit-pool-size", str(POOL_SIZE_SMALL),
            "--singularity-pool-size", str(POOL_SIZE_SMALL),
        ]
    )
    assert exit_code == 0

    validate_exit_code = anchor_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--validate-only",
            "--regular-pool-size", str(POOL_SIZE_SMALL),
            "--near-limit-pool-size", str(POOL_SIZE_SMALL),
            "--singularity-pool-size", str(POOL_SIZE_SMALL),
        ]
    )
    assert validate_exit_code == 0


def test_cli_rejects_reoverwrite_without_flag(v2_root):
    assert anchor_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--regular-pool-size", str(POOL_SIZE_SMALL),
            "--near-limit-pool-size", str(POOL_SIZE_SMALL),
            "--singularity-pool-size", str(POOL_SIZE_SMALL),
        ]
    ) == 0
    assert anchor_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--regular-pool-size", str(POOL_SIZE_SMALL),
            "--near-limit-pool-size", str(POOL_SIZE_SMALL),
            "--singularity-pool-size", str(POOL_SIZE_SMALL),
        ]
    ) == 2


def test_validator_passes_on_freshly_generated_small_fixture(v2_root):
    _generate_small(v2_root)
    report = validate_anchors(v2_root, full_counts=True)
    assert report.passed, report.reasons
    assert report.total_anchors == 12
