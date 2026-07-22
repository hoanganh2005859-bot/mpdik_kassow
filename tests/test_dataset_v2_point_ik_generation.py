"""Tests for the Dataset v2 Tier 1 Point-IK generator (Phase 3).

Uses a small fixture (20 samples/group, split 4/4/12) for speed, except one integration test that
confirms the *resolved* full configuration is exactly 6 groups x 1000, split 200/200/600 per
group (1200/1200/3600 overall) per ``specs/DLS_DATASET_V2_SPEC.md`` section B. Never touches
Dataset v1.
"""

import json

import jsonschema
import numpy as np
import pytest

from dataset_v2.config_templates import DIFFICULTY_GROUPS
from dataset_v2.locator import dataset_v2_paths
from dataset_v2.point_ik_generation import (
    DIFFICULTY_GROUP_IDS,
    run_point_ik_generation,
    stratified_diversity_select,
)
from dataset_v2.point_ik_validation import validate_point_ik
from dataset_v2.scaffold import create_dataset_v2_scaffold
from kinematics.model_loader import load_model_context
from pipelines.run_dataset_v2_tier1_generation import main as point_ik_cli_main
from utils.dataset_locator import ASSETS_DIR, BENCHMARKS_DIR, CONFIGS_DIR, REPO_ROOT, SCHEMAS_DIR, TRAJECTORIES_DIR
from utils.exceptions import ModelConfigurationError
from utils.npz_utils import load_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()

SAMPLES_PER_GROUP_SMALL = 20
SPLIT_SIZES_SMALL = {"development": 4, "validation": 4, "frozen_test": 12}
POOL_SIZE_SMALL = 1500

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
    return run_point_ik_generation(
        root,
        master_seed=master_seed,
        overwrite=overwrite,
        samples_per_group=SAMPLES_PER_GROUP_SMALL,
        pool_size=POOL_SIZE_SMALL,
        split_sizes_per_group=dict(SPLIT_SIZES_SMALL),
        model_context=model_context,
    )


def _load_all_splits(tier1_dir):
    return {
        "development": load_npz(tier1_dir / "development.npz"),
        "validation": load_npz(tier1_dir / "validation.npz"),
        "frozen_test": load_npz(tier1_dir / "frozen_test.npz"),
    }


def test_generation_never_touches_dataset_v1(v2_root):
    before = _snapshot_v1()
    _generate_small(v2_root)
    after = _snapshot_v1()
    assert before == after, "Point-IK v2 generation must never touch Dataset v1"


def test_generation_writes_expected_files(v2_root):
    result = _generate_small(v2_root)
    tier1_dir = result.tier1_point_ik_dir
    assert (tier1_dir / "development.npz").is_file()
    assert (tier1_dir / "validation.npz").is_file()
    assert (tier1_dir / "frozen_test.npz").is_file()
    assert (tier1_dir / "point_ik_manifest.csv").is_file()
    assert (tier1_dir / "difficulty_definition.json").is_file()
    assert (tier1_dir / "point_ik_generation_report.json").is_file()


def test_exact_group_and_split_counts_in_small_fixture(v2_root):
    result = _generate_small(v2_root)
    assert result.total_samples == SAMPLES_PER_GROUP_SMALL * len(DIFFICULTY_GROUPS)
    assert result.group_counts == {name: SAMPLES_PER_GROUP_SMALL for name in DIFFICULTY_GROUPS}
    expected_split_totals = {name: count * len(DIFFICULTY_GROUPS) for name, count in SPLIT_SIZES_SMALL.items()}
    assert result.split_counts == expected_split_totals
    for name in DIFFICULTY_GROUPS:
        assert result.group_split_counts[name] == SPLIT_SIZES_SMALL
    assert result.full_locked_counts is False


def test_npz_loads_with_allow_pickle_false(v2_root):
    result = _generate_small(v2_root)
    for arrays in _load_all_splits(result.tier1_point_ik_dir).values():
        assert arrays
        for name, arr in arrays.items():
            assert arr.dtype != object, f"array '{name}' has object dtype"


def test_operational_limit_compliance(v2_root):
    result = _generate_small(v2_root)
    lower = MODEL_CONTEXT.operational_lower_rad
    upper = MODEL_CONTEXT.operational_upper_rad
    for arrays in _load_all_splits(result.tier1_point_ik_dir).values():
        for field in ("q_initial", "q_target_reference"):
            q = arrays[field]
            assert np.all(np.isfinite(q))
            assert np.all(q >= lower) and np.all(q <= upper)


def test_quaternions_are_normalized(v2_root):
    result = _generate_small(v2_root)
    for arrays in _load_all_splits(result.tier1_point_ik_dir).values():
        for field in ("initial_quaternion_wxyz", "target_quaternion_wxyz"):
            norms = np.linalg.norm(arrays[field], axis=1)
            assert np.allclose(norms, 1.0, atol=1e-6)


def test_target_pose_matches_fk_of_q_target_reference(v2_root):
    from kinematics.forward_kinematics import forward_kinematics

    result = _generate_small(v2_root)
    data = MODEL_CONTEXT.new_data()
    for arrays in _load_all_splits(result.tier1_point_ik_dir).values():
        for i in range(arrays["sample_id"].shape[0]):
            fk = forward_kinematics(MODEL_CONTEXT, arrays["q_target_reference"][i], data=data)
            assert np.linalg.norm(fk.position - arrays["target_position"][i]) < 1e-6
            assert np.linalg.norm(fk.rotation_matrix - _to_matrix(arrays["target_quaternion_wxyz"][i])) < 1e-5


def _to_matrix(q):
    from kinematics.quaternion_utils import quaternion_wxyz_to_matrix

    return quaternion_wxyz_to_matrix(q)


def test_q_target_reference_not_used_as_q_initial(v2_root):
    """Structural check: q_target_reference is provenance-only, never identical to q_initial in
    the same record (which would mean an evaluator trivially reusing it as an initial guess)."""
    result = _generate_small(v2_root)
    for arrays in _load_all_splits(result.tier1_point_ik_dir).values():
        diffs = np.linalg.norm(arrays["q_target_reference"] - arrays["q_initial"], axis=1)
        assert np.all(diffs > 1e-6), "q_target_reference must never equal q_initial for a valid sample"

    difficulty_path = result.tier1_point_ik_dir / "difficulty_definition.json"
    definition = json.loads(difficulty_path.read_text(encoding="utf-8"))
    assert "q_target_reference_usage_policy" in definition
    assert "never" in definition["q_target_reference_usage_policy"]


def test_no_duplicate_pairs_within_or_across_splits(v2_root):
    result = _generate_small(v2_root)
    all_arrays = _load_all_splits(result.tier1_point_ik_dir)
    pair_keys = set()
    total = 0
    for arrays in all_arrays.values():
        q_i = arrays["q_initial"]
        q_t = arrays["q_target_reference"]
        for i in range(q_i.shape[0]):
            pair_keys.add(q_i[i].tobytes() + q_t[i].tobytes())
            total += 1
    assert len(pair_keys) == total


def test_no_split_leakage_in_sample_id_or_content_hash(v2_root):
    result = _generate_small(v2_root)
    all_arrays = _load_all_splits(result.tier1_point_ik_dir)
    all_sample_ids = []
    all_content_hashes = []
    for split_name, arrays in all_arrays.items():
        all_sample_ids.extend(str(s) for s in arrays["sample_id"])
        all_content_hashes.extend(str(s) for s in arrays["content_hash"])
        assert np.all(arrays["sample_id"].astype(str) != "")
        for sid in arrays["sample_id"]:
            assert str(sid).startswith(f"pik_{split_name}_")
    assert len(set(all_sample_ids)) == len(all_sample_ids)
    assert len(set(all_content_hashes)) == len(all_content_hashes)


def test_content_hash_stability_same_seed(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    create_dataset_v2_scaffold(root_a, master_seed=MASTER_SEED)
    create_dataset_v2_scaffold(root_b, master_seed=MASTER_SEED)

    result_a = _generate_small(root_a)
    result_b = _generate_small(root_b)

    arrays_a = load_npz(result_a.tier1_point_ik_dir / "development.npz")
    arrays_b = load_npz(result_b.tier1_point_ik_dir / "development.npz")
    assert np.array_equal(arrays_a["content_hash"], arrays_b["content_hash"])


def test_deterministic_generation_same_seed(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    create_dataset_v2_scaffold(root_a, master_seed=MASTER_SEED)
    create_dataset_v2_scaffold(root_b, master_seed=MASTER_SEED)

    result_a = _generate_small(root_a)
    result_b = _generate_small(root_b)

    for filename in ("development.npz", "validation.npz", "frozen_test.npz"):
        arrays_a = load_npz(result_a.tier1_point_ik_dir / filename)
        arrays_b = load_npz(result_b.tier1_point_ik_dir / filename)
        assert set(arrays_a.keys()) == set(arrays_b.keys())
        for name in arrays_a:
            assert np.array_equal(arrays_a[name], arrays_b[name]), f"{filename}:{name} differs across identical-seed runs"


def test_different_seed_produces_different_content(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    create_dataset_v2_scaffold(root_a, master_seed=1)
    create_dataset_v2_scaffold(root_b, master_seed=2)

    result_a = _generate_small(root_a, master_seed=1)
    result_b = _generate_small(root_b, master_seed=2)

    arrays_a = load_npz(result_a.tier1_point_ik_dir / "development.npz")
    arrays_b = load_npz(result_b.tier1_point_ik_dir / "development.npz")
    assert not np.array_equal(arrays_a["q_initial"], arrays_b["q_initial"])


def test_difficulty_boundary_and_priority(v2_root):
    result = _generate_small(v2_root)
    definition = json.loads((result.tier1_point_ik_dir / "difficulty_definition.json").read_text(encoding="utf-8"))
    thresholds = definition["quantile_thresholds"]
    assert thresholds["position_distance_m_low_quantile"] < thresholds["position_distance_m_high_quantile"]
    assert definition["priority_order_highest_first"] == [
        "near_singularity",
        "near_joint_limit",
        "large_orientation_change",
        "far_target",
        "medium_target",
        "near_target",
    ]

    all_arrays = _load_all_splits(result.tier1_point_ik_dir)
    for arrays in all_arrays.values():
        near_mask = arrays["difficulty_id"] == DIFFICULTY_GROUP_IDS["near_target"]
        far_mask = arrays["difficulty_id"] == DIFFICULTY_GROUP_IDS["far_target"]
        if np.any(near_mask) and np.any(far_mask):
            assert arrays["position_distance_m"][near_mask].max() <= thresholds["position_distance_m_low_quantile"] + 1e-9
            assert arrays["position_distance_m"][far_mask].min() >= thresholds["position_distance_m_high_quantile"] - 1e-9


def test_large_orientation_change_uses_so3_geodesic_not_euler(v2_root):
    result = _generate_small(v2_root)
    definition = json.loads((result.tier1_point_ik_dir / "difficulty_definition.json").read_text(encoding="utf-8"))
    top_quantile = definition["quantile_thresholds"]["orientation_distance_rad_top_quantile"]

    all_arrays = _load_all_splits(result.tier1_point_ik_dir)
    for arrays in all_arrays.values():
        mask = arrays["difficulty_id"] == DIFFICULTY_GROUP_IDS["large_orientation_change"]
        if np.any(mask):
            assert np.all(arrays["orientation_distance_rad"][mask] >= top_quantile - 1e-9)
            # orientation_distance_rad must be in [0, pi] (SO(3) geodesic range), never > pi
            assert np.all(arrays["orientation_distance_rad"][mask] <= np.pi + 1e-9)


def test_diversity_selection_is_not_a_plain_first_n_cut():
    rng_a = np.random.default_rng(123)
    idx_pool = np.arange(500)
    covariates = [np.arange(500, dtype=np.float64), (np.arange(500, dtype=np.float64) % 7)]
    selected = stratified_diversity_select(rng_a, idx_pool, covariates, n_bins=4, target_count=50, label="test_group")
    assert selected.shape[0] == 50
    assert not np.array_equal(np.sort(selected), np.arange(50)), "selection must not be a plain first-N cut"


def test_diversity_selection_raises_actionable_error_when_pool_insufficient():
    rng = np.random.default_rng(1)
    idx_pool = np.arange(10)
    covariates = [np.arange(10, dtype=np.float64)]
    with pytest.raises(ValueError, match="only has 10 eligible"):
        stratified_diversity_select(rng, idx_pool, covariates, n_bins=4, target_count=50, label="near_target")


def test_overwrite_protection(v2_root):
    _generate_small(v2_root)
    with pytest.raises(FileExistsError):
        _generate_small(v2_root, overwrite=False)
    _generate_small(v2_root, overwrite=True)  # allowed and does not raise


def test_dry_run_writes_nothing(v2_root):
    result = run_point_ik_generation(
        v2_root,
        master_seed=MASTER_SEED,
        samples_per_group=SAMPLES_PER_GROUP_SMALL,
        pool_size=POOL_SIZE_SMALL,
        split_sizes_per_group=dict(SPLIT_SIZES_SMALL),
        dry_run=True,
    )
    assert result.dry_run is True
    assert not (result.tier1_point_ik_dir / "development.npz").exists()
    assert list(result.tier1_point_ik_dir.iterdir()) == [result.tier1_point_ik_dir / ".gitkeep"]


def test_explicit_dataset_root_independent_of_cwd(v2_root, tmp_path, monkeypatch):
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    result = _generate_small(v2_root)
    assert result.total_samples == SAMPLES_PER_GROUP_SMALL * len(DIFFICULTY_GROUPS)


def test_manifest_updated_with_actual_counts_after_generation(v2_root):
    result = _generate_small(v2_root)
    paths = dataset_v2_paths(v2_root)
    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    point_ik_counts = manifest["counts"]["point_ik"]
    assert point_ik_counts["total_samples"] == result.total_samples
    assert point_ik_counts["generated"] is True
    assert point_ik_counts["full_locked_counts"] is False
    # dataset-wide flags are untouched by a Point-IK-only generation
    assert manifest["generated"] is False
    assert manifest["frozen"] is False


def test_checksum_manifest_includes_generated_point_ik_files(v2_root):
    _generate_small(v2_root)
    paths = dataset_v2_paths(v2_root)
    checksum_manifest = json.loads(paths.checksum_manifest_file.read_text(encoding="utf-8"))
    filenames = {e["filename"] for e in checksum_manifest["categories"]["generated_data_checksum"]}
    assert "tier1_point_ik/development.npz" in filenames
    assert "tier1_point_ik/validation.npz" in filenames
    assert "tier1_point_ik/frozen_test.npz" in filenames


def test_no_absolute_paths_in_generated_metadata_or_report(v2_root):
    result = _generate_small(v2_root)
    root_str = str(v2_root)
    root_str_posix = root_str.replace("\\", "/")
    for filename in ("difficulty_definition.json", "point_ik_generation_report.json"):
        content = (result.tier1_point_ik_dir / filename).read_text(encoding="utf-8")
        assert root_str not in content
        assert root_str_posix not in content
        assert str(REPO_ROOT) not in content


def test_schema_validates_a_representative_sample_record(v2_root):
    result = _generate_small(v2_root)
    paths = dataset_v2_paths(v2_root)
    schema = json.loads((paths.schemas_dir / "point_ik_schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    arrays = load_npz(result.tier1_point_ik_dir / "development.npz")
    i = 0
    record = {
        "sample_id": str(arrays["sample_id"][i]),
        "split": "development",
        "difficulty_group": DIFFICULTY_GROUPS[int(arrays["difficulty_id"][i])],
        "q_initial": arrays["q_initial"][i].tolist(),
        "q_target_reference": arrays["q_target_reference"][i].tolist(),
        "initial_position": arrays["initial_position"][i].tolist(),
        "initial_quaternion": arrays["initial_quaternion_wxyz"][i].tolist(),
        "target_position": arrays["target_position"][i].tolist(),
        "target_quaternion": arrays["target_quaternion_wxyz"][i].tolist(),
        "position_distance_m": float(arrays["position_distance_m"][i]),
        "orientation_distance_rad": float(arrays["orientation_distance_rad"][i]),
        "joint_distance_rad": float(arrays["joint_distance_rad"][i]),
        "initial_sigma_min": float(arrays["initial_sigma_min"][i]),
        "target_sigma_min": float(arrays["target_sigma_min"][i]),
        "initial_sigma_max": float(arrays["initial_sigma_max"][i]),
        "target_sigma_max": float(arrays["target_sigma_max"][i]),
        "initial_condition_number": float(arrays["initial_condition_number"][i]),
        "target_condition_number": float(arrays["target_condition_number"][i]),
        "minimum_initial_limit_margin_normalized": float(arrays["minimum_initial_limit_margin_normalized"][i]),
        "minimum_target_limit_margin_normalized": float(arrays["minimum_target_limit_margin_normalized"][i]),
        "source_seed": int(arrays["source_seed"][i]),
        "content_hash": str(arrays["content_hash"][i]),
    }
    validator.validate(record)  # must not raise


def test_generation_requires_existing_scaffold(tmp_path):
    missing_root = tmp_path / "no_scaffold_here"
    with pytest.raises(ModelConfigurationError):
        run_point_ik_generation(
            missing_root,
            master_seed=MASTER_SEED,
            samples_per_group=SAMPLES_PER_GROUP_SMALL,
            pool_size=POOL_SIZE_SMALL,
            split_sizes_per_group=dict(SPLIT_SIZES_SMALL),
        )


def test_resolved_full_counts_are_locked_6000_1000_per_group_1200_1200_3600(v2_root):
    """Integration check (section 12/13): the counts a full (no-override) generation run would
    resolve to must be exactly the locked spec numbers, without actually running the full
    6,000-sample generation (too slow for the regular test suite; see
    docs/V2_IMPLEMENTATION_LOG.md for a manual full-run confirmation).
    """
    from dataset_v2.point_ik_generation import load_point_ik_generation_settings

    paths = dataset_v2_paths(v2_root)
    settings = load_point_ik_generation_settings(paths)
    assert settings.samples_per_group == 1000
    assert len(DIFFICULTY_GROUPS) == 6
    assert settings.samples_per_group * len(DIFFICULTY_GROUPS) == 6000
    assert settings.split_sizes_per_group == {"development": 200, "validation": 200, "frozen_test": 600}
    assert sum(settings.split_sizes_per_group.values()) == 1000
    assert {name: settings.split_sizes_per_group["development"] * len(DIFFICULTY_GROUPS) for name in ["development"]}[
        "development"
    ] == 1200
    assert settings.split_sizes_per_group["frozen_test"] * len(DIFFICULTY_GROUPS) == 3600


def test_cli_dry_run(v2_root):
    exit_code = point_ik_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--master-seed", str(MASTER_SEED),
            "--dry-run",
            "--sample-limit-per-group", str(SAMPLES_PER_GROUP_SMALL),
        ]
    )
    assert exit_code == 0
    paths = dataset_v2_paths(v2_root)
    assert not (paths.tier1_point_ik_dir / "development.npz").exists()


def test_cli_generate_then_validate_only(v2_root):
    exit_code = point_ik_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--master-seed", str(MASTER_SEED),
            "--sample-limit-per-group", str(SAMPLES_PER_GROUP_SMALL),
            "--pool-size", str(POOL_SIZE_SMALL),
        ]
    )
    assert exit_code == 0

    validate_exit_code = point_ik_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--validate-only",
            "--sample-limit-per-group", str(SAMPLES_PER_GROUP_SMALL),
        ]
    )
    assert validate_exit_code == 0


def test_cli_rejects_reoverwrite_without_flag(v2_root):
    assert point_ik_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--sample-limit-per-group", str(SAMPLES_PER_GROUP_SMALL),
            "--pool-size", str(POOL_SIZE_SMALL),
        ]
    ) == 0
    assert point_ik_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--sample-limit-per-group", str(SAMPLES_PER_GROUP_SMALL),
            "--pool-size", str(POOL_SIZE_SMALL),
        ]
    ) == 2


def test_validator_passes_on_freshly_generated_small_fixture(v2_root):
    _generate_small(v2_root)
    report = validate_point_ik(v2_root, full_counts=False, expected_samples_per_group=SAMPLES_PER_GROUP_SMALL)
    assert report.passed, report.reasons
    assert report.total_samples == SAMPLES_PER_GROUP_SMALL * len(DIFFICULTY_GROUPS)
