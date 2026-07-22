"""Tests for the Dataset v2 Tier 0 generator (Phase 2): FK/Jacobian/singularity validation states.

Uses small fixture counts (not the full 1000/1000/600) for speed, except one integration test
that confirms the *resolved* full counts in ``configs/tier0_config.json`` are exactly 1000/1000/600
per ``specs/DLS_DATASET_V2_SPEC.md`` section B. Never touches Dataset v1.
"""

import json

import jsonschema
import numpy as np
import pytest

from dataset_v2.locator import dataset_v2_paths
from dataset_v2.scaffold import create_dataset_v2_scaffold
from dataset_v2.tier0_generation import (
    FK_GROUPS,
    JACOBIAN_GROUPS,
    SINGULARITY_GROUPS,
    load_singularity_threshold,
    run_tier0_generation,
)
from kinematics.model_loader import load_model_context
from pipelines.run_dataset_v2_tier0_generation import main as tier0_cli_main
from utils.dataset_locator import ASSETS_DIR, BENCHMARKS_DIR, CONFIGS_DIR, REPO_ROOT, SCHEMAS_DIR, TRAJECTORIES_DIR
from utils.npz_utils import load_npz

MASTER_SEED = 42
MODEL_CONTEXT = load_model_context()

# Small fixture counts (must divide evenly: FK/Jacobian by 5 groups, singularity by 3 groups).
FK_SMALL = 10
JACOBIAN_SMALL = 10
SINGULARITY_SMALL = 9
SINGULARITY_POOL_SMALL = 4000
JACOBIAN_POOL_SMALL = 1000

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
    return run_tier0_generation(
        root,
        master_seed=master_seed,
        overwrite=overwrite,
        fk_total=FK_SMALL,
        jacobian_total=JACOBIAN_SMALL,
        singularity_total=SINGULARITY_SMALL,
        jacobian_candidate_pool_size=JACOBIAN_POOL_SMALL,
        singularity_candidate_pool_size=SINGULARITY_POOL_SMALL,
        model_context=model_context,
    )


def test_generation_never_touches_dataset_v1(v2_root):
    before = _snapshot_v1()
    _generate_small(v2_root)
    after = _snapshot_v1()
    assert before == after, "Tier 0 v2 generation must never touch Dataset v1"


def test_generation_writes_expected_files(v2_root):
    result = _generate_small(v2_root)
    tier0_dir = result.tier0_validation_dir
    assert (tier0_dir / "fk_test_states_v2.npz").is_file()
    assert (tier0_dir / "jacobian_test_states_v2.npz").is_file()
    assert (tier0_dir / "singularity_test_states_v2.npz").is_file()
    assert (tier0_dir / "fk_test_states_v2_metadata.json").is_file()
    assert (tier0_dir / "jacobian_test_states_v2_metadata.json").is_file()
    assert (tier0_dir / "singularity_test_states_v2_metadata.json").is_file()
    assert (tier0_dir / "tier0_generation_report.json").is_file()


def test_exact_group_counts_in_small_fixture(v2_root):
    result = _generate_small(v2_root)
    assert result.fk_total == FK_SMALL
    assert result.jacobian_total == JACOBIAN_SMALL
    assert result.singularity_total == SINGULARITY_SMALL
    assert result.fk_group_counts == {name: FK_SMALL // len(FK_GROUPS) for name in FK_GROUPS.values()}
    assert result.jacobian_group_counts == {name: JACOBIAN_SMALL // len(JACOBIAN_GROUPS) for name in JACOBIAN_GROUPS.values()}
    assert result.singularity_group_counts == {name: SINGULARITY_SMALL // len(SINGULARITY_GROUPS) for name in SINGULARITY_GROUPS.values()}
    assert result.full_locked_counts is False


def test_npz_loads_with_allow_pickle_false(v2_root):
    result = _generate_small(v2_root)
    for filename in ("fk_test_states_v2.npz", "jacobian_test_states_v2.npz", "singularity_test_states_v2.npz"):
        arrays = load_npz(result.tier0_validation_dir / filename)
        assert arrays  # not empty
        for arr in arrays.values():
            assert arr.dtype != object


def test_operational_limit_compliance(v2_root):
    result = _generate_small(v2_root)
    lower = MODEL_CONTEXT.operational_lower_rad
    upper = MODEL_CONTEXT.operational_upper_rad
    for filename in ("fk_test_states_v2.npz", "jacobian_test_states_v2.npz", "singularity_test_states_v2.npz"):
        arrays = load_npz(result.tier0_validation_dir / filename)
        q = arrays["q_samples"]
        assert np.all(np.isfinite(q))
        assert np.all(q >= lower) and np.all(q <= upper)


def test_no_duplicate_states_within_each_file(v2_root):
    result = _generate_small(v2_root)
    for filename in ("fk_test_states_v2.npz", "jacobian_test_states_v2.npz", "singularity_test_states_v2.npz"):
        arrays = load_npz(result.tier0_validation_dir / filename)
        q = arrays["q_samples"]
        _, first_idx = np.unique(q, axis=0, return_index=True)
        assert first_idx.shape[0] == q.shape[0], f"{filename} contains duplicate joint states"


def test_deterministic_generation_same_seed(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    create_dataset_v2_scaffold(root_a, master_seed=MASTER_SEED)
    create_dataset_v2_scaffold(root_b, master_seed=MASTER_SEED)

    result_a = _generate_small(root_a)
    result_b = _generate_small(root_b)

    for filename in ("fk_test_states_v2.npz", "jacobian_test_states_v2.npz", "singularity_test_states_v2.npz"):
        arrays_a = load_npz(result_a.tier0_validation_dir / filename)
        arrays_b = load_npz(result_b.tier0_validation_dir / filename)
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

    arrays_a = load_npz(result_a.tier0_validation_dir / "fk_test_states_v2.npz")
    arrays_b = load_npz(result_b.tier0_validation_dir / "fk_test_states_v2.npz")
    assert not np.array_equal(arrays_a["q_samples"], arrays_b["q_samples"])


def test_low_sigma_group_uses_real_computed_sigma_min(v2_root):
    result = _generate_small(v2_root)
    arrays = load_npz(result.tier0_validation_dir / "jacobian_test_states_v2.npz")
    name_to_id = {name: gid for gid, name in JACOBIAN_GROUPS.items()}
    low_sigma_mask = arrays["group_id"] == name_to_id["low_sigma"]
    regular_mask = arrays["group_id"] == name_to_id["regular"]

    low_sigma_values = arrays["sigma_min"][low_sigma_mask]
    regular_values = arrays["sigma_min"][regular_mask]
    assert low_sigma_values.max() <= regular_values.min(), (
        "low_sigma group must have smaller sigma_min than the regular (interior) group -- "
        "selection must be driven by computed sigma_min, not the group name alone"
    )


def test_singularity_classification_matches_threshold(v2_root):
    result = _generate_small(v2_root)
    arrays = load_npz(result.tier0_validation_dir / "singularity_test_states_v2.npz")
    threshold, _source = load_singularity_threshold()
    metadata = json.loads((result.tier0_validation_dir / "singularity_test_states_v2_metadata.json").read_text(encoding="utf-8"))
    moderate_upper = metadata["moderately_conditioned_upper_bound"]

    name_to_id = {name: gid for gid, name in SINGULARITY_GROUPS.items()}
    sigma_min = arrays["sigma_min"]
    group_id = arrays["group_id"]

    assert np.all(sigma_min[group_id == name_to_id["near_singular"]] <= threshold)
    moderate_sigma = sigma_min[group_id == name_to_id["moderately_conditioned"]]
    assert np.all((moderate_sigma > threshold) & (moderate_sigma <= moderate_upper))
    assert np.all(sigma_min[group_id == name_to_id["regular"]] > moderate_upper)


def test_singularity_condition_number_never_nan(v2_root):
    result = _generate_small(v2_root)
    arrays = load_npz(result.tier0_validation_dir / "singularity_test_states_v2.npz")
    assert not np.any(np.isnan(arrays["condition_number"]))


def test_overwrite_protection(v2_root):
    _generate_small(v2_root)
    with pytest.raises(FileExistsError):
        _generate_small(v2_root, overwrite=False)
    _generate_small(v2_root, overwrite=True)  # allowed and does not raise


def test_dry_run_writes_nothing(v2_root):
    result = run_tier0_generation(
        v2_root,
        master_seed=MASTER_SEED,
        fk_total=FK_SMALL,
        jacobian_total=JACOBIAN_SMALL,
        singularity_total=SINGULARITY_SMALL,
        dry_run=True,
    )
    assert result.dry_run is True
    assert not (result.tier0_validation_dir / "fk_test_states_v2.npz").exists()
    assert list(result.tier0_validation_dir.iterdir()) == [result.tier0_validation_dir / ".gitkeep"]


def test_explicit_dataset_root_independent_of_cwd(v2_root, tmp_path, monkeypatch):
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    result = _generate_small(v2_root)
    assert result.fk_total == FK_SMALL


def test_manifest_updated_with_actual_counts_after_generation(v2_root):
    result = _generate_small(v2_root)
    paths = dataset_v2_paths(v2_root)
    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    tier0_counts = manifest["counts"]["tier0"]
    assert tier0_counts["fk_states"] == FK_SMALL
    assert tier0_counts["jacobian_states"] == JACOBIAN_SMALL
    assert tier0_counts["singularity_states"] == SINGULARITY_SMALL
    assert tier0_counts["generated"] is True
    assert tier0_counts["full_locked_counts"] is False
    # dataset-wide flags are untouched by a Tier-0-only generation
    assert manifest["generated"] is False
    assert manifest["frozen"] is False


def test_checksum_manifest_includes_generated_tier0_files(v2_root):
    _generate_small(v2_root)
    paths = dataset_v2_paths(v2_root)
    checksum_manifest = json.loads(paths.checksum_manifest_file.read_text(encoding="utf-8"))
    generated_entries = checksum_manifest["categories"]["generated_data_checksum"]
    filenames = {e["filename"] for e in generated_entries}
    assert "tier0_validation/fk_test_states_v2.npz" in filenames
    assert "tier0_validation/jacobian_test_states_v2.npz" in filenames
    assert "tier0_validation/singularity_test_states_v2.npz" in filenames
    assert checksum_manifest["status"] == "partial_generation_in_progress"


def test_metadata_json_has_required_fields(v2_root):
    result = _generate_small(v2_root)
    for filename in (
        "fk_test_states_v2_metadata.json",
        "jacobian_test_states_v2_metadata.json",
        "singularity_test_states_v2_metadata.json",
    ):
        metadata = json.loads((result.tier0_validation_dir / filename).read_text(encoding="utf-8"))
        for field in (
            "dataset_version",
            "schema_version",
            "generator_version",
            "asset_fingerprint",
            "master_seed",
            "counts",
            "group_counts",
            "arrays",
            "output_sha256",
            "generation_timestamp_utc",
            "generation_status",
        ):
            assert field in metadata, f"{filename} missing required field '{field}'"
        assert metadata["generation_status"] == "development"


def test_no_absolute_paths_in_generated_metadata_or_report(v2_root):
    result = _generate_small(v2_root)
    root_str = str(v2_root)
    root_str_posix = root_str.replace("\\", "/")
    for filename in (
        "fk_test_states_v2_metadata.json",
        "jacobian_test_states_v2_metadata.json",
        "singularity_test_states_v2_metadata.json",
        "tier0_generation_report.json",
    ):
        content = (result.tier0_validation_dir / filename).read_text(encoding="utf-8")
        assert root_str not in content
        assert root_str_posix not in content
        assert str(REPO_ROOT) not in content


def test_schema_validates_a_representative_state_record(v2_root):
    result = _generate_small(v2_root)
    paths = dataset_v2_paths(v2_root)
    schema = json.loads((paths.schemas_dir / "tier0_state_schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    fk_arrays = load_npz(result.tier0_validation_dir / "fk_test_states_v2.npz")
    record = {
        "state_type": "fk",
        "sample_id": int(fk_arrays["sample_id"][0]),
        "q_sample": fk_arrays["q_samples"][0].tolist(),
        "source_seed": int(fk_arrays["source_seed"][0]),
        "sample_count": int(fk_arrays["sample_id"].shape[0]),
    }
    validator.validate(record)  # must not raise


def test_generation_requires_existing_scaffold(tmp_path):
    from utils.exceptions import ModelConfigurationError

    missing_root = tmp_path / "no_scaffold_here"
    with pytest.raises(ModelConfigurationError):
        run_tier0_generation(missing_root, master_seed=MASTER_SEED, fk_total=FK_SMALL, jacobian_total=JACOBIAN_SMALL, singularity_total=SINGULARITY_SMALL)


def test_resolved_full_counts_in_config_are_locked_1000_1000_600(v2_root):
    """Integration check (section 12/13): the counts a full (no-override) generation run would
    resolve to must be exactly the locked spec numbers, without actually running the full
    2,600-state generation (too slow for the regular test suite; see docs/V2_IMPLEMENTATION_LOG.md
    for a manual full-run confirmation).
    """
    from dataset_v2.tier0_generation import load_tier0_generation_settings

    paths = dataset_v2_paths(v2_root)
    settings = load_tier0_generation_settings(paths)
    assert settings.fk_total == 1000
    assert settings.jacobian_total == 1000
    assert settings.singularity_total == 600


def test_cli_dry_run(v2_root):
    exit_code = tier0_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--master-seed", str(MASTER_SEED),
            "--dry-run",
            "--fk-count", str(FK_SMALL),
            "--jacobian-count", str(JACOBIAN_SMALL),
            "--singularity-count", str(SINGULARITY_SMALL),
        ]
    )
    assert exit_code == 0
    paths = dataset_v2_paths(v2_root)
    assert not (paths.tier0_validation_dir / "fk_test_states_v2.npz").exists()


def test_cli_generate_then_validate_only(v2_root):
    exit_code = tier0_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--master-seed", str(MASTER_SEED),
            "--fk-count", str(FK_SMALL),
            "--jacobian-count", str(JACOBIAN_SMALL),
            "--singularity-count", str(SINGULARITY_SMALL),
            "--jacobian-candidate-pool-size", str(JACOBIAN_POOL_SMALL),
            "--singularity-candidate-pool-size", str(SINGULARITY_POOL_SMALL),
        ]
    )
    assert exit_code == 0

    validate_exit_code = tier0_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--validate-only",
            "--fk-count", str(FK_SMALL),
            "--jacobian-count", str(JACOBIAN_SMALL),
            "--singularity-count", str(SINGULARITY_SMALL),
        ]
    )
    assert validate_exit_code == 0


def test_cli_rejects_reoverwrite_without_flag(v2_root):
    assert tier0_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--fk-count", str(FK_SMALL),
            "--jacobian-count", str(JACOBIAN_SMALL),
            "--singularity-count", str(SINGULARITY_SMALL),
            "--jacobian-candidate-pool-size", str(JACOBIAN_POOL_SMALL),
            "--singularity-candidate-pool-size", str(SINGULARITY_POOL_SMALL),
        ]
    ) == 0
    assert tier0_cli_main(
        [
            "--dataset-root", str(v2_root),
            "--fk-count", str(FK_SMALL),
            "--jacobian-count", str(JACOBIAN_SMALL),
            "--singularity-count", str(SINGULARITY_SMALL),
        ]
    ) == 2
