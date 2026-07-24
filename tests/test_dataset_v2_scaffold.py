"""Tests for the Dataset v2 path/config/schema/checksum scaffold (Phase 1).

Everything here operates against ``tmp_path`` -- never the repository itself -- since Dataset v2
has no committed root; the scaffold is created on demand at a caller-supplied dataset root. Also
asserts nothing under the repository's Dataset v1 paths is ever touched by any of this.
"""

import json

import jsonschema
import numpy as np
import pytest

from dataset_v2.checksums import build_checksum_manifest, content_hash_of_record, verify_checksum_manifest
from dataset_v2.locator import dataset_v2_paths, require_dataset_v2_root
from dataset_v2.scaffold import create_dataset_v2_scaffold
from pipelines.run_dataset_v2_scaffold import main as scaffold_cli_main
from utils.dataset_locator import ASSETS_DIR, BENCHMARKS_DIR, CONFIGS_DIR, REPO_ROOT, SCHEMAS_DIR, TRAJECTORIES_DIR
from utils.exceptions import ModelConfigurationError
from utils.file_checksum import sha256_file

_WATCHED_V1_DIRS = [ASSETS_DIR, BENCHMARKS_DIR, TRAJECTORIES_DIR, CONFIGS_DIR, SCHEMAS_DIR]

MASTER_SEED = 42


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
    return tmp_path / "kr810_dataset_v2"


def test_scaffold_creates_expected_layout(v2_root):
    before = _snapshot_v1()
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    after = _snapshot_v1()

    assert before == after, "scaffold creation must never touch Dataset v1"
    assert paths.version_file.is_file()
    assert paths.manifest_file.is_file()
    for directory in (
        paths.configs_dir,
        paths.schemas_dir,
        paths.checksums_dir,
        paths.tier0_validation_dir,
        paths.tier1_point_ik_dir,
        paths.anchors_dir,
        paths.trajectories_development_dir,
        paths.trajectories_validation_dir,
        paths.trajectories_frozen_test_dir,
        paths.trials_dir,
        paths.references_dir,
        paths.reports_dir,
    ):
        assert directory.is_dir(), f"missing directory {directory}"

    # empty scaffold dirs are placeholder-only (.gitkeep), never fake NPZ/CSV data
    for directory in (paths.tier0_validation_dir, paths.anchors_dir, paths.trials_dir):
        contents = list(directory.iterdir())
        assert contents == [directory / ".gitkeep"]


def test_scaffold_refuses_v1_paths(tmp_path):
    with pytest.raises(ModelConfigurationError, match="Dataset v1"):
        create_dataset_v2_scaffold(REPO_ROOT, master_seed=MASTER_SEED)
    with pytest.raises(ModelConfigurationError, match="Dataset v1"):
        create_dataset_v2_scaffold(BENCHMARKS_DIR, master_seed=MASTER_SEED)


def test_scaffold_refuses_overwrite_without_flag(v2_root):
    create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    with pytest.raises(FileExistsError):
        create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    # overwrite=True is allowed and idempotent
    create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED, overwrite=True)


def test_manifest_declares_not_generated_not_frozen(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    assert manifest["generated"] is False
    assert manifest["frozen"] is False
    assert manifest["status"] == "scaffold"
    assert manifest["scope"]["includes_ppo"] is False
    assert manifest["scope"]["includes_mpdik"] is False
    assert manifest["scope"]["includes_mappo"] is False
    assert manifest["scope"]["includes_dynamic_control"] is False


def test_manifest_counts_match_locked_spec(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    counts = json.loads(paths.manifest_file.read_text(encoding="utf-8"))["counts"]

    assert counts["tier0"] == {"fk_states": 1000, "jacobian_states": 1000, "singularity_states": 600}
    assert counts["point_ik"]["total_samples"] == 6000
    assert counts["point_ik"]["samples_per_group"] == 1000
    assert counts["point_ik"]["groups"] == 6
    assert counts["point_ik"]["split_sizes"] == {"development": 1200, "validation": 1200, "frozen_test": 3600}
    assert counts["anchors"] == {"total": 12, "regular": 6, "near_limit": 3, "near_singular": 3}
    assert counts["core_trajectories"]["total"] == 120
    assert counts["random_challenge_trajectories"]["total"] == 90
    assert counts["random_challenge_trajectories"]["split_sizes"] == {"development": 30, "validation": 30, "frozen_test": 30}
    assert counts["trajectories_total"] == 210
    assert counts["canonical_waypoints_per_trajectory"] == 400
    assert counts["canonical_poses_total"] == 84000
    assert counts["trials_total"] == 630


def test_configs_all_parse_as_json(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    config_files = sorted(paths.configs_dir.glob("*.json"))
    assert len(config_files) == 13
    for path in config_files:
        json.loads(path.read_text(encoding="utf-8"))  # must not raise


def test_seed_policy_records_master_seed_explicitly(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=123456)
    seed_policy = json.loads((paths.configs_dir / "seed_policy.json").read_text(encoding="utf-8"))
    assert seed_policy["master_seed"] == 123456
    assert "component_tags" in seed_policy
    assert "split_tags" in seed_policy


def test_anchor_config_thresholds_are_locked_by_calibration_not_invented(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    anchor_config = json.loads((paths.configs_dir / "anchor_config.json").read_text(encoding="utf-8"))
    difficulty_thresholds = json.loads((paths.configs_dir / "difficulty_thresholds.json").read_text(encoding="utf-8"))

    assert difficulty_thresholds["status"] == "locked"
    assert anchor_config["acceptance_criteria"]["near_limit"]["status"] == "locked"
    assert anchor_config["acceptance_criteria"]["near_singular"]["status"] == "locked"
    # thresholds must match the calibrated values in configs/difficulty_thresholds.json, not a
    # second, independently-invented number
    assert (
        anchor_config["acceptance_criteria"]["near_limit"]["threshold_normalized_joint_limit_margin"]
        == difficulty_thresholds["near_joint_limit"]["threshold_normalized"]
    )
    assert (
        anchor_config["acceptance_criteria"]["near_singular"]["threshold_sigma_min"]
        == difficulty_thresholds["near_singularity"]["threshold_sigma_min"]
    )
    # regular is not blocked -- v1's select_anchor predicate already covers it
    assert isinstance(anchor_config["acceptance_criteria"]["regular"], str)


def test_difficulty_thresholds_config_is_well_formed(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    config = json.loads((paths.configs_dir / "difficulty_thresholds.json").read_text(encoding="utf-8"))

    assert config["status"] == "locked"
    assert 0.0 < config["near_joint_limit"]["threshold_normalized"] < 1.0
    assert config["near_singularity"]["threshold_sigma_min"] > 0.0
    assert (
        config["moderately_conditioned"]["upper_bound_sigma_min"]
        > config["near_singularity"]["threshold_sigma_min"]
    )
    assert config["regular"]["min_sigma_min"] == config["moderately_conditioned"]["upper_bound_sigma_min"]
    assert config["regular"]["min_normalized_joint_limit_margin"] == config["near_joint_limit"]["threshold_normalized"]
    assert config["classification_priority_highest_first"] == [
        "near_singularity",
        "near_joint_limit",
        "large_orientation_change",
        "far_target",
        "medium_target",
        "near_target",
    ]


def test_evaluation_defaults_are_not_invented(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    evaluation_defaults = json.loads((paths.configs_dir / "evaluation_defaults.json").read_text(encoding="utf-8"))
    assert evaluation_defaults["status"] == "not_yet_defined"


def test_schemas_are_all_valid_draft202012(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    schema_files = sorted(paths.schemas_dir.glob("*.json"))
    assert len(schema_files) == 10
    for path in schema_files:
        schema = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)


def _point_ik_conforming_record():
    return {
        "sample_id": "pik_development_near_target_00042",
        "split": "development",
        "difficulty_group": "near_target",
        "q_initial": [0.0] * 7,
        "q_target_reference": [0.1] * 7,
        "initial_position": [0.0, 0.1, 0.2],
        "initial_quaternion": [1.0, 0.0, 0.0, 0.0],
        "target_position": [0.1, 0.2, 0.3],
        "target_quaternion": [1.0, 0.0, 0.0, 0.0],
        "position_distance_m": 0.05,
        "orientation_distance_rad": 0.01,
        "joint_distance_rad": 0.1,
        "initial_sigma_min": 0.2,
        "target_sigma_min": 0.2,
        "initial_sigma_max": 0.5,
        "target_sigma_max": 0.5,
        "initial_condition_number": 2.5,
        "target_condition_number": 2.5,
        "minimum_initial_limit_margin_normalized": 0.5,
        "minimum_target_limit_margin_normalized": 0.5,
        "source_seed": 12345,
        "content_hash": "a" * 64,
    }


def test_point_ik_schema_accepts_a_conforming_record(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    schema = json.loads((paths.schemas_dir / "point_ik_schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(_point_ik_conforming_record())  # must not raise


def test_point_ik_schema_rejects_bad_split(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    schema = json.loads((paths.schemas_dir / "point_ik_schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    record = _point_ik_conforming_record()
    record["split"] = "not_a_real_split"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(record)


def test_checksum_manifest_scaffold_only_contains_real_files(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    checksum_manifest = json.loads(paths.checksum_manifest_file.read_text(encoding="utf-8"))

    assert checksum_manifest["categories"]["generated_data_checksum"] == []
    assert checksum_manifest["categories"]["release_archive_checksum"] == []

    entries = checksum_manifest["categories"]["source_config_fingerprint"]
    assert len(entries) > 0
    for entry in entries:
        file_path = paths.root / entry["filename"]
        assert file_path.is_file(), f"checksum entry references nonexistent file {entry['filename']}"
        assert entry["sha256"] == sha256_file(file_path)
        # never the checksum manifest's own file (would be circular)
        assert entry["filename"] != "checksums/CHECKSUM_MANIFEST.json"


def test_checksum_paths_are_relative_and_deterministically_ordered(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    checksum_manifest = json.loads(paths.checksum_manifest_file.read_text(encoding="utf-8"))
    entries = checksum_manifest["categories"]["source_config_fingerprint"]
    filenames = [e["filename"] for e in entries]
    assert filenames == sorted(filenames)
    for filename in filenames:
        assert not filename.startswith("/")
        assert ":" not in filename  # no Windows drive letter
        assert str(v2_root) not in filename


def test_verify_checksum_manifest_passes_on_untouched_scaffold(v2_root):
    create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    assert verify_checksum_manifest(v2_root) == []


def test_verify_checksum_manifest_detects_tampering(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    tampered_file = paths.configs_dir / "tier0_config.json"
    tampered_file.write_text(tampered_file.read_text(encoding="utf-8") + "\n// tampered\n", encoding="utf-8")

    mismatches = verify_checksum_manifest(v2_root)
    assert len(mismatches) == 1
    assert mismatches[0].filename == "configs/tier0_config.json"
    assert mismatches[0].expected_sha256 != mismatches[0].actual_sha256


def test_build_checksum_manifest_matches_written_manifest(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    on_disk = json.loads(paths.checksum_manifest_file.read_text(encoding="utf-8"))
    rebuilt = build_checksum_manifest(v2_root)
    assert on_disk["categories"]["source_config_fingerprint"] == rebuilt["categories"]["source_config_fingerprint"]


def test_content_hash_is_deterministic_and_order_independent_over_keys():
    record_a = {"b": 1.0000000000001, "a": 2}
    record_b = {"a": 2, "b": 1.0000000000002}  # differs below rounding precision, same key order irrelevance
    assert content_hash_of_record({"a": 1, "b": 2}) == content_hash_of_record({"b": 2, "a": 1})
    assert content_hash_of_record(record_a) == content_hash_of_record(record_b)
    assert content_hash_of_record({"a": 1}) != content_hash_of_record({"a": 2})


def test_dataset_v2_loader_parses_freshly_created_scaffold(v2_root):
    create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    paths = require_dataset_v2_root(v2_root)
    assert paths.manifest_file.is_file()


def test_dataset_v2_loader_raises_actionable_error_on_missing_root(tmp_path):
    missing = tmp_path / "no_such_root"
    with pytest.raises(ModelConfigurationError, match="does not exist"):
        require_dataset_v2_root(missing)


def test_dataset_v2_loader_raises_actionable_error_on_missing_manifest(tmp_path):
    empty_dir = tmp_path / "empty_root"
    empty_dir.mkdir()
    with pytest.raises(ModelConfigurationError, match="DATASET_MANIFEST.json"):
        require_dataset_v2_root(empty_dir)


def test_dataset_v2_paths_rejects_none_root():
    with pytest.raises(ModelConfigurationError):
        dataset_v2_paths(None)


def test_no_absolute_paths_in_generated_manifest_or_configs(v2_root):
    paths = create_dataset_v2_scaffold(v2_root, master_seed=MASTER_SEED)
    root_str = str(v2_root)
    root_str_posix = root_str.replace("\\", "/")
    for path in list(paths.configs_dir.glob("*.json")) + [paths.manifest_file, paths.checksum_manifest_file]:
        content = path.read_text(encoding="utf-8")
        assert root_str not in content
        assert root_str_posix not in content
        assert str(REPO_ROOT) not in content


def test_scaffold_cli_end_to_end(tmp_path):
    before = _snapshot_v1()
    v2_root = tmp_path / "cli_v2_root"

    exit_code = scaffold_cli_main(["--dataset-root", str(v2_root), "--master-seed", "7"])

    after = _snapshot_v1()
    assert before == after, "scaffold CLI must never modify Dataset v1"
    assert exit_code == 0
    assert (v2_root / "DATASET_MANIFEST.json").is_file()

    for path in v2_root.rglob("*"):
        assert tmp_path in path.parents


def test_scaffold_cli_rejects_v1_root(tmp_path):
    exit_code = scaffold_cli_main(["--dataset-root", str(REPO_ROOT)])
    assert exit_code == 2


def test_scaffold_cli_rejects_reoverwrite_without_flag(tmp_path):
    v2_root = tmp_path / "cli_v2_root2"
    assert scaffold_cli_main(["--dataset-root", str(v2_root)]) == 0
    assert scaffold_cli_main(["--dataset-root", str(v2_root)]) == 2
    assert scaffold_cli_main(["--dataset-root", str(v2_root), "--overwrite"]) == 0
