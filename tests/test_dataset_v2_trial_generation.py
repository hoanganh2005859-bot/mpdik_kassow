"""Tests for the Dataset v2 trial generator/validator/calibration (Phase 7).

Uses the fast FK-only fixture (``tests/_dataset_v2_trial_helpers.py``). The locked full 630-trial
counts (210/210/210 split, 210/difficulty, 360 core / 270 challenge) are verified against the
persistent working dataset and reported in docs/V2_IMPLEMENTATION_LOG.md, not inside this suite.
Never touches Dataset v1.
"""

import csv
import json

import jsonschema
import numpy as np
import pytest

from _dataset_v2_trial_helpers import build_fixture
from dataset_v2.locator import dataset_v2_paths
from dataset_v2.schemas import trial_schema
from dataset_v2.trial_calibration import apply_calibration_to_config, calibrate
from dataset_v2.trial_generation import (
    ANTI_LEAKAGE_REPORT_NAME,
    PROTECTED_DIR_NAME,
    TRIAL_MANIFEST_NAME,
    run_trial_generation,
)
from dataset_v2.trial_validation import validate_trials
from kinematics.model_loader import load_model_context
from utils.dataset_locator import ASSETS_DIR, BENCHMARKS_DIR, CONFIGS_DIR, REPO_ROOT, SCHEMAS_DIR, TRAJECTORIES_DIR
from utils.npz_utils import load_npz, save_npz

MODEL_CONTEXT = load_model_context()
POOL = 40
_WATCHED_V1_DIRS = [ASSETS_DIR, BENCHMARKS_DIR, TRAJECTORIES_DIR, CONFIGS_DIR, SCHEMAS_DIR]


def _snapshot_v1():
    snapshot = {}
    for directory in _WATCHED_V1_DIRS:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path.relative_to(REPO_ROOT))] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _prepare(root, master_seed=42, n_core=6, n_challenge=6):
    build_fixture(root, MODEL_CONTEXT, master_seed=master_seed, n_core=n_core, n_challenge=n_challenge)
    cal = calibrate(root, master_seed=master_seed, model_context=MODEL_CONTEXT, pool_scale_override=POOL)
    apply_calibration_to_config(root, cal)
    return root


def _generate(root, master_seed=42, overwrite=False, **kwargs):
    return run_trial_generation(root, master_seed=master_seed, model_context=MODEL_CONTEXT, pool_scale_override=POOL, overwrite=overwrite, **kwargs)


def _read_manifest(root):
    paths = dataset_v2_paths(root)
    with open(paths.trials_dir / TRIAL_MANIFEST_NAME, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def gen_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2_trials") / "ds"
    _prepare(root)
    _generate(root)
    return root


def test_counts_three_per_trajectory_and_distribution(gen_root):
    result_rows = _read_manifest(gen_root)
    assert len(result_rows) == 36  # 12 trajectories x 3
    by_traj = {}
    for r in result_rows:
        by_traj.setdefault(r["trajectory_id"], set()).add(r["difficulty"])
    assert all(v == {"easy", "medium", "hard"} for v in by_traj.values())
    assert len(by_traj) == 12
    assert sum(1 for r in result_rows if r["difficulty"] == "easy") == 12
    assert sum(1 for r in result_rows if r["difficulty"] == "medium") == 12
    assert sum(1 for r in result_rows if r["difficulty"] == "hard") == 12
    assert sum(1 for r in result_rows if r["trajectory_family"] == "core") == 18
    assert sum(1 for r in result_rows if r["trajectory_family"] == "random_challenge") == 18
    assert {r["split"] for r in result_rows} == {"development", "validation", "frozen_test"}


def test_independent_validator_passes(gen_root):
    report = validate_trials(gen_root, model_context=MODEL_CONTEXT, full_counts=False)
    assert report.passed, report.reasons
    assert report.statistics["max_position_error_recompute_disagreement_m"] < 1e-6
    assert report.statistics["max_primary_metric_recompute_disagreement"] < 1e-6


def test_split_inheritance(gen_root):
    catalog = {r["trajectory_id"]: r for r in _read_manifest(gen_root)}
    from dataset_v2.trajectory_catalog import load_combined_catalog

    traj_split = {r["trajectory_id"]: r["split"] for r in load_combined_catalog(gen_root)}
    for tid, row in catalog.items():
        assert row["split"] == traj_split[tid]


def test_q_initial_within_limits_and_independent_of_reference(gen_root):
    paths = dataset_v2_paths(gen_root)
    lower = MODEL_CONTEXT.operational_lower_rad
    upper = MODEL_CONTEXT.operational_upper_rad
    for split in ("development", "validation", "frozen_test"):
        pub = load_npz(paths.trials_dir / f"{split}.npz")
        prot = load_npz(paths.trials_dir / PROTECTED_DIR_NAME / f"{split}_evidence.npz")
        q_init = pub["q_initial"]
        assert np.all(q_init >= lower - 1e-9) and np.all(q_init <= upper + 1e-9)
        # q_initial never equals the protected reference start
        assert np.all(np.max(np.abs(q_init - prot["q_reference_start"]), axis=1) > 1e-6)


def test_public_npz_has_no_protected_arrays_but_evidence_does(gen_root):
    from dataset_v2.trajectory_loading import PROTECTED_ARRAY_KEYS

    paths = dataset_v2_paths(gen_root)
    pub = load_npz(paths.trials_dir / "development.npz")
    assert not (set(pub.keys()) & PROTECTED_ARRAY_KEYS)
    assert "q_reference" not in pub
    evidence = load_npz(paths.trials_dir / PROTECTED_DIR_NAME / "development_evidence.npz")
    assert "q_reference_start" in evidence
    assert "normalized_joint_distance_to_reference_start" in evidence


def test_npz_loads_with_allow_pickle_false(gen_root):
    paths = dataset_v2_paths(gen_root)
    # load_npz enforces allow_pickle=False; a successful load proves no object arrays were written.
    data = load_npz(paths.trials_dir / "development.npz")
    assert data["q_initial"].dtype != object


def test_fk_metadata_matches_recomputation(gen_root):
    from kinematics.forward_kinematics import forward_kinematics

    paths = dataset_v2_paths(gen_root)
    pub = load_npz(paths.trials_dir / "development.npz")
    for i in range(pub["q_initial"].shape[0]):
        fk = forward_kinematics(MODEL_CONTEXT, pub["q_initial"][i])
        assert np.allclose(fk.position, pub["initial_position"][i], atol=1e-9)


def test_monotonicity_per_trajectory(gen_root):
    rows = _read_manifest(gen_root)
    by_traj = {}
    for r in rows:
        by_traj.setdefault(r["trajectory_id"], {})[r["difficulty"]] = float(r["primary_difficulty_metric"])
    for tid, m in by_traj.items():
        assert m["easy"] < m["medium"] < m["hard"], tid


def test_difficulty_classification_within_bands(gen_root):
    paths = dataset_v2_paths(gen_root)
    cfg = json.loads((paths.configs_dir / "trial_config.json").read_text())["difficulty"]
    easy_upper = cfg["bands"]["easy"]["primary_max_inclusive"]
    medium_lower = cfg["bands"]["medium"]["primary_min_inclusive"]
    medium_upper = cfg["bands"]["medium"]["primary_max_inclusive"]
    hard_lower = cfg["bands"]["hard"]["primary_min_inclusive"]
    for r in _read_manifest(gen_root):
        prim = float(r["primary_difficulty_metric"])
        if r["difficulty"] == "easy":
            assert prim <= easy_upper + 1e-9
        elif r["difficulty"] == "medium":
            assert medium_lower - 1e-9 <= prim <= medium_upper + 1e-9
        else:
            assert prim >= hard_lower - 1e-9


def test_unique_ids_and_hashes(gen_root):
    rows = _read_manifest(gen_root)
    assert len({r["trial_id"] for r in rows}) == len(rows)
    assert len({r["content_hash"] for r in rows}) == len(rows)


def test_anti_leakage_report_present_and_pass(gen_root):
    paths = dataset_v2_paths(gen_root)
    report = json.loads((paths.trials_dir / ANTI_LEAKAGE_REPORT_NAME).read_text())
    assert report["pass"]
    assert report["collisions_found"] == 0


def test_manifest_record_matches_schema(gen_root):
    schema = trial_schema()
    row = _read_manifest(gen_root)[0]
    record = {
        "trial_id": row["trial_id"],
        "trajectory_id": row["trajectory_id"],
        "trajectory_family": row["trajectory_family"],
        "split": row["split"],
        "difficulty": row["difficulty"],
        "q_initial": json.loads(row["q_initial"]),
        "initial_position": json.loads(row["initial_position"]),
        "initial_quaternion_wxyz": json.loads(row["initial_quaternion_wxyz"]),
        "first_target_position": json.loads(row["first_target_position"]),
        "first_target_quaternion_wxyz": json.loads(row["first_target_quaternion_wxyz"]),
        "initial_position_error_m": float(row["initial_position_error_m"]),
        "initial_orientation_error_rad": float(row["initial_orientation_error_rad"]),
        "primary_difficulty_metric": float(row["primary_difficulty_metric"]),
        "initial_sigma_min": float(row["initial_sigma_min"]),
        "initial_sigma_max": float(row["initial_sigma_max"]),
        "initial_condition_number": float(row["initial_condition_number"]),
        "minimum_initial_limit_margin_normalized": float(row["minimum_initial_limit_margin_normalized"]),
        "minimum_initial_limit_margin_absolute_rad": float(row["minimum_initial_limit_margin_absolute_rad"]),
        "controlling_joint_index": int(row["controlling_joint_index"]),
        "candidate_source_pool": row["candidate_source_pool"],
        "source_seed": int(row["source_seed"]),
        "trajectory_content_hash": row["trajectory_content_hash"],
        "model_fingerprint": row["model_fingerprint"],
        "config_fingerprint": row["config_fingerprint"],
        "content_hash": row["content_hash"],
    }
    jsonschema.validate(record, schema)


def test_deterministic_same_seed_and_sensitive_to_seed(gen_root):
    paths = dataset_v2_paths(gen_root)
    first = load_npz(paths.trials_dir / "development.npz")["q_initial"].copy()
    # same seed, regenerate -> byte identical
    _generate(gen_root, master_seed=42, overwrite=True)
    again = load_npz(paths.trials_dir / "development.npz")["q_initial"]
    assert np.array_equal(first, again)
    # different seed -> different content
    _generate(gen_root, master_seed=43, overwrite=True)
    different = load_npz(paths.trials_dir / "development.npz")["q_initial"]
    assert not np.array_equal(first, different)
    # restore the module fixture's canonical (seed 42) state for any later-ordered test
    _generate(gen_root, master_seed=42, overwrite=True)


def test_overwrite_protection(tmp_path):
    root = tmp_path / "ds"
    _prepare(root)
    _generate(root)
    with pytest.raises(FileExistsError):
        _generate(root, overwrite=False)


def test_dry_run_writes_nothing(tmp_path):
    root = tmp_path / "ds"
    _prepare(root)
    result = run_trial_generation(root, master_seed=42, model_context=MODEL_CONTEXT, pool_scale_override=POOL, dry_run=True)
    assert result.dry_run
    paths = dataset_v2_paths(root)
    assert not (paths.trials_dir / TRIAL_MANIFEST_NAME).is_file()


def test_generation_never_touches_dataset_v1(tmp_path):
    root = tmp_path / "ds"
    _prepare(root)
    before = _snapshot_v1()
    _generate(root)
    assert _snapshot_v1() == before, "Trial v2 generation must never touch Dataset v1"


def test_frozen_trial_revision_locked(gen_root):
    paths = dataset_v2_paths(gen_root)
    seed_policy = json.loads((paths.configs_dir / "seed_policy.json").read_text())
    assert seed_policy["frozen_trial_seed_revision"] == 1
    assert seed_policy["frozen_core_seed_revision"] == 4
    assert seed_policy["frozen_challenge_seed_revision"] == 1


def test_validator_detects_out_of_limit_q_initial(tmp_path):
    root = tmp_path / "ds"
    _prepare(root)
    _generate(root)
    paths = dataset_v2_paths(root)
    pub_path = paths.trials_dir / "development.npz"
    data = load_npz(pub_path)
    data["q_initial"][0, 0] = MODEL_CONTEXT.operational_upper_rad[0] + 5.0
    save_npz(pub_path, data, overwrite=True)
    report = validate_trials(root, model_context=MODEL_CONTEXT, full_counts=False)
    assert not report.passed
    assert any("joint limits" in r or "recomputation" in r or "out of band" in r for r in report.reasons)


def test_calibration_uses_development_only_and_is_deterministic(tmp_path):
    root = tmp_path / "ds"
    build_fixture(root, MODEL_CONTEXT, master_seed=42, n_core=6, n_challenge=6)
    a = calibrate(root, master_seed=42, model_context=MODEL_CONTEXT, pool_scale_override=POOL)
    b = calibrate(root, master_seed=42, model_context=MODEL_CONTEXT, pool_scale_override=POOL)
    assert a.development_trajectories == 4  # 12 trajectories / 3 splits
    assert a.easy_upper == b.easy_upper and a.hard_lower == b.hard_lower
    assert a.easy_upper < a.medium_lower <= a.medium_upper < a.hard_lower
