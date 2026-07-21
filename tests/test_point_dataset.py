"""Tests for the generated point-IK benchmark structure and schema conformance."""

import csv
import json

import jsonschema
import numpy as np
import pytest

from generators._common import REPO_ROOT
from generators.generate_point_ik_dataset import DIFFICULTY_GROUPS
from kinematics.forward_kinematics import forward_kinematics
from kinematics.model_loader import load_model_context
from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from kinematics.rotation_utils import rotation_geodesic_angle
from utils.file_checksum import sha256_file

POINT_IK_DIR = REPO_ROOT / "benchmarks" / "point_ik"
NPZ_PATH = POINT_IK_DIR / "point_ik_v1.npz"
MANIFEST_PATH = POINT_IK_DIR / "point_ik_manifest.csv"
DIFFICULTY_PATH = POINT_IK_DIR / "difficulty_definition.json"
CHECKSUM_PATH = POINT_IK_DIR / "point_ik_checksum.json"
SCHEMA_PATH = REPO_ROOT / "schemas" / "point_ik_schema.json"

EXPECTED_TOTAL_SAMPLES = 1200
EXPECTED_SAMPLES_PER_GROUP = 200

CONTEXT = load_model_context()

REQUIRED_ARRAYS = {
    "sample_id": np.int64,
    "q_initial": np.float64,
    "q_target": np.float64,
    "initial_position": np.float64,
    "initial_quaternion": np.float64,
    "target_position": np.float64,
    "target_quaternion": np.float64,
    "position_distance_m": np.float64,
    "orientation_distance_rad": np.float64,
    "joint_distance_rad": np.float64,
    "initial_sigma_min": np.float64,
    "target_sigma_min": np.float64,
    "minimum_initial_limit_margin": np.float64,
    "minimum_target_limit_margin": np.float64,
    "difficulty_id": np.int32,
    "source_seed": np.int64,
}


def _load():
    assert NPZ_PATH.is_file(), f"missing {NPZ_PATH}"
    return np.load(NPZ_PATH, allow_pickle=False)


def test_point_ik_files_exist():
    for path in (NPZ_PATH, MANIFEST_PATH, DIFFICULTY_PATH, CHECKSUM_PATH):
        assert path.is_file(), f"missing {path}"


def test_npz_loads_with_allow_pickle_false():
    data = np.load(NPZ_PATH, allow_pickle=False)
    assert set(REQUIRED_ARRAYS).issubset(set(data.files))


def test_required_arrays_present_with_expected_dtype():
    data = _load()
    for name, dtype in REQUIRED_ARRAYS.items():
        assert name in data.files, f"missing array '{name}'"
        assert data[name].dtype == np.dtype(dtype), f"'{name}' dtype {data[name].dtype} != {dtype}"


def test_sample_count_is_1200():
    data = _load()
    assert data["sample_id"].shape[0] == EXPECTED_TOTAL_SAMPLES


def test_each_difficulty_group_has_200_samples():
    data = _load()
    counts = {int(g): int(c) for g, c in zip(*np.unique(data["difficulty_id"], return_counts=True))}
    assert counts == {g: EXPECTED_SAMPLES_PER_GROUP for g in DIFFICULTY_GROUPS}


def test_array_shapes():
    data = _load()
    n = data["sample_id"].shape[0]
    assert data["q_initial"].shape == (n, 7)
    assert data["q_target"].shape == (n, 7)
    assert data["initial_position"].shape == (n, 3)
    assert data["initial_quaternion"].shape == (n, 4)
    assert data["target_position"].shape == (n, 3)
    assert data["target_quaternion"].shape == (n, 4)
    scalar_fields = [
        "position_distance_m",
        "orientation_distance_rad",
        "joint_distance_rad",
        "initial_sigma_min",
        "target_sigma_min",
        "minimum_initial_limit_margin",
        "minimum_target_limit_margin",
        "difficulty_id",
        "source_seed",
    ]
    for name in scalar_fields:
        assert data[name].shape == (n,), f"'{name}' has shape {data[name].shape}, expected ({n},)"


def test_no_nan_or_inf():
    data = _load()
    for name in REQUIRED_ARRAYS:
        arr = data[name]
        if np.issubdtype(arr.dtype, np.floating):
            assert np.all(np.isfinite(arr)), f"'{name}' contains NaN/Inf"


def test_sample_id_unique():
    data = _load()
    assert len(np.unique(data["sample_id"])) == data["sample_id"].shape[0]


def test_q_within_operational_limits():
    data = _load()
    lower, upper = CONTEXT.operational_lower_rad, CONTEXT.operational_upper_rad
    assert np.all(data["q_initial"] >= lower) and np.all(data["q_initial"] <= upper)
    assert np.all(data["q_target"] >= lower) and np.all(data["q_target"] <= upper)


def test_quaternions_are_normalized():
    data = _load()
    for name in ("initial_quaternion", "target_quaternion"):
        norms = np.linalg.norm(data[name], axis=1)
        assert np.max(np.abs(norms - 1.0)) < 1e-6, f"'{name}' is not unit-norm"


@pytest.mark.parametrize("index", [0, 1, 150, 300, 599, 600, 900, 1198, 1199])
def test_fk_of_q_target_matches_stored_target_pose(index):
    data = _load()
    fk = forward_kinematics(CONTEXT, data["q_target"][index])
    position_err = np.linalg.norm(fk.position - data["target_position"][index])
    assert position_err < 1e-6, f"sample {index}: FK(q_target) position mismatch {position_err:.3e} m"
    R_stored = quaternion_wxyz_to_matrix(data["target_quaternion"][index])
    orientation_err = rotation_geodesic_angle(fk.rotation_matrix, R_stored)
    assert orientation_err < 1e-6, f"sample {index}: FK(q_target) orientation mismatch {orientation_err:.3e} rad"


def test_distance_fields_match_recomputation():
    data = _load()
    n = data["sample_id"].shape[0]
    idx = np.linspace(0, n - 1, 60).round().astype(int)
    for i in idx:
        pos_dist = float(np.linalg.norm(data["target_position"][i] - data["initial_position"][i]))
        assert abs(pos_dist - data["position_distance_m"][i]) < 1e-8
        joint_dist = float(np.linalg.norm(data["q_target"][i] - data["q_initial"][i]))
        assert abs(joint_dist - data["joint_distance_rad"][i]) < 1e-8


def test_sigma_min_is_finite_and_nonnegative():
    data = _load()
    assert np.all(np.isfinite(data["initial_sigma_min"]))
    assert np.all(np.isfinite(data["target_sigma_min"]))
    assert np.all(data["initial_sigma_min"] >= 0.0)
    assert np.all(data["target_sigma_min"] >= 0.0)


def test_limit_margins_never_indicate_a_violation():
    data = _load()
    assert np.all(data["minimum_initial_limit_margin"] >= -1e-9)
    assert np.all(data["minimum_target_limit_margin"] >= -1e-9)


def test_manifest_matches_npz():
    data = _load()
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == data["sample_id"].shape[0]
    for i in (0, 1, 500, 1198, 1199):
        assert int(rows[i]["sample_id"]) == int(data["sample_id"][i])
        assert int(rows[i]["difficulty_id"]) == int(data["difficulty_id"][i])
        assert rows[i]["difficulty_name"] == DIFFICULTY_GROUPS[int(data["difficulty_id"][i])]
        assert abs(float(rows[i]["position_distance_m"]) - float(data["position_distance_m"][i])) < 1e-6


def test_difficulty_definition_is_complete():
    definition = json.loads(DIFFICULTY_PATH.read_text(encoding="utf-8"))
    for key in ("difficulty_groups", "priority_order_highest_first", "priority_note", "criteria", "quantile_thresholds", "sample_counts"):
        assert key in definition, f"difficulty_definition.json missing key '{key}'"
    assert definition["sample_counts"] == {name: EXPECTED_SAMPLES_PER_GROUP for name in DIFFICULTY_GROUPS.values()}


def test_checksum_matches_npz_file():
    checksum = json.loads(CHECKSUM_PATH.read_text(encoding="utf-8"))
    entry = next(f for f in checksum["files"] if f["filename"].endswith("point_ik_v1.npz"))
    assert entry["sha256"] == sha256_file(NPZ_PATH)
    assert entry["sample_count"] == EXPECTED_TOTAL_SAMPLES


def test_schema_is_valid_and_accepts_generated_samples():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    data = _load()
    for i in (0, 300, 900, 1199):
        record = {
            "sample_id": int(data["sample_id"][i]),
            "difficulty_id": int(data["difficulty_id"][i]),
            "q_initial": data["q_initial"][i].tolist(),
            "q_target": data["q_target"][i].tolist(),
            "initial_position": data["initial_position"][i].tolist(),
            "initial_quaternion": data["initial_quaternion"][i].tolist(),
            "target_position": data["target_position"][i].tolist(),
            "target_quaternion": data["target_quaternion"][i].tolist(),
            "position_distance_m": float(data["position_distance_m"][i]),
            "orientation_distance_rad": float(data["orientation_distance_rad"][i]),
            "joint_distance_rad": float(data["joint_distance_rad"][i]),
            "initial_sigma_min": float(data["initial_sigma_min"][i]),
            "target_sigma_min": float(data["target_sigma_min"][i]),
            "minimum_initial_limit_margin": float(data["minimum_initial_limit_margin"][i]),
            "minimum_target_limit_margin": float(data["minimum_target_limit_margin"][i]),
            "source_seed": int(data["source_seed"][i]),
        }
        validator.validate(record)


def test_no_absolute_paths_in_manifest_or_metadata():
    for path in (MANIFEST_PATH, DIFFICULTY_PATH, CHECKSUM_PATH):
        content = path.read_text(encoding="utf-8")
        assert str(REPO_ROOT) not in content
        assert str(REPO_ROOT).replace("\\", "/") not in content
