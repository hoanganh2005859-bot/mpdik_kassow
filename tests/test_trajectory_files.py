"""Tests for generated trajectory manifest/CSV/NPZ structure and schema conformance."""

import csv
import json

import jsonschema
import numpy as np
import pytest

from generators._common import REPO_ROOT
from kinematics.model_loader import load_model_context
from utils.file_checksum import sha256_file

TRAJECTORIES_DIR = REPO_ROOT / "trajectories"
MANIFEST_PATH = TRAJECTORIES_DIR / "trajectory_manifest.csv"
TRIALS_PATH = TRAJECTORIES_DIR / "trajectory_trials.csv"
SCHEMA_PATH = REPO_ROOT / "schemas" / "trajectory_schema.json"

CONTEXT = load_model_context()

TRAJECTORY_FILES = [
    ("line", "line_fixed_orientation"),
    ("line", "line_variable_orientation"),
    ("circle", "circle_fixed_orientation"),
    ("circle", "circle_variable_orientation"),
    ("figure8", "figure8_fixed_orientation"),
    ("figure8", "figure8_variable_orientation"),
    ("helix", "helix_fixed_orientation"),
    ("helix", "helix_variable_orientation"),
]
TRAJECTORY_IDS = [trajectory_id for _, trajectory_id in TRAJECTORY_FILES]

REQUIRED_ARRAYS = {
    "waypoint_id": np.int64,
    "time_s": np.float64,
    "path_parameter_s": np.float64,
    "target_position": np.float64,
    "target_quaternion": np.float64,
    "target_linear_velocity": np.float64,
    "target_linear_acceleration": np.float64,
}

EXPECTED_NUM_WAYPOINTS = 400
QUATERNION_NORM_TOL = 1e-6
CONTINUITY_MIN_DOT = 0.9
ENDPOINT_ZERO_TOL_M_S = 1e-6


def _npz_path(trajectory_type, trajectory_id):
    return TRAJECTORIES_DIR / trajectory_type / f"{trajectory_id}.npz"


def _load(trajectory_type, trajectory_id):
    path = _npz_path(trajectory_type, trajectory_id)
    assert path.is_file(), f"missing {path}"
    return np.load(path, allow_pickle=False)


def _manifest_rows():
    assert MANIFEST_PATH.is_file()
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as handle:
        return {row["trajectory_id"]: row for row in csv.DictReader(handle)}


@pytest.mark.parametrize("trajectory_type,trajectory_id", TRAJECTORY_FILES)
def test_trajectory_file_exists(trajectory_type, trajectory_id):
    assert _npz_path(trajectory_type, trajectory_id).is_file()


@pytest.mark.parametrize("trajectory_type,trajectory_id", TRAJECTORY_FILES)
def test_npz_loads_with_allow_pickle_false_and_has_required_arrays(trajectory_type, trajectory_id):
    data = _load(trajectory_type, trajectory_id)
    for name, dtype in REQUIRED_ARRAYS.items():
        assert name in data.files, f"{trajectory_id}: missing array '{name}'"
        assert data[name].dtype == np.dtype(dtype), f"{trajectory_id}: '{name}' dtype {data[name].dtype} != {dtype}"


@pytest.mark.parametrize("trajectory_type,trajectory_id", TRAJECTORY_FILES)
def test_array_shapes_are_consistent(trajectory_type, trajectory_id):
    data = _load(trajectory_type, trajectory_id)
    n = data["waypoint_id"].shape[0]
    assert n == EXPECTED_NUM_WAYPOINTS
    for name in ("time_s", "path_parameter_s"):
        assert data[name].shape == (n,)
    for name in ("target_position", "target_linear_velocity", "target_linear_acceleration"):
        assert data[name].shape == (n, 3)
    assert data["target_quaternion"].shape == (n, 4)


@pytest.mark.parametrize("trajectory_type,trajectory_id", TRAJECTORY_FILES)
def test_waypoint_id_unique_and_sorted(trajectory_type, trajectory_id):
    data = _load(trajectory_type, trajectory_id)
    n = data["waypoint_id"].shape[0]
    assert np.array_equal(data["waypoint_id"], np.arange(n, dtype=np.int64))


@pytest.mark.parametrize("trajectory_type,trajectory_id", TRAJECTORY_FILES)
def test_no_nan_or_inf(trajectory_type, trajectory_id):
    data = _load(trajectory_type, trajectory_id)
    for name in REQUIRED_ARRAYS:
        arr = data[name]
        if np.issubdtype(arr.dtype, np.floating):
            assert np.all(np.isfinite(arr)), f"{trajectory_id}: '{name}' contains NaN/Inf"


@pytest.mark.parametrize("trajectory_type,trajectory_id", TRAJECTORY_FILES)
def test_time_strictly_increasing(trajectory_type, trajectory_id):
    data = _load(trajectory_type, trajectory_id)
    assert np.all(np.diff(data["time_s"]) > 0.0)


@pytest.mark.parametrize("trajectory_type,trajectory_id", TRAJECTORY_FILES)
def test_path_parameter_in_unit_interval_and_nondecreasing(trajectory_type, trajectory_id):
    data = _load(trajectory_type, trajectory_id)
    s = data["path_parameter_s"]
    assert s[0] == pytest.approx(0.0, abs=1e-9)
    assert s[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.min(s) >= -1e-9
    assert np.max(s) <= 1.0 + 1e-9
    assert np.all(np.diff(s) >= -1e-9)


@pytest.mark.parametrize("trajectory_type,trajectory_id", TRAJECTORY_FILES)
def test_quaternions_normalized(trajectory_type, trajectory_id):
    data = _load(trajectory_type, trajectory_id)
    norms = np.linalg.norm(data["target_quaternion"], axis=1)
    assert np.max(np.abs(norms - 1.0)) < QUATERNION_NORM_TOL


@pytest.mark.parametrize("trajectory_type,trajectory_id", TRAJECTORY_FILES)
def test_quaternion_continuity(trajectory_type, trajectory_id):
    data = _load(trajectory_type, trajectory_id)
    quats = data["target_quaternion"]
    dots = np.sum(quats[1:] * quats[:-1], axis=1)
    assert np.min(dots) >= CONTINUITY_MIN_DOT, f"{trajectory_id}: discontinuous quaternion sequence (min dot={np.min(dots):.4f})"


@pytest.mark.parametrize("trajectory_type,trajectory_id", TRAJECTORY_FILES)
def test_first_and_last_velocity_near_zero(trajectory_type, trajectory_id):
    data = _load(trajectory_type, trajectory_id)
    assert np.linalg.norm(data["target_linear_velocity"][0]) < ENDPOINT_ZERO_TOL_M_S
    assert np.linalg.norm(data["target_linear_velocity"][-1]) < ENDPOINT_ZERO_TOL_M_S


@pytest.mark.parametrize("trajectory_type,trajectory_id", TRAJECTORY_FILES)
def test_first_and_last_acceleration_near_zero(trajectory_type, trajectory_id):
    data = _load(trajectory_type, trajectory_id)
    assert np.linalg.norm(data["target_linear_acceleration"][0]) < ENDPOINT_ZERO_TOL_M_S
    assert np.linalg.norm(data["target_linear_acceleration"][-1]) < ENDPOINT_ZERO_TOL_M_S


@pytest.mark.parametrize("trajectory_type,trajectory_id", TRAJECTORY_FILES)
def test_manifest_row_matches_file(trajectory_type, trajectory_id):
    rows = _manifest_rows()
    assert trajectory_id in rows, f"trajectory_manifest.csv has no row for '{trajectory_id}'"
    row = rows[trajectory_id]
    data = _load(trajectory_type, trajectory_id)

    assert row["type"] == trajectory_type
    assert int(row["num_waypoints"]) == data["waypoint_id"].shape[0]
    assert abs(float(row["duration_s"]) - float(data["time_s"][-1])) < 1e-6
    assert abs(float(row["control_period_s"]) - float(data["time_s"][1] - data["time_s"][0])) < 1e-9
    assert row["generation_status"] == "validated"
    assert float(row["validation_waypoint_success_rate"]) == pytest.approx(1.0)

    npz_path = _npz_path(trajectory_type, trajectory_id)
    assert row["sha256"] == sha256_file(npz_path)

    resolved = REPO_ROOT / row["file_path"]
    assert resolved.is_file()
    assert not row["file_path"].startswith("/")
    assert ":" not in row["file_path"]


def test_manifest_has_exactly_one_row_per_trajectory():
    rows = _manifest_rows()
    assert set(rows.keys()) == set(TRAJECTORY_IDS)


def test_manifest_anchor_q_json_is_valid_and_within_limits():
    rows = _manifest_rows()
    lower, upper = CONTEXT.operational_lower_rad, CONTEXT.operational_upper_rad
    for trajectory_id, row in rows.items():
        anchor_q = np.array(json.loads(row["anchor_q_json"]), dtype=np.float64)
        assert anchor_q.shape == (7,)
        assert np.all(np.isfinite(anchor_q))
        assert np.all(anchor_q >= lower) and np.all(anchor_q <= upper)


def test_no_absolute_paths_in_manifest_or_trials():
    for path in (MANIFEST_PATH, TRIALS_PATH):
        content = path.read_text(encoding="utf-8")
        assert str(REPO_ROOT) not in content
        assert str(REPO_ROOT).replace("\\", "/") not in content


def test_trial_categories_and_counts():
    assert TRIALS_PATH.is_file()
    with open(TRIALS_PATH, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(set(r["trial_id"] for r in rows)), "trial_id values must be unique"

    categories = {r["trial_category"] for r in rows}
    assert categories == {"repeatability", "robustness"}

    by_trajectory = {}
    for row in rows:
        by_trajectory.setdefault(row["trajectory_id"], []).append(row)

    assert set(by_trajectory.keys()) == set(TRAJECTORY_IDS)
    for trajectory_id, trial_rows in by_trajectory.items():
        repeat_rows = [r for r in trial_rows if r["trial_category"] == "repeatability"]
        robustness_rows = [r for r in trial_rows if r["trial_category"] == "robustness"]
        assert len(repeat_rows) > 0, f"{trajectory_id}: no repeatability trials"
        assert len(robustness_rows) > 0, f"{trajectory_id}: no robustness trials"

        repeat_q = {tuple(r[f"q{i}_init"] for i in range(1, 8)) for r in repeat_rows}
        assert len(repeat_q) == 1, f"{trajectory_id}: repeatability trials must share the same q_initial"


def test_trial_initial_configurations_are_within_operational_limits():
    lower, upper = CONTEXT.operational_lower_rad, CONTEXT.operational_upper_rad
    with open(TRIALS_PATH, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        q_init = np.array([float(row[f"q{i}_init"]) for i in range(1, 8)])
        assert np.all(q_init >= lower - 1e-9) and np.all(q_init <= upper + 1e-9), f"{row['trial_id']}: q_init out of operational limits"


def test_schema_is_valid_and_accepts_generated_waypoints():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    rows = _manifest_rows()
    for trajectory_type, trajectory_id in TRAJECTORY_FILES:
        data = _load(trajectory_type, trajectory_id)
        row = rows[trajectory_id]
        n = data["waypoint_id"].shape[0]
        waypoints = [
            {
                "waypoint_id": int(data["waypoint_id"][i]),
                "time_s": float(data["time_s"][i]),
                "path_parameter_s": float(data["path_parameter_s"][i]),
                "target_position": data["target_position"][i].tolist(),
                "target_quaternion": data["target_quaternion"][i].tolist(),
                "target_linear_velocity": data["target_linear_velocity"][i].tolist(),
                "target_linear_acceleration": data["target_linear_acceleration"][i].tolist(),
            }
            for i in (0, n // 2, n - 1)
        ]
        record = {
            "trajectory_id": trajectory_id,
            "type": row["type"],
            "orientation_mode": row["orientation_mode"],
            "closed_path": row["closed_path"] == "True",
            "duration_s": float(row["duration_s"]),
            "control_period_s": float(row["control_period_s"]),
            "generation_status": row["generation_status"],
            "waypoints": waypoints,
        }
        validator.validate(record)
