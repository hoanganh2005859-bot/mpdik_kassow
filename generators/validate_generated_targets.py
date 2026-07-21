"""Validates generated point-IK and trajectory targets against schemas/ and reachability constraints.

Checks (see module functions for detail):
    validate_point_ik_benchmark()  - benchmarks/point_ik/* structural, numerical, and FK-consistency checks
    validate_trajectories()        - trajectories/<type>/*.npz structural, kinematic, and manifest checks
    validate_schema_files()        - schemas/*.json parse as valid JSON Schema and accept real records

Run as a script for a pass/fail summary; import the functions for reuse from tests.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import jsonschema
import numpy as np

from generators._common import REPO_ROOT, get_model_context
from kinematics.forward_kinematics import forward_kinematics
from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from utils.file_checksum import sha256_file

BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
TRAJECTORIES_DIR = REPO_ROOT / "trajectories"
SCHEMAS_DIR = REPO_ROOT / "schemas"

POINT_IK_REQUIRED_ARRAYS = {
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

TRAJECTORY_REQUIRED_ARRAYS = {
    "waypoint_id": np.int64,
    "time_s": np.float64,
    "path_parameter_s": np.float64,
    "target_position": np.float64,
    "target_quaternion": np.float64,
    "target_linear_velocity": np.float64,
    "target_linear_acceleration": np.float64,
}

DIFFICULTY_NAMES = {
    0: "near_target",
    1: "medium_target",
    2: "far_target",
    3: "large_orientation_change",
    4: "near_joint_limit",
    5: "near_singularity",
}

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

QUATERNION_NORM_TOL = 1e-6
CONTINUITY_MIN_DOT = 0.9
ENDPOINT_ZERO_TOL = 1e-6
FK_MATCH_POSITION_TOL_M = 1e-6
FK_MATCH_ORIENTATION_TOL_RAD = 1e-6


def _no_absolute_path(path_str: str) -> bool:
    p = Path(path_str)
    return not p.is_absolute() and ":" not in path_str.split("/")[0]


def validate_schema_files():
    issues = []
    schemas = {}
    for name in ("point_ik_schema.json", "trajectory_schema.json"):
        path = SCHEMAS_DIR / name
        if not path.is_file():
            issues.append(f"missing schema file: {name}")
            continue
        schema = json.loads(path.read_text(encoding="utf-8"))
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.exceptions.SchemaError as exc:
            issues.append(f"invalid JSON Schema in {name}: {exc}")
            continue
        schemas[name] = schema
    return issues, schemas


def validate_point_ik_benchmark(model_context=None):
    issues = []
    point_ik_dir = BENCHMARKS_DIR / "point_ik"
    npz_path = point_ik_dir / "point_ik_v1.npz"
    manifest_path = point_ik_dir / "point_ik_manifest.csv"
    difficulty_path = point_ik_dir / "difficulty_definition.json"
    checksum_path = point_ik_dir / "point_ik_checksum.json"

    for path in (npz_path, manifest_path, difficulty_path, checksum_path):
        if not path.is_file():
            issues.append(f"missing point-IK file: {path.relative_to(REPO_ROOT).as_posix()}")
    if issues:
        return issues

    data = np.load(npz_path, allow_pickle=False)

    for name, dtype in POINT_IK_REQUIRED_ARRAYS.items():
        if name not in data.files:
            issues.append(f"point_ik_v1.npz missing required array '{name}'")
            continue
        if data[name].dtype != np.dtype(dtype):
            issues.append(f"point_ik_v1.npz array '{name}' has dtype {data[name].dtype}, expected {np.dtype(dtype)}")

    if issues:
        return issues

    n = data["sample_id"].shape[0]
    if n != 1200:
        issues.append(f"point_ik_v1.npz has {n} samples, expected 1200")

    for group_id, count in zip(*np.unique(data["difficulty_id"], return_counts=True)):
        if count != 200:
            issues.append(f"difficulty group {int(group_id)} has {int(count)} samples, expected 200")
    if set(np.unique(data["difficulty_id"]).tolist()) != set(DIFFICULTY_NAMES.keys()):
        issues.append("difficulty_id values do not exactly cover {0,1,2,3,4,5}")

    if len(np.unique(data["sample_id"])) != n:
        issues.append("sample_id values are not unique")

    for name in POINT_IK_REQUIRED_ARRAYS:
        arr = data[name]
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            issues.append(f"array '{name}' contains NaN/Inf")

    model_context = model_context or get_model_context()
    lower, upper = model_context.operational_lower_rad, model_context.operational_upper_rad
    if np.any(data["q_initial"] < lower) or np.any(data["q_initial"] > upper):
        issues.append("q_initial contains values outside operational limits")
    if np.any(data["q_target"] < lower) or np.any(data["q_target"] > upper):
        issues.append("q_target contains values outside operational limits")

    for name in ("initial_quaternion", "target_quaternion"):
        norms = np.linalg.norm(data[name], axis=1)
        if np.max(np.abs(norms - 1.0)) > QUATERNION_NORM_TOL:
            issues.append(f"'{name}' contains non-unit quaternions (max |norm-1|={np.max(np.abs(norms - 1.0)):.3e})")

    if np.any(data["initial_sigma_min"] < 0.0) or np.any(data["target_sigma_min"] < 0.0):
        issues.append("sigma_min values must be non-negative")

    # Spot-check FK(q_target) against the stored target pose across the full dataset (cheap: mujoco calls are fast).
    fk_data = model_context.new_data()
    max_pos_err = 0.0
    max_orient_err = 0.0
    for i in range(n):
        fk = forward_kinematics(model_context, data["q_target"][i], data=fk_data)
        pos_err = float(np.linalg.norm(fk.position - data["target_position"][i]))
        R_stored = quaternion_wxyz_to_matrix(data["target_quaternion"][i])
        from kinematics.rotation_utils import rotation_geodesic_angle

        orient_err = rotation_geodesic_angle(fk.rotation_matrix, R_stored)
        max_pos_err = max(max_pos_err, pos_err)
        max_orient_err = max(max_orient_err, orient_err)
        if pos_err > FK_MATCH_POSITION_TOL_M or orient_err > FK_MATCH_ORIENTATION_TOL_RAD:
            issues.append(f"sample {i}: stored target pose does not match FK(q_target) (pos_err={pos_err:.3e} m, orient_err={orient_err:.3e} rad)")
            break

    # Recompute distance fields for a deterministic subset and compare.
    check_idx = np.linspace(0, n - 1, min(n, 60)).round().astype(int)
    for i in check_idx:
        pos_dist = float(np.linalg.norm(data["target_position"][i] - data["initial_position"][i]))
        if abs(pos_dist - data["position_distance_m"][i]) > 1e-8:
            issues.append(f"sample {i}: position_distance_m does not match recomputed value")
        joint_dist = float(np.linalg.norm(data["q_target"][i] - data["q_initial"][i]))
        if abs(joint_dist - data["joint_distance_rad"][i]) > 1e-8:
            issues.append(f"sample {i}: joint_distance_rad does not match recomputed value")

    if np.any(data["minimum_initial_limit_margin"] < -1e-9) or np.any(data["minimum_target_limit_margin"] < -1e-9):
        issues.append("limit margin values indicate an operational-limit violation")

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != n:
        issues.append(f"point_ik_manifest.csv has {len(manifest_rows)} rows, expected {n}")
    else:
        for i in (0, n // 2, n - 1):
            row = manifest_rows[i]
            if int(row["sample_id"]) != int(data["sample_id"][i]):
                issues.append(f"manifest row {i}: sample_id mismatch with NPZ")
            if int(row["difficulty_id"]) != int(data["difficulty_id"][i]):
                issues.append(f"manifest row {i}: difficulty_id mismatch with NPZ")

    checksum_payload = json.loads(checksum_path.read_text(encoding="utf-8"))
    npz_entry = next((f for f in checksum_payload["files"] if f["filename"].endswith("point_ik_v1.npz")), None)
    if npz_entry is None:
        issues.append("point_ik_checksum.json missing an entry for point_ik_v1.npz")
    else:
        actual_sha = sha256_file(npz_path)
        if actual_sha != npz_entry["sha256"]:
            issues.append("point_ik_v1.npz SHA256 does not match point_ik_checksum.json")

    difficulty_definition = json.loads(difficulty_path.read_text(encoding="utf-8"))
    for required_key in ("difficulty_groups", "priority_order_highest_first", "criteria", "quantile_thresholds"):
        if required_key not in difficulty_definition:
            issues.append(f"difficulty_definition.json missing key '{required_key}'")

    schema_issues, schemas = validate_schema_files()
    issues.extend(schema_issues)
    if "point_ik_schema.json" in schemas:
        validator = jsonschema.Draft202012Validator(schemas["point_ik_schema.json"])
        for i in check_idx[:10]:
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
            errors = sorted(validator.iter_errors(record), key=str)
            if errors:
                issues.append(f"sample {i} fails point_ik_schema.json: {errors[0].message}")

    return issues


def validate_trajectories():
    issues = []
    schema_issues, schemas = validate_schema_files()
    issues.extend(schema_issues)
    trajectory_validator = jsonschema.Draft202012Validator(schemas["trajectory_schema.json"]) if "trajectory_schema.json" in schemas else None

    manifest_path = TRAJECTORIES_DIR / "trajectory_manifest.csv"
    if not manifest_path.is_file():
        issues.append("missing trajectories/trajectory_manifest.csv")
        return issues
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        manifest_rows = {row["trajectory_id"]: row for row in csv.DictReader(handle)}

    for trajectory_type, trajectory_id in TRAJECTORY_FILES:
        npz_path = TRAJECTORIES_DIR / trajectory_type / f"{trajectory_id}.npz"
        if not npz_path.is_file():
            issues.append(f"missing trajectory file: {npz_path.relative_to(REPO_ROOT).as_posix()}")
            continue
        if trajectory_id not in manifest_rows:
            issues.append(f"trajectory_manifest.csv has no row for '{trajectory_id}'")
            continue
        row = manifest_rows[trajectory_id]

        if not _no_absolute_path(row["file_path"]):
            issues.append(f"'{trajectory_id}': manifest file_path looks absolute: {row['file_path']}")
        resolved = REPO_ROOT / row["file_path"]
        if not resolved.is_file():
            issues.append(f"'{trajectory_id}': manifest file_path does not resolve to a file: {row['file_path']}")

        data = np.load(npz_path, allow_pickle=False)
        for name, dtype in TRAJECTORY_REQUIRED_ARRAYS.items():
            if name not in data.files:
                issues.append(f"{trajectory_id}: missing required array '{name}'")
        if any(f"{trajectory_id}: missing" in msg for msg in issues):
            continue

        n = data["waypoint_id"].shape[0]
        for name in TRAJECTORY_REQUIRED_ARRAYS:
            if data[name].shape[0] != n:
                issues.append(f"{trajectory_id}: array '{name}' has length {data[name].shape[0]}, expected {n}")
            if np.issubdtype(data[name].dtype, np.floating) and not np.all(np.isfinite(data[name])):
                issues.append(f"{trajectory_id}: array '{name}' contains NaN/Inf")

        if int(row["num_waypoints"]) != n:
            issues.append(f"{trajectory_id}: manifest num_waypoints={row['num_waypoints']} != NPZ length {n}")

        if len(np.unique(data["waypoint_id"])) != n or not np.array_equal(data["waypoint_id"], np.arange(n)):
            issues.append(f"{trajectory_id}: waypoint_id is not 0..N-1 unique/sorted")

        if not np.all(np.diff(data["time_s"]) > 0):
            issues.append(f"{trajectory_id}: time_s is not strictly increasing")

        s = data["path_parameter_s"]
        if np.min(s) < -1e-9 or np.max(s) > 1.0 + 1e-9 or np.any(np.diff(s) < -1e-9):
            issues.append(f"{trajectory_id}: path_parameter_s is not within [0,1] and non-decreasing")

        quat_norms = np.linalg.norm(data["target_quaternion"], axis=1)
        if np.max(np.abs(quat_norms - 1.0)) > QUATERNION_NORM_TOL:
            issues.append(f"{trajectory_id}: target_quaternion is not unit-norm")

        dots = np.sum(data["target_quaternion"][1:] * data["target_quaternion"][:-1], axis=1)
        if np.min(dots) < CONTINUITY_MIN_DOT:
            issues.append(f"{trajectory_id}: quaternion sequence is discontinuous (min consecutive dot={np.min(dots):.4f})")

        if np.linalg.norm(data["target_linear_velocity"][0]) > ENDPOINT_ZERO_TOL or np.linalg.norm(data["target_linear_velocity"][-1]) > ENDPOINT_ZERO_TOL:
            issues.append(f"{trajectory_id}: first/last target_linear_velocity is not ~0")
        if np.linalg.norm(data["target_linear_acceleration"][0]) > ENDPOINT_ZERO_TOL or np.linalg.norm(data["target_linear_acceleration"][-1]) > ENDPOINT_ZERO_TOL:
            issues.append(f"{trajectory_id}: first/last target_linear_acceleration is not ~0")

        actual_sha = sha256_file(npz_path)
        if actual_sha != row["sha256"]:
            issues.append(f"{trajectory_id}: SHA256 does not match trajectory_manifest.csv")

        if row["generation_status"] != "validated":
            issues.append(f"{trajectory_id}: generation_status is '{row['generation_status']}', expected 'validated'")
        success_rate = float(row["validation_waypoint_success_rate"])
        if success_rate < 1.0:
            issues.append(f"{trajectory_id}: validation_waypoint_success_rate={success_rate} < 1.0 (not 100% DLS reachable)")

        if trajectory_validator is not None:
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
            errors = sorted(trajectory_validator.iter_errors(record), key=str)
            if errors:
                issues.append(f"{trajectory_id} fails trajectory_schema.json: {errors[0].message}")

    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    model_context = get_model_context()
    point_ik_issues = validate_point_ik_benchmark(model_context)
    trajectory_issues = validate_trajectories()

    all_issues = point_ik_issues + trajectory_issues
    if all_issues:
        print(f"[validate_generated_targets] FAIL ({len(all_issues)} issue(s)):")
        for issue in all_issues:
            print(f"  - {issue}")
        sys.exit(1)

    print("[validate_generated_targets] PASS: point-IK benchmark and all 8 trajectories are valid.")
    sys.exit(0)


if __name__ == "__main__":
    main()
