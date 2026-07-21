"""Generates the Tier 1 point-IK benchmark (200 samples x 6 difficulty groups) from the real KR810 model.

Every target is produced by drawing a valid joint configuration and running it through forward
kinematics, so every target_position/target_quaternion is reachable by construction (never a
freely chosen Cartesian point assumed reachable).

Difficulty groups are assigned from data, not guessed: a large generic pool of (q_initial,
q_target) pairs is sampled, five real quantities are computed for every pair (joint distance,
Cartesian position distance, SO(3) orientation distance, initial/target sigma_min, initial/target
joint-limit margin), and group thresholds are derived from the empirical quantiles of that pool.
See difficulty_definition.json for the exact thresholds and priority rule used to break ties
when a pair would otherwise qualify for more than one group.

Writes:
    benchmarks/point_ik/point_ik_v1.npz
    benchmarks/point_ik/point_ik_manifest.csv
    benchmarks/point_ik/difficulty_definition.json
    benchmarks/point_ik/point_ik_checksum.json
"""

import argparse
import csv

import numpy as np

from generators._common import (
    GENERATOR_VERSION,
    REPO_ROOT,
    build_checksum_entry,
    config_hash,
    ensure_output_path,
    get_model_context,
    load_benchmark_config,
    load_dls_config,
    relative_to_repo,
    rng_from,
    save_npz,
    write_json,
)
from kinematics.forward_kinematics import forward_kinematics
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import minimum_joint_limit_margin
from kinematics.rotation_utils import rotation_geodesic_angle
from kinematics.singularity_metrics import minimum_singular_value
from utils.file_checksum import sha256_file

OUTPUT_DIR = REPO_ROOT / "benchmarks" / "point_ik"

DIFFICULTY_GROUPS = {
    0: "near_target",
    1: "medium_target",
    2: "far_target",
    3: "large_orientation_change",
    4: "near_joint_limit",
    5: "near_singularity",
}
# Higher-priority groups claim a candidate pair first; later groups only get pairs not already
# claimed. This resolves pairs that would otherwise qualify for more than one group.
PRIORITY_ORDER = [5, 4, 3, 2, 1, 0]

INTERIOR_MARGIN_RAD = 0.10
MIN_JOINT_DISTANCE_RAD = 1e-6

ORIENTATION_TOP_QUANTILE = 0.85
LIMIT_MARGIN_BOTTOM_QUANTILE = 0.10
SIGMA_MIN_BOTTOM_QUANTILE = 0.10
POSITION_LOW_QUANTILE = 1.0 / 3.0
POSITION_HIGH_QUANTILE = 2.0 / 3.0

POOL_MAGNITUDE_LOG_MIN = -2.0  # log10(rad), ~0.01 rad average joint delta
POOL_MAGNITUDE_LOG_MAX = 0.5  # log10(rad), ~3.16 rad average joint delta


def _sample_generic_pool(rng, model_context, pool_size):
    """Sample a generic pool of (q_initial, q_target) pairs with a broad spread of separations."""
    nq = model_context.nq
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad

    q_initial = rng.uniform(lower + INTERIOR_MARGIN_RAD, upper - INTERIOR_MARGIN_RAD, size=(pool_size, nq))

    directions = rng.normal(size=(pool_size, nq))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    directions = directions / norms

    log_magnitude = rng.uniform(POOL_MAGNITUDE_LOG_MIN, POOL_MAGNITUDE_LOG_MAX, size=(pool_size, 1))
    magnitude = 10.0**log_magnitude

    q_target = np.clip(q_initial + directions * magnitude, lower, upper)
    return q_initial, q_target


def _compute_pair_metrics(model_context, q_initial_batch, q_target_batch):
    n = q_initial_batch.shape[0]
    initial_position = np.empty((n, 3))
    initial_quaternion = np.empty((n, 4))
    target_position = np.empty((n, 3))
    target_quaternion = np.empty((n, 4))
    position_distance_m = np.empty(n)
    orientation_distance_rad = np.empty(n)
    joint_distance_rad = np.empty(n)
    initial_sigma_min = np.empty(n)
    target_sigma_min = np.empty(n)
    minimum_initial_limit_margin = np.empty(n)
    minimum_target_limit_margin = np.empty(n)

    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    data = model_context.new_data()

    for i in range(n):
        q_i = q_initial_batch[i]
        q_t = q_target_batch[i]

        fk_i = forward_kinematics(model_context, q_i, data=data)
        fk_t = forward_kinematics(model_context, q_t, data=data)
        J_i = geometric_jacobian_world(model_context, q_i, data=data)
        J_t = geometric_jacobian_world(model_context, q_t, data=data)

        initial_position[i] = fk_i.position
        initial_quaternion[i] = fk_i.quaternion_wxyz
        target_position[i] = fk_t.position
        target_quaternion[i] = fk_t.quaternion_wxyz

        position_distance_m[i] = np.linalg.norm(fk_t.position - fk_i.position)
        orientation_distance_rad[i] = rotation_geodesic_angle(fk_i.rotation_matrix, fk_t.rotation_matrix)
        joint_distance_rad[i] = np.linalg.norm(q_t - q_i)
        initial_sigma_min[i] = minimum_singular_value(J_i)
        target_sigma_min[i] = minimum_singular_value(J_t)
        minimum_initial_limit_margin[i] = minimum_joint_limit_margin(q_i, lower, upper)
        minimum_target_limit_margin[i] = minimum_joint_limit_margin(q_t, lower, upper)

    return {
        "initial_position": initial_position,
        "initial_quaternion": initial_quaternion,
        "target_position": target_position,
        "target_quaternion": target_quaternion,
        "position_distance_m": position_distance_m,
        "orientation_distance_rad": orientation_distance_rad,
        "joint_distance_rad": joint_distance_rad,
        "initial_sigma_min": initial_sigma_min,
        "target_sigma_min": target_sigma_min,
        "minimum_initial_limit_margin": minimum_initial_limit_margin,
        "minimum_target_limit_margin": minimum_target_limit_margin,
    }


def _derive_thresholds(metrics):
    position = metrics["position_distance_m"]
    orientation = metrics["orientation_distance_rad"]
    pair_margin = np.minimum(metrics["minimum_initial_limit_margin"], metrics["minimum_target_limit_margin"])
    pair_sigma_min = np.minimum(metrics["initial_sigma_min"], metrics["target_sigma_min"])

    thresholds = {
        "position_distance_m_low_quantile": float(np.quantile(position, POSITION_LOW_QUANTILE)),
        "position_distance_m_high_quantile": float(np.quantile(position, POSITION_HIGH_QUANTILE)),
        "orientation_distance_rad_top_quantile": float(np.quantile(orientation, ORIENTATION_TOP_QUANTILE)),
        "pair_limit_margin_bottom_quantile": float(np.quantile(pair_margin, LIMIT_MARGIN_BOTTOM_QUANTILE)),
        "pair_sigma_min_bottom_quantile": float(np.quantile(pair_sigma_min, SIGMA_MIN_BOTTOM_QUANTILE)),
    }
    return thresholds, pair_margin, pair_sigma_min


def _classify_pool(metrics, thresholds, pair_margin, pair_sigma_min):
    n = metrics["position_distance_m"].shape[0]
    position = metrics["position_distance_m"]
    orientation = metrics["orientation_distance_rad"]
    joint_distance = metrics["joint_distance_rad"]

    is_near_singularity = pair_sigma_min <= thresholds["pair_sigma_min_bottom_quantile"]
    is_near_joint_limit = pair_margin <= thresholds["pair_limit_margin_bottom_quantile"]
    is_large_orientation = orientation >= thresholds["orientation_distance_rad_top_quantile"]
    is_far = position >= thresholds["position_distance_m_high_quantile"]
    is_medium = (position > thresholds["position_distance_m_low_quantile"]) & (
        position < thresholds["position_distance_m_high_quantile"]
    )
    is_near = (position <= thresholds["position_distance_m_low_quantile"]) & (joint_distance > MIN_JOINT_DISTANCE_RAD)

    eligibility = {5: is_near_singularity, 4: is_near_joint_limit, 3: is_large_orientation, 2: is_far, 1: is_medium, 0: is_near}

    assigned = np.full(n, -1, dtype=np.int32)
    claimed = np.zeros(n, dtype=bool)
    for group_id in PRIORITY_ORDER:
        eligible_unclaimed = eligibility[group_id] & (~claimed)
        assigned[eligible_unclaimed] = group_id
        claimed |= eligible_unclaimed

    pool_by_group = {group_id: np.flatnonzero(assigned == group_id) for group_id in DIFFICULTY_GROUPS}
    return pool_by_group


def _select_per_group(rng, pool_by_group, metrics, samples_per_group):
    selected_idx = {}
    for group_id, idx_pool in pool_by_group.items():
        name = DIFFICULTY_GROUPS[group_id]
        if idx_pool.shape[0] < samples_per_group:
            raise ValueError(
                f"difficulty group '{name}' only has {idx_pool.shape[0]} eligible pairs in the pool, "
                f"need {samples_per_group}; increase the generic pool size"
            )
        chosen = rng.choice(idx_pool, size=samples_per_group, replace=False)
        selected_idx[group_id] = np.sort(chosen)
    return selected_idx


def generate_point_ik_dataset(seed: int, samples_per_group=None, pool_size=None):
    model_context = get_model_context()
    benchmark_config = load_benchmark_config()

    samples_per_group = int(samples_per_group if samples_per_group is not None else benchmark_config["point_ik_samples_per_group"])
    n_groups = len(DIFFICULTY_GROUPS)
    pool_size = int(pool_size if pool_size is not None else max(samples_per_group * n_groups * 25, 30000))

    pool_rng = rng_from(seed, 20, 0)
    q_initial_pool, q_target_pool = _sample_generic_pool(pool_rng, model_context, pool_size)
    metrics = _compute_pair_metrics(model_context, q_initial_pool, q_target_pool)

    thresholds, pair_margin, pair_sigma_min = _derive_thresholds(metrics)
    pool_by_group = _classify_pool(metrics, thresholds, pair_margin, pair_sigma_min)

    select_rng = rng_from(seed, 20, 1)
    selected_idx = _select_per_group(select_rng, pool_by_group, metrics, samples_per_group)

    q_initial_chunks, q_target_chunks, difficulty_chunks, seed_chunks = [], [], [], []
    field_chunks = {
        key: []
        for key in (
            "initial_position",
            "initial_quaternion",
            "target_position",
            "target_quaternion",
            "position_distance_m",
            "orientation_distance_rad",
            "joint_distance_rad",
            "initial_sigma_min",
            "target_sigma_min",
            "minimum_initial_limit_margin",
            "minimum_target_limit_margin",
        )
    }

    for group_id in sorted(DIFFICULTY_GROUPS):
        idx = selected_idx[group_id]
        q_initial_chunks.append(q_initial_pool[idx])
        q_target_chunks.append(q_target_pool[idx])
        difficulty_chunks.append(np.full(idx.shape[0], group_id, dtype=np.int32))
        seed_chunks.append(np.full(idx.shape[0], seed * 100000 + 20000 + group_id, dtype=np.int64))
        for key in field_chunks:
            field_chunks[key].append(metrics[key][idx])

    q_initial = np.concatenate(q_initial_chunks, axis=0).astype(np.float64)
    q_target = np.concatenate(q_target_chunks, axis=0).astype(np.float64)
    difficulty_id = np.concatenate(difficulty_chunks, axis=0)
    source_seed = np.concatenate(seed_chunks, axis=0)
    sample_id = np.arange(q_initial.shape[0], dtype=np.int64)

    arrays = {
        "sample_id": sample_id,
        "q_initial": q_initial,
        "q_target": q_target,
        "initial_position": np.concatenate(field_chunks["initial_position"], axis=0).astype(np.float64),
        "initial_quaternion": np.concatenate(field_chunks["initial_quaternion"], axis=0).astype(np.float64),
        "target_position": np.concatenate(field_chunks["target_position"], axis=0).astype(np.float64),
        "target_quaternion": np.concatenate(field_chunks["target_quaternion"], axis=0).astype(np.float64),
        "position_distance_m": np.concatenate(field_chunks["position_distance_m"], axis=0).astype(np.float64),
        "orientation_distance_rad": np.concatenate(field_chunks["orientation_distance_rad"], axis=0).astype(np.float64),
        "joint_distance_rad": np.concatenate(field_chunks["joint_distance_rad"], axis=0).astype(np.float64),
        "initial_sigma_min": np.concatenate(field_chunks["initial_sigma_min"], axis=0).astype(np.float64),
        "target_sigma_min": np.concatenate(field_chunks["target_sigma_min"], axis=0).astype(np.float64),
        "minimum_initial_limit_margin": np.concatenate(field_chunks["minimum_initial_limit_margin"], axis=0).astype(np.float64),
        "minimum_target_limit_margin": np.concatenate(field_chunks["minimum_target_limit_margin"], axis=0).astype(np.float64),
        "difficulty_id": difficulty_id,
        "source_seed": source_seed,
    }

    for key, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            raise ValueError(f"array '{key}' contains non-finite values")

    return arrays, thresholds, pool_size


def _write_manifest(path, arrays, overwrite):
    path = ensure_output_path(path, overwrite)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "difficulty_id",
                "difficulty_name",
                "source_seed",
                "position_distance_m",
                "orientation_distance_deg",
                "joint_distance_rad",
                "initial_sigma_min",
                "target_sigma_min",
                "minimum_initial_limit_margin",
                "minimum_target_limit_margin",
            ]
        )
        n = arrays["sample_id"].shape[0]
        for i in range(n):
            writer.writerow(
                [
                    int(arrays["sample_id"][i]),
                    int(arrays["difficulty_id"][i]),
                    DIFFICULTY_GROUPS[int(arrays["difficulty_id"][i])],
                    int(arrays["source_seed"][i]),
                    f"{arrays['position_distance_m'][i]:.8f}",
                    f"{np.degrees(arrays['orientation_distance_rad'][i]):.6f}",
                    f"{arrays['joint_distance_rad'][i]:.8f}",
                    f"{arrays['initial_sigma_min'][i]:.8f}",
                    f"{arrays['target_sigma_min'][i]:.8f}",
                    f"{arrays['minimum_initial_limit_margin'][i]:.8f}",
                    f"{arrays['minimum_target_limit_margin'][i]:.8f}",
                ]
            )
    return path


def run(seed: int, overwrite: bool, output_dir=OUTPUT_DIR) -> dict:
    benchmark_config = load_benchmark_config()
    arrays, thresholds, pool_size = generate_point_ik_dataset(seed)

    npz_path = save_npz(output_dir / "point_ik_v1.npz", overwrite, arrays)
    manifest_path = _write_manifest(output_dir / "point_ik_manifest.csv", arrays, overwrite)

    counts = {DIFFICULTY_GROUPS[g]: int(np.sum(arrays["difficulty_id"] == g)) for g in DIFFICULTY_GROUPS}

    difficulty_definition = {
        "generator": "generate_point_ik_dataset",
        "generator_version": GENERATOR_VERSION,
        "generation_seed": seed,
        "difficulty_groups": DIFFICULTY_GROUPS,
        "samples_per_group": int(benchmark_config["point_ik_samples_per_group"]),
        "sample_counts": counts,
        "priority_order_highest_first": [DIFFICULTY_GROUPS[g] for g in PRIORITY_ORDER],
        "priority_note": (
            "A candidate pair qualifying for more than one group is assigned to the highest-"
            "priority group in priority_order_highest_first; lower-priority groups only draw "
            "from pairs not already claimed."
        ),
        "generic_pool_size": pool_size,
        "generic_pool_construction": (
            "q_initial sampled uniformly over the operational interior (margin "
            f"{INTERIOR_MARGIN_RAD} rad from each bound); q_target = clip(q_initial + "
            "magnitude * random_unit_direction, lower, upper) with magnitude log-uniformly "
            f"sampled in [10^{POOL_MAGNITUDE_LOG_MIN}, 10^{POOL_MAGNITUDE_LOG_MAX}] rad so the pool "
            "spans a wide range of joint/Cartesian separations."
        ),
        "criteria": {
            "near_target": (
                "position_distance_m <= position_distance_m_low_quantile (33rd percentile of the "
                "generic pool) and joint_distance_rad > 1e-6 (not an identical pose)."
            ),
            "medium_target": (
                "position_distance_m_low_quantile < position_distance_m < "
                "position_distance_m_high_quantile (33rd-66th percentile of the generic pool)."
            ),
            "far_target": "position_distance_m >= position_distance_m_high_quantile (66th percentile of the generic pool).",
            "large_orientation_change": (
                f"orientation_distance_rad >= its {int(ORIENTATION_TOP_QUANTILE * 100)}th percentile "
                "of the generic pool, independent of joint/position distance."
            ),
            "near_joint_limit": (
                "min(minimum_initial_limit_margin, minimum_target_limit_margin) <= its "
                f"{int(LIMIT_MARGIN_BOTTOM_QUANTILE * 100)}th percentile of the generic pool "
                "(normalized margin; never negative, i.e. never an operational-limit violation)."
            ),
            "near_singularity": (
                "min(initial_sigma_min, target_sigma_min) <= its "
                f"{int(SIGMA_MIN_BOTTOM_QUANTILE * 100)}th percentile of the generic pool "
                "(actual computed Jacobian singular value, not an assumed label)."
            ),
        },
        "quantile_thresholds": thresholds,
    }
    difficulty_path = write_json(output_dir / "difficulty_definition.json", True, difficulty_definition)

    config_for_hash = {
        "benchmark_config": benchmark_config,
        "difficulty_groups": DIFFICULTY_GROUPS,
        "priority_order": PRIORITY_ORDER,
    }
    checksum_payload = {
        "generator": "generate_point_ik_dataset",
        "generator_version": GENERATOR_VERSION,
        "generation_seed": seed,
        "config_hash": config_hash(config_for_hash),
        "sample_counts": counts,
        "files": [
            build_checksum_entry(
                npz_path,
                arrays,
                seed,
                extra={"generator_version": GENERATOR_VERSION, "config_hash": config_hash(config_for_hash)},
            ),
            {
                "filename": relative_to_repo(manifest_path),
                "sha256": sha256_file(manifest_path),
                "file_size_bytes": manifest_path.stat().st_size,
            },
            {
                "filename": relative_to_repo(difficulty_path),
                "sha256": sha256_file(difficulty_path),
                "file_size_bytes": difficulty_path.stat().st_size,
            },
        ],
    }
    write_json(output_dir / "point_ik_checksum.json", True, checksum_payload)

    print(f"[generate_point_ik_dataset] wrote {relative_to_repo(npz_path)} ({arrays['sample_id'].shape[0]} samples)")
    print(f"[generate_point_ik_dataset] group counts: {counts}")
    return checksum_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Base seed (default: benchmark_config.json random_seed)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else load_benchmark_config()["random_seed"]
    run(seed=seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
