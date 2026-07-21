"""Generates Tier 0 FK/Jacobian/singularity validation joint states from the real KR810 model.

Writes three deterministic, seed-driven NPZ files under benchmarks/validation/:
    fk_test_states.npz          - joint states for FK sanity/consistency checks
    jacobian_test_states.npz    - joint states for analytic-vs-finite-difference Jacobian checks
    singularity_test_states.npz - joint states labeled by their *actually computed* sigma_min

All group labels are derived from real model quantities (operational limits, computed
Jacobian singular values) rather than assumed; near-limit samples keep a safety margin
away from the bound so a finite-difference perturbation cannot cross the operational range.
"""

import argparse

import numpy as np

from generators._common import (
    GENERATOR_VERSION,
    REPO_ROOT,
    build_checksum_entry,
    get_model_context,
    load_benchmark_config,
    load_dls_config,
    relative_to_repo,
    rng_from,
    save_npz,
    write_json,
)
from kinematics.jacobian import geometric_jacobian_world
from kinematics.singularity_metrics import condition_number, minimum_singular_value

OUTPUT_DIR = REPO_ROOT / "benchmarks" / "validation"

NEAR_LIMIT_MARGIN_RAD = 0.05
NEAR_LIMIT_BAND_RAD = 0.05
INTERIOR_MARGIN_RAD = 0.15
HOME_PERTURBATION_RAD = 0.10

FK_GROUPS = {
    0: "zero_or_home",
    1: "random_interior",
    2: "near_operational_lower_limit",
    3: "near_operational_upper_limit",
    4: "mixed_near_limits",
}

JACOBIAN_GROUPS = {
    0: "regular",
    1: "near_lower_limit",
    2: "near_upper_limit",
    3: "mixed_near_limits",
    4: "low_sigma",
}

SINGULARITY_GROUPS = {
    0: "regular",
    1: "moderately_conditioned",
    2: "near_singular",
}


def _group_zero_or_home(rng, nq, count, lower, upper):
    q = np.zeros((count, nq), dtype=np.float64)
    if count > 1:
        q[1:] = rng.uniform(-HOME_PERTURBATION_RAD, HOME_PERTURBATION_RAD, size=(count - 1, nq))
    return np.clip(q, lower + 1e-3, upper - 1e-3)


def _group_random_interior(rng, nq, count, lower, upper, margin=INTERIOR_MARGIN_RAD):
    lo = lower + margin
    hi = upper - margin
    return rng.uniform(lo, hi, size=(count, nq))


def _group_near_lower(rng, nq, count, lower, upper, margin=NEAR_LIMIT_MARGIN_RAD, band=NEAR_LIMIT_BAND_RAD):
    lo = lower + margin
    hi = lower + margin + band
    return rng.uniform(lo, hi, size=(count, nq))


def _group_near_upper(rng, nq, count, lower, upper, margin=NEAR_LIMIT_MARGIN_RAD, band=NEAR_LIMIT_BAND_RAD):
    lo = upper - margin - band
    hi = upper - margin
    return rng.uniform(lo, hi, size=(count, nq))


def _group_mixed_near_limits(
    rng, nq, count, lower, upper, margin=NEAR_LIMIT_MARGIN_RAD, band=NEAR_LIMIT_BAND_RAD, interior_margin=INTERIOR_MARGIN_RAD
):
    choices = rng.integers(0, 3, size=(count, nq))
    lower_offsets = rng.uniform(0.0, band, size=(count, nq))
    upper_offsets = rng.uniform(0.0, band, size=(count, nq))
    interior_values = rng.uniform(lower + interior_margin, upper - interior_margin, size=(count, nq))
    near_lower_values = lower + margin + lower_offsets
    near_upper_values = upper - margin - upper_offsets
    return np.where(choices == 0, near_lower_values, np.where(choices == 1, near_upper_values, interior_values))


def generate_fk_states(model_context, benchmark_config, seed, total=None):
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    nq = model_context.nq

    total = int(total if total is not None else benchmark_config["validation_fk_samples"])
    n_groups = len(FK_GROUPS)
    if total % n_groups != 0:
        raise ValueError(f"validation_fk_samples ({total}) must be divisible by {n_groups} groups")
    per_group = total // n_groups

    builders = {
        0: _group_zero_or_home,
        1: _group_random_interior,
        2: _group_near_lower,
        3: _group_near_upper,
        4: _group_mixed_near_limits,
    }

    q_chunks, group_chunks, seed_chunks = [], [], []
    for group_id in sorted(FK_GROUPS):
        rng = rng_from(seed, 10, group_id)
        derived_seed = seed * 100000 + 10000 + group_id
        qs = builders[group_id](rng, nq, per_group, lower, upper)
        q_chunks.append(qs)
        group_chunks.append(np.full(per_group, group_id, dtype=np.int32))
        seed_chunks.append(np.full(per_group, derived_seed, dtype=np.int64))

    q_samples = np.concatenate(q_chunks, axis=0)
    group_id_arr = np.concatenate(group_chunks, axis=0)
    source_seed = np.concatenate(seed_chunks, axis=0)
    sample_id = np.arange(q_samples.shape[0], dtype=np.int64)

    if not np.all(np.isfinite(q_samples)):
        raise ValueError("generated FK validation states contain non-finite values")
    if np.any(q_samples < lower) or np.any(q_samples > upper):
        raise ValueError("generated FK validation states violate operational limits")

    return {
        "sample_id": sample_id,
        "q_samples": q_samples.astype(np.float64),
        "group_id": group_id_arr,
        "source_seed": source_seed,
    }


def _build_singularity_candidate_pool(rng, nq, lower, upper, pool_size):
    """Uniform-interior candidates plus elbow/wrist-biased candidates (increases singular yield).

    Bias only *proposes* candidates; the actual group assignment always comes from a real
    computed sigma_min, never from the bias itself.
    """
    n_uniform = pool_size // 2
    n_elbow = pool_size // 4
    n_wrist = pool_size - n_uniform - n_elbow

    uniform_pool = rng.uniform(lower + INTERIOR_MARGIN_RAD, upper - INTERIOR_MARGIN_RAD, size=(n_uniform, nq))

    elbow_pool = rng.uniform(lower + INTERIOR_MARGIN_RAD, upper - INTERIOR_MARGIN_RAD, size=(n_elbow, nq))
    elbow_pool[:, 3] = rng.uniform(max(lower[3], -0.05), min(upper[3], 0.05), size=n_elbow)

    wrist_pool = rng.uniform(lower + INTERIOR_MARGIN_RAD, upper - INTERIOR_MARGIN_RAD, size=(n_wrist, nq))
    wrist_pool[:, 5] = rng.uniform(max(lower[5], -0.05), min(upper[5], 0.05), size=n_wrist)

    return np.concatenate([uniform_pool, elbow_pool, wrist_pool], axis=0)


def generate_jacobian_states(model_context, benchmark_config, seed, total=None):
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    nq = model_context.nq
    fd_epsilon = float(benchmark_config["finite_difference_epsilon"])

    total = int(total if total is not None else benchmark_config["validation_jacobian_samples"])
    n_groups = len(JACOBIAN_GROUPS)
    if total % n_groups != 0:
        raise ValueError(f"validation_jacobian_samples ({total}) must be divisible by {n_groups} groups")
    per_group = total // n_groups

    q_chunks, group_chunks, seed_chunks = [], [], []

    for group_id, builder in (
        (0, _group_random_interior),
        (1, _group_near_lower),
        (2, _group_near_upper),
        (3, _group_mixed_near_limits),
    ):
        rng = rng_from(seed, 11, group_id)
        derived_seed = seed * 100000 + 11000 + group_id
        qs = builder(rng, nq, per_group, lower, upper)
        q_chunks.append(qs)
        group_chunks.append(np.full(per_group, group_id, dtype=np.int32))
        seed_chunks.append(np.full(per_group, derived_seed, dtype=np.int64))

    # low_sigma group: draw a large candidate pool (uniform + elbow/wrist-biased), compute the
    # real sigma_min for each candidate, and keep the ones with the smallest actual sigma_min.
    rng_low = rng_from(seed, 11, 4)
    pool_size = max(per_group * 20, 2000)
    candidates = _build_singularity_candidate_pool(rng_low, nq, lower, upper, pool_size)
    sigma_mins = np.array([minimum_singular_value(geometric_jacobian_world(model_context, q)) for q in candidates])
    order = np.argsort(sigma_mins)
    low_sigma_qs = candidates[order[:per_group]]
    q_chunks.append(low_sigma_qs)
    group_chunks.append(np.full(per_group, 4, dtype=np.int32))
    seed_chunks.append(np.full(per_group, seed * 100000 + 11000 + 4, dtype=np.int64))

    q_samples = np.concatenate(q_chunks, axis=0)
    group_id_arr = np.concatenate(group_chunks, axis=0)
    source_seed = np.concatenate(seed_chunks, axis=0)
    sample_id = np.arange(q_samples.shape[0], dtype=np.int64)
    fd_epsilon_arr = np.full(q_samples.shape[0], fd_epsilon, dtype=np.float64)

    if not np.all(np.isfinite(q_samples)):
        raise ValueError("generated Jacobian validation states contain non-finite values")
    if np.any(q_samples < lower) or np.any(q_samples > upper):
        raise ValueError("generated Jacobian validation states violate operational limits")

    return {
        "sample_id": sample_id,
        "q_samples": q_samples.astype(np.float64),
        "group_id": group_id_arr,
        "finite_difference_epsilon": fd_epsilon_arr,
        "source_seed": source_seed,
    }


def generate_singularity_states(model_context, benchmark_config, dls_config, seed, total=None):
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    nq = model_context.nq

    total = int(total if total is not None else benchmark_config["validation_singularity_samples"])
    n_groups = len(SINGULARITY_GROUPS)
    if total % n_groups != 0:
        raise ValueError(f"validation_singularity_samples ({total}) must be divisible by {n_groups} groups")
    per_group = total // n_groups

    threshold = float(dls_config["singularity_sigma_threshold"])
    moderate_upper = threshold * float(benchmark_config["validation_singularity_moderate_upper_multiplier"])
    pool_size = int(benchmark_config["validation_singularity_candidate_pool_size"])

    rng = rng_from(seed, 12, 0)
    candidates = _build_singularity_candidate_pool(rng, nq, lower, upper, pool_size)

    sigma_mins = np.empty(candidates.shape[0], dtype=np.float64)
    condition_numbers = np.empty(candidates.shape[0], dtype=np.float64)
    for i, q in enumerate(candidates):
        J = geometric_jacobian_world(model_context, q)
        sigma_mins[i] = minimum_singular_value(J)
        condition_numbers[i] = condition_number(J)

    near_singular_mask = sigma_mins <= threshold
    moderate_mask = (sigma_mins > threshold) & (sigma_mins <= moderate_upper)
    regular_mask = sigma_mins > moderate_upper

    near_singular_pool_idx = np.flatnonzero(near_singular_mask)
    near_singular_pool_idx = near_singular_pool_idx[np.argsort(sigma_mins[near_singular_pool_idx])]

    moderate_pool_idx = np.flatnonzero(moderate_mask)
    moderate_pool_idx = moderate_pool_idx[np.argsort(sigma_mins[moderate_pool_idx])]

    regular_pool_idx = np.flatnonzero(regular_mask)
    regular_pool_idx = regular_pool_idx[np.argsort(-sigma_mins[regular_pool_idx])]

    def _take(idx_pool, count, label):
        if idx_pool.shape[0] < count:
            raise ValueError(
                f"singularity candidate pool only produced {idx_pool.shape[0]} '{label}' samples, "
                f"need {count}; increase validation_singularity_candidate_pool_size"
            )
        if idx_pool.shape[0] == count:
            return idx_pool
        # Evenly spaced pick across the sorted pool for good within-group coverage.
        pick = np.linspace(0, idx_pool.shape[0] - 1, count).round().astype(int)
        return idx_pool[pick]

    selections = [
        (0, _take(regular_pool_idx, per_group, "regular")),
        (1, _take(moderate_pool_idx, per_group, "moderately_conditioned")),
        (2, _take(near_singular_pool_idx, per_group, "near_singular")),
    ]

    q_chunks, group_chunks, sigma_chunks, cond_chunks, seed_chunks = [], [], [], [], []
    for group_id, idx in selections:
        q_chunks.append(candidates[idx])
        group_chunks.append(np.full(len(idx), group_id, dtype=np.int32))
        sigma_chunks.append(sigma_mins[idx])
        cond_chunks.append(condition_numbers[idx])
        seed_chunks.append(np.full(len(idx), seed * 100000 + 12000 + group_id, dtype=np.int64))

    q_samples = np.concatenate(q_chunks, axis=0)
    group_id_arr = np.concatenate(group_chunks, axis=0)
    sigma_min_arr = np.concatenate(sigma_chunks, axis=0)
    condition_number_arr = np.concatenate(cond_chunks, axis=0)
    source_seed = np.concatenate(seed_chunks, axis=0)
    sample_id = np.arange(q_samples.shape[0], dtype=np.int64)

    if not np.all(np.isfinite(q_samples)):
        raise ValueError("generated singularity validation states contain non-finite values")
    if np.any(q_samples < lower) or np.any(q_samples > upper):
        raise ValueError("generated singularity validation states violate operational limits")
    if not np.all(np.isfinite(sigma_min_arr)) or np.any(sigma_min_arr < 0.0):
        raise ValueError("sigma_min values must be finite and non-negative")

    arrays = {
        "sample_id": sample_id,
        "q_samples": q_samples.astype(np.float64),
        "sigma_min": sigma_min_arr,
        "condition_number": condition_number_arr,
        "group_id": group_id_arr,
        "source_seed": source_seed,
    }
    return arrays, threshold, moderate_upper


def run(seed: int, overwrite: bool, output_dir=OUTPUT_DIR) -> dict:
    model_context = get_model_context()
    benchmark_config = load_benchmark_config()
    dls_config = load_dls_config()

    out_dir = output_dir

    fk_arrays = generate_fk_states(model_context, benchmark_config, seed)
    jacobian_arrays = generate_jacobian_states(model_context, benchmark_config, seed)
    singularity_arrays, threshold, moderate_upper = generate_singularity_states(
        model_context, benchmark_config, dls_config, seed
    )

    fk_path = save_npz(out_dir / "fk_test_states.npz", overwrite, fk_arrays)
    jacobian_path = save_npz(out_dir / "jacobian_test_states.npz", overwrite, jacobian_arrays)
    singularity_path = save_npz(out_dir / "singularity_test_states.npz", overwrite, singularity_arrays)

    checksum_payload = {
        "generator": "generate_fk_validation_states",
        "generator_version": GENERATOR_VERSION,
        "generation_seed": seed,
        "source_config": {
            "benchmark_config": benchmark_config,
            "dls_config_singularity_threshold": dls_config["singularity_sigma_threshold"],
        },
        "group_definitions": {
            "fk_test_states": FK_GROUPS,
            "jacobian_test_states": JACOBIAN_GROUPS,
            "singularity_test_states": SINGULARITY_GROUPS,
        },
        "singularity_thresholds": {
            "singularity_sigma_threshold": threshold,
            "moderately_conditioned_upper_bound": moderate_upper,
            "source": "configs/dls_config.json:singularity_sigma_threshold",
        },
        "files": [
            build_checksum_entry(fk_path, fk_arrays, seed),
            build_checksum_entry(jacobian_path, jacobian_arrays, seed),
            build_checksum_entry(singularity_path, singularity_arrays, seed),
        ],
    }
    write_json(out_dir / "validation_checksum.json", True, checksum_payload)

    print(f"[generate_fk_validation_states] wrote {relative_to_repo(fk_path)} ({fk_arrays['sample_id'].shape[0]} samples)")
    print(
        f"[generate_fk_validation_states] wrote {relative_to_repo(jacobian_path)} "
        f"({jacobian_arrays['sample_id'].shape[0]} samples)"
    )
    print(
        f"[generate_fk_validation_states] wrote {relative_to_repo(singularity_path)} "
        f"({singularity_arrays['sample_id'].shape[0]} samples)"
    )
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
