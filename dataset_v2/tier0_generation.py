"""Dataset v2 Tier 0 generator: FK/Jacobian/singularity validation states.

Writes three deterministic, seed-driven NPZ files under
``<dataset_v2_root>/tier0_validation/`` (locked counts 1000 FK / 1000 Jacobian / 600
singularity, spec section B), plus per-file metadata JSON and a generation report.

This module reuses the already-verified ``kinematics/`` FK, Jacobian, and singularity-metric
implementations unchanged (``forward_kinematics``, ``geometric_jacobian_world``,
``finite_difference_jacobian_world``, ``singularity_metrics``) -- it only adds v2-specific
sampling/orchestration on top, parameterized by an explicit ``dataset_root`` (never CWD, never a
hardcoded path) and seeded from ``configs/seed_policy.json`` via
``generators/_common.py::derive_seed`` (no global ``numpy.random`` state is ever touched).
"""

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from dataset_v2.checksums import build_checksum_manifest
from dataset_v2.config_templates import DATASET_SCHEMA_VERSION, DATASET_VERSION
from dataset_v2.locator import DatasetV2Paths, relative_to_dataset_v2_root, require_dataset_v2_root
from dataset_v2.manifest import apply_tier0_generation_status
from generators._common import derive_seed
from kinematics.forward_kinematics import forward_kinematics
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import minimum_joint_limit_margin
from kinematics.manipulability import positional_manipulability
from kinematics.model_loader import ModelContext, load_model_context
from kinematics.singularity_metrics import condition_number, minimum_singular_value, numerical_rank, singular_values
from utils.config_loader import load_json_config
from utils.dataset_locator import CONFIGS_DIR as V1_CONFIGS_DIR, MODEL_PATH as V1_MODEL_PATH, REPO_ROOT
from utils.file_checksum import sha256_file
from utils.npz_utils import save_npz

GENERATOR_VERSION = "1.0.0"

FK_NPZ_NAME = "fk_test_states_v2.npz"
JACOBIAN_NPZ_NAME = "jacobian_test_states_v2.npz"
SINGULARITY_NPZ_NAME = "singularity_test_states_v2.npz"
FK_METADATA_NAME = "fk_test_states_v2_metadata.json"
JACOBIAN_METADATA_NAME = "jacobian_test_states_v2_metadata.json"
SINGULARITY_METADATA_NAME = "singularity_test_states_v2_metadata.json"
REPORT_NAME = "tier0_generation_report.json"

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

# State-type tags, subordinate to the "tier0" component seed (configs/seed_policy.json
# component_tags["tier0"] == 10 per spec section E's derivation scheme).
STATE_TYPE_TAGS = {"fk": 1, "jacobian": 2, "singularity": 3}


@dataclass(frozen=True)
class Tier0GenerationSettings:
    fk_total: int
    jacobian_total: int
    singularity_total: int
    interior_margin_rad: float
    near_limit_margin_rad: float
    near_limit_band_rad: float
    home_perturbation_rad: float
    finite_difference_epsilon: float
    jacobian_low_sigma_candidate_pool_multiplier: int
    jacobian_low_sigma_candidate_pool_min: int
    singularity_candidate_pool_size: int
    singularity_moderate_upper_multiplier: float


@dataclass(frozen=True)
class Tier0GenerationResult:
    dataset_root: Path
    tier0_validation_dir: Path
    dry_run: bool
    fk_total: int
    jacobian_total: int
    singularity_total: int
    fk_group_counts: Dict[str, int] = field(default_factory=dict)
    jacobian_group_counts: Dict[str, int] = field(default_factory=dict)
    singularity_group_counts: Dict[str, int] = field(default_factory=dict)
    full_locked_counts: bool = False
    report: Optional[dict] = None


def load_singularity_threshold() -> "tuple[float, str]":
    """The near-singular sigma_min threshold, sourced from v1's shared DLS config (reused
    unchanged; Dataset v2 does not duplicate it), per spec section 5's requirement to document
    the threshold's source rather than invent a new one.
    """
    dls_config = load_json_config(V1_CONFIGS_DIR / "dls_config.json")
    threshold = float(dls_config["singularity_sigma_threshold"])
    source = "repo root configs/dls_config.json:singularity_sigma_threshold (v1 shared DLS config, reused unchanged)"
    return threshold, source


def load_tier0_generation_settings(
    paths: DatasetV2Paths,
    fk_total: Optional[int] = None,
    jacobian_total: Optional[int] = None,
    singularity_total: Optional[int] = None,
) -> Tier0GenerationSettings:
    config = load_json_config(paths.configs_dir / "tier0_config.json")
    policy = config["sampling_policy"]
    return Tier0GenerationSettings(
        fk_total=int(fk_total if fk_total is not None else config["fk_state_count"]),
        jacobian_total=int(jacobian_total if jacobian_total is not None else config["jacobian_state_count"]),
        singularity_total=int(singularity_total if singularity_total is not None else config["singularity_state_count"]),
        interior_margin_rad=float(policy["interior_margin_rad"]),
        near_limit_margin_rad=float(policy["near_limit_margin_rad"]),
        near_limit_band_rad=float(policy["near_limit_band_rad"]),
        home_perturbation_rad=float(policy["home_perturbation_rad"]),
        finite_difference_epsilon=float(policy["finite_difference_epsilon"]),
        jacobian_low_sigma_candidate_pool_multiplier=int(policy["jacobian_low_sigma_candidate_pool_multiplier"]),
        jacobian_low_sigma_candidate_pool_min=int(policy["jacobian_low_sigma_candidate_pool_min"]),
        singularity_candidate_pool_size=int(policy["singularity_candidate_pool_size"]),
        singularity_moderate_upper_multiplier=float(policy["singularity_moderate_upper_multiplier"]),
    )


# ---------------------------------------------------------------------------------------------
# Group sampling builders (generation policy, not kinematics math -- FK/Jacobian/singularity
# formulas themselves are never reimplemented here, only called via kinematics/).
# ---------------------------------------------------------------------------------------------


def _group_zero_or_home(rng, nq, count, lower, upper, home_perturbation_rad):
    q = np.zeros((count, nq), dtype=np.float64)
    if count > 1:
        q[1:] = rng.uniform(-home_perturbation_rad, home_perturbation_rad, size=(count - 1, nq))
    return np.clip(q, lower + 1e-3, upper - 1e-3)


def _group_random_interior(rng, nq, count, lower, upper, margin):
    lo = lower + margin
    hi = upper - margin
    return rng.uniform(lo, hi, size=(count, nq))


def _group_near_lower(rng, nq, count, lower, upper, margin, band):
    lo = lower + margin
    hi = lower + margin + band
    return rng.uniform(lo, hi, size=(count, nq))


def _group_near_upper(rng, nq, count, lower, upper, margin, band):
    lo = upper - margin - band
    hi = upper - margin
    return rng.uniform(lo, hi, size=(count, nq))


def _group_mixed_near_limits(rng, nq, count, lower, upper, margin, band, interior_margin):
    choices = rng.integers(0, 3, size=(count, nq))
    lower_offsets = rng.uniform(0.0, band, size=(count, nq))
    upper_offsets = rng.uniform(0.0, band, size=(count, nq))
    interior_values = rng.uniform(lower + interior_margin, upper - interior_margin, size=(count, nq))
    near_lower_values = lower + margin + lower_offsets
    near_upper_values = upper - margin - upper_offsets
    return np.where(choices == 0, near_lower_values, np.where(choices == 1, near_upper_values, interior_values))


def _build_singularity_candidate_pool(rng, nq, lower, upper, pool_size, interior_margin_rad):
    """Uniform-interior candidates plus elbow/wrist-biased candidates (increases singular yield).

    Bias only *proposes* candidates; group assignment always comes from a real computed
    sigma_min, never from the bias itself.
    """
    n_uniform = pool_size // 2
    n_elbow = pool_size // 4
    n_wrist = pool_size - n_uniform - n_elbow

    uniform_pool = rng.uniform(lower + interior_margin_rad, upper - interior_margin_rad, size=(n_uniform, nq))

    elbow_pool = rng.uniform(lower + interior_margin_rad, upper - interior_margin_rad, size=(n_elbow, nq))
    elbow_pool[:, 3] = rng.uniform(max(lower[3], -0.05), min(upper[3], 0.05), size=n_elbow)

    wrist_pool = rng.uniform(lower + interior_margin_rad, upper - interior_margin_rad, size=(n_wrist, nq))
    wrist_pool[:, 5] = rng.uniform(max(lower[5], -0.05), min(upper[5], 0.05), size=n_wrist)

    return np.concatenate([uniform_pool, elbow_pool, wrist_pool], axis=0)


def _unique_rows(q: np.ndarray) -> bool:
    _, first_idx = np.unique(q, axis=0, return_index=True)
    return first_idx.shape[0] == q.shape[0]


def _generate_unique_group(rng, nq, count, builder, args, max_attempts=25) -> np.ndarray:
    """Draw ``count`` rows from ``builder``, redrawing any exact-duplicate rows in place until
    the group is duplicate-free (or raising after ``max_attempts`` -- never silently keeps a
    duplicate, per spec section 3/4's "loại exact duplicates").
    """
    if count == 0:
        return np.zeros((0, nq), dtype=np.float64)
    q = np.asarray(builder(rng, nq, count, *args), dtype=np.float64)
    for _ in range(max_attempts):
        _, first_idx = np.unique(q, axis=0, return_index=True)
        if first_idx.shape[0] == q.shape[0]:
            return q
        dup_mask = np.ones(q.shape[0], dtype=bool)
        dup_mask[first_idx] = False
        n_dup = int(dup_mask.sum())
        q[dup_mask] = builder(rng, nq, n_dup, *args)
    raise ValueError("could not eliminate duplicate joint states within a Tier 0 group after max redraw attempts")


def _select_top_k_unique(sorted_candidates: np.ndarray, k: int, label: str) -> np.ndarray:
    seen = set()
    selected = []
    for row in sorted_candidates:
        key = row.tobytes()
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) == k:
            break
    if len(selected) < k:
        raise ValueError(
            f"only found {len(selected)} unique '{label}' candidates, need {k}; increase the "
            "candidate pool size rather than allowing duplicates"
        )
    return np.array(selected, dtype=np.float64)


def _assert_finite_and_within_limits(q_samples, lower, upper, label):
    if not np.all(np.isfinite(q_samples)):
        raise ValueError(f"generated {label} states contain non-finite values")
    if np.any(q_samples < lower) or np.any(q_samples > upper):
        raise ValueError(f"generated {label} states violate operational limits")


def _assert_no_duplicates(q_samples, label):
    if not _unique_rows(q_samples):
        raise ValueError(f"generated {label} states contain duplicate joint configurations across groups")


def _group_counts(group_id_arr: np.ndarray, groups: Dict[int, str]) -> Dict[str, int]:
    return {name: int(np.sum(group_id_arr == gid)) for gid, name in groups.items()}


# ---------------------------------------------------------------------------------------------
# Per-state-type generation
# ---------------------------------------------------------------------------------------------


def generate_fk_states(model_context: ModelContext, seed: int, total: int, settings: Tier0GenerationSettings):
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    nq = model_context.nq

    n_groups = len(FK_GROUPS)
    if total % n_groups != 0:
        raise ValueError(f"fk_state_count ({total}) must be divisible by {n_groups} groups")
    per_group = total // n_groups

    builders = {
        0: (_group_zero_or_home, (settings.home_perturbation_rad,)),
        1: (_group_random_interior, (settings.interior_margin_rad,)),
        2: (_group_near_lower, (settings.near_limit_margin_rad, settings.near_limit_band_rad)),
        3: (_group_near_upper, (settings.near_limit_margin_rad, settings.near_limit_band_rad)),
        4: (
            _group_mixed_near_limits,
            (settings.near_limit_margin_rad, settings.near_limit_band_rad, settings.interior_margin_rad),
        ),
    }

    q_chunks, group_chunks, seed_chunks = [], [], []
    group_seeds: Dict[str, int] = {}
    for group_id in sorted(FK_GROUPS):
        group_seed = derive_seed(seed, group_id)
        group_seeds[FK_GROUPS[group_id]] = group_seed
        rng = np.random.default_rng(group_seed)
        builder, extra_args = builders[group_id]
        qs = _generate_unique_group(rng, nq, per_group, builder, (lower, upper) + extra_args)
        q_chunks.append(qs)
        group_chunks.append(np.full(per_group, group_id, dtype=np.int32))
        seed_chunks.append(np.full(per_group, group_seed, dtype=np.int64))

    q_samples = np.concatenate(q_chunks, axis=0)
    group_id_arr = np.concatenate(group_chunks, axis=0)
    source_seed = np.concatenate(seed_chunks, axis=0)
    sample_id = np.arange(q_samples.shape[0], dtype=np.int64)

    _assert_finite_and_within_limits(q_samples, lower, upper, "FK")
    _assert_no_duplicates(q_samples, "FK")

    data = model_context.new_data()
    positions = np.empty((q_samples.shape[0], 3), dtype=np.float64)
    quaternions = np.empty((q_samples.shape[0], 4), dtype=np.float64)
    margins = np.empty(q_samples.shape[0], dtype=np.float64)
    for i in range(q_samples.shape[0]):
        fk = forward_kinematics(model_context, q_samples[i], data=data)
        positions[i] = fk.position
        quaternions[i] = fk.quaternion_wxyz
        margins[i] = minimum_joint_limit_margin(q_samples[i], lower, upper)

    arrays = {
        "sample_id": sample_id,
        "q_samples": q_samples,
        "group_id": group_id_arr,
        "source_seed": source_seed,
        "ee_position": positions,
        "ee_quaternion_wxyz": quaternions,
        "minimum_joint_limit_margin": margins,
    }
    return arrays, group_seeds


def generate_jacobian_states(
    model_context: ModelContext,
    seed: int,
    total: int,
    settings: Tier0GenerationSettings,
    candidate_pool_size_override: Optional[int] = None,
):
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    nq = model_context.nq

    n_groups = len(JACOBIAN_GROUPS)
    if total % n_groups != 0:
        raise ValueError(f"jacobian_state_count ({total}) must be divisible by {n_groups} groups")
    per_group = total // n_groups

    builders = {
        0: (_group_random_interior, (settings.interior_margin_rad,)),
        1: (_group_near_lower, (settings.near_limit_margin_rad, settings.near_limit_band_rad)),
        2: (_group_near_upper, (settings.near_limit_margin_rad, settings.near_limit_band_rad)),
        3: (
            _group_mixed_near_limits,
            (settings.near_limit_margin_rad, settings.near_limit_band_rad, settings.interior_margin_rad),
        ),
    }

    q_chunks, group_chunks, seed_chunks = [], [], []
    group_seeds: Dict[str, int] = {}
    for group_id in (0, 1, 2, 3):
        group_seed = derive_seed(seed, group_id)
        group_seeds[JACOBIAN_GROUPS[group_id]] = group_seed
        rng = np.random.default_rng(group_seed)
        builder, extra_args = builders[group_id]
        qs = _generate_unique_group(rng, nq, per_group, builder, (lower, upper) + extra_args)
        q_chunks.append(qs)
        group_chunks.append(np.full(per_group, group_id, dtype=np.int32))
        seed_chunks.append(np.full(per_group, group_seed, dtype=np.int64))

    # low_sigma (group 4): draw a real candidate pool, compute actual sigma_min for each, keep
    # the smallest -- never a name-only selection (spec section 4).
    group_id = 4
    group_seed = derive_seed(seed, group_id)
    group_seeds[JACOBIAN_GROUPS[group_id]] = group_seed
    rng = np.random.default_rng(group_seed)
    pool_size = (
        candidate_pool_size_override
        if candidate_pool_size_override is not None
        else max(per_group * settings.jacobian_low_sigma_candidate_pool_multiplier, settings.jacobian_low_sigma_candidate_pool_min)
    )
    candidates = _build_singularity_candidate_pool(rng, nq, lower, upper, pool_size, settings.interior_margin_rad)
    data = model_context.new_data()
    candidate_sigma_mins = np.array(
        [minimum_singular_value(geometric_jacobian_world(model_context, q, data=data)) for q in candidates]
    )
    order = np.argsort(candidate_sigma_mins)
    low_sigma_qs = _select_top_k_unique(candidates[order], per_group, "low_sigma")
    q_chunks.append(low_sigma_qs)
    group_chunks.append(np.full(per_group, group_id, dtype=np.int32))
    seed_chunks.append(np.full(per_group, group_seed, dtype=np.int64))

    q_samples = np.concatenate(q_chunks, axis=0)
    group_id_arr = np.concatenate(group_chunks, axis=0)
    source_seed = np.concatenate(seed_chunks, axis=0)
    sample_id = np.arange(q_samples.shape[0], dtype=np.int64)
    fd_epsilon_arr = np.full(q_samples.shape[0], settings.finite_difference_epsilon, dtype=np.float64)

    _assert_finite_and_within_limits(q_samples, lower, upper, "Jacobian")
    _assert_no_duplicates(q_samples, "Jacobian")

    sigma_min = np.empty(q_samples.shape[0], dtype=np.float64)
    sigma_max = np.empty(q_samples.shape[0], dtype=np.float64)
    cond = np.empty(q_samples.shape[0], dtype=np.float64)
    rank = np.empty(q_samples.shape[0], dtype=np.int32)
    margin = np.empty(q_samples.shape[0], dtype=np.float64)
    for i in range(q_samples.shape[0]):
        J = geometric_jacobian_world(model_context, q_samples[i], data=data)
        sv = singular_values(J)
        sigma_min[i] = float(sv[-1])
        sigma_max[i] = float(sv[0])
        cond[i] = condition_number(J)
        rank[i] = numerical_rank(J)
        margin[i] = minimum_joint_limit_margin(q_samples[i], lower, upper)

    arrays = {
        "sample_id": sample_id,
        "q_samples": q_samples,
        "group_id": group_id_arr,
        "source_seed": source_seed,
        "finite_difference_epsilon": fd_epsilon_arr,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": cond,
        "numerical_rank": rank,
        "minimum_joint_limit_margin": margin,
    }
    return arrays, group_seeds


def generate_singularity_states(
    model_context: ModelContext,
    seed: int,
    total: int,
    settings: Tier0GenerationSettings,
    threshold: float,
    candidate_pool_size_override: Optional[int] = None,
):
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    nq = model_context.nq

    n_groups = len(SINGULARITY_GROUPS)
    if total % n_groups != 0:
        raise ValueError(f"singularity_state_count ({total}) must be divisible by {n_groups} groups")
    per_group = total // n_groups

    moderate_upper = threshold * settings.singularity_moderate_upper_multiplier
    pool_size = candidate_pool_size_override if candidate_pool_size_override is not None else settings.singularity_candidate_pool_size

    group_seed = derive_seed(seed, 0)
    group_seeds = {"pool": group_seed}
    rng = np.random.default_rng(group_seed)
    candidates = _build_singularity_candidate_pool(rng, nq, lower, upper, pool_size, settings.interior_margin_rad)

    data = model_context.new_data()
    sigma_mins = np.empty(candidates.shape[0], dtype=np.float64)
    condition_numbers = np.empty(candidates.shape[0], dtype=np.float64)
    for i in range(candidates.shape[0]):
        J = geometric_jacobian_world(model_context, candidates[i], data=data)
        sigma_mins[i] = minimum_singular_value(J)
        condition_numbers[i] = condition_number(J)

    near_mask = sigma_mins <= threshold
    moderate_mask = (sigma_mins > threshold) & (sigma_mins <= moderate_upper)
    regular_mask = sigma_mins > moderate_upper

    near_idx = np.flatnonzero(near_mask)
    near_idx = near_idx[np.argsort(sigma_mins[near_idx])]
    moderate_idx = np.flatnonzero(moderate_mask)
    moderate_idx = moderate_idx[np.argsort(sigma_mins[moderate_idx])]
    regular_idx = np.flatnonzero(regular_mask)
    regular_idx = regular_idx[np.argsort(-sigma_mins[regular_idx])]

    def _take(idx_pool, count, label):
        if idx_pool.shape[0] < count:
            raise ValueError(
                f"singularity candidate pool only produced {idx_pool.shape[0]} '{label}' sample(s), "
                f"need {count} (pool_size={pool_size}; pool distribution: "
                f"regular={regular_idx.shape[0]}, moderately_conditioned={moderate_idx.shape[0]}, "
                f"near_singular={near_idx.shape[0]}); increase singularity_candidate_pool_size in "
                "tier0_config.json rather than relaxing thresholds or duplicating states"
            )
        if idx_pool.shape[0] == count:
            return idx_pool
        pick = np.linspace(0, idx_pool.shape[0] - 1, count).round().astype(int)
        return idx_pool[pick]

    selections = [
        (0, _take(regular_idx, per_group, "regular")),
        (1, _take(moderate_idx, per_group, "moderately_conditioned")),
        (2, _take(near_idx, per_group, "near_singular")),
    ]

    q_chunks, group_chunks, sigma_min_chunks, sigma_max_chunks, cond_chunks, rank_chunks = [], [], [], [], [], []
    manip_chunks, margin_chunks, seed_chunks = [], [], []
    for group_id, idx in selections:
        qs = candidates[idx]
        q_chunks.append(qs)
        group_chunks.append(np.full(len(idx), group_id, dtype=np.int32))
        sigma_min_chunks.append(sigma_mins[idx])
        sigma_max_local = np.empty(len(idx), dtype=np.float64)
        rank_local = np.empty(len(idx), dtype=np.int32)
        manip_local = np.empty(len(idx), dtype=np.float64)
        margin_local = np.empty(len(idx), dtype=np.float64)
        for j, q in enumerate(qs):
            J = geometric_jacobian_world(model_context, q, data=data)
            sv = singular_values(J)
            sigma_max_local[j] = float(sv[0])
            rank_local[j] = numerical_rank(J)
            manip_local[j] = positional_manipulability(J)
            margin_local[j] = minimum_joint_limit_margin(q, lower, upper)
        sigma_max_chunks.append(sigma_max_local)
        cond_chunks.append(condition_numbers[idx])
        rank_chunks.append(rank_local)
        manip_chunks.append(manip_local)
        margin_chunks.append(margin_local)
        seed_chunks.append(np.full(len(idx), group_seed, dtype=np.int64))

    q_samples = np.concatenate(q_chunks, axis=0)
    group_id_arr = np.concatenate(group_chunks, axis=0)
    sigma_min_arr = np.concatenate(sigma_min_chunks, axis=0)
    sigma_max_arr = np.concatenate(sigma_max_chunks, axis=0)
    condition_number_arr = np.concatenate(cond_chunks, axis=0)
    rank_arr = np.concatenate(rank_chunks, axis=0)
    manipulability_arr = np.concatenate(manip_chunks, axis=0)
    margin_arr = np.concatenate(margin_chunks, axis=0)
    source_seed = np.concatenate(seed_chunks, axis=0)
    sample_id = np.arange(q_samples.shape[0], dtype=np.int64)

    _assert_finite_and_within_limits(q_samples, lower, upper, "singularity")
    _assert_no_duplicates(q_samples, "singularity")
    if not np.all(np.isfinite(sigma_min_arr)) or np.any(sigma_min_arr < 0.0):
        raise ValueError("sigma_min values must be finite and non-negative")
    if np.any(np.isnan(condition_number_arr)):
        raise ValueError("condition_number must never be NaN (use inf for a singular Jacobian)")

    arrays = {
        "sample_id": sample_id,
        "q_samples": q_samples,
        "group_id": group_id_arr,
        "sigma_min": sigma_min_arr,
        "sigma_max": sigma_max_arr,
        "condition_number": condition_number_arr,
        "numerical_rank": rank_arr,
        "manipulability": manipulability_arr,
        "minimum_joint_limit_margin": margin_arr,
        "source_seed": source_seed,
    }
    return arrays, group_seeds, moderate_upper


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _array_metadata(arrays: Dict[str, np.ndarray]) -> dict:
    return {name: {"shape": list(arr.shape), "dtype": str(arr.dtype)} for name, arr in arrays.items()}


def _atomic_write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp.json")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path


def run_tier0_generation(
    dataset_root,
    master_seed: Optional[int] = None,
    overwrite: bool = False,
    fk_total: Optional[int] = None,
    jacobian_total: Optional[int] = None,
    singularity_total: Optional[int] = None,
    jacobian_candidate_pool_size: Optional[int] = None,
    singularity_candidate_pool_size: Optional[int] = None,
    model_context: Optional[ModelContext] = None,
    dry_run: bool = False,
) -> Tier0GenerationResult:
    """Generate Dataset v2 Tier 0 FK/Jacobian/singularity validation states.

    ``fk_total``/``jacobian_total``/``singularity_total`` and the two candidate-pool-size
    overrides exist for tests/smoke runs only -- the locked full mode is 1000/1000/600
    (``specs/DLS_DATASET_V2_SPEC.md`` section B); passing them does not relax any acceptance
    threshold, only the sample counts/pool sizes.
    """
    paths = require_dataset_v2_root(dataset_root)
    tier0_dir = paths.tier0_validation_dir

    output_paths = [
        tier0_dir / FK_NPZ_NAME,
        tier0_dir / JACOBIAN_NPZ_NAME,
        tier0_dir / SINGULARITY_NPZ_NAME,
        tier0_dir / FK_METADATA_NAME,
        tier0_dir / JACOBIAN_METADATA_NAME,
        tier0_dir / SINGULARITY_METADATA_NAME,
        tier0_dir / REPORT_NAME,
    ]
    existing = [p for p in output_paths if p.is_file()]
    if existing and not overwrite:
        existing_relative = ", ".join(str(p.relative_to(paths.root)) for p in existing)
        raise FileExistsError(
            f"Tier 0 v2 output already exists ({existing_relative}); pass overwrite=True "
            "(--overwrite on the CLI) to regenerate it."
        )

    seed_policy = load_json_config(paths.configs_dir / "seed_policy.json")
    resolved_master_seed = int(master_seed if master_seed is not None else seed_policy["master_seed"])
    tier0_component_tag = int(seed_policy["component_tags"]["tier0"])
    tier0_component_seed = derive_seed(resolved_master_seed, tier0_component_tag)

    settings = load_tier0_generation_settings(paths, fk_total, jacobian_total, singularity_total)
    threshold, threshold_source = load_singularity_threshold()
    full_locked_counts = bool(settings.fk_total == 1000 and settings.jacobian_total == 1000 and settings.singularity_total == 600)

    if dry_run:
        n_fk_groups, n_jac_groups, n_sing_groups = len(FK_GROUPS), len(JACOBIAN_GROUPS), len(SINGULARITY_GROUPS)
        return Tier0GenerationResult(
            dataset_root=paths.root,
            tier0_validation_dir=tier0_dir,
            dry_run=True,
            fk_total=settings.fk_total,
            jacobian_total=settings.jacobian_total,
            singularity_total=settings.singularity_total,
            fk_group_counts={name: settings.fk_total // n_fk_groups for name in FK_GROUPS.values()},
            jacobian_group_counts={name: settings.jacobian_total // n_jac_groups for name in JACOBIAN_GROUPS.values()},
            singularity_group_counts={name: settings.singularity_total // n_sing_groups for name in SINGULARITY_GROUPS.values()},
            full_locked_counts=full_locked_counts,
            report=None,
        )

    model_context = model_context if model_context is not None else load_model_context()

    fk_seed = derive_seed(tier0_component_seed, STATE_TYPE_TAGS["fk"])
    jacobian_seed = derive_seed(tier0_component_seed, STATE_TYPE_TAGS["jacobian"])
    singularity_seed = derive_seed(tier0_component_seed, STATE_TYPE_TAGS["singularity"])

    fk_arrays, fk_group_seeds = generate_fk_states(model_context, fk_seed, settings.fk_total, settings)
    jacobian_arrays, jacobian_group_seeds = generate_jacobian_states(
        model_context, jacobian_seed, settings.jacobian_total, settings, jacobian_candidate_pool_size
    )
    singularity_arrays, singularity_group_seeds, moderate_upper = generate_singularity_states(
        model_context, singularity_seed, settings.singularity_total, settings, threshold, singularity_candidate_pool_size
    )

    fk_path = save_npz(tier0_dir / FK_NPZ_NAME, fk_arrays, overwrite=overwrite)
    jacobian_path = save_npz(tier0_dir / JACOBIAN_NPZ_NAME, jacobian_arrays, overwrite=overwrite)
    singularity_path = save_npz(tier0_dir / SINGULARITY_NPZ_NAME, singularity_arrays, overwrite=overwrite)

    fk_group_counts = _group_counts(fk_arrays["group_id"], FK_GROUPS)
    jacobian_group_counts = _group_counts(jacobian_arrays["group_id"], JACOBIAN_GROUPS)
    singularity_group_counts = _group_counts(singularity_arrays["group_id"], SINGULARITY_GROUPS)

    asset_fingerprint = sha256_file(V1_MODEL_PATH)
    git_commit = _git_commit(REPO_ROOT)
    generated_at = datetime.now(timezone.utc).isoformat()

    common_meta = {
        "dataset_version": DATASET_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "git_commit": git_commit,
        "asset_fingerprint": asset_fingerprint,
        "asset_fingerprint_source": "assets/kr810.xml (v1 shared asset, reused unchanged)",
        "master_seed": resolved_master_seed,
        "tier0_component_seed": tier0_component_seed,
        "generation_timestamp_utc": generated_at,
        "generation_status": "development",
    }

    fk_metadata = {
        **common_meta,
        "state_type": "fk",
        "seed": fk_seed,
        "group_seeds": fk_group_seeds,
        "counts": {"total": int(fk_arrays["sample_id"].shape[0]), "requested_total": settings.fk_total},
        "group_counts": fk_group_counts,
        "arrays": _array_metadata(fk_arrays),
        "output_sha256": sha256_file(fk_path),
    }
    _atomic_write_json(tier0_dir / FK_METADATA_NAME, fk_metadata)

    jacobian_metadata = {
        **common_meta,
        "state_type": "jacobian",
        "seed": jacobian_seed,
        "group_seeds": jacobian_group_seeds,
        "counts": {"total": int(jacobian_arrays["sample_id"].shape[0]), "requested_total": settings.jacobian_total},
        "group_counts": jacobian_group_counts,
        "finite_difference_epsilon": settings.finite_difference_epsilon,
        "arrays": _array_metadata(jacobian_arrays),
        "output_sha256": sha256_file(jacobian_path),
    }
    _atomic_write_json(tier0_dir / JACOBIAN_METADATA_NAME, jacobian_metadata)

    singularity_metadata = {
        **common_meta,
        "state_type": "singularity",
        "seed": singularity_seed,
        "group_seeds": singularity_group_seeds,
        "counts": {"total": int(singularity_arrays["sample_id"].shape[0]), "requested_total": settings.singularity_total},
        "group_counts": singularity_group_counts,
        "singularity_threshold": threshold,
        "singularity_threshold_source": threshold_source,
        "moderately_conditioned_upper_bound": moderate_upper,
        "arrays": _array_metadata(singularity_arrays),
        "output_sha256": sha256_file(singularity_path),
    }
    _atomic_write_json(tier0_dir / SINGULARITY_METADATA_NAME, singularity_metadata)

    report = {
        **common_meta,
        "component": "tier0",
        "seed_derivation": {
            "tier0_component_tag": tier0_component_tag,
            "tier0_component_seed": tier0_component_seed,
            "state_type_tags": STATE_TYPE_TAGS,
            "fk_seed": fk_seed,
            "jacobian_seed": jacobian_seed,
            "singularity_seed": singularity_seed,
            "fk_group_seeds": fk_group_seeds,
            "jacobian_group_seeds": jacobian_group_seeds,
            "singularity_group_seeds": singularity_group_seeds,
        },
        "counts": {
            "fk": {"total": int(fk_arrays["sample_id"].shape[0]), "group_counts": fk_group_counts},
            "jacobian": {"total": int(jacobian_arrays["sample_id"].shape[0]), "group_counts": jacobian_group_counts},
            "singularity": {"total": int(singularity_arrays["sample_id"].shape[0]), "group_counts": singularity_group_counts},
        },
        "full_locked_counts": full_locked_counts,
        "singularity_threshold": threshold,
        "singularity_threshold_source": threshold_source,
        "moderately_conditioned_upper_bound": moderate_upper,
        "sampling_policy": {
            "interior_margin_rad": settings.interior_margin_rad,
            "near_limit_margin_rad": settings.near_limit_margin_rad,
            "near_limit_band_rad": settings.near_limit_band_rad,
            "home_perturbation_rad": settings.home_perturbation_rad,
            "finite_difference_epsilon": settings.finite_difference_epsilon,
        },
        "output_files": {
            "fk": {"filename": relative_to_dataset_v2_root(fk_path, paths.root), "sha256": sha256_file(fk_path)},
            "jacobian": {
                "filename": relative_to_dataset_v2_root(jacobian_path, paths.root),
                "sha256": sha256_file(jacobian_path),
            },
            "singularity": {
                "filename": relative_to_dataset_v2_root(singularity_path, paths.root),
                "sha256": sha256_file(singularity_path),
            },
        },
    }
    _atomic_write_json(tier0_dir / REPORT_NAME, report)

    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    manifest = apply_tier0_generation_status(
        manifest,
        fk_count=int(fk_arrays["sample_id"].shape[0]),
        jacobian_count=int(jacobian_arrays["sample_id"].shape[0]),
        singularity_count=int(singularity_arrays["sample_id"].shape[0]),
        fk_group_counts=fk_group_counts,
        jacobian_group_counts=jacobian_group_counts,
        singularity_group_counts=singularity_group_counts,
        full_locked_counts=full_locked_counts,
    )
    _atomic_write_json(paths.manifest_file, manifest)

    checksum_manifest = build_checksum_manifest(paths.root)
    _atomic_write_json(paths.checksum_manifest_file, checksum_manifest)

    return Tier0GenerationResult(
        dataset_root=paths.root,
        tier0_validation_dir=tier0_dir,
        dry_run=False,
        fk_total=int(fk_arrays["sample_id"].shape[0]),
        jacobian_total=int(jacobian_arrays["sample_id"].shape[0]),
        singularity_total=int(singularity_arrays["sample_id"].shape[0]),
        fk_group_counts=fk_group_counts,
        jacobian_group_counts=jacobian_group_counts,
        singularity_group_counts=singularity_group_counts,
        full_locked_counts=full_locked_counts,
        report=report,
    )
