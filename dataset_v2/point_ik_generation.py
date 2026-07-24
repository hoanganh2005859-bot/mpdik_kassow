"""Dataset v2 Tier 1 Point-IK generator: 6,000 samples, 6 difficulty groups x 1,000, split
development/validation/frozen_test 1,200/1,200/3,600 (200/200/600 per group).

Every target is produced by drawing a valid ``q_target_reference`` and computing its pose via the
existing forward-kinematics implementation (``kinematics/forward_kinematics.py``, unchanged) --
never a freely chosen Cartesian point assumed reachable. ``q_target_reference`` is stored only as
a reference/provenance value; nothing in this module (or any consumer of its output within this
phase) ever uses it as an IK solver's initial guess.

Difficulty-group thresholds:
- ``near_joint_limit``/``near_singularity`` reuse the single-configuration thresholds locked by
  Phase 2.5 (``configs/difficulty_thresholds.json``) applied to the pair-minimum of the initial/
  target covariate -- unchanged from v1's ``generate_point_ik_dataset.py`` pattern.
- ``near_target``/``medium_target``/``far_target``/``large_orientation_change`` are derived fresh
  at generation time from this phase's own candidate pool (never copied from v1's stored
  quantiles), using the same quantile levels (33rd/66th percentile position, 85th percentile
  orientation) v1 already validated.

Selection is diversity-aware (spec section 6): each group's exact quota is drawn via deterministic
stratified sampling over quantile-binned covariates (joint-space location, target workspace
location, orientation distance, position distance, sigma_min, joint-limit margin), never a plain
"first N eligible" cut and never based on solver outcome.
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from dataset_v2.checksums import build_checksum_manifest, content_hash_of_record
from dataset_v2.config_templates import (
    CLASSIFICATION_PRIORITY_HIGHEST_FIRST,
    DATASET_SCHEMA_VERSION,
    DATASET_VERSION,
    DIFFICULTY_GROUPS,
    SPLITS,
)
from dataset_v2.locator import DatasetV2Paths, relative_to_dataset_v2_root, require_dataset_v2_root
from dataset_v2.manifest import apply_point_ik_generation_status
from dataset_v2.seeds import derive_seed
from kinematics.forward_kinematics import forward_kinematics
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import joint_center, minimum_joint_limit_margin
from kinematics.model_loader import ModelContext, load_model_context
from kinematics.rotation_utils import rotation_geodesic_angle
from kinematics.singularity_metrics import condition_number, singular_values
from utils.config_loader import load_json_config
from utils.dataset_locator import MODEL_PATH as V1_MODEL_PATH, REPO_ROOT
from utils.file_checksum import sha256_file
from utils.npz_utils import save_npz

GENERATOR_VERSION = "1.0.0"

DIFFICULTY_GROUP_IDS: Dict[str, int] = {name: i for i, name in enumerate(DIFFICULTY_GROUPS)}
DIFFICULTY_GROUP_ID_TO_NAME: Dict[int, str] = {i: name for name, i in DIFFICULTY_GROUP_IDS.items()}
SPLIT_IDS: Dict[str, int] = {"development": 0, "validation": 1, "frozen_test": 2}

MIN_JOINT_DISTANCE_RAD = 1e-6  # reused from v1: avoids classifying an identical pose as near_target

# Point-IK is component tag 20 (configs/seed_policy.json component_tags["point_ik"]); subordinate
# tags below are Point-IK-generation-only and distinct from Tier 0's fk/jacobian/singularity tags.
POOL_TAG = 1
SELECT_TAG = 2
SPLIT_TAG = 3

NPZ_NAMES = {"development": "development.npz", "validation": "validation.npz", "frozen_test": "frozen_test.npz"}
MANIFEST_NAME = "point_ik_manifest.csv"
DIFFICULTY_DEFINITION_NAME = "difficulty_definition.json"
REPORT_NAME = "point_ik_generation_report.json"

DIVERSITY_COVARIATES = (
    "initial_joint_space_radius_rad",
    "target_workspace_radius_m",
    "orientation_distance_rad",
    "position_distance_m",
    "pair_sigma_min",
    "pair_limit_margin_normalized",
)


@dataclass(frozen=True)
class PointIKGenerationSettings:
    samples_per_group: int
    split_sizes_per_group: Dict[str, int]
    pool_size: int
    interior_margin_fraction: float
    magnitude_log_min: float
    magnitude_log_max: float
    position_low_quantile: float
    position_high_quantile: float
    orientation_top_quantile: float
    diversity_bins_per_covariate: int
    near_joint_limit_threshold: float
    near_singularity_threshold: float


@dataclass(frozen=True)
class PointIKGenerationResult:
    dataset_root: Path
    tier1_point_ik_dir: Path
    dry_run: bool
    total_samples: int
    group_counts: Dict[str, int] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)
    group_split_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    full_locked_counts: bool = False
    report: Optional[dict] = None


def load_point_ik_generation_settings(
    paths: DatasetV2Paths,
    samples_per_group: Optional[int] = None,
    pool_size: Optional[int] = None,
    split_sizes_per_group: Optional[Dict[str, int]] = None,
) -> PointIKGenerationSettings:
    config = load_json_config(paths.configs_dir / "point_ik_config.json")
    thresholds = load_json_config(paths.configs_dir / "difficulty_thresholds.json")
    pool_policy = config["pair_pool_policy"]
    diversity_policy = config["diversity_selection_policy"]

    resolved_split_sizes = (
        dict(split_sizes_per_group) if split_sizes_per_group is not None else dict(config["split_sizes_per_group"])
    )
    return PointIKGenerationSettings(
        samples_per_group=int(samples_per_group if samples_per_group is not None else config["samples_per_group"]),
        split_sizes_per_group=resolved_split_sizes,
        pool_size=int(pool_size if pool_size is not None else pool_policy["pool_size_default"]),
        interior_margin_fraction=float(pool_policy["interior_margin_fraction"]),
        magnitude_log_min=float(pool_policy["magnitude_log_min"]),
        magnitude_log_max=float(pool_policy["magnitude_log_max"]),
        position_low_quantile=float(pool_policy["position_low_quantile"]),
        position_high_quantile=float(pool_policy["position_high_quantile"]),
        orientation_top_quantile=float(pool_policy["orientation_top_quantile"]),
        diversity_bins_per_covariate=int(diversity_policy["bins_per_covariate"]),
        near_joint_limit_threshold=float(thresholds["near_joint_limit"]["threshold_normalized"]),
        near_singularity_threshold=float(thresholds["near_singularity"]["threshold_sigma_min"]),
    )


# ---------------------------------------------------------------------------------------------
# Candidate pool construction
# ---------------------------------------------------------------------------------------------


def _sample_generic_pool(rng, model_context: ModelContext, pool_size, interior_margin_fraction, magnitude_log_min, magnitude_log_max):
    """Deterministic (q_initial, q_target_reference) pair pool with a broad spread of separations.

    q_initial: uniform over the operational interior, margin proportional to each joint's own
    half-range (not v1's flat rad margin -- see config_templates.py::point_ik_config's
    interior_margin_note). q_target_reference: q_initial perturbed by a random unit direction
    scaled by a log-uniform magnitude, clipped back into the operational limits -- reused
    unchanged from v1's generic-pool construction so the pool spans near-identical to
    far-separated pairs.
    """
    nq = model_context.nq
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    half_range = (upper - lower) / 2.0
    margin = interior_margin_fraction * half_range

    q_initial = rng.uniform(lower + margin, upper - margin, size=(pool_size, nq))

    directions = rng.normal(size=(pool_size, nq))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    directions = directions / norms

    log_magnitude = rng.uniform(magnitude_log_min, magnitude_log_max, size=(pool_size, 1))
    magnitude = 10.0**log_magnitude

    q_target = np.clip(q_initial + directions * magnitude, lower, upper)
    return q_initial, q_target


def _redraw_duplicate_pairs(rng, model_context, q_initial, q_target, builder_args, max_attempts=5):
    """Redraw any exact-duplicate (q_initial, q_target) row in place; never silently keeps one."""
    pool_size = q_initial.shape[0]
    combined = np.concatenate([q_initial, q_target], axis=1)
    for _ in range(max_attempts):
        _, first_idx = np.unique(combined, axis=0, return_index=True)
        if first_idx.shape[0] == pool_size:
            return q_initial, q_target
        dup_mask = np.ones(pool_size, dtype=bool)
        dup_mask[first_idx] = False
        n_dup = int(dup_mask.sum())
        new_initial, new_target = _sample_generic_pool(rng, model_context, n_dup, *builder_args)
        q_initial[dup_mask] = new_initial
        q_target[dup_mask] = new_target
        combined = np.concatenate([q_initial, q_target], axis=1)
    raise ValueError("could not eliminate duplicate (q_initial, q_target) pairs in the Point-IK candidate pool")


def _compute_pair_metrics(model_context: ModelContext, q_initial_batch, q_target_batch):
    n = q_initial_batch.shape[0]
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    center = joint_center(lower, upper)
    data = model_context.new_data()

    initial_position = np.empty((n, 3))
    initial_quaternion = np.empty((n, 4))
    target_position = np.empty((n, 3))
    target_quaternion = np.empty((n, 4))
    position_distance_m = np.empty(n)
    orientation_distance_rad = np.empty(n)
    joint_distance_rad = np.empty(n)
    initial_sigma_min = np.empty(n)
    target_sigma_min = np.empty(n)
    initial_sigma_max = np.empty(n)
    target_sigma_max = np.empty(n)
    initial_condition_number = np.empty(n)
    target_condition_number = np.empty(n)
    minimum_initial_limit_margin = np.empty(n)
    minimum_target_limit_margin = np.empty(n)
    minimum_initial_limit_margin_rad = np.empty(n)
    minimum_target_limit_margin_rad = np.empty(n)
    initial_joint_space_radius_rad = np.empty(n)
    target_workspace_radius_m = np.empty(n)

    for i in range(n):
        q_i = q_initial_batch[i]
        q_t = q_target_batch[i]

        fk_i = forward_kinematics(model_context, q_i, data=data)
        fk_t = forward_kinematics(model_context, q_t, data=data)
        J_i = geometric_jacobian_world(model_context, q_i, data=data)
        J_t = geometric_jacobian_world(model_context, q_t, data=data)
        sv_i = singular_values(J_i)
        sv_t = singular_values(J_t)

        initial_position[i] = fk_i.position
        initial_quaternion[i] = fk_i.quaternion_wxyz
        target_position[i] = fk_t.position
        target_quaternion[i] = fk_t.quaternion_wxyz

        position_distance_m[i] = np.linalg.norm(fk_t.position - fk_i.position)
        orientation_distance_rad[i] = rotation_geodesic_angle(fk_i.rotation_matrix, fk_t.rotation_matrix)
        joint_distance_rad[i] = np.linalg.norm(q_t - q_i)

        initial_sigma_min[i] = float(sv_i[-1])
        target_sigma_min[i] = float(sv_t[-1])
        initial_sigma_max[i] = float(sv_i[0])
        target_sigma_max[i] = float(sv_t[0])
        initial_condition_number[i] = condition_number(J_i)
        target_condition_number[i] = condition_number(J_t)

        minimum_initial_limit_margin[i] = minimum_joint_limit_margin(q_i, lower, upper)
        minimum_target_limit_margin[i] = minimum_joint_limit_margin(q_t, lower, upper)
        minimum_initial_limit_margin_rad[i] = float(np.min(np.minimum(q_i - lower, upper - q_i)))
        minimum_target_limit_margin_rad[i] = float(np.min(np.minimum(q_t - lower, upper - q_t)))

        initial_joint_space_radius_rad[i] = float(np.linalg.norm(q_i - center))
        target_workspace_radius_m[i] = float(np.linalg.norm(fk_t.position))

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
        "initial_sigma_max": initial_sigma_max,
        "target_sigma_max": target_sigma_max,
        "initial_condition_number": initial_condition_number,
        "target_condition_number": target_condition_number,
        "minimum_initial_limit_margin": minimum_initial_limit_margin,
        "minimum_target_limit_margin": minimum_target_limit_margin,
        "minimum_initial_limit_margin_rad": minimum_initial_limit_margin_rad,
        "minimum_target_limit_margin_rad": minimum_target_limit_margin_rad,
        "initial_joint_space_radius_rad": initial_joint_space_radius_rad,
        "target_workspace_radius_m": target_workspace_radius_m,
    }


def _derive_position_orientation_thresholds(metrics, position_low_q, position_high_q, orientation_top_q):
    position = metrics["position_distance_m"]
    orientation = metrics["orientation_distance_rad"]
    return {
        "position_distance_m_low_quantile": float(np.quantile(position, position_low_q)),
        "position_distance_m_high_quantile": float(np.quantile(position, position_high_q)),
        "orientation_distance_rad_top_quantile": float(np.quantile(orientation, orientation_top_q)),
    }


def _classify_pool(metrics, position_orientation_thresholds, near_joint_limit_threshold, near_singularity_threshold):
    n = metrics["position_distance_m"].shape[0]
    position = metrics["position_distance_m"]
    orientation = metrics["orientation_distance_rad"]
    joint_distance = metrics["joint_distance_rad"]
    pair_margin = np.minimum(metrics["minimum_initial_limit_margin"], metrics["minimum_target_limit_margin"])
    pair_sigma_min = np.minimum(metrics["initial_sigma_min"], metrics["target_sigma_min"])

    eligibility = {
        "near_singularity": pair_sigma_min <= near_singularity_threshold,
        "near_joint_limit": pair_margin <= near_joint_limit_threshold,
        "large_orientation_change": orientation >= position_orientation_thresholds["orientation_distance_rad_top_quantile"],
        "far_target": position >= position_orientation_thresholds["position_distance_m_high_quantile"],
        "medium_target": (position > position_orientation_thresholds["position_distance_m_low_quantile"])
        & (position < position_orientation_thresholds["position_distance_m_high_quantile"]),
        "near_target": (position <= position_orientation_thresholds["position_distance_m_low_quantile"])
        & (joint_distance > MIN_JOINT_DISTANCE_RAD),
    }

    assigned = np.full(n, -1, dtype=np.int32)
    claimed = np.zeros(n, dtype=bool)
    for name in CLASSIFICATION_PRIORITY_HIGHEST_FIRST:
        group_id = DIFFICULTY_GROUP_IDS[name]
        eligible_unclaimed = eligibility[name] & (~claimed)
        assigned[eligible_unclaimed] = group_id
        claimed |= eligible_unclaimed

    return {name: np.flatnonzero(assigned == DIFFICULTY_GROUP_IDS[name]) for name in DIFFICULTY_GROUPS}, pair_margin, pair_sigma_min


# ---------------------------------------------------------------------------------------------
# Diversity-aware selection (spec section 6)
# ---------------------------------------------------------------------------------------------


def _quantile_bin_ids(values: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1 or values.shape[0] == 0:
        return np.zeros(values.shape[0], dtype=np.int64)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    return np.digitize(values, edges)


def stratified_diversity_select(rng, idx_pool: np.ndarray, covariate_columns: List[np.ndarray], n_bins: int, target_count: int, label: str) -> np.ndarray:
    """Deterministically select exactly ``target_count`` indices from ``idx_pool``.

    Stratifies on a composite key built from quantile-binning each covariate (computed over the
    group's own eligible pool), shuffles draw order within each stratum via ``rng``, then draws
    round-robin across occupied strata. Never uses solver outcome, never relaxes a threshold,
    never duplicates a candidate; raises an actionable error (naming the group and available
    count) rather than silently reducing the quota if the pool is insufficient.
    """
    available = idx_pool.shape[0]
    if available < target_count:
        raise ValueError(
            f"difficulty group '{label}' only has {available} eligible candidate pair(s) after "
            f"classification, need {target_count}; increase configs/point_ik_config.json's "
            "pair_pool_policy.pool_size_default rather than relaxing thresholds or duplicating samples"
        )

    composite = np.zeros(available, dtype=np.int64)
    for column in covariate_columns:
        bin_ids = _quantile_bin_ids(column[idx_pool], n_bins)
        composite = composite * n_bins + bin_ids

    order = rng.permutation(available)
    strata: Dict[int, List[int]] = {}
    for local_pos in order:
        key = int(composite[local_pos])
        strata.setdefault(key, []).append(int(idx_pool[local_pos]))

    selected: List[int] = []
    stratum_keys = sorted(strata.keys())
    while len(selected) < target_count:
        progressed = False
        for key in stratum_keys:
            bucket = strata[key]
            if bucket:
                selected.append(bucket.pop())
                progressed = True
                if len(selected) == target_count:
                    break
        if not progressed:
            break

    if len(selected) < target_count:
        raise ValueError(
            f"difficulty group '{label}' diversity selection only reached {len(selected)} of "
            f"{target_count} required samples despite {available} available candidates -- this "
            "indicates a bug in stratified_diversity_select, not a data shortage"
        )
    return np.array(sorted(selected), dtype=np.int64)


def _split_group_selection(rng, selected_idx: np.ndarray, split_sizes: Dict[str, int]) -> Dict[str, np.ndarray]:
    n = selected_idx.shape[0]
    if sum(split_sizes[name] for name in SPLITS) != n:
        raise ValueError(f"split sizes {split_sizes} do not sum to the group's selected count {n}")
    perm = rng.permutation(n)
    shuffled = selected_idx[perm]
    result = {}
    offset = 0
    for split_name in SPLITS:
        count = split_sizes[split_name]
        result[split_name] = shuffled[offset : offset + count]
        offset += count
    return result


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp.json")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path


def _atomic_write_csv(path: Path, header: List[str], rows: List[list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp.csv")
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    tmp_path.replace(path)
    return path


def _assert_no_leakage(all_sample_ids: List[str], all_content_hashes: List[str], q_initial_all: np.ndarray, q_target_all: np.ndarray) -> None:
    if len(set(all_sample_ids)) != len(all_sample_ids):
        raise ValueError("duplicate sample_id found across the generated Point-IK dataset")
    if len(set(all_content_hashes)) != len(all_content_hashes):
        raise ValueError("duplicate content_hash found across the generated Point-IK dataset")
    pair_keys = [q_initial_all[i].tobytes() + q_target_all[i].tobytes() for i in range(q_initial_all.shape[0])]
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("duplicate (q_initial, q_target_reference) pair found across the generated Point-IK dataset")


def _git_commit(repo_root: Path) -> Optional[str]:
    import subprocess

    try:
        result = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def run_point_ik_generation(
    dataset_root,
    master_seed: Optional[int] = None,
    overwrite: bool = False,
    samples_per_group: Optional[int] = None,
    pool_size: Optional[int] = None,
    split_sizes_per_group: Optional[Dict[str, int]] = None,
    model_context: Optional[ModelContext] = None,
    dry_run: bool = False,
) -> PointIKGenerationResult:
    """Generate Dataset v2 Tier 1 Point-IK samples.

    ``samples_per_group``/``pool_size``/``split_sizes_per_group`` exist for tests/smoke runs
    only -- the locked full mode is 1000 samples/group, split 200/200/600 (spec section B).
    """
    paths = require_dataset_v2_root(dataset_root)
    tier1_dir = paths.tier1_point_ik_dir

    output_paths = [tier1_dir / name for name in NPZ_NAMES.values()] + [
        tier1_dir / MANIFEST_NAME,
        tier1_dir / DIFFICULTY_DEFINITION_NAME,
        tier1_dir / REPORT_NAME,
    ]
    existing = [p for p in output_paths if p.is_file()]
    if existing and not overwrite:
        existing_relative = ", ".join(str(p.relative_to(paths.root)) for p in existing)
        raise FileExistsError(
            f"Point-IK v2 output already exists ({existing_relative}); pass overwrite=True "
            "(--overwrite on the CLI) to regenerate it."
        )

    seed_policy = load_json_config(paths.configs_dir / "seed_policy.json")
    resolved_master_seed = int(master_seed if master_seed is not None else seed_policy["master_seed"])
    point_ik_component_tag = int(seed_policy["component_tags"]["point_ik"])
    point_ik_component_seed = derive_seed(resolved_master_seed, point_ik_component_tag)

    settings = load_point_ik_generation_settings(paths, samples_per_group, pool_size, split_sizes_per_group)
    full_locked_counts = bool(
        settings.samples_per_group == 1000
        and settings.split_sizes_per_group == {"development": 200, "validation": 200, "frozen_test": 600}
    )

    if dry_run:
        group_counts = {name: settings.samples_per_group for name in DIFFICULTY_GROUPS}
        split_counts = {
            split_name: settings.split_sizes_per_group[split_name] * len(DIFFICULTY_GROUPS) for split_name in SPLITS
        }
        return PointIKGenerationResult(
            dataset_root=paths.root,
            tier1_point_ik_dir=tier1_dir,
            dry_run=True,
            total_samples=settings.samples_per_group * len(DIFFICULTY_GROUPS),
            group_counts=group_counts,
            split_counts=split_counts,
            group_split_counts={name: dict(settings.split_sizes_per_group) for name in DIFFICULTY_GROUPS},
            full_locked_counts=full_locked_counts,
            report=None,
        )

    model_context = model_context if model_context is not None else load_model_context()

    pool_seed = derive_seed(point_ik_component_seed, POOL_TAG)
    pool_rng = np.random.default_rng(pool_seed)
    builder_args = (settings.interior_margin_fraction, settings.magnitude_log_min, settings.magnitude_log_max)
    q_initial_pool, q_target_pool = _sample_generic_pool(pool_rng, model_context, settings.pool_size, *builder_args)
    q_initial_pool, q_target_pool = _redraw_duplicate_pairs(pool_rng, model_context, q_initial_pool, q_target_pool, builder_args)

    metrics = _compute_pair_metrics(model_context, q_initial_pool, q_target_pool)
    position_orientation_thresholds = _derive_position_orientation_thresholds(
        metrics, settings.position_low_quantile, settings.position_high_quantile, settings.orientation_top_quantile
    )
    pool_by_group, pair_margin, pair_sigma_min = _classify_pool(
        metrics, position_orientation_thresholds, settings.near_joint_limit_threshold, settings.near_singularity_threshold
    )

    covariate_arrays = [
        metrics["initial_joint_space_radius_rad"],
        metrics["target_workspace_radius_m"],
        metrics["orientation_distance_rad"],
        metrics["position_distance_m"],
        pair_sigma_min,
        pair_margin,
    ]

    per_group_selected: Dict[str, np.ndarray] = {}
    per_group_select_seed: Dict[str, int] = {}
    per_group_split_seed: Dict[str, int] = {}
    per_group_split_assignment: Dict[str, Dict[str, np.ndarray]] = {}

    for name in DIFFICULTY_GROUPS:
        group_id = DIFFICULTY_GROUP_IDS[name]
        select_seed = derive_seed(point_ik_component_seed, SELECT_TAG, group_id)
        select_rng = np.random.default_rng(select_seed)
        selected = stratified_diversity_select(
            select_rng, pool_by_group[name], covariate_arrays, settings.diversity_bins_per_covariate, settings.samples_per_group, name
        )
        per_group_selected[name] = selected
        per_group_select_seed[name] = select_seed

        split_seed = derive_seed(point_ik_component_seed, SPLIT_TAG, group_id)
        split_rng = np.random.default_rng(split_seed)
        per_group_split_assignment[name] = _split_group_selection(split_rng, selected, settings.split_sizes_per_group)
        per_group_split_seed[name] = split_seed

    # Assemble per-split arrays.
    split_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    split_group_counts: Dict[str, Dict[str, int]] = {name: {} for name in SPLITS}
    all_sample_ids: List[str] = []
    all_content_hashes: List[str] = []
    all_q_initial: List[np.ndarray] = []
    all_q_target: List[np.ndarray] = []

    for split_name in SPLITS:
        chunks: Dict[str, List[np.ndarray]] = {}
        sample_id_chunk: List[str] = []
        difficulty_id_chunk: List[int] = []
        content_hash_chunk: List[str] = []

        for name in DIFFICULTY_GROUPS:
            idx = per_group_split_assignment[name][split_name]
            split_group_counts[split_name][name] = int(idx.shape[0])
            group_id = DIFFICULTY_GROUP_IDS[name]

            q_i = q_initial_pool[idx]
            q_t = q_target_pool[idx]
            for local_i in range(idx.shape[0]):
                sample_id = f"pik_{split_name}_{name}_{local_i:05d}"
                sample_id_chunk.append(sample_id)
                difficulty_id_chunk.append(group_id)
                record = {
                    "q_initial": [round(float(v), 12) for v in q_i[local_i]],
                    "q_target_reference": [round(float(v), 12) for v in q_t[local_i]],
                }
                content_hash_chunk.append(content_hash_of_record(record))

            for field_name in (
                "initial_position",
                "initial_quaternion",
                "target_position",
                "target_quaternion",
                "position_distance_m",
                "orientation_distance_rad",
                "joint_distance_rad",
                "initial_sigma_min",
                "target_sigma_min",
                "initial_sigma_max",
                "target_sigma_max",
                "initial_condition_number",
                "target_condition_number",
                "minimum_initial_limit_margin",
                "minimum_target_limit_margin",
                "minimum_initial_limit_margin_rad",
                "minimum_target_limit_margin_rad",
            ):
                chunks.setdefault(field_name, []).append(metrics[field_name][idx])
            chunks.setdefault("q_initial", []).append(q_i)
            chunks.setdefault("q_target_reference", []).append(q_t)

        n_split = len(sample_id_chunk)
        source_seed_arr = np.empty(n_split, dtype=np.int64)
        offset = 0
        for name in DIFFICULTY_GROUPS:
            count = split_group_counts[split_name][name]
            source_seed_arr[offset : offset + count] = per_group_select_seed[name]
            offset += count

        arrays = {
            "sample_id": np.array(sample_id_chunk, dtype=f"<U{max(len(s) for s in sample_id_chunk)}"),
            "split_id": np.full(n_split, SPLIT_IDS[split_name], dtype=np.int32),
            "difficulty_id": np.array(difficulty_id_chunk, dtype=np.int32),
            "q_initial": np.concatenate(chunks["q_initial"], axis=0).astype(np.float64),
            "q_target_reference": np.concatenate(chunks["q_target_reference"], axis=0).astype(np.float64),
            "initial_position": np.concatenate(chunks["initial_position"], axis=0).astype(np.float64),
            "initial_quaternion_wxyz": np.concatenate(chunks["initial_quaternion"], axis=0).astype(np.float64),
            "target_position": np.concatenate(chunks["target_position"], axis=0).astype(np.float64),
            "target_quaternion_wxyz": np.concatenate(chunks["target_quaternion"], axis=0).astype(np.float64),
            "position_distance_m": np.concatenate(chunks["position_distance_m"], axis=0).astype(np.float64),
            "orientation_distance_rad": np.concatenate(chunks["orientation_distance_rad"], axis=0).astype(np.float64),
            "joint_distance_rad": np.concatenate(chunks["joint_distance_rad"], axis=0).astype(np.float64),
            "initial_sigma_min": np.concatenate(chunks["initial_sigma_min"], axis=0).astype(np.float64),
            "target_sigma_min": np.concatenate(chunks["target_sigma_min"], axis=0).astype(np.float64),
            "initial_sigma_max": np.concatenate(chunks["initial_sigma_max"], axis=0).astype(np.float64),
            "target_sigma_max": np.concatenate(chunks["target_sigma_max"], axis=0).astype(np.float64),
            "initial_condition_number": np.concatenate(chunks["initial_condition_number"], axis=0).astype(np.float64),
            "target_condition_number": np.concatenate(chunks["target_condition_number"], axis=0).astype(np.float64),
            "minimum_initial_limit_margin_normalized": np.concatenate(chunks["minimum_initial_limit_margin"], axis=0).astype(np.float64),
            "minimum_target_limit_margin_normalized": np.concatenate(chunks["minimum_target_limit_margin"], axis=0).astype(np.float64),
            "minimum_initial_limit_margin_rad": np.concatenate(chunks["minimum_initial_limit_margin_rad"], axis=0).astype(np.float64),
            "minimum_target_limit_margin_rad": np.concatenate(chunks["minimum_target_limit_margin_rad"], axis=0).astype(np.float64),
            "source_seed": source_seed_arr,
            "content_hash": np.array(content_hash_chunk, dtype="<U64"),
        }
        split_arrays[split_name] = arrays

        all_sample_ids.extend(sample_id_chunk)
        all_content_hashes.extend(content_hash_chunk)
        all_q_initial.append(arrays["q_initial"])
        all_q_target.append(arrays["q_target_reference"])

    _assert_no_leakage(
        all_sample_ids, all_content_hashes, np.concatenate(all_q_initial, axis=0), np.concatenate(all_q_target, axis=0)
    )

    output_files: Dict[str, dict] = {}
    for split_name in SPLITS:
        out_path = save_npz(tier1_dir / NPZ_NAMES[split_name], split_arrays[split_name], overwrite=overwrite)
        output_files[split_name] = {"filename": relative_to_dataset_v2_root(out_path, paths.root), "sha256": sha256_file(out_path)}

    manifest_header = [
        "sample_id",
        "split",
        "difficulty_group",
        "source_seed",
        "position_distance_m",
        "orientation_distance_deg",
        "joint_distance_rad",
        "initial_sigma_min",
        "target_sigma_min",
        "minimum_initial_limit_margin_normalized",
        "minimum_target_limit_margin_normalized",
        "content_hash",
    ]
    manifest_rows: List[list] = []
    for split_name in SPLITS:
        arrays = split_arrays[split_name]
        for i in range(arrays["sample_id"].shape[0]):
            manifest_rows.append(
                [
                    str(arrays["sample_id"][i]),
                    split_name,
                    DIFFICULTY_GROUP_ID_TO_NAME[int(arrays["difficulty_id"][i])],
                    int(arrays["source_seed"][i]),
                    f"{arrays['position_distance_m'][i]:.8f}",
                    f"{np.degrees(arrays['orientation_distance_rad'][i]):.6f}",
                    f"{arrays['joint_distance_rad'][i]:.8f}",
                    f"{arrays['initial_sigma_min'][i]:.8f}",
                    f"{arrays['target_sigma_min'][i]:.8f}",
                    f"{arrays['minimum_initial_limit_margin_normalized'][i]:.8f}",
                    f"{arrays['minimum_target_limit_margin_normalized'][i]:.8f}",
                    str(arrays["content_hash"][i]),
                ]
            )
    manifest_path = _atomic_write_csv(tier1_dir / MANIFEST_NAME, manifest_header, manifest_rows)

    total_samples = sum(arrays["sample_id"].shape[0] for arrays in split_arrays.values())
    group_counts = {name: per_group_selected[name].shape[0] for name in DIFFICULTY_GROUPS}
    split_counts = {split_name: split_arrays[split_name]["sample_id"].shape[0] for split_name in SPLITS}
    group_split_counts = {name: {split_name: split_group_counts[split_name][name] for split_name in SPLITS} for name in DIFFICULTY_GROUPS}

    generated_at = datetime.now(timezone.utc).isoformat()
    git_commit = _git_commit(REPO_ROOT)
    asset_fingerprint = sha256_file(V1_MODEL_PATH)

    difficulty_definition = {
        "generator": "dataset_v2.point_ik_generation",
        "generator_version": GENERATOR_VERSION,
        "dataset_version": DATASET_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "master_seed": resolved_master_seed,
        "point_ik_component_seed": point_ik_component_seed,
        "difficulty_groups": {i: name for i, name in DIFFICULTY_GROUP_ID_TO_NAME.items()},
        "samples_per_group": settings.samples_per_group,
        "split_sizes_per_group": settings.split_sizes_per_group,
        "sample_counts": group_counts,
        "group_split_counts": group_split_counts,
        "priority_order_highest_first": list(CLASSIFICATION_PRIORITY_HIGHEST_FIRST),
        "priority_note": (
            "A candidate pair qualifying for more than one group is assigned to the highest-"
            "priority group in priority_order_highest_first; lower-priority groups only draw "
            "from pairs not already claimed."
        ),
        "generic_pool_size": settings.pool_size,
        "generic_pool_seed": pool_seed,
        "generic_pool_construction": (
            "q_initial sampled uniformly over the operational interior (margin proportional to "
            f"each joint's own half-range, fraction={settings.interior_margin_fraction}); "
            "q_target_reference = clip(q_initial + magnitude * random_unit_direction, lower, "
            f"upper) with magnitude log-uniformly sampled in "
            f"[10^{settings.magnitude_log_min}, 10^{settings.magnitude_log_max}] rad."
        ),
        "criteria": {
            "near_target": "position_distance_m <= position_distance_m_low_quantile and joint_distance_rad > 1e-6 (not an identical pose).",
            "medium_target": "position_distance_m_low_quantile < position_distance_m < position_distance_m_high_quantile.",
            "far_target": "position_distance_m >= position_distance_m_high_quantile.",
            "large_orientation_change": "orientation_distance_rad >= orientation_distance_rad_top_quantile, independent of joint/position distance.",
            "near_joint_limit": (
                "min(minimum_initial_limit_margin_normalized, minimum_target_limit_margin_normalized) <= "
                f"{settings.near_joint_limit_threshold} (Phase 2.5 locked single-configuration threshold)."
            ),
            "near_singularity": (
                f"min(initial_sigma_min, target_sigma_min) <= {settings.near_singularity_threshold} "
                "(Phase 2.5 locked single-configuration threshold, reused from v1's configs/dls_config.json)."
            ),
        },
        "quantile_thresholds": position_orientation_thresholds,
        "near_joint_limit_threshold": settings.near_joint_limit_threshold,
        "near_singularity_threshold": settings.near_singularity_threshold,
        "diversity_selection": {
            "covariates": list(DIVERSITY_COVARIATES),
            "bins_per_covariate": settings.diversity_bins_per_covariate,
            "method": "stratified_diversity_select (see dataset_v2/point_ik_generation.py)",
        },
        "q_target_reference_usage_policy": (
            "q_target_reference is a reference/provenance value only; it must never be used as "
            "q_initial for any IK solve evaluating this sample."
        ),
        "full_locked_counts": full_locked_counts,
    }
    difficulty_path = _atomic_write_json(tier1_dir / DIFFICULTY_DEFINITION_NAME, difficulty_definition)

    report = {
        "dataset_version": DATASET_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "git_commit": git_commit,
        "asset_fingerprint": asset_fingerprint,
        "master_seed": resolved_master_seed,
        "generation_timestamp_utc": generated_at,
        "component": "point_ik",
        "seed_derivation": {
            "point_ik_component_tag": point_ik_component_tag,
            "point_ik_component_seed": point_ik_component_seed,
            "pool_tag": POOL_TAG,
            "pool_seed": pool_seed,
            "select_tag": SELECT_TAG,
            "split_tag": SPLIT_TAG,
            "per_group_select_seed": per_group_select_seed,
            "per_group_split_seed": per_group_split_seed,
        },
        "total_samples": total_samples,
        "group_counts": group_counts,
        "split_counts": split_counts,
        "group_split_counts": group_split_counts,
        "full_locked_counts": full_locked_counts,
        "quantile_thresholds": position_orientation_thresholds,
        "near_joint_limit_threshold": settings.near_joint_limit_threshold,
        "near_singularity_threshold": settings.near_singularity_threshold,
        "generic_pool_size": settings.pool_size,
        "output_files": {
            **output_files,
            "manifest": {"filename": relative_to_dataset_v2_root(manifest_path, paths.root), "sha256": sha256_file(manifest_path)},
            "difficulty_definition": {
                "filename": relative_to_dataset_v2_root(difficulty_path, paths.root),
                "sha256": sha256_file(difficulty_path),
            },
        },
    }
    _atomic_write_json(tier1_dir / REPORT_NAME, report)

    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    manifest = apply_point_ik_generation_status(
        manifest,
        total_samples=total_samples,
        group_counts=group_counts,
        split_counts=split_counts,
        group_split_counts=group_split_counts,
        full_locked_counts=full_locked_counts,
    )
    _atomic_write_json(paths.manifest_file, manifest)

    checksum_manifest = build_checksum_manifest(paths.root)
    _atomic_write_json(paths.checksum_manifest_file, checksum_manifest)

    return PointIKGenerationResult(
        dataset_root=paths.root,
        tier1_point_ik_dir=tier1_dir,
        dry_run=False,
        total_samples=total_samples,
        group_counts=group_counts,
        split_counts=split_counts,
        group_split_counts=group_split_counts,
        full_locked_counts=full_locked_counts,
        report=report,
    )
