"""Dataset v2 anchor generator: 12 anchor configurations (6 regular, 3 near_limit,
3 near_singular), split 4/4/4 (2 regular + 1 near_limit + 1 near_singular per split).

Anchors are never chosen by DLS convergence -- every classification below depends only on real
computed FK/Jacobian/joint-limit quantities (``kinematics/``, unchanged). Candidate pools reuse
Tier 0's already-verified sampling constructions
(``dataset_v2/tier0_generation.py::_group_random_interior``/``_group_mixed_near_limits``/
``_build_singularity_candidate_pool``) -- bias only *proposes* candidates; classification always
uses the real computed metric.

Selection is diversity-aware (spec section 5): each class's exact quota is drawn via a
deterministic greedy farthest-point (max-min) selection over a normalized composite feature
vector (joint-space, workspace position, orientation, sigma_min, joint-limit margin, plus a
controlling-joint one-hot for the near_limit class) -- never a plain "first K after sort" cut.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataset_v2.checksums import build_checksum_manifest, content_hash_of_record
from dataset_v2.config_templates import (
    ANCHOR_CLASS_PRIORITY_HIGHEST_FIRST,
    DATASET_SCHEMA_VERSION,
    DATASET_VERSION,
    SPLITS,
)
from dataset_v2.locator import DatasetV2Paths, relative_to_dataset_v2_root, require_dataset_v2_root
from dataset_v2.manifest import apply_anchor_generation_status
from dataset_v2.tier0_generation import (
    _build_singularity_candidate_pool,
    _generate_unique_group,
    _group_mixed_near_limits,
    _group_random_interior,
)
from dataset_v2.seeds import derive_seed
from kinematics.forward_kinematics import forward_kinematics
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import normalized_joint_limit_margin
from kinematics.manipulability import positional_manipulability
from kinematics.model_loader import ModelContext, load_model_context
from kinematics.rotation_utils import so3_log
from kinematics.singularity_metrics import condition_number, numerical_rank, singular_values
from utils.config_loader import load_json_config
from utils.dataset_locator import MODEL_PATH as V1_MODEL_PATH, REPO_ROOT
from utils.file_checksum import sha256_file
from utils.npz_utils import save_npz

GENERATOR_VERSION = "1.0.0"

ANCHOR_CLASSES = ("regular", "near_limit", "near_singular")
ANCHOR_CLASS_IDS: Dict[str, int] = {"regular": 0, "near_limit": 1, "near_singular": 2}
ANCHOR_CLASS_ID_TO_NAME: Dict[int, str] = {v: k for k, v in ANCHOR_CLASS_IDS.items()}
ANCHOR_CLASS_TOTAL_COUNTS = {"regular": 6, "near_limit": 3, "near_singular": 3}
SPLIT_IDS: Dict[str, int] = {"development": 0, "validation": 1, "frozen_test": 2}

NPZ_NAME = "anchors.npz"
MANIFEST_NAME = "anchor_manifest.csv"
REPORT_NAME = "anchor_generation_report.json"

# Anchors component tag is configs/seed_policy.json component_tags["anchors"] (30); subordinate
# tags below are anchor-generation-only.
REGULAR_POOL_TAG = 1
NEAR_LIMIT_POOL_TAG = 2
SINGULARITY_POOL_TAG = 3
SELECT_TAG = 10
SPLIT_TAG = 20


@dataclass(frozen=True)
class AnchorGenerationSettings:
    regular_pool_size: int
    near_limit_biased_pool_size: int
    singularity_biased_pool_size: int
    regular_interior_margin_fraction: float
    near_limit_margin_rad: float
    near_limit_band_rad: float
    near_limit_interior_margin_rad: float
    singularity_interior_margin_rad: float
    near_joint_limit_threshold: float
    near_singularity_threshold: float
    moderately_conditioned_upper_bound: float
    near_limit_min_sigma_min: float
    frozen_core_seed_revision: int
    controlling_joint_emphasis: float
    near_duplicate_joint_space_rad: float
    near_duplicate_position_m: float
    near_duplicate_orientation_rad: float
    split_counts_per_class: Dict[str, Dict[str, int]]


@dataclass(frozen=True)
class AnchorGenerationResult:
    dataset_root: Path
    anchors_dir: Path
    dry_run: bool
    total_anchors: int
    class_counts: Dict[str, int] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)
    class_split_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    report: Optional[dict] = None


def load_anchor_generation_settings(
    paths: DatasetV2Paths,
    regular_pool_size: Optional[int] = None,
    near_limit_biased_pool_size: Optional[int] = None,
    singularity_biased_pool_size: Optional[int] = None,
) -> AnchorGenerationSettings:
    anchor_config = load_json_config(paths.configs_dir / "anchor_config.json")
    tier0_config = load_json_config(paths.configs_dir / "tier0_config.json")
    thresholds = load_json_config(paths.configs_dir / "difficulty_thresholds.json")
    seed_policy = load_json_config(paths.configs_dir / "seed_policy.json")

    isolation = anchor_config["class_eligibility_predicates"]
    if anchor_config.get("anchor_class_isolation_status") != "locked":
        raise ValueError(
            "configs/anchor_config.json:anchor_class_isolation_status must be 'locked' "
            "(Phase 5.2 requires mutually-exclusive anchor classes)"
        )

    pool_policy = anchor_config["candidate_pool_policy"]
    tier0_policy = tier0_config["sampling_policy"]
    diversity_policy = anchor_config["diversity_selection_policy"]
    duplicate_policy = anchor_config["near_duplicate_tolerance"]

    return AnchorGenerationSettings(
        regular_pool_size=int(regular_pool_size if regular_pool_size is not None else pool_policy["regular_pool_size_default"]),
        near_limit_biased_pool_size=int(
            near_limit_biased_pool_size if near_limit_biased_pool_size is not None else pool_policy["near_limit_biased_pool_size_default"]
        ),
        singularity_biased_pool_size=int(
            singularity_biased_pool_size if singularity_biased_pool_size is not None else pool_policy["singularity_biased_pool_size_default"]
        ),
        regular_interior_margin_fraction=float(pool_policy["regular_interior_margin_fraction"]),
        near_limit_margin_rad=float(tier0_policy["near_limit_margin_rad"]),
        near_limit_band_rad=float(tier0_policy["near_limit_band_rad"]),
        near_limit_interior_margin_rad=float(tier0_policy["interior_margin_rad"]),
        singularity_interior_margin_rad=float(tier0_policy["interior_margin_rad"]),
        near_joint_limit_threshold=float(thresholds["near_joint_limit"]["threshold_normalized"]),
        near_singularity_threshold=float(thresholds["near_singularity"]["threshold_sigma_min"]),
        moderately_conditioned_upper_bound=float(thresholds["moderately_conditioned"]["upper_bound_sigma_min"]),
        near_limit_min_sigma_min=float(isolation["near_limit"]["min_sigma_min_exclusive"]),
        frozen_core_seed_revision=int(seed_policy["frozen_core_seed_revision"]),
        controlling_joint_emphasis=float(diversity_policy["controlling_joint_emphasis"]),
        near_duplicate_joint_space_rad=float(duplicate_policy["joint_space_rad"]),
        near_duplicate_position_m=float(duplicate_policy["position_m"]),
        near_duplicate_orientation_rad=float(duplicate_policy["orientation_rad"]),
        split_counts_per_class={
            name: dict(value) for name, value in anchor_config["split_assignment"]["counts_per_class_per_split"].items()
        },
    )


# ---------------------------------------------------------------------------------------------
# Candidate pool construction (bias only proposes; classification always uses real metrics)
# ---------------------------------------------------------------------------------------------


def _build_regular_pool(rng, model_context: ModelContext, pool_size: int, interior_margin_fraction: float) -> np.ndarray:
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    half_range = (upper - lower) / 2.0
    margin = interior_margin_fraction * half_range
    return _generate_unique_group(rng, model_context.nq, pool_size, _group_random_interior, (lower, upper, margin))


def _build_near_limit_biased_pool(rng, model_context: ModelContext, pool_size: int, margin_rad: float, band_rad: float, interior_margin_rad: float) -> np.ndarray:
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    return _generate_unique_group(
        rng, model_context.nq, pool_size, _group_mixed_near_limits, (lower, upper, margin_rad, band_rad, interior_margin_rad)
    )


def _build_singularity_biased_pool_dedup(rng, model_context: ModelContext, pool_size: int, interior_margin_rad: float) -> np.ndarray:
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad

    def builder(r, nq, count, lo, up, margin):
        return _build_singularity_candidate_pool(r, nq, lo, up, count, margin)

    return _generate_unique_group(rng, model_context.nq, pool_size, builder, (lower, upper, interior_margin_rad))


def _compute_candidate_metrics(model_context: ModelContext, q_batch: np.ndarray) -> Dict[str, np.ndarray]:
    n = q_batch.shape[0]
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    data = model_context.new_data()

    position = np.empty((n, 3))
    quaternion = np.empty((n, 4))
    sigma_min = np.empty(n)
    sigma_max = np.empty(n)
    cond = np.empty(n)
    rank = np.empty(n, dtype=np.int32)
    manipulability = np.empty(n)
    normalized_margin = np.empty(n)
    absolute_margin_rad = np.empty(n)
    controlling_joint = np.empty(n, dtype=np.int32)

    for i in range(n):
        q = q_batch[i]
        fk = forward_kinematics(model_context, q, data=data)
        J = geometric_jacobian_world(model_context, q, data=data)
        sv = singular_values(J)

        position[i] = fk.position
        quaternion[i] = fk.quaternion_wxyz
        sigma_min[i] = float(sv[-1])
        sigma_max[i] = float(sv[0])
        cond[i] = condition_number(J)
        rank[i] = numerical_rank(J)
        manipulability[i] = positional_manipulability(J)

        per_joint_normalized = normalized_joint_limit_margin(q, lower, upper)
        per_joint_absolute = np.minimum(q - lower, upper - q)
        controlling_joint[i] = int(np.argmin(per_joint_normalized))
        normalized_margin[i] = float(per_joint_normalized[controlling_joint[i]])
        absolute_margin_rad[i] = float(per_joint_absolute[controlling_joint[i]])

    return {
        "q": q_batch,
        "position": position,
        "quaternion": quaternion,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": cond,
        "numerical_rank": rank,
        "manipulability": manipulability,
        "normalized_margin": normalized_margin,
        "absolute_margin_rad": absolute_margin_rad,
        "controlling_joint": controlling_joint,
    }


def _classify(metrics: Dict[str, np.ndarray], near_joint_limit_threshold: float, near_singularity_threshold: float, moderately_conditioned_upper_bound: float):
    sigma_min = metrics["sigma_min"]
    margin = metrics["normalized_margin"]

    is_near_singular = sigma_min <= near_singularity_threshold
    is_near_limit = margin <= near_joint_limit_threshold
    is_moderately_conditioned = (sigma_min > near_singularity_threshold) & (sigma_min <= moderately_conditioned_upper_bound)
    is_regular = (sigma_min > moderately_conditioned_upper_bound) & (margin > near_joint_limit_threshold)

    n = sigma_min.shape[0]
    primary_class = np.full(n, "", dtype=object)
    claimed = np.zeros(n, dtype=bool)
    eligibility = {"near_singular": is_near_singular, "near_limit": is_near_limit, "regular": is_regular}
    for name in ANCHOR_CLASS_PRIORITY_HIGHEST_FIRST:
        eligible_unclaimed = eligibility[name] & (~claimed)
        primary_class[eligible_unclaimed] = name
        claimed |= eligible_unclaimed

    return {
        "is_near_singular": is_near_singular,
        "is_near_limit": is_near_limit,
        "is_moderately_conditioned": is_moderately_conditioned,
        "is_regular": is_regular,
        "primary_class": primary_class,
    }


def class_eligibility_masks(
    metrics: Dict[str, np.ndarray],
    classification: Dict[str, np.ndarray],
    near_joint_limit_threshold: float,
    near_singularity_threshold: float,
    near_limit_min_sigma_min: float,
) -> Dict[str, np.ndarray]:
    """Phase 5.2 [LOCKED] mutually-exclusive anchor-class eligibility.

    These predicates govern *anchor selection only*. The global difficulty definitions and the
    four diagnostic flags in ``classification`` are computed independently and are not altered
    here. An anchor may be near a joint limit OR near-singular, never both -- Phase 5.1 measured
    the cost of the compound case (a near-limit *and* near-singular anchor forced its closed-shape
    trajectories to scale 0.12/0.20).
    """
    sigma_min = metrics["sigma_min"]
    margin = metrics["normalized_margin"]

    regular = (sigma_min > near_limit_min_sigma_min) & (margin > near_joint_limit_threshold)
    near_limit = (
        (margin <= near_joint_limit_threshold)
        & (sigma_min > near_limit_min_sigma_min)
        & (~classification["is_near_singular"])
    )
    near_singular = (
        (sigma_min <= near_singularity_threshold)
        & (margin > near_joint_limit_threshold)
        & (~classification["is_near_limit"])
    )
    return {"regular": regular, "near_limit": near_limit, "near_singular": near_singular}


def candidate_availability_report(
    metrics: Dict[str, np.ndarray],
    classification: Dict[str, np.ndarray],
    eligibility: Dict[str, np.ndarray],
    near_joint_limit_threshold: float,
    near_singularity_threshold: float,
    near_limit_min_sigma_min: float,
) -> dict:
    """Candidate-availability breakdown reported before any anchor is selected (spec section 3)."""
    sigma_min = metrics["sigma_min"]
    margin = metrics["normalized_margin"]
    raw_near_limit = classification["is_near_limit"]
    raw_near_singular = classification["is_near_singular"]

    return {
        "total_candidates": int(sigma_min.shape[0]),
        "raw_near_limit_total": int(np.sum(raw_near_limit)),
        "near_limit_well_conditioned_count": int(np.sum(raw_near_limit & (sigma_min > near_limit_min_sigma_min))),
        "near_limit_moderately_conditioned_count": int(
            np.sum(raw_near_limit & (sigma_min > near_singularity_threshold) & (sigma_min <= near_limit_min_sigma_min))
        ),
        "near_limit_near_singular_overlap_count": int(np.sum(raw_near_limit & raw_near_singular)),
        "raw_near_singular_total": int(np.sum(raw_near_singular)),
        "near_singular_clean_count": int(np.sum(raw_near_singular & (margin > near_joint_limit_threshold))),
        "near_singular_near_limit_overlap_count": int(np.sum(raw_near_singular & raw_near_limit)),
        "eligible_regular_count": int(np.sum(eligibility["regular"])),
        "eligible_near_limit_count": int(np.sum(eligibility["near_limit"])),
        "eligible_near_singular_count": int(np.sum(eligibility["near_singular"])),
        "required": dict(ANCHOR_CLASS_TOTAL_COUNTS),
    }


# ---------------------------------------------------------------------------------------------
# Diversity-aware selection (spec section 5)
# ---------------------------------------------------------------------------------------------


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    """Min-max normalize each column of a 2D array to [0, 1]; constant columns map to 0."""
    lo = values.min(axis=0)
    hi = values.max(axis=0)
    span = np.where((hi - lo) > 1e-12, (hi - lo), 1.0)
    return (values - lo) / span


def build_feature_vectors(
    model_context: ModelContext,
    metrics: Dict[str, np.ndarray],
    idx_pool: np.ndarray,
    include_controlling_joint_emphasis: bool = False,
    controlling_joint_emphasis: float = 2.0,
) -> np.ndarray:
    """Normalized composite feature vector for greedy farthest-point selection.

    Feature groups (each divided by sqrt(its own dimensionality) so no group dominates the
    Euclidean distance purely by having more raw dimensions): joint-space (7d), workspace
    position (3d), orientation log-vector (3d), sigma_min (1d), joint-limit margin (1d), and --
    near_limit class only -- a controlling-joint one-hot (7d) scaled by
    ``controlling_joint_emphasis`` to actively spread controlling joints. See
    ``configs/anchor_config.json:diversity_selection_policy`` for the documented formula.
    """
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    range_ = upper - lower

    q = metrics["q"][idx_pool]
    q_feat = ((q - lower) / range_) / np.sqrt(7)

    position = metrics["position"][idx_pool]
    pos_feat = _normalize_columns(position) / np.sqrt(3)

    quaternion = metrics["quaternion"][idx_pool]
    rot_logs = np.array([so3_log(_quat_to_matrix(quaternion[i])) for i in range(quaternion.shape[0])])
    rot_feat = (rot_logs / np.pi) / np.sqrt(3)

    sigma_min = metrics["sigma_min"][idx_pool].reshape(-1, 1)
    sigma_feat = _normalize_columns(sigma_min)

    margin = metrics["normalized_margin"][idx_pool].reshape(-1, 1)
    margin_feat = _normalize_columns(margin)

    feature = np.concatenate([q_feat, pos_feat, rot_feat, sigma_feat, margin_feat], axis=1)

    if include_controlling_joint_emphasis:
        controlling = metrics["controlling_joint"][idx_pool]
        one_hot = np.zeros((idx_pool.shape[0], 7))
        one_hot[np.arange(idx_pool.shape[0]), controlling] = 1.0
        feature = np.concatenate([feature, (one_hot / np.sqrt(7)) * controlling_joint_emphasis], axis=1)

    return feature


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    from kinematics.quaternion_utils import quaternion_wxyz_to_matrix

    return quaternion_wxyz_to_matrix(q)


def greedy_farthest_point_select(rng: np.random.Generator, features: np.ndarray, k: int, label: str) -> np.ndarray:
    """Deterministically select exactly ``k`` local indices (into ``features``) maximizing
    minimum pairwise diversity (greedy farthest-point / max-min sampling).

    The first point is the candidate farthest from the pool's centroid; each subsequent point
    maximizes the minimum distance to the already-selected set. Ties (within floating precision)
    are broken by a seeded permutation rank derived from ``rng`` -- deterministic, never
    unseeded/arbitrary. Never uses solver outcome.
    """
    n = features.shape[0]
    if n < k:
        raise ValueError(f"diversity selection for '{label}' only has {n} candidate(s), need {k}")
    if n == k:
        return np.arange(n, dtype=np.int64)

    tie_rank = np.argsort(rng.permutation(n))

    centroid = features.mean(axis=0)
    dist_to_centroid = np.linalg.norm(features - centroid, axis=1)
    order0 = np.lexsort((tie_rank, -dist_to_centroid))
    first = int(order0[0])

    selected = [first]
    available = np.ones(n, dtype=bool)
    available[first] = False
    min_dist = np.linalg.norm(features - features[first], axis=1)

    for _ in range(k - 1):
        masked = np.where(available, min_dist, -np.inf)
        order = np.lexsort((tie_rank, -masked))
        next_idx = int(order[0])
        selected.append(next_idx)
        available[next_idx] = False
        new_dist = np.linalg.norm(features - features[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    return np.array(selected, dtype=np.int64)


# ---------------------------------------------------------------------------------------------
# Overlap-aware target pool selection + split assignment
# ---------------------------------------------------------------------------------------------


def _screen_class_for_feasibility(
    model_context: ModelContext,
    metrics: Dict[str, np.ndarray],
    ordered_candidates: np.ndarray,
    class_name: str,
    target_count: int,
    feasibility_settings,
    core_settings,
    probe_reach_settings,
    master_seed: int,
    cache: dict,
    cache_context: dict,
    stats,
) -> Tuple[np.ndarray, Dict[int, dict]]:
    """Stage A: keep only candidates that pass all ten locked combinations at >= the gate.

    Candidates are screened in the deterministic diversity order supplied by the caller, so the
    surviving subset is itself diverse and the replacement order is fixed -- there is no random
    retry loop. Screening stops once the class's budget is spent or enough feasible candidates
    exist for Stage B to choose from. A rejected candidate is always replaced from the SAME class
    pool; candidates never move between classes.
    """
    from dataset_v2.anchor_feasibility import feasibility_cache_key, probe_anchor_feasibility

    budget = feasibility_settings.screening_budget_per_class.get(class_name, 4 * target_count)
    feasible_idx: List[int] = []
    evidence: Dict[int, dict] = {}
    screened = 0

    for global_idx in ordered_candidates:
        if screened >= budget or len(feasible_idx) >= max(target_count * 2, target_count + 4):
            break
        screened += 1
        q = metrics["q"][int(global_idx)]
        key = feasibility_cache_key(
            q,
            cache_context["model_fingerprint"],
            cache_context["geometry_config_fingerprint"],
            cache_context["reachability_config_fingerprint"],
            feasibility_settings,
        )
        verdict = probe_anchor_feasibility(
            model_context,
            q,
            feasibility_settings,
            core_settings,
            probe_reach_settings,
            master_seed,
            cache=cache,
            cache_key=key,
            stats=stats,
        )
        if verdict["feasible"]:
            feasible_idx.append(int(global_idx))
            evidence[int(global_idx)] = verdict
        else:
            failing = verdict.get("first_failing_combination")
            if failing:
                stats.failure_histogram[failing] = stats.failure_histogram.get(failing, 0) + 1

    stats.screened[class_name] = screened
    stats.passed[class_name] = len(feasible_idx)
    stats.rejected[class_name] = screened - len(feasible_idx)
    return np.array(feasible_idx, dtype=np.int64), evidence


def _select_class_candidates(
    model_context: ModelContext,
    metrics: Dict[str, np.ndarray],
    classification: Dict[str, np.ndarray],
    eligibility: Dict[str, np.ndarray],
    class_name: str,
    select_rng: np.random.Generator,
    controlling_joint_emphasis: float,
    availability: dict,
    feasibility_context: Optional[dict] = None,
) -> Tuple[np.ndarray, dict]:
    """Select the exact quota for one anchor class from its **isolated** eligible pool.

    Phase 5.2: there is no overlap fallback. The eligibility predicates are mutually exclusive by
    construction, so an insufficient pool is a hard failure reporting the candidate-availability
    breakdown -- thresholds are never relaxed and anchors are never duplicated.
    """
    if class_name not in eligibility:
        raise ValueError(f"unknown anchor class '{class_name}'")

    target_count = ANCHOR_CLASS_TOTAL_COUNTS[class_name]
    pool = np.flatnonzero(eligibility[class_name])

    if pool.shape[0] < target_count:
        raise ValueError(
            f"anchor class '{class_name}' has only {pool.shape[0]} eligible candidate(s) under the "
            f"locked Phase 5.2 class-isolation predicate, need {target_count}. Candidate "
            f"availability: {json.dumps(availability, sort_keys=True)}. Increase "
            "configs/anchor_config.json's candidate_pool_policy pool size(s); never relax the "
            "class-isolation thresholds and never duplicate an anchor."
        )

    features = build_feature_vectors(
        model_context,
        metrics,
        pool,
        include_controlling_joint_emphasis=(class_name == "near_limit"),
        controlling_joint_emphasis=controlling_joint_emphasis,
    )

    if feasibility_context is None or not feasibility_context["settings"].enabled:
        local_selected = greedy_farthest_point_select(select_rng, features, target_count, class_name)
        return pool[local_selected], {
            "eligible_count": int(pool.shape[0]),
            "selected_count": int(target_count),
            "selected_source": "isolated_eligible_pool",
            "overlap_fallback_used": False,
            "feasibility_screened": False,
        }

    # --- Stage A: feasibility screen, in deterministic diversity order --------------------------
    screening_order_local = greedy_farthest_point_select(select_rng, features, pool.shape[0], class_name)
    ordered_candidates = pool[screening_order_local]
    feasible_pool, evidence = _screen_class_for_feasibility(
        model_context,
        metrics,
        ordered_candidates,
        class_name,
        target_count,
        feasibility_context["settings"],
        feasibility_context["core_settings"],
        feasibility_context["probe_reach_settings"],
        feasibility_context["master_seed"],
        feasibility_context["cache"],
        feasibility_context["cache_context"],
        feasibility_context["stats"],
    )

    if feasible_pool.shape[0] < target_count:
        raise ValueError(
            f"anchor class '{class_name}' has only {feasible_pool.shape[0]} candidate(s) passing all "
            f"{feasibility_context['settings'].required_combinations} core-trajectory feasibility "
            f"combinations at scale >= {feasibility_context['settings'].minimum_scale}, need {target_count}. "
            f"Screened {feasibility_context['stats'].screened.get(class_name, 0)} candidate(s). "
            "Increase configs/anchor_config.json's feasibility_screening.screening_budget_per_class or the "
            "candidate pool size; never relax the scale gate, the class predicate or the reachability tolerances."
        )

    # --- Stage B: diversity selection over the FEASIBLE subset only ------------------------------
    feasible_features = build_feature_vectors(
        model_context,
        metrics,
        feasible_pool,
        include_controlling_joint_emphasis=(class_name == "near_limit"),
        controlling_joint_emphasis=controlling_joint_emphasis,
    )
    local_selected = greedy_farthest_point_select(select_rng, feasible_features, target_count, class_name)
    selected_global = feasible_pool[local_selected]

    selection_report = {
        "eligible_count": int(pool.shape[0]),
        "screened_count": int(feasibility_context["stats"].screened.get(class_name, 0)),
        "feasible_count": int(feasible_pool.shape[0]),
        "rejected_count": int(feasibility_context["stats"].rejected.get(class_name, 0)),
        "selected_count": int(target_count),
        "selected_source": "feasible_subset_of_isolated_eligible_pool",
        "overlap_fallback_used": False,
        "feasibility_screened": True,
    }
    feasibility_context["evidence"].update({int(i): evidence[int(i)] for i in selected_global})
    return selected_global, selection_report


def _assign_splits(rng: np.random.Generator, selected_idx: np.ndarray, split_counts: Dict[str, int]) -> Dict[str, np.ndarray]:
    n = selected_idx.shape[0]
    if sum(split_counts[name] for name in SPLITS) != n:
        raise ValueError(f"split counts {split_counts} do not sum to the class's selected count {n}")
    perm = rng.permutation(n)
    shuffled = selected_idx[perm]
    result = {}
    offset = 0
    for split_name in SPLITS:
        count = split_counts[split_name]
        result[split_name] = shuffled[offset : offset + count]
        offset += count
    return result


def _rotation_geodesic_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    from kinematics.rotation_utils import rotation_geodesic_angle

    return rotation_geodesic_angle(_quat_to_matrix(q1), _quat_to_matrix(q2))


def _check_no_cross_split_near_duplicates(
    all_q: np.ndarray, all_position: np.ndarray, all_quaternion: np.ndarray, all_split: List[str], settings: AnchorGenerationSettings
) -> int:
    """Assert no two selected anchors in *different* splits are near-duplicates (spec section 6).

    Returns the near-duplicate-pair count (across any split assignment, for reporting); raises if
    any such pair spans two different splits.
    """
    n = all_q.shape[0]
    near_duplicate_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            joint_dist = float(np.linalg.norm(all_q[i] - all_q[j]))
            pos_dist = float(np.linalg.norm(all_position[i] - all_position[j]))
            orient_dist = _rotation_geodesic_distance(all_quaternion[i], all_quaternion[j])
            is_near_duplicate = (
                joint_dist <= settings.near_duplicate_joint_space_rad
                and pos_dist <= settings.near_duplicate_position_m
                and orient_dist <= settings.near_duplicate_orientation_rad
            )
            if is_near_duplicate:
                near_duplicate_pairs += 1
                if all_split[i] != all_split[j]:
                    raise ValueError(
                        f"near-duplicate anchors detected in different splits (indices {i}, {j}; "
                        f"splits {all_split[i]!r}/{all_split[j]!r}; joint_dist={joint_dist:.6f}, "
                        f"pos_dist={pos_dist:.6f}, orient_dist={orient_dist:.6f}); refusing to "
                        "silently relax the near-duplicate tolerance or place them in different splits"
                    )
    return near_duplicate_pairs


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
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp.csv")
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    tmp_path.replace(path)
    return path


def _git_commit(repo_root: Path) -> Optional[str]:
    import subprocess

    try:
        result = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _config_fingerprint(paths: DatasetV2Paths) -> str:
    import hashlib

    anchor_config = load_json_config(paths.configs_dir / "anchor_config.json")
    thresholds = load_json_config(paths.configs_dir / "difficulty_thresholds.json")
    payload = {"anchor_config": anchor_config, "difficulty_thresholds": thresholds}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_anchor_generation(
    dataset_root,
    master_seed: Optional[int] = None,
    overwrite: bool = False,
    regular_pool_size: Optional[int] = None,
    near_limit_biased_pool_size: Optional[int] = None,
    singularity_biased_pool_size: Optional[int] = None,
    model_context: Optional[ModelContext] = None,
    dry_run: bool = False,
) -> AnchorGenerationResult:
    """Generate Dataset v2's 12 anchor configurations (6 regular/3 near_limit/3 near_singular),
    split 4/4/4 (2 regular + 1 near_limit + 1 near_singular per split).

    Pool-size overrides exist for tests/smoke runs only -- the locked counts (12 total, 6/3/3
    class split, 4/4/4 split totals, 2/1/1 per-class-per-split) never change.
    """
    paths = require_dataset_v2_root(dataset_root)
    anchors_dir = paths.anchors_dir

    output_paths = [anchors_dir / NPZ_NAME, anchors_dir / MANIFEST_NAME, anchors_dir / REPORT_NAME]
    existing = [p for p in output_paths if p.is_file()]
    if existing and not overwrite:
        existing_relative = ", ".join(str(p.relative_to(paths.root)) for p in existing)
        raise FileExistsError(
            f"Anchor v2 output already exists ({existing_relative}); pass overwrite=True "
            "(--overwrite on the CLI) to regenerate it."
        )

    seed_policy = load_json_config(paths.configs_dir / "seed_policy.json")
    resolved_master_seed = int(master_seed if master_seed is not None else seed_policy["master_seed"])
    anchors_component_tag = int(seed_policy["component_tags"]["anchors"])
    anchors_component_seed = derive_seed(resolved_master_seed, anchors_component_tag)

    settings = load_anchor_generation_settings(paths, regular_pool_size, near_limit_biased_pool_size, singularity_biased_pool_size)

    if dry_run:
        return AnchorGenerationResult(
            dataset_root=paths.root,
            anchors_dir=anchors_dir,
            dry_run=True,
            total_anchors=12,
            class_counts=dict(ANCHOR_CLASS_TOTAL_COUNTS),
            split_counts={split_name: 4 for split_name in SPLITS},
            class_split_counts=dict(settings.split_counts_per_class),
            report=None,
        )

    model_context = model_context if model_context is not None else load_model_context()

    regular_seed = derive_seed(anchors_component_seed, REGULAR_POOL_TAG)
    near_limit_seed = derive_seed(anchors_component_seed, NEAR_LIMIT_POOL_TAG)
    singularity_seed = derive_seed(anchors_component_seed, SINGULARITY_POOL_TAG)

    regular_pool = _build_regular_pool(np.random.default_rng(regular_seed), model_context, settings.regular_pool_size, settings.regular_interior_margin_fraction)
    near_limit_pool = _build_near_limit_biased_pool(
        np.random.default_rng(near_limit_seed), model_context, settings.near_limit_biased_pool_size,
        settings.near_limit_margin_rad, settings.near_limit_band_rad, settings.near_limit_interior_margin_rad,
    )
    singularity_pool = _build_singularity_biased_pool_dedup(
        np.random.default_rng(singularity_seed), model_context, settings.singularity_biased_pool_size, settings.singularity_interior_margin_rad
    )

    pool_labels = (
        ["regular_pool"] * regular_pool.shape[0]
        + ["near_limit_biased_pool"] * near_limit_pool.shape[0]
        + ["singularity_biased_pool"] * singularity_pool.shape[0]
    )
    pool_source_seeds = (
        [regular_seed] * regular_pool.shape[0] + [near_limit_seed] * near_limit_pool.shape[0] + [singularity_seed] * singularity_pool.shape[0]
    )
    q_all = np.concatenate([regular_pool, near_limit_pool, singularity_pool], axis=0)

    metrics = _compute_candidate_metrics(model_context, q_all)
    classification = _classify(
        metrics, settings.near_joint_limit_threshold, settings.near_singularity_threshold, settings.moderately_conditioned_upper_bound
    )
    eligibility = class_eligibility_masks(
        metrics,
        classification,
        settings.near_joint_limit_threshold,
        settings.near_singularity_threshold,
        settings.near_limit_min_sigma_min,
    )
    availability = candidate_availability_report(
        metrics,
        classification,
        eligibility,
        settings.near_joint_limit_threshold,
        settings.near_singularity_threshold,
        settings.near_limit_min_sigma_min,
    )

    # --- feasibility screening context (Phase 5.4) ------------------------------------------------
    from dataset_v2.anchor_feasibility import (
        FeasibilityStats,
        config_fingerprints,
        load_feasibility_settings,
        process_feasibility_cache,
    )
    from dataset_v2.core_trajectory_generation import load_core_trajectory_generation_settings
    from dataset_v2.generation_reachability import load_generation_reachability_settings, probe_settings_from

    feasibility_settings = load_feasibility_settings(paths)
    feasibility_stats = FeasibilityStats()
    feasibility_context = None
    if feasibility_settings.enabled:
        geometry_fp, reachability_fp = config_fingerprints(paths)
        feasibility_context = {
            "settings": feasibility_settings,
            "core_settings": load_core_trajectory_generation_settings(paths),
            "probe_reach_settings": probe_settings_from(paths, load_generation_reachability_settings(paths)),
            "master_seed": resolved_master_seed,
            "cache": process_feasibility_cache(),
            "cache_context": {
                "model_fingerprint": sha256_file(V1_MODEL_PATH),
                "geometry_config_fingerprint": geometry_fp,
                "reachability_config_fingerprint": reachability_fp,
            },
            "evidence": {},
            "stats": feasibility_stats,
        }

    selected_global_by_class: Dict[str, np.ndarray] = {}
    selection_report_by_class: Dict[str, dict] = {}
    select_seed_by_class: Dict[str, int] = {}

    screening_started = time.perf_counter()
    for class_name in ANCHOR_CLASSES:
        class_id = ANCHOR_CLASS_IDS[class_name]
        select_seed = derive_seed(anchors_component_seed, SELECT_TAG, class_id)
        select_seed_by_class[class_name] = select_seed
        select_rng = np.random.default_rng(select_seed)
        selected_global, selection_report = _select_class_candidates(
            model_context,
            metrics,
            classification,
            eligibility,
            class_name,
            select_rng,
            settings.controlling_joint_emphasis,
            availability,
            feasibility_context,
        )
        selected_global_by_class[class_name] = selected_global
        selection_report_by_class[class_name] = selection_report
    feasibility_stats.runtime_seconds = time.perf_counter() - screening_started

    split_assignment_by_class: Dict[str, Dict[str, np.ndarray]] = {}
    split_seed_by_class: Dict[str, int] = {}
    for class_name in ANCHOR_CLASSES:
        class_id = ANCHOR_CLASS_IDS[class_name]
        # The frozen-test seed revision participates in the split permutation: with exact 2/1/1
        # quotas, frozen membership is the complement of development+validation, so a fresh frozen
        # namespace necessarily re-permutes the whole class assignment (documented in
        # configs/seed_policy.json:frozen_core_seed_policy). Phase 5.2 regenerates every anchor
        # anyway, so no split retains prior content.
        split_seed = derive_seed(anchors_component_seed, SPLIT_TAG, class_id, settings.frozen_core_seed_revision)
        split_seed_by_class[class_name] = split_seed
        split_assignment_by_class[class_name] = _assign_splits(
            np.random.default_rng(split_seed), selected_global_by_class[class_name], settings.split_counts_per_class[class_name]
        )

    # Assemble the final 12-anchor catalog, ordered class-then-split-then-local-index.
    rows: List[dict] = []
    for class_name in ANCHOR_CLASSES:
        for split_name in SPLITS:
            idx_in_split = split_assignment_by_class[class_name][split_name]
            for local_i, global_idx in enumerate(idx_in_split):
                rows.append({"class_name": class_name, "split_name": split_name, "global_idx": int(global_idx)})

    # anchor_id index is 0-based within the class (spec section D), not within the split.
    per_class_running_index: Dict[str, int] = {name: 0 for name in ANCHOR_CLASSES}
    model_fingerprint = sha256_file(V1_MODEL_PATH)
    config_fingerprint = _config_fingerprint(paths)

    anchor_ids: List[str] = []
    splits_list: List[str] = []
    class_ids: List[int] = []
    q_out = np.empty((len(rows), model_context.nq))
    position_out = np.empty((len(rows), 3))
    quaternion_out = np.empty((len(rows), 4))
    sigma_min_out = np.empty(len(rows))
    sigma_max_out = np.empty(len(rows))
    condition_number_out = np.empty(len(rows))
    numerical_rank_out = np.empty(len(rows), dtype=np.int32)
    manipulability_out = np.empty(len(rows))
    normalized_margin_out = np.empty(len(rows))
    absolute_margin_out = np.empty(len(rows))
    controlling_joint_out = np.empty(len(rows), dtype=np.int32)
    is_near_limit_out = np.empty(len(rows), dtype=bool)
    is_near_singular_out = np.empty(len(rows), dtype=bool)
    is_moderately_conditioned_out = np.empty(len(rows), dtype=bool)
    is_regular_out = np.empty(len(rows), dtype=bool)
    feasibility_passed_out = np.empty(len(rows), dtype=bool)
    feasibility_combinations_out = np.empty(len(rows), dtype=np.int32)
    feasibility_worst_scale_out = np.empty(len(rows))
    source_pool_out: List[str] = []
    source_seed_out = np.empty(len(rows), dtype=np.int64)
    content_hash_out: List[str] = []

    for row_i, row in enumerate(rows):
        class_name = row["class_name"]
        split_name = row["split_name"]
        global_idx = row["global_idx"]

        local_index = per_class_running_index[class_name]
        per_class_running_index[class_name] += 1
        anchor_id = f"anchor_{class_name}_{local_index:02d}"

        anchor_ids.append(anchor_id)
        splits_list.append(split_name)
        class_ids.append(ANCHOR_CLASS_IDS[class_name])
        q_out[row_i] = metrics["q"][global_idx]
        position_out[row_i] = metrics["position"][global_idx]
        quaternion_out[row_i] = metrics["quaternion"][global_idx]
        sigma_min_out[row_i] = metrics["sigma_min"][global_idx]
        sigma_max_out[row_i] = metrics["sigma_max"][global_idx]
        condition_number_out[row_i] = metrics["condition_number"][global_idx]
        numerical_rank_out[row_i] = metrics["numerical_rank"][global_idx]
        manipulability_out[row_i] = metrics["manipulability"][global_idx]
        normalized_margin_out[row_i] = metrics["normalized_margin"][global_idx]
        absolute_margin_out[row_i] = metrics["absolute_margin_rad"][global_idx]
        controlling_joint_out[row_i] = metrics["controlling_joint"][global_idx]
        is_near_limit_out[row_i] = bool(classification["is_near_limit"][global_idx])
        is_near_singular_out[row_i] = bool(classification["is_near_singular"][global_idx])
        is_moderately_conditioned_out[row_i] = bool(classification["is_moderately_conditioned"][global_idx])
        is_regular_out[row_i] = bool(classification["is_regular"][global_idx])
        source_pool_out.append(pool_labels[global_idx])
        source_seed_out[row_i] = pool_source_seeds[global_idx]

        # per-anchor proof that all ten locked combinations passed the feasibility screen
        verdict = (feasibility_context or {}).get("evidence", {}).get(global_idx)
        if verdict is None:
            feasibility_passed_out[row_i] = not feasibility_settings.enabled
            feasibility_combinations_out[row_i] = 0
            feasibility_worst_scale_out[row_i] = -1.0
        else:
            feasibility_passed_out[row_i] = bool(verdict["feasible"])
            feasibility_combinations_out[row_i] = int(verdict["combinations_passed"])
            feasibility_worst_scale_out[row_i] = float(verdict["worst_accepted_scale"])
        content_hash_out.append(
            content_hash_of_record(
                {
                    "q": [round(float(v), 12) for v in metrics["q"][global_idx]],
                    "anchor_class": class_name,
                    "split": split_name,
                    "model_fingerprint": model_fingerprint,
                    "config_fingerprint": config_fingerprint,
                }
            )
        )

    near_duplicate_pairs = _check_no_cross_split_near_duplicates(q_out, position_out, quaternion_out, splits_list, settings)

    if len(set(anchor_ids)) != len(anchor_ids):
        raise ValueError("duplicate anchor_id found across the generated anchor catalog")
    if len(set(content_hash_out)) != len(content_hash_out):
        raise ValueError("duplicate content_hash found across the generated anchor catalog")
    pair_keys = [q_out[i].tobytes() for i in range(q_out.shape[0])]
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("duplicate exact q found across the generated anchor catalog")

    max_len = max(len(s) for s in anchor_ids)
    arrays = {
        "anchor_id": np.array(anchor_ids, dtype=f"<U{max_len}"),
        "split": np.array(splits_list, dtype="<U11"),
        "split_id": np.array([SPLIT_IDS[s] for s in splits_list], dtype=np.int32),
        "anchor_class_id": np.array(class_ids, dtype=np.int32),
        "q": q_out,
        "position": position_out,
        "quaternion_wxyz": quaternion_out,
        "sigma_min": sigma_min_out,
        "sigma_max": sigma_max_out,
        "condition_number": condition_number_out,
        "numerical_rank": numerical_rank_out,
        "manipulability": manipulability_out,
        "minimum_normalized_limit_margin": normalized_margin_out,
        "minimum_absolute_limit_margin_rad": absolute_margin_out,
        "controlling_joint_index": controlling_joint_out,
        "is_near_limit": is_near_limit_out,
        "is_near_singular": is_near_singular_out,
        "is_moderately_conditioned": is_moderately_conditioned_out,
        "is_regular": is_regular_out,
        "source_pool": np.array(source_pool_out, dtype="<U24"),
        "source_seed": source_seed_out,
        "content_hash": np.array(content_hash_out, dtype="<U64"),
        "feasibility_passed": feasibility_passed_out,
        "feasibility_combinations_passed": feasibility_combinations_out,
        "feasibility_worst_accepted_scale": feasibility_worst_scale_out,
    }

    npz_path = save_npz(anchors_dir / NPZ_NAME, arrays, overwrite=overwrite)

    manifest_header = [
        "anchor_id", "split", "anchor_class", "sigma_min", "condition_number",
        "minimum_normalized_limit_margin", "controlling_joint_index", "is_near_limit",
        "is_near_singular", "source_pool", "content_hash",
    ]
    manifest_rows = []
    for i in range(len(anchor_ids)):
        manifest_rows.append(
            [
                anchor_ids[i], splits_list[i], ANCHOR_CLASS_ID_TO_NAME[class_ids[i]],
                f"{sigma_min_out[i]:.8f}", f"{condition_number_out[i]:.6f}",
                f"{normalized_margin_out[i]:.8f}", int(controlling_joint_out[i]),
                bool(is_near_limit_out[i]), bool(is_near_singular_out[i]),
                source_pool_out[i], content_hash_out[i],
            ]
        )
    manifest_path = _atomic_write_csv(anchors_dir / MANIFEST_NAME, manifest_header, manifest_rows)

    class_counts = {name: int(np.sum(np.array(class_ids) == ANCHOR_CLASS_IDS[name])) for name in ANCHOR_CLASSES}
    split_counts = {split_name: int(np.sum(np.array(splits_list) == split_name)) for split_name in SPLITS}
    class_split_counts = {
        name: {split_name: int(np.sum((np.array(class_ids) == ANCHOR_CLASS_IDS[name]) & (np.array(splits_list) == split_name))) for split_name in SPLITS}
        for name in ANCHOR_CLASSES
    }

    controlling_joint_histogram = {
        name: {int(j): int(np.sum(controlling_joint_out[np.array(class_ids) == ANCHOR_CLASS_IDS[name]] == j)) for j in range(7)}
        for name in ANCHOR_CLASSES
    }
    workspace_bbox = {"min": position_out.min(axis=0).tolist(), "max": position_out.max(axis=0).tolist()}

    def _pairwise_min_distance(arr):
        if arr.shape[0] < 2:
            return None
        dists = []
        for i in range(arr.shape[0]):
            for j in range(i + 1, arr.shape[0]):
                dists.append(float(np.linalg.norm(arr[i] - arr[j])))
        return {"min": min(dists), "max": max(dists), "mean": float(np.mean(dists))}

    diversity_summary = {
        name: {
            "joint_space": _pairwise_min_distance(q_out[np.array(class_ids) == ANCHOR_CLASS_IDS[name]]),
            "workspace_position": _pairwise_min_distance(position_out[np.array(class_ids) == ANCHOR_CLASS_IDS[name]]),
        }
        for name in ANCHOR_CLASSES
    }

    generated_at = datetime.now(timezone.utc).isoformat()
    git_commit = _git_commit(REPO_ROOT)

    report = {
        "dataset_version": DATASET_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "git_commit": git_commit,
        "model_fingerprint": model_fingerprint,
        "config_fingerprint": config_fingerprint,
        "master_seed": resolved_master_seed,
        "generation_timestamp_utc": generated_at,
        "generation_status": "development",
        "seed_derivation": {
            "anchors_component_tag": anchors_component_tag,
            "anchors_component_seed": anchors_component_seed,
            "regular_pool_seed": regular_seed,
            "near_limit_biased_pool_seed": near_limit_seed,
            "singularity_biased_pool_seed": singularity_seed,
            "select_seed_by_class": select_seed_by_class,
            "split_seed_by_class": split_seed_by_class,
        },
        "thresholds": {
            "near_joint_limit_threshold": settings.near_joint_limit_threshold,
            "near_singularity_threshold": settings.near_singularity_threshold,
            "moderately_conditioned_upper_bound": settings.moderately_conditioned_upper_bound,
            "near_limit_min_sigma_min": settings.near_limit_min_sigma_min,
        },
        "anchor_class_isolation_status": "locked",
        "frozen_core_seed_revision": settings.frozen_core_seed_revision,
        "candidate_availability": availability,
        "candidate_pool_sizes": {
            "regular_pool": int(regular_pool.shape[0]),
            "near_limit_biased_pool": int(near_limit_pool.shape[0]),
            "singularity_biased_pool": int(singularity_pool.shape[0]),
        },
        "selection_report_by_class": selection_report_by_class,
        "feasibility_screening": {
            "enabled": feasibility_settings.enabled,
            "required_combinations": feasibility_settings.required_combinations,
            "minimum_scale": feasibility_settings.minimum_scale,
            "coarse_canonical_waypoints": feasibility_settings.coarse_canonical_waypoints,
            "coarse_source_waypoints": feasibility_settings.coarse_source_waypoints,
            "full_verification_canonical_waypoints": 400,
            "candidates_screened_per_class": dict(feasibility_stats.screened),
            "candidates_passed_per_class": dict(feasibility_stats.passed),
            "candidates_rejected_per_class": dict(feasibility_stats.rejected),
            "failure_histogram_by_combination": dict(sorted(feasibility_stats.failure_histogram.items())),
            "cache_hits": feasibility_stats.cache_hits,
            "cache_misses": feasibility_stats.cache_misses,
            "worst_accepted_scale_semantics": (
                "the probe tests the gate rung (smallest scheduled scale >= minimum_scale), so this "
                "is the scale at which feasibility was VERIFIED, not the best achievable scale -- "
                "the generator maximizes the accepted scale afterwards and can only do better"
            ),
            "screening_runtime_seconds": round(feasibility_stats.runtime_seconds, 3),
            "geometry_config_fingerprint": (feasibility_context or {}).get("cache_context", {}).get("geometry_config_fingerprint"),
            "reachability_config_fingerprint": (feasibility_context or {}).get("cache_context", {}).get("reachability_config_fingerprint"),
            "selected_anchor_feasibility": {
                anchor_ids[i]: {
                    "passed": bool(feasibility_passed_out[i]),
                    "combinations_passed": int(feasibility_combinations_out[i]),
                    "worst_accepted_scale": float(feasibility_worst_scale_out[i]),
                }
                for i in range(len(anchor_ids))
            },
            "selected_anchor_worst_combination": {
                anchor_ids[i]: min(
                    (
                        (label, v["best_scale"])
                        for label, v in ((feasibility_context or {}).get("evidence", {}).get(rows[i]["global_idx"], {}) or {})
                        .get("matrix", {})
                        .items()
                        if v.get("best_scale") is not None
                    ),
                    key=lambda kv: kv[1],
                    default=None,
                )
                for i in range(len(anchor_ids))
            },
        },
        "total_anchors": len(anchor_ids),
        "class_counts": class_counts,
        "split_counts": split_counts,
        "class_split_counts": class_split_counts,
        "near_duplicate_pairs": near_duplicate_pairs,
        "controlling_joint_histogram": controlling_joint_histogram,
        "workspace_bounding_box": workspace_bbox,
        "diversity_summary": diversity_summary,
        "selected_sigma_min_distribution": {
            name: {
                "min": float(sigma_min_out[np.array(class_ids) == ANCHOR_CLASS_IDS[name]].min()),
                "max": float(sigma_min_out[np.array(class_ids) == ANCHOR_CLASS_IDS[name]].max()),
            }
            for name in ANCHOR_CLASSES
        },
        "selected_limit_margin_distribution": {
            name: {
                "min": float(normalized_margin_out[np.array(class_ids) == ANCHOR_CLASS_IDS[name]].min()),
                "max": float(normalized_margin_out[np.array(class_ids) == ANCHOR_CLASS_IDS[name]].max()),
            }
            for name in ANCHOR_CLASSES
        },
        "output_files": {
            "anchors_npz": {"filename": relative_to_dataset_v2_root(npz_path, paths.root), "sha256": sha256_file(npz_path)},
            "manifest": {"filename": relative_to_dataset_v2_root(manifest_path, paths.root), "sha256": sha256_file(manifest_path)},
        },
    }
    _atomic_write_json(anchors_dir / REPORT_NAME, report)

    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    manifest = apply_anchor_generation_status(
        manifest, total_anchors=len(anchor_ids), class_counts=class_counts, split_counts=split_counts, class_split_counts=class_split_counts
    )
    _atomic_write_json(paths.manifest_file, manifest)

    checksum_manifest = build_checksum_manifest(paths.root)
    _atomic_write_json(paths.checksum_manifest_file, checksum_manifest)

    return AnchorGenerationResult(
        dataset_root=paths.root,
        anchors_dir=anchors_dir,
        dry_run=False,
        total_anchors=len(anchor_ids),
        class_counts=class_counts,
        split_counts=split_counts,
        class_split_counts=class_split_counts,
        report=report,
    )
