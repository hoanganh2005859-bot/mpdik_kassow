"""Dataset v2 random-challenge trajectory generator (Phase 6): 90 trajectories = 6 challenge
families x 5 per split x 3 splits (development / validation / frozen_test), each with exactly 400
canonical waypoints -> 36,000 canonical challenge poses.

Design (spec section I; the Phase 0 `[PROVISIONAL]` generation policy is locked here). Unlike core
trajectories -- Cartesian closed-form shapes anchored at the 12 locked anchors -- challenge
trajectories are drawn from smooth, seeded JOINT-SPACE reference families and pushed through
forward kinematics ("known-reachable joint-space reference family converted through FK", task
section 6). Each trajectory starts from an *independent* reachable start state (not one of the 12
anchors) and follows a bounded-Fourier joint-space curve whose per-joint amplitude is capped at a
fraction of that joint's own start joint-limit margin, so the reference stays inside operational
limits with no clipping and every source pose is reachable by construction. The SAME strict,
DLS-baseline-independent reachability engine (``dataset_v2/generation_reachability.py``, 1e-4 m /
0.01 deg) still verifies every source and canonical waypoint independently -- the numerical IK
engine's success flag is never sufficient.

Feasibility-aware diversity selection (task section 7) mirrors Phase 5.4: per (family, split) a
seeded candidate pool is coarse-probe screened for strict reachability + family coverage floors,
the feasible subset is diversity-selected (greedy farthest-point) down to the exact quota, and the
selected candidates are then re-validated end-to-end at full 400/source resolution. A selected
candidate that fails full validation is replaced deterministically from the same pool -- never by
loosening reachability, counts, or the family policy.

This module never modifies Dataset v1, never touches FK/Jacobian/pose-error/DLS formulas, and
never uses global ``numpy.random`` state.
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dataset_v2.anchor_generation import _compute_candidate_metrics, greedy_farthest_point_select
from dataset_v2.checksums import build_checksum_manifest, content_hash_of_record
from dataset_v2.config_templates import (
    CHALLENGE_FAMILIES,
    CHALLENGE_FAMILY_TAGS,
    DATASET_SCHEMA_VERSION,
    DATASET_VERSION,
    SEED_COMPONENT_TAGS,
    SEED_SPLIT_TAGS,
    SPLITS,
)
from dataset_v2.core_trajectory_generation import (
    _atomic_write_csv,
    _atomic_write_json,
    _canonicalize_quat_sequence,
    _git_commit,
    resample_canonical,
)
from dataset_v2.generation_reachability import (
    error_distribution,
    load_generation_reachability_settings,
    probe_settings_from,
    validate_path_strict,
)
from dataset_v2.locator import DatasetV2Paths, relative_to_dataset_v2_root, require_dataset_v2_root
from dataset_v2.manifest import apply_challenge_trajectory_generation_status
from dataset_v2.seeds import derive_seed
from dataset_v2.tier0_generation import (
    _build_singularity_candidate_pool,
    _group_mixed_near_limits,
    _group_random_interior,
)
from generators._trajectory_common import build_time_and_path_parameter
from kinematics.forward_kinematics import forward_kinematics
from kinematics.model_loader import ModelContext, load_model_context
from utils.config_loader import load_json_config
from utils.dataset_locator import MODEL_PATH as V1_MODEL_PATH, REPO_ROOT
from utils.file_checksum import sha256_file
from utils.npz_utils import save_npz

GENERATOR_VERSION = "1.0.0"

# Random-challenge component tag is configs/seed_policy.json component_tags["random_challenge"]
# (50); subordinate integer tags below are challenge-generation-only.
CANDIDATE_TAG = 7
COEFF_TAG = 11

NPZ_SUFFIX = ".npz"
SOURCE_SUFFIX = "_source.npz"
MANIFEST_NAME = "challenge_trajectory_manifest.csv"
REPORT_NAME = "challenge_trajectory_generation_report.json"
REACHABILITY_REPORT_NAME = "challenge_trajectory_reachability_report.json"
DIVERSITY_REPORT_NAME = "challenge_trajectory_diversity_report.json"
FEASIBILITY_REPORT_NAME = "challenge_trajectory_feasibility_report.json"
ANTI_LEAKAGE_REPORT_NAME = "challenge_trajectory_anti_leakage_report.json"

MANIFEST_COLUMNS = [
    "trajectory_id",
    "family",
    "challenge_family",
    "split",
    "family_candidate_index",
    "source_seed",
    "path_seed",
    "frozen_challenge_seed_revision",
    "source_waypoint_count",
    "canonical_waypoint_count",
    "quaternion_convention",
    "duration_s",
    "canonical_control_period_s",
    "start_position",
    "start_sigma_min",
    "start_sigma_max",
    "start_condition_number",
    "start_normalized_limit_margin",
    "start_absolute_limit_margin_rad",
    "start_controlling_joint_index",
    "start_content_hash",
    "envelope_margin_fraction",
    "harmonics_json",
    "geometry_parameters_json",
    "arc_length_m",
    "cumulative_angular_displacement_rad",
    "mean_curvature_1_per_m",
    "max_curvature_1_per_m",
    "non_planarity",
    "reachability_status",
    "reachability_tolerance_position_m",
    "reachability_tolerance_orientation_deg",
    "canonical_position_reconstruction_max_m",
    "canonical_orientation_reconstruction_max_deg",
    "source_position_reconstruction_max_m",
    "source_orientation_reconstruction_max_deg",
    "canonical_waypoints_reachable",
    "source_waypoints_reachable",
    "generation_status",
    "model_fingerprint",
    "config_fingerprint",
    "content_hash",
    "sha256",
    "source_sha256",
]


@dataclass(frozen=True)
class ChallengeGenerationSettings:
    total: int
    split_sizes: Dict[str, int]
    per_family_per_split: int
    canonical_waypoint_count: int
    source_waypoint_count: int
    duration_s: float
    families: List[str]
    family_definitions: Dict[str, dict]
    candidate_pool_size: int
    coarse_canonical_waypoints: int
    coarse_source_waypoints: int
    near_joint_limit_threshold: float
    near_singularity_threshold: float
    start_near_duplicate_rad: float
    path_metric_relative: float
    frozen_challenge_seed_revision: int


@dataclass(frozen=True)
class ChallengeGenerationResult:
    dataset_root: Path
    trajectories_dir: Path
    dry_run: bool
    total_trajectories: int
    full_locked_counts: bool
    split_counts: Dict[str, int] = field(default_factory=dict)
    family_counts: Dict[str, int] = field(default_factory=dict)
    family_split_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    reachability_statistics: dict = field(default_factory=dict)
    diversity_statistics: dict = field(default_factory=dict)
    report: Optional[dict] = None


def load_challenge_generation_settings(
    paths: DatasetV2Paths,
    source_waypoint_count: Optional[int] = None,
    candidate_pool_size: Optional[int] = None,
    per_family_per_split: Optional[int] = None,
) -> ChallengeGenerationSettings:
    challenge_config = load_json_config(paths.configs_dir / "random_challenge_config.json")
    difficulty = load_json_config(paths.configs_dir / "difficulty_thresholds.json")
    seed_policy = load_json_config(paths.configs_dir / "seed_policy.json")
    feasibility = challenge_config["feasibility_aware_selection"]
    return ChallengeGenerationSettings(
        total=int(challenge_config["total"]),
        split_sizes={k: int(v) for k, v in challenge_config["split_sizes"].items()},
        per_family_per_split=int(
            per_family_per_split if per_family_per_split is not None else challenge_config["per_family_per_split"]
        ),
        canonical_waypoint_count=int(challenge_config["canonical_waypoints_per_trajectory"]),
        source_waypoint_count=int(
            source_waypoint_count if source_waypoint_count is not None else challenge_config["source_waypoint_count_nominal"]
        ),
        duration_s=float(challenge_config["duration_s"]),
        families=list(challenge_config["families"]),
        family_definitions=dict(challenge_config["family_definitions"]),
        candidate_pool_size=int(
            candidate_pool_size
            if candidate_pool_size is not None
            else feasibility["candidate_pool_size_per_family_per_split"]
        ),
        coarse_canonical_waypoints=int(feasibility["coarse_probe_canonical_waypoints"]),
        coarse_source_waypoints=int(feasibility["coarse_probe_source_waypoints"]),
        near_joint_limit_threshold=float(difficulty["near_joint_limit"]["threshold_normalized"]),
        near_singularity_threshold=float(difficulty["near_singularity"]["threshold_sigma_min"]),
        start_near_duplicate_rad=float(challenge_config["near_duplicate_tolerance"]["start_joint_space_rad"]),
        path_metric_relative=float(challenge_config["near_duplicate_tolerance"]["path_metric_relative"]),
        frozen_challenge_seed_revision=int(seed_policy["frozen_challenge_seed_revision"]),
    )


# ---------------------------------------------------------------------------------------------
# Start-state region draw (independent reachable starts, never the 12 anchors)
# ---------------------------------------------------------------------------------------------


def _draw_region_starts(
    model_context: ModelContext,
    rng: np.random.Generator,
    region: str,
    needed: int,
    settings: ChallengeGenerationSettings,
) -> Tuple[np.ndarray, dict]:
    """Draw ``needed`` distinct start configurations from a family's joint-space region.

    ``interior`` is a uniform interior draw with a per-joint range-proportional margin (the same
    joint_2/joint_4-bias-free construction Point-IK/anchor generation use). ``near_limit`` and
    ``near_singular`` over-draw a biased pool and keep only the candidates whose real computed
    metric satisfies the Phase 2.5 threshold. Bias only proposes candidates; the predicate always
    uses the real computed metric. Raises with the availability breakdown if too few survive.
    """
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    nq = model_context.nq

    if region == "interior":
        margin = 0.05 * (upper - lower)
        q_batch = _group_random_interior(rng, nq, needed * 3, lower, upper, margin)
        metrics = _compute_candidate_metrics(model_context, q_batch)
        keep = np.arange(q_batch.shape[0])
    elif region == "near_limit":
        # tier0 sampling-policy margins drive the biased near-limit pool (single source of truth)
        tier0 = _tier0_sampling_policy()
        q_batch = _group_mixed_near_limits(
            rng, nq, needed * 40, lower, upper, tier0["near_limit_margin_rad"], tier0["near_limit_band_rad"], tier0["interior_margin_rad"]
        )
        metrics = _compute_candidate_metrics(model_context, q_batch)
        keep = np.where(metrics["normalized_margin"] <= settings.near_joint_limit_threshold)[0]
    elif region == "near_singular":
        tier0 = _tier0_sampling_policy()
        q_batch = _build_singularity_candidate_pool(rng, nq, lower, upper, needed * 60, tier0["interior_margin_rad"])
        metrics = _compute_candidate_metrics(model_context, q_batch)
        keep = np.where(metrics["sigma_min"] <= settings.near_singularity_threshold)[0]
    else:
        raise ValueError(f"unknown challenge start region '{region}'")

    if keep.shape[0] < needed:
        raise RuntimeError(
            f"challenge start region '{region}' produced only {keep.shape[0]} eligible candidate(s) "
            f"of {q_batch.shape[0]} drawn; need {needed}. Increase the pool draw or the biased pool "
            "size (never relax the predicate)."
        )
    keep = keep[:needed]
    kept_metrics = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
    return q_batch[keep], kept_metrics


def _tier0_sampling_policy() -> dict:
    return {
        "interior_margin_rad": 0.15,
        "near_limit_margin_rad": 0.05,
        "near_limit_band_rad": 0.05,
    }


# ---------------------------------------------------------------------------------------------
# Bounded-Fourier joint-space reference curve (smooth, within-limits by construction)
# ---------------------------------------------------------------------------------------------


def _reference_joint_path(
    q_start: np.ndarray,
    coeff_a: np.ndarray,
    coeff_b: np.ndarray,
    harmonics: Sequence[int],
    weights: np.ndarray,
    envelope_fraction: float,
    lower: np.ndarray,
    upper: np.ndarray,
    s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """q(s) = q_start + offset(s), a smooth joint-space curve that vanishes at s=0 and whose
    per-joint amplitude is capped at ``envelope_fraction`` of the start's own joint-limit margin.

    Returns (q_of_s [n, nq], per_joint_scale [nq]). Guaranteed inside operational limits with no
    clipping, so the reference is C-infinity smooth (no kink).
    """
    n = s.shape[0]
    nq = q_start.shape[0]
    harmonics = np.asarray(harmonics, dtype=np.float64)
    # raw_j(s) = weight_j * sum_k [a_kj sin(pi m_k s) + b_kj (1 - cos(pi m_k s))]
    raw = np.zeros((n, nq), dtype=np.float64)
    for k, m in enumerate(harmonics):
        sin_term = np.sin(np.pi * m * s)[:, None]
        cos_term = (1.0 - np.cos(np.pi * m * s))[:, None]
        raw += sin_term * coeff_a[k][None, :] + cos_term * coeff_b[k][None, :]
    raw *= weights[None, :]

    margin = np.minimum(q_start - lower, upper - q_start)
    max_abs = np.max(np.abs(raw), axis=0)
    safe = np.where(max_abs > 1e-9, max_abs, 1.0)
    scale = np.minimum(1.0, envelope_fraction * margin / safe)
    scale = np.where(max_abs > 1e-9, scale, 0.0)

    offset = raw * scale[None, :]
    q_of_s = q_start[None, :] + offset
    # defensive clip (should be a no-op given the margin cap)
    q_of_s = np.clip(q_of_s, lower, upper)
    return q_of_s, scale


def _fk_path(model_context: ModelContext, q_of_s: np.ndarray, data=None) -> Tuple[np.ndarray, np.ndarray]:
    """Forward kinematics of a joint-space path -> (positions [n,3], sign-continuous quats [n,4])."""
    n = q_of_s.shape[0]
    data = data if data is not None else model_context.new_data()
    positions = np.empty((n, 3), dtype=np.float64)
    quaternions = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        fk = forward_kinematics(model_context, q_of_s[i], data=data)
        positions[i] = fk.position
        quaternions[i] = fk.quaternion_wxyz
    quaternions = _canonicalize_quat_sequence(quaternions)
    return positions, quaternions


# ---------------------------------------------------------------------------------------------
# Path geometry diagnostics (curvature, non-planarity)
# ---------------------------------------------------------------------------------------------


def path_curvature(positions: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Discrete Cartesian curvature kappa = |p' x p''| / |p'|^3 per waypoint (mean, max, array)."""
    n = positions.shape[0]
    if n < 3:
        return 0.0, 0.0, np.zeros(n)
    d1 = np.gradient(positions, axis=0)
    d2 = np.gradient(d1, axis=0)
    cross = np.cross(d1, d2)
    num = np.linalg.norm(cross, axis=1)
    denom = np.linalg.norm(d1, axis=1) ** 3
    kappa = np.where(denom > 1e-12, num / np.where(denom > 1e-12, denom, 1.0), 0.0)
    return float(np.mean(kappa)), float(np.max(kappa)), kappa


def path_non_planarity(positions: np.ndarray) -> float:
    """RMS distance of the path from its own best-fit plane, normalized by bounding-box diagonal."""
    centered = positions - positions.mean(axis=0)
    bbox = positions.max(axis=0) - positions.min(axis=0)
    diag = float(np.linalg.norm(bbox))
    if diag < 1e-12:
        return 0.0
    # plane normal = right-singular vector of smallest singular value
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    residual = centered @ normal
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return rms / diag


# ---------------------------------------------------------------------------------------------
# Candidate build (one family/split candidate: start + coefficients -> full built representation)
# ---------------------------------------------------------------------------------------------


def _draw_coefficients(rng: np.random.Generator, harmonics: Sequence[int], nq: int, base_amplitude: float) -> Tuple[np.ndarray, np.ndarray]:
    k = len(harmonics)
    a = rng.uniform(-1.0, 1.0, size=(k, nq)) * base_amplitude
    b = rng.uniform(-1.0, 1.0, size=(k, nq)) * base_amplitude
    return a, b


def build_challenge_representation(
    model_context: ModelContext,
    q_start: np.ndarray,
    coeff_a: np.ndarray,
    coeff_b: np.ndarray,
    family_def: dict,
    source_count: int,
    canonical_count: int,
    duration_s: float,
    data=None,
) -> dict:
    """Build the dual (source + canonical) representation for one challenge candidate."""
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    harmonics = family_def["harmonics"]
    weights = np.asarray(family_def["joint_amplitude_weights"], dtype=np.float64)
    envelope_fraction = float(family_def["envelope_margin_fraction"])

    time_s, tau, s, _sd, _sdd, control_period = build_time_and_path_parameter(source_count, duration_s)
    q_of_s, scale = _reference_joint_path(q_start, coeff_a, coeff_b, harmonics, weights, envelope_fraction, lower, upper, s)
    source_positions, source_quaternions = _fk_path(model_context, q_of_s, data=data)
    canonical = resample_canonical(time_s, tau, source_positions, source_quaternions, canonical_count)

    mean_curv, max_curv, _kappa = path_curvature(canonical["target_position"])
    non_planarity = path_non_planarity(canonical["target_position"])

    return {
        "source": {
            "time_s": time_s,
            "tau": tau,
            "q_of_s": q_of_s,
            "target_position": source_positions,
            "target_quaternion": source_quaternions,
        },
        "canonical": canonical,
        "control_period_s": control_period,
        "per_joint_scale": scale,
        "mean_curvature": mean_curv,
        "max_curvature": max_curv,
        "non_planarity": non_planarity,
    }


def _coverage_floor_ok(family_def: dict, built: dict) -> Tuple[bool, Optional[str]]:
    """Family coverage-floor acceptance (non_planar / large_orientation), plus curvature ceiling."""
    mean_curv = built["mean_curvature"]
    if not np.isfinite(mean_curv) or mean_curv > float(family_def["max_mean_curvature_1_per_m"]):
        return False, f"mean curvature {mean_curv} exceeds ceiling {family_def['max_mean_curvature_1_per_m']}"
    if "min_non_planarity" in family_def and built["non_planarity"] < float(family_def["min_non_planarity"]):
        return False, f"non_planarity {built['non_planarity']:.4f} < floor {family_def['min_non_planarity']}"
    if "min_angular_displacement_rad" in family_def:
        ang = float(built["canonical"]["cumulative_angular_displacement_rad"][-1])
        if ang < float(family_def["min_angular_displacement_rad"]):
            return False, f"angular displacement {ang:.4f} < floor {family_def['min_angular_displacement_rad']}"
    return True, None


# ---------------------------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------------------------


def _split_dir(paths: DatasetV2Paths, split_name: str) -> Path:
    return {
        "development": paths.trajectories_development_dir,
        "validation": paths.trajectories_validation_dir,
        "frozen_test": paths.trajectories_frozen_test_dir,
    }[split_name]


def _config_fingerprint(paths: DatasetV2Paths) -> str:
    import hashlib

    challenge_config = load_json_config(paths.configs_dir / "random_challenge_config.json")
    reachability_config = load_json_config(paths.configs_dir / "generation_reachability_config.json")
    difficulty = load_json_config(paths.configs_dir / "difficulty_thresholds.json")
    payload = {
        "random_challenge_config": challenge_config,
        "generation_reachability_config": reachability_config,
        "difficulty_thresholds": difficulty,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _start_content_hash(q_start: np.ndarray, metrics: dict, idx: int) -> str:
    return content_hash_of_record(
        {
            "q_start": [round(float(v), 12) for v in q_start],
            "sigma_min": round(float(metrics["sigma_min"][idx]), 12),
            "normalized_margin": round(float(metrics["normalized_margin"][idx]), 12),
        }
    )


def _path_metric_vector(built: dict) -> np.ndarray:
    canonical = built["canonical"]
    return np.array(
        [
            float(canonical["cumulative_arc_length_m"][-1]),
            float(canonical["cumulative_angular_displacement_rad"][-1]),
            built["mean_curvature"],
            built["non_planarity"],
        ],
        dtype=np.float64,
    )


def run_challenge_trajectory_generation(
    dataset_root,
    master_seed: Optional[int] = None,
    overwrite: bool = False,
    families: Optional[Sequence[str]] = None,
    splits: Optional[Sequence[str]] = None,
    per_family_per_split: Optional[int] = None,
    source_waypoint_count: Optional[int] = None,
    candidate_pool_size: Optional[int] = None,
    model_context: Optional[ModelContext] = None,
    dry_run: bool = False,
    progress: bool = False,
) -> ChallengeGenerationResult:
    """Generate Dataset v2's random-challenge trajectories: 6 families x 5 per split x 3 splits =
    90 (full/locked mode). ``families``/``splits``/``per_family_per_split``/``source_waypoint_count``
    /``candidate_pool_size`` exist for tests/smoke runs only -- passing any marks the run as not
    ``full_locked_counts``; the canonical waypoint count (400) never changes.
    """
    paths = require_dataset_v2_root(dataset_root)
    trajectories_dir = paths.trajectories_dir

    full_locked_counts = (
        families is None
        and splits is None
        and per_family_per_split is None
        and source_waypoint_count is None
        and candidate_pool_size is None
    )

    settings = load_challenge_generation_settings(paths, source_waypoint_count, candidate_pool_size, per_family_per_split)
    family_list = list(families) if families is not None else list(settings.families)
    split_list = list(splits) if splits is not None else list(SPLITS)
    for family in family_list:
        if family not in CHALLENGE_FAMILIES:
            raise ValueError(f"unknown challenge family '{family}'; must be one of {CHALLENGE_FAMILIES}")
    for split in split_list:
        if split not in SPLITS:
            raise ValueError(f"unknown split '{split}'; must be one of {SPLITS}")

    per_split_total = settings.per_family_per_split * len(family_list)

    # Plan output paths for the existence/overwrite check.
    planned: List[Tuple[str, str, int]] = []  # (split, family, family_slot)
    for split in split_list:
        index = 0
        for family in family_list:
            for _slot in range(settings.per_family_per_split):
                planned.append((split, family, index))
                index += 1
    output_paths = [
        trajectories_dir / MANIFEST_NAME,
        trajectories_dir / REPORT_NAME,
        trajectories_dir / REACHABILITY_REPORT_NAME,
        trajectories_dir / DIVERSITY_REPORT_NAME,
        trajectories_dir / FEASIBILITY_REPORT_NAME,
        trajectories_dir / ANTI_LEAKAGE_REPORT_NAME,
    ]
    for split, _family, index in planned:
        tid = f"challenge_{split}_{index:03d}"
        output_paths.append(_split_dir(paths, split) / f"{tid}{NPZ_SUFFIX}")
        output_paths.append(_split_dir(paths, split) / f"{tid}{SOURCE_SUFFIX}")

    existing = [p for p in output_paths if p.is_file()]
    if existing and not overwrite:
        existing_relative = ", ".join(str(p.relative_to(paths.root)) for p in existing[:10])
        more = "" if len(existing) <= 10 else f" (+{len(existing) - 10} more)"
        raise FileExistsError(
            f"Challenge trajectory v2 output already exists ({existing_relative}{more}); pass overwrite=True "
            "(--overwrite on the CLI) to regenerate it."
        )

    seed_policy = load_json_config(paths.configs_dir / "seed_policy.json")
    resolved_master_seed = int(master_seed if master_seed is not None else seed_policy["master_seed"])
    component_tag = int(SEED_COMPONENT_TAGS["random_challenge"])
    component_seed = derive_seed(resolved_master_seed, component_tag)

    if dry_run:
        return ChallengeGenerationResult(
            dataset_root=paths.root,
            trajectories_dir=trajectories_dir,
            dry_run=True,
            total_trajectories=len(planned),
            full_locked_counts=full_locked_counts,
            split_counts={s: per_split_total for s in split_list},
            family_counts={f: settings.per_family_per_split * len(split_list) for f in family_list},
        )

    model_context = model_context if model_context is not None else load_model_context()
    model_fingerprint = sha256_file(V1_MODEL_PATH)
    config_fingerprint = _config_fingerprint(paths)

    reach_settings = load_generation_reachability_settings(paths)
    probe = probe_settings_from(paths, reach_settings)

    manifest_rows: List[list] = []
    trajectory_records: List[dict] = []  # rich per-trajectory records for reports / anti-leakage
    feasibility_records: List[dict] = []
    diversity_records: List[dict] = []

    data = model_context.new_data()

    import sys as _sys
    import time as _time

    run_started = _time.perf_counter()

    for split in split_list:
        split_index = 0
        for family in family_list:
            family_def = settings.family_definitions[family]
            family_tag = int(CHALLENGE_FAMILY_TAGS[family])
            split_tag = int(SEED_SPLIT_TAGS[split])
            if split == "frozen_test":
                family_split_seed = derive_seed(component_seed, family_tag, split_tag, settings.frozen_challenge_seed_revision)
            else:
                family_split_seed = derive_seed(component_seed, family_tag, split_tag)

            selected, fam_feas, fam_div = _select_family_split(
                model_context,
                data,
                family,
                family_def,
                split,
                family_split_seed,
                settings,
                reach_settings,
                probe,
                progress,
            )
            feasibility_records.append(fam_feas)
            diversity_records.append(fam_div)

            for built, cand in selected:
                tid = f"challenge_{split}_{split_index:03d}"
                split_index += 1
                record = _finalize_trajectory(
                    paths,
                    tid,
                    family,
                    split,
                    cand,
                    built,
                    settings,
                    reach_settings,
                    component_seed,
                    model_fingerprint,
                    config_fingerprint,
                    overwrite,
                )
                trajectory_records.append(record)
                manifest_rows.append(record["manifest_row"])

            if progress:
                print(
                    f"[challenge-generate] {split}/{family} selected {len(selected)} "
                    f"[total {(_time.perf_counter() - run_started) / 60:.1f}min]",
                    flush=True,
                    file=_sys.stderr,
                )

    # ---- global integrity: uniqueness + anti-leakage ------------------------------------------
    _assert_and_report(
        paths,
        trajectory_records,
        manifest_rows,
        feasibility_records,
        diversity_records,
        settings,
        split_list,
        family_list,
        reach_settings,
        resolved_master_seed,
        component_tag,
        component_seed,
        model_fingerprint,
        config_fingerprint,
        full_locked_counts,
    )

    split_counts: Dict[str, int] = {s: 0 for s in split_list}
    family_counts: Dict[str, int] = {f: 0 for f in family_list}
    family_split_counts: Dict[str, Dict[str, int]] = {f: {s: 0 for s in split_list} for f in family_list}
    for rec in trajectory_records:
        split_counts[rec["split"]] += 1
        family_counts[rec["challenge_family"]] += 1
        family_split_counts[rec["challenge_family"]][rec["split"]] += 1

    reachability_statistics = _reachability_statistics(trajectory_records, settings, reach_settings)
    diversity_statistics = {"families": diversity_records}

    return ChallengeGenerationResult(
        dataset_root=paths.root,
        trajectories_dir=trajectories_dir,
        dry_run=False,
        total_trajectories=len(trajectory_records),
        full_locked_counts=full_locked_counts,
        split_counts=split_counts,
        family_counts=family_counts,
        family_split_counts=family_split_counts,
        reachability_statistics=reachability_statistics,
        diversity_statistics=diversity_statistics,
        report=None,
    )


def _select_family_split(
    model_context,
    data,
    family,
    family_def,
    split,
    family_split_seed,
    settings: ChallengeGenerationSettings,
    reach_settings,
    probe,
    progress,
) -> Tuple[List[Tuple[dict, dict]], dict, dict]:
    """Feasibility-aware diversity selection for one (family, split): draw a candidate pool,
    coarse-probe screen for strict reachability + coverage floors, diversity-select the quota, then
    full-validate; replace a full-validation failure from the same feasible pool.
    """
    rng = np.random.default_rng(family_split_seed)
    region = family_def["region"]
    nq = model_context.nq
    base_amplitude = float(family_def["base_amplitude_rad"])
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad

    q_starts, start_metrics = _draw_region_starts(model_context, rng, region, settings.candidate_pool_size, settings)

    # Per-candidate coefficients drawn from a stable per-candidate seed.
    candidates: List[dict] = []
    for i in range(settings.candidate_pool_size):
        cand_seed = derive_seed(family_split_seed, CANDIDATE_TAG, i)
        coeff_a, coeff_b = _draw_coefficients(np.random.default_rng(derive_seed(cand_seed, COEFF_TAG)), family_def["harmonics"], nq, base_amplitude)
        candidates.append(
            {
                "index": i,
                "q_start": q_starts[i],
                "coeff_a": coeff_a,
                "coeff_b": coeff_b,
                "cand_seed": cand_seed,
                "sigma_min": float(start_metrics["sigma_min"][i]),
                "sigma_max": float(start_metrics["sigma_max"][i]),
                "condition_number": float(start_metrics["condition_number"][i]),
                "normalized_margin": float(start_metrics["normalized_margin"][i]),
                "absolute_margin_rad": float(start_metrics["absolute_margin_rad"][i]),
                "controlling_joint": int(start_metrics["controlling_joint"][i]),
                "position": start_metrics["position"][i],
                "start_content_hash": _start_content_hash(q_starts[i], start_metrics, i),
            }
        )

    # Coarse feasibility screen (strict reachability at coarse resolution + coverage floors).
    feasible: List[dict] = []
    screen_details: List[dict] = []
    for cand in candidates:
        built = build_challenge_representation(
            model_context,
            cand["q_start"],
            cand["coeff_a"],
            cand["coeff_b"],
            family_def,
            settings.coarse_source_waypoints,
            settings.coarse_canonical_waypoints,
            settings.duration_s,
            data=data,
        )
        floor_ok, floor_reason = _coverage_floor_ok(family_def, built)
        reachable = False
        reason = floor_reason
        if floor_ok:
            result = validate_path_strict(
                model_context,
                probe,
                cand["q_start"],
                built["canonical"]["target_position"],
                built["canonical"]["target_quaternion"],
                path_seed=cand["cand_seed"],
                stop_on_first_failure=True,
                data=data,
            )
            reachable = bool(result["all_reachable"])
            if not reachable:
                reason = f"coarse waypoint {int(result['first_failure_index'])} unreachable"
        eligible = floor_ok and reachable
        screen_details.append({"index": cand["index"], "eligible": eligible, "reason": None if eligible else reason})
        if eligible:
            cand = dict(cand)
            cand["coarse_features"] = _diversity_feature(cand, built)
            feasible.append(cand)

    quota = settings.per_family_per_split
    if len(feasible) < quota:
        raise RuntimeError(
            f"challenge family '{family}' split '{split}' has only {len(feasible)} feasible candidate(s) "
            f"of {settings.candidate_pool_size} screened; need {quota}. Increase the candidate pool "
            "(never loosen strict reachability or the coverage floors)."
        )

    # Diversity selection over the feasible subset.
    feature_matrix = _normalize_features(np.array([c["coarse_features"] for c in feasible]))
    order_rng = np.random.default_rng(derive_seed(family_split_seed, 99))
    # Select more than the quota so full-validation failures can be replaced deterministically.
    select_k = min(len(feasible), quota + 4)
    selected_local = greedy_farthest_point_select(order_rng, feature_matrix, select_k, f"challenge_{family}_{split}")
    ranked = [feasible[int(i)] for i in selected_local]

    accepted: List[Tuple[dict, dict]] = []
    full_attempts: List[dict] = []
    for cand in ranked:
        if len(accepted) >= quota:
            break
        built_full = build_challenge_representation(
            model_context,
            cand["q_start"],
            cand["coeff_a"],
            cand["coeff_b"],
            family_def,
            settings.source_waypoint_count,
            settings.canonical_waypoint_count,
            settings.duration_s,
            data=data,
        )
        floor_ok, floor_reason = _coverage_floor_ok(family_def, built_full)
        canonical_result = None
        source_result = None
        ok = False
        if floor_ok:
            canonical_result = validate_path_strict(
                model_context, reach_settings, cand["q_start"], built_full["canonical"]["target_position"],
                built_full["canonical"]["target_quaternion"], path_seed=cand["cand_seed"], data=data,
            )
            if canonical_result["all_reachable"]:
                source_result = validate_path_strict(
                    model_context, reach_settings, cand["q_start"], built_full["source"]["target_position"],
                    built_full["source"]["target_quaternion"], path_seed=cand["cand_seed"], data=data,
                )
                ok = bool(source_result["all_reachable"])
        full_attempts.append({"index": cand["index"], "accepted": ok, "reason": None if ok else (floor_reason or "full-resolution reachability failed")})
        if ok:
            built_full["canonical_result"] = canonical_result
            built_full["source_result"] = source_result
            accepted.append((built_full, cand))

    if len(accepted) < quota:
        raise RuntimeError(
            f"challenge family '{family}' split '{split}' produced only {len(accepted)} fully-validated "
            f"trajectory(ies) of the required {quota}; refusing to reduce counts or loosen reachability. "
            f"Diversity-ranked candidates tried: {[a['index'] for a in full_attempts]}"
        )
    accepted = accepted[:quota]

    feasibility_report = {
        "family": family,
        "split": split,
        "candidates_screened": len(candidates),
        "candidates_feasible": len(feasible),
        "quota": quota,
        "screen_details": screen_details,
        "full_validation_attempts": full_attempts,
    }
    diversity_report = {
        "family": family,
        "split": split,
        "feasible_count": len(feasible),
        "diversity_ranked_indices": [cand["index"] for cand in ranked],
        "selected_indices": [cand["index"] for _b, cand in accepted],
        "feature_names": [
            "start_q(7)",
            "workspace_centroid(3)",
            "arc_length",
            "angular_displacement",
            "mean_curvature",
            "non_planarity",
            "start_sigma_min",
            "start_normalized_margin",
        ],
    }
    return accepted, feasibility_report, diversity_report


def _diversity_feature(cand: dict, built: dict) -> np.ndarray:
    canonical = built["canonical"]
    centroid = canonical["target_position"].mean(axis=0)
    return np.concatenate(
        [
            np.asarray(cand["q_start"], dtype=np.float64),
            centroid,
            np.array(
                [
                    float(canonical["cumulative_arc_length_m"][-1]),
                    float(canonical["cumulative_angular_displacement_rad"][-1]),
                    built["mean_curvature"],
                    built["non_planarity"],
                    cand["sigma_min"],
                    cand["normalized_margin"],
                ]
            ),
        ]
    )


def _normalize_features(features: np.ndarray) -> np.ndarray:
    """Column min-max normalization; constant columns collapse to 0 (no divide-by-zero)."""
    lo = features.min(axis=0)
    hi = features.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (features - lo) / span


def _finalize_trajectory(
    paths,
    tid,
    family,
    split,
    cand,
    built,
    settings: ChallengeGenerationSettings,
    reach_settings,
    component_seed,
    model_fingerprint,
    config_fingerprint,
    overwrite,
) -> dict:
    canonical = built["canonical"]
    source = built["source"]
    canonical_result = built["canonical_result"]
    source_result = built["source_result"]
    family_def = settings.family_definitions[family]

    canonical_errors = error_distribution(canonical_result["position_errors_m"], canonical_result["orientation_errors_rad"])
    source_errors = error_distribution(source_result["position_errors_m"], source_result["orientation_errors_rad"])

    arc_length_m = float(canonical["cumulative_arc_length_m"][-1])
    angular_displacement_rad = float(canonical["cumulative_angular_displacement_rad"][-1])
    _mean_curv, _max_curv, kappa = path_curvature(canonical["target_position"])

    frozen_rev = settings.frozen_challenge_seed_revision if split == "frozen_test" else 0
    harmonics_json = json.dumps({"harmonics": list(family_def["harmonics"]), "base_amplitude_rad": family_def["base_amplitude_rad"]}, sort_keys=True)
    geometry_params = {
        "region": family_def["region"],
        "joint_amplitude_weights": list(family_def["joint_amplitude_weights"]),
        "envelope_margin_fraction": float(family_def["envelope_margin_fraction"]),
        "per_joint_scale": [round(float(v), 12) for v in built["per_joint_scale"]],
        "coeff_a": [[round(float(v), 12) for v in row] for row in cand["coeff_a"]],
        "coeff_b": [[round(float(v), 12) for v in row] for row in cand["coeff_b"]],
        "q_start": [round(float(v), 12) for v in cand["q_start"]],
    }

    content_hash = content_hash_of_record(
        {
            "challenge_family": family,
            "split": split,
            "start_content_hash": cand["start_content_hash"],
            "geometry_parameters": geometry_params,
            "target_position": [[round(float(v), 9) for v in p] for p in canonical["target_position"]],
            "target_quaternion": [[round(float(v), 9) for v in q] for q in canonical["target_quaternion"]],
            "model_fingerprint": model_fingerprint,
            "config_fingerprint": config_fingerprint,
        }
    )
    canonical_path_hash = content_hash_of_record({"p": [[round(float(v), 9) for v in p] for p in canonical["target_position"]]})
    source_path_hash = content_hash_of_record({"p": [[round(float(v), 9) for v in p] for p in source["target_position"]]})

    split_dir = _split_dir(paths, split)
    canonical_npz_path = split_dir / f"{tid}{NPZ_SUFFIX}"
    source_npz_path = split_dir / f"{tid}{SOURCE_SUFFIX}"

    canonical_arrays = {
        "waypoint_id": np.arange(settings.canonical_waypoint_count, dtype=np.int64),
        "time_s": canonical["time_s"].astype(np.float64),
        "source_parameter_u": canonical["source_parameter_u"].astype(np.float64),
        "cumulative_arc_length_m": canonical["cumulative_arc_length_m"].astype(np.float64),
        "target_position": canonical["target_position"].astype(np.float64),
        "target_quaternion": canonical["target_quaternion"].astype(np.float64),
        "cumulative_angular_displacement_rad": canonical["cumulative_angular_displacement_rad"].astype(np.float64),
        "curvature_1_per_m": kappa.astype(np.float64),
        "q_reference": canonical_result["q_reference"].astype(np.float64),
        "position_reconstruction_error_m": canonical_result["position_errors_m"].astype(np.float64),
        "orientation_reconstruction_error_rad": canonical_result["orientation_errors_rad"].astype(np.float64),
        "waypoint_reachable": canonical_result["reachable"].astype(bool),
    }
    source_arrays = {
        "waypoint_id": np.arange(settings.source_waypoint_count, dtype=np.int64),
        "time_s": source["time_s"].astype(np.float64),
        "tau": source["tau"].astype(np.float64),
        "q_source_reference": source["q_of_s"].astype(np.float64),
        "target_position": source["target_position"].astype(np.float64),
        "target_quaternion": source["target_quaternion"].astype(np.float64),
        "q_reference": source_result["q_reference"].astype(np.float64),
        "position_reconstruction_error_m": source_result["position_errors_m"].astype(np.float64),
        "orientation_reconstruction_error_rad": source_result["orientation_errors_rad"].astype(np.float64),
        "waypoint_reachable": source_result["reachable"].astype(bool),
    }

    save_npz(canonical_npz_path, canonical_arrays, overwrite=overwrite)
    save_npz(source_npz_path, source_arrays, overwrite=overwrite)
    sha = sha256_file(canonical_npz_path)
    source_sha = sha256_file(source_npz_path)

    manifest_row = [
        tid,
        "random_challenge",
        family,
        split,
        int(cand["index"]),
        int(component_seed),
        int(cand["cand_seed"]),
        int(frozen_rev),
        settings.source_waypoint_count,
        settings.canonical_waypoint_count,
        "wxyz",
        f"{settings.duration_s:.10f}",
        f"{built['control_period_s']:.10f}",
        json.dumps([round(float(v), 10) for v in cand["position"]]),
        f"{cand['sigma_min']:.12g}",
        f"{cand['sigma_max']:.12g}",
        f"{cand['condition_number']:.12g}",
        f"{cand['normalized_margin']:.12g}",
        f"{cand['absolute_margin_rad']:.12g}",
        int(cand["controlling_joint"]),
        cand["start_content_hash"],
        f"{float(family_def['envelope_margin_fraction']):.10f}",
        harmonics_json,
        json.dumps(geometry_params, sort_keys=True),
        f"{arc_length_m:.10f}",
        f"{angular_displacement_rad:.10f}",
        f"{built['mean_curvature']:.12g}",
        f"{built['max_curvature']:.12g}",
        f"{built['non_planarity']:.12g}",
        "validated",
        f"{reach_settings.position_tolerance_m:.10g}",
        f"{reach_settings.orientation_tolerance_deg:.10g}",
        f"{canonical_errors['position_max_m']:.12g}",
        f"{canonical_errors['orientation_max_deg']:.12g}",
        f"{source_errors['position_max_m']:.12g}",
        f"{source_errors['orientation_max_deg']:.12g}",
        int(np.sum(canonical_result["reachable"])),
        int(np.sum(source_result["reachable"])),
        "development",
        model_fingerprint,
        config_fingerprint,
        content_hash,
        sha,
        source_sha,
    ]

    return {
        "trajectory_id": tid,
        "challenge_family": family,
        "split": split,
        "content_hash": content_hash,
        "canonical_path_hash": canonical_path_hash,
        "source_path_hash": source_path_hash,
        "path_seed": int(cand["cand_seed"]),
        "q_start": np.asarray(cand["q_start"], dtype=np.float64),
        "path_metric_vector": _path_metric_vector(built),
        "canonical_errors": canonical_errors,
        "source_errors": source_errors,
        "canonical_waypoints_reachable": int(np.sum(canonical_result["reachable"])),
        "source_waypoints_reachable": int(np.sum(source_result["reachable"])),
        "arc_length_m": arc_length_m,
        "angular_displacement_rad": angular_displacement_rad,
        "mean_curvature": built["mean_curvature"],
        "non_planarity": built["non_planarity"],
        "manifest_row": manifest_row,
    }


def _reachability_statistics(trajectory_records, settings, reach_settings) -> dict:
    return {
        "tolerance_position_m": reach_settings.position_tolerance_m,
        "tolerance_orientation_deg": reach_settings.orientation_tolerance_deg,
        "tolerance_source": "configs/generation_reachability_config.json (Dataset v2's own; never the DLS baseline evaluation config)",
        "all_canonical_waypoints_reachable": all(
            r["canonical_waypoints_reachable"] == settings.canonical_waypoint_count for r in trajectory_records
        ),
        "all_source_waypoints_reachable": all(
            r["source_waypoints_reachable"] == settings.source_waypoint_count for r in trajectory_records
        ),
        "canonical_position_max_m": max(r["canonical_errors"]["position_max_m"] for r in trajectory_records),
        "canonical_position_p95_m": float(np.percentile([r["canonical_errors"]["position_p95_m"] for r in trajectory_records], 95)),
        "canonical_orientation_max_deg": max(r["canonical_errors"]["orientation_max_deg"] for r in trajectory_records),
        "source_position_max_m": max(r["source_errors"]["position_max_m"] for r in trajectory_records),
        "source_position_p95_m": float(np.percentile([r["source_errors"]["position_p95_m"] for r in trajectory_records], 95)),
        "source_orientation_max_deg": max(r["source_errors"]["orientation_max_deg"] for r in trajectory_records),
        "arc_length_range_m": [min(r["arc_length_m"] for r in trajectory_records), max(r["arc_length_m"] for r in trajectory_records)],
        "angular_displacement_range_rad": [
            min(r["angular_displacement_rad"] for r in trajectory_records),
            max(r["angular_displacement_rad"] for r in trajectory_records),
        ],
        "mean_curvature_range": [min(r["mean_curvature"] for r in trajectory_records), max(r["mean_curvature"] for r in trajectory_records)],
        "non_planarity_range": [min(r["non_planarity"] for r in trajectory_records), max(r["non_planarity"] for r in trajectory_records)],
    }


def _assert_and_report(
    paths,
    trajectory_records,
    manifest_rows,
    feasibility_records,
    diversity_records,
    settings: ChallengeGenerationSettings,
    split_list,
    family_list,
    reach_settings,
    resolved_master_seed,
    component_tag,
    component_seed,
    model_fingerprint,
    config_fingerprint,
    full_locked_counts,
) -> None:
    trajectories_dir = paths.trajectories_dir

    ids = [r["trajectory_id"] for r in trajectory_records]
    hashes = [r["content_hash"] for r in trajectory_records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate trajectory_id found across the generated challenge set")
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate content_hash found across the generated challenge set")

    # Cross-split anti-leakage: exact + near-duplicate.
    by_split: Dict[str, List[dict]] = {s: [] for s in split_list}
    for r in trajectory_records:
        by_split[r["split"]].append(r)

    collisions: List[dict] = []
    for i in range(len(split_list)):
        for j in range(i + 1, len(split_list)):
            a, b = split_list[i], split_list[j]
            # exact identifier overlaps
            for dim in ("trajectory_id", "content_hash", "canonical_path_hash", "source_path_hash", "path_seed"):
                sa = {r[dim] for r in by_split[a]}
                sb = {r[dim] for r in by_split[b]}
                overlap = sa & sb
                if overlap:
                    collisions.append({"splits": [a, b], "dimension": dim, "overlap": sorted(str(x) for x in overlap)})
            # near-duplicate start q + path metrics
            for ra in by_split[a]:
                for rb in by_split[b]:
                    dq = float(np.max(np.abs(ra["q_start"] - rb["q_start"])))
                    denom = np.maximum(np.abs(ra["path_metric_vector"]), 1e-9)
                    dmetric = float(np.max(np.abs(ra["path_metric_vector"] - rb["path_metric_vector"]) / denom))
                    if dq < settings.start_near_duplicate_rad and dmetric < settings.path_metric_relative:
                        collisions.append(
                            {
                                "splits": [a, b],
                                "dimension": "near_duplicate_start_and_path_metrics",
                                "pair": [ra["trajectory_id"], rb["trajectory_id"]],
                                "start_q_linf_rad": dq,
                                "path_metric_rel": dmetric,
                            }
                        )

    # No duplicate with any core trajectory (by content hash), if a core manifest exists.
    core_manifest = trajectories_dir / "core_trajectory_manifest.csv"
    core_overlap: List[str] = []
    if core_manifest.is_file():
        core_hashes = set()
        with open(core_manifest, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                core_hashes.add(row.get("content_hash", ""))
        overlap = set(hashes) & core_hashes
        if overlap:
            core_overlap = sorted(overlap)
            collisions.append({"dimension": "duplicate_with_core_trajectory", "overlap": core_overlap})

    anti_leakage_report = {
        "dimensions_checked": [
            "trajectory_id",
            "trajectory_content_hash",
            "canonical_path_hash",
            "source_path_hash",
            "path_seed",
            "near_duplicate_start_and_path_metrics",
            "duplicate_with_core_trajectory",
        ],
        "collisions_found": len(collisions),
        "collision_details": collisions,
        "core_manifest_present": core_manifest.is_file(),
        "pass": len(collisions) == 0,
    }
    if collisions:
        raise ValueError(f"challenge trajectory anti-leakage failed: {collisions}")

    # ---- write manifest, reports, checksum, manifest update -----------------------------------
    manifest_path = _atomic_write_csv(trajectories_dir / MANIFEST_NAME, MANIFEST_COLUMNS, manifest_rows)
    _atomic_write_json(trajectories_dir / ANTI_LEAKAGE_REPORT_NAME, anti_leakage_report)
    _atomic_write_json(trajectories_dir / FEASIBILITY_REPORT_NAME, {"families": feasibility_records})
    _atomic_write_json(trajectories_dir / DIVERSITY_REPORT_NAME, {"families": diversity_records})

    reachability_statistics = _reachability_statistics(trajectory_records, settings, reach_settings)
    _atomic_write_json(
        trajectories_dir / REACHABILITY_REPORT_NAME,
        {
            "statistics": reachability_statistics,
            "trajectories": [
                {
                    "trajectory_id": r["trajectory_id"],
                    "challenge_family": r["challenge_family"],
                    "split": r["split"],
                    "canonical_position_max_m": r["canonical_errors"]["position_max_m"],
                    "source_position_max_m": r["source_errors"]["position_max_m"],
                    "arc_length_m": r["arc_length_m"],
                    "angular_displacement_rad": r["angular_displacement_rad"],
                    "mean_curvature": r["mean_curvature"],
                    "non_planarity": r["non_planarity"],
                }
                for r in trajectory_records
            ],
        },
    )

    split_counts: Dict[str, int] = {s: 0 for s in split_list}
    family_counts: Dict[str, int] = {f: 0 for f in family_list}
    family_split_counts: Dict[str, Dict[str, int]] = {f: {s: 0 for s in split_list} for f in family_list}
    for rec in trajectory_records:
        split_counts[rec["split"]] += 1
        family_counts[rec["challenge_family"]] += 1
        family_split_counts[rec["challenge_family"]][rec["split"]] += 1

    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "dataset_version": DATASET_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "git_commit": _git_commit(REPO_ROOT),
        "model_fingerprint": model_fingerprint,
        "config_fingerprint": config_fingerprint,
        "master_seed": resolved_master_seed,
        "generation_timestamp_utc": generated_at,
        "generation_status": "development",
        "seed_derivation": {
            "random_challenge_component_tag": component_tag,
            "random_challenge_component_seed": component_seed,
            "frozen_challenge_seed_revision": settings.frozen_challenge_seed_revision,
            "frozen_seed_policy": (
                "frozen_test challenge path seeds and coefficient draws mix in "
                "frozen_challenge_seed_revision; development/validation use the unrevised namespace"
            ),
        },
        "full_locked_counts": full_locked_counts,
        "total_trajectories": len(trajectory_records),
        "split_counts": split_counts,
        "family_counts": family_counts,
        "family_split_counts": family_split_counts,
        "canonical_waypoints_per_trajectory": settings.canonical_waypoint_count,
        "source_waypoint_count": settings.source_waypoint_count,
        "canonical_poses_total": len(trajectory_records) * settings.canonical_waypoint_count,
        "reachability_statistics": reachability_statistics,
        "anti_leakage_report": anti_leakage_report,
        "output_files": {
            "manifest": {"filename": relative_to_dataset_v2_root(manifest_path, paths.root), "sha256": sha256_file(manifest_path)},
        },
    }
    _atomic_write_json(trajectories_dir / REPORT_NAME, report)

    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    manifest = apply_challenge_trajectory_generation_status(
        manifest,
        total_trajectories=len(trajectory_records),
        split_counts=split_counts,
        family_counts=family_counts,
        family_split_counts=family_split_counts,
        canonical_waypoints_per_trajectory=settings.canonical_waypoint_count,
        full_locked_counts=full_locked_counts,
    )
    _atomic_write_json(paths.manifest_file, manifest)

    checksum_manifest = build_checksum_manifest(paths.root)
    _atomic_write_json(paths.checksum_manifest_file, checksum_manifest)
