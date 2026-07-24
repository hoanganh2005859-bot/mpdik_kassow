"""Dataset v2 trial generator (Phase 7): 3 trials (easy/medium/hard) per trajectory across all 210
core + random-challenge trajectories -> 630 trials.

Each trial's ``q_initial`` is drawn INDEPENDENTLY from the operational joint limits (a deterministic
four-sub-pool mixture, ``dataset_v2/trial_candidates.py``) and classified only against the
trajectory's FIRST canonical target pose. ``q_initial`` is never derived from, or perturbed around,
the trajectory's protected ``q_reference`` (spec section J); no future waypoint's solution is ever
used. Difficulty is the LOCKED combined-normalized-pose-error metric with non-overlapping
easy/medium/hard bands (``configs/trial_config.json`` ``difficulty`` block), calibrated on
development trajectories only.

Public trial records (id / split / family / difficulty, ``q_initial``, FK pose, first target pose,
pose-error covariates, conditioning covariates, seeds, hashes) are written per split; the protected
reference evidence (``q_reference_start``, reference FK consistency, diagnostic joint distances) is
written to a separate ``trials/protected/`` tree the public trial loader never reads.

This module never modifies Dataset v1, never runs DLS, and never uses global ``numpy.random``
state.
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from dataset_v2.checksums import build_checksum_manifest, content_hash_of_record
from dataset_v2.config_templates import DATASET_SCHEMA_VERSION, DATASET_VERSION, TRIAL_INIT_CLASSES, TRIAL_INIT_CLASS_TAGS
from dataset_v2.core_trajectory_generation import _atomic_write_csv, _atomic_write_json, _git_commit
from dataset_v2.generation_reachability import fk_reconstruction_error
from dataset_v2.locator import DatasetV2Paths, relative_to_dataset_v2_root, require_dataset_v2_root
from dataset_v2.manifest import apply_trial_generation_status
from dataset_v2.seeds import derive_seed
from dataset_v2.trajectory_catalog import build_combined_catalog, load_combined_catalog
from dataset_v2.trajectory_loading import load_protected_trajectory, load_public_trajectory
from dataset_v2.trial_candidates import (
    build_candidate_pool,
    compute_candidate_covariates,
    pose_errors,
    primary_metric,
    trial_pool_seed,
    trial_source_seed,
)
from kinematics.model_loader import ModelContext, load_model_context
from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from utils.config_loader import load_json_config
from utils.dataset_locator import MODEL_PATH as V1_MODEL_PATH, REPO_ROOT
from utils.file_checksum import sha256_file
from utils.npz_utils import save_npz

GENERATOR_VERSION = "1.0.0"

SPLITS = ("development", "validation", "frozen_test")
INIT_CLASS_ID = {"easy": 0, "medium": 1, "hard": 2}

TRIAL_MANIFEST_NAME = "trial_manifest.csv"
GENERATION_REPORT_NAME = "trial_generation_report.json"
DISTRIBUTION_REPORT_NAME = "trial_distribution_report.json"
ANTI_LEAKAGE_REPORT_NAME = "trial_anti_leakage_report.json"
PROTECTED_DIR_NAME = "protected"

# Reference-start integrity: FK(q_reference_start) must reproduce the first target within a loose
# gate (well above the strict generation tolerance) or the trajectory data itself is inconsistent.
REFERENCE_START_POSITION_GATE_M = 1e-3
REFERENCE_START_ORIENTATION_GATE_RAD = np.deg2rad(0.1)

# Near-duplicate q_initial threshold for the anti-leakage report (L-inf, rad).
Q_INITIAL_NEAR_DUPLICATE_RAD = 1e-6

TRIAL_MANIFEST_COLUMNS = [
    "trial_id",
    "trajectory_id",
    "trajectory_family",
    "split",
    "difficulty",
    "q_initial",
    "initial_position",
    "initial_quaternion_wxyz",
    "first_target_position",
    "first_target_quaternion_wxyz",
    "initial_position_error_m",
    "initial_orientation_error_rad",
    "primary_difficulty_metric",
    "initial_sigma_min",
    "initial_sigma_max",
    "initial_condition_number",
    "minimum_initial_limit_margin_normalized",
    "minimum_initial_limit_margin_absolute_rad",
    "controlling_joint_index",
    "candidate_source_pool",
    "source_seed",
    "trajectory_content_hash",
    "model_fingerprint",
    "config_fingerprint",
    "content_hash",
]


@dataclass(frozen=True)
class TrialDifficultyPolicy:
    primary_metric: str
    position_weight: float
    orientation_weight: float
    position_scale_m: float
    orientation_scale_rad: float
    easy_upper: float
    medium_lower: float
    medium_upper: float
    hard_lower: float
    minimum_inter_level_separation: float


@dataclass(frozen=True)
class TrialGenerationSettings:
    interior_samples: int
    near_limit_samples: int
    singular_samples: int
    stratified_samples: int
    interior_margin_fraction: float
    stratified_margin_fraction: float
    difficulty: TrialDifficultyPolicy
    frozen_core_revision: int
    frozen_challenge_revision: int
    frozen_trial_revision: int


@dataclass
class TrialGenerationResult:
    dataset_root: Path
    trials_dir: Path
    dry_run: bool
    total_trials: int
    full_locked_counts: bool
    split_counts: Dict[str, int] = field(default_factory=dict)
    difficulty_counts: Dict[str, int] = field(default_factory=dict)
    family_counts: Dict[str, int] = field(default_factory=dict)
    split_difficulty_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    separation_statistics: dict = field(default_factory=dict)
    report: Optional[dict] = None


def load_trial_generation_settings(paths: DatasetV2Paths, pool_scale_override: Optional[int] = None) -> TrialGenerationSettings:
    trial_config = load_json_config(paths.configs_dir / "trial_config.json")
    seed_policy = load_json_config(paths.configs_dir / "seed_policy.json")
    policy = trial_config["candidate_pool_policy"]
    difficulty = trial_config["difficulty"]

    if pool_scale_override is None:
        interior = int(policy["interior_samples"])
        near_limit = int(policy["near_limit_samples"])
        singular = int(policy["singular_samples"])
        stratified = int(policy["stratified_samples"])
    else:
        interior = near_limit = singular = stratified = int(pool_scale_override)

    diff = TrialDifficultyPolicy(
        primary_metric=str(difficulty["primary_metric"]),
        position_weight=float(difficulty["position_weight"]),
        orientation_weight=float(difficulty["orientation_weight"]),
        position_scale_m=float(difficulty["position_scale_m"]),
        orientation_scale_rad=float(difficulty["orientation_scale_rad"]),
        easy_upper=float(difficulty["bands"]["easy"]["primary_max_inclusive"]),
        medium_lower=float(difficulty["bands"]["medium"]["primary_min_inclusive"]),
        medium_upper=float(difficulty["bands"]["medium"]["primary_max_inclusive"]),
        hard_lower=float(difficulty["bands"]["hard"]["primary_min_inclusive"]),
        minimum_inter_level_separation=float(difficulty["minimum_inter_level_separation"]),
    )
    return TrialGenerationSettings(
        interior_samples=interior,
        near_limit_samples=near_limit,
        singular_samples=singular,
        stratified_samples=stratified,
        interior_margin_fraction=float(policy["interior_margin_fraction"]),
        stratified_margin_fraction=float(policy["stratified_margin_fraction"]),
        difficulty=diff,
        frozen_core_revision=int(seed_policy["frozen_core_seed_revision"]),
        frozen_challenge_revision=int(seed_policy["frozen_challenge_seed_revision"]),
        frozen_trial_revision=int(seed_policy["frozen_trial_seed_revision"]),
    )


def classify_primary(primary: np.ndarray, diff: TrialDifficultyPolicy) -> Dict[str, np.ndarray]:
    """Return boolean masks for each band on the primary metric. Guard gaps are unassigned."""
    return {
        "easy": primary <= diff.easy_upper,
        "medium": (primary >= diff.medium_lower) & (primary <= diff.medium_upper),
        "hard": primary >= diff.hard_lower,
    }


def _select_representative(primary: np.ndarray, mask: np.ndarray) -> int:
    """Deterministically pick a representative candidate index for a band: the in-band candidate
    whose primary metric is closest to the band's own median (never the first/most-trivial one);
    ties broken by lowest global index."""
    idx = np.where(mask)[0]
    band_primary = primary[idx]
    center = float(np.median(band_primary))
    dist = np.abs(band_primary - center)
    order = np.lexsort((idx, dist))  # primary key: distance, tie-break: lowest index
    return int(idx[order[0]])


def _config_fingerprint(paths: DatasetV2Paths) -> str:
    import hashlib

    trial_config = load_json_config(paths.configs_dir / "trial_config.json")
    seed_policy = load_json_config(paths.configs_dir / "seed_policy.json")
    payload = {"trial_config": trial_config, "seed_policy": seed_policy}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_joint_distance(q_initial: np.ndarray, q_reference_start: np.ndarray, span: np.ndarray) -> float:
    return float(np.linalg.norm((q_initial - q_reference_start) / span))


def run_trial_generation(
    dataset_root,
    master_seed: Optional[int] = None,
    overwrite: bool = False,
    trajectory_ids: Optional[Sequence[str]] = None,
    pool_scale_override: Optional[int] = None,
    model_context: Optional[ModelContext] = None,
    dry_run: bool = False,
    progress: bool = False,
    rebuild_catalog: bool = True,
    difficulty_override: Optional[TrialDifficultyPolicy] = None,
) -> TrialGenerationResult:
    """Generate Dataset v2's trials: 3 per trajectory (easy/medium/hard) across all 210 trajectories
    = 630 (full/locked mode). ``trajectory_ids``/``pool_scale_override`` exist for tests/smoke runs
    only -- passing either marks the run as not ``full_locked_counts``.
    """
    paths = require_dataset_v2_root(dataset_root)
    trials_dir = paths.trials_dir
    protected_dir = trials_dir / PROTECTED_DIR_NAME

    full_locked_counts = trajectory_ids is None and pool_scale_override is None

    # Ensure the combined catalog exists (build it if missing; never overwrite silently otherwise).
    combined_manifest = paths.trajectories_dir / "combined_trajectory_manifest.csv"
    if rebuild_catalog or not combined_manifest.is_file():
        build_combined_catalog(paths.root, overwrite=True, full_counts=full_locked_counts)
    catalog = load_combined_catalog(paths.root)

    if trajectory_ids is not None:
        wanted = set(trajectory_ids)
        catalog = [r for r in catalog if r["trajectory_id"] in wanted]
        missing = wanted - {r["trajectory_id"] for r in catalog}
        if missing:
            raise ValueError(f"requested trajectory_ids not found in combined catalog: {sorted(missing)}")
    catalog.sort(key=lambda r: r["trajectory_id"])

    # Output existence / overwrite check.
    output_paths = [
        trials_dir / TRIAL_MANIFEST_NAME,
        trials_dir / GENERATION_REPORT_NAME,
        trials_dir / DISTRIBUTION_REPORT_NAME,
        trials_dir / ANTI_LEAKAGE_REPORT_NAME,
    ]
    for split in SPLITS:
        output_paths.append(trials_dir / f"{split}.npz")
        output_paths.append(protected_dir / f"{split}_evidence.npz")
    existing = [p for p in output_paths if p.is_file()]
    if existing and not overwrite:
        existing_relative = ", ".join(str(p.relative_to(paths.root)) for p in existing[:10])
        raise FileExistsError(
            f"Trial v2 output already exists ({existing_relative}); pass overwrite=True "
            "(--overwrite on the CLI) to regenerate it."
        )

    seed_policy = load_json_config(paths.configs_dir / "seed_policy.json")
    resolved_master_seed = int(master_seed if master_seed is not None else seed_policy["master_seed"])
    trials_component_tag = int(seed_policy["component_tags"]["trials"])
    component_seed = derive_seed(resolved_master_seed, trials_component_tag)

    settings = load_trial_generation_settings(paths, pool_scale_override)
    if difficulty_override is not None:
        # Test/smoke-only injection of a difficulty policy derived from the fixture's own candidate
        # distribution, so band population does not depend on the production-locked thresholds.
        settings = TrialGenerationSettings(
            interior_samples=settings.interior_samples,
            near_limit_samples=settings.near_limit_samples,
            singular_samples=settings.singular_samples,
            stratified_samples=settings.stratified_samples,
            interior_margin_fraction=settings.interior_margin_fraction,
            stratified_margin_fraction=settings.stratified_margin_fraction,
            difficulty=difficulty_override,
            frozen_core_revision=settings.frozen_core_revision,
            frozen_challenge_revision=settings.frozen_challenge_revision,
            frozen_trial_revision=settings.frozen_trial_revision,
        )
    diff = settings.difficulty

    if dry_run:
        return TrialGenerationResult(
            dataset_root=paths.root,
            trials_dir=trials_dir,
            dry_run=True,
            total_trials=len(catalog) * 3,
            full_locked_counts=full_locked_counts,
            split_counts={s: sum(3 for r in catalog if r["split"] == s) for s in SPLITS},
            difficulty_counts={c: len(catalog) for c in TRIAL_INIT_CLASSES},
            family_counts={
                "core": sum(3 for r in catalog if r["family"] == "core"),
                "random_challenge": sum(3 for r in catalog if r["family"] == "random_challenge"),
            },
        )

    model_context = model_context if model_context is not None else load_model_context()
    model_fingerprint = sha256_file(V1_MODEL_PATH)
    config_fingerprint = _config_fingerprint(paths)
    span = model_context.operational_upper_rad - model_context.operational_lower_rad

    public_records: List[dict] = []
    protected_records: List[dict] = []
    separation_records: List[dict] = []

    import sys as _sys
    import time as _time

    run_started = _time.perf_counter()
    total = len(catalog)

    for tindex, row in enumerate(catalog, start=1):
        trajectory_id = row["trajectory_id"]
        family = row["family"]
        split = row["split"]
        content_hash = row["content_hash"]

        public = load_public_trajectory(paths.root, trajectory_id, catalog_row=row)
        protected = load_protected_trajectory(paths.root, trajectory_id, catalog_row=row)
        first_target_position = public.first_target_position.astype(np.float64)
        first_target_quaternion = public.first_target_quaternion_wxyz.astype(np.float64)
        q_reference_start = protected.q_reference_start.astype(np.float64)

        # Confirm the first target is reachable via the PROTECTED reference start (diagnostic only).
        ref_pos_err, ref_ori_err = fk_reconstruction_error(
            model_context, q_reference_start, first_target_position, quaternion_wxyz_to_matrix(first_target_quaternion)
        )
        if ref_pos_err > REFERENCE_START_POSITION_GATE_M or ref_ori_err > REFERENCE_START_ORIENTATION_GATE_RAD:
            raise RuntimeError(
                f"{trajectory_id}: FK(q_reference_start) does not reproduce the first target pose "
                f"(pos {ref_pos_err:.3e} m, ori {np.rad2deg(ref_ori_err):.3e} deg); trajectory data is inconsistent."
            )

        pool_seed = trial_pool_seed(
            component_seed, content_hash, family, split,
            settings.frozen_core_revision, settings.frozen_challenge_revision, settings.frozen_trial_revision,
        )
        q_batch, source_labels = build_candidate_pool(
            model_context, pool_seed,
            settings.interior_samples, settings.near_limit_samples, settings.singular_samples, settings.stratified_samples,
            settings.interior_margin_fraction, settings.stratified_margin_fraction,
        )
        metrics = compute_candidate_covariates(model_context, q_batch)
        pos_err, ori_err = pose_errors(metrics["position"], metrics["quaternion"], first_target_position, first_target_quaternion)
        primary = primary_metric(pos_err, ori_err, diff.position_scale_m, diff.orientation_scale_rad, diff.position_weight, diff.orientation_weight)

        masks = classify_primary(primary, diff)
        chosen: Dict[str, int] = {}
        for init_class in TRIAL_INIT_CLASSES:
            mask = masks[init_class]
            if not np.any(mask):
                band_counts = {c: int(np.sum(masks[c])) for c in TRIAL_INIT_CLASSES}
                raise RuntimeError(
                    f"{trajectory_id}: difficulty band '{init_class}' has 0 candidates of "
                    f"{q_batch.shape[0]} (band_counts={band_counts}). Phase 7 incomplete for this "
                    "trajectory: never widen the threshold, duplicate, use q_reference, or swap the trajectory."
                )
            chosen[init_class] = _select_representative(primary, mask)

        # No duplicate q_initial / content-hash within a trajectory (disjoint bands guarantee this,
        # but assert it explicitly and fail loudly rather than dedupe).
        chosen_idx = [chosen[c] for c in TRIAL_INIT_CLASSES]
        if len({tuple(np.round(q_batch[i], 12)) for i in chosen_idx}) != 3:
            raise RuntimeError(f"{trajectory_id}: selected q_initial values are not all distinct")

        # Monotonicity check (primary metric): easy < medium < hard, with configured separation.
        p_easy = float(primary[chosen["easy"]])
        p_medium = float(primary[chosen["medium"]])
        p_hard = float(primary[chosen["hard"]])
        if not (p_easy < p_medium < p_hard):
            raise RuntimeError(f"{trajectory_id}: primary-metric monotonicity violated ({p_easy}, {p_medium}, {p_hard})")
        separation_records.append(
            {
                "trajectory_id": trajectory_id,
                "easy_primary": p_easy,
                "medium_primary": p_medium,
                "hard_primary": p_hard,
                "easy_to_medium": p_medium - p_easy,
                "medium_to_hard": p_hard - p_medium,
                "position_error_ordering_ok": bool(pos_err[chosen["easy"]] <= pos_err[chosen["hard"]]),
                "orientation_error_ordering_ok": bool(ori_err[chosen["easy"]] <= ori_err[chosen["hard"]]),
            }
        )

        for init_class in TRIAL_INIT_CLASSES:
            i = chosen[init_class]
            q_initial = q_batch[i].astype(np.float64)
            trial_id = f"{trajectory_id}_trial_{init_class}"
            src_seed = trial_source_seed(pool_seed, int(TRIAL_INIT_CLASS_TAGS[init_class]))

            trial_content_hash = content_hash_of_record(
                {
                    "trajectory_content_hash": content_hash,
                    "difficulty": init_class,
                    "q_initial": [round(float(v), 12) for v in q_initial],
                    "model_fingerprint": model_fingerprint,
                    "config_fingerprint": config_fingerprint,
                }
            )
            protected_reference_hash = content_hash_of_record(
                {"q_reference_start": [round(float(v), 12) for v in q_reference_start], "trajectory_content_hash": content_hash}
            )

            public_records.append(
                {
                    "trial_id": trial_id,
                    "trajectory_id": trajectory_id,
                    "trajectory_family": family,
                    "split": split,
                    "difficulty": init_class,
                    "difficulty_id": INIT_CLASS_ID[init_class],
                    "q_initial": q_initial,
                    "initial_position": metrics["position"][i].astype(np.float64),
                    "initial_quaternion_wxyz": metrics["quaternion"][i].astype(np.float64),
                    "first_target_position": first_target_position,
                    "first_target_quaternion_wxyz": first_target_quaternion,
                    "initial_position_error_m": float(pos_err[i]),
                    "initial_orientation_error_rad": float(ori_err[i]),
                    "primary_difficulty_metric": float(primary[i]),
                    "initial_sigma_min": float(metrics["sigma_min"][i]),
                    "initial_sigma_max": float(metrics["sigma_max"][i]),
                    "initial_condition_number": float(metrics["condition_number"][i]),
                    "minimum_initial_limit_margin_normalized": float(metrics["normalized_margin"][i]),
                    "minimum_initial_limit_margin_absolute_rad": float(metrics["absolute_margin_rad"][i]),
                    "controlling_joint_index": int(metrics["controlling_joint"][i]),
                    "candidate_source_pool": source_labels[i],
                    "source_seed": int(src_seed),
                    "trajectory_content_hash": content_hash,
                    "model_fingerprint": model_fingerprint,
                    "config_fingerprint": config_fingerprint,
                    "content_hash": trial_content_hash,
                }
            )
            protected_records.append(
                {
                    "trial_id": trial_id,
                    "trajectory_id": trajectory_id,
                    "split": split,
                    "difficulty": init_class,
                    "q_reference_start": q_reference_start,
                    "reference_fk_position_error_m": float(ref_pos_err),
                    "reference_fk_orientation_error_rad": float(ref_ori_err),
                    "protected_reference_content_hash": protected_reference_hash,
                    "normalized_joint_distance_to_reference_start": _normalized_joint_distance(q_initial, q_reference_start, span),
                    "absolute_joint_distance_to_reference_start_rad": float(np.linalg.norm(q_initial - q_reference_start)),
                }
            )

        if progress and (tindex % 20 == 0 or tindex == total):
            print(
                f"[trial-generate] {tindex}/{total} trajectories [{(_time.perf_counter() - run_started) / 60:.1f}min]",
                flush=True,
                file=_sys.stderr,
            )

    _assert_and_write(
        paths,
        public_records,
        protected_records,
        separation_records,
        settings,
        resolved_master_seed,
        trials_component_tag,
        component_seed,
        model_fingerprint,
        config_fingerprint,
        full_locked_counts,
        overwrite,
    )

    split_counts = {s: 0 for s in SPLITS}
    difficulty_counts = {c: 0 for c in TRIAL_INIT_CLASSES}
    family_counts = {"core": 0, "random_challenge": 0}
    split_difficulty_counts = {s: {c: 0 for c in TRIAL_INIT_CLASSES} for s in SPLITS}
    for rec in public_records:
        split_counts[rec["split"]] += 1
        difficulty_counts[rec["difficulty"]] += 1
        family_counts[rec["trajectory_family"]] += 1
        split_difficulty_counts[rec["split"]][rec["difficulty"]] += 1

    min_sep = min((r["easy_to_medium"] for r in separation_records), default=0.0)
    min_sep = min(min_sep, min((r["medium_to_hard"] for r in separation_records), default=0.0))

    return TrialGenerationResult(
        dataset_root=paths.root,
        trials_dir=trials_dir,
        dry_run=False,
        total_trials=len(public_records),
        full_locked_counts=full_locked_counts,
        split_counts=split_counts,
        difficulty_counts=difficulty_counts,
        family_counts=family_counts,
        split_difficulty_counts=split_difficulty_counts,
        separation_statistics={
            "minimum_observed_separation": float(min_sep),
            "configured_minimum_inter_level_separation": diff.minimum_inter_level_separation,
        },
    )


def _stack_public(records: List[dict], split: str) -> Dict[str, np.ndarray]:
    subset = [r for r in records if r["split"] == split]
    subset.sort(key=lambda r: r["trial_id"])
    return {
        "trial_id": np.array([r["trial_id"] for r in subset]),
        "trajectory_id": np.array([r["trajectory_id"] for r in subset]),
        "trajectory_family": np.array([r["trajectory_family"] for r in subset]),
        "difficulty": np.array([r["difficulty"] for r in subset]),
        "difficulty_id": np.array([r["difficulty_id"] for r in subset], dtype=np.int64),
        "q_initial": np.array([r["q_initial"] for r in subset], dtype=np.float64),
        "initial_position": np.array([r["initial_position"] for r in subset], dtype=np.float64),
        "initial_quaternion_wxyz": np.array([r["initial_quaternion_wxyz"] for r in subset], dtype=np.float64),
        "first_target_position": np.array([r["first_target_position"] for r in subset], dtype=np.float64),
        "first_target_quaternion_wxyz": np.array([r["first_target_quaternion_wxyz"] for r in subset], dtype=np.float64),
        "initial_position_error_m": np.array([r["initial_position_error_m"] for r in subset], dtype=np.float64),
        "initial_orientation_error_rad": np.array([r["initial_orientation_error_rad"] for r in subset], dtype=np.float64),
        "primary_difficulty_metric": np.array([r["primary_difficulty_metric"] for r in subset], dtype=np.float64),
        "initial_sigma_min": np.array([r["initial_sigma_min"] for r in subset], dtype=np.float64),
        "initial_sigma_max": np.array([r["initial_sigma_max"] for r in subset], dtype=np.float64),
        "initial_condition_number": np.array([r["initial_condition_number"] for r in subset], dtype=np.float64),
        "minimum_initial_limit_margin_normalized": np.array([r["minimum_initial_limit_margin_normalized"] for r in subset], dtype=np.float64),
        "minimum_initial_limit_margin_absolute_rad": np.array([r["minimum_initial_limit_margin_absolute_rad"] for r in subset], dtype=np.float64),
        "controlling_joint_index": np.array([r["controlling_joint_index"] for r in subset], dtype=np.int64),
        "source_seed": np.array([r["source_seed"] for r in subset], dtype=np.int64),
        "content_hash": np.array([r["content_hash"] for r in subset]),
    }


def _stack_protected(records: List[dict], split: str) -> Dict[str, np.ndarray]:
    subset = [r for r in records if r["split"] == split]
    subset.sort(key=lambda r: r["trial_id"])
    return {
        "trial_id": np.array([r["trial_id"] for r in subset]),
        "trajectory_id": np.array([r["trajectory_id"] for r in subset]),
        "difficulty": np.array([r["difficulty"] for r in subset]),
        "q_reference_start": np.array([r["q_reference_start"] for r in subset], dtype=np.float64),
        "reference_fk_position_error_m": np.array([r["reference_fk_position_error_m"] for r in subset], dtype=np.float64),
        "reference_fk_orientation_error_rad": np.array([r["reference_fk_orientation_error_rad"] for r in subset], dtype=np.float64),
        "protected_reference_content_hash": np.array([r["protected_reference_content_hash"] for r in subset]),
        "normalized_joint_distance_to_reference_start": np.array(
            [r["normalized_joint_distance_to_reference_start"] for r in subset], dtype=np.float64
        ),
        "absolute_joint_distance_to_reference_start_rad": np.array(
            [r["absolute_joint_distance_to_reference_start_rad"] for r in subset], dtype=np.float64
        ),
    }


def _manifest_row(rec: dict) -> list:
    return [
        rec["trial_id"],
        rec["trajectory_id"],
        rec["trajectory_family"],
        rec["split"],
        rec["difficulty"],
        json.dumps([round(float(v), 12) for v in rec["q_initial"]]),
        json.dumps([round(float(v), 10) for v in rec["initial_position"]]),
        json.dumps([round(float(v), 10) for v in rec["initial_quaternion_wxyz"]]),
        json.dumps([round(float(v), 10) for v in rec["first_target_position"]]),
        json.dumps([round(float(v), 10) for v in rec["first_target_quaternion_wxyz"]]),
        f"{rec['initial_position_error_m']:.12g}",
        f"{rec['initial_orientation_error_rad']:.12g}",
        f"{rec['primary_difficulty_metric']:.12g}",
        f"{rec['initial_sigma_min']:.12g}",
        f"{rec['initial_sigma_max']:.12g}",
        f"{rec['initial_condition_number']:.12g}",
        f"{rec['minimum_initial_limit_margin_normalized']:.12g}",
        f"{rec['minimum_initial_limit_margin_absolute_rad']:.12g}",
        int(rec["controlling_joint_index"]),
        rec["candidate_source_pool"],
        int(rec["source_seed"]),
        rec["trajectory_content_hash"],
        rec["model_fingerprint"],
        rec["config_fingerprint"],
        rec["content_hash"],
    ]


def _anti_leakage(public_records: List[dict]) -> dict:
    by_split: Dict[str, List[dict]] = {s: [] for s in SPLITS}
    for r in public_records:
        by_split[r["split"]].append(r)

    collisions: List[dict] = []
    for i in range(len(SPLITS)):
        for j in range(i + 1, len(SPLITS)):
            a, b = SPLITS[i], SPLITS[j]
            for dim in ("trial_id", "content_hash", "source_seed", "trajectory_id", "trajectory_content_hash"):
                sa = {r[dim] for r in by_split[a]}
                sb = {r[dim] for r in by_split[b]}
                overlap = sa & sb
                if overlap:
                    collisions.append({"splits": [a, b], "dimension": dim, "overlap": sorted(str(x) for x in overlap)[:5]})
            # (trajectory_id, difficulty) tuples
            ta = {(r["trajectory_id"], r["difficulty"]) for r in by_split[a]}
            tb = {(r["trajectory_id"], r["difficulty"]) for r in by_split[b]}
            if ta & tb:
                collisions.append({"splits": [a, b], "dimension": "(trajectory_id,difficulty)", "overlap": sorted(str(x) for x in (ta & tb))[:5]})
            # q_initial exact-byte + near-duplicate
            qa = [(r["trial_id"], np.asarray(r["q_initial"], dtype=np.float64)) for r in by_split[a]]
            qb = [(r["trial_id"], np.asarray(r["q_initial"], dtype=np.float64)) for r in by_split[b]]
            exact_a = {tuple(np.round(q, 12)) for _t, q in qa}
            exact_b = {tuple(np.round(q, 12)) for _t, q in qb}
            if exact_a & exact_b:
                collisions.append({"splits": [a, b], "dimension": "q_initial_exact_bytes", "overlap_count": len(exact_a & exact_b)})
            for ta_id, qav in qa:
                for tb_id, qbv in qb:
                    if float(np.max(np.abs(qav - qbv))) < Q_INITIAL_NEAR_DUPLICATE_RAD:
                        collisions.append({"splits": [a, b], "dimension": "q_initial_near_duplicate", "pair": [ta_id, tb_id]})

    return {
        "dimensions_checked": [
            "trial_id",
            "content_hash",
            "source_seed",
            "(trajectory_id,difficulty)",
            "q_initial_exact_bytes",
            "q_initial_near_duplicate",
            "trajectory_content_hash",
            "split_inheritance",
        ],
        "collisions_found": len(collisions),
        "collision_details": collisions,
        "pass": len(collisions) == 0,
    }


def _assert_and_write(
    paths,
    public_records,
    protected_records,
    separation_records,
    settings: TrialGenerationSettings,
    resolved_master_seed,
    trials_component_tag,
    component_seed,
    model_fingerprint,
    config_fingerprint,
    full_locked_counts,
    overwrite,
) -> None:
    trials_dir = paths.trials_dir
    protected_dir = trials_dir / PROTECTED_DIR_NAME

    ids = [r["trial_id"] for r in public_records]
    hashes = [r["content_hash"] for r in public_records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate trial_id across the generated trial set")
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate trial content_hash across the generated trial set")

    # Every trajectory contributes exactly one easy/medium/hard.
    by_traj: Dict[str, set] = {}
    for r in public_records:
        by_traj.setdefault(r["trajectory_id"], set()).add(r["difficulty"])
    for tid, classes in by_traj.items():
        if classes != set(TRIAL_INIT_CLASSES):
            raise ValueError(f"trajectory {tid} does not have exactly easy/medium/hard trials: {sorted(classes)}")

    anti_leakage = _anti_leakage(public_records)
    if not anti_leakage["pass"]:
        raise ValueError(f"trial anti-leakage failed: {anti_leakage['collision_details']}")

    # Write per-split public + protected NPZ (allow_pickle=False enforced by save_npz).
    for split in SPLITS:
        save_npz(trials_dir / f"{split}.npz", _stack_public(public_records, split), overwrite=overwrite)
        save_npz(protected_dir / f"{split}_evidence.npz", _stack_protected(protected_records, split), overwrite=overwrite)

    # Public manifest.
    manifest_rows = [_manifest_row(r) for r in sorted(public_records, key=lambda r: r["trial_id"])]
    manifest_path = _atomic_write_csv(trials_dir / TRIAL_MANIFEST_NAME, TRIAL_MANIFEST_COLUMNS, manifest_rows)

    # Counts.
    split_counts = {s: 0 for s in SPLITS}
    difficulty_counts = {c: 0 for c in TRIAL_INIT_CLASSES}
    family_counts = {"core": 0, "random_challenge": 0}
    split_difficulty_counts = {s: {c: 0 for c in TRIAL_INIT_CLASSES} for s in SPLITS}
    for r in public_records:
        split_counts[r["split"]] += 1
        difficulty_counts[r["difficulty"]] += 1
        family_counts[r["trajectory_family"]] += 1
        split_difficulty_counts[r["split"]][r["difficulty"]] += 1

    diff = settings.difficulty
    distribution = {
        "total_trials": len(public_records),
        "split_counts": split_counts,
        "difficulty_counts": difficulty_counts,
        "family_counts": family_counts,
        "split_difficulty_counts": split_difficulty_counts,
        "primary_metric": diff.primary_metric,
        "separation": {
            "configured_minimum_inter_level_separation": diff.minimum_inter_level_separation,
            "observed_min_easy_to_medium": min((r["easy_to_medium"] for r in separation_records), default=0.0),
            "observed_min_medium_to_hard": min((r["medium_to_hard"] for r in separation_records), default=0.0),
            "position_error_ordering_ok_count": sum(1 for r in separation_records if r["position_error_ordering_ok"]),
            "orientation_error_ordering_ok_count": sum(1 for r in separation_records if r["orientation_error_ordering_ok"]),
            "trajectories": len(separation_records),
        },
        "difficulty_bands": {
            "position_scale_m": diff.position_scale_m,
            "orientation_scale_rad": diff.orientation_scale_rad,
            "easy_upper": diff.easy_upper,
            "medium_lower": diff.medium_lower,
            "medium_upper": diff.medium_upper,
            "hard_lower": diff.hard_lower,
        },
    }
    _atomic_write_json(trials_dir / DISTRIBUTION_REPORT_NAME, distribution)
    _atomic_write_json(trials_dir / ANTI_LEAKAGE_REPORT_NAME, anti_leakage)

    report = {
        "dataset_version": DATASET_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "git_commit": _git_commit(REPO_ROOT),
        "model_fingerprint": model_fingerprint,
        "config_fingerprint": config_fingerprint,
        "master_seed": resolved_master_seed,
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "generation_status": "development",
        "seed_derivation": {
            "trials_component_tag": trials_component_tag,
            "trials_component_seed": component_seed,
            "frozen_core_seed_revision": settings.frozen_core_revision,
            "frozen_challenge_seed_revision": settings.frozen_challenge_revision,
            "frozen_trial_seed_revision": settings.frozen_trial_revision,
        },
        "full_locked_counts": full_locked_counts,
        "total_trials": len(public_records),
        "split_counts": split_counts,
        "difficulty_counts": difficulty_counts,
        "family_counts": family_counts,
        "split_difficulty_counts": split_difficulty_counts,
        "anti_leakage_report": anti_leakage,
        "output_files": {"manifest": {"filename": relative_to_dataset_v2_root(manifest_path, paths.root), "sha256": sha256_file(manifest_path)}},
    }
    _atomic_write_json(trials_dir / GENERATION_REPORT_NAME, report)

    # Manifest count update + checksum manifest rebuild.
    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    manifest = apply_trial_generation_status(
        manifest,
        total_trials=len(public_records),
        split_counts=split_counts,
        difficulty_counts=difficulty_counts,
        family_counts=family_counts,
        split_difficulty_counts=split_difficulty_counts,
        full_locked_counts=full_locked_counts,
    )
    _atomic_write_json(paths.manifest_file, manifest)

    checksum_manifest = build_checksum_manifest(paths.root)
    _atomic_write_json(paths.checksum_manifest_file, checksum_manifest)
