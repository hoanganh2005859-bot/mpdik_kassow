"""Dataset v2 trial difficulty-threshold calibration (Phase 7, task section 9).

Calibrates the easy/medium/hard bands for the LOCKED primary difficulty metric -- a combined
NORMALIZED first-target pose error (position + orientation), never normalized joint distance alone
(the KR810 is redundant, so many joint configurations map to nearly the same pose). Uses ONLY the
development trajectories; never validation, never frozen_test (spec section J / frozen-test
protocol). The numbers this module produces are recorded in
``docs/V2_TRIAL_DIFFICULTY_CALIBRATION.md`` and baked into
``dataset_v2/config_templates.py`` as the [LOCKED] ``TRIAL_*`` constants, exactly as Phase 2.5's
threshold calibration was.

Procedure:
1. For every development trajectory, build the same deterministic q_initial candidate pool the trial
   generator uses (``dataset_v2/trial_candidates.py``), draw ONLY from joint limits.
2. Compute FK covariates and the raw position/orientation error of each candidate to the
   trajectory's FIRST canonical target pose (public geometry only; q_reference never read).
3. Pool all development candidates. The calibration scales are the pooled medians of the raw
   position and orientation errors (so neither unit dominates the combined metric).
4. The bands are non-overlapping percentile bands of the combined metric with guard gaps
   (easy <= P30, medium in [P40, P60], hard >= P70), guaranteeing easy < medium < hard by
   construction and a positive minimum inter-level separation.

Never runs DLS; never uses global ``numpy.random`` state.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from dataset_v2.locator import require_dataset_v2_root
from dataset_v2.seeds import derive_seed
from dataset_v2.trajectory_catalog import load_combined_catalog
from dataset_v2.trajectory_loading import load_public_trajectory
from dataset_v2.trial_candidates import (
    build_candidate_pool,
    compute_candidate_covariates,
    pose_errors,
    primary_metric,
    trial_pool_seed,
)
from kinematics.model_loader import ModelContext, load_model_context
from utils.config_loader import load_json_config

# Percentile band edges (guard gaps between them make the three bands non-overlapping).
EASY_UPPER_PERCENTILE = 30.0
MEDIUM_LOWER_PERCENTILE = 40.0
MEDIUM_UPPER_PERCENTILE = 60.0
HARD_LOWER_PERCENTILE = 70.0


@dataclass
class TrialCalibrationResult:
    position_scale_m: float
    orientation_scale_rad: float
    easy_upper: float
    medium_lower: float
    medium_upper: float
    hard_lower: float
    minimum_inter_level_separation: float
    position_weight: float
    orientation_weight: float
    development_trajectories: int
    total_candidates: int
    per_band_candidate_counts: Dict[str, int] = field(default_factory=dict)
    raw_position_error_stats: Dict[str, float] = field(default_factory=dict)
    raw_orientation_error_stats: Dict[str, float] = field(default_factory=dict)
    primary_metric_stats: Dict[str, float] = field(default_factory=dict)
    per_trajectory_band_min_counts: Dict[str, int] = field(default_factory=dict)


def _pool_sizes(trial_config: dict, pool_scale: Optional[int]) -> Dict[str, int]:
    policy = trial_config["candidate_pool_policy"]
    if pool_scale is None:
        return {
            "interior": int(policy["interior_samples"]),
            "near_limit": int(policy["near_limit_samples"]),
            "singular": int(policy["singular_samples"]),
            "stratified": int(policy["stratified_samples"]),
        }
    # test/smoke override: uniform small pool
    return {"interior": pool_scale, "near_limit": pool_scale, "singular": pool_scale, "stratified": pool_scale}


def calibrate(
    dataset_root,
    master_seed: Optional[int] = None,
    model_context: Optional[ModelContext] = None,
    pool_scale_override: Optional[int] = None,
) -> TrialCalibrationResult:
    """Run the calibration against the development trajectories under ``dataset_root``.

    ``pool_scale_override`` (tests only) replaces every sub-pool size with a small uniform value.
    """
    paths = require_dataset_v2_root(dataset_root)
    model_context = model_context if model_context is not None else load_model_context()

    trial_config = load_json_config(paths.configs_dir / "trial_config.json")
    seed_policy = load_json_config(paths.configs_dir / "seed_policy.json")
    resolved_master_seed = int(master_seed if master_seed is not None else seed_policy["master_seed"])
    component_seed = derive_seed(resolved_master_seed, int(seed_policy["component_tags"]["trials"]))

    difficulty = trial_config["difficulty"]
    w_pos = float(difficulty["position_weight"])
    w_ori = float(difficulty["orientation_weight"])

    pool_sizes = _pool_sizes(trial_config, pool_scale_override)
    policy = trial_config["candidate_pool_policy"]
    interior_margin = float(policy["interior_margin_fraction"])
    stratified_margin = float(policy["stratified_margin_fraction"])

    frozen_core = int(seed_policy["frozen_core_seed_revision"])
    frozen_challenge = int(seed_policy["frozen_challenge_seed_revision"])
    frozen_trial = int(seed_policy["frozen_trial_seed_revision"])

    catalog = load_combined_catalog(paths.root)
    dev_rows = [r for r in catalog if r["split"] == "development"]
    if not dev_rows:
        raise RuntimeError("no development trajectories found in the combined catalog; cannot calibrate")

    all_pos_err: List[np.ndarray] = []
    all_ori_err: List[np.ndarray] = []
    per_trajectory: List[Dict[str, np.ndarray]] = []

    for row in dev_rows:
        public = load_public_trajectory(paths.root, row["trajectory_id"], catalog_row=row)
        pool_seed = trial_pool_seed(
            component_seed, row["content_hash"], row["family"], row["split"], frozen_core, frozen_challenge, frozen_trial
        )
        q_batch, _labels = build_candidate_pool(
            model_context,
            pool_seed,
            pool_sizes["interior"],
            pool_sizes["near_limit"],
            pool_sizes["singular"],
            pool_sizes["stratified"],
            interior_margin,
            stratified_margin,
        )
        metrics = compute_candidate_covariates(model_context, q_batch)
        pos_err, ori_err = pose_errors(
            metrics["position"], metrics["quaternion"], public.first_target_position, public.first_target_quaternion_wxyz
        )
        all_pos_err.append(pos_err)
        all_ori_err.append(ori_err)
        per_trajectory.append({"pos": pos_err, "ori": ori_err})

    pooled_pos = np.concatenate(all_pos_err)
    pooled_ori = np.concatenate(all_ori_err)

    pos_scale = float(np.median(pooled_pos))
    ori_scale = float(np.median(pooled_ori))
    if pos_scale <= 0.0 or ori_scale <= 0.0:
        raise RuntimeError(f"degenerate calibration scale (pos_scale={pos_scale}, ori_scale={ori_scale})")

    pooled_primary = primary_metric(pooled_pos, pooled_ori, pos_scale, ori_scale, w_pos, w_ori)

    easy_upper = float(np.percentile(pooled_primary, EASY_UPPER_PERCENTILE))
    medium_lower = float(np.percentile(pooled_primary, MEDIUM_LOWER_PERCENTILE))
    medium_upper = float(np.percentile(pooled_primary, MEDIUM_UPPER_PERCENTILE))
    hard_lower = float(np.percentile(pooled_primary, HARD_LOWER_PERCENTILE))
    minimum_separation = min(medium_lower - easy_upper, hard_lower - medium_upper)

    # Per-band pooled counts and per-trajectory minimum band populations (must be >= 1 everywhere
    # for a full run to succeed; reported here so an under-populated band surfaces at calibration).
    def _band_counts(primary: np.ndarray) -> Dict[str, int]:
        return {
            "easy": int(np.sum(primary <= easy_upper)),
            "medium": int(np.sum((primary >= medium_lower) & (primary <= medium_upper))),
            "hard": int(np.sum(primary >= hard_lower)),
        }

    per_band = _band_counts(pooled_primary)
    per_traj_min = {"easy": 10**9, "medium": 10**9, "hard": 10**9}
    for rec in per_trajectory:
        p = primary_metric(rec["pos"], rec["ori"], pos_scale, ori_scale, w_pos, w_ori)
        counts = _band_counts(p)
        for band in per_traj_min:
            per_traj_min[band] = min(per_traj_min[band], counts[band])

    return TrialCalibrationResult(
        position_scale_m=pos_scale,
        orientation_scale_rad=ori_scale,
        easy_upper=easy_upper,
        medium_lower=medium_lower,
        medium_upper=medium_upper,
        hard_lower=hard_lower,
        minimum_inter_level_separation=minimum_separation,
        position_weight=w_pos,
        orientation_weight=w_ori,
        development_trajectories=len(dev_rows),
        total_candidates=int(pooled_primary.shape[0]),
        per_band_candidate_counts=per_band,
        raw_position_error_stats={
            "min": float(np.min(pooled_pos)),
            "median": pos_scale,
            "p95": float(np.percentile(pooled_pos, 95)),
            "max": float(np.max(pooled_pos)),
        },
        raw_orientation_error_stats={
            "min": float(np.min(pooled_ori)),
            "median": ori_scale,
            "p95": float(np.percentile(pooled_ori, 95)),
            "max": float(np.max(pooled_ori)),
        },
        primary_metric_stats={
            "min": float(np.min(pooled_primary)),
            "p30": easy_upper,
            "p40": medium_lower,
            "p60": medium_upper,
            "p70": hard_lower,
            "max": float(np.max(pooled_primary)),
        },
        per_trajectory_band_min_counts=per_traj_min,
    )


def calibration_report(result: TrialCalibrationResult, master_seed: int) -> dict:
    return {
        "calibration": "trial_difficulty_thresholds",
        "master_seed": int(master_seed),
        "primary_metric": "combined_normalized_first_target_pose_error",
        "split_used": "development_only",
        "position_weight": result.position_weight,
        "orientation_weight": result.orientation_weight,
        "position_scale_m": result.position_scale_m,
        "orientation_scale_rad": result.orientation_scale_rad,
        "bands": {
            "easy_upper": result.easy_upper,
            "medium_lower": result.medium_lower,
            "medium_upper": result.medium_upper,
            "hard_lower": result.hard_lower,
        },
        "minimum_inter_level_separation": result.minimum_inter_level_separation,
        "development_trajectories": result.development_trajectories,
        "total_candidates": result.total_candidates,
        "pooled_band_candidate_counts": result.per_band_candidate_counts,
        "per_trajectory_band_min_counts": result.per_trajectory_band_min_counts,
        "raw_position_error_stats": result.raw_position_error_stats,
        "raw_orientation_error_stats": result.raw_orientation_error_stats,
        "primary_metric_stats": result.primary_metric_stats,
    }


def apply_calibration_to_config(dataset_root, result: TrialCalibrationResult) -> Path:
    """Write a calibration result into an on-disk ``configs/trial_config.json`` difficulty block and
    return the config path.

    Used by tests to lock fixture-derived bands so the generator and validator both read the same
    numbers from the config (the production lock step bakes the numbers into
    ``config_templates.py`` instead, so a fresh scaffold already carries them). Does not rebuild the
    checksum manifest.
    """
    paths = require_dataset_v2_root(dataset_root)
    config_path = paths.configs_dir / "trial_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    difficulty = config["difficulty"]
    difficulty["position_scale_m"] = result.position_scale_m
    difficulty["orientation_scale_rad"] = result.orientation_scale_rad
    difficulty["bands"] = {
        "easy": {"primary_max_inclusive": result.easy_upper},
        "medium": {"primary_min_inclusive": result.medium_lower, "primary_max_inclusive": result.medium_upper},
        "hard": {"primary_min_inclusive": result.hard_lower},
    }
    difficulty["minimum_inter_level_separation"] = result.minimum_inter_level_separation
    config["difficulty"] = difficulty
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def write_calibration_report(result: TrialCalibrationResult, master_seed: int, report_path) -> Path:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration_report(result, master_seed), indent=2) + "\n", encoding="utf-8")
    return path
