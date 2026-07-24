"""Independent validator for Dataset v2 trials (Phase 7).

Reuses ``kinematics/`` unchanged for every recomputation and NEVER runs DLS to decide validity. It
recomputes, from the on-disk public trial NPZ/manifest only: FK(q_initial), the first-target pose
error covariates, conditioning covariates, joint-limit margins, controlling joint, the primary
difficulty metric, the difficulty classification, and per-trajectory monotonicity. It independently
re-checks counts, integrity, uniqueness, split inheritance, anti-leakage, and -- crucially -- the
public/protected isolation:

* the public trial NPZ carries no ``q_reference`` / protected field;
* every ``q_initial`` differs from the trajectory's protected ``q_reference_start`` and from every
  future ``q_reference`` waypoint (so no reference solution leaked into the initial state);
* the protected reference content hashes recompute.

Never uses global ``numpy.random`` state.
"""

import csv
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from dataset_v2.config_templates import TRIAL_INIT_CLASSES
from dataset_v2.locator import require_dataset_v2_root
from dataset_v2.trajectory_catalog import load_combined_catalog
from dataset_v2.trajectory_loading import PROTECTED_ARRAY_KEYS, load_protected_trajectory, load_public_trajectory
from dataset_v2.trial_candidates import compute_candidate_covariates, pose_errors, primary_metric
from dataset_v2.trial_generation import (
    ANTI_LEAKAGE_REPORT_NAME,
    PROTECTED_DIR_NAME,
    TRIAL_MANIFEST_NAME,
    load_trial_generation_settings,
)
from kinematics.model_loader import ModelContext, load_model_context
from utils.npz_utils import load_npz

SPLITS = ("development", "validation", "frozen_test")
EXPECTED_TOTAL = 630
EXPECTED_PER_SPLIT = 210
EXPECTED_PER_DIFFICULTY = 210
EXPECTED_PER_DIFFICULTY_PER_SPLIT = 70
EXPECTED_CORE = 360
EXPECTED_CHALLENGE = 270

QUATERNION_NORM_TOL = 1e-6
COVARIATE_TOL = 1e-6
POSE_ERROR_TOL = 1e-6
PRIMARY_TOL = 1e-9
Q_REFERENCE_ISOLATION_RAD = 1e-6


@dataclass
class TrialValidationReport:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    total_trials: int = 0
    split_counts: Dict[str, int] = field(default_factory=dict)
    difficulty_counts: Dict[str, int] = field(default_factory=dict)
    family_counts: Dict[str, int] = field(default_factory=dict)
    split_difficulty_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    statistics: dict = field(default_factory=dict)


def _read_manifest_rows(path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_trials(
    dataset_root,
    model_context: Optional[ModelContext] = None,
    full_counts: bool = True,
) -> TrialValidationReport:
    paths = require_dataset_v2_root(dataset_root)
    model_context = model_context if model_context is not None else load_model_context()
    reasons: List[str] = []

    trials_dir = paths.trials_dir
    protected_dir = trials_dir / PROTECTED_DIR_NAME
    manifest_path = trials_dir / TRIAL_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"trial manifest not found: {manifest_path}")

    manifest_rows = _read_manifest_rows(manifest_path)
    settings = load_trial_generation_settings(paths)
    diff = settings.difficulty

    catalog = {r["trajectory_id"]: r for r in load_combined_catalog(paths.root)}
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad

    # ---- load per-split public + protected NPZ ------------------------------------------------
    public_by_split: Dict[str, dict] = {}
    protected_by_split: Dict[str, dict] = {}
    for split in SPLITS:
        pub_path = trials_dir / f"{split}.npz"
        prot_path = protected_dir / f"{split}_evidence.npz"
        if not pub_path.is_file():
            reasons.append(f"public trial NPZ missing for split '{split}' ({pub_path.name})")
            continue
        if not prot_path.is_file():
            reasons.append(f"protected trial NPZ missing for split '{split}' ({prot_path.name})")
            continue
        public_by_split[split] = load_npz(pub_path)
        protected_by_split[split] = load_npz(prot_path)
        # Public NPZ must never contain a protected array key.
        leaked = set(public_by_split[split].keys()) & PROTECTED_ARRAY_KEYS
        if leaked:
            reasons.append(f"public trial NPZ for split '{split}' leaks protected arrays: {sorted(leaked)}")

    if reasons:
        # A missing/corrupt NPZ makes the deeper checks meaningless.
        return TrialValidationReport(passed=False, reasons=reasons)

    # ---- counts (manifest) --------------------------------------------------------------------
    total = len(manifest_rows)
    split_counts = {s: 0 for s in SPLITS}
    difficulty_counts = {c: 0 for c in TRIAL_INIT_CLASSES}
    family_counts = {"core": 0, "random_challenge": 0}
    split_difficulty_counts = {s: {c: 0 for c in TRIAL_INIT_CLASSES} for s in SPLITS}
    by_trajectory: Dict[str, set] = {}
    for row in manifest_rows:
        if row["split"] in split_counts:
            split_counts[row["split"]] += 1
        if row["difficulty"] in difficulty_counts:
            difficulty_counts[row["difficulty"]] += 1
        if row["trajectory_family"] in family_counts:
            family_counts[row["trajectory_family"]] += 1
        if row["split"] in split_difficulty_counts and row["difficulty"] in split_difficulty_counts[row["split"]]:
            split_difficulty_counts[row["split"]][row["difficulty"]] += 1
        by_trajectory.setdefault(row["trajectory_id"], set()).add(row["difficulty"])

    if full_counts:
        if total != EXPECTED_TOTAL:
            reasons.append(f"expected {EXPECTED_TOTAL} trials, found {total}")
        for split in SPLITS:
            if split_counts[split] != EXPECTED_PER_SPLIT:
                reasons.append(f"split '{split}': expected {EXPECTED_PER_SPLIT} trials, found {split_counts[split]}")
        for c in TRIAL_INIT_CLASSES:
            if difficulty_counts[c] != EXPECTED_PER_DIFFICULTY:
                reasons.append(f"difficulty '{c}': expected {EXPECTED_PER_DIFFICULTY}, found {difficulty_counts[c]}")
        if family_counts["core"] != EXPECTED_CORE:
            reasons.append(f"core family: expected {EXPECTED_CORE} trials, found {family_counts['core']}")
        if family_counts["random_challenge"] != EXPECTED_CHALLENGE:
            reasons.append(f"random_challenge family: expected {EXPECTED_CHALLENGE} trials, found {family_counts['random_challenge']}")
        for split in SPLITS:
            for c in TRIAL_INIT_CLASSES:
                if split_difficulty_counts[split][c] != EXPECTED_PER_DIFFICULTY_PER_SPLIT:
                    reasons.append(
                        f"split '{split}' difficulty '{c}': expected {EXPECTED_PER_DIFFICULTY_PER_SPLIT}, "
                        f"found {split_difficulty_counts[split][c]}"
                    )

    # exactly easy/medium/hard per trajectory
    for tid, classes in by_trajectory.items():
        if classes != set(TRIAL_INIT_CLASSES):
            reasons.append(f"trajectory {tid} missing a difficulty: has {sorted(classes)}")

    # uniqueness of ids/hashes (manifest)
    ids = [r["trial_id"] for r in manifest_rows]
    hashes = [r["content_hash"] for r in manifest_rows]
    if len(set(ids)) != len(ids):
        reasons.append("duplicate trial_id in manifest")
    if len(set(hashes)) != len(hashes):
        reasons.append("duplicate trial content_hash in manifest")

    # ---- per-split integrity + recomputation --------------------------------------------------
    per_split_records: Dict[str, List[dict]] = {s: [] for s in SPLITS}
    max_position_error_disagreement = 0.0
    max_primary_disagreement = 0.0

    for split in SPLITS:
        pub = public_by_split[split]
        prot = protected_by_split[split]

        # dtype / object / finiteness
        for name, arr in pub.items():
            if arr.dtype == object:
                reasons.append(f"public split '{split}' array '{name}' has object dtype")
            if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
                reasons.append(f"public split '{split}' array '{name}' has non-finite values")

        q_initial = pub["q_initial"]
        n = q_initial.shape[0]
        if q_initial.shape[1] != model_context.nq:
            reasons.append(f"split '{split}': q_initial has {q_initial.shape[1]} joints, expected {model_context.nq}")
            continue

        # joint limits
        if np.any(q_initial < lower - 1e-9) or np.any(q_initial > upper + 1e-9):
            reasons.append(f"split '{split}': a q_initial violates operational joint limits")

        # quaternion normalization
        quats = pub["initial_quaternion_wxyz"]
        norms = np.linalg.norm(quats, axis=1)
        if np.any(np.abs(norms - 1.0) > QUATERNION_NORM_TOL):
            reasons.append(f"split '{split}': initial_quaternion_wxyz not unit-normalized")

        # recompute covariates
        metrics = compute_candidate_covariates(model_context, q_initial)
        # controlling joint / sigma / condition / margin agreement
        if np.any(metrics["controlling_joint"] != pub["controlling_joint_index"]):
            reasons.append(f"split '{split}': controlling_joint_index disagrees with recomputation")
        for key_stored, key_metric in (
            ("initial_sigma_min", "sigma_min"),
            ("initial_sigma_max", "sigma_max"),
            ("initial_condition_number", "condition_number"),
            ("minimum_initial_limit_margin_normalized", "normalized_margin"),
            ("minimum_initial_limit_margin_absolute_rad", "absolute_margin_rad"),
        ):
            rel = np.abs(pub[key_stored] - metrics[key_metric])
            denom = np.maximum(np.abs(metrics[key_metric]), 1.0)
            if np.any(rel / denom > 1e-6):
                reasons.append(f"split '{split}': {key_stored} disagrees with recomputation")

        # recompute pose errors + primary metric, verify difficulty classification
        for i in range(n):
            first_pos = pub["first_target_position"][i]
            first_quat = pub["first_target_quaternion_wxyz"][i]
            pos_err, ori_err = pose_errors(
                metrics["position"][i : i + 1], metrics["quaternion"][i : i + 1], first_pos, first_quat
            )
            pos_err = float(pos_err[0])
            ori_err = float(ori_err[0])
            max_position_error_disagreement = max(max_position_error_disagreement, abs(pos_err - float(pub["initial_position_error_m"][i])))
            prim = float(
                primary_metric(
                    np.array([pos_err]), np.array([ori_err]),
                    diff.position_scale_m, diff.orientation_scale_rad, diff.position_weight, diff.orientation_weight,
                )[0]
            )
            max_primary_disagreement = max(max_primary_disagreement, abs(prim - float(pub["primary_difficulty_metric"][i])))

            difficulty = str(pub["difficulty"][i])
            band_ok = {
                "easy": prim <= diff.easy_upper + PRIMARY_TOL,
                "medium": (prim >= diff.medium_lower - PRIMARY_TOL) and (prim <= diff.medium_upper + PRIMARY_TOL),
                "hard": prim >= diff.hard_lower - PRIMARY_TOL,
            }[difficulty]
            if not band_ok:
                reasons.append(f"split '{split}': trial {str(pub['trial_id'][i])} classified '{difficulty}' but primary {prim:.6f} is out of band")

            per_split_records[split].append(
                {
                    "trial_id": str(pub["trial_id"][i]),
                    "trajectory_id": str(pub["trajectory_id"][i]),
                    "difficulty": difficulty,
                    "primary": prim,
                    "q_initial": q_initial[i],
                    "content_hash": str(pub["content_hash"][i]),
                    "source_seed": int(pub["source_seed"][i]),
                }
            )

        # ---- isolation vs protected reference ---------------------------------------------------
        prot_trial_ids = list(prot["trial_id"])
        q_ref_start = prot["q_reference_start"]
        if prot_trial_ids != list(pub["trial_id"]):
            reasons.append(f"split '{split}': protected evidence trial_id order does not match public NPZ")
        else:
            dist_ref_start = np.max(np.abs(q_initial - q_ref_start), axis=1)
            if np.any(dist_ref_start < Q_REFERENCE_ISOLATION_RAD):
                reasons.append(f"split '{split}': a q_initial equals the protected q_reference_start")

    # ---- monotonicity per trajectory ----------------------------------------------------------
    prim_by_traj: Dict[str, Dict[str, float]] = {}
    for split in SPLITS:
        for rec in per_split_records[split]:
            prim_by_traj.setdefault(rec["trajectory_id"], {})[rec["difficulty"]] = rec["primary"]
    monotonicity_ok = 0
    for tid, m in prim_by_traj.items():
        if set(m.keys()) != set(TRIAL_INIT_CLASSES):
            continue
        if not (m["easy"] < m["medium"] < m["hard"]):
            reasons.append(f"trajectory {tid}: primary-metric monotonicity violated (easy<medium<hard)")
        else:
            monotonicity_ok += 1
        if (m["medium"] - m["easy"]) < diff.minimum_inter_level_separation - 1e-9 or (m["hard"] - m["medium"]) < diff.minimum_inter_level_separation - 1e-9:
            reasons.append(f"trajectory {tid}: inter-level separation below configured minimum")

    # ---- trajectory references + split inheritance --------------------------------------------
    for row in manifest_rows:
        tid = row["trajectory_id"]
        if tid not in catalog:
            reasons.append(f"trial references unknown trajectory_id '{tid}'")
            continue
        if catalog[tid]["split"] != row["split"]:
            reasons.append(f"trial split '{row['split']}' does not match trajectory split '{catalog[tid]['split']}' for {tid}")
        if catalog[tid]["family"] != row["trajectory_family"]:
            reasons.append(f"trial family '{row['trajectory_family']}' does not match trajectory family for {tid}")

    # ---- q_initial not drawn from any future q_reference waypoint ------------------------------
    # (load each trajectory's protected q_reference once; check all its trials' q_initial)
    trials_by_traj: Dict[str, List[np.ndarray]] = {}
    for split in SPLITS:
        for rec in per_split_records[split]:
            trials_by_traj.setdefault(rec["trajectory_id"], []).append(rec["q_initial"])
    for tid, q_inits in trials_by_traj.items():
        if tid not in catalog:
            continue
        protected = load_protected_trajectory(paths.root, tid, catalog_row=catalog[tid])
        q_ref = protected.q_reference  # [400, 7]
        for q in q_inits:
            min_linf = float(np.min(np.max(np.abs(q_ref - q[None, :]), axis=1)))
            if min_linf < Q_REFERENCE_ISOLATION_RAD:
                reasons.append(f"trajectory {tid}: a q_initial coincides with a q_reference waypoint (isolation breach)")
                break

    # ---- cross-split anti-leakage (independent recompute) -------------------------------------
    for i in range(len(SPLITS)):
        for j in range(i + 1, len(SPLITS)):
            a, b = SPLITS[i], SPLITS[j]
            for dim in ("trial_id", "content_hash", "source_seed"):
                sa = {r[dim] for r in per_split_records[a]}
                sb = {r[dim] for r in per_split_records[b]}
                if sa & sb:
                    reasons.append(f"cross-split {dim} leakage between {a} and {b}")

    # ---- anti-leakage report presence ---------------------------------------------------------
    anti_leakage_path = trials_dir / ANTI_LEAKAGE_REPORT_NAME
    if not anti_leakage_path.is_file():
        reasons.append("trial anti-leakage report missing")
    else:
        report = json.loads(anti_leakage_path.read_text(encoding="utf-8"))
        if not report.get("pass", False):
            reasons.append("trial anti-leakage report is not pass=true")

    statistics = {
        "max_position_error_recompute_disagreement_m": max_position_error_disagreement,
        "max_primary_metric_recompute_disagreement": max_primary_disagreement,
        "monotonic_trajectories": monotonicity_ok,
        "configured_minimum_inter_level_separation": diff.minimum_inter_level_separation,
    }

    return TrialValidationReport(
        passed=len(reasons) == 0,
        reasons=reasons,
        total_trials=total,
        split_counts=split_counts,
        difficulty_counts=difficulty_counts,
        family_counts=family_counts,
        split_difficulty_counts=split_difficulty_counts,
        statistics=statistics,
    )
