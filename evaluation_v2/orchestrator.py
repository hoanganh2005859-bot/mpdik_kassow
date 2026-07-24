"""End-to-end Dataset v2 DLS evaluation for a single candidate config, over development and/or
validation, with atomic checkpointing and deterministic resume.

Pipeline (task section 8): Tier 0 gate -> Tier 1 point-IK -> Tier 2 warm/cold trajectory ->
Tier 3 tracking -> Tier 4 smoothness/feasibility/runtime. Tier 3/4 never call the solver. If the
Tier 0 gate fails, Tier 1-4 are skipped and the run is marked failed.

The evaluator reads ONLY the public evaluation root; ``frozen_test`` is never referenced. Every
solver/waypoint result is written to a checkpoint shard the first time it is produced, so a resume
recomputes nothing already completed and never duplicates a row.
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation_v2 import fingerprints
from evaluation_v2.candidate_configs import CandidateConfig
from evaluation_v2.checkpoint import CheckpointManager, build_run_fingerprints
from evaluation_v2.locator import EVAL_SPLITS, eval_output_paths, require_public_eval_root
from evaluation_v2 import metrics as metrics_mod
from evaluation_v2.point_eval import evaluate_point_ik_split
from evaluation_v2.protected_guard import assert_no_protected_fields
from evaluation_v2.tier0_gate import run_tier0_gate
from evaluation_v2.trajectory_eval import evaluate_trajectory_trial
from dataset_v2.locator import require_dataset_v2_root
from kinematics.model_loader import load_model_context
from utils.npz_utils import load_npz

_CLOSED_SHAPES = {"circle", "figure8"}


def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    tmp.replace(path)


def _load_public_manifest(public_paths) -> Dict[str, dict]:
    index = {}
    with open(public_paths.combined_manifest_file(), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index[row["trajectory_id"]] = row
    return index


def run_evaluation(
    dataset_root,
    public_root,
    output_root,
    candidate: CandidateConfig,
    *,
    splits=EVAL_SPLITS,
    run_name: Optional[str] = None,
    resume: bool = False,
    overwrite: bool = False,
    point_sample_limit: Optional[int] = None,
    trial_limit: Optional[int] = None,
    waypoint_limit: Optional[int] = None,
    methods=("warm_start", "cold_start"),
    show_progress: bool = False,
    model_context=None,
) -> dict:
    for split in splits:
        if split not in EVAL_SPLITS:
            raise ValueError(f"split '{split}' is not allowed in Phase 8A (frozen_test is off-limits)")

    ds = require_dataset_v2_root(dataset_root)
    public = require_public_eval_root(public_root)
    out = eval_output_paths(output_root, require_exists=False)
    run_name = run_name or candidate.candidate_id
    run_dir = out.config_run_dir(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    model_context = model_context or load_model_context()

    dataset_fp = fingerprints.directory_fingerprint(ds.root)["sha256"]
    public_manifest_doc = json.loads(public.manifest_file.read_text(encoding="utf-8"))
    public_bundle_fp = public_manifest_doc.get("public_bundle_fingerprint", "")

    resolved_config = {
        "candidate": candidate.to_record(),
        "splits": list(splits),
        "methods": list(methods),
        "limits": {
            "point_sample_limit": point_sample_limit,
            "trial_limit": trial_limit,
            "waypoint_limit": waypoint_limit,
        },
    }
    _atomic_json(run_dir / "resolved_config.json", resolved_config)

    run_fps = build_run_fingerprints(resolved_config, dataset_fp, public_bundle_fp)
    ckpt = CheckpointManager(run_dir)
    ckpt.begin(run_fps, resume=resume, overwrite=overwrite)

    manifest = {
        "run_name": run_name,
        "candidate_id": candidate.candidate_id,
        "splits": list(splits),
        "methods": list(methods),
        "fingerprints": run_fps.to_dict(),
        "environment": fingerprints.environment_fingerprint(),
        "tiers": {},
        "frozen_test_accessed": False,
    }

    # ---- Tier 0 gate ----
    gate = run_tier0_gate(
        ds.root,
        model_context=model_context,
        output_file=run_dir / "tier0_kinematics" / "tier0_gate_summary.json",
    )
    manifest["tiers"]["tier0"] = {
        "gate_pass": bool(gate.gate_pass),
        "max_jacobian_relative_error": gate.max_jacobian_relative_error,
        "reasons": gate.reasons,
    }
    if not gate.gate_pass:
        manifest["overall_status"] = "failed_tier0_gate"
        for t in ("tier1", "tier2", "tier3", "tier4"):
            manifest["tiers"][t] = {"status": "not_run", "reason": "Tier 0 gate failed"}
        _atomic_json(run_dir / "run_manifest.json", manifest)
        _atomic_json(run_dir / "FINAL_SUMMARY.json", {"status": "failed_tier0_gate", "run_name": run_name})
        return manifest

    # ---- Tier 1 point-IK ----
    tier1_dir = run_dir / "tier1_point_dls"
    point_frames = []
    for split in splits:
        shard_key = f"tier1/{split}"
        if not ckpt.has_shard(shard_key):
            df = evaluate_point_ik_split(
                public.point_ik_split_file(split), candidate,
                model_context=model_context, sample_limit=point_sample_limit, show_progress=show_progress,
            )
            df.insert(0, "split", split)
            ckpt.write_shard(shard_key, df)
        point_frames.append(ckpt.read_shard(shard_key))
    point_df = pd.concat(point_frames, ignore_index=True)
    _write_csv(tier1_dir / "point_results.csv", point_df)
    point_metrics_df, point_failures_df = metrics_mod.compute_point_metrics(point_df)
    _write_csv(tier1_dir / "point_metrics.csv", point_metrics_df)
    _write_csv(tier1_dir / "point_failures.csv", point_failures_df)
    manifest["tiers"]["tier1"] = {
        "status": "completed",
        "sample_count": int(len(point_df)),
        "success_rate_standard": float(point_df["success_standard"].mean()),
    }

    # ---- Tier 2 trajectory ----
    manifest_index = _load_public_manifest(public)
    closed_path_map = {tid: (row.get("shape") in _CLOSED_SHAPES) for tid, row in manifest_index.items()}
    tier2_dir = run_dir / "tier2_sequential_dls"
    waypoint_shard_keys: List[str] = []
    for split in splits:
        trials = load_npz(public.trials_split_file(split))
        assert_no_protected_fields(trials, f"public trials NPZ {split}")
        n_trials = int(trials["trial_id"].shape[0])
        selected = range(n_trials if trial_limit is None else min(trial_limit, n_trials))
        for method in methods:
            for idx in selected:
                trial_id = str(trials["trial_id"][idx])
                trajectory_id = str(trials["trajectory_id"][idx])
                shard_key = f"tier2/{split}/{method}/{trial_id}"
                waypoint_shard_keys.append(shard_key)
                if ckpt.has_shard(shard_key):
                    continue
                canonical_rel = manifest_index[trajectory_id]["public_canonical_path"]
                df = evaluate_trajectory_trial(
                    public.root / canonical_rel,
                    trial_id=trial_id,
                    trajectory_id=trajectory_id,
                    trajectory_family=str(trials["trajectory_family"][idx]),
                    difficulty=str(trials["difficulty"][idx]),
                    split=split,
                    q_initial=np.asarray(trials["q_initial"][idx], dtype=np.float64),
                    method=method,
                    candidate=candidate,
                    model_context=model_context,
                    waypoint_limit=waypoint_limit,
                    show_progress=show_progress,
                )
                ckpt.write_shard(shard_key, df)

    waypoint_df = ckpt.assemble(waypoint_shard_keys)
    _write_csv(tier2_dir / "waypoint_results.csv", waypoint_df)
    expected_wp = waypoint_limit if waypoint_limit is not None else None
    summaries_df = metrics_mod.compute_trial_summaries(waypoint_df, expected_waypoints=expected_wp)
    _write_csv(tier2_dir / "trajectory_trial_summaries.csv", summaries_df)
    warm_cold_df = metrics_mod.compute_warm_vs_cold(summaries_df)
    _write_csv(tier2_dir / "warm_vs_cold.csv", warm_cold_df)
    manifest["tiers"]["tier2"] = {
        "status": "completed",
        "waypoint_count": int(len(waypoint_df)),
        "trial_method_count": int(len(summaries_df)),
        "warm_vs_cold_pairs": int(len(warm_cold_df)),
    }

    # ---- Tier 3 tracking ----
    tracking_df, tracking_summary = metrics_mod.compute_tracking(waypoint_df, closed_path_map)
    _write_csv(run_dir / "tier3_trajectory_tracking" / "trajectory_metrics.csv", tracking_df)
    _atomic_json(run_dir / "tier3_trajectory_tracking" / "tracking_summary.json", tracking_summary)
    manifest["tiers"]["tier3"] = {"status": "completed", "trial_method_count": int(len(tracking_df))}

    # ---- Tier 4 smoothness/feasibility/runtime ----
    smooth_df, feas_df, runtime_df = metrics_mod.compute_tier4(waypoint_df, model_context)
    tier4_dir = run_dir / "tier4_joint_feasibility"
    _write_csv(tier4_dir / "smoothness_metrics.csv", smooth_df)
    _write_csv(tier4_dir / "feasibility_metrics.csv", feas_df)
    _write_csv(tier4_dir / "runtime_metrics.csv", runtime_df)
    manifest["tiers"]["tier4"] = {"status": "completed", "trial_method_count": int(len(smooth_df))}

    manifest["overall_status"] = "completed"
    _atomic_json(run_dir / "run_manifest.json", manifest)

    final_summary = {
        "status": "completed",
        "run_name": run_name,
        "candidate_id": candidate.candidate_id,
        "splits": list(splits),
        "point_sample_count": int(len(point_df)),
        "point_success_rate_standard": float(point_df["success_standard"].mean()),
        "waypoint_count": int(len(waypoint_df)),
        "tracking_summary": tracking_summary,
    }
    _atomic_json(run_dir / "FINAL_SUMMARY.json", final_summary)

    _atomic_json(run_dir / "evaluation_lock_candidate.json", {
        "candidate_id": candidate.candidate_id,
        "resolved_config": resolved_config,
        "fingerprints": run_fps.to_dict(),
        "lock_status": "candidate_run_completed_pending_selection",
    })

    ckpt.mark_complete()
    return manifest


def _write_csv(path: Path, df) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
