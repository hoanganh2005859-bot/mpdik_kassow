"""Combine an already-validated development run and a validation-confirmation run for the SAME
selected candidate into one final Phase 8A summary, without recomputing development.

Development and validation are separate sample pools (different splits) -- they are never pooled
into one aggregate number here. This module only reads each run's own already-written per-tier
CSV/JSON outputs and reports them side by side plus the delta between them.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _tier_summary(run_dir: Path) -> dict:
    pr = pd.read_csv(run_dir / "tier1_point_dls" / "point_results.csv")
    wp = pd.read_csv(run_dir / "tier2_sequential_dls" / "waypoint_results.csv")
    ts = pd.read_csv(run_dir / "tier2_sequential_dls" / "trajectory_trial_summaries.csv")
    tm = pd.read_csv(run_dir / "tier3_trajectory_tracking" / "trajectory_metrics.csv")
    warm = ts[ts["method"] == "warm_start"]
    cold = ts[ts["method"] == "cold_start"]
    all_pos = np.concatenate([pr["position_error_m"].to_numpy(), wp["position_error_m"].to_numpy()])
    all_ori = np.concatenate([pr["orientation_error_deg"].to_numpy(), wp["orientation_error_deg"].to_numpy()])
    all_rt = np.concatenate([pr["solve_time_ms"].to_numpy(), wp["solve_time_ms"].to_numpy()])
    nonfinite = int((~np.isfinite(pr["position_error_m"])).sum()) + int((~np.isfinite(wp["position_error_m"])).sum())

    return {
        "solve_counts": {"point": int(len(pr)), "waypoint": int(len(wp)), "total": int(len(pr) + len(wp))},
        "point_ik": {
            "success_coarse": float(pr["success_coarse"].mean()),
            "success_standard": float(pr["success_standard"].mean()),
            "success_strict": float(pr["success_strict"].mean()),
            "position_error_p95_m": float(np.percentile(pr["position_error_m"], 95)),
        },
        "warm_start": {
            "waypoint_success_standard": float(wp[wp["method"] == "warm_start"]["success_standard"].mean()),
            "trial_macro_success_standard": float(warm["success_rate_standard"].mean()),
            "completion_rate": float(warm["full_trajectory_completed"].mean()),
            "recovery_rate_mean": float(warm["recovery_rate"].dropna().mean()) if warm["recovery_rate"].notna().any() else None,
        },
        "cold_start": {
            "waypoint_success_standard": float(wp[wp["method"] == "cold_start"]["success_standard"].mean()),
            "trial_macro_success_standard": float(cold["success_rate_standard"].mean()),
            "completion_rate": float(cold["full_trajectory_completed"].mean()),
        },
        "completion_and_coverage": {
            "completion_macro": float(ts["full_trajectory_completed"].mean()),
            "coverage_macro": float(tm["coverage_ratio"].clip(upper=1.0).mean()),
        },
        "error_tails": {
            "position_p50_m": float(np.percentile(all_pos, 50)),
            "position_p95_m": float(np.percentile(all_pos, 95)),
            "position_max_m": float(np.max(all_pos)),
            "orientation_p50_deg": float(np.percentile(all_ori, 50)),
            "orientation_p95_deg": float(np.percentile(all_ori, 95)),
            "orientation_max_deg": float(np.max(all_ori)),
        },
        "runtime": {
            "p50_ms": float(np.percentile(all_rt, 50)),
            "p95_ms": float(np.percentile(all_rt, 95)),
            "p99_ms": float(np.percentile(all_rt, 99)),
            "max_ms": float(np.max(all_rt)),
        },
        "failures": {
            "non_finite_count": nonfinite,
            "failure_reason_counts": {
                k: int(v) for k, v in pd.concat(
                    [pr["failure_reason"].fillna(""), wp["failure_reason"].fillna("")]
                ).value_counts().items() if k
            },
        },
    }


def build_final_phase8a_summary(
    candidate_id: str, development_run_dir: Path, validation_run_dir: Path
) -> Dict:
    dev = _tier_summary(Path(development_run_dir))
    val = _tier_summary(Path(validation_run_dir))
    dev_man = json.loads((Path(development_run_dir) / "run_manifest.json").read_text())
    val_man = json.loads((Path(validation_run_dir) / "run_manifest.json").read_text())
    return {
        "selected_candidate_id": candidate_id,
        "development": dev,
        "validation": val,
        "tier0_gate": {
            "development": dev_man["tiers"]["tier0"]["gate_pass"],
            "validation": val_man["tiers"]["tier0"]["gate_pass"],
        },
        "frozen_test_accessed": {
            "development": dev_man["frozen_test_accessed"],
            "validation": val_man["frozen_test_accessed"],
        },
    }


def write_final_summary(output_path: Path, summary: dict) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(output_path)
