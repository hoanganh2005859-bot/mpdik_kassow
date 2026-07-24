"""Deterministic lexicographic candidate-selection objective (task section 5), development-only.

The objective is defined over **task-level aggregates**, NOT over pooled raw solves: Point-IK
(1,200 samples/dev) and trajectory waypoints (168,000/dev) are kept as separate terms so the far
larger waypoint count can never dominate the Point-IK signal. Warm-start and cold-start trajectory
success appear as explicit, separate levels.

Every reporting-tier success used here is the POST-HOC classification of a solve's final pose
error (``success_standard`` etc.), recomputed for every solve including max-iteration /
non-converged cases -- never the solver's ``converged`` flag.

Locked lexicographic tuple (each level compared in order; higher is better unless noted; the final
tie-break is candidate registration order):

  L1  safe_execution            finite/safe gate: fraction of dev solves with a finite pose error
  L2  pointik_standard_macro     Point-IK standard-success, macro-averaged across the 6 difficulty
                                 groups (task-level; not pooled with waypoints)
  L3  trajectory_standard_warm   warm-start standard-success, macro-averaged across trials
  L4  trajectory_standard_cold   cold-start standard-success, macro-averaged across trials
  L5  completion_macro           mean full-trajectory completion across all (trial, method)
  L6  coverage_macro             mean path-length coverage ratio across all (trial, method)
  L7  neg_pose_error_p95         negative P95 position error over all dev solves (lower is better)
  L8  warm_continuity            warm-start recovery rate, macro-averaged across warm trials
  L9  neg_runtime_p99            negative P99 solve time over all dev solves (lower is better)
  ... registration order         deterministic final tie-break

This module never reads validation or frozen_test, and the objective is recorded (see
``selection_objective_spec``) before any development result is inspected.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from evaluation_v2.candidate_configs import candidate_set

# Ordered lexicographic levels; every value is stored so that HIGHER is better.
SELECTION_LEVELS = (
    "safe_execution",
    "pointik_standard_macro",
    "trajectory_standard_warm",
    "trajectory_standard_cold",
    "completion_macro",
    "coverage_macro",
    "neg_pose_error_p95",
    "warm_continuity",
    "neg_runtime_p99",
)


def selection_objective_spec() -> dict:
    """The exact, deterministic objective definition -- recorded before any run is inspected."""
    return {
        "type": "lexicographic",
        "aggregation": "task_level_macros_not_pooled_rows",
        "success_definition": "post_hoc_reporting_tier_from_final_pose_error_all_solves",
        "levels": [
            {"order": 1, "name": "safe_execution", "direction": "max",
             "definition": "fraction of development solves (point + waypoint) with a finite pose error"},
            {"order": 2, "name": "pointik_standard_macro", "direction": "max",
             "definition": "mean over the 6 difficulty groups of that group's standard-tier "
                           "(3mm/2deg) success rate on the 1,200 development Point-IK samples"},
            {"order": 3, "name": "trajectory_standard_warm", "direction": "max",
             "definition": "mean over warm-start trials of each trial's standard-tier waypoint "
                           "success rate"},
            {"order": 4, "name": "trajectory_standard_cold", "direction": "max",
             "definition": "mean over cold-start trials of each trial's standard-tier waypoint "
                           "success rate"},
            {"order": 5, "name": "completion_macro", "direction": "max",
             "definition": "mean full_trajectory_completed over all (trial, method)"},
            {"order": 6, "name": "coverage_macro", "direction": "max",
             "definition": "mean path-length coverage ratio (min(actual/target,1)) over all "
                           "(trial, method)"},
            {"order": 7, "name": "pose_error_p95", "direction": "min",
             "definition": "P95 of position_error_m over all development solves (stored negated)"},
            {"order": 8, "name": "warm_continuity", "direction": "max",
             "definition": "mean warm-start recovery_rate over warm trials"},
            {"order": 9, "name": "runtime_p99", "direction": "min",
             "definition": "P99 of solve_time_ms over all development solves (stored negated)"},
            {"order": 10, "name": "registration_order", "direction": "tie_break",
             "definition": "candidate registration order in candidate_set(); lowest index wins"},
        ],
        "candidate_registration_order": [c.candidate_id for c in candidate_set()],
    }


def extract_selection_metrics(run_dir: Path) -> Dict[str, float]:
    """Extract the task-level selection metrics for one candidate's development run directory."""
    run_dir = Path(run_dir)
    point_df = pd.read_csv(run_dir / "tier1_point_dls" / "point_results.csv")
    waypoint_df = pd.read_csv(run_dir / "tier2_sequential_dls" / "waypoint_results.csv")
    summaries_df = pd.read_csv(run_dir / "tier2_sequential_dls" / "trajectory_trial_summaries.csv")
    tracking_df = pd.read_csv(run_dir / "tier3_trajectory_tracking" / "trajectory_metrics.csv")

    # L1 safe execution: fraction of finite pose errors across all dev solves.
    finite = int(np.isfinite(point_df["position_error_m"]).sum()) + int(
        np.isfinite(waypoint_df["position_error_m"]).sum()
    )
    total = len(point_df) + len(waypoint_df)
    safe_execution = float(finite / total) if total else 0.0

    # L2 Point-IK standard success, macro over the 6 difficulty groups (task-level).
    group_rates = point_df.groupby("difficulty_id")["success_standard"].mean()
    pointik_standard_macro = float(group_rates.mean()) if len(group_rates) else 0.0

    # L3/L4 trajectory standard success, macro over trials, per method.
    warm_summ = summaries_df[summaries_df["method"] == "warm_start"]
    cold_summ = summaries_df[summaries_df["method"] == "cold_start"]
    trajectory_standard_warm = float(warm_summ["success_rate_standard"].mean()) if len(warm_summ) else 0.0
    trajectory_standard_cold = float(cold_summ["success_rate_standard"].mean()) if len(cold_summ) else 0.0

    # L5/L6 completion + coverage, macro over all (trial, method).
    completion_macro = float(summaries_df["full_trajectory_completed"].astype(float).mean())
    coverage = tracking_df["coverage_ratio"].clip(upper=1.0)
    coverage_macro = float(coverage.mean()) if len(tracking_df) else 0.0

    # L7 pose error P95 tail over all dev solves.
    all_pos = np.concatenate([
        point_df["position_error_m"].to_numpy(), waypoint_df["position_error_m"].to_numpy()
    ])
    all_pos = all_pos[np.isfinite(all_pos)]
    pose_error_p95 = float(np.percentile(all_pos, 95)) if all_pos.size else float("inf")

    # L8 warm continuity.
    warm_recovery = warm_summ["recovery_rate"].dropna()
    warm_continuity = float(warm_recovery.mean()) if len(warm_recovery) else 0.0
    if np.isnan(warm_continuity):
        warm_continuity = 0.0

    # L9 runtime P99 tail over all dev solves.
    all_rt = np.concatenate([
        point_df["solve_time_ms"].to_numpy(), waypoint_df["solve_time_ms"].to_numpy()
    ])
    runtime_p99 = float(np.percentile(all_rt, 99)) if all_rt.size else float("inf")

    return {
        "safe_execution": safe_execution,
        "pointik_standard_macro": pointik_standard_macro,
        "trajectory_standard_warm": trajectory_standard_warm,
        "trajectory_standard_cold": trajectory_standard_cold,
        "completion_macro": completion_macro,
        "coverage_macro": coverage_macro,
        "neg_pose_error_p95": -pose_error_p95,
        "warm_continuity": warm_continuity,
        "neg_runtime_p99": -runtime_p99,
    }


def score_tuple(metrics: Dict[str, float]) -> Tuple[float, ...]:
    return tuple(round(float(metrics[level]), 9) for level in SELECTION_LEVELS)


def select_candidate(candidate_metrics: Dict[str, Dict[str, float]]) -> dict:
    """Pick the single best candidate. ``candidate_metrics`` maps candidate_id -> metrics dict."""
    order = [c.candidate_id for c in candidate_set()]
    ranking: List[dict] = []
    for cid, metrics in candidate_metrics.items():
        ranking.append(
            {
                "candidate_id": cid,
                "score": score_tuple(metrics),
                "order_index": order.index(cid) if cid in order else len(order),
                "metrics": metrics,
            }
        )
    ranking.sort(key=lambda r: (tuple(-s for s in r["score"]), r["order_index"]))
    selected = ranking[0]["candidate_id"]
    return {
        "selected_candidate_id": selected,
        "selection_levels": list(SELECTION_LEVELS),
        "selection_objective": selection_objective_spec(),
        "ranking": [
            {"candidate_id": r["candidate_id"], "score": r["score"], "metrics": r["metrics"]}
            for r in ranking
        ],
    }


def write_selection_report(output_path: Path, report: dict) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=list), encoding="utf-8")
    tmp.replace(output_path)
