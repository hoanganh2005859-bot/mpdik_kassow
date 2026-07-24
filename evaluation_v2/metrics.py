"""Aggregate Tier 1-4 metrics for Dataset v2 evaluation, computed purely from the raw per-sample
(Tier 1) and per-waypoint (Tier 2) result frames. Tier 3 and Tier 4 never invoke a solver -- they
are deterministic functions of the already-solved target/actual poses and joint solutions.

Reuses ``evaluation/`` metric math (position tracking, orientation, cross-track, smoothness,
runtime, joint feasibility) unchanged.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.cross_track_metrics import compute_cross_track_metrics
from evaluation.orientation_metrics import summarize_orientation_errors
from evaluation.runtime_metrics import compute_runtime_metrics
from evaluation.smoothness_metrics import compute_smoothness_metrics
from evaluation.trajectory_metrics import compute_position_tracking_metrics
from kinematics.model_loader import ModelContext

_Q_COLS = [f"q_solution_q{i}" for i in range(1, 8)]
_REPORT_TIERS = ("coarse", "standard", "strict")


def _err_stats(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("rmse", "mean", "median", "p95", "max")}
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(values ** 2))),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


# ---------------------------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------------------------
def compute_point_metrics(point_df: pd.DataFrame):
    """Return (point_metrics_df, point_failures_df) from the raw per-sample frame."""
    groups = [("overall", point_df)]
    for diff_id, sub in point_df.groupby("difficulty_id"):
        groups.append((f"difficulty_{int(diff_id)}", sub))

    rows = []
    for name, sub in groups:
        n = len(sub)
        row = {"group": name, "sample_count": int(n)}
        for tier in _REPORT_TIERS:
            rate = float(sub[f"success_{tier}"].mean()) if n else float("nan")
            row[f"success_rate_{tier}"] = rate
            row[f"success_count_{tier}"] = int(sub[f"success_{tier}"].sum()) if n else 0
        row["converged_rate"] = float(sub["converged"].mean()) if n else float("nan")
        row.update(_err_stats(sub["position_error_m"].to_numpy(), "position_error_m"))
        row.update(_err_stats(sub["orientation_error_deg"].to_numpy(), "orientation_error_deg"))
        row.update(_err_stats(sub["iterations"].to_numpy(), "iterations"))
        row.update(_err_stats(sub["solve_time_ms"].to_numpy(), "solve_time_ms"))
        row["joint_limit_violation_rate"] = float(sub["joint_limit_violation"].mean()) if n else float("nan")
        rows.append(row)

    failures = point_df[~point_df["converged"]].copy()
    fail_cols = [
        "sample_id", "difficulty_id", "failure_reason", "position_error_m",
        "orientation_error_deg", "iterations",
    ]
    failures_df = failures[fail_cols].reset_index(drop=True) if len(failures) else pd.DataFrame(columns=fail_cols)
    return pd.DataFrame(rows), failures_df


# ---------------------------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------------------------
def _failure_streaks(success_mask: np.ndarray) -> int:
    longest = current = 0
    for ok in success_mask:
        current = 0 if ok else current + 1
        longest = max(longest, current)
    return int(longest)


def _recovery_rate(success_mask: np.ndarray) -> float:
    attempts = successes = 0
    for i in range(1, len(success_mask)):
        if not success_mask[i - 1]:
            attempts += 1
            if success_mask[i]:
                successes += 1
    return float(successes / attempts) if attempts else float("nan")


def compute_trial_summaries(waypoint_df: pd.DataFrame, expected_waypoints: Optional[int] = None) -> pd.DataFrame:
    """One row per (trial_id, method) with success/completion/continuity/error/runtime summaries."""
    rows = []
    for (trial_id, method), sub in waypoint_df.groupby(["trial_id", "method"]):
        sub = sub.sort_values("waypoint_id")
        n = len(sub)
        conv = sub["converged"].to_numpy().astype(bool)
        exp = expected_waypoints if expected_waypoints is not None else n
        row = {
            "trial_id": trial_id,
            "trajectory_id": sub["trajectory_id"].iloc[0],
            "trajectory_family": sub["trajectory_family"].iloc[0],
            "difficulty": sub["difficulty"].iloc[0],
            "split": sub["split"].iloc[0],
            "method": method,
            "waypoint_count": int(n),
            "full_trajectory_completed": bool(n == exp),
            "converged_waypoints": int(conv.sum()),
            "converged_rate": float(conv.mean()) if n else float("nan"),
            "maximum_failure_streak": _failure_streaks(conv),
            "recovery_rate": _recovery_rate(conv),
        }
        for tier in _REPORT_TIERS:
            row[f"success_rate_{tier}"] = float(sub[f"success_{tier}"].mean()) if n else float("nan")
        row.update(_err_stats(sub["position_error_m"].to_numpy(), "position_error_m"))
        row.update(_err_stats(sub["orientation_error_deg"].to_numpy(), "orientation_error_deg"))
        row.update(_err_stats(sub["iterations"].to_numpy(), "iterations"))
        row.update(_err_stats(sub["solve_time_ms"].to_numpy(), "solve_time_ms"))
        row["minimum_sigma_min"] = float(sub["sigma_min"].min())
        row["minimum_joint_limit_margin"] = float(sub["minimum_joint_limit_margin"].min())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["trial_id", "method"]).reset_index(drop=True)


_WARM_VS_COLD_METRICS = (
    "success_rate_standard", "success_rate_strict", "converged_rate",
    "full_trajectory_completed", "maximum_failure_streak", "recovery_rate",
    "position_error_m_rmse", "position_error_m_p95", "orientation_error_deg_rmse",
    "iterations_mean", "solve_time_ms_mean", "solve_time_ms_p95", "minimum_sigma_min",
)


def compute_warm_vs_cold(summaries_df: pd.DataFrame) -> pd.DataFrame:
    """Paired warm-vs-cold table: one row per trial present in both methods."""
    warm = summaries_df[summaries_df["method"] == "warm_start"].set_index("trial_id")
    cold = summaries_df[summaries_df["method"] == "cold_start"].set_index("trial_id")
    common = sorted(set(warm.index) & set(cold.index))
    rows = []
    for trial_id in common:
        row = {
            "trial_id": trial_id,
            "trajectory_id": warm.loc[trial_id, "trajectory_id"],
            "trajectory_family": warm.loc[trial_id, "trajectory_family"],
            "difficulty": warm.loc[trial_id, "difficulty"],
        }
        for metric in _WARM_VS_COLD_METRICS:
            row[f"warm_{metric}"] = warm.loc[trial_id, metric]
            row[f"cold_{metric}"] = cold.loc[trial_id, metric]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Tier 3 (tracking) -- no solver calls
# ---------------------------------------------------------------------------------------------
def compute_tracking(waypoint_df: pd.DataFrame, closed_path_map: Dict[str, bool]):
    """Per (trial_id, method) position/orientation/cross-track metrics + an aggregate summary."""
    rows = []
    for (trial_id, method), sub in waypoint_df.groupby(["trial_id", "method"]):
        sub = sub.sort_values("waypoint_id")
        target_pos = sub[[f"target_position_{a}" for a in "xyz"]].to_numpy()
        actual_pos = sub[[f"actual_position_{a}" for a in "xyz"]].to_numpy()
        pos = compute_position_tracking_metrics(target_pos, actual_pos)
        orient = summarize_orientation_errors(np.radians(sub["orientation_error_deg"].to_numpy()))
        closed = bool(closed_path_map.get(sub["trajectory_id"].iloc[0], False))
        ct = compute_cross_track_metrics(actual_pos, target_pos, closed_path=closed).summary
        row = {
            "trial_id": trial_id,
            "trajectory_id": sub["trajectory_id"].iloc[0],
            "method": method,
            "difficulty": sub["difficulty"].iloc[0],
            "position_rmse_m": pos.rmse_m,
            "position_median_m": pos.median_m,
            "position_p95_m": pos.p95_m,
            "position_max_m": pos.max_m,
            "endpoint_error_m": pos.endpoint_error_m,
            "start_point_error_m": pos.start_point_error_m,
            "target_path_length_m": pos.target_path_length_m,
            "actual_path_length_m": pos.actual_path_length_m,
            "path_length_ratio": pos.path_length_ratio,
            "orientation_rmse_deg": orient.rmse_deg,
            "orientation_p95_deg": orient.p95_deg,
            "orientation_max_deg": orient.max_deg,
            "cross_track_rmse_m": float(ct.cross_track_rmse_m),
            "cross_track_p95_m": float(ct.cross_track_p95_m),
            "cross_track_max_m": float(ct.cross_track_max_m),
            "final_progress_ratio": float(ct.final_progress_ratio),
            "coverage_ratio": pos.path_length_ratio,
        }
        rows.append(row)
    tracking_df = pd.DataFrame(rows)
    summary = {}
    if len(tracking_df):
        for method, sub in tracking_df.groupby("method"):
            summary[method] = {
                "trial_count": int(len(sub)),
                "position_rmse_m_median": float(sub["position_rmse_m"].median()),
                "position_rmse_m_p95": float(sub["position_rmse_m"].quantile(0.95)),
                "orientation_rmse_deg_median": float(sub["orientation_rmse_deg"].median()),
                "cross_track_rmse_m_median": float(sub["cross_track_rmse_m"].median()),
            }
    return tracking_df, summary


# ---------------------------------------------------------------------------------------------
# Tier 4 (smoothness / feasibility / runtime) -- no solver calls
# ---------------------------------------------------------------------------------------------
def compute_tier4(waypoint_df: pd.DataFrame, model_context: ModelContext, deadline_ms: Optional[float] = None):
    """Per (trial_id, method) smoothness, joint feasibility, and runtime tables."""
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    vlimits = model_context.velocity_limits_rad_s
    smooth_rows, feas_rows, runtime_rows = [], [], []
    for (trial_id, method), sub in waypoint_df.groupby(["trial_id", "method"]):
        sub = sub.sort_values("waypoint_id")
        q = sub[_Q_COLS].to_numpy(dtype=np.float64)
        t = sub["time_s"].to_numpy(dtype=np.float64)
        base = {
            "trial_id": trial_id,
            "trajectory_id": sub["trajectory_id"].iloc[0],
            "method": method,
            "difficulty": sub["difficulty"].iloc[0],
        }
        # Smoothness
        sm = compute_smoothness_metrics(q, t)
        smooth_rows.append({
            **base,
            "max_joint_jump_rad": sm.max_joint_jump_rad,
            "total_joint_variation_rad": float(np.sum(sm.total_joint_variation_per_joint)),
            "global_rms_jerk": sm.global_rms_jerk,
            "max_abs_velocity_rad_s": float(np.max(sm.max_abs_velocity_per_joint)) if sm.velocity_available else float("nan"),
            "max_abs_acceleration_rad_s2": float(np.max(sm.max_abs_acceleration_per_joint)) if sm.acceleration_available else float("nan"),
            "max_abs_jerk_rad_s3": float(np.max(sm.max_abs_jerk_per_joint)) if sm.jerk_available else float("nan"),
        })
        # Feasibility
        if len(t) >= 2:
            vel = np.gradient(q, t, axis=0)
        else:
            vel = np.zeros_like(q)
        margin = np.min([
            np.min((q - lower)), np.min((upper - q)),
        ])
        vel_util = float(np.max(np.abs(vel) / vlimits)) if q.shape[0] else float("nan")
        feas_rows.append({
            **base,
            "minimum_joint_limit_margin_rad": float(np.minimum((q - lower).min(), (upper - q).min())),
            "operational_limit_violation_count": int(np.sum((q < lower) | (q > upper))),
            "minimum_sigma_min": float(sub["sigma_min"].min()),
            "maximum_velocity_utilization": vel_util,
            # Acceleration feasibility is descriptive only: the model has no locked acceleration
            # limits, so no acceleration acceptance is asserted and no dynamic feasibility claimed.
            "acceleration_status": "unavailable_no_locked_acceleration_limits",
        })
        # Runtime
        rt = compute_runtime_metrics(sub["solve_time_ms"].to_numpy(dtype=np.float64), deadline_ms=deadline_ms)
        runtime_rows.append({
            **base,
            "count": rt.count,
            "mean_ms": rt.mean_ms,
            "median_ms": rt.median_ms,
            "p90_ms": rt.p90_ms,
            "p95_ms": rt.p95_ms,
            "p99_ms": rt.p99_ms,
            "max_ms": rt.max_ms,
        })
    return pd.DataFrame(smooth_rows), pd.DataFrame(feas_rows), pd.DataFrame(runtime_rows)
