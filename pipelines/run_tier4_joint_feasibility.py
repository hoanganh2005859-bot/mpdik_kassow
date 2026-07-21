"""Tier 4 pipeline: joint trajectory smoothness and operational-limit feasibility from Tier 2 output.

Reads tier2_sequential_dls/waypoint_results.csv and never re-runs the solver: joint smoothness
(velocity/acceleration/jerk), operational-limit feasibility, singularity-along-path, and runtime
metrics are all deterministic functions of the already-solved q(t) and per-waypoint solver
telemetry (sigma_min, condition_number, solve_time_ms) already present in that CSV.

Cold-start solutions can jump discontinuously between waypoints (each solve starts fresh from
the trial's initial configuration; see algorithms.cold_start_dls). This is measured, not
smoothed away -- ``max_joint_jump_rad`` is reported as-is for both methods.

Coverage: if a trial did not process every waypoint of its trajectory (e.g. a fatal solver
failure truncated it early, or a smoke run's --waypoint-limit), ``coverage_ratio`` reports the
processed fraction rather than silently treating the truncated data as complete.

Can be run standalone:
    python -m pipelines.run_tier4_joint_feasibility --tier2-dir results/run/tier2_sequential_dls --output results/tier4_only
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from evaluation.joint_feasibility_metrics import compute_joint_feasibility_metrics
from evaluation.plotting import (
    plot_joint_acceleration,
    plot_joint_jerk,
    plot_joint_trajectory,
    plot_joint_velocity,
    plot_sigma_min_over_time,
)
from evaluation.runtime_metrics import compute_runtime_metrics
from evaluation.smoothness_metrics import compute_smoothness_metrics
from utils.config_loader import load_json_config
from utils.dataset_locator import CONFIGS_DIR, TRAJECTORY_MANIFEST_PATH, TRAJECTORY_TRIALS_PATH
from utils.result_logger import write_result_csv

logger = logging.getLogger(__name__)

_Q_COLS = [f"q_solution_q{i}" for i in range(1, 8)]


def _q_trajectory(group: pd.DataFrame) -> np.ndarray:
    return group[_Q_COLS].to_numpy(dtype=np.float64)


def run_tier4(
    waypoint_results_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    make_plots: bool = True,
) -> dict:
    """Compute Tier 4 smoothness/feasibility/singularity-path/runtime metrics from Tier 2 output."""
    robot_config = load_json_config(CONFIGS_DIR / "robot_config.json")
    dls_config = load_json_config(CONFIGS_DIR / "dls_config.json")
    lower = np.asarray(robot_config["operational_lower_rad"], dtype=np.float64)
    upper = np.asarray(robot_config["operational_upper_rad"], dtype=np.float64)
    velocity_limits = np.asarray(robot_config["velocity_limits_rad_s"], dtype=np.float64)
    near_singular_threshold = float(dls_config["singularity_sigma_threshold"])

    manifest_df = pd.read_csv(TRAJECTORY_MANIFEST_PATH)
    num_waypoints_by_trajectory = dict(zip(manifest_df["trajectory_id"], manifest_df["num_waypoints"]))
    trials_df = pd.read_csv(TRAJECTORY_TRIALS_PATH)
    control_period_by_trial = dict(zip(trials_df["trial_id"], trials_df["control_period_s"]))
    speed_scale_by_trial = dict(zip(trials_df["trial_id"], trials_df["speed_scale"]))

    smoothness_rows = []
    feasibility_rows = []
    singularity_rows = []
    runtime_rows = []
    representative_groups = {}

    for (trial_id, method), group in waypoint_results_df.groupby(["trial_id", "method"], sort=True):
        group = group.sort_values("waypoint_id")
        trajectory_id = group["trajectory_id"].iloc[0]
        expected = int(num_waypoints_by_trajectory.get(trajectory_id, len(group)))
        coverage_ratio = float(len(group) / expected) if expected > 0 else float("nan")

        q_trajectory = _q_trajectory(group)
        time_s = group["time_s"].to_numpy(dtype=np.float64)
        smoothness = compute_smoothness_metrics(q_trajectory, time_s)

        smoothness_rows.append({
            "trial_id": trial_id, "trajectory_id": trajectory_id, "method": method,
            "joint_count": smoothness.joint_count, "sample_count": smoothness.sample_count,
            "velocity_available": smoothness.velocity_available,
            "global_max_abs_velocity_rad_s": float(np.max(smoothness.max_abs_velocity_per_joint)) if smoothness.velocity_available else None,
            "global_rms_velocity_rad_s": float(np.sqrt(np.mean(smoothness.rms_velocity_per_joint ** 2))) if smoothness.velocity_available else None,
            "acceleration_available": smoothness.acceleration_available,
            "global_max_abs_acceleration_rad_s2": float(np.max(smoothness.max_abs_acceleration_per_joint)) if smoothness.acceleration_available else None,
            "global_rms_acceleration_rad_s2": float(np.sqrt(np.mean(smoothness.rms_acceleration_per_joint ** 2))) if smoothness.acceleration_available else None,
            "jerk_available": smoothness.jerk_available,
            "global_max_abs_jerk_rad_s3": float(np.max(smoothness.max_abs_jerk_per_joint)) if smoothness.jerk_available else None,
            "global_rms_jerk_rad_s3": smoothness.global_rms_jerk,
            "max_joint_jump_rad": smoothness.max_joint_jump_rad,
            "max_joint_jump_joint_index": smoothness.max_joint_jump_joint_index,
            "max_joint_jump_timestep_index": smoothness.max_joint_jump_timestep_index,
            "max_total_joint_variation_rad": float(np.max(smoothness.total_joint_variation_per_joint)),
            "second_difference_norm_rad": smoothness.second_difference_norm_rad,
            "coverage_ratio": coverage_ratio,
        })

        if smoothness.velocity_available:
            feasibility = compute_joint_feasibility_metrics(
                q_trajectory,
                smoothness.velocity,
                lower,
                upper,
                velocity_limits,
                joint_acceleration=smoothness.acceleration if smoothness.acceleration_available else None,
                acceleration_limits=None,
            )
            feasibility_rows.append({
                "trial_id": trial_id, "trajectory_id": trajectory_id, "method": method,
                "operational_limit_violation_count": feasibility.operational_limit_violation_count,
                "operational_limit_violation_rate": feasibility.operational_limit_violation_rate,
                "minimum_normalized_joint_limit_margin": feasibility.minimum_normalized_joint_limit_margin,
                "maximum_velocity_utilization": feasibility.maximum_velocity_utilization,
                "velocity_violation_count": feasibility.velocity_violation_count,
                "acceleration_status": feasibility.acceleration_status,
                "maximum_acceleration_utilization": feasibility.maximum_acceleration_utilization,
                "acceleration_violation_count": feasibility.acceleration_violation_count,
                "coverage_ratio": coverage_ratio,
            })
        else:
            feasibility_rows.append({
                "trial_id": trial_id, "trajectory_id": trajectory_id, "method": method,
                "operational_limit_violation_count": None, "operational_limit_violation_rate": None,
                "minimum_normalized_joint_limit_margin": None, "maximum_velocity_utilization": None,
                "velocity_violation_count": None, "acceleration_status": "unavailable",
                "maximum_acceleration_utilization": None, "acceleration_violation_count": None,
                "coverage_ratio": coverage_ratio,
            })

        sigma_min = group["sigma_min"].to_numpy(dtype=np.float64)
        condition_number = group["condition_number"].to_numpy(dtype=np.float64)
        finite_condition = condition_number[np.isfinite(condition_number)]
        near_singular_mask = sigma_min <= near_singular_threshold
        singularity_rows.append({
            "trial_id": trial_id, "trajectory_id": trajectory_id, "method": method,
            "waypoint_count": len(group),
            "minimum_sigma_min": float(np.min(sigma_min)),
            "p05_sigma_min": float(np.percentile(sigma_min, 5)),
            "maximum_condition_number": float(np.max(finite_condition)) if finite_condition.size else float("inf"),
            "near_singular_count": int(np.sum(near_singular_mask)),
            "near_singular_fraction": float(np.mean(near_singular_mask)),
            "worst_waypoint_index": int(np.argmin(sigma_min)),
        })

        solve_time_ms = group["solve_time_ms"].to_numpy(dtype=np.float64)
        control_period_s = float(control_period_by_trial.get(trial_id, np.nan))
        speed_scale = float(speed_scale_by_trial.get(trial_id, 1.0))
        deadline_ms = (control_period_s / speed_scale) * 1000.0 if np.isfinite(control_period_s) else None
        runtime = compute_runtime_metrics(solve_time_ms, deadline_ms=deadline_ms)
        runtime_rows.append({
            "trial_id": trial_id, "trajectory_id": trajectory_id, "method": method,
            "count": runtime.count, "mean_ms": runtime.mean_ms, "median_ms": runtime.median_ms,
            "p90_ms": runtime.p90_ms, "p95_ms": runtime.p95_ms, "p99_ms": runtime.p99_ms, "max_ms": runtime.max_ms,
            "deadline_miss_count": runtime.deadline_miss_count, "deadline_miss_rate": runtime.deadline_miss_rate,
        })

        rep_key = f"{trajectory_id}::{method}"
        if rep_key not in representative_groups:
            representative_groups[rep_key] = (group, smoothness)

    smoothness_df = pd.DataFrame(smoothness_rows)
    feasibility_df = pd.DataFrame(feasibility_rows)
    singularity_df = pd.DataFrame(singularity_rows)
    runtime_df = pd.DataFrame(runtime_rows)

    if output_dir is not None:
        output_dir = Path(output_dir)
        write_result_csv(smoothness_df, output_dir / "smoothness_metrics.csv")
        write_result_csv(feasibility_df, output_dir / "joint_feasibility_metrics.csv")
        write_result_csv(singularity_df, output_dir / "singularity_path_metrics.csv")
        write_result_csv(runtime_df, output_dir / "runtime_metrics.csv")

        if make_plots:
            figures_dir = output_dir / "figures"
            for rep_key, (group, smoothness) in sorted(representative_groups.items()):
                safe_name = rep_key.replace("::", "_")
                time_s = group["time_s"].to_numpy(dtype=np.float64)
                q_trajectory = _q_trajectory(group)
                sigma_min = group["sigma_min"].to_numpy(dtype=np.float64)

                plot_joint_trajectory(time_s, q_trajectory, figures_dir / f"joint_trajectory_{safe_name}.png", title=rep_key)
                plot_sigma_min_over_time(time_s, sigma_min, figures_dir / f"sigma_min_{safe_name}.png", title=rep_key)
                if smoothness.velocity_available:
                    plot_joint_velocity(time_s, smoothness.velocity, figures_dir / f"joint_velocity_{safe_name}.png", title=rep_key)
                if smoothness.acceleration_available:
                    plot_joint_acceleration(time_s, smoothness.acceleration, figures_dir / f"joint_acceleration_{safe_name}.png", title=rep_key)
                if smoothness.jerk_available:
                    plot_joint_jerk(time_s, smoothness.jerk, figures_dir / f"joint_jerk_{safe_name}.png", title=rep_key)

    return {
        "smoothness_df": smoothness_df,
        "feasibility_df": feasibility_df,
        "singularity_df": singularity_df,
        "runtime_df": runtime_df,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tier 4: joint smoothness and feasibility metrics from Tier 2 output.")
    parser.add_argument("--tier2-dir", required=True, type=Path, help="Directory containing tier2 waypoint_results.csv.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(message)s")

    waypoint_df = pd.read_csv(args.tier2_dir / "waypoint_results.csv")
    run_tier4(waypoint_df, output_dir=args.output, make_plots=not args.no_plots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
