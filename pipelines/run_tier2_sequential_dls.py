"""Tier 2 pipeline: sequential (warm/cold-start) DLS across each selected trajectory trial.

Selects (trajectory_id, trial_id) combinations from trajectories/trajectory_manifest.csv /
trajectory_trials.csv (via ``pipelines._common.select_trials``), runs each with every requested
method through algorithms.sequential_dls.run_sequential_trial, and aggregates the results into a
per-(trial_id, method) TrajectoryTrialSummary plus a paired warm-vs-cold comparison.

A fatal (non-finite) failure in one trial is caught and recorded; it never aborts the remaining
trials or discards already-completed results (see module docstring of
algorithms.sequential_dls / algorithms.warm_start_dls for how ordinary per-waypoint failures are
handled, which is a different, non-fatal case that never raises).

Can be run standalone:
    python -m pipelines.run_tier2_sequential_dls --output results/tier2_only --trajectory-ids line_fixed_orientation
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from algorithms.result_types import WaypointResult, waypoint_results_to_dataframe
from algorithms.sequential_dls import run_sequential_trial
from evaluation.plotting import plot_target_vs_actual_3d, plot_warm_vs_cold_summary
from evaluation.result_types import TrajectoryTrialSummary, trajectory_trial_summaries_to_dataframe
from evaluation.runtime_metrics import compute_runtime_metrics
from evaluation.trajectory_metrics import compute_position_tracking_metrics
from evaluation.orientation_metrics import geodesic_angles_from_quaternions, summarize_orientation_errors
from evaluation.waypoint_metrics import compute_waypoint_metrics
from kinematics.dls_solver import load_dls_config
from kinematics.model_loader import ModelContext, load_model_context
from pipelines._common import select_trials
from utils.csv_utils import json_safe_scalar
from utils.dataset_locator import TRAJECTORY_MANIFEST_PATH, TRAJECTORY_TRIALS_PATH
from utils.result_logger import write_result_csv, write_result_json

logger = logging.getLogger(__name__)


def _build_trial_summary(
    trial_id: str,
    trajectory_id: str,
    trial_category: str,
    method: str,
    repeat_id: int,
    seed: int,
    speed_scale: float,
    control_period_s_scaled: float,
    results: List[WaypointResult],
    expected_waypoint_count: int,
    confidence_level: float,
) -> TrajectoryTrialSummary:
    waypoint_metrics = compute_waypoint_metrics(
        results, expected_waypoint_count, confidence_level=confidence_level, deadline_s=control_period_s_scaled
    )

    target_positions = np.array([r.target_position for r in results], dtype=np.float64)
    actual_positions = np.array([r.actual_position for r in results], dtype=np.float64)
    target_quaternions = np.array([r.target_quaternion for r in results], dtype=np.float64)
    actual_quaternions = np.array([r.actual_quaternion for r in results], dtype=np.float64)

    position_metrics = compute_position_tracking_metrics(target_positions, actual_positions)
    angles_rad = geodesic_angles_from_quaternions(target_quaternions, actual_quaternions)
    orientation_metrics = summarize_orientation_errors(angles_rad)

    solve_times_ms = np.array([r.solve_time_ms for r in results], dtype=np.float64)
    runtime = compute_runtime_metrics(solve_times_ms, deadline_ms=control_period_s_scaled * 1000.0)

    sigma_mins = np.array([r.sigma_min for r in results], dtype=np.float64)
    condition_numbers = np.array([r.condition_number for r in results], dtype=np.float64)
    finite_condition = condition_numbers[np.isfinite(condition_numbers)]
    margins = np.array([r.minimum_joint_limit_margin for r in results], dtype=np.float64)

    return TrajectoryTrialSummary(
        trial_id=trial_id,
        trajectory_id=trajectory_id,
        trial_category=trial_category,
        method=method,
        repeat_id=repeat_id,
        seed=seed,
        speed_scale=speed_scale,
        control_period_s=control_period_s_scaled,
        waypoint_count=waypoint_metrics.waypoint_count,
        successful_waypoints=waypoint_metrics.successful_waypoints,
        failed_waypoints=waypoint_metrics.failed_waypoints,
        waypoint_success_rate=waypoint_metrics.waypoint_success_rate,
        full_trajectory_completed=waypoint_metrics.full_trajectory_completed,
        maximum_failure_streak=waypoint_metrics.maximum_failure_streak,
        recovery_rate=waypoint_metrics.recovery_rate,
        position_rmse_m=position_metrics.rmse_m,
        position_mae_m=position_metrics.mae_m,
        position_median_m=position_metrics.median_m,
        position_p95_m=position_metrics.p95_m,
        position_max_m=position_metrics.max_m,
        orientation_rmse_deg=orientation_metrics.rmse_deg,
        orientation_p95_deg=orientation_metrics.p95_deg,
        orientation_max_deg=orientation_metrics.max_deg,
        mean_iterations=waypoint_metrics.mean_iterations,
        p95_iterations=waypoint_metrics.p95_iterations,
        mean_solve_time_ms=runtime.mean_ms,
        p95_solve_time_ms=runtime.p95_ms,
        deadline_miss_rate=float(runtime.deadline_miss_rate),
        minimum_sigma_min=float(np.min(sigma_mins)),
        maximum_condition_number=float(np.max(finite_condition)) if finite_condition.size else float("inf"),
        minimum_joint_limit_margin=float(np.min(margins)),
    )


def _build_warm_vs_cold(summaries_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "waypoint_success_rate",
        "full_trajectory_completed",
        "maximum_failure_streak",
        "recovery_rate",
        "position_rmse_m",
        "position_p95_m",
        "orientation_rmse_deg",
        "mean_iterations",
        "mean_solve_time_ms",
        "p95_solve_time_ms",
        "deadline_miss_rate",
        "minimum_sigma_min",
    ]
    warm = summaries_df[summaries_df["method"] == "warm_start"].set_index("trial_id")
    cold = summaries_df[summaries_df["method"] == "cold_start"].set_index("trial_id")
    common_ids = sorted(set(warm.index) & set(cold.index))

    rows = []
    for trial_id in common_ids:
        row = {
            "trial_id": trial_id,
            "trajectory_id": warm.loc[trial_id, "trajectory_id"],
            "trial_category": warm.loc[trial_id, "trial_category"],
            "speed_scale": warm.loc[trial_id, "speed_scale"],
        }
        for col in metric_cols:
            row[f"warm_{col}"] = json_safe_scalar(warm.loc[trial_id, col])
            row[f"cold_{col}"] = json_safe_scalar(cold.loc[trial_id, col])
        rows.append(row)

    columns = (
        ["trial_id", "trajectory_id", "trial_category", "speed_scale"]
        + [f"warm_{col}" for col in metric_cols]
        + [f"cold_{col}" for col in metric_cols]
    )
    return pd.DataFrame(rows, columns=columns)


def run_tier2(
    model_context: Optional[ModelContext] = None,
    dls_config: Optional[dict] = None,
    trajectory_ids: Optional[Sequence[str]] = None,
    trial_category: str = "all",
    methods: Optional[Sequence[str]] = None,
    trial_limit: Optional[int] = None,
    waypoint_limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    confidence_level: float = 0.95,
    make_plots: bool = True,
    show_progress: bool = True,
) -> dict:
    """Run the full Tier 2 sequential-DLS pass over the selected (trial, method) combinations."""
    model_context = model_context if model_context is not None else load_model_context()
    dls_config = dls_config if dls_config is not None else load_dls_config()
    methods = list(methods) if methods is not None else ["warm_start", "cold_start"]

    trials_df = pd.read_csv(TRAJECTORY_TRIALS_PATH)
    manifest_df = pd.read_csv(TRAJECTORY_MANIFEST_PATH)
    selected_trials = select_trials(trials_df, trajectory_ids, trial_category, trial_limit)

    logger.info(
        "tier2: running %d trial(s) x %d method(s) = %d combination(s)",
        len(selected_trials), len(methods), len(selected_trials) * len(methods),
    )

    all_waypoint_results: List[WaypointResult] = []
    summaries: List[TrajectoryTrialSummary] = []
    failure_cases: List[dict] = []
    representative_by_key: Dict[str, List[WaypointResult]] = {}

    for _, trial_row in selected_trials.iterrows():
        trial_id = trial_row["trial_id"]
        trajectory_id = trial_row["trajectory_id"]
        manifest_row = manifest_df[manifest_df["trajectory_id"] == trajectory_id].iloc[0]
        expected_waypoint_count = int(manifest_row["num_waypoints"])
        if waypoint_limit is not None:
            expected_waypoint_count = min(expected_waypoint_count, waypoint_limit)

        for method in methods:
            try:
                results = run_sequential_trial(
                    trajectory_id=trajectory_id,
                    trial_id=trial_id,
                    method=method,
                    model_context=model_context,
                    dls_config=dls_config,
                    waypoint_limit=waypoint_limit,
                    fail_fast=False,
                    show_progress=show_progress,
                )
            except Exception as exc:  # fatal, unrecoverable numerical failure for this trial only
                logger.error("tier2: trial '%s' method '%s' failed fatally: %s", trial_id, method, exc)
                failure_cases.append(
                    {
                        "trial_id": trial_id,
                        "trajectory_id": trajectory_id,
                        "method": method,
                        "fatal_error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                continue

            all_waypoint_results.extend(results)

            failed = [
                {
                    "trial_id": trial_id,
                    "trajectory_id": trajectory_id,
                    "method": method,
                    "waypoint_id": r.waypoint_id,
                    "failure_reason": r.failure_reason,
                    "position_error_m": r.position_error_m,
                    "orientation_error_deg": r.orientation_error_deg,
                }
                for r in results
                if not r.success
            ]
            failure_cases.extend(failed)

            speed_scale = float(trial_row["speed_scale"])
            control_period_scaled = float(trial_row["control_period_s"]) / speed_scale
            summaries.append(
                _build_trial_summary(
                    trial_id=trial_id,
                    trajectory_id=trajectory_id,
                    trial_category=trial_row["trial_category"],
                    method=method,
                    repeat_id=int(trial_row["repeat_id"]),
                    seed=int(trial_row["seed"]),
                    speed_scale=speed_scale,
                    control_period_s_scaled=control_period_scaled,
                    results=results,
                    expected_waypoint_count=expected_waypoint_count,
                    confidence_level=confidence_level,
                )
            )

            rep_key = f"{trajectory_id}::{method}"
            if rep_key not in representative_by_key:
                representative_by_key[rep_key] = results

    waypoint_df = (
        waypoint_results_to_dataframe(all_waypoint_results)
        if all_waypoint_results
        else pd.DataFrame(columns=["trial_id", "trajectory_id", "trial_category", "method", "waypoint_id"])
    )
    summaries_df = (
        trajectory_trial_summaries_to_dataframe(summaries)
        if summaries
        else pd.DataFrame(columns=list(TrajectoryTrialSummary.__dataclass_fields__))
    )
    warm_vs_cold_df = _build_warm_vs_cold(summaries_df)

    if output_dir is not None:
        output_dir = Path(output_dir)
        write_result_csv(waypoint_df, output_dir / "waypoint_results.csv")
        write_result_csv(summaries_df, output_dir / "trajectory_trial_summaries.csv")
        write_result_csv(warm_vs_cold_df, output_dir / "warm_vs_cold.csv")
        write_result_json(failure_cases, output_dir / "failure_cases.json")

        if make_plots and not summaries_df.empty:
            figures_dir = output_dir / "figures"
            if not warm_vs_cold_df.empty:
                labels = warm_vs_cold_df["trial_id"].tolist()
                plot_warm_vs_cold_summary(
                    warm_vs_cold_df["warm_waypoint_success_rate"].to_numpy(),
                    warm_vs_cold_df["cold_waypoint_success_rate"].to_numpy(),
                    labels,
                    figures_dir / "warm_vs_cold_success.png",
                    ylabel="Waypoint success rate",
                )
                plot_warm_vs_cold_summary(
                    warm_vs_cold_df["warm_mean_iterations"].to_numpy(),
                    warm_vs_cold_df["cold_mean_iterations"].to_numpy(),
                    labels,
                    figures_dir / "warm_vs_cold_iterations.png",
                    ylabel="Mean iterations",
                )
                plot_warm_vs_cold_summary(
                    warm_vs_cold_df["warm_mean_solve_time_ms"].to_numpy(),
                    warm_vs_cold_df["cold_mean_solve_time_ms"].to_numpy(),
                    labels,
                    figures_dir / "warm_vs_cold_runtime.png",
                    ylabel="Mean solve time (ms)",
                )

            for rep_key, results in sorted(representative_by_key.items()):
                safe_name = rep_key.replace("::", "_")
                target_positions = np.array([r.target_position for r in results], dtype=np.float64)
                actual_positions = np.array([r.actual_position for r in results], dtype=np.float64)
                plot_target_vs_actual_3d(
                    target_positions, actual_positions, figures_dir / f"target_vs_actual_{safe_name}.png", title=rep_key
                )

    return {
        "waypoint_results": all_waypoint_results,
        "waypoint_df": waypoint_df,
        "summaries": summaries,
        "summaries_df": summaries_df,
        "warm_vs_cold_df": warm_vs_cold_df,
        "failure_cases": failure_cases,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tier 2: sequential (warm/cold-start) DLS.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trajectory-ids", type=str, default=None, help="Comma-separated trajectory_id list.")
    parser.add_argument("--trial-category", choices=["repeatability", "robustness", "all"], default="all")
    parser.add_argument("--methods", type=str, default="warm_start,cold_start")
    parser.add_argument("--trial-limit", type=int, default=None)
    parser.add_argument("--waypoint-limit", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(message)s")

    trajectory_ids = args.trajectory_ids.split(",") if args.trajectory_ids else None
    methods = args.methods.split(",")

    run_tier2(
        trajectory_ids=trajectory_ids,
        trial_category=args.trial_category,
        methods=methods,
        trial_limit=args.trial_limit,
        waypoint_limit=args.waypoint_limit,
        output_dir=args.output,
        make_plots=not args.no_plots,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
