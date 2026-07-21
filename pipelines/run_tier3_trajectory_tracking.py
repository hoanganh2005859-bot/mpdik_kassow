"""Tier 3 pipeline: Cartesian trajectory tracking accuracy from Tier 2 joint solutions.

Reads tier2_sequential_dls/waypoint_results.csv (or an in-process DataFrame from
pipelines.run_tier2_sequential_dls) and never re-runs the DLS solver: every metric here is a
deterministic function of the already-solved target/actual pose columns.

ISO 9283-inspired accuracy/repeatability (ATp/RTp) is computed only from repeatability trials,
grouped by (trajectory_id, method, speed_scale) -- never mixing robustness trials, methods, or
speed scales (see evaluation.iso9283_metrics module docstring for why). Bootstrap confidence
intervals are computed at the trial level (one RMSE value per trial), never by resampling
individual waypoints.

Can be run standalone:
    python -m pipelines.run_tier3_trajectory_tracking --tier2-dir results/run/tier2_sequential_dls --output results/tier3_only
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from evaluation.confidence_intervals import bootstrap_confidence_interval
from evaluation.cross_track_metrics import compute_cross_track_metrics
from evaluation.iso9283_metrics import compute_path_accuracy, compute_path_repeatability
from evaluation.plotting import (
    plot_orientation_error_over_time,
    plot_position_error_over_time,
    plot_target_vs_actual_3d,
    plot_xyz_tracking,
)
from evaluation.trajectory_metrics import compute_trajectory_tracking_metrics
from utils.csv_utils import json_safe_scalar
from utils.dataset_locator import TRAJECTORY_MANIFEST_PATH, TRAJECTORY_TRIALS_PATH
from utils.result_logger import write_result_csv, write_result_json

logger = logging.getLogger(__name__)

_POS_COLS = {"target": ["target_position_x", "target_position_y", "target_position_z"],
             "actual": ["actual_position_x", "actual_position_y", "actual_position_z"]}
_QUAT_COLS = {
    "target": ["target_quaternion_qw", "target_quaternion_qx", "target_quaternion_qy", "target_quaternion_qz"],
    "actual": ["actual_quaternion_qw", "actual_quaternion_qx", "actual_quaternion_qy", "actual_quaternion_qz"],
}


def _positions(df: pd.DataFrame, which: str) -> np.ndarray:
    return df[_POS_COLS[which]].to_numpy(dtype=np.float64)


def _quaternions(df: pd.DataFrame, which: str) -> np.ndarray:
    return df[_QUAT_COLS[which]].to_numpy(dtype=np.float64)


def run_tier3(
    waypoint_results_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 10000,
    seed: int = 42,
    make_plots: bool = True,
) -> dict:
    """Compute Tier 3 tracking/cross-track/ISO9283-inspired metrics from Tier 2 waypoint results."""
    manifest_df = pd.read_csv(TRAJECTORY_MANIFEST_PATH)
    closed_path_by_trajectory = dict(zip(manifest_df["trajectory_id"], manifest_df["closed_path"]))
    trials_df = pd.read_csv(TRAJECTORY_TRIALS_PATH)
    speed_scale_by_trial = dict(zip(trials_df["trial_id"], trials_df["speed_scale"]))

    tracking_rows = []
    cross_track_rows = []
    representative_groups = {}

    for (trial_id, method), group in waypoint_results_df.groupby(["trial_id", "method"], sort=True):
        group = group.sort_values("waypoint_id")
        trajectory_id = group["trajectory_id"].iloc[0]
        target_positions = _positions(group, "target")
        actual_positions = _positions(group, "actual")
        target_quats = _quaternions(group, "target")
        actual_quats = _quaternions(group, "actual")

        tracking = compute_trajectory_tracking_metrics(target_positions, actual_positions, target_quats, actual_quats)
        closed_path = bool(closed_path_by_trajectory.get(trajectory_id, False))
        cross_track = compute_cross_track_metrics(actual_positions, target_positions, closed_path=closed_path)

        tracking_rows.append({
            "trial_id": trial_id, "trajectory_id": trajectory_id, "method": method,
            "waypoint_count": tracking.position.count,
            "position_rmse_m": tracking.position.rmse_m,
            "position_mae_m": tracking.position.mae_m,
            "position_median_m": tracking.position.median_m,
            "position_p95_m": tracking.position.p95_m,
            "position_max_m": tracking.position.max_m,
            "endpoint_error_m": tracking.position.endpoint_error_m,
            "start_point_error_m": tracking.position.start_point_error_m,
            "centroid_offset_m": tracking.position.centroid_offset_m,
            "target_path_length_m": tracking.position.target_path_length_m,
            "actual_path_length_m": tracking.position.actual_path_length_m,
            "path_length_ratio": tracking.position.path_length_ratio,
            "orientation_rmse_deg": tracking.orientation.rmse_deg,
            "orientation_mae_deg": tracking.orientation.mae_deg,
            "orientation_median_deg": tracking.orientation.median_deg,
            "orientation_p95_deg": tracking.orientation.p95_deg,
            "orientation_max_deg": tracking.orientation.max_deg,
        })

        cross_track_rows.append({
            "trial_id": trial_id, "trajectory_id": trajectory_id, "method": method,
            "point_count": cross_track.summary.point_count,
            "cross_track_rmse_m": cross_track.summary.cross_track_rmse_m,
            "cross_track_mae_m": cross_track.summary.cross_track_mae_m,
            "cross_track_p95_m": cross_track.summary.cross_track_p95_m,
            "cross_track_max_m": cross_track.summary.cross_track_max_m,
            "total_path_length_m": cross_track.summary.total_path_length_m,
            "final_progress_ratio": cross_track.summary.final_progress_ratio,
            "backward_progress_count": cross_track.summary.backward_progress_count,
            "synchronized_along_track_rmse_m": cross_track.summary.synchronized_along_track_rmse_m,
        })

        rep_key = f"{trajectory_id}::{method}"
        if rep_key not in representative_groups:
            representative_groups[rep_key] = group

    _TRACKING_COLUMNS = [
        "trial_id", "trajectory_id", "method", "waypoint_count",
        "position_rmse_m", "position_mae_m", "position_median_m", "position_p95_m", "position_max_m",
        "endpoint_error_m", "start_point_error_m", "centroid_offset_m",
        "target_path_length_m", "actual_path_length_m", "path_length_ratio",
        "orientation_rmse_deg", "orientation_mae_deg", "orientation_median_deg",
        "orientation_p95_deg", "orientation_max_deg",
    ]
    _CROSS_TRACK_COLUMNS = [
        "trial_id", "trajectory_id", "method", "point_count",
        "cross_track_rmse_m", "cross_track_mae_m", "cross_track_p95_m", "cross_track_max_m",
        "total_path_length_m", "final_progress_ratio", "backward_progress_count",
        "synchronized_along_track_rmse_m",
    ]
    tracking_df = pd.DataFrame(tracking_rows, columns=_TRACKING_COLUMNS)
    cross_track_df = pd.DataFrame(cross_track_rows, columns=_CROSS_TRACK_COLUMNS)

    # --- ISO 9283-inspired ATp/RTp: repeatability trials only, grouped by (trajectory_id, method, speed_scale)
    iso_rows = []
    repeatability_df = waypoint_results_df[waypoint_results_df["trial_category"] == "repeatability"]
    if not repeatability_df.empty:
        repeatability_df = repeatability_df.copy()
        repeatability_df["speed_scale"] = repeatability_df["trial_id"].map(speed_scale_by_trial)
        group_keys = repeatability_df[["trajectory_id", "method", "speed_scale"]].drop_duplicates()
        for _, key_row in group_keys.iterrows():
            trajectory_id, method, speed_scale = key_row["trajectory_id"], key_row["method"], key_row["speed_scale"]
            subset = repeatability_df[
                (repeatability_df["trajectory_id"] == trajectory_id)
                & (repeatability_df["method"] == method)
                & (repeatability_df["speed_scale"] == speed_scale)
            ]
            by_repeat = {}
            for trial_id, trial_group in subset.groupby("trial_id"):
                trial_group = trial_group.sort_values("waypoint_id")
                by_repeat[trial_id] = trial_group

            if len(by_repeat) < 2:
                continue

            reference_positions = None
            actual_stack = []
            consistent = True
            for trial_id, trial_group in sorted(by_repeat.items()):
                target = _positions(trial_group, "target")
                if reference_positions is None:
                    reference_positions = target
                elif target.shape != reference_positions.shape or not np.allclose(target, reference_positions, atol=1e-9):
                    consistent = False
                    break
                actual_stack.append(_positions(trial_group, "actual"))

            if not consistent:
                logger.warning(
                    "tier3: skipping ISO9283 group (trajectory_id=%s, method=%s, speed_scale=%s): "
                    "mismatched commanded path across repeats",
                    trajectory_id, method, speed_scale,
                )
                continue

            repeated_actual = np.stack(actual_stack, axis=0)
            accuracy = compute_path_accuracy(reference_positions, repeated_actual)
            repeatability = compute_path_repeatability(repeated_actual)

            iso_rows.append({
                "trajectory_id": trajectory_id, "method": method, "speed_scale": speed_scale,
                "n_repeats": repeatability.n_repeats,
                "atp_m": accuracy.atp_m, "atp_waypoint_index": accuracy.atp_waypoint_index,
                "mean_deviation_m": accuracy.mean_deviation_m, "rmse_deviation_m": accuracy.rmse_deviation_m,
                "p95_deviation_m": accuracy.p95_deviation_m,
                "rtp_m": repeatability.rtp_m, "rtp_waypoint_index": repeatability.rtp_waypoint_index,
                "maximum_radial_spread_m": repeatability.maximum_radial_spread_m,
                "warning": repeatability.warning,
            })

    _ISO_COLUMNS = [
        "trajectory_id", "method", "speed_scale", "n_repeats",
        "atp_m", "atp_waypoint_index", "mean_deviation_m", "rmse_deviation_m", "p95_deviation_m",
        "rtp_m", "rtp_waypoint_index", "maximum_radial_spread_m", "warning",
    ]
    iso_df = pd.DataFrame(iso_rows, columns=_ISO_COLUMNS)

    # --- Bootstrap confidence intervals over trial-level position/orientation RMSE, per method
    ci_rows = []
    for method, method_df in tracking_df.groupby("method"):
        for metric_col in ("position_rmse_m", "orientation_rmse_deg"):
            ci = bootstrap_confidence_interval(
                method_df[metric_col].to_numpy(),
                statistic="mean",
                confidence_level=confidence_level,
                n_resamples=bootstrap_resamples,
                seed=seed,
            )
            ci_rows.append({
                "method": method, "metric": metric_col,
                "estimate": ci.estimate, "lower": ci.lower, "upper": ci.upper,
                "n_resamples": ci.n_resamples, "confidence_level": ci.confidence_level, "sample_size": ci.sample_size,
            })
    _CI_COLUMNS = ["method", "metric", "estimate", "lower", "upper", "n_resamples", "confidence_level", "sample_size"]
    ci_df = pd.DataFrame(ci_rows, columns=_CI_COLUMNS)

    if iso_df.empty:
        logger.warning(
            "tier3: iso9283_metrics.csv has no rows -- no (trajectory_id, method, speed_scale) group in this "
            "selection had >= 2 repeatability repeats (see evaluation.iso9283_metrics.MIN_REPEATS_FOR_STD); "
            "this is expected under a small --trial-limit, not a computation failure."
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        write_result_csv(tracking_df, output_dir / "trajectory_metrics.csv")
        write_result_csv(cross_track_df, output_dir / "cross_track_metrics.csv")
        write_result_csv(iso_df, output_dir / "iso9283_metrics.csv")
        write_result_csv(ci_df, output_dir / "confidence_intervals.csv")

        if make_plots:
            figures_dir = output_dir / "figures"
            for rep_key, group in sorted(representative_groups.items()):
                safe_name = rep_key.replace("::", "_")
                target_positions = _positions(group, "target")
                actual_positions = _positions(group, "actual")
                time_s = group["time_s"].to_numpy(dtype=np.float64)
                position_errors = group["position_error_m"].to_numpy(dtype=np.float64)
                orientation_errors = group["orientation_error_deg"].to_numpy(dtype=np.float64)

                plot_target_vs_actual_3d(target_positions, actual_positions, figures_dir / f"target_vs_actual_3d_{safe_name}.png", title=rep_key)
                plot_xyz_tracking(time_s, target_positions, actual_positions, figures_dir / f"xyz_tracking_{safe_name}.png", title=rep_key)
                plot_position_error_over_time(time_s, position_errors, figures_dir / f"position_error_{safe_name}.png", title=rep_key)
                plot_orientation_error_over_time(time_s, orientation_errors, figures_dir / f"orientation_error_{safe_name}.png", title=rep_key)

            if not tracking_df.empty:
                from evaluation.plotting import plot_position_error_cdf
                plot_position_error_cdf(
                    waypoint_results_df["position_error_m"].to_numpy(dtype=np.float64),
                    figures_dir / "trajectory_position_error_cdf.png",
                )

    return {
        "tracking_df": tracking_df,
        "cross_track_df": cross_track_df,
        "iso9283_df": iso_df,
        "confidence_intervals_df": ci_df,
        "representative_trial_selection_policy": "first trial encountered per (trajectory_id, method) in sorted trial_id order",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tier 3: Cartesian trajectory tracking metrics from Tier 2 output.")
    parser.add_argument("--tier2-dir", required=True, type=Path, help="Directory containing tier2 waypoint_results.csv.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(message)s")

    waypoint_df = pd.read_csv(args.tier2_dir / "waypoint_results.csv")
    run_tier3(
        waypoint_df,
        output_dir=args.output,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        make_plots=not args.no_plots,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
