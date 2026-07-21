"""Tier 1 pipeline: solves the point-IK benchmark with DLS and reports per-group metrics.

Runs every selected sample independently through algorithms.point_dls, then
evaluation.point_ik_metrics for overall/per-difficulty aggregates. Tier 1 never gates the
pipeline: a low success rate is recorded (``acceptance_status``) but Tier 2-4 still run
(see module docstring of pipelines.run_tier0_to_tier4).

Can be run standalone:
    python -m pipelines.run_tier1_point_dls --output results/tier1_only --point-sample-limit 30
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np

from algorithms.point_dls import load_point_ik_benchmark, run_point_dls
from algorithms.result_types import point_ik_results_to_dataframe
from evaluation.plotting import (
    plot_iterations_histogram,
    plot_orientation_error_cdf,
    plot_position_error_cdf,
    plot_runtime_histogram,
    plot_success_rate_by_group_bar,
)
from evaluation.point_ik_metrics import compute_point_ik_metrics
from kinematics.dls_solver import load_dls_config
from kinematics.model_loader import ModelContext, load_model_context
from pipelines._common import select_stratified_point_sample_ids
from utils.csv_utils import json_safe_scalar
from utils.result_logger import write_result_csv, write_result_json

logger = logging.getLogger(__name__)

DIFFICULTY_NAMES = {
    0: "near_target",
    1: "medium_target",
    2: "far_target",
    3: "large_orientation_change",
    4: "near_joint_limit",
    5: "near_singularity",
}


def _group_metrics_to_row(key: str, metrics) -> dict:
    row = {
        "group": key,
        "difficulty_name": DIFFICULTY_NAMES.get(int(key)) if key != "overall" and key.isdigit() else key,
        "sample_count": metrics.sample_count,
        "success_count": metrics.success_count,
        "success_rate": metrics.success_rate,
        "success_rate_wilson_lower": metrics.success_rate_wilson_ci.lower,
        "success_rate_wilson_upper": metrics.success_rate_wilson_ci.upper,
        "position_rmse_m": metrics.position_rmse_m,
        "position_p95_m": metrics.position_p95_m,
        "position_max_m": metrics.position_max_m,
        "orientation_rmse_deg": metrics.orientation_rmse_deg,
        "orientation_p95_deg": metrics.orientation_p95_deg,
        "orientation_max_deg": metrics.orientation_max_deg,
        "mean_iterations": metrics.mean_iterations,
        "p95_iterations": metrics.p95_iterations,
        "mean_solve_time_ms": metrics.mean_solve_time_ms,
        "p95_solve_time_ms": metrics.p95_solve_time_ms,
        "joint_limit_violation_rate": metrics.joint_limit_violation_rate,
    }
    return {k: json_safe_scalar(v) for k, v in row.items()}


def run_tier1(
    model_context: Optional[ModelContext] = None,
    dls_config: Optional[dict] = None,
    sample_ids: Optional[List[int]] = None,
    sample_limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    confidence_level: float = 0.95,
    minimum_success_rate: float = 0.95,
    make_plots: bool = True,
    show_progress: bool = True,
) -> dict:
    """Run the full Tier 1 point-IK pass and (if ``output_dir`` is given) write its output.

    If ``sample_ids`` is None and ``sample_limit`` is given, samples are selected with
    ``select_stratified_point_sample_ids`` so every difficulty group is represented.
    """
    model_context = model_context if model_context is not None else load_model_context()
    dls_config = dls_config if dls_config is not None else load_dls_config()
    benchmark = load_point_ik_benchmark()

    if sample_ids is None and sample_limit is not None:
        sample_ids = select_stratified_point_sample_ids(
            benchmark["sample_id"], benchmark["difficulty_id"], sample_limit
        )

    logger.info("tier1: running point DLS on %s samples", len(sample_ids) if sample_ids is not None else "all")
    results = run_point_dls(
        benchmark=benchmark,
        model_context=model_context,
        dls_config=dls_config,
        sample_ids=sample_ids,
        show_progress=show_progress,
    )

    results_df = point_ik_results_to_dataframe(results)
    metrics = compute_point_ik_metrics(results, confidence_level=confidence_level)

    overall = metrics["overall"]
    by_difficulty_rows = [_group_metrics_to_row(key, m) for key, m in metrics.items() if key != "overall"]

    failure_rows = []
    for key, m in metrics.items():
        for reason, count in m.failure_reason_counts.items():
            failure_rows.append({"group": key, "failure_reason": reason, "count": count})
    if not failure_rows:
        failure_rows = [{"group": "overall", "failure_reason": None, "count": 0}]

    failed_samples = [r for r in results if not r.success]
    failure_cases = [
        {
            "sample_id": r.sample_id,
            "difficulty_id": r.difficulty_id,
            "failure_reason": r.failure_reason,
            "position_error_m": r.position_error_m,
            "orientation_error_deg": r.orientation_error_deg,
            "iterations": r.iterations,
        }
        for r in failed_samples
    ]

    acceptance_passed = bool(overall.success_rate >= minimum_success_rate)
    overall_summary = {
        "execution_status": "completed",
        "acceptance_status": "passed" if acceptance_passed else "failed",
        "acceptance_criterion": {
            "name": "tier1_minimum_success_rate",
            "value": overall.success_rate,
            "threshold": minimum_success_rate,
            "passed": acceptance_passed,
            "unit": "fraction",
            "source": "project_criterion",
        },
        "sample_count": overall.sample_count,
        "success_count": overall.success_count,
        "success_rate": overall.success_rate,
        "success_rate_wilson_ci": {
            "lower": overall.success_rate_wilson_ci.lower,
            "upper": overall.success_rate_wilson_ci.upper,
            "confidence_level": overall.success_rate_wilson_ci.confidence_level,
        },
        "position_rmse_m": overall.position_rmse_m,
        "position_p95_m": overall.position_p95_m,
        "position_max_m": overall.position_max_m,
        "orientation_rmse_deg": overall.orientation_rmse_deg,
        "orientation_p95_deg": overall.orientation_p95_deg,
        "orientation_max_deg": overall.orientation_max_deg,
        "mean_iterations": overall.mean_iterations,
        "p95_iterations": overall.p95_iterations,
        "mean_solve_time_ms": overall.mean_solve_time_ms,
        "p95_solve_time_ms": overall.p95_solve_time_ms,
        "failure_reason_counts": overall.failure_reason_counts,
        "joint_limit_violation_rate": overall.joint_limit_violation_rate,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        write_result_csv(results_df, output_dir / "point_results.csv")
        write_result_json(overall_summary, output_dir / "metrics_overall.json")
        write_result_csv(_rows_to_df(by_difficulty_rows), output_dir / "metrics_by_difficulty.csv")
        write_result_csv(_rows_to_df(failure_rows), output_dir / "failure_reasons.csv")
        write_result_json(failure_cases, output_dir / "failure_cases.json")

        if make_plots:
            figures_dir = output_dir / "figures"
            plot_position_error_cdf(results_df["position_error_m"].to_numpy(), figures_dir / "position_error_cdf.png")
            plot_orientation_error_cdf(
                results_df["orientation_error_deg"].to_numpy(), figures_dir / "orientation_error_cdf.png"
            )
            plot_iterations_histogram(results_df["iterations"].to_numpy(), figures_dir / "iterations_histogram.png")
            plot_runtime_histogram(results_df["solve_time_ms"].to_numpy(), figures_dir / "runtime_histogram.png")

            difficulty_keys = sorted((k for k in metrics if k != "overall"), key=int)
            if difficulty_keys:
                labels = [DIFFICULTY_NAMES.get(int(k), k) for k in difficulty_keys]
                rates = [metrics[k].success_rate for k in difficulty_keys]
                plot_success_rate_by_group_bar(labels, rates, figures_dir / "success_by_difficulty.png")

    return {
        "results": results,
        "results_df": results_df,
        "metrics": metrics,
        "overall_summary": overall_summary,
    }


def _rows_to_df(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tier 1: point-IK DLS benchmark.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for tier1_point_dls/ content.")
    parser.add_argument("--point-sample-limit", type=int, default=None)
    parser.add_argument("--sample-ids", type=str, default=None, help="Comma-separated explicit sample ids.")
    parser.add_argument("--minimum-success-rate", type=float, default=0.95)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(message)s")

    sample_ids = [int(s) for s in args.sample_ids.split(",")] if args.sample_ids else None

    run_tier1(
        sample_ids=sample_ids,
        sample_limit=args.point_sample_limit,
        output_dir=args.output,
        minimum_success_rate=args.minimum_success_rate,
        make_plots=not args.no_plots,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
