"""Tier 0 pipeline: FK/Jacobian/singularity validation and the mandatory Tier 0-4 gate.

Loads benchmarks/validation/{fk,jacobian,singularity}_test_states.npz, runs
evaluation.kinematics_validation against the compiled KR810 model, and writes per-sample CSVs
plus a summary.json carrying the gate_pass decision that run_tier0_to_tier4.py enforces before
Tier 1 is allowed to run.

Can be run standalone:
    python -m pipelines.run_tier0_kinematics --output results/tier0_only
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from evaluation.kinematics_validation import (
    DEFAULT_JACOBIAN_RELATIVE_ERROR_THRESHOLD,
    compute_gate_result,
    fk_validation_results_to_dataframe,
    jacobian_validation_results_to_dataframe,
    singularity_validation_results_to_dataframe,
    validate_fk_states,
    validate_jacobian_states,
    validate_singularity_states,
)
from evaluation.plotting import plot_iterations_histogram, plot_sigma_min_over_time
from kinematics.dls_solver import load_dls_config
from kinematics.model_loader import ModelContext, load_model_context
from utils.dataset_locator import FK_VALIDATION_PATH, JACOBIAN_VALIDATION_PATH, SINGULARITY_VALIDATION_PATH
from utils.npz_utils import load_npz
from utils.result_logger import write_result_csv, write_result_json

logger = logging.getLogger(__name__)


def _truncate(arrays: dict, limit: Optional[int]) -> dict:
    if limit is None:
        return arrays
    return {name: arr[:limit] for name, arr in arrays.items()}


def run_tier0(
    model_context: Optional[ModelContext] = None,
    dls_config: Optional[dict] = None,
    fk_sample_limit: Optional[int] = None,
    jacobian_sample_limit: Optional[int] = None,
    singularity_sample_limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    jacobian_relative_error_threshold: float = DEFAULT_JACOBIAN_RELATIVE_ERROR_THRESHOLD,
    make_plots: bool = True,
) -> dict:
    """Run the full Tier 0 validation pass and (if ``output_dir`` is given) write its output.

    Returns a dict with the gate result plus the per-check DataFrames, so
    ``pipelines.run_tier0_to_tier4`` can consume it without re-reading files from disk.
    """
    model_context = model_context if model_context is not None else load_model_context()
    dls_config = dls_config if dls_config is not None else load_dls_config()
    near_singular_threshold = float(dls_config["singularity_sigma_threshold"])

    fk_data = _truncate(load_npz(FK_VALIDATION_PATH), fk_sample_limit)
    jacobian_data = _truncate(load_npz(JACOBIAN_VALIDATION_PATH), jacobian_sample_limit)
    singularity_data = _truncate(load_npz(SINGULARITY_VALIDATION_PATH), singularity_sample_limit)

    logger.info(
        "tier0: validating %d FK, %d Jacobian, %d singularity states",
        fk_data["q_samples"].shape[0],
        jacobian_data["q_samples"].shape[0],
        singularity_data["q_samples"].shape[0],
    )

    fk_results = validate_fk_states(
        model_context, fk_data["q_samples"], fk_data["sample_id"], fk_data["group_id"]
    )
    jacobian_results = validate_jacobian_states(
        model_context,
        jacobian_data["q_samples"],
        jacobian_data["sample_id"],
        jacobian_data["group_id"],
        finite_difference_epsilon=jacobian_data["finite_difference_epsilon"],
        relative_error_threshold=jacobian_relative_error_threshold,
    )
    singularity_results = validate_singularity_states(
        model_context,
        singularity_data["q_samples"],
        singularity_data["sample_id"],
        singularity_data["group_id"],
        near_singular_threshold=near_singular_threshold,
    )

    gate = compute_gate_result(fk_results, jacobian_results, singularity_results, jacobian_relative_error_threshold)

    fk_df = fk_validation_results_to_dataframe(fk_results)
    jacobian_df = jacobian_validation_results_to_dataframe(jacobian_results)
    singularity_df = singularity_validation_results_to_dataframe(singularity_results)

    summary = {
        "gate_pass": gate.gate_pass,
        "gate_reasons": gate.reasons,
        "fk_sample_count": gate.fk_sample_count,
        "fk_rotation_failures": gate.fk_rotation_failures,
        "fk_determinism_failures": gate.fk_determinism_failures,
        "fk_reference_discrepancy_status": gate.fk_reference_discrepancy_status,
        "jacobian_sample_count": gate.jacobian_sample_count,
        "max_jacobian_relative_error": gate.max_jacobian_relative_error,
        "mean_jacobian_relative_error": gate.mean_jacobian_relative_error,
        "p95_jacobian_relative_error": gate.p95_jacobian_relative_error,
        "jacobian_relative_error_threshold": gate.jacobian_relative_error_threshold,
        "jacobian_over_threshold_count": gate.jacobian_over_threshold_count,
        "minimum_sigma_min": gate.minimum_sigma_min,
        "maximum_condition_number": gate.maximum_condition_number,
        "singularity_sample_count": gate.singularity_sample_count,
        "singularity_minimum_sigma_min": gate.singularity_minimum_sigma_min,
        "singularity_maximum_condition_number": gate.singularity_maximum_condition_number,
        "near_singular_threshold": near_singular_threshold,
        "note": "near-singular states are expected test data, not a gate failure criterion.",
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        write_result_csv(fk_df, output_dir / "fk_validation.csv")
        write_result_csv(jacobian_df, output_dir / "jacobian_validation.csv")
        write_result_csv(singularity_df, output_dir / "singularity_validation.csv")
        write_result_json(summary, output_dir / "summary.json")

        if make_plots:
            figures_dir = output_dir / "figures"
            plot_iterations_histogram(
                jacobian_df["relative_error"].to_numpy(),
                figures_dir / "jacobian_relative_error_histogram.png",
                title="Jacobian relative error",
            )
            plot_sigma_min_over_time(
                np.arange(len(singularity_df)),
                singularity_df["sigma_min"].to_numpy(),
                figures_dir / "singularity_sigma_min.png",
                title="Singularity validation states",
            )

    return {
        "gate": gate,
        "summary": summary,
        "fk_results": fk_df,
        "jacobian_results": jacobian_df,
        "singularity_results": singularity_df,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tier 0: FK/Jacobian/singularity validation gate.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for tier0_kinematics/ content.")
    parser.add_argument("--fk-samples", type=int, default=None, help="Limit FK validation states (default: all).")
    parser.add_argument("--jacobian-samples", type=int, default=None, help="Limit Jacobian validation states.")
    parser.add_argument("--singularity-samples", type=int, default=None, help="Limit singularity validation states.")
    parser.add_argument(
        "--jacobian-error-threshold", type=float, default=DEFAULT_JACOBIAN_RELATIVE_ERROR_THRESHOLD
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(message)s")

    result = run_tier0(
        fk_sample_limit=args.fk_samples,
        jacobian_sample_limit=args.jacobian_samples,
        singularity_sample_limit=args.singularity_samples,
        output_dir=args.output,
        jacobian_relative_error_threshold=args.jacobian_error_threshold,
        make_plots=not args.no_plots,
    )
    logger.info("tier0: gate_pass=%s", result["gate"].gate_pass)
    return 0 if result["gate"].gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
