"""CLI: Dataset v2 Phase 2.5 threshold calibration (near_joint_limit / near_singularity /
moderately_conditioned / regular).

    python -m pipelines.run_dataset_v2_threshold_calibration --seed N
        [--generic-pool-size N] [--singularity-pool-size N] [--report-json PATH]

Prints the calibration distribution summary and selected thresholds to stdout. Never writes a
candidate pool to disk; the optional ``--report-json`` only writes the small scalar summary (no
per-candidate arrays), and only to a path the caller supplies (never into a Dataset v2 root or the
repository). Generates no official Dataset v2 data.
"""

import argparse
import json
import sys

from dataset_v2.threshold_calibration import calibrate


def _report_dict(result) -> dict:
    return {
        "calibration_seed": result.calibration_seed,
        "generic_pool_seed": result.generic_pool_seed,
        "singularity_pool_seed": result.singularity_pool_seed,
        "generic_pool_size": result.generic_pool_size,
        "singularity_pool_size": result.singularity_pool_size,
        "normalized_margin_distribution": result.normalized_margin_distribution,
        "absolute_margin_distribution_rad": result.absolute_margin_distribution,
        "sigma_min_distribution_generic_pool": result.sigma_min_distribution_generic,
        "sigma_min_distribution_singularity_biased_pool": result.sigma_min_distribution_biased,
        "thresholds": {
            "near_joint_limit_normalized_margin": result.near_joint_limit_normalized_threshold,
            "near_joint_limit_absolute_margin_rad_diagnostic": result.near_joint_limit_absolute_threshold_rad_diagnostic,
            "near_singularity_sigma_min": result.near_singularity_sigma_threshold,
            "near_singularity_threshold_source": result.near_singularity_threshold_source,
            "moderately_conditioned_upper_bound": result.moderately_conditioned_upper_bound,
            "regular_min_sigma_min": result.regular_min_sigma_min,
            "regular_min_normalized_margin": result.regular_min_normalized_margin,
        },
        "candidate_counts": result.candidate_counts,
        "controlling_joint_normalized_near_limit": result.controlling_joint_normalized,
        "controlling_joint_absolute_near_limit_diagnostic": result.controlling_joint_absolute,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, help="Calibration seed (explicit; never a global default).")
    parser.add_argument("--generic-pool-size", type=int, default=None, help="Override the default generic candidate pool size.")
    parser.add_argument("--singularity-pool-size", type=int, default=None, help="Override the default singularity-biased candidate pool size.")
    parser.add_argument("--report-json", type=str, default=None, help="Optional path to write the scalar report as JSON (no per-candidate arrays, no pool data).")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    kwargs = {}
    if args.generic_pool_size is not None:
        kwargs["generic_pool_size"] = args.generic_pool_size
    if args.singularity_pool_size is not None:
        kwargs["singularity_pool_size"] = args.singularity_pool_size

    result = calibrate(args.seed, **kwargs)
    report = _report_dict(result)
    print(json.dumps(report, indent=2, sort_keys=False))

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=False)
            handle.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
