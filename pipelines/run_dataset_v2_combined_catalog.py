"""CLI: build/validate the Dataset v2 combined trajectory catalog (Phase 7).

    python -m pipelines.run_dataset_v2_combined_catalog --dataset-root PATH [--overwrite]
        [--validate-only]

Builds ``trajectories/combined_trajectory_manifest.csv`` -- the deterministic 210-row union of the
core (120) and random-challenge (90) per-family manifests -- or validates an existing one.
"""

import argparse
import sys
from pathlib import Path

from dataset_v2.trajectory_catalog import build_combined_catalog, validate_combined_catalog
from utils.exceptions import ModelConfigurationError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild an existing combined catalog.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate an existing combined catalog.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.validate_only:
        try:
            report = validate_combined_catalog(args.dataset_root)
        except (ModelConfigurationError, FileNotFoundError) as exc:
            print(f"[combined-catalog] error: {exc}", file=sys.stderr)
            return 2
        for reason in report.reasons:
            print(f"[combined-catalog] FAIL: {reason}", file=sys.stderr)
        print(f"[combined-catalog] total={report.total} family_counts={report.family_counts} split_counts={report.split_counts} passed={report.passed}")
        return 0 if report.passed else 1

    try:
        path = build_combined_catalog(args.dataset_root, overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"[combined-catalog] error: {exc}", file=sys.stderr)
        return 2
    except (ModelConfigurationError, FileNotFoundError, ValueError) as exc:
        print(f"[combined-catalog] error: {exc}", file=sys.stderr)
        return 2
    report = validate_combined_catalog(args.dataset_root)
    for reason in report.reasons:
        print(f"[combined-catalog] FAIL: {reason}", file=sys.stderr)
    print(f"[combined-catalog] wrote {path} total={report.total} passed={report.passed}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
