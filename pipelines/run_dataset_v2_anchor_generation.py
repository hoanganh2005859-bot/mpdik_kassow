"""CLI: Dataset v2 anchor generation/validation.

    python -m pipelines.run_dataset_v2_anchor_generation --dataset-root PATH [--master-seed N]
        [--overwrite] [--validate-only] [--dry-run]

The locked full mode is 12 anchors (6 regular/3 near_limit/3 near_singular), split 4/4/4
(2 regular + 1 near_limit + 1 near_singular per split) per
``specs/DLS_DATASET_V2_SPEC.md`` section G. The candidate-pool-size overrides exist for
tests/smoke runs only and never relax a classification threshold.
"""

import argparse
import sys
from pathlib import Path

from dataset_v2.anchor_generation import run_anchor_generation
from dataset_v2.anchor_validation import validate_anchors
from utils.exceptions import ModelConfigurationError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Explicit Dataset v2 root (must already hold a scaffold).")
    parser.add_argument("--master-seed", type=int, default=None, help="Override configs/seed_policy.json's recorded master seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow regenerating existing anchor v2 output.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing anchor v2 output; generate nothing.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be generated; write nothing.")
    parser.add_argument("--regular-pool-size", type=int, default=None, help="Test/smoke override only.")
    parser.add_argument("--near-limit-pool-size", type=int, default=None, help="Test/smoke override only.")
    parser.add_argument("--singularity-pool-size", type=int, default=None, help="Test/smoke override only.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.validate_only:
        full_counts = args.regular_pool_size is None and args.near_limit_pool_size is None and args.singularity_pool_size is None
        try:
            report = validate_anchors(args.dataset_root, full_counts=full_counts)
        except (ModelConfigurationError, FileNotFoundError) as exc:
            print(f"[anchor-validate] error: {exc}", file=sys.stderr)
            return 2

        for reason in report.reasons:
            print(f"[anchor-validate] FAIL: {reason}", file=sys.stderr)
        print(f"[anchor-validate] total={report.total_anchors} passed={report.passed}")
        return 0 if report.passed else 1

    try:
        result = run_anchor_generation(
            args.dataset_root,
            master_seed=args.master_seed,
            overwrite=args.overwrite,
            regular_pool_size=args.regular_pool_size,
            near_limit_biased_pool_size=args.near_limit_pool_size,
            singularity_biased_pool_size=args.singularity_pool_size,
            dry_run=args.dry_run,
        )
    except FileExistsError as exc:
        print(f"[anchor-generate] error: {exc}", file=sys.stderr)
        return 2
    except ModelConfigurationError as exc:
        print(f"[anchor-generate] error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[anchor-generate] error: {exc}", file=sys.stderr)
        return 2

    if result.dry_run:
        print(
            f"[anchor-generate] dry-run: would write {result.total_anchors} anchors to "
            f"{result.anchors_dir} (class_counts={result.class_counts}, split_counts={result.split_counts})"
        )
        return 0

    print(
        f"[anchor-generate] wrote {result.total_anchors} anchors to {result.anchors_dir} "
        f"(class_counts={result.class_counts}, split_counts={result.split_counts}, "
        f"class_split_counts={result.class_split_counts})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
