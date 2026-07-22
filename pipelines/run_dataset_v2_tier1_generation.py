"""CLI: Dataset v2 Tier 1 Point-IK generation/validation.

    python -m pipelines.run_dataset_v2_tier1_generation --dataset-root PATH [--master-seed N]
        [--overwrite] [--validate-only] [--dry-run] [--sample-limit-per-group N]

The locked full mode is 6,000 samples (6 difficulty groups x 1,000), split
development/validation/frozen_test 1,200/1,200/3,600 (200/200/600 per group), per
``specs/DLS_DATASET_V2_SPEC.md`` section B. ``--sample-limit-per-group`` (and the pool-size
override) exist for tests/smoke runs only and never relax a difficulty threshold.
"""

import argparse
import sys
from pathlib import Path

from dataset_v2.point_ik_generation import run_point_ik_generation
from dataset_v2.point_ik_validation import validate_point_ik
from utils.exceptions import ModelConfigurationError


def _split_sizes_for_limit(sample_limit_per_group: int) -> dict:
    if sample_limit_per_group % 5 != 0:
        raise SystemExit(
            f"--sample-limit-per-group must be divisible by 5 (200:200:600 = 1:1:3 ratio), "
            f"got {sample_limit_per_group}"
        )
    unit = sample_limit_per_group // 5
    return {"development": unit, "validation": unit, "frozen_test": sample_limit_per_group - 2 * unit}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Explicit Dataset v2 root (must already hold a scaffold).")
    parser.add_argument("--master-seed", type=int, default=None, help="Override configs/seed_policy.json's recorded master seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow regenerating existing Point-IK v2 output.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing Point-IK v2 output; generate nothing.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be generated; write nothing.")
    parser.add_argument(
        "--sample-limit-per-group", type=int, default=None, help="Test/smoke override only; locked full mode is 1000/group."
    )
    parser.add_argument("--pool-size", type=int, default=None, help="Test/smoke override only for the generic candidate pool size.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.validate_only:
        full_counts = args.sample_limit_per_group is None
        expected_samples_per_group = args.sample_limit_per_group if not full_counts else None
        try:
            report = validate_point_ik(args.dataset_root, full_counts=full_counts, expected_samples_per_group=expected_samples_per_group)
        except (ModelConfigurationError, FileNotFoundError) as exc:
            print(f"[point-ik-validate] error: {exc}", file=sys.stderr)
            return 2

        for reason in report.reasons:
            print(f"[point-ik-validate] FAIL: {reason}", file=sys.stderr)
        print(f"[point-ik-validate] total={report.total_samples} passed={report.passed}")
        return 0 if report.passed else 1

    split_sizes_per_group = _split_sizes_for_limit(args.sample_limit_per_group) if args.sample_limit_per_group is not None else None

    try:
        result = run_point_ik_generation(
            args.dataset_root,
            master_seed=args.master_seed,
            overwrite=args.overwrite,
            samples_per_group=args.sample_limit_per_group,
            pool_size=args.pool_size,
            split_sizes_per_group=split_sizes_per_group,
            dry_run=args.dry_run,
        )
    except FileExistsError as exc:
        print(f"[point-ik-generate] error: {exc}", file=sys.stderr)
        return 2
    except ModelConfigurationError as exc:
        print(f"[point-ik-generate] error: {exc}", file=sys.stderr)
        return 2

    if result.dry_run:
        print(
            f"[point-ik-generate] dry-run: would write {result.total_samples} samples to "
            f"{result.tier1_point_ik_dir} (full_locked_counts={result.full_locked_counts})"
        )
        return 0

    print(
        f"[point-ik-generate] wrote {result.total_samples} samples to {result.tier1_point_ik_dir} "
        f"(group_counts={result.group_counts}, split_counts={result.split_counts}, "
        f"full_locked_counts={result.full_locked_counts})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
