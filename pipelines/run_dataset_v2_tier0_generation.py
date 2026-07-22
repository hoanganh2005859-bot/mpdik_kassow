"""CLI: Dataset v2 Tier 0 generation/validation (FK/Jacobian/singularity validation states).

    python -m pipelines.run_dataset_v2_tier0_generation --dataset-root PATH [--master-seed N]
        [--overwrite] [--validate-only] [--dry-run]

The locked full mode is 1000 FK / 1000 Jacobian / 600 singularity states
(``specs/DLS_DATASET_V2_SPEC.md`` section B). ``--fk-count``/``--jacobian-count``/
``--singularity-count`` and the candidate-pool-size overrides exist for tests/smoke runs only.
"""

import argparse
import sys
from pathlib import Path

from dataset_v2.tier0_generation import run_tier0_generation
from dataset_v2.tier0_validation import validate_tier0
from utils.exceptions import ModelConfigurationError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Explicit Dataset v2 root (must already hold a scaffold).")
    parser.add_argument("--master-seed", type=int, default=None, help="Override configs/seed_policy.json's recorded master seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow regenerating existing Tier 0 v2 output.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing Tier 0 v2 output; generate nothing.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be generated; write nothing.")
    parser.add_argument("--fk-count", type=int, default=None, help="Test/smoke override only; locked full mode is 1000.")
    parser.add_argument("--jacobian-count", type=int, default=None, help="Test/smoke override only; locked full mode is 1000.")
    parser.add_argument("--singularity-count", type=int, default=None, help="Test/smoke override only; locked full mode is 600.")
    parser.add_argument("--jacobian-candidate-pool-size", type=int, default=None, help="Test/smoke override only.")
    parser.add_argument("--singularity-candidate-pool-size", type=int, default=None, help="Test/smoke override only.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.validate_only:
        full_counts = not any((args.fk_count, args.jacobian_count, args.singularity_count))
        expected_totals = None
        if not full_counts:
            expected_totals = {
                key: value
                for key, value in (
                    ("fk", args.fk_count),
                    ("jacobian", args.jacobian_count),
                    ("singularity", args.singularity_count),
                )
                if value is not None
            }
        try:
            report = validate_tier0(args.dataset_root, full_counts=full_counts, expected_totals=expected_totals)
        except (ModelConfigurationError, FileNotFoundError) as exc:
            print(f"[tier0-validate] error: {exc}", file=sys.stderr)
            return 2

        for reason in report.reasons:
            print(f"[tier0-validate] FAIL: {reason}", file=sys.stderr)
        print(
            f"[tier0-validate] fk={report.fk_count} jacobian={report.jacobian_count} "
            f"singularity={report.singularity_count} max_jacobian_relative_error="
            f"{report.max_jacobian_relative_error:.3e} passed={report.passed}"
        )
        return 0 if report.passed else 1

    try:
        result = run_tier0_generation(
            args.dataset_root,
            master_seed=args.master_seed,
            overwrite=args.overwrite,
            fk_total=args.fk_count,
            jacobian_total=args.jacobian_count,
            singularity_total=args.singularity_count,
            jacobian_candidate_pool_size=args.jacobian_candidate_pool_size,
            singularity_candidate_pool_size=args.singularity_candidate_pool_size,
            dry_run=args.dry_run,
        )
    except FileExistsError as exc:
        print(f"[tier0-generate] error: {exc}", file=sys.stderr)
        return 2
    except ModelConfigurationError as exc:
        print(f"[tier0-generate] error: {exc}", file=sys.stderr)
        return 2

    if result.dry_run:
        print(
            f"[tier0-generate] dry-run: would write {result.fk_total} FK / {result.jacobian_total} "
            f"Jacobian / {result.singularity_total} singularity states under {result.tier0_validation_dir} "
            f"(full_locked_counts={result.full_locked_counts})"
        )
        return 0

    print(
        f"[tier0-generate] wrote {result.fk_total} FK, {result.jacobian_total} Jacobian, "
        f"{result.singularity_total} singularity states to {result.tier0_validation_dir} "
        f"(full_locked_counts={result.full_locked_counts})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
