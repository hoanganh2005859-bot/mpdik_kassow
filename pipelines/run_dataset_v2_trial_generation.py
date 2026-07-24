"""CLI: Dataset v2 trial generation/validation (Phase 7).

    python -m pipelines.run_dataset_v2_trial_generation --dataset-root PATH
        [--master-seed N] [--overwrite] [--validate-only] [--dry-run] [--progress]
        [--trajectory-id ID ...] [--candidate-pool-size N] [--no-rebuild-catalog]

The locked full mode is 630 trials (3 per trajectory -- easy/medium/hard -- across all 210 core +
random-challenge trajectories) per ``specs/DLS_DATASET_V2_SPEC.md`` section J. Each trial's
``q_initial`` is drawn independently from the operational joint limits and classified only against
its trajectory's first canonical target pose; ``q_reference`` is never used to construct a trial.
The ``--trajectory-id``/``--candidate-pool-size`` overrides exist for tests/smoke runs only and
never change the locked 630-trial full mode.

No DLS is run in this phase.
"""

import argparse
import sys
from pathlib import Path

from dataset_v2.trajectory_catalog import build_combined_catalog, validate_combined_catalog
from dataset_v2.trial_generation import run_trial_generation
from dataset_v2.trial_validation import validate_trials
from utils.exceptions import ModelConfigurationError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Explicit Dataset v2 root (must already hold a scaffold + trajectories).")
    parser.add_argument("--master-seed", type=int, default=None, help="Override configs/seed_policy.json's recorded master seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow regenerating existing trial v2 output.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing trial v2 output; generate nothing.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be generated; write nothing.")
    parser.add_argument("--trajectory-id", action="append", default=None, dest="trajectory_ids", help="Test/smoke override only: restrict to this trajectory (repeatable).")
    parser.add_argument("--candidate-pool-size", type=int, default=None, help="Test/smoke override only: uniform sub-pool size for the q_initial candidate pool.")
    parser.add_argument("--no-rebuild-catalog", action="store_true", help="Do not rebuild the combined trajectory catalog before generating trials.")
    parser.add_argument("--progress", action="store_true", help="Print progress to stderr.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.validate_only:
        full_counts = args.trajectory_ids is None and args.candidate_pool_size is None
        try:
            catalog_report = validate_combined_catalog(args.dataset_root, full_counts=full_counts)
            report = validate_trials(args.dataset_root, full_counts=full_counts)
        except (ModelConfigurationError, FileNotFoundError) as exc:
            print(f"[trial-validate] error: {exc}", file=sys.stderr)
            return 2
        for reason in catalog_report.reasons:
            print(f"[trial-validate] CATALOG FAIL: {reason}", file=sys.stderr)
        for reason in report.reasons:
            print(f"[trial-validate] FAIL: {reason}", file=sys.stderr)
        print(
            f"[trial-validate] total={report.total_trials} split_counts={report.split_counts} "
            f"difficulty_counts={report.difficulty_counts} family_counts={report.family_counts} "
            f"catalog_passed={catalog_report.passed} passed={report.passed}"
        )
        return 0 if (report.passed and catalog_report.passed) else 1

    try:
        if not args.no_rebuild_catalog and not args.dry_run:
            build_combined_catalog(args.dataset_root, overwrite=True)
        result = run_trial_generation(
            args.dataset_root,
            master_seed=args.master_seed,
            overwrite=args.overwrite,
            trajectory_ids=args.trajectory_ids,
            pool_scale_override=args.candidate_pool_size,
            dry_run=args.dry_run,
            progress=args.progress,
            rebuild_catalog=not args.no_rebuild_catalog,
        )
    except FileExistsError as exc:
        print(f"[trial-generate] error: {exc}", file=sys.stderr)
        return 2
    except ModelConfigurationError as exc:
        print(f"[trial-generate] error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"[trial-generate] error: {exc}", file=sys.stderr)
        return 2

    if result.dry_run:
        print(
            f"[trial-generate] dry-run: would write {result.total_trials} trials to {result.trials_dir} "
            f"(split_counts={result.split_counts}, difficulty_counts={result.difficulty_counts})"
        )
        return 0

    print(
        f"[trial-generate] wrote {result.total_trials} trials to {result.trials_dir} "
        f"(split_counts={result.split_counts}, difficulty_counts={result.difficulty_counts}, "
        f"family_counts={result.family_counts}, separation={result.separation_statistics})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
