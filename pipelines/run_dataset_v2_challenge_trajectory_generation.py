"""CLI: Dataset v2 random-challenge trajectory generation/validation (Phase 6).

    python -m pipelines.run_dataset_v2_challenge_trajectory_generation --dataset-root PATH
        [--master-seed N] [--overwrite] [--validate-only] [--dry-run] [--progress]

The locked full mode is 90 random-challenge trajectories (6 families x 5 per split x 3 splits) per
``specs/DLS_DATASET_V2_SPEC.md`` section I, each with exactly 400 canonical waypoints (36,000
canonical challenge poses). The ``--family``/``--split``/``--per-family-per-split``/
``--source-waypoints``/``--candidate-pool-size`` overrides exist for tests/smoke runs only and
never change the locked 400-waypoint canonical count.

Reachability (spec section H.1/I, Phase 5.1/5.4): a waypoint is accepted only when a reference
configuration exists whose *independently recomputed* FK reproduces the target pose within Dataset
v2's own strict tolerances (``configs/generation_reachability_config.json``: 1e-4 m / 0.01 deg) --
never because the numerical IK engine reported success, and never against Dataset v1's DLS baseline
thresholds. Both the canonical and the high-resolution source path are verified.
"""

import argparse
import sys
from pathlib import Path

from dataset_v2.challenge_trajectory_generation import run_challenge_trajectory_generation
from dataset_v2.challenge_trajectory_validation import validate_challenge_trajectories
from utils.exceptions import ModelConfigurationError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Explicit Dataset v2 root (must already hold a scaffold).")
    parser.add_argument("--master-seed", type=int, default=None, help="Override configs/seed_policy.json's recorded master seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow regenerating existing challenge trajectory v2 output.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing challenge trajectory v2 output; generate nothing.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be generated; write nothing.")
    parser.add_argument("--family", action="append", default=None, dest="families", help="Test/smoke override only: restrict to this challenge family (repeatable).")
    parser.add_argument("--split", action="append", default=None, dest="splits", choices=["development", "validation", "frozen_test"], help="Test/smoke override only: restrict to this split (repeatable).")
    parser.add_argument("--per-family-per-split", type=int, default=None, help="Test/smoke override only: trajectories per family per split (locked full mode = 5).")
    parser.add_argument("--source-waypoints", type=int, default=None, help="Test/smoke override only: high-resolution source waypoint count. Never affects the locked 400 canonical count.")
    parser.add_argument("--candidate-pool-size", type=int, default=None, help="Test/smoke override only: feasibility-aware candidate pool size per family per split.")
    parser.add_argument("--progress", action="store_true", help="Print per-(family,split) progress to stderr.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.validate_only:
        full_counts = (
            args.families is None
            and args.splits is None
            and args.per_family_per_split is None
            and args.source_waypoints is None
            and args.candidate_pool_size is None
        )
        try:
            report = validate_challenge_trajectories(args.dataset_root, full_counts=full_counts)
        except (ModelConfigurationError, FileNotFoundError) as exc:
            print(f"[challenge-validate] error: {exc}", file=sys.stderr)
            return 2

        for reason in report.reasons:
            print(f"[challenge-validate] FAIL: {reason}", file=sys.stderr)
        print(
            f"[challenge-validate] total={report.total_trajectories} canonical_poses={report.canonical_poses_total} "
            f"combined_total={report.combined_trajectories_total} combined_poses={report.combined_canonical_poses_total} "
            f"passed={report.passed}"
        )
        return 0 if report.passed else 1

    try:
        result = run_challenge_trajectory_generation(
            args.dataset_root,
            master_seed=args.master_seed,
            overwrite=args.overwrite,
            families=args.families,
            splits=args.splits,
            per_family_per_split=args.per_family_per_split,
            source_waypoint_count=args.source_waypoints,
            candidate_pool_size=args.candidate_pool_size,
            dry_run=args.dry_run,
            progress=args.progress,
        )
    except FileExistsError as exc:
        print(f"[challenge-generate] error: {exc}", file=sys.stderr)
        return 2
    except ModelConfigurationError as exc:
        print(f"[challenge-generate] error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError) as exc:
        print(f"[challenge-generate] error: {exc}", file=sys.stderr)
        return 2

    if result.dry_run:
        print(
            f"[challenge-generate] dry-run: would write {result.total_trajectories} trajectories to "
            f"{result.trajectories_dir} (split_counts={result.split_counts}, family_counts={result.family_counts})"
        )
        return 0

    print(
        f"[challenge-generate] wrote {result.total_trajectories} trajectories to {result.trajectories_dir} "
        f"(split_counts={result.split_counts}, family_counts={result.family_counts}, "
        f"reachability_statistics={result.reachability_statistics})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
