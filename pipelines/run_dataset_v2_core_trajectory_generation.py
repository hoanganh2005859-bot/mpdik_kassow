"""CLI: Dataset v2 core trajectory generation/validation.

    python -m pipelines.run_dataset_v2_core_trajectory_generation --dataset-root PATH
        [--master-seed N] [--overwrite] [--validate-only] [--dry-run] [--progress]

The locked full mode is 120 core trajectories (5 shapes x 2 orientation modes x 12 anchors) per
``specs/DLS_DATASET_V2_SPEC.md`` section H. The ``--anchor-id``/``--shape``/``--orientation-mode``/
``--source-waypoints`` overrides exist for tests/smoke runs only and never change the locked
400-waypoint canonical count.

Reachability (spec section H.1, Phase 5.1): a waypoint is accepted only when a reference
configuration exists whose *independently recomputed* FK reproduces the target pose within Dataset
v2's own strict tolerances (``configs/generation_reachability_config.json``: 1e-4 m / 0.01 deg) --
never because the numerical IK engine reported success, and never against Dataset v1's DLS
*baseline evaluation* thresholds. Both the canonical and the high-resolution source path are
verified. Geometry-basis alternatives are searched before any scale reduction, and the
accepted-scale distribution is reported so small-scale trajectories cannot pass unnoticed.
"""

import argparse
import sys
from pathlib import Path

from dataset_v2.core_trajectory_generation import run_core_trajectory_generation
from dataset_v2.core_trajectory_validation import validate_core_trajectories
from utils.exceptions import ModelConfigurationError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Explicit Dataset v2 root (must already hold an anchor catalog).")
    parser.add_argument("--master-seed", type=int, default=None, help="Override configs/seed_policy.json's recorded master seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow regenerating existing core trajectory v2 output.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing core trajectory v2 output; generate nothing.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be generated; write nothing.")
    parser.add_argument("--anchor-id", action="append", default=None, dest="anchor_ids", help="Test/smoke override only: restrict to this anchor_id (repeatable).")
    parser.add_argument("--shape", action="append", default=None, dest="shapes", choices=["line", "circle", "figure8", "helix", "free_form"], help="Test/smoke override only: restrict to this shape (repeatable).")
    parser.add_argument("--orientation-mode", action="append", default=None, dest="orientation_modes", choices=["fixed", "variable"], help="Test/smoke override only: restrict to this orientation mode (repeatable).")
    parser.add_argument("--source-waypoints", type=int, default=None, help="Test/smoke override only: high-resolution source waypoint count. Never affects the locked 400-waypoint canonical count.")
    parser.add_argument("--progress", action="store_true", help="Print per-trajectory progress (geometry alternative, accepted scale, reconstruction errors, timing) to stderr.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.validate_only:
        full_counts = (
            args.anchor_ids is None and args.shapes is None and args.orientation_modes is None and args.source_waypoints is None
        )
        try:
            report = validate_core_trajectories(args.dataset_root, full_counts=full_counts)
        except (ModelConfigurationError, FileNotFoundError) as exc:
            print(f"[core-trajectory-validate] error: {exc}", file=sys.stderr)
            return 2

        for reason in report.reasons:
            print(f"[core-trajectory-validate] FAIL: {reason}", file=sys.stderr)
        print(f"[core-trajectory-validate] total={report.total_trajectories} canonical_poses={report.canonical_poses_total} passed={report.passed}")
        return 0 if report.passed else 1

    try:
        result = run_core_trajectory_generation(
            args.dataset_root,
            master_seed=args.master_seed,
            overwrite=args.overwrite,
            anchor_ids=args.anchor_ids,
            shapes=args.shapes,
            orientation_modes=args.orientation_modes,
            source_waypoint_count=args.source_waypoints,
            dry_run=args.dry_run,
            progress=args.progress,
        )
    except FileExistsError as exc:
        print(f"[core-trajectory-generate] error: {exc}", file=sys.stderr)
        return 2
    except ModelConfigurationError as exc:
        print(f"[core-trajectory-generate] error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError) as exc:
        print(f"[core-trajectory-generate] error: {exc}", file=sys.stderr)
        return 2

    if result.dry_run:
        print(
            f"[core-trajectory-generate] dry-run: would write {result.total_trajectories} trajectories to "
            f"{result.trajectories_dir} (split_counts={result.split_counts}, shape_counts={result.shape_counts}, "
            f"orientation_counts={result.orientation_counts})"
        )
        return 0

    print(
        f"[core-trajectory-generate] wrote {result.total_trajectories} trajectories to {result.trajectories_dir} "
        f"(split_counts={result.split_counts}, shape_counts={result.shape_counts}, "
        f"orientation_counts={result.orientation_counts}, scale_statistics={result.scale_statistics}, "
        f"reachability_statistics={result.reachability_statistics})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
