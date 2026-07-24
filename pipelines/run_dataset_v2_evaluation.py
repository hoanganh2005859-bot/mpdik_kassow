"""CLI: run the Dataset v2 DLS evaluation for ONE candidate config over development/validation.

    python -m pipelines.run_dataset_v2_evaluation \
        --dataset-root <v2 work root> \
        --public-root <public eval root> \
        --evaluation-output-root <eval output root> \
        --candidate <candidate_id> \
        --splits development,validation \
        (--resume | --overwrite) \
        [--point-sample-limit N] [--trial-limit N] [--waypoint-limit N] \
        [--methods warm_start,cold_start] [--run-name NAME] [--progress]

Reads only the public root; frozen_test is never accessed. The ``--*-limit`` flags are for smoke
runs only and must be omitted for a full run.
"""

import argparse
import json
import sys

from evaluation_v2.candidate_configs import candidate_by_id, candidate_set
from evaluation_v2.orchestrator import run_evaluation


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Dataset v2 DLS evaluation for one candidate config.")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--public-root", required=True)
    p.add_argument("--evaluation-output-root", required=True)
    p.add_argument("--candidate", required=True, help=f"one of: {[c.candidate_id for c in candidate_set()]}")
    p.add_argument("--splits", default="development", help="comma-separated; development,validation only.")
    p.add_argument("--run-name", default=None)
    p.add_argument("--methods", default="warm_start,cold_start")
    p.add_argument("--point-sample-limit", type=int, default=None, help="smoke only")
    p.add_argument("--trial-limit", type=int, default=None, help="smoke only")
    p.add_argument("--waypoint-limit", type=int, default=None, help="smoke only")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    try:
        candidate = candidate_by_id(args.candidate)
    except KeyError as exc:
        print(f"[evaluation] ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        manifest = run_evaluation(
            args.dataset_root, args.public_root, args.evaluation_output_root, candidate,
            splits=splits, run_name=args.run_name, resume=args.resume, overwrite=args.overwrite,
            point_sample_limit=args.point_sample_limit, trial_limit=args.trial_limit,
            waypoint_limit=args.waypoint_limit, methods=methods, show_progress=args.progress,
        )
    except (FileExistsError, ValueError, RuntimeError) as exc:
        print(f"[evaluation] ERROR: {exc}", file=sys.stderr)
        return 2
    status = manifest.get("overall_status", "unknown")
    print(f"[evaluation] candidate={args.candidate} status={status} "
          f"tiers={json.dumps({k: v.get('status', v.get('gate_pass')) for k, v in manifest['tiers'].items()})}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
