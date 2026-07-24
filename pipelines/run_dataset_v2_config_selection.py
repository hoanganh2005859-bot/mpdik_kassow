"""CLI: run ALL pre-registered candidates on development, then select exactly one (task section 5).

    python -m pipelines.run_dataset_v2_config_selection \
        --dataset-root <v2 work root> \
        --public-root <public eval root> \
        --evaluation-output-root <eval output root> \
        [--resume | --overwrite] [--progress] [--pre-register-only]

Development split ONLY. Validation and frozen_test are never touched here. ``--pre-register-only``
writes the candidate pre-registration document and exits WITHOUT running anything (used to lock the
candidate set before any compute). The full selection run is large -- see the harness review report
for the estimated workload before invoking it without ``--pre-register-only``.
"""

import argparse
import json
import sys
from pathlib import Path

from evaluation_v2.candidate_configs import candidate_set, pre_registration_record
from evaluation_v2.locator import eval_output_paths
from evaluation_v2.orchestrator import run_evaluation
from evaluation_v2.selection import (
    extract_selection_metrics,
    select_candidate,
    selection_objective_spec,
    write_selection_report,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Development-split candidate selection for Dataset v2 DLS.")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--public-root", required=True)
    p.add_argument("--evaluation-output-root", required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--pre-register-only", action="store_true",
                   help="Write the candidate pre-registration document and exit without running.")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = eval_output_paths(args.evaluation_output_root, require_exists=False)
    out.root.mkdir(parents=True, exist_ok=True)

    prereg_path = out.root / "candidate_preregistration.json"
    prereg_doc = pre_registration_record()
    prereg_doc["selection_objective"] = selection_objective_spec()
    prereg_tmp = prereg_path.with_name(prereg_path.name + ".tmp")
    prereg_tmp.write_text(json.dumps(prereg_doc, indent=2, sort_keys=True), encoding="utf-8")
    prereg_tmp.replace(prereg_path)
    print(f"[selection] pre-registration (candidates + locked selection objective) written: {prereg_path}")

    if args.pre_register_only:
        print("[selection] --pre-register-only: candidate set locked, no evaluation run.")
        return 0

    candidate_metrics = {}
    for candidate in candidate_set():
        run_name = f"development/{candidate.candidate_id}"
        run_evaluation(
            args.dataset_root, args.public_root, args.evaluation_output_root, candidate,
            splits=("development",), run_name=run_name, resume=args.resume, overwrite=args.overwrite,
            show_progress=args.progress,
        )
        run_dir = out.config_run_dir(run_name)
        candidate_metrics[candidate.candidate_id] = extract_selection_metrics(run_dir)

    report = select_candidate(candidate_metrics)
    write_selection_report(out.root / "development_selection_report.json", report)
    print(f"[selection] selected={report['selected_candidate_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
