"""CLI: Dataset v2 trial difficulty-threshold calibration (Phase 7).

    python -m pipelines.run_dataset_v2_trial_calibration --dataset-root PATH [--master-seed N]
        [--report-json PATH] [--pool-scale N]

Calibrates the easy/medium/hard bands for the combined-normalized-pose-error primary metric using
ONLY the development trajectories under ``--dataset-root``. Prints the calibration report as JSON;
the resulting numbers are baked into ``dataset_v2/config_templates.py`` as the [LOCKED] ``TRIAL_*``
constants (see ``docs/V2_TRIAL_DIFFICULTY_CALIBRATION.md``). Never runs DLS; never touches
validation or frozen_test.
"""

import argparse
import json
import sys
from pathlib import Path

from dataset_v2.trial_calibration import calibrate, calibration_report, write_calibration_report
from utils.config_loader import load_json_config
from dataset_v2.locator import require_dataset_v2_root
from utils.exceptions import ModelConfigurationError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--master-seed", type=int, default=None)
    parser.add_argument("--report-json", type=Path, default=None, help="Optional path to write the calibration report JSON (never a dataset root or the repo).")
    parser.add_argument("--pool-scale", type=int, default=None, help="Test/smoke override only: uniform small sub-pool size.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        paths = require_dataset_v2_root(args.dataset_root)
        seed_policy = load_json_config(paths.configs_dir / "seed_policy.json")
        master_seed = int(args.master_seed if args.master_seed is not None else seed_policy["master_seed"])
        result = calibrate(args.dataset_root, master_seed=master_seed, pool_scale_override=args.pool_scale)
    except (ModelConfigurationError, FileNotFoundError, RuntimeError) as exc:
        print(f"[trial-calibration] error: {exc}", file=sys.stderr)
        return 2

    report = calibration_report(result, master_seed)
    print(json.dumps(report, indent=2))
    if args.report_json is not None:
        write_calibration_report(result, master_seed, args.report_json)
        print(f"[trial-calibration] wrote {args.report_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
