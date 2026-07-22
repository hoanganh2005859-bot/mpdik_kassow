"""Creates the Dataset v2 path/config/schema/checksum scaffold at an explicit dataset root.

Phase 1 only: writes directory structure, config/schema templates, and a checksum manifest --
never NPZ/CSV data, never a fabricated sample count. Dataset v2 has no repo-root or CWD-implicit
default (unlike Dataset v1's pipelines, which keep working unchanged with no flags at all); a
dataset root must always be passed explicitly.

Usage:
    python -m pipelines.run_dataset_v2_scaffold --dataset-root /path/to/kr810_dataset_v2 --master-seed 42
"""

import argparse
import sys

from dataset_v2.scaffold import create_dataset_v2_scaffold
from utils.exceptions import ModelConfigurationError

DEFAULT_MASTER_SEED = 42


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", required=True, type=str, help="Explicit Dataset v2 root (created if it does not exist)."
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help=f"Master seed recorded in configs/seed_policy.json (default: {DEFAULT_MASTER_SEED}). "
        "Not used for generation in Phase 1 -- there is no generation logic yet.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate an existing scaffold in place.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        paths = create_dataset_v2_scaffold(
            dataset_root=args.dataset_root, master_seed=args.master_seed, overwrite=args.overwrite
        )
    except (ModelConfigurationError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"[run_dataset_v2_scaffold] scaffold created at {paths.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
