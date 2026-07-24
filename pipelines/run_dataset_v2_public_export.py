"""CLI: build the public evaluation root and protected validation root from a Dataset v2 root.

    python -m pipelines.run_dataset_v2_public_export \
        --dataset-root <v2 work root> \
        --public-root <public eval root> \
        --protected-root <protected validation root> \
        [--overwrite]

Only development + validation are exported; frozen_test is never touched. The public root carries
no protected reference array (q_reference / q_target_reference / reconstruction evidence).
"""

import argparse
import json
import sys

from evaluation_v2.public_export import export_public_and_protected


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export the Dataset v2 public/protected evaluation roots.")
    p.add_argument("--dataset-root", required=True, help="Dataset v2 working root (input).")
    p.add_argument("--public-root", required=True, help="Public evaluation root to write (output).")
    p.add_argument("--protected-root", required=True, help="Protected validation root to write (output).")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting non-empty output roots.")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = export_public_and_protected(
            args.dataset_root, args.public_root, args.protected_root, overwrite=args.overwrite
        )
    except (FileExistsError, ValueError) as exc:
        print(f"[public-export] ERROR: {exc}", file=sys.stderr)
        return 2
    print("[public-export] " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
