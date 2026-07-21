"""Generates the illustrative custom-trajectory template (schema example only, not benchmark data).

Writes a small, real-model-derived (not fabricated) 5-waypoint straight-line example so
trajectories/custom/custom_trajectory_template.csv and .npz show a concrete, valid instance of
the custom-trajectory schema. Both outputs are marked template_only=true in
custom_trajectory_metadata.json and must never be included in official Tier 2-4 benchmark runs.

Writes:
    trajectories/custom/custom_trajectory_template.csv
    trajectories/custom/custom_trajectory_template.npz
    trajectories/custom/custom_trajectory_metadata.json
"""

import argparse
import csv

import numpy as np

from generators._common import GENERATOR_VERSION, REPO_ROOT, ensure_output_path, get_model_context, load_dls_config, relative_to_repo, save_npz, write_json
from generators._trajectory_common import ANCHOR_TAG, select_anchor
from generators.generate_orientation_profile import fixed_orientation_profile
from utils.file_checksum import sha256_file

OUTPUT_DIR = REPO_ROOT / "trajectories" / "custom"
TEMPLATE_TAG = 95
NUM_WAYPOINTS = 5
LENGTH_M = 0.05
CONTROL_PERIOD_S = 0.1


def build_template(seed: int):
    model_context = get_model_context()
    dls_config = load_dls_config()
    q_anchor, fk_anchor = select_anchor(model_context, dls_config, seed, TEMPLATE_TAG)

    direction_u = fk_anchor.rotation_matrix[:, 0]
    s = np.linspace(0.0, 1.0, NUM_WAYPOINTS)
    positions = fk_anchor.position[None, :] + s[:, None] * LENGTH_M * direction_u[None, :]
    quats = fixed_orientation_profile(fk_anchor.rotation_matrix, NUM_WAYPOINTS)
    time_s = np.arange(NUM_WAYPOINTS, dtype=np.float64) * CONTROL_PERIOD_S
    waypoint_id = np.arange(NUM_WAYPOINTS, dtype=np.int64)

    arrays = {
        "waypoint_id": waypoint_id,
        "time_s": time_s,
        "x_m": positions[:, 0].astype(np.float64),
        "y_m": positions[:, 1].astype(np.float64),
        "z_m": positions[:, 2].astype(np.float64),
        "qw": quats[:, 0].astype(np.float64),
        "qx": quats[:, 1].astype(np.float64),
        "qy": quats[:, 2].astype(np.float64),
        "qz": quats[:, 3].astype(np.float64),
    }
    return arrays, q_anchor


def _write_csv(path, overwrite, arrays):
    path = ensure_output_path(path, overwrite)
    n = arrays["waypoint_id"].shape[0]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["waypoint_id", "time_s", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz"])
        for i in range(n):
            writer.writerow(
                [
                    int(arrays["waypoint_id"][i]),
                    f"{arrays['time_s'][i]:.6f}",
                    f"{arrays['x_m'][i]:.8f}",
                    f"{arrays['y_m'][i]:.8f}",
                    f"{arrays['z_m'][i]:.8f}",
                    f"{arrays['qw'][i]:.8f}",
                    f"{arrays['qx'][i]:.8f}",
                    f"{arrays['qy'][i]:.8f}",
                    f"{arrays['qz'][i]:.8f}",
                ]
            )
    return path


def run(seed: int, overwrite: bool, output_dir=OUTPUT_DIR) -> dict:
    arrays, q_anchor = build_template(seed)

    csv_path = _write_csv(output_dir / "custom_trajectory_template.csv", overwrite, arrays)
    npz_path = save_npz(output_dir / "custom_trajectory_template.npz", overwrite, arrays)

    metadata = {
        "template_only": True,
        "note": (
            "Illustrative example of the custom-trajectory schema only. Not part of any official "
            "Tier 2-4 benchmark and must not be consumed by evaluation pipelines."
        ),
        "generator": "generate_custom_trajectory_template",
        "generator_version": GENERATOR_VERSION,
        "generation_seed": seed,
        "num_waypoints": int(arrays["waypoint_id"].shape[0]),
        "anchor_q_rad": [float(v) for v in q_anchor],
        "csv_path": relative_to_repo(csv_path),
        "npz_path": relative_to_repo(npz_path),
        "csv_sha256": sha256_file(csv_path),
        "npz_sha256": sha256_file(npz_path),
    }
    metadata_path = write_json(output_dir / "custom_trajectory_metadata.json", True, metadata)

    print(f"[generate_custom_trajectory_template] wrote {relative_to_repo(csv_path)}, {relative_to_repo(npz_path)}, {relative_to_repo(metadata_path)}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Base seed (default: benchmark_config.json random_seed)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    from generators._common import load_benchmark_config

    seed = args.seed if args.seed is not None else load_benchmark_config()["random_seed"]
    run(seed=seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
