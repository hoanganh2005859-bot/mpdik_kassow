"""Generates straight-line Cartesian trajectories from a validated anchor pose on the KR810 model.

p(s) = p0 + s * (p1 - p0), starting exactly at the anchor position (p0 = anchor position) so the
first waypoint's sequential-DLS solve is trivial. Every waypoint is validated via warm-started
DLS before the trajectory is accepted; if any waypoint fails, the line length (and, for the
variable-orientation file, the orientation change) is shrunk and the whole sequence is retried.

Writes:
    trajectories/line/line_fixed_orientation.npz
    trajectories/line/line_variable_orientation.npz
"""

import argparse

import numpy as np

from generators._common import REPO_ROOT, get_model_context, load_dls_config, load_trajectory_config, relative_to_repo
from generators._trajectory_common import (
    ANCHOR_TAG,
    build_time_and_path_parameter,
    build_trajectory_arrays,
    build_trials,
    generate_validated_geometry,
    orientation_arrays,
    replace_trial_rows,
    select_anchor,
    write_trajectory_outputs,
)

OUTPUT_DIR = REPO_ROOT / "trajectories" / "line"

NOMINAL_LENGTH_M = 0.12
ROTATION_ANGLE_RAD = 0.35
ROTATION_AXIS = np.array([0.0, 0.0, 1.0])
CLOSED_PATH = False


def _line_position_and_derivatives(p0, p1, s, num_waypoints):
    delta = p1 - p0
    positions = p0[None, :] + s[:, None] * delta[None, :]
    dp_ds = np.tile(delta, (num_waypoints, 1))
    d2p_ds2 = np.zeros((num_waypoints, 3))
    return positions, dp_ds, d2p_ds2


def _make_build_fn(fk_anchor, direction_u, s, num_waypoints, orientation_mode):
    def build_fn(scale):
        length = NOMINAL_LENGTH_M * scale
        p0 = fk_anchor.position
        p1 = p0 + length * direction_u
        positions, dp_ds, d2p_ds2 = _line_position_and_derivatives(p0, p1, s, num_waypoints)

        rotation_vector = ROTATION_ANGLE_RAD * scale * ROTATION_AXIS if orientation_mode == "variable" else None
        quats = orientation_arrays(orientation_mode, fk_anchor.rotation_matrix, rotation_vector, s, CLOSED_PATH, num_waypoints)

        extra = {
            "dp_ds": dp_ds,
            "d2p_ds2": d2p_ds2,
            "center": (p0 + p1) / 2.0,
            "scales": [length],
        }
        return positions, quats, extra

    return build_fn


def run(seed: int, overwrite: bool, output_dir=OUTPUT_DIR) -> dict:
    model_context = get_model_context()
    dls_config = load_dls_config()
    trajectory_config = load_trajectory_config()

    num_waypoints = int(trajectory_config["default_waypoints"])
    duration_s = float(trajectory_config["default_duration_s"])
    time_s, tau, s, s_dot_dtau, s_ddot_dtau, control_period_s = build_time_and_path_parameter(num_waypoints, duration_s)

    q_anchor, fk_anchor = select_anchor(model_context, dls_config, seed, ANCHOR_TAG)
    direction_u = fk_anchor.rotation_matrix[:, 0]

    results = {}
    for orientation_mode, trajectory_id, filename in (
        ("fixed", "line_fixed_orientation", "line_fixed_orientation.npz"),
        ("variable", "line_variable_orientation", "line_variable_orientation.npz"),
    ):
        build_fn = _make_build_fn(fk_anchor, direction_u, s, num_waypoints, orientation_mode)
        validated = generate_validated_geometry(model_context, dls_config, q_anchor, build_fn)

        arrays = build_trajectory_arrays(
            time_s, s, validated["positions"], validated["quaternions"],
            validated["extra"]["dp_ds"], validated["extra"]["d2p_ds2"], s_dot_dtau, s_ddot_dtau, duration_s,
        )

        saved_path, sha, manifest_row = write_trajectory_outputs(
            trajectory_id=trajectory_id,
            trajectory_type="line",
            orientation_mode=orientation_mode,
            closed_path=CLOSED_PATH,
            npz_path=output_dir / filename,
            arrays=arrays,
            q_anchor=q_anchor,
            center=validated["extra"]["center"],
            path_scales=validated["extra"]["scales"],
            seed=seed,
            validation_result=validated,
            overwrite=overwrite,
        )

        trial_rows = build_trials(
            model_context, dls_config, trajectory_id, q_anchor,
            arrays["target_position"][0], arrays["target_quaternion"][0], seed, control_period_s,
        )
        replace_trial_rows(trajectory_id, trial_rows)

        print(
            f"[generate_line_trajectory] wrote {relative_to_repo(saved_path)} "
            f"({num_waypoints} waypoints, success_rate={validated['success_rate']:.3f}, scale={validated['scale']:.3f})"
        )
        results[trajectory_id] = manifest_row

    return results


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
