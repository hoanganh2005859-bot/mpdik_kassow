"""Generates helical Cartesian trajectories from a validated anchor pose on the KR810 model.

p(s) = c + r*cos(2*pi*s)*u + r*sin(2*pi*s)*v + h*(s-0.5)*w, where (u, v, w) are the anchor
end-effector's local x/y/z axes (world frame) and c is chosen so p(0) equals the anchor position
exactly (trivial first waypoint). Every waypoint is validated via warm-started sequential DLS
before acceptance; on failure the radius/height (and, for the variable-orientation file, the
orientation change) are shrunk and the whole open path is retried.

Writes:
    trajectories/helix/helix_fixed_orientation.npz
    trajectories/helix/helix_variable_orientation.npz
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

OUTPUT_DIR = REPO_ROOT / "trajectories" / "helix"

NOMINAL_RADIUS_M = 0.04
NOMINAL_HEIGHT_M = 0.08
ROTATION_ANGLE_RAD = 0.35
ROTATION_AXIS = np.array([0.0, 0.0, 1.0])
CLOSED_PATH = False
TWO_PI = 2.0 * np.pi


def _helix_position_and_derivatives(center, r, h, u, v, w, s):
    angle = TWO_PI * s
    cos_a = np.cos(angle)[:, None]
    sin_a = np.sin(angle)[:, None]

    positions = center[None, :] + r * cos_a * u[None, :] + r * sin_a * v[None, :] + h * (s[:, None] - 0.5) * w[None, :]
    dp_ds = -TWO_PI * r * sin_a * u[None, :] + TWO_PI * r * cos_a * v[None, :] + h * w[None, :]
    d2p_ds2 = -(TWO_PI**2) * r * cos_a * u[None, :] - (TWO_PI**2) * r * sin_a * v[None, :]
    return positions, dp_ds, d2p_ds2


def _make_build_fn(fk_anchor, u, v, w, s, num_waypoints, orientation_mode):
    def build_fn(scale):
        r = NOMINAL_RADIUS_M * scale
        h = NOMINAL_HEIGHT_M * scale
        center = fk_anchor.position - r * u + 0.5 * h * w
        positions, dp_ds, d2p_ds2 = _helix_position_and_derivatives(center, r, h, u, v, w, s)

        rotation_vector = ROTATION_ANGLE_RAD * scale * ROTATION_AXIS if orientation_mode == "variable" else None
        quats = orientation_arrays(orientation_mode, fk_anchor.rotation_matrix, rotation_vector, s, CLOSED_PATH, num_waypoints)

        extra = {"dp_ds": dp_ds, "d2p_ds2": d2p_ds2, "center": center, "scales": [r, h]}
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
    u = fk_anchor.rotation_matrix[:, 0]
    v = fk_anchor.rotation_matrix[:, 1]
    w = fk_anchor.rotation_matrix[:, 2]

    results = {}
    for orientation_mode, trajectory_id, filename in (
        ("fixed", "helix_fixed_orientation", "helix_fixed_orientation.npz"),
        ("variable", "helix_variable_orientation", "helix_variable_orientation.npz"),
    ):
        build_fn = _make_build_fn(fk_anchor, u, v, w, s, num_waypoints, orientation_mode)
        validated = generate_validated_geometry(model_context, dls_config, q_anchor, build_fn)

        arrays = build_trajectory_arrays(
            time_s, s, validated["positions"], validated["quaternions"],
            validated["extra"]["dp_ds"], validated["extra"]["d2p_ds2"], s_dot_dtau, s_ddot_dtau, duration_s,
        )

        saved_path, sha, manifest_row = write_trajectory_outputs(
            trajectory_id=trajectory_id,
            trajectory_type="helix",
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
            f"[generate_helix_trajectory] wrote {relative_to_repo(saved_path)} "
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
