# trajectories/

Trajectory definitions and trial manifests for Tier 2-4, generated deterministically from the
real KR810 model (`assets/kr810.xml`) by the scripts in `generators/`.

## Regenerating

```
.venv\Scripts\python.exe -m generators.generate_line_trajectory --overwrite
.venv\Scripts\python.exe -m generators.generate_circle_trajectory --overwrite
.venv\Scripts\python.exe -m generators.generate_figure8_trajectory --overwrite
.venv\Scripts\python.exe -m generators.generate_helix_trajectory --overwrite
.venv\Scripts\python.exe -m generators.generate_custom_trajectory_template --overwrite
.venv\Scripts\python.exe -m generators.validate_generated_targets
```

Each generator accepts `--seed <int>` (default: `random_seed` in `configs/benchmark_config.json`,
currently `42`) and requires `--overwrite` to replace existing output files. Each of the four
shape generators writes both the fixed- and variable-orientation file for its shape and updates
the shared `trajectory_manifest.csv` / `trajectory_trials.csv` rows for those two trajectories
only (rows for other trajectories are left untouched).

## Anchor selection

All 8 generated trajectories (and the custom template) share one anchor joint configuration,
found by sampling candidates, computing their real Jacobian sigma_min and normalized joint-limit
margin, and keeping the candidate with the largest margin among those whose sigma_min is
comfortably above `configs/dls_config.json:singularity_sigma_threshold`. The anchor's
end-effector orientation (from forward kinematics) supplies the orthonormal frame `(u, v, w)`
used as the path plane/axis for every shape, and every shape's position path starts exactly at
the anchor position (`s=0` maps to the anchor pose), so the first sequential-DLS solve is trivial.

## Geometry (path parameter `s` in `[0, 1]`)

- **Line**: `p(s) = p0 + s * (p1 - p0)`, `p0` = anchor position, `p1 = p0 + length * u`.
- **Circle**: `p(s) = c + r*cos(2*pi*s)*u + r*sin(2*pi*s)*v`, closed path.
- **Figure-8**: `p(s) = c + a*sin(2*pi*s)*u + b*sin(4*pi*s)*v`, closed path.
- **Helix**: `p(s) = c + r*cos(2*pi*s)*u + r*sin(2*pi*s)*v + h*(s-0.5)*w`, open path.

Nominal sizes (shrunk automatically, see below, if any waypoint fails validation): line length
0.12 m, circle radius 0.045 m, figure-8 amplitudes a=0.05 m/b=0.03 m, helix radius 0.04 m / height
0.08 m. All 8 generated files validated at the full nominal size (no shrinking was needed); the
realized size is always recorded in `trajectory_manifest.csv`
(`path_scale_1_m`/`path_scale_2_m`/`path_scale_3_m`).

## Time scaling

Waypoints are sampled at `tau_i = i/(N-1)` for `i = 0..N-1` (`N` = `default_waypoints` in
`configs/trajectory_config.json`, currently 400) over `[0, duration_s]`
(`default_duration_s`, currently 10.0 s), giving a realized `control_period_s =
duration_s/(N-1)`. The path parameter uses quintic time scaling
`s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5`, so `s(0)=0`, `s(1)=1`, and `s'(0)=s'(1)=s''(0)=s''(1)=0`
(velocity/acceleration are ~0 at both endpoints). Cartesian `target_linear_velocity` and
`target_linear_acceleration` are computed analytically via the chain rule through this scaling
(`dp/dt = dp/ds * ds/dtau / duration_s`, etc.), not via raw finite differences.

## Fixed vs. variable orientation

- **Fixed**: every waypoint uses the anchor's orientation (`generators/generate_orientation_profile.py:
  fixed_orientation_profile`).
- **Variable**: SO(3) geodesic interpolation `R(s) = R0 @ Exp(phase(s) * Log(R0.T @ R1))`
  (`variable_orientation_profile`), never independent Euler-angle interpolation, with `R1` a
  bounded local rotation from the anchor (0.35 rad about the anchor's local z-axis). For closed
  paths (circle, figure-8) `phase(s)` goes out-and-back (`0 -> 1 -> 0`) so orientation returns to
  its start alongside the closed position path; for open paths (line, helix) `phase(s) = s`.
  Output quaternions are wxyz and canonicalized so consecutive samples have a non-negative dot
  product (no sign-flip discontinuities).

## Generation-time reachability validation

Every waypoint of every trajectory is checked with warm-started sequential DLS (each solve
initialized from the previous waypoint's solution, using `configs/dls_config.json` thresholds)
before the file is written. If any waypoint fails, the path size (and, for variable orientation,
the rotation magnitude) is shrunk by 15% and the entire sequence is retried, up to 15 attempts;
waypoints are never silently dropped. `generation_status = validated` in
`trajectory_manifest.csv` means 100% of waypoints converged; `validation_waypoint_success_rate`,
`validation_position_max_mm`, and `validation_orientation_max_deg` record the outcome. This is a
**data-generation reachability check only**, not a Tier 2 evaluation result.

## Files

- `trajectory_manifest.csv` - one row per generated trajectory: `trajectory_id, type, file_path,
  num_waypoints, duration_s, control_period_s, orientation_mode, closed_path, anchor_q_json,
  center_x_m, center_y_m, center_z_m, path_scale_1_m, path_scale_2_m, path_scale_3_m,
  generation_seed, validation_waypoint_success_rate, validation_position_max_mm,
  validation_orientation_max_deg, generation_status, sha256`.
- `trajectory_trials.csv` - trial definitions per trajectory (see below).
- `line/`, `circle/`, `figure8/`, `helix/` - the 8 generated `.npz` files (fixed + variable
  orientation per shape).
- `custom/custom_trajectory_template.csv` / `.npz` / `custom_trajectory_metadata.json` - a
  minimal, real-model-derived (not fabricated) 5-waypoint illustration of the custom-trajectory
  schema, flagged `template_only: true`; **not** part of the official benchmark.

## Waypoint NPZ arrays (`line/`, `circle/`, `figure8/`, `helix/`)

| array | shape | dtype | unit |
| --- | --- | --- | --- |
| `waypoint_id` | `[N]` | int64 | - |
| `time_s` | `[N]` | float64 | second |
| `path_parameter_s` | `[N]` | float64 | - (in `[0,1]`) |
| `target_position` | `[N,3]` | float64 | meter |
| `target_quaternion` | `[N,4]` | float64 | - (wxyz, unit norm) |
| `target_linear_velocity` | `[N,3]` | float64 | m/s |
| `target_linear_acceleration` | `[N,3]` | float64 | m/s^2 |

`N = 400` for every generated file. Loads with `np.load(path, allow_pickle=False)`.

## Repeatability vs. robustness trials

`trajectory_trials.csv` columns: `trial_id, trajectory_id, trial_category, repeat_id, seed,
speed_scale, control_period_s, q1_init..q7_init`. Both categories are generated at each speed
scale in `configs/trajectory_config.json:speed_scales` (0.5, 1.0, 1.5):

- **repeatability**: the same trajectory, the same `q_initial` (the anchor), the same conditions,
  `repeat_id` 0-9, no random perturbation - deterministic kinematics (Tier 0-4 has no
  dynamics/noise) means ideal repeatability is expected here; these rows only record trial
  *definitions*, not a repeatability *result*.
- **robustness**: 5 independently sampled, verified-solvable `q_initial` configurations per
  trajectory (each confirmed able to solve the trajectory's first waypoint via DLS before being
  accepted), with distinct seeds. Never described as ISO repeatability.

45 trial rows per trajectory (30 repeatability + 15 robustness) x 8 trajectories = 360 rows
total. All `q*_init` values are within `configs/robot_config.json` operational limits.

## Reachability guarantee

Every waypoint position/orientation in every generated trajectory file is either the anchor pose
itself or was validated reachable via warm-started sequential DLS at generation time (see above).
