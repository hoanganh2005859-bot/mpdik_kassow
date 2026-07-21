# benchmarks/

Benchmark data for Tier 0 (kinematics validation) and Tier 1 (point IK). All data below is
generated deterministically from the real KR810 model (`assets/kr810.xml`) by the scripts in
`generators/`; nothing here is hand-authored or fabricated.

## Regenerating

```
.venv\Scripts\python.exe -m generators.generate_fk_validation_states --overwrite
.venv\Scripts\python.exe -m generators.generate_point_ik_dataset --overwrite
.venv\Scripts\python.exe -m generators.validate_generated_targets
```

Both generators accept `--seed <int>` (default: `random_seed` in `configs/benchmark_config.json`,
currently `42`) and require `--overwrite` to replace existing output files.

## validation/ (Tier 0)

Three NPZ files of joint states used to sanity-check FK/Jacobian/singularity computations before
any solver is trusted, plus a `validation_checksum.json` summary (SHA256, shapes, dtypes, group
definitions, and the singularity thresholds used, all traceable to `configs/dls_config.json` and
`configs/benchmark_config.json`).

- `fk_test_states.npz` - 1000 samples, arrays `sample_id` (int64), `q_samples` (float64 [N,7]),
  `group_id` (int32), `source_seed` (int64). Groups (200 samples each):
  `0=zero_or_home, 1=random_interior, 2=near_operational_lower_limit,
  3=near_operational_upper_limit, 4=mixed_near_limits`. Near-limit samples keep a margin
  (>= 0.05 rad) from the actual bound so a finite-difference perturbation cannot cross it.
- `jacobian_test_states.npz` - 200 samples, arrays as above plus
  `finite_difference_epsilon` (float64, constant = `configs/benchmark_config.json:
  finite_difference_epsilon`). Groups (40 samples each):
  `0=regular, 1=near_lower_limit, 2=near_upper_limit, 3=mixed_near_limits, 4=low_sigma`. The
  `low_sigma` group is selected by computing the real Jacobian sigma_min for a large candidate
  pool and keeping the smallest, never by assumption.
- `singularity_test_states.npz` - 300 samples, arrays `sample_id`, `q_samples`, `sigma_min`
  (float64), `condition_number` (float64), `group_id`, `source_seed`. Groups (100 samples each):
  `0=regular, 1=moderately_conditioned, 2=near_singular`, split by the actual computed sigma_min
  against `configs/dls_config.json:singularity_sigma_threshold` (near_singular) and a
  configurable multiple of it (moderately_conditioned upper bound), both recorded in
  `validation_checksum.json`.

## point_ik/ (Tier 1)

`point_ik_v1.npz` - 1200 samples (200 per difficulty group x 6 groups). Every target is produced
by drawing a valid `q_target` and running forward kinematics, so every `target_position`/
`target_quaternion` is reachable by construction, never a freely chosen Cartesian point assumed
reachable.

Arrays (position: meter, quaternion: wxyz, angles: radian):

| array | shape | dtype |
| --- | --- | --- |
| `sample_id` | `[N]` | int64 |
| `q_initial`, `q_target` | `[N,7]` | float64 |
| `initial_position`, `target_position` | `[N,3]` | float64 |
| `initial_quaternion`, `target_quaternion` | `[N,4]` | float64 |
| `position_distance_m`, `orientation_distance_rad`, `joint_distance_rad` | `[N]` | float64 |
| `initial_sigma_min`, `target_sigma_min` | `[N]` | float64 |
| `minimum_initial_limit_margin`, `minimum_target_limit_margin` | `[N]` | float64 |
| `difficulty_id` | `[N]` | int32 |
| `source_seed` | `[N]` | int64 |

Difficulty groups (`difficulty_id`): `0=near_target, 1=medium_target, 2=far_target,
3=large_orientation_change, 4=near_joint_limit, 5=near_singularity`. Thresholds are **not**
arbitrary: a large generic pool of `(q_initial, q_target)` pairs is sampled first, five real
quantities are computed for every pair (joint distance, Cartesian position distance, SO(3)
orientation distance, sigma_min, joint-limit margin), and group boundaries are the empirical
quantiles of that pool (33rd/66th percentile of position distance for near/medium/far, 85th
percentile of orientation distance, 10th percentile of limit margin and of sigma_min). A pair
that qualifies for more than one group is resolved by a fixed priority order (most specific
first: near_singularity > near_joint_limit > large_orientation_change > far_target >
medium_target > near_target). The exact thresholds, quantiles, and priority rule used for the
current data are recorded in `difficulty_definition.json`.

Other files:
- `point_ik_manifest.csv` - one row per sample (`sample_id, difficulty_id, difficulty_name,
  source_seed, position_distance_m, orientation_distance_deg, joint_distance_rad,
  initial_sigma_min, target_sigma_min, minimum_initial_limit_margin,
  minimum_target_limit_margin`).
- `difficulty_definition.json` - exact criteria, quantile thresholds, priority order, and
  per-group sample counts used to generate `point_ik_v1.npz`.
- `point_ik_checksum.json` - SHA256, file size, sample count, array names/shapes/dtypes,
  generation seed, and generator version/config hash for every generated file.

## Reachability guarantee

Every Tier 0 joint state and every Tier 1 target is either a directly sampled valid joint
configuration or the forward-kinematics image of one. No Cartesian target is chosen freely and
then merely assumed reachable.

## Loading

All `.npz` files load with `np.load(path, allow_pickle=False)`. No object-dtype arrays are used;
all per-sample metadata that would otherwise require an object array (difficulty names, group
names) is instead stored in the accompanying CSV/JSON files.
