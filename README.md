# Kassow KR810 Trajectory Tier 0-4 Dataset

A trajectory-first research dataset for the Kassow KR810 (7-DOF collaborative
manipulator), built around progressively harder tiers of kinematic
validation rather than a single end-to-end benchmark.

## Trajectory-first philosophy

The dataset is organized around trajectories (sequences of end-effector
waypoints and the joint solutions that realize them), not around isolated
point-to-point IK samples. Point IK (Tier 1) exists to validate the solver
before it is trusted on full trajectories (Tier 2-4).

## Tiers

- **Tier 0 - Kinematics validation**: verifies the forward kinematics (FK)
  and Jacobian implementations against the compiled MuJoCo model (finite
  difference checks, consistency checks) before any solver is trusted.
- **Tier 1 - Point DLS**: solves inverse kinematics (Damped Least Squares)
  for individual target poses, grouped by difficulty (near/medium/far,
  large orientation change, near joint limit, near singularity).
- **Tier 2 - Sequential DLS**: solves IK sequentially across an ordered
  chain of waypoints (warm-started from the previous solution), rather than
  independently per point.
- **Tier 3 - Trajectory tracking**: evaluates how well the sequential joint
  solutions reproduce the intended Cartesian trajectory (position/orientation
  tracking error, cross-track error).
- **Tier 4 - Joint feasibility and smoothness**: evaluates whether the
  resulting joint trajectories are smooth and kinematically feasible
  (velocity/acceleration/jerk behavior within operational limits).

## Out of scope

This dataset does **not** include:
- Tier 5 dynamics / torque-level control.
- Tier 6 MPDIK, PPO, MAPPO, or any learning-based control.

## Conventions

- Position: meters.
- Joint angle: radians.
- Orientation reporting: degrees.
- Quaternion order: `wxyz`.
- Rotation error: SO(3) logarithm (geodesic angle).

## Project structure

```
assets/         MuJoCo model, URDF, and meshes (not yet integrated - Stage 1)
configs/        Robot, frame, solver, benchmark, trajectory, evaluation configs
kinematics/     FK, Jacobian, DLS solver, rotation/quaternion utilities
generators/     Point-IK and trajectory target generators
algorithms/     DLS variants (point, sequential, warm/cold start)
evaluation/     Metrics: kinematics validation, IK, trajectory, ISO 9283-style,
                smoothness, feasibility, singularity, runtime, confidence intervals
pipelines/      Tier 0-4 runners and the combined Tier 0-4 entry point
benchmarks/     Point-IK and validation benchmark data (generated in a later stage)
trajectories/   Trajectory definitions and trial manifests (generated in a later stage)
schemas/        JSON Schema definitions for generated data
tests/          Test suite
notebooks/      Kaggle/local notebook scaffolding
```

## Asset status

The KR810 MuJoCo/URDF/mesh assets have **not** been integrated in Stage 1.
`assets/kr810.xml`, `assets/kr810.urdf`, and `assets/meshes/a810/` are
expected paths for a later stage; they do not exist yet in this repository.

## Running the Tier 0-4 pipeline

The combined entry point runs Tier 0 through Tier 4 in sequence for a given
experiment preset. Tier 0 is a mandatory gate: if it fails (non-finite
FK/Jacobian output, invalid rotations, or a Jacobian relative error above
threshold), the run stops before Tier 1 and Tier 1-4 are recorded as
`not_run`. Tier 1 always runs to completion and never gates the rest of the
pipeline (a low point-IK success rate is recorded but does not stop Tier 2-4).

### Smoke run (small subset, fast, for local iteration)

```bash
python -m pipelines.run_tier0_to_tier4 \
  --preset smoke \
  --output results/smoke_run
```

### Full run (entire dataset: 1200 point samples, all 8 trajectories, both methods)

```bash
python -m pipelines.run_tier0_to_tier4 \
  --preset full \
  --output results/full_run
```

A full run can be CPU- and time-intensive (8 trajectories x up to 360 trial
definitions x up to 400 waypoints x 2 methods); it does not require a GPU and
never renders MuJoCo. Use the CLI overrides below to scope it down if needed.

### Windows PowerShell (one-liner)

```powershell
.venv\Scripts\python.exe -m pipelines.run_tier0_to_tier4 --preset smoke --output results/smoke_run
```

### CLI overrides

CLI flags always take priority over the selected preset, which in turn takes
priority over `configs/*.json` defaults:

```
--preset smoke|full          (required)
--output PATH                (required)
--seed INT
--methods warm_start,cold_start
--trajectory-ids ID1,ID2
--trial-category repeatability|robustness|all
--trial-limit INT
--point-sample-limit INT
--waypoint-limit INT
--overwrite                  # remove and recreate an existing --output directory
--resume                     # reuse valid, up-to-date tier output instead of recomputing it
--no-plots
--log-level LEVEL
```

If `--output` already exists and neither `--overwrite` nor `--resume` is
given, the run refuses to proceed (no silent overwrite). `--resume` re-checks
each tier's required output files, JSON/CSV validity, and input signature
(configuration + relevant checksums); a tier is only skipped if all of those
still match, and Tier 3/4 are always recomputed whenever Tier 2 itself
recomputes. Each tier also has its own standalone CLI, e.g.
`python -m pipelines.run_tier0_kinematics --output results/tier0_only`.

### Output structure

```
results/<run>/
├── run_manifest.json        # provenance: versions, checksums, command line, per-tier status
├── resolved_config.json     # merged config snapshot (CLI > preset > configs/*.json)
├── FINAL_SUMMARY.json       # per-tier metrics + project acceptance criteria
├── tier0_kinematics/        # FK/Jacobian/singularity validation, gate result
├── tier1_point_dls/         # point-IK results, per-difficulty metrics, failures
├── tier2_sequential_dls/    # per-waypoint warm/cold-start results, warm-vs-cold comparison
├── tier3_trajectory_tracking/  # tracking, cross-track, ISO-9283-inspired, confidence intervals
└── tier4_joint_feasibility/    # smoothness, joint/velocity feasibility, singularity-along-path
```

Every tier folder also has a `figures/` subfolder (unless `--no-plots` is
passed). No file is created just to satisfy this structure: if a metric is
legitimately unavailable (e.g. acceleration utilization, since no
acceleration limits are configured), its status is reported as
`"unavailable"` rather than a fabricated number.

## Running locally vs. Kaggle

- **Local**: create a virtual environment, install `requirements.txt`, then
  run `python -m pipelines.run_tier0_to_tier4` as above.
- **Kaggle**: a notebook under `notebooks/` will wrap the same pipeline
  entry points once one exists; no notebook has been generated yet (Stage 6
  does not add one).

## On acceptance thresholds

Values such as 4 mm (RMSE), 6 mm (P95), 10 mm (max), and 10 degrees
(orientation P95) referenced in `configs/evaluation_config.json` and reported
in `FINAL_SUMMARY.json`'s `acceptance_criteria` are **project-defined
acceptance criteria** (`source: "project_criterion"`) used to judge this
dataset's own results. They are not ISO 9283 certification limits and no ISO
9283 certification is claimed; `evaluation/iso9283_metrics.py` computes
simulation-adapted accuracy/repeatability measures *inspired by* ISO 9283, not
a certified result. A smoke run's results are a fast sanity check, not the
dataset's official research result -- use a full run for that.

Tier 0-4 evaluates kinematics only (FK/Jacobian/IK and the resulting
trajectory tracking and joint feasibility); it does not include dynamics or a
controller.
