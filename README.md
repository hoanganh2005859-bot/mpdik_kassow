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

## Expected usage (not yet functional)

```
python -m pipelines.run_tier0_to_tier4 --preset smoke --output results
```

This command is a scaffold reference only. It will not run until the asset
integration, kinematics, algorithm, generator, and evaluation stages are
implemented in later stages.

## Expected Tier 0-4 output (once implemented)

- Tier 0: FK/Jacobian validation states and pass/fail summary.
- Tier 1: per-point IK results grouped by difficulty, with success/error metrics.
- Tier 2: per-waypoint sequential IK results per trajectory.
- Tier 3: trajectory tracking error metrics (position, orientation, cross-track).
- Tier 4: joint smoothness and feasibility metrics per trajectory trial.

## Running locally vs. Kaggle (framework only)

- **Local**: create a virtual environment, install `requirements.txt`, then
  invoke the relevant `pipelines/run_tier*.py` module once implemented.
- **Kaggle**: a notebook under `notebooks/` will wrap the same pipeline
  entry points once they exist; no notebook has been generated yet.

## On acceptance thresholds

Values such as 4 mm (RMSE), 6 mm (P95), 10 mm (max), and 10 degrees
(orientation P95) referenced in `configs/evaluation_config.json` are
**project-defined acceptance criteria** used to judge this dataset's own
results. They are not ISO 9283 certification limits and no ISO 9283
certification is claimed.
