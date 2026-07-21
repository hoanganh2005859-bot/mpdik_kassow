# benchmarks/

Benchmark data for Tier 0 (kinematics validation) and Tier 1 (point IK).

- `point_ik/` - point-to-point IK benchmark samples, grouped by difficulty
  (populated in a later stage; empty in this scaffold).
- `validation/` - FK, Jacobian, and singularity validation states used to
  sanity-check the kinematics implementation before it is used by any
  solver (populated in a later stage; empty in this scaffold).

No benchmark data (including `.npz` files) has been generated in this
stage.
