# MPDIK Kassow KR810

Trajectory-first kinematic validation/evaluation dataset for the Kassow KR810 (7-DOF). Current
branch: `feature/dataset-v2` (Phase 0: Dataset v2 audit + spec, no implementation yet).

## Dataset v1 vs Dataset v2

- **Dataset v1** (`VERSION`, `DATASET_MANIFEST.json`, `benchmarks/`, `trajectories/` at repo
  root) is an **immutable regression baseline**. Never modify its content, checksums, or the
  files listed in its checksum manifests (`benchmarks/point_ik/point_ik_checksum.json`,
  `benchmarks/validation/validation_checksum.json`, `trajectories/trajectory_manifest.csv`
  `sha256` column).
- **Dataset v2** covers Tier 0-4 kinematics only, under its own dataset root (see
  `specs/DLS_DATASET_V2_SPEC.md` for the locked layout). It must not reuse or mutate v1's
  `VERSION`, `DATASET_MANIFEST.json`, `configs/`, or `benchmarks/`/`trajectories/` paths.
- Out of scope for both v1 and v2: PPO, MPDIK, MAPPO, dynamics, actuators, controllers, torque
  control, collision, real-robot/TCP calibration, ISO certification claims.

## Rules

- Never edit, reformat, rename, or regenerate anything under Dataset v1's generated paths.
- All dataset generation must be **deterministic**: derive seeds explicitly (see
  `generators/_common.py::derive_seed`/`rng_from`), never touch global `numpy.random` state.
- Every dataset-v2 path must be resolved against an **explicit dataset root** — no
  current-working-directory assumptions, no absolute paths baked into configs/manifests/generated
  files (mirror `utils/dataset_locator.py`'s `Path(__file__)`-relative pattern, but parameterized
  by dataset root rather than hardcoded to repo root).
- The `frozen_test` split is never used to design generators, pick thresholds, or tune
  DLS/PPO/MPDIK — only to run after an evaluation config is locked.
- Run targeted tests (`pytest tests/test_<area>.py -q`) before a full `pytest -q` pass.
- Don't commit or push unless explicitly asked.

## Docs

- `specs/DLS_DATASET_V2_SPEC.md` — locked Dataset v2 design (counts, layout, naming, seed/split
  policy, generation policies, versioning/checksum policy).
- `docs/V2_REPO_AUDIT.md` — Phase 0 audit of the current repo against the v2 spec.
- `docs/V2_IMPLEMENTATION_LOG.md` — phase-by-phase implementation log for Dataset v2.
