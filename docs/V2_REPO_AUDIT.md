# Dataset v2 — Repository Audit (Phase 0)

- **Audit date**: 2026-07-22
- **Current branch**: `feature/dataset-v2` (matches required branch; confirmed via
  `git branch --show-current` before any file was touched)
- **Baseline commit**: `ce53bb10e05fa0d873b32fd6f211c7dd89398d89`
- **Tag/version**: no git tags exist (`git describe --tags --always` → `ce53bb10`, the short
  commit hash); repo-level `VERSION` file → `1.0.0`
- **Working tree at start**: clean, up to date with `origin/feature/dataset-v2`
  (`git status --short --branch` → `## feature/dataset-v2...origin/feature/dataset-v2`, no
  other output)
- **Baseline pytest result**: `pytest -q` → **322 passed, 0 failed, 0 skipped, 0 errors** (26.7s).
  No missing dependencies; `requirements.txt` deps (numpy, scipy, pandas, matplotlib, mujoco,
  pytest, tqdm, nbformat, jsonschema) were all importable. No environment blockers, no source
  defects observed.

## Dataset v1 structure (as it exists today)

- Root-level, dataset-scoped (not versioned-directory-scoped): `VERSION` (`"1.0.0"`),
  `DATASET_MANIFEST.json` (single manifest for the whole `kassow-kr810-trajectory-tier0-tier4`
  dataset; `status: "scaffold"`).
- `benchmarks/validation/` — Tier 0: `fk_test_states.npz`, `jacobian_test_states.npz`,
  `singularity_test_states.npz`, `validation_checksum.json`. Counts per
  `configs/benchmark_config.json`: 1000 FK, 200 Jacobian, 300 singularity samples.
- `benchmarks/point_ik/` — Tier 1: `point_ik_v1.npz` (1200 samples, 6 difficulty groups × 200,
  produced by `generators/generate_point_ik_dataset.py`), `point_ik_manifest.csv`,
  `difficulty_definition.json`, `point_ik_checksum.json`. No `split` field anywhere in the
  schema/arrays (`schemas/point_ik_schema.json`) — development/validation/frozen_test does not
  exist as a concept in v1.
- `trajectories/{line,circle,figure8,helix}/` — Tier 2-4: 8 files total (4 shapes × {fixed,
  variable} orientation), each 400 waypoints, plus `trajectory_manifest.csv` (one row per file,
  `sha256` column) and `trajectory_trials.csv` (360 rows: per trajectory, 10 repeatability repeats
  × 3 speed scales + 5 robustness initial configs × 3 speed scales). No `figure8`-style anchor
  variety: **all four shape generators call
  `generators/_trajectory_common.py::select_anchor(model_context, dls_config, seed, ANCHOR_TAG)`
  with the same `seed` (benchmark_config `random_seed=42`) and the same module-level
  `ANCHOR_TAG = 90`**, so line/circle/figure8/helix all anchor at the identical joint
  configuration — v1 effectively has one shared anchor, not per-shape/per-difficulty anchors.
- No `free_form` shape, no random-challenge trajectories, no canonical-vs-high-resolution
  distinction, no arc-length or cumulative-angular-displacement metadata anywhere in
  `trajectories/`.

## Reusable components (as-is)

- `kinematics/` — `forward_kinematics.py`, `jacobian.py`, `dls_solver.py`,
  `adaptive_damping.py`, `pose_error.py`, `rotation_utils.py`, `quaternion_utils.py`
  (already wxyz-only, per project instructions this must not be touched anyway),
  `joint_limit_utils.py`, `manipulability.py`, `singularity_metrics.py`, `model_loader.py`.
  Pure functions/model context, no v1-specific coupling.
- `algorithms/` — `point_dls.py`, `sequential_dls.py`, `warm_start_dls.py`,
  `cold_start_dls.py`, `result_types.py`. Not tied to v1 dataset paths.
- `utils/npz_utils.py` — `load_npz`/`save_npz` already enforce `allow_pickle=False`, reject
  object dtype, reject non-finite floats, and write atomically (temp file + `os.replace`).
  Directly satisfies the v2 NPZ rule with no change.
- `utils/file_checksum.py::sha256_file` — streaming SHA256, reusable for v2 checksum manifests.
- `utils/reproducibility.py::seed_everything`, `generators/_common.py::derive_seed`/`rng_from` —
  already `np.random.SeedSequence`-based, explicit-seed, no global random state. This is exactly
  the seed-derivation pattern the v2 seed policy should reuse.
- `generators/_trajectory_common.py` — quintic time scaling, sequential-DLS reachability
  validation (`validate_sequential_reachability`, `generate_validated_geometry`), orientation
  profile plumbing (`orientation_arrays`, calling `generators/generate_orientation_profile.py`).
  Geometry/validation logic is reusable; the anchor-selection and manifest/trial I/O parts are
  not (see "Requires extension" and "New components").
- `schemas/` — `point_ik_schema.json`, `trajectory_schema.json` are well-formed Draft 2020-12
  schemas with `additionalProperties: false`; the *pattern* (schema mirrors NPZ arrays 1:1) is
  reusable for v2 schemas, though the v2 schemas themselves must be new files (v1 schemas lack
  `split`, anchor metadata, arc-length, checksum fields).

## Components requiring extension

- `utils/dataset_locator.py` — `REPO_ROOT = Path(__file__).resolve().parent.parent` and every
  derived path (`BENCHMARKS_DIR`, `TRAJECTORIES_DIR`, `CONFIGS_DIR`, `SCHEMAS_DIR`, ...) is
  hardcoded to the repo root. Needs an explicit `dataset_root` parameter/function so v2 can point
  at a separate root without touching v1's constants.
- `generators/_common.py` — `REPO_ROOT`, `CONFIGS_DIR` are similarly hardcoded module constants;
  `load_benchmark_config`/`load_dls_config`/etc. always read from the single repo-root
  `configs/`. A v2 generator package needs the equivalent loaders parameterized by dataset root.
- `generators/_trajectory_common.py::select_anchor` — supports exactly one anchor-selection
  strategy (max joint-limit margin, `sigma_min` comfortably above threshold). V2's anchor policy
  needs three classes (`regular`, `near_limit`, `near_singular`); this function's search loop is
  reusable machinery but needs new acceptance predicates per class.
- `generators/_trajectory_common.py::upsert_manifest_row` / `replace_trial_rows` — hardcoded to
  `trajectories/trajectory_manifest.csv` / `trajectory_trials.csv` at repo root and to the v1
  `MANIFEST_COLUMNS`/`TRIAL_COLUMNS` (no `split`, no `trajectory_family`, no checksum column,
  no easy/medium/hard trial taxonomy). New column sets and a dataset-root-parameterized path are
  needed for v2.
- `pipelines/_common.py::_PROTECTED_DIRS` — the `--overwrite` safety allowlist is a hardcoded
  list of repo-root subdirectories. Should be extended (or given a v2 analogue) so v2 output
  writers get the same "never delete the repo root or a source dataset dir" protection without
  hardcoding v2's dataset root into this v1 list.
- `DATASET_MANIFEST.json` / `VERSION` — the pattern (single JSON manifest + single version
  string) is reusable structurally, but each must become dataset-root-scoped: v1 keeps its
  current root-level files untouched, v2 needs its own `VERSION` and `DATASET_MANIFEST.json`
  under its own root.

## New components required (per the locked v2 design)

- A v2 dataset-root resolver (v2 analogue of `utils/dataset_locator.py`, parameterized rather
  than hardcoded).
- V2 generators for: Tier 0 (1000 FK / 1000 Jacobian / 600 singularity — different counts than
  v1's 1000/200/300), Tier 1 Point-IK with development/validation/frozen_test splits and 1000
  samples/group (vs. v1's 200/group, no splits), core trajectories with 3 anchor classes × 12
  anchors × 5 shapes (including new `free_form`) × 2 orientation modes, random challenge
  trajectories (90 total, split 30/30/30), and the easy/medium/hard trial generator (630 trials
  total, 3 per trajectory — structurally different from v1's repeatability/robustness taxonomy).
- Canonical (400-waypoint) + high-resolution source dual representation per trajectory, plus
  arc-length and cumulative-angular-displacement metadata — none of this exists in v1's
  `build_trajectory_arrays`.
- Per-trajectory/per-sample SHA256 checksum fields stored *inside* the record (v1 only has
  file-level checksums in manifest CSV/JSON, not embedded per-sample/per-trajectory content
  hashes for leakage checks).
- V2 schemas (`schemas/` equivalents) covering split, anchor class, trajectory family,
  orientation mode, arc-length/angular-displacement metadata, checksum fields.
- V2 checksum manifest and generation report, separate from v1's
  `benchmarks/point_ik/point_ik_checksum.json` / `validation_checksum.json` /
  `trajectory_manifest.csv`.
- Anti-leakage validation across splits (no shared `anchor_id`, Point-IK `sample_id`/content
  hash, random path seed, `trajectory_id`/hash, or `trial_id` across development/validation/
  frozen_test) — no such check exists anywhere in the current test suite or generators.

## Immutable components (must not be modified in any phase)

- `VERSION`, `DATASET_MANIFEST.json` (root-level, v1).
- `benchmarks/point_ik/point_ik_v1.npz`, `point_ik_manifest.csv`, `difficulty_definition.json`,
  `point_ik_checksum.json`.
- `benchmarks/validation/fk_test_states.npz`, `jacobian_test_states.npz`,
  `singularity_test_states.npz`, `validation_checksum.json`.
- `trajectories/**/*.npz`, `trajectory_manifest.csv`, `trajectory_trials.csv`.
- `kinematics/` FK, Jacobian, pose-error, and DLS-solver formulas (per explicit task
  instructions, regardless of dataset version).
- `algorithms/` DLS variants (same reason).

## Path-resolution audit

- `utils/dataset_locator.py`: every path is `Path(__file__).resolve().parent.parent / <subdir>`
  — no CWD dependency, no absolute-path literals. Good pattern; not yet generalized to a
  parameterized root.
- `generators/_common.py`, `generators/_trajectory_common.py`, `pipelines/_common.py`: same
  `Path(__file__)`-relative `REPO_ROOT` pattern, independently redefined in each module rather
  than imported from one place (`generators/_common.py::REPO_ROOT` and
  `utils/dataset_locator.py::REPO_ROOT` are two separate constants that happen to compute the
  same value). No CWD dependency found in any of the three.
  Note: `pipelines/_common.py` imports `REPO_ROOT` from `utils.dataset_locator` (single source),
  while `generators/_common.py` and `generators/_trajectory_common.py` (which imports `REPO_ROOT`
  from `generators._common`) redefine it independently — an existing minor duplication, not a bug.
- `utils/config_loader.py::load_json_config`: explicitly documents that relative paths are
  resolved against CWD, not a fixed root — callers (`generators/_common.py`,
  `pipelines/_common.py`) always pass absolute `Path` objects built from `REPO_ROOT`, so no CWD
  bug surfaces today, but the function itself has no root-anchoring guarantee.
- No absolute path literals (`D:\`, `C:\`, `/home`, etc.) found in any config, generator, or
  manifest file inspected; `tests/test_point_dataset.py::test_no_absolute_paths_in_manifest_or_metadata`
  and `tests/test_trajectory_files.py::test_no_absolute_paths_in_manifest_or_trials` actively
  assert this for v1's generated manifests.
- `pipelines/_common.py::_PROTECTED_DIRS` / `is_safe_to_remove`: existing safety mechanism to
  stop `--overwrite` from deleting source directories; hardcodes the v1 repo-root subdirectory
  list.

## Dataset-root readiness

Not ready. No module in the repo accepts a `dataset_root` argument; `REPO_ROOT` is always
derived from file location and always points at the single repo root. Every generator, loader,
and pipeline writes into (or reads from) `benchmarks/`, `trajectories/`, `configs/`, `schemas/`
directly under that one root. Introducing Dataset v2 requires adding an explicit root parameter
threaded through the v2-equivalents of `utils/dataset_locator.py` and `generators/_common.py`,
without changing the v1 constants those v1 modules currently export.

## Version/schema/checksum readiness

- Versioning: only a single flat `VERSION` file and a single `DATASET_MANIFEST.json["version"]`
  field exist; no concept of multiple coexisting dataset versions in one repo today.
- Schema: `schemas/*.json` are per-artifact-type, not per-dataset-version; v1's schemas hardcode
  v1's field set (no split, no anchor class, no checksum-in-record).
- Checksum: `utils/file_checksum.py::sha256_file` plus per-generator checksum JSON files
  (`point_ik_checksum.json`, `validation_checksum.json`) already give a solid, reusable pattern:
  file-level SHA256 + array shape/dtype/sample-count metadata, built by
  `generators/_common.py::build_checksum_entry`. V2 needs its own checksum manifest file(s) using
  the same pattern, rooted at the v2 dataset root, and (per the locked design) content hashes at
  the sample/trajectory level for anti-leakage checks, which `build_checksum_entry` does not
  currently compute.

## Backward-compatibility risks

- Any change to `utils/dataset_locator.py` or `generators/_common.py` module-level constants
  (`REPO_ROOT`, `BENCHMARKS_DIR`, etc.) would affect v1 code paths too, since v1 and v2 generators
  would otherwise share these modules. Mitigation direction: add new v2-specific,
  root-parameterized functions alongside the existing constants rather than editing them in
  place.
- `pipelines/_common.py::_PROTECTED_DIRS` currently only protects v1 directories; if a v2
  generator is later pointed at an output path inside the v1 tree by mistake, nothing in the
  current codebase would stop it except this v1-scoped allowlist (which would correctly refuse,
  since v2's dataset root should not be one of the listed v1 dirs — but only if v2's root is
  chosen outside the repo's existing top-level directories).
- Test suite (`tests/test_point_dataset.py`, `tests/test_trajectory_files.py`) hardcodes v1
  counts (`EXPECTED_TOTAL_SAMPLES = 1200`, `EXPECTED_NUM_WAYPOINTS = 400`,
  `TRAJECTORY_FILES` = the 8 v1 files) and imports `generators.generate_point_ik_dataset`
  directly — safe as long as v2 generators live in new modules/paths rather than modifying these.

## Blockers

- None that block continued audit/spec work. No missing dependency, no failing test, no
  ambiguous git state.
- Open design blocker for Phase 1 (not resolved here, per the spec's "unresolved blockers"
  section): the anchor-selection acceptance criteria for the `near_limit` and `near_singular`
  anchor classes are not yet defined with concrete thresholds (v1's `select_anchor` only
  implements a "well-conditioned interior" class).

## Recommended implementation order

1. Add a root-parameterized dataset-root resolver (v2 analogue of `utils/dataset_locator.py`)
   and v2 config/schema/checksum-manifest scaffolding under the new dataset root — no generation
   logic yet.
2. Tier 0 generators (FK/Jacobian/singularity states) at the new counts (1000/1000/600), reusing
   `kinematics/` unchanged.
3. Point-IK Tier 1 generator with the 6 difficulty groups × 1000 samples and the
   development/validation/frozen_test split logic, including anti-leakage checks.
4. Anchor generator supporting all three classes (`regular`, `near_limit`, `near_singular`),
   12 anchors total, with split isolation.
5. Core trajectory generator (5 shapes × 2 orientation modes × 12 anchors = 120), producing both
   canonical (400-waypoint) and high-resolution source representations plus arc-length/angular
   metadata.
6. Random challenge trajectory generator (90, split 30/30/30).
7. Trial generator (easy/medium/hard × 210 trajectories = 630 trials).
8. Checksum/manifest/generation-report writers, then full schema + anti-leakage validation pass.
9. Frozen-test protocol enforcement (lock evaluation config before any frozen_test run).
