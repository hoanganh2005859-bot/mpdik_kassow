# Dataset v2 — Implementation Log

## Baseline

- Branch: `feature/dataset-v2`
- Baseline commit: `ce53bb10e05fa0d873b32fd6f211c7dd89398d89`
- Baseline version/tag: `VERSION` = `1.0.0`; no git tags exist (`git describe --tags --always` →
  `ce53bb10`)
- Baseline test result: `pytest -q` → 322 passed, 0 failed, 0 skipped, 0 errors

## Phase 0 — Audit + Spec

- **Status**: complete.
- **Files created/updated**:
  - `CLAUDE.md` (new)
  - `docs/V2_REPO_AUDIT.md` (new)
  - `specs/DLS_DATASET_V2_SPEC.md` (new)
  - `docs/V2_IMPLEMENTATION_LOG.md` (this file, new)
- **Decisions locked** (see `specs/DLS_DATASET_V2_SPEC.md` for full detail): all Dataset v2
  counts from the user's design (Tier 0: 1000/1000/600; Point-IK: 6,000 total, 1,000/group,
  split 1,200/1,200/3,600; core trajectories: 120 = 5 shapes × 2 orientation modes × 12 anchors;
  random challenge: 90, split 30/30/30; 210 trajectories total, 84,000 canonical poses, 630
  trials); dataset-v2-own `VERSION`/`DATASET_MANIFEST.json`/configs/schemas/checksum
  manifest/generation report, separate dataset root, no CWD dependence, no absolute paths,
  deterministic seeding with no global random state, `allow_pickle=False` NPZ, split
  anti-leakage rules, frozen-test protocol.
- **Blockers**: numeric acceptance thresholds for `near_limit`/`near_singular` anchor classes are
  not yet defined (v1's `generators/_trajectory_common.py::select_anchor` only implements the
  `regular` predicate) — flagged as `[BLOCKER]` in `specs/DLS_DATASET_V2_SPEC.md` section G, to
  be resolved at the start of the anchor-generation implementation phase.
- **Recommended Phase 1**: dataset-v2 root resolver + config/schema/checksum-manifest scaffolding
  only (no generation logic) — see `docs/V2_REPO_AUDIT.md` "Recommended implementation order",
  step 1.
- **Dataset v1 confirmation**: not modified in this phase. `git status --short` after document
  creation shows only the four files listed above as new/changed; no file under `assets/`,
  `benchmarks/`, `trajectories/`, `configs/`, `kinematics/`, `algorithms/`, `generators/`,
  `evaluation/`, `pipelines/`, `tests/`, `DATASET_MANIFEST.json`, or `VERSION` was touched.

## Phase 1 — Dataset-root, config/schema, and checksum-manifest scaffolding

- **Status**: complete. No Tier 0-4 Dataset v2 data was generated in this phase.
- **Baseline before this phase**: branch `feature/dataset-v2`, commit `aab50a0dc09371523346a90aae37c8b107cdb8b7`,
  clean working tree, `pytest -q` → 322 passed, 0 failed/skipped/errored.

### Files added/modified

- **Modified** (extended, not rewritten): `utils/dataset_locator.py` -- added
  `resolve_dataset_root(dataset_root=None, require_exists=True)` (the one central root-resolution
  implementation: `None` → unchanged `REPO_ROOT` default; an explicit root is anchored to
  `Path.cwd()` only if relative, then `.resolve()`d, then existence/directory-checked with an
  actionable `ModelConfigurationError`) and `dataset_paths_for(root) -> DatasetPaths` (the v1
  layout, parameterized by root). The existing module-level constants
  (`ASSETS_DIR`/`BENCHMARKS_DIR`/`TRAJECTORIES_DIR`/`CONFIGS_DIR`/`SCHEMAS_DIR`/`MODEL_PATH`/etc.)
  are now derived by calling `dataset_paths_for(REPO_ROOT)` once at import time -- their values
  are byte-identical to before, verified by
  `tests/test_dataset_root_resolution.py::test_dataset_paths_for_explicit_root_matches_v1_constants`.
- **New package** `dataset_v2/` (parallel to `generators/`/`pipelines/`/`utils/`, kept separate
  from v1's `generators/` so nothing v1-specific is imported):
  - `dataset_v2/locator.py` -- `DatasetV2Paths`, `dataset_v2_paths(dataset_root)` (v2's own
    layout: `configs/`, `schemas/`, `checksums/`, `tier0_validation/`, `tier1_point_ik/`,
    `anchors/`, `trajectories/{development,validation,frozen_test}/`, `trials/`, `references/`,
    `reports/`), `require_dataset_v2_root(dataset_root)` (validates root exists + has
    `DATASET_MANIFEST.json`, actionable error otherwise), `relative_to_dataset_v2_root`. Builds on
    `utils.dataset_locator.resolve_dataset_root` rather than re-implementing root normalization.
  - `dataset_v2/config_templates.py` -- builders for all 11 `configs/*.json` files (dataset
    metadata, robot reference, seed policy, Tier 0, Point-IK, anchor, core trajectory, random
    challenge, trial, split policy, evaluation defaults), using only [LOCKED] counts/names from
    `specs/DLS_DATASET_V2_SPEC.md` section B/D. `near_limit`/`near_singular` anchor acceptance
    thresholds are written as `{"status": "unresolved", ...}` (spec section G `[BLOCKER]`) --
    no numeric threshold was invented.
  - `dataset_v2/schemas.py` -- builders for 9 Draft-2020-12 `schemas/*.json` files
    (`dataset_manifest`, `generation_config`, `anchor`, `tier0_state`, `point_ik`, `trajectory`,
    `trial`, `validation_report`, `checksum_manifest`), all `additionalProperties: false`,
    covering split/difficulty/orientation-mode/trajectory-family enums, SHA256 pattern, wxyz
    quaternion arrays, non-negative counts.
  - `dataset_v2/manifest.py` -- builds `DATASET_MANIFEST.json` content (scope,
    `includes_ppo/mpdik/mappo/dynamic_control: false`, all locked counts, split sizes, pointers to
    `checksums/CHECKSUM_MANIFEST.json` and `reports/GENERATION_REPORT.json`); `generated`/`frozen`
    are always `false` from this builder.
  - `dataset_v2/checksums.py` -- `build_source_config_fingerprint`/`build_checksum_manifest`
    (streaming SHA256 via the existing `utils.file_checksum.sha256_file`, dataset-root-relative
    filenames, deterministic sort order, never checksums its own manifest file, never a fabricated
    entry for a nonexistent file), `verify_checksum_manifest` (recompute + report mismatches),
    `content_hash_of_record` (sorted-key, fixed-precision content hash for future anti-leakage
    checks -- infra only, not yet called by any generator).
  - `dataset_v2/scaffold.py` -- `create_dataset_v2_scaffold(dataset_root, master_seed,
    overwrite=False)`: creates the full directory tree, writes `VERSION`, all configs/schemas,
    `DATASET_MANIFEST.json`, then the checksum manifest over what was actually written. Refuses to
    scaffold onto `REPO_ROOT` or any existing Dataset v1 subdirectory. Refuses to overwrite an
    existing scaffold unless `overwrite=True`. Writes no NPZ/CSV, no fabricated sample data.
- **New CLI** `pipelines/run_dataset_v2_scaffold.py --dataset-root PATH [--master-seed N]
  [--overwrite]` -- the CLI entry point for the above; `--dataset-root` is required (Dataset v2 has
  no repo-root/CWD-implicit default by design). No existing Dataset v1 CLI was modified, and none
  needed a `--dataset-root` flag added since none of them changed behavior.
- **New tests**:
  - `tests/test_dataset_root_resolution.py` -- default resolution == `REPO_ROOT`; explicit
    `dataset_root=REPO_ROOT` resolves identically and Dataset v1 loads through it; resolution is
    independent of CWD (`monkeypatch.chdir`); invalid/non-directory roots raise
    `ModelConfigurationError` with an actionable message; no absolute-path literals in the module's
    code (docstring examples excluded).
  - `tests/test_dataset_v2_scaffold.py` -- scaffold layout/idempotency/overwrite semantics; refuses
    v1 paths; manifest declares `generated: false`/`frozen: false` and matches every locked count
    (1000/1000/600 Tier 0; 6000 Point-IK, 1000/group, 1200/1200/3600 split; 12 anchors 6/3/3; 120
    core trajectories; 90 random challenge 30/30/30; 210 total; 84,000 canonical poses; 630
    trials); all 11 configs parse; `near_limit`/`near_singular` thresholds are `"unresolved"`, not
    invented; all 9 schemas are valid Draft 2020-12 and accept/reject sample point-IK records
    correctly; checksum manifest only lists real files with correct relative paths and SHA256,
    detects tampering, never self-references; content-hash determinism; CLI end-to-end round trip
    including v1-untouched snapshot diffing (mirrors `tests/test_pipeline_smoke.py`'s pattern) and
    rejection of a v1 root/re-overwrite-without-flag.

### Dataset-root behavior

- Default (no `dataset_root` argument, no `--dataset-root` flag on any existing v1 CLI): 100%
  unchanged -- confirmed by the full `pytest -q` pass below and by
  `tests/test_dataset_root_resolution.py`.
- Explicit root: `resolve_dataset_root`/`dataset_paths_for` (v1-shaped layout) and
  `dataset_v2_paths`/`require_dataset_v2_root` (v2-shaped layout) both take an explicit
  `str | Path`, never assume CWD, normalize + validate with actionable errors, and produce only
  root-relative paths in any generated JSON/manifest.

### Config/schema/checksum scaffold status

- 11 configs, 9 schemas, 1 root manifest, 1 checksum manifest -- all generated by code (not
  committed as a fixed instance in the repo, since Dataset v2's root is caller-supplied and must
  never be baked into the repository per spec section C/O). Exercised end-to-end by the test
  suite against `tmp_path`.
- All counts match `specs/DLS_DATASET_V2_SPEC.md` section B exactly. Unresolved items (anchor
  `near_limit`/`near_singular` thresholds, evaluation-acceptance thresholds) are explicitly marked
  `"unresolved"`/`"not_yet_defined"`, not guessed.

### Tests

- Targeted: `pytest tests/test_dataset_root_resolution.py tests/test_dataset_v2_scaffold.py -q` →
  36 passed.
- Dataset v1 smoke/backward-compat: `pytest tests/test_pipeline_smoke.py tests/test_point_dataset.py
  tests/test_trajectory_files.py -q` → 130 passed.
- Full suite: `pytest -q` → **358 passed, 0 failed, 0 skipped, 0 errors** (322 baseline + 36 new).

### Blockers

- Same one carried over from Phase 0, still unresolved and correctly not touched: the exact
  numeric acceptance thresholds for the `near_limit`/`near_singular` anchor classes
  (`specs/DLS_DATASET_V2_SPEC.md` section G `[BLOCKER]`). Recorded as `"unresolved"` in
  `configs/anchor_config.json`; must be decided before anchor generation (Phase 2+) can run.

### Confirmations

- Dataset v2 data generation: **not performed**. No NPZ/CSV was written anywhere; the scaffold
  writes only `VERSION`, JSON configs/schemas/manifests, and `.gitkeep` placeholders in otherwise-
  empty directories.
- Dataset v1: **not modified**. `git diff --name-only` after this phase shows only
  `utils/dataset_locator.py` as a modified file (extended, values unchanged); every new file is
  additive (`dataset_v2/`, `pipelines/run_dataset_v2_scaffold.py`,
  `tests/test_dataset_root_resolution.py`, `tests/test_dataset_v2_scaffold.py`,
  `docs/V2_IMPLEMENTATION_LOG.md`). No file under `assets/`, `benchmarks/`, `trajectories/`,
  `configs/`, `schemas/`, `kinematics/`, `algorithms/`, `DATASET_MANIFEST.json`, or `VERSION` (v1)
  was touched.

### Recommended Phase 2

Per `docs/V2_REPO_AUDIT.md`'s recommended order, next is the Tier 0 generator (FK/Jacobian/
singularity states at 1000/1000/600) writing into `<dataset_v2_root>/tier0_validation/`, reusing
`kinematics/` unchanged and the seed-derivation scheme now recorded in
`configs/seed_policy.json`. The anchor `near_limit`/`near_singular` threshold blocker should be
resolved (a decision from the user, not an implementation guess) before or alongside the anchor
generator that Point-IK/core-trajectory generation will depend on.

## Phase 2 — Tier 0 generator, validator, CLI (FK / Jacobian / singularity)

- **Status**: complete. Tier 0 only -- no Point-IK, anchors, trajectories, or trials.
- **Baseline before this phase**: branch `feature/dataset-v2`, commit
  `67f753195deb40b300e9815f8b93389544d173ab`, clean working tree, `pytest -q` -> 358 passed,
  0 failed/skipped/errored.

### Files added/modified

- **New**: `dataset_v2/tier0_generation.py` -- the generator. Builds FK (5 groups x 200),
  Jacobian (5 groups x 200, including a real-`sigma_min`-ranked `low_sigma` group), and
  singularity (3 groups x 200, classified by real computed `sigma_min` against v1's shared
  `configs/dls_config.json:singularity_sigma_threshold`) states. Calls
  `kinematics/forward_kinematics.py`, `kinematics/jacobian.py`,
  `kinematics/singularity_metrics.py`, `kinematics/joint_limit_utils.py`, and
  `kinematics/manipulability.py` unchanged -- no FK/Jacobian/SO(3)/singularity/DLS math was
  touched or reimplemented anywhere in this phase. Seeds are derived exclusively via
  `generators/_common.py::derive_seed` from `configs/seed_policy.json`'s `master_seed` and the
  `tier0` component tag (10) -> per-state-type tags (fk=1, jacobian=2, singularity=3) -> per-group
  tags; no `numpy.random` global state is touched anywhere. Exact-duplicate joint states are
  redrawn in place (never silently kept, never padded); an insufficient singularity candidate
  pool raises a hard error reporting the pool's regular/moderate/near-singular distribution
  rather than relaxing thresholds or duplicating states. Writes atomically (NPZ via
  `utils/npz_utils.py::save_npz`, JSON via temp-file-then-`replace`), rejects pre-existing output
  without `overwrite=True`, and after a real (non-dry-run) generation updates
  `DATASET_MANIFEST.json`'s `counts.tier0` with the actual generated counts/group counts (via the
  new `dataset_v2/manifest.py::apply_tier0_generation_status`) and rebuilds
  `checksums/CHECKSUM_MANIFEST.json` to include the new files.
- **New**: `dataset_v2/tier0_validation.py` -- the independent validator. Reuses
  `evaluation/kinematics_validation.py` (`validate_fk_states`/`validate_jacobian_states`, the
  same functions the existing v1 Tier 0 gate uses) for the actual FK/Jacobian numerical checks,
  adding only v2-specific structural checks: exact total/group counts, duplicate detection,
  operational-limit compliance, and singularity-classification-vs-threshold consistency (using
  the threshold/`moderately_conditioned_upper_bound` recorded in
  `singularity_test_states_v2_metadata.json` at generation time, not re-derived). Returns a
  `Tier0ValidationReport` with a `passed` flag and itemized `reasons`; never raises on a failed
  check, only on a missing/malformed input file.
- **New**: `pipelines/run_dataset_v2_tier0_generation.py` -- CLI
  (`python -m pipelines.run_dataset_v2_tier0_generation --dataset-root PATH`). Supports
  `--master-seed`, `--overwrite`, `--validate-only`, `--dry-run`, and count/candidate-pool-size
  overrides documented as test/smoke-only (the locked full mode is 1000/1000/600). Exit codes:
  `0` success, `1` validation failure, `2` usage/configuration error (missing scaffold, output
  already exists without `--overwrite`, etc.).
- **Modified** (extended, not rewritten):
  - `dataset_v2/config_templates.py::tier0_config()` -- added the sampling-policy parameters
    (margins, home perturbation, FD epsilon, candidate-pool sizes/multipliers) the generator
    reads, plus an explicit `singularity_threshold_source` string; counts (1000/1000/600) and all
    Phase 1 config files/tests are unaffected.
  - `dataset_v2/manifest.py` -- added `apply_tier0_generation_status` (returns a copy of the
    manifest dict with `counts.tier0` replaced by actual generated counts; never touches the
    dataset-wide `generated`/`frozen`/`status` flags, since Tier 0 alone being generated does not
    mean Tier 1-4 are).
  - `dataset_v2/checksums.py` -- added `build_generated_data_fingerprint` (fingerprints any files
    under the generation-output directories, empty at scaffold time) and wired it into
    `build_checksum_manifest`'s `generated_data_checksum` category (previously always `[]`);
    `source_config_fingerprint` behavior is unchanged, verified by the untouched Phase 1 scaffold
    tests.
- **New tests**: `tests/test_dataset_v2_tier0_generation.py` (24 tests: determinism/reseed,
  exact small-fixture group counts, `allow_pickle=False` loads, operational-limit compliance, no
  duplicates, `low_sigma` group driven by real computed `sigma_min`, singularity classification
  vs. threshold, condition-number-never-NaN, overwrite protection, dry-run writes nothing,
  CWD-independence, manifest/checksum-manifest updates, metadata required fields, no absolute
  paths, `tier0_state_schema.json` accepts a representative record, missing-scaffold rejection,
  the locked-1000/1000/600 config-resolution integration check, and CLI dry-run/generate/
  validate/overwrite-rejection flows) and `tests/test_dataset_v2_tier0_validation.py` (9 tests:
  passes on a clean fixture, and separately detects a wrong total count, wrong group count,
  duplicate state, out-of-limit state, oversized-FD-epsilon Jacobian relative-error violation,
  and a hand-corrupted singularity misclassification, plus CLI exit-code 0/1 on success/failure).
  All fixtures use small counts (FK=10, Jacobian=10, singularity=9) for speed; one CLI-level test
  runs the full locked 1000/1000/600 generation manually (see below), not as part of the regular
  suite.

### Seed policy actually used

`master_seed` (from `configs/seed_policy.json`, override via `--master-seed`) ->
`derive_seed(master_seed, component_tags["tier0"]=10)` = tier0 component seed ->
`derive_seed(tier0_component_seed, {fk:1, jacobian:2, singularity:3})` = per-state-type seed ->
`derive_seed(state_type_seed, group_id)` = per-group seed (the RNG actually used to draw that
group's candidates). Every seed used is recorded verbatim in
`tier0_validation/tier0_generation_report.json` and each per-file metadata JSON, so any seed is
traceable back to `(master_seed, tag_path)` per spec section E. Verified byte-identical NPZ output
across two independent scaffold+generate runs at the same seed
(`test_deterministic_generation_same_seed`), and differing output at a different seed
(`test_different_seed_produces_different_content`).

### Counts

Resolved from `configs/tier0_config.json` (written by the Phase 1 scaffold, extended this phase
with the sampling-policy fields): 1000 FK (5 groups x 200), 1000 Jacobian (5 groups x 200), 600
singularity (3 groups x 200) -- confirmed locked via
`test_resolved_full_counts_in_config_are_locked_1000_1000_600`.

### Full generation (manual, temporary directory, not committed)

Ran the CLI against a scaffold in the OS temp directory (outside the repo, deleted afterward),
`--master-seed 42`, no overrides (full locked mode):

- `python -m pipelines.run_dataset_v2_tier0_generation --dataset-root <temp> --master-seed 42` ->
  wrote 1000 FK / 1000 Jacobian / 600 singularity states in ~7 seconds; group counts exactly
  200/group for all three files (confirmed from `tier0_generation_report.json`).
- `python -m pipelines.run_dataset_v2_tier0_generation --dataset-root <temp> --validate-only` ->
  `fk=1000 jacobian=1000 singularity=600 max_jacobian_relative_error=2.859e-10 passed=True`.
- `singularity_threshold=0.03` (source: repo root `configs/dls_config.json`, v1's shared,
  unmodified DLS config), `moderately_conditioned_upper_bound=0.09`.
- `DATASET_MANIFEST.json`'s `counts.tier0` updated with `generated: true`,
  `full_locked_counts: true`, and the exact group counts above; dataset-wide `generated`/`frozen`
  flags remain `false` (Tier 1-4 are not generated).
- No NPZ/CSV/generated JSON from this run was added to the repository or Git; the temp directory
  was deleted after inspection.

### Tests

- Targeted: `pytest tests/test_dataset_v2_tier0_generation.py tests/test_dataset_v2_tier0_validation.py -q`
  -> 33 passed.
- Existing Tier 0/Dataset v2 regression:
  `pytest tests/test_dataset_v2_scaffold.py tests/test_dataset_root_resolution.py tests/test_pipeline_tier0.py -q`
  -> 39 passed.
- Dataset v1 backward compatibility:
  `pytest tests/test_pipeline_smoke.py tests/test_point_dataset.py tests/test_trajectory_files.py -q`
  -> 130 passed.
- Full suite: `pytest -q` -> **391 passed, 0 failed, 0 skipped, 0 errors** (358 baseline + 33 new).

### Blockers

- None new. The anchor `near_limit`/`near_singular` threshold blocker (spec section G) carries
  over unresolved and untouched -- out of scope for this Tier-0-only phase.

### Confirmations

- Dataset v1: **not modified**. `git status --short`/`git diff --name-only` after this phase show
  only `dataset_v2/checksums.py`, `dataset_v2/config_templates.py`, `dataset_v2/manifest.py`
  modified (all Phase 1 files, extended additively) and `dataset_v2/tier0_generation.py`,
  `dataset_v2/tier0_validation.py`, `pipelines/run_dataset_v2_tier0_generation.py`,
  `tests/test_dataset_v2_tier0_generation.py`, `tests/test_dataset_v2_tier0_validation.py` as new,
  untracked files. No file under `assets/`, `benchmarks/`, `trajectories/`, `configs/`,
  `schemas/`, `kinematics/`, `algorithms/`, root `DATASET_MANIFEST.json`, or root `VERSION` was
  touched.
- Full Dataset v2 Tier 0 (2,600 states): generated and validated in a temporary directory (see
  above) to confirm the locked counts/runtime are real, but **not** committed to the repository --
  no NPZ/CSV/generated JSON under this phase's file list above is committed data.

### Recommended Phase 3

Per `docs/V2_REPO_AUDIT.md`'s recommended order, next is the Point-IK Tier 1 generator (6,000
samples, 6 difficulty groups x 1,000, `development`/`validation`/`frozen_test` split
1,200/1,200/3,600, anti-leakage checks per spec section K) -- or, if tackled first, the anchor
generator's `near_limit`/`near_singular` threshold blocker (spec section G) that Point-IK's
`near_joint_limit`/`near_singularity` groups and the later core-trajectory anchors both depend on.
Both remain explicitly out of scope for Phase 2.

## Phase 2.5 — Threshold calibration (near_joint_limit / near_singularity / moderately_conditioned / regular)

- **Status**: complete. Calibrates thresholds only; no Point-IK samples, anchors, trajectories, or
  trials were generated.
- **Baseline before this phase**: branch `feature/dataset-v2`, commit
  `77674d023156e05ebe7ce1740fb5dd1317d65df1`, clean working tree, `pytest -q` -> 391 passed,
  0 failed/skipped/errored.

### Files added/modified

- **New**: `dataset_v2/threshold_calibration.py` -- calibration module. Builds two deterministic,
  duplicate-free candidate pools (a generic uniform-interior pool with a per-joint
  range-proportional margin, and a singularity-biased pool reusing
  `dataset_v2/tier0_generation.py::_build_singularity_candidate_pool` unchanged), computes real
  normalized/absolute joint-limit margin, `sigma_min`/`sigma_max`/`condition_number`/
  `numerical_rank`/manipulability/FK position per candidate (`kinematics/` unchanged, no DLS), and
  derives/reports the four locked thresholds. Never writes a candidate pool to disk, never touches
  `numpy`'s global random state, never uses DLS convergence/error, `q_target`-as-solution, or
  frozen-test data.
- **New**: `pipelines/run_dataset_v2_threshold_calibration.py` -- CLI
  (`python -m pipelines.run_dataset_v2_threshold_calibration --seed N`), prints the scalar
  calibration report as JSON; `--report-json PATH` optionally writes it to a caller-supplied path
  (never a Dataset v2 root or the repo).
- **Modified** (extended, not rewritten): `dataset_v2/config_templates.py` -- added
  `difficulty_threshold_config()` (new `configs/difficulty_thresholds.json`, `status: "locked"`,
  containing the calibrated thresholds, candidate pool sizes, calibration seed, and classification
  priority) and updated `anchor_config()`'s `near_limit`/`near_singular` acceptance criteria from
  `"unresolved"` to `"locked"`, referencing the same calibrated values (never a second,
  independently-invented number). `all_configs()` now returns 12 files (was 11).
- **Modified**: `specs/DLS_DATASET_V2_SPEC.md` section G ("Anchor policy") -- the previously
  `[BLOCKER]` near_limit/near_singular acceptance thresholds are now `[LOCKED]` with concrete
  numeric values, sourced from this phase's calibration; the anchor *selection procedure* itself
  remains `[PROVISIONAL]`/unimplemented. Section F ("Point-IK generation policy") gained a note on
  the shared thresholds/classification priority (unchanged from v1).
- **New**: `docs/V2_THRESHOLD_CALIBRATION.md` -- full calibration report (baseline, seed, pool
  sizes, formulas, distribution tables, selected thresholds, classification priority, expected
  candidate counts, rationale, limitations, reproducibility command).
- **New tests**: `tests/test_dataset_v2_threshold_calibration.py` (14 tests: determinism at a fixed
  seed; different seed produces different pool/thresholds; pools are duplicate-free and
  limit-respecting; normalized-margin formula matches `kinematics/joint_limit_utils.py` directly;
  `sigma_min` matches `kinematics/singularity_metrics.py` directly; classification boundary
  behavior (`<=` is near, exclusive above); the sigma-axis tri-state and the "regular anchor"
  combination are non-overlapping; a regular interior configuration classifies as regular (and a
  note that `q=0` is itself an exact singularity for this model, discovered while picking that
  fixture); near-singular classification traced back to a directly recomputed `sigma_min`;
  candidate counts are sufficient (proportional check plus the biased-pool absolute counts);
  `difficulty_thresholds.json` parses and matches the calibration module's constants; the new
  config file is present in `all_configs()`; Dataset v1's `configs/dls_config.json` is read but
  never written by calibration).
- **Modified**: `tests/test_dataset_v2_scaffold.py` -- `test_configs_all_parse_as_json` updated
  11 -> 12 config files; the old `test_anchor_config_does_not_invent_unresolved_thresholds` was
  replaced with `test_anchor_config_thresholds_are_locked_by_calibration_not_invented` (asserts
  `"locked"` status and that `anchor_config.json`'s numeric thresholds equal
  `difficulty_thresholds.json`'s, not a second invented value) plus a new
  `test_difficulty_thresholds_config_is_well_formed`.

### Calibration run actually executed

`python -m pipelines.run_dataset_v2_threshold_calibration --seed 42` (locked default pool sizes:
50,000 generic + 20,000 singularity-biased) -- ~31s wall-clock. Key results (full detail in
`docs/V2_THRESHOLD_CALIBRATION.md`):

- `near_joint_limit`: normalized margin `<= 0.024991237796029034` (P10 of the generic pool's
  per-configuration normalized minimum joint-limit margin). An initial flat-rad interior margin
  (matching v1 Point-IK's `INTERIOR_MARGIN_RAD=0.10`) was found to structurally exclude
  `joint_2`/`joint_4` from ever being the near-limit controlling joint (their half-range is ~2.9x
  smaller than the other five joints'); switched the generic pool to a per-joint
  range-proportional margin (1% of half-range) before deriving the threshold, confirmed by the
  controlling-joint histogram becoming even across all seven joints.
- `near_singularity`: `sigma_min <= 0.03`, reused unchanged from v1's
  `configs/dls_config.json:singularity_sigma_threshold` -- audited against
  `generators/_trajectory_common.py::select_anchor` and Tier 0's own singularity classifier and
  found consistent with both (same threshold, same `3.0` moderate-upper multiplier).
- `moderately_conditioned_upper_bound = 0.09`; `regular` = `sigma_min > 0.09` **and**
  `normalized_joint_limit_margin > 0.024991237796029034`.
- Candidate sufficiency confirmed empirically: 5,000/50,000 generic-pool candidates near-limit,
  6,038/20,000 singularity-biased-pool candidates near-singular, 24,736/50,000 generic-pool
  candidates regular -- all far exceeding Point-IK's 1,000-per-group need and anchors' 3-6-per-class
  need.
- Determinism verified: identical seed/pool-size reruns produce byte-identical candidate arrays;
  seed 42 vs. 43 produce different pools and different thresholds.

### Tests

- Targeted: `pytest tests/test_dataset_v2_threshold_calibration.py -q` -> 14 passed.
- Dataset v2 regression: `pytest tests/test_dataset_v2_scaffold.py tests/test_dataset_root_resolution.py tests/test_pipeline_tier0.py tests/test_dataset_v2_tier0_generation.py tests/test_dataset_v2_tier0_validation.py -q` -> passed (see final report for the exact count).
- Dataset v1 backward compatibility: `pytest tests/test_pipeline_smoke.py tests/test_point_dataset.py tests/test_trajectory_files.py -q` -> passed, no regression.
- Full suite: `pytest -q` -> see final report for the exact pass count.

### Blockers

- None new. The anchor **selection procedure** (not just the now-locked acceptance thresholds)
  remains unimplemented and is recommended as part of Phase 3's anchor generator.

### Confirmations

- Dataset v1: **not modified**. Only `dataset_v2/`, `pipelines/`, `specs/`, `docs/`, and `tests/`
  files were touched; no file under `assets/`, `benchmarks/`, `trajectories/`, root
  `DATASET_MANIFEST.json`, or root `VERSION` was touched.
- Official Dataset v2 data: **not generated**. No Point-IK samples, anchors, trajectories, or
  trials were produced; only threshold values (a handful of scalars) were computed and recorded in
  `dataset_v2/config_templates.py`'s constants / `configs/difficulty_thresholds.json`'s builder.
- No candidate pool was written to disk or committed at any point in this phase.

### Recommended Phase 3

Point-IK Tier 1 generator (6,000 samples, 6 groups x 1,000, `development`/`validation`/
`frozen_test` split 1,200/1,200/3,600) can now reuse this phase's locked `near_joint_limit`/
`near_singularity` thresholds directly, deriving only the remaining four groups'
(`near_target`/`medium_target`/`far_target`/`large_orientation_change`) quantile thresholds from
its own generation-time pool (analogous to v1's `_derive_thresholds`). Alternatively, the anchor
generator's **selection procedure** (searching for diverse anchors satisfying this phase's locked
thresholds) can be implemented next, unblocked by this phase.

## Phase 3 -- Tier 1 Point-IK generator, validator, CLI

- **Status**: complete. Point-IK only -- no anchors, trajectories, trials, or DLS evaluation.
- **Baseline before this phase**: branch `feature/dataset-v2`, commit
  `e53617755149ec43f43079f6641e86438c8ed5e8`, clean working tree, `pytest -q` -> 406 passed,
  0 failed/skipped/errored.

### Files added/modified

- **New**: `dataset_v2/point_ik_generation.py` -- the generator. Draws a deterministic
  `(q_initial, q_target_reference)` candidate pool (`q_initial` uniform over the operational
  interior with a margin proportional to each joint's own half-range -- a deliberate change from
  v1's flat-0.10-rad margin, justified by the joint_2/joint_4 bias documented in
  `docs/V2_THRESHOLD_CALIBRATION.md`; `q_target_reference` = `q_initial` perturbed by a
  log-uniform-magnitude random unit direction, clipped to limits -- reused unchanged from v1),
  computes real FK/Jacobian covariates for both endpoints (position, SO(3) geodesic orientation
  distance, joint distance, `sigma_min`/`sigma_max`/`condition_number` at both endpoints, normalized
  and absolute-rad joint-limit margins), classifies each pair into exactly one of the six locked
  difficulty groups (`near_joint_limit`/`near_singularity` reuse Phase 2.5's locked
  single-configuration thresholds applied to the pair-minimum; `near_target`/`medium_target`/
  `far_target`/`large_orientation_change` use v1's unchanged 33rd/66th-percentile-position and
  85th-percentile-orientation quantile levels, re-derived fresh from this phase's own pool), then
  selects each group's exact 1,000-sample quota via a new deterministic
  `stratified_diversity_select` (quantile-binned strata over six covariates -- joint-space
  location, target-workspace location, orientation distance, position distance, pair `sigma_min`,
  pair joint-limit margin -- seeded shuffle, round-robin draw across strata; never a plain
  first-N cut, never solver-derived, raises an actionable pool-size error rather than relaxing a
  threshold or duplicating a sample) before splitting each group's 1,000 into 200/200/600 via a
  further seeded permutation. `q_target_reference` is never passed to any solver in this phase and
  is documented (config + generated `difficulty_definition.json`) as reference/provenance-only. All
  seeds trace to `(master_seed, tag_path)` via `generators/_common.py::derive_seed`; no
  `numpy.random` global state is touched. Writes `development.npz`/`validation.npz`/
  `frozen_test.npz`, `point_ik_manifest.csv`, `difficulty_definition.json`,
  `point_ik_generation_report.json` atomically under `<dataset_v2_root>/tier1_point_ik/`, asserts
  zero sample_id/content_hash/`(q_initial, q_target_reference)` collisions across all three splits
  before writing, rejects pre-existing output without `overwrite=True`, and updates
  `DATASET_MANIFEST.json`'s `counts.point_ik` (via new `dataset_v2/manifest.py::
  apply_point_ik_generation_status`) and `checksums/CHECKSUM_MANIFEST.json` after a real
  (non-dry-run) generation.
- **New**: `dataset_v2/point_ik_validation.py` -- the independent validator. Recomputes FK of both
  `q_initial` and `q_target_reference` (via `kinematics/forward_kinematics.py`, unchanged) and
  compares to the stored pose; recomputes every covariate and the difficulty classification
  (same priority/threshold logic as the generator, read back from the generated
  `difficulty_definition.json`, never re-invented) and flags any mismatch; checks exact/group/
  split counts, shapes/dtypes, no object dtype, operational-limit compliance, quaternion
  normalization, and global uniqueness of sample_id/content_hash/`(q_initial, q_target_reference)`
  pairs. Never calls DLS. Returns a `PointIKValidationReport` with a `passed` flag and itemized
  `reasons`; never raises on a failed check, only on a missing/malformed input file.
- **New**: `pipelines/run_dataset_v2_tier1_generation.py` -- CLI
  (`python -m pipelines.run_dataset_v2_tier1_generation --dataset-root PATH`). Supports
  `--master-seed`, `--overwrite`, `--validate-only`, `--dry-run`, `--sample-limit-per-group`
  (test/smoke only, must be divisible by 5 to keep the 1:1:3 development:validation:frozen_test
  ratio), and `--pool-size` (test/smoke only). Exit codes: `0` success, `1` validation failure,
  `2` usage/configuration error.
- **Modified** (extended, not rewritten):
  - `dataset_v2/config_templates.py::point_ik_config()` -- added `split_sizes_per_group`
    (200/200/600), `pair_pool_policy` (pool size formula/default 150,000, range-proportional
    interior margin, magnitude log-range, quantile levels), and `diversity_selection_policy`
    (covariates, bin count, procedure description); locked counts (1000/group, 1200/1200/3600)
    and all Phase 1/2.5 config files/tests are unaffected. Status updated from
    `counts_locked_generation_not_implemented` to `counts_locked_generation_implemented`.
  - `dataset_v2/schemas.py::point_ik_schema()` -- extended to the full Phase 3 field list
    (`q_target_reference` replacing `q_target` to make the reference-only usage explicit in the
    field name itself; added `initial_position`/`initial_quaternion`, `initial_sigma_max`/
    `target_sigma_max`, `initial_condition_number`/`target_condition_number`, and renamed the
    margin fields to `minimum_*_limit_margin_normalized` with optional diagnostic
    `minimum_*_limit_margin_rad` fields) -- `tests/test_dataset_v2_scaffold.py`'s two
    `point_ik_schema` tests were updated to match (same pattern as Phase 2.5's `anchor_config`
    test update), not to hide any implementation defect.
  - `dataset_v2/manifest.py` -- added `apply_point_ik_generation_status` (returns a copy of the
    manifest dict with `counts.point_ik` replaced by actual generated counts; never touches the
    dataset-wide `generated`/`frozen`/`status` flags, mirroring `apply_tier0_generation_status`).
- **New tests**: `tests/test_dataset_v2_point_ik_generation.py` (29 tests: exact small-fixture
  group/split counts, `allow_pickle=False` loads, operational-limit compliance, quaternion
  normalization, `target_position`/`target_quaternion` matching FK(`q_target_reference`),
  `q_target_reference` never equal to `q_initial` plus the documented usage policy, no duplicate
  pairs within/across splits, no sample_id/content_hash leakage across splits, content-hash and
  full-array determinism at a fixed seed, differing content at a different seed, difficulty
  boundary/priority correctness, large-orientation classification range-checked to `[0, pi]`
  (SO(3), never Euler), diversity selection is not a plain first-N cut and raises an actionable
  error when the pool is insufficient, overwrite protection, dry-run writes nothing,
  CWD-independence, manifest/checksum-manifest updates, no absolute paths, schema validation of a
  representative record, missing-scaffold rejection, the locked-6000/1000-per-group/
  200-200-600-per-split config-resolution integration check, and CLI dry-run/generate/validate/
  overwrite-rejection flows) and `tests/test_dataset_v2_point_ik_validation.py` (11 tests: passes
  on a clean fixture, and separately detects a wrong total count, a wrong group count under
  full-counts mode, a duplicate sample_id, a duplicate content_hash, an out-of-limit joint state,
  an FK/target-pose mismatch, and a hand-corrupted difficulty classification, plus CLI exit-code
  0/1 on success/failure). All fixtures use a small pool (1,500 candidates, 20 samples/group,
  split 4/4/12) for speed; the full locked 6,000-sample generation was run manually (see below),
  not as part of the regular suite.

### Seed policy actually used

`master_seed` -> `derive_seed(master_seed, component_tags["point_ik"]=20)` = point_ik component
seed -> `derive_seed(point_ik_component_seed, 1)` = generic-pool seed (the RNG that draws the
candidate pool) -> per difficulty group, `derive_seed(point_ik_component_seed, 2, group_id)` =
diversity-selection seed and `derive_seed(point_ik_component_seed, 3, group_id)` = split-assignment
seed. Every seed used is recorded verbatim in `tier1_point_ik/point_ik_generation_report.json`, so
any seed is traceable back to `(master_seed, tag_path)` per spec section E. Verified byte-identical
NPZ/content-hash output across two independent scaffold+generate runs at the same seed, and
differing output at a different seed.

### Thresholds, pool, and counts (full locked run)

Reproducibility command used for the full run below:
`python -m pipelines.run_dataset_v2_tier1_generation --dataset-root <temp> --master-seed 42`

- `near_joint_limit_threshold = 0.024991237796029034` (Phase 2.5 locked, applied to
  `min(minimum_initial_limit_margin_normalized, minimum_target_limit_margin_normalized)`).
- `near_singularity_threshold = 0.03` (Phase 2.5 locked, applied to
  `min(initial_sigma_min, target_sigma_min)`).
- `generic_pool_size = 150000` (formula `max(samples_per_group * n_groups * 25, 30000)`, same
  ratio as v1's `generate_point_ik_dataset.py`), `generic_pool_seed` recorded in the generation
  report.
- Derived quantile thresholds from this run's pool: `position_distance_m_low_quantile =
  0.019073062799465516`, `position_distance_m_high_quantile = 0.1271916326883878`,
  `orientation_distance_rad_top_quantile = 1.1148948540219186`.
- Exact counts: 6,000 total, 1,000/group for all six groups, split 1,200/1,200/3,600 overall and
  200/200/600 for every group (`full_locked_counts: true`).

### Full generation (manual, temporary directory, not committed)

Ran the CLI against a scaffold in the OS temp directory (outside the repo, deleted afterward),
`--master-seed 42`, no overrides (full locked mode):

- Generation: **~104 s** wall-clock (`time python -m pipelines.run_dataset_v2_tier1_generation
  --dataset-root <temp> --master-seed 42 --overwrite`) -> wrote exactly 6,000 samples,
  1,000/group, 1,200/1,200/3,600 split, 200/200/600 per group per split.
- Validation: **~6 s** wall-clock (`... --validate-only`) -> `total=6000 passed=True`.
- `DATASET_MANIFEST.json`'s `counts.point_ik` updated with `generated: true`,
  `full_locked_counts: true`, and the exact group/split/group-split counts above; dataset-wide
  `generated`/`frozen` flags remain `false` (Tier 0 and Tier 2-4 are not generated by this phase).
- No NPZ/CSV/generated JSON from this run was added to the repository or Git; the temp directory
  was deleted after inspection.

### Tests

- Targeted: `pytest tests/test_dataset_v2_point_ik_generation.py
  tests/test_dataset_v2_point_ik_validation.py -q` -> 40 passed (60 s).
- Threshold calibration regression: `pytest tests/test_dataset_v2_threshold_calibration.py -q` ->
  14 passed.
- Tier 0 v2 regression: `pytest tests/test_dataset_v2_scaffold.py
  tests/test_dataset_root_resolution.py tests/test_pipeline_tier0.py
  tests/test_dataset_v2_tier0_generation.py tests/test_dataset_v2_tier0_validation.py -q` ->
  87 passed.
- Dataset v1 backward compatibility: `pytest tests/test_pipeline_smoke.py
  tests/test_point_dataset.py tests/test_trajectory_files.py -q` -> 130 passed, no regression.
- Full suite: `pytest -q` -> **446 passed, 0 failed, 0 skipped, 0 errors** (406 baseline + 40 new).

### Blockers

- None. The anchor generator's **selection procedure** (spec section G, unimplemented since
  Phase 1) remains out of scope for this Point-IK-only phase.

### Confirmations

- Dataset v1: **not modified**. `git status --short`/`git diff --name-only` after this phase show
  only `dataset_v2/config_templates.py`, `dataset_v2/manifest.py`, `dataset_v2/schemas.py`, and
  `tests/test_dataset_v2_scaffold.py` modified (all Phase 1/2.5 files, extended additively) and
  `dataset_v2/point_ik_generation.py`, `dataset_v2/point_ik_validation.py`,
  `pipelines/run_dataset_v2_tier1_generation.py`, `tests/test_dataset_v2_point_ik_generation.py`,
  `tests/test_dataset_v2_point_ik_validation.py` as new, untracked files. No file under `assets/`,
  `benchmarks/`, `trajectories/`, `configs/`, `schemas/`, `kinematics/`, `algorithms/`, root
  `DATASET_MANIFEST.json`, or root `VERSION` was touched.
- Official Dataset v2 Point-IK data: generated and validated in a temporary directory (see above)
  to confirm the locked counts/thresholds/runtime are real, but **not** committed to the
  repository -- no NPZ/CSV/generated JSON from this phase is committed data.
- DLS was never run, `q_target_reference` was never used as an initial guess anywhere in this
  phase's code, and `frozen_test` was never used to design the generator, choose a threshold, or
  tune anything.

### Recommended Phase 4

Per `docs/V2_REPO_AUDIT.md`'s recommended order, next is the anchor generator (12 anchors, 6
`regular`/3 `near_limit`/3 `near_singular`, split-isolated per spec section G) -- its acceptance
thresholds are already locked (Phase 2.5) but its **selection procedure** is not yet implemented.
Anchors are a prerequisite for the core-trajectory phase (120 trajectories, 5 shapes x 2
orientation modes x 12 anchors) that follows.
