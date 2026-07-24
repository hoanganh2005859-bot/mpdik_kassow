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

## Phase 4 -- Anchor generator, validator, CLI

- **Status**: complete. Anchors only -- no core trajectories, random-challenge trajectories,
  trials, or DLS evaluation.
- **Baseline before this phase**: branch `feature/dataset-v2`, commit
  `6c3d736e4d6c18347300d8a86a529586e6fd3706`, clean working tree, `pytest -q` -> 446 passed,
  0 failed/skipped/errored.

### Files added/modified

- **New**: `dataset_v2/anchor_generation.py` -- the generator. Draws three sub-pools of
  candidate joint configurations, all reusing Tier 0's already-verified sampling constructions
  unchanged (`dataset_v2/tier0_generation.py::_group_random_interior` for a `regular` pool with a
  range-proportional interior margin -- same rationale as Point-IK/threshold-calibration's generic
  pool; `_group_mixed_near_limits` for a near-limit-biased pool, whose per-joint independent
  lower/upper/interior randomization already gives natural controlling-joint diversity;
  `_build_singularity_candidate_pool` for a singularity-biased pool). Computes real FK/Jacobian
  covariates for every candidate (position, quaternion wxyz, `sigma_min`/`sigma_max`/
  `condition_number`/`numerical_rank`/positional manipulability, normalized and absolute-rad
  joint-limit margin, controlling joint index), then classifies every candidate against Phase
  2.5's locked thresholds (`near_joint_limit`/`near_singularity`/`moderately_conditioned` from
  `configs/difficulty_thresholds.json`, never copied to a second location) with the locked
  priority `near_singular > near_limit > regular`, storing all four diagnostic flags
  (`is_near_limit`/`is_near_singular`/`is_moderately_conditioned`/`is_regular`) independently of
  the resolved `primary_class`/assigned `anchor_class`. For `near_limit`/`near_singular`, prefers
  the "clean" (non-overlapping) eligible subset per spec section 3, falling back to the full
  overlap-inclusive pool only if the clean subset is smaller than the 3 needed (both counts always
  recorded in the report); never relaxes a threshold, never duplicates. Selects each class's exact
  quota via a new deterministic `greedy_farthest_point_select` (max-min diversity over a
  normalized composite feature vector -- joint-space, workspace position, `so3_log`-based
  orientation, `sigma_min`, joint-limit margin, plus a controlling-joint one-hot emphasized for
  `near_limit` -- each feature group divided by `sqrt(dimensionality)` so no group dominates
  Euclidean distance; ties broken by a seeded permutation, never arbitrary). Assigns splits per
  class via a further seeded permutation (2/2/2 regular, 1/1/1 near_limit, 1/1/1 near_singular),
  then asserts no two anchors in *different* splits are near-duplicates (joint-space/Cartesian/
  orientation tolerances from `configs/anchor_config.json`, never silently relaxed) before writing.
  Writes `anchors.npz`, `anchor_manifest.csv`, `anchor_generation_report.json` atomically under
  `<dataset_v2_root>/anchors/`, rejects pre-existing output without `overwrite=True`, and updates
  `DATASET_MANIFEST.json`'s `counts.anchors` (via new `dataset_v2/manifest.py::
  apply_anchor_generation_status`) and `checksums/CHECKSUM_MANIFEST.json` after a real
  (non-dry-run) generation. No DLS call anywhere in this module.
- **New**: `dataset_v2/anchor_validation.py` -- the independent validator. Recomputes FK, every
  covariate, all four diagnostic flags, and the anchor's own stored `anchor_class`'s raw
  eligibility (via `kinematics/`, unchanged) and flags any mismatch; checks exact/class/split/
  per-class-per-split counts, shapes/dtypes, no object dtype, operational-limit compliance,
  quaternion normalization, global uniqueness of `anchor_id`/`content_hash`/exact `q`, and
  split-anti-leakage (no near-duplicate anchor pair spanning two different splits, using the same
  configured tolerance). Never calls DLS. Returns an `AnchorValidationReport` with a `passed` flag
  and itemized `reasons`; never raises on a failed check, only on a missing/malformed input file.
- **New**: `pipelines/run_dataset_v2_anchor_generation.py` -- CLI
  (`python -m pipelines.run_dataset_v2_anchor_generation --dataset-root PATH`). Supports
  `--master-seed`, `--overwrite`, `--validate-only`, `--dry-run`, and `--regular-pool-size`/
  `--near-limit-pool-size`/`--singularity-pool-size` (test/smoke only). Exit codes: `0` success,
  `1` validation failure, `2` usage/configuration error.
- **Modified** (extended, not rewritten):
  - `dataset_v2/config_templates.py::anchor_config()` -- added
    `classification_priority_highest_first` (`["near_singular", "near_limit", "regular"]`),
    `split_assignment` (counts-per-class-per-split: 2/2/2 regular, 1/1/1 near_limit, 1/1/1
    near_singular), `candidate_pool_policy` (three sub-pool size defaults of 5,000 each, plus
    construction notes pointing at the single-source Tier 0 functions/config rather than copying
    numbers), `overlap_policy` (clean-preference + fallback rule, documented report fields), the
    full `diversity_selection_policy` (feature groups, weighting formula, tie-breaking rule), and
    `near_duplicate_tolerance` (joint-space/position/orientation thresholds). Status updated from
    `counts_and_thresholds_locked_generation_not_implemented` to
    `counts_and_thresholds_locked_generation_implemented`; the locked 6/3/3 class counts and the
    already-locked near_limit/near_singular acceptance thresholds are unchanged.
  - `dataset_v2/schemas.py::anchor_schema()` -- extended to the full Phase 4 field list (`q`/
    `position`/`quaternion_wxyz`/`split`/`sigma_max`/`condition_number`/`numerical_rank`/
    `manipulability`/`minimum_normalized_limit_margin`/`minimum_absolute_limit_margin_rad`/
    `controlling_joint_index`/all four diagnostic flags/`source_pool`/`content_hash`, replacing
    the old placeholder `q_anchor`/`consuming_trajectory_ids` fields, which no test referenced).
  - `dataset_v2/manifest.py` -- added `apply_anchor_generation_status` (mirrors
    `apply_tier0_generation_status`/`apply_point_ik_generation_status`; never touches the
    dataset-wide `generated`/`frozen`/`status` flags).
- **New tests**: `tests/test_dataset_v2_anchor_generation.py` (32 tests: exact 12/6-3-3/4-4-4/
  2-1-1 counts on a small fixture, determinism at a fixed seed, differing output at a different
  seed, classification-boundary checks against the config-read thresholds, overlap-report
  bookkeeping, regular/near_limit/near_singular anchor requirements, controlling-joint
  recomputation matches the stored value, no exact-duplicate `q`, split anti-leakage (no
  ID/hash reuse), the greedy farthest-point selector is deterministic and not a plain first-K cut
  and raises an actionable error when the pool is insufficient, FK/metadata consistency,
  `allow_pickle=False` loads, schema validation of a representative record, checksum-manifest
  verification, overwrite protection, dry-run writes nothing, CWD-independence, manifest/
  checksum-manifest updates, no absolute paths, missing-scaffold rejection, the locked-
  12/6-3-3/4-4-4/2-1-1 config-resolution integration check, and CLI dry-run/generate/validate/
  overwrite-rejection flows) and `tests/test_dataset_v2_anchor_validation.py` (10 tests: passes
  on a clean fixture, and separately detects a wrong total count, a wrong class count, a wrong
  split assignment, a duplicate anchor_id, a duplicate content_hash, an out-of-limit joint state,
  an FK/position mismatch, and a hand-corrupted covariate, plus CLI exit-code 0/1 on
  success/failure). All fixtures use small sub-pools (200 candidates each) for speed; the full
  locked-pool-size (5,000/sub-pool) generation was run manually (see below), not as part of the
  regular suite.

### Seed policy actually used

`master_seed` -> `derive_seed(master_seed, component_tags["anchors"]=30)` = anchors component
seed -> `derive_seed(anchors_component_seed, {regular_pool:1, near_limit_biased_pool:2,
singularity_biased_pool:3})` = per-sub-pool seed (the RNG that draws that sub-pool's candidates)
-> per anchor class, `derive_seed(anchors_component_seed, 10, class_id)` = diversity-selection seed
and `derive_seed(anchors_component_seed, 20, class_id)` = split-assignment seed. Every seed used is
recorded verbatim in `anchors/anchor_generation_report.json`, so any seed is traceable back to
`(master_seed, tag_path)` per spec section E. Verified byte-identical NPZ output across two
independent scaffold+generate runs at the same seed, and differing output at a different seed.

### Thresholds, pools, and counts (full locked run)

Reproducibility command used for the full run below:
`python -m pipelines.run_dataset_v2_anchor_generation --dataset-root <temp> --master-seed 42`

- `near_joint_limit_threshold = 0.024991237796029034`, `near_singularity_threshold = 0.03`,
  `moderately_conditioned_upper_bound = 0.09` -- all read from `configs/difficulty_thresholds.json`
  (Phase 2.5 locked), never copied into a second constant.
- Candidate pool sizes (locked defaults): `regular_pool = 5000`, `near_limit_biased_pool = 5000`,
  `singularity_biased_pool = 5000`.
- Clean/overlap candidate counts from this run: `regular` 4,250 eligible (no overlap axis);
  `near_limit` 4,658 clean / 855 overlapping (5,513 total eligible) -- selected from the clean
  subset; `near_singular` 2,145 clean / 855 overlapping (3,000 total eligible) -- selected from the
  clean subset. `near_duplicate_pairs = 0`.
- Exact counts: 12 total, 6/3/3 by class, 4/4/4 by split, 2/1/1 per class per split.

### Full generation (manual, temporary directory, not committed)

Ran the CLI against a scaffold in the OS temp directory (outside the repo, deleted afterward),
`--master-seed 42`, no overrides (locked default pool sizes):

- Generation: **~6.7 s** wall-clock -> wrote exactly 12 anchors, 6/3/3 by class, 4/4/4 by split,
  2/1/1 per class per split.
- Validation: **~1.0 s** wall-clock (`... --validate-only`) -> `total=12 passed=True`.
- Controlling-joint histogram: `regular` spread across 5 of 7 joints (0,1,2,4,5); `near_limit`
  spread across 3 distinct joints (0,1,2) -- no single joint dominates; `near_singular` spread
  across joints 2 and 3.
- Workspace bounding box (12 anchors): x in [-0.725, 0.920] m, y in [-0.866, 0.850] m, z in
  [-0.389, 1.092] m.
- Selected `sigma_min` ranges: `regular` [0.0905, 0.2518], `near_limit` [0.0625, 0.2101],
  `near_singular` [0.0030, 0.0282] -- all consistent with their class thresholds.
- `DATASET_MANIFEST.json`'s `counts.anchors` updated with `generated: true` and the exact
  class/split/class-split counts above; dataset-wide `generated`/`frozen` flags remain `false`.
- No NPZ/CSV/generated JSON from this run was added to the repository or Git; the temp directory
  was deleted after inspection.

### Tests

- Targeted: `pytest tests/test_dataset_v2_anchor_generation.py
  tests/test_dataset_v2_anchor_validation.py -q` -> 46 passed (20 s).
- Point-IK v2 regression: `pytest tests/test_dataset_v2_point_ik_generation.py
  tests/test_dataset_v2_point_ik_validation.py -q` -> 40 passed.
- Threshold calibration + Tier 0 v2 regression: `pytest
  tests/test_dataset_v2_threshold_calibration.py tests/test_dataset_v2_scaffold.py
  tests/test_dataset_root_resolution.py tests/test_pipeline_tier0.py
  tests/test_dataset_v2_tier0_generation.py tests/test_dataset_v2_tier0_validation.py -q` ->
  87 passed.
- Dataset v1 backward compatibility: `pytest tests/test_pipeline_smoke.py
  tests/test_point_dataset.py tests/test_trajectory_files.py -q` -> 130 passed, no regression.
- Full suite: `pytest -q` -> **492 passed, 0 failed, 0 skipped, 0 errors** (446 baseline + 46 new).

### Blockers

- None.

### Confirmations

- Dataset v1: **not modified**. `git status --short`/`git diff --name-only` after this phase show
  only `dataset_v2/config_templates.py`, `dataset_v2/manifest.py`, and `dataset_v2/schemas.py`
  modified (all Phase 1-3 files, extended additively) and `dataset_v2/anchor_generation.py`,
  `dataset_v2/anchor_validation.py`, `pipelines/run_dataset_v2_anchor_generation.py`,
  `tests/test_dataset_v2_anchor_generation.py`, `tests/test_dataset_v2_anchor_validation.py` as
  new, untracked files. No file under `assets/`, `benchmarks/`, `trajectories/`, `configs/`,
  `schemas/`, `kinematics/`, `algorithms/`, root `DATASET_MANIFEST.json`, or root `VERSION` was
  touched.
- Official Dataset v2 anchor data: generated and validated in a temporary directory (see above) to
  confirm the locked counts/thresholds/runtime are real, but **not** committed to the repository --
  no NPZ/CSV/generated JSON from this phase is committed data.
- DLS was never called anywhere in this phase's code (generator or validator); anchor selection
  used only real computed FK/Jacobian/joint-limit quantities; `frozen_test` anchors were generated
  and integrity-validated but never used to tune thresholds or selection weights.

### Recommended Phase 5

Per `docs/V2_REPO_AUDIT.md`'s recommended order, next is the core trajectory generator (120
trajectories = 5 shapes x 2 orientation modes x 12 anchors), producing both a canonical
(400-waypoint) and high-resolution source representation per trajectory plus arc-length/
cumulative-angular-displacement metadata, reusing `generators/_trajectory_common.py`'s quintic
time-scaling and sequential-DLS reachability validation and `generators/generate_orientation_profile.py`'s
SLERP-based variable-orientation profile (both unchanged) against the anchors this phase produced.

## Phase 5 -- Core trajectory generator, validator, CLI

- **Status**: complete. Core trajectories only -- no random-challenge trajectories, no trials, no
  DLS baseline evaluation.
- **Baseline before this phase**: branch `feature/dataset-v2`, commit
  `d1f017f0cba49e51e05c60a8a2124c343d70bc6d`, clean working tree, `pytest -q` -> 492 passed,
  0 failed/skipped/errored.

### Files added/modified

- **New**: `dataset_v2/core_trajectory_generation.py` -- the generator. For each of the 120
  (anchor, shape, orientation_mode) triples: builds a high-resolution source path (provisional
  2001 waypoints, `configs/trajectory_config.json:source_waypoint_count_nominal`) via the same
  closed-form position formulas v1's `generators/generate_{line,circle,figure8,helix}_trajectory.py`
  already use (unchanged math, now parameterized per-anchor instead of one shared anchor) plus a
  new `free_form` shape (a deterministic seeded `scipy.interpolate.CubicSpline` through 5 control
  points -- the anchor position, seeded interior offsets, and a seeded endpoint); orientation
  reuses `generators/generate_orientation_profile.py`/`generators/_trajectory_common.py`'s
  `orientation_arrays`/`orientation_phase_for_shape` unchanged (SO(3) SLERP for `variable`,
  constant for `fixed`, closed shapes' phase mapping already makes `variable` orientation return
  to start). Canonical (400-waypoint) output is **arc-length-uniformly resampled** from the source
  path -- position linearly interpolated between bracketing source samples, orientation via a new
  SO(3) geodesic SLERP helper (`so3_log`/`so3_exp`, never linear-then-normalize), endpoints
  preserved exactly, canonical never independently regenerated by re-evaluating the shape formula.
  Reachability is validated by sequential warm-started DLS (unchanged `kinematics`/DLS math) over
  the 400-waypoint **canonical** path (spec section 8 permits validating either source or
  canonical), capturing `q_reference` and FK-reconstruction error at every waypoint; on any
  waypoint failure the geometry scale shrinks (`configs/trajectory_config.json:
  scale_reduction_policy`, reused unchanged from v1's 0.85/15/0.05) and the whole 400-waypoint
  canonical path is retried -- never a partial accept, never a silently swapped trajectory,
  failure after `min_scale` raises a `RuntimeError` naming the trajectory/anchor/shape/orientation
  and every scale attempted. Writes `<trajectory_id>.npz` (canonical) and
  `<trajectory_id>_source.npz` (source) under `trajectories/{development,validation,frozen_test}/`
  (split inherited from the anchor, per `CLAUDE.md`), plus `core_trajectory_manifest.csv`,
  `core_trajectory_generation_report.json`, `core_trajectory_reachability_report.json`, and
  `core_trajectory_anti_leakage_report.json` under `trajectories/`, atomically; asserts zero
  `trajectory_id`/content-hash collisions (including cross-split) before writing; rejects
  pre-existing output without `overwrite=True`; refuses to run against a missing or
  independent-validator-failing anchor catalog; updates `DATASET_MANIFEST.json`'s
  `counts.core_trajectories` (via new `dataset_v2/manifest.py::apply_core_trajectory_generation_status`)
  and `checksums/CHECKSUM_MANIFEST.json` after a real (non-dry-run) generation. No DLS baseline
  evaluation result is ever published from this phase; `q_target`/`q_reference` are
  reference/provenance only.
- **New**: `dataset_v2/core_trajectory_validation.py` -- the independent validator. Recomputes, from
  the on-disk manifest + NPZ files only (never re-running DLS to decide validity): exact/split/
  shape/orientation/per-anchor counts and 48,000 total canonical poses; shapes/dtypes/finiteness/
  no-object-dtype; quaternion normalization and sign continuity; monotonic source/canonical
  parameter and timestamps; endpoint preservation; a **fresh arc-length resample of the stored
  source path** compared against the stored canonical arrays (catches a canonical path that wasn't
  actually derived from its own source); arc-length and angular-displacement recomputation against
  the manifest; closed-path closure; fixed-orientation-actually-fixed /
  variable-orientation-actually-varies; joint-limit compliance and FK(`q_reference`)-vs-target
  reconstruction error (tolerance sourced from `configs/dls_config.json`'s own DLS convergence
  thresholds, since `q_reference` is a DLS solution, not an exact FK inversion); no waypoint
  marked unreachable; global `trajectory_id`/content-hash uniqueness and split leakage (no
  `anchor_id` spanning two splits); presence and `pass=True` of the anti-leakage report. Returns a
  `CoreTrajectoryValidationReport` with a `passed` flag and itemized `reasons`; never raises on a
  failed check, only on a missing/malformed input file.
- **New**: `pipelines/run_dataset_v2_core_trajectory_generation.py` -- CLI
  (`python -m pipelines.run_dataset_v2_core_trajectory_generation --dataset-root PATH`). Supports
  `--master-seed`, `--overwrite`, `--validate-only`, `--dry-run`, repeatable `--anchor-id`/
  `--shape`/`--orientation-mode` (test/smoke-only subset selection), and `--source-waypoints`
  (test/smoke-only; never changes the locked 400-waypoint canonical count). Exit codes: `0`
  success, `1` validation failure, `2` usage/configuration error.
- **Modified** (extended, not rewritten):
  - `dataset_v2/config_templates.py::trajectory_config()` -- added `duration_s` (10.0, reused from
    v1's default), `source_waypoint_count_nominal` (2001, `[PROVISIONAL]`), per-shape `geometry`
    parameters (line/circle/figure8/helix nominal sizes reused unchanged from v1's
    `generate_*_trajectory.py` constants; `free_form`'s control-point/spline policy, new),
    `canonical_resampling_policy`, `scale_reduction_policy` (0.85/15/0.05, reused unchanged from
    v1), and `reachability_policy` documentation dicts; locked counts (120/400/48,000) and all
    Phase 1-4 config files/tests are unaffected. Status updated to
    `counts_locked_generation_implemented`.
  - `dataset_v2/schemas.py::trajectory_schema()` -- rewritten to the full Phase 5 field list
    (`shape`/`anchor_id`/`anchor_class`/`source_waypoint_count`/`canonical_waypoint_count`/
    `quaternion_convention`/`duration_s`/`canonical_control_period_s`/`nominal_scale`/
    `accepted_scale`/`geometry_parameters_json`/`closed_path`/`closure_*_error`/
    `reachability_status`/fingerprints/`content_hash`/`sha256`), replacing the Phase-1 placeholder
    (`num_canonical_waypoints` etc., which no test referenced by field name).
  - `dataset_v2/manifest.py` -- added `apply_core_trajectory_generation_status` (mirrors
    `apply_tier0_generation_status`/`apply_anchor_generation_status`/
    `apply_point_ik_generation_status`; never touches the dataset-wide `generated`/`frozen`/
    `status` flags).
  - `specs/DLS_DATASET_V2_SPEC.md` section H -- added a `[PROVISIONAL]` subsection documenting the
    concrete source-resolution/canonical-resampling/`free_form`/reachability-validation-scope
    decisions this phase made to fill gaps the locked design left open, without changing the
    locked 120/400/48,000 counts.
- **New tests**: `tests/test_dataset_v2_core_trajectory_generation.py` (34 tests: anchor-catalog
  dependency, expected-files/counts on a 1-anchor/10-trajectory fixture, same-seed determinism
  (content-hash and byte-identical NPZ) and different-seed content change, line/circle/figure8/
  helix/free_form geometry (start-at-anchor, closure, open-vs-closed, control-point/smoothness
  checks), fixed-orientation-constant and variable-orientation-varies, quaternion wxyz
  normalization and sign continuity, arc-length/angular-displacement recomputation, canonical/
  source endpoint consistency, `q_reference`/FK consistency, no-waypoint-unreachable, no
  duplicate IDs/hashes across anchors, anchor-inheritance fields, `allow_pickle=False` loads,
  schema validation, overwrite protection, dry-run writes nothing, CWD independence, unknown
  shape/anchor-id filter rejection, a config-arithmetic check that the locked configuration
  resolves to exactly 120/400/48,000/40-40-40/24-per-shape/60-60, and CLI generate/validate/
  overwrite-rejection round trips) and `tests/test_dataset_v2_core_trajectory_validation.py` (16
  tests: passes on a clean fixture, and separately detects duplicate `trajectory_id`/
  `content_hash`, wrong total count, a missing NPZ file, non-finite values, a non-unit quaternion,
  a wrong `arc_length_m`/`cumulative_angular_displacement_rad` in the manifest, an unreachable-
  waypoint flag, an FK/`q_reference` mismatch, a canonical path that no longer matches a fresh
  resample of its own stored source, a `fixed`-orientation trajectory that was tampered to
  actually vary, split leakage, a missing anti-leakage report, plus CLI exit-code 0/1). Both files
  use a small shared fixture (1 anchor x 5 shapes x 2 orientation modes = 10 trajectories,
  `source_waypoint_count=450` -- reduced from the 2001 nominal for speed, since DLS-validation
  runtime is dominated by the 400-waypoint canonical count, not the source count) and, for the
  validator's corruption tests, a directory-copy-then-tamper pattern to avoid re-running expensive
  reachability validation per test case; the full locked 120-trajectory generation was run
  manually (see below), not as part of the regular suite.

### Seed policy actually used

`master_seed` -> `derive_seed(master_seed, component_tags["core_trajectories"]=40)` =
core-trajectories component seed (the only seed derivation point in this phase; per-anchor
`free_form` control-point offsets use `derive_seed(core_trajectories_component_seed, 1,
zlib.crc32(anchor_id))` -- `zlib.crc32`, not Python's randomized-per-process `hash()`, so the
derivation is reproducible across processes/platforms). Every other geometric choice
(line/circle/figure8/helix parameters, `variable`-orientation rotation vector, scale-shrink
sequence) is a deterministic function of the anchor pose and `scale`, not an independent random
draw. Verified byte-identical NPZ/content-hash output across two independent scaffold+anchor+
core-trajectory runs at the same seed, and differing output at a different seed.

### Geometry, resolution, and policy actually used (full locked run)

Reproducibility commands used for the full run below (temporary directory, `--master-seed 42`):
`python -m pipelines.run_dataset_v2_scaffold ...`, then
`python -m pipelines.run_dataset_v2_anchor_generation --dataset-root <temp> --master-seed 42`
(locked default 5000/5000/5000 pool sizes, same as Phase 4's own full run), then
`python -m pipelines.run_dataset_v2_core_trajectory_generation --dataset-root <temp> --master-seed 42`
(no overrides -- full locked mode, source waypoint count 2001).

- Source resolution: 2001 waypoints/trajectory (`> 400` as required); canonical: exactly 400
  waypoints/trajectory (locked).
- Geometry (nominal, reused unchanged from v1): line length 0.12 m; circle radius 0.045 m;
  figure8 amplitudes 0.05 m / 0.03 m; helix radius 0.04 m / height 0.08 m; `free_form` (new)
  endpoint distance 0.10 m / lateral amplitude 0.02 m, 5 control points via
  `scipy.interpolate.CubicSpline`. `variable`-orientation rotation angle 0.35 rad about the
  anchor's local z-axis (reused unchanged from v1).
- Scale-reduction policy: shrink factor 0.85, up to 15 attempts, floor 0.05 (reused unchanged from
  v1's `generators/_trajectory_common.py`).

### Full generation (manual, temporary directory, not committed)

Ran the CLI against a scaffold + full anchor catalog in the OS temp directory (outside the repo,
deleted afterward), `--master-seed 42`, no overrides (full locked mode):

- Generation: **5 min 6 s** wall-clock -> wrote exactly 120 trajectories: split counts
  development/validation/frozen_test = 40/40/40; shape counts line/circle/figure8/helix/free_form
  = 24/24/24/24/24; orientation counts fixed/variable = 60/60; 400 canonical waypoints/trajectory
  -> 48,000 canonical poses total.
- Validation: **14.8 s** wall-clock (`... --validate-only`) -> `total=120 canonical_poses=48000
  passed=True`.
- Scale-reduction statistics: 46/120 trajectories required at least one shrink attempt (the rest
  succeeded at nominal scale=1.0); accepted scale range **0.1422 - 1.0** (mean 0.812); the hardest
  trajectory needed 13 shrink attempts before validating (near_limit/near_singular anchors, as
  expected -- their geometry has far less clearance than `regular` anchors).
  `all_trajectories_100pct_reachable = True`.
- Reachability/reconstruction error ranges: position error max **6.0 mm** (matches
  `configs/dls_config.json:position_success_threshold_m=0.006`, the DLS solver's own convergence
  threshold, as expected for a successful solve at the boundary), orientation error max **4.17
  deg** (well inside the 10 deg DLS threshold).
  > **[SUPERSEDED BY PHASE 5.1]** These two numbers are artifacts of accepting the DLS baseline's
  > *own* success thresholds as the definition of "reachable", not real kinematic limits. Phase 5.1
  > replaced that circular criterion with an independent-FK check at 1e-4 m / 0.01 deg and found
  > the same targets reconstruct to ~1e-7 m. The scale statistics in this section are likewise
  > superseded — see the Phase 5.1 section below.
- Arc length range: **0.0327 m - 0.3357 m**; cumulative angular displacement range: **0.0 -
  0.6988 rad** (0.0 for every `fixed`-orientation trajectory, as expected).
- Closure: position closure error **0.0 m** and variable-orientation closure error **0.0 rad**
  for every closed-path (`circle`/`figure8`) trajectory (exact by construction: closed-form
  position formulas are exactly periodic in `s`, and `orientation_phase_for_shape`'s
  0->1->0 phase mapping returns `variable` orientation to the start exactly).
- `DATASET_MANIFEST.json`'s `counts.core_trajectories` updated with `generated: true` and the
  exact split/shape/orientation counts and 48,000 canonical-pose total above; dataset-wide
  `generated`/`frozen` flags remain `false` (Tier 0-1/random-challenge/trials are not generated).
- No NPZ/CSV/generated JSON from this run was added to the repository or Git; the temp directory
  was deleted after inspection (validator run + stats extraction), per phase instructions.

### Tests

- Targeted: `pytest tests/test_dataset_v2_core_trajectory_generation.py
  tests/test_dataset_v2_core_trajectory_validation.py -q` -> 50 passed (~7 min, dominated by
  sequential-DLS reachability validation on 400-waypoint canonical paths across ~15 independent
  small generations).
- Anchor v2 regression: `pytest tests/test_dataset_v2_anchor_generation.py
  tests/test_dataset_v2_anchor_validation.py -q` -> 46 passed.
- Point-IK v2 regression: `pytest tests/test_dataset_v2_point_ik_generation.py
  tests/test_dataset_v2_point_ik_validation.py -q` -> 40 passed.
- Threshold calibration + Tier 0 v2 regression: `pytest
  tests/test_dataset_v2_threshold_calibration.py tests/test_dataset_v2_scaffold.py
  tests/test_dataset_root_resolution.py tests/test_pipeline_tier0.py
  tests/test_dataset_v2_tier0_generation.py tests/test_dataset_v2_tier0_validation.py -q` ->
  87 passed.
- Dataset v1 backward compatibility: `pytest tests/test_pipeline_smoke.py
  tests/test_point_dataset.py tests/test_trajectory_files.py -q` -> 130 passed, no regression.
- Full suite: `pytest -q` -> **542 passed, 0 failed, 0 skipped, 0 errors** (492 baseline + 50 new),
  6 min 36 s.

### Blockers

- None.

### Confirmations

- Dataset v1: **not modified**. `git status --short`/`git diff --name-only` after this phase show
  only `dataset_v2/config_templates.py`, `dataset_v2/manifest.py`, `dataset_v2/schemas.py`, and
  `specs/DLS_DATASET_V2_SPEC.md` modified (all Phase 1-4 files or the spec's own provisional
  clarification section, extended additively) and `dataset_v2/core_trajectory_generation.py`,
  `dataset_v2/core_trajectory_validation.py`,
  `pipelines/run_dataset_v2_core_trajectory_generation.py`,
  `tests/test_dataset_v2_core_trajectory_generation.py`,
  `tests/test_dataset_v2_core_trajectory_validation.py` as new, untracked files. No file under
  `assets/`, `benchmarks/`, `trajectories/`, `configs/`, `schemas/`, `kinematics/`, `algorithms/`,
  root `DATASET_MANIFEST.json`, or root `VERSION` was touched. HEAD unchanged
  (`d1f017f0cba49e51e05c60a8a2124c343d70bc6d`) -- nothing was committed.
- Official Dataset v2 core trajectory data: generated and independently validated in a temporary
  directory (see above) to confirm the locked counts/thresholds/runtime are real, but **not**
  committed to the repository -- no NPZ/CSV/generated JSON from this phase is committed data.
- DLS was used only as a generation-time reachability *validator* (never published as a DLS
  baseline evaluation result); `frozen_test` trajectories were generated and integrity-validated
  but never used to design the generator, choose a threshold, or tune scale/geometry; no random
  challenge trajectories, no trials, no Tier 1-4 DLS baseline evaluation was performed.

### Recommended Phase 6

Per `docs/V2_REPO_AUDIT.md`'s recommended order, next is the random-challenge trajectory generator
(90 trajectories, split 30/30/30, reusing the same dual-representation/reachability-validation
machinery this phase built, but interpolating between individually-validated randomly drawn
control poses per spec section I rather than closed-form shapes) and, after that, the
easy/medium/hard trial generator (630 trials = 3 per trajectory across all 210 core + random-
challenge trajectories, spec section J).

## Phase 5.1 -- Reachability and geometric-scale hardening of the 120 core trajectories

- **Baseline before this phase**: branch `feature/dataset-v2`, commit
  `d1f017f0cba49e51e05c60a8a2124c343d70bc6d`, working tree carrying the uncommitted Phase 5 files,
  `pytest -q` -> 542 passed, 0 failed/skipped/errored.

### Reachability audit of the Phase 5 implementation

| Question | Phase 5 (audited) | Phase 5.1 (after hardening) |
| --- | --- | --- |
| Generation target source | Cartesian geometry derived from the anchor pose (closed-form shape formulas); **not** FK of a drawn `q`, so IK is genuinely required | unchanged |
| Solver | `kinematics/dls_solver.py::solve_dls_until_converged` | same engine, used only as a generation-time numerical IK engine |
| Solver config source | **`configs/dls_config.json` (Dataset v1 DLS *baseline evaluation* config)** | `configs/generation_reachability_config.json` (Dataset v2's own; v1's DLS config never read) |
| Solver success threshold | **`position_success_threshold_m = 0.006`, `orientation_success_threshold_deg = 10.0`** | generation solver thresholds = half the acceptance tolerance (5e-5 m / 0.005 deg) |
| What decided "reachable" | **`result.success` alone** -- the DLS baseline's own evaluation criterion | an **independent** `FK(q_reference)` recomputation vs. the target pose |
| Independent FK reconstruction threshold | **none at generation**; the validator re-used the same v1 DLS thresholds (circular -- it could never catch the problem) | position `<= 1e-4 m`, orientation `<= 0.01 deg`, from Dataset v2's own config |
| Initial guess / restart policy | warm start from the previous waypoint only; no refinement, no restarts | warm start, then up to 6 refinement re-entries, then deterministic restarts (anchor configuration, then seeded perturbations) |
| `q_reference` stored | canonical path only | canonical **and** source path |
| Source-path validation | **not performed** (canonical only) | every source waypoint validated under the full policy |
| Canonical-path validation | all 400 waypoints | all 400 waypoints |
| Failure handling | shrink scale, retry whole path; waypoints never dropped | geometry alternatives first, then shrink; waypoints never dropped; an unresolved trajectory aborts the run rather than writing a partial core set |

**Audit conclusion (question 1)**: yes -- Phase 5 confirmed reachability using the DLS baseline's
own success threshold, and the validator inherited the same threshold, so the circularity was
undetectable from inside. The reported "max FK reconstruction error = 0.006 m / 4.17 deg" was an
artifact of that acceptance boundary, not a kinematic limit: re-verified under the strict
independent criterion, the same targets reconstruct to ~1e-7 m for well-conditioned anchors.

**Audit conclusion (question 2)**: yes -- the scale collapse was largely an artifact of a *fixed*
geometry basis. v1's generators hardcode one direction/plane/axis per shape (line `+x`, circle
`(x,y)`, ...), which Phase 5 inherited; when that single basis pointed into an unreachable region
the only remedy available was shrinking. Direct measurement: for `anchor_near_singular_00` a line
along `+x` fails strict reachability at waypoint 166 at scale 1.0, while `-x`, `-y` and `-z` are
each fully reachable at scale 1.0.

### Files added/modified

- **New**: `dataset_v2/generation_reachability.py` -- the strict reachability policy and engine.
  Loads Dataset v2's own tolerances/solver settings (refusing to run if the config declares any
  dependence on DLS evaluation thresholds); `fk_reconstruction_error` (the independent evidence);
  `_solve_with_refinement` (re-enters the solver so its internal stagnation window cannot halt
  descent above the strict tolerance); `solve_reference_configuration` (deterministic
  warm-start -> anchor -> seeded-perturbation restart ladder); `validate_path_strict` (per-waypoint
  reachability over a whole path, with a `stop_on_first_failure` search optimization);
  `probe_settings_from` (strictly weaker settings used only while searching, never for acceptance);
  `error_distribution` (max/P95/median reporting).
- **New config**: `configs/generation_reachability_config.json` (via
  `dataset_v2/config_templates.py::generation_reachability_config`) -- strict tolerances, the
  generation solver settings, refinement/restart/search-probe policies, an explicit
  `independence.reads_dls_evaluation_thresholds: false` declaration, and the DLS baseline's
  thresholds recorded *for reference only* so the separation stays auditable.
- **Modified**: `dataset_v2/core_trajectory_generation.py` -- reachability now goes through
  `generation_reachability` (v1's `configs/dls_config.json` is no longer read anywhere in the
  module); added `enumerate_geometry_alternatives` + `search_geometry_and_scale` (scale is the
  **outer** loop and geometry alternative the **inner** loop, so every alternative is tried at the
  larger scale before anything shrinks; ties broken by configured order); the accepted candidate is
  re-validated end-to-end under the **full** policy on both the canonical and source path before
  anything is written; `q_reference`/per-waypoint errors/reachability flags are stored for both
  paths; scale band, geometry-alternative id, alternatives-attempted count and scale-reduction
  reason are recorded per trajectory; a new `core_trajectory_geometry_search_report.json` records
  every attempt and rejection reason; any trajectory that cannot be generated aborts the run (no
  partial core set); optional `--progress` reporting.
- **Modified**: `dataset_v2/core_trajectory_validation.py` -- now fails when the generation config
  declares DLS dependence, when the generation tolerances are not strictly tighter than the DLS
  baseline's, when any canonical **or source** waypoint exceeds the strict tolerance, when
  `q_reference` is missing/truncated/out-of-limits, when stored per-waypoint errors disagree with
  the independent recomputation, when scale-band / scale-reduction-reason / geometry-alternative
  metadata is missing or inconsistent, and when an enforced minimum-scale gate is violated.
  Reports max/P95/median position and orientation reconstruction error, per-shape error
  distribution, accepted-scale distribution, per-band counts, and the worst 10 trajectories.
- **Modified**: `dataset_v2/config_templates.py::trajectory_config` -- added `geometry_alternatives`
  (6 line directions, 3 circle/figure-8 plane bases, 6 helix axis/sign combinations, 4 free-form
  templates) and `minimum_scale_gate` (`minimum_core_accepted_scale: null`, `minimum_scale_status:
  "not_yet_locked_pending_user_decision"`, `enforced: false`, diagnostic bands
  `[1.0, 0.75, 0.5, 0.25]`) -- deliberately **not** inventing a locked minimum scale.
- **Modified**: `dataset_v2/schemas.py::trajectory_schema` -- added and required the reachability
  tolerance, geometry alternative, accepted scale and reconstruction-error fields; the tolerance
  fields are schema-bounded strictly below the DLS baseline's thresholds.
- **Modified**: `specs/DLS_DATASET_V2_SPEC.md` -- new section H.1 locking the independent-FK
  acceptance rule, the generation/evaluation separation, dual-path validation, and the
  geometry-alternative-before-scale-reduction policy; the minimum-scale gate is recorded as
  `[PROVISIONAL]` pending a user decision. Locked counts unchanged.
- **New tests**: `tests/test_dataset_v2_core_trajectory_reachability.py` (25 tests).
- **Modified tests**: `tests/test_dataset_v2_core_trajectory_generation.py` (strict tolerance and
  new schema fields), `tests/test_dataset_v2_core_trajectory_validation.py` (new rejection
  message), `tests/test_dataset_v2_scaffold.py` (12 -> 13 config files).

### Generation tolerance and solver policy actually used

- Acceptance tolerances: position `<= 1e-4 m`, orientation `<= 0.01 deg` -- 60x and 1000x tighter
  than the DLS baseline's 0.006 m / 10.0 deg respectively. Source:
  `configs/generation_reachability_config.json` only.
- Generation solver (numerical IK engine only): `max_iterations=400`, success thresholds 5e-5 m /
  0.005 deg (half the acceptance tolerance, so the independent FK check has headroom),
  `position_weight=orientation_weight=1.0`, `lambda_min=1e-6`, `lambda_max=0.05`,
  `joint_limit_avoidance=false`, `null_space_gain=0.0`.
- Refinement: up to 6 solver re-entries per waypoint. Restarts: up to 3 deterministic restarts
  (anchor configuration, then seeded perturbations of the previous reference, perturbation
  0.05 rad, seed derived from the master seed + trajectory tag + waypoint index).
- Search probe (speed only, strictly weaker: 2 refinement rounds, 0 restarts) is used solely to
  rank `(geometry alternative, scale)` candidates; the winner is always re-validated end-to-end
  under the full policy before being written, so the shipped guarantee comes from the full policy.

### Geometry-alternative search policy

For each `(anchor, shape, orientation_mode)` the search enumerates a deterministic alternative set
(`configs/trajectory_config.json:geometry_alternatives`) and runs **scale as the outer loop,
alternative as the inner loop**, so shrinking only happens once every alternative has failed at the
larger scale:

- `line`: 6 direction axes (`+x -x +y -y +z -z` in the anchor's end-effector frame). v1 hardcoded `+x`.
- `circle` / `figure8`: 3 plane bases (`xy`, `yz`, `zx`). v1 hardcoded `xy`.
- `helix`: 3 axis bases x 2 travel-direction signs = 6. v1 hardcoded `xyz`, `+`.
- `free_form`: 4 deterministic control-point templates (seeded per anchor + template index).

Ties at equal scale are broken by the configured alternative order. Every attempt and its
rejection reason is recorded in `trajectories/core_trajectory_geometry_search_report.json`. The
search never consults evaluation results, DLS iteration count, DLS runtime, or frozen-test data.

### Full generation (manual, temporary directory, not committed)

Scaffold + full anchor catalog (locked 5000-per-sub-pool sizes) + core generation, all in the OS
temp directory outside the repo, `--master-seed 42`, no overrides (full locked mode, source
waypoint count 2001):

- **Generation: 24.3 min** wall-clock -> exactly 120 trajectories; splits 40/40/40; shapes
  24/24/24/24/24; orientations 60/60; 400 canonical waypoints each -> **48,000 canonical poses**.
  Every canonical **and** every source waypoint reachable (`all_canonical_waypoints_reachable` and
  `all_source_waypoints_reachable` both true). Zero generation failures.
- **Independent validation: 51 s** -> `total=120 canonical_poses=48000 passed=True`, with the
  validator recomputing FK from the stored `q_reference` for all 48,000 canonical **and** 240,120
  source waypoints.

Strict independent-FK reconstruction error (tolerances: 1e-4 m, 0.01 deg):

| Path | position max | position P95 | position median | orientation max | orientation P95 | orientation median |
| --- | --- | --- | --- | --- | --- | --- |
| canonical | 9.924e-05 m | 5.000e-05 m | 3.460e-06 m | 9.667e-03 deg | 4.999e-03 deg | 2.976e-04 deg |
| source | 8.457e-05 m | 5.000e-05 m | 4.996e-05 m | 9.999e-03 deg | 9.882e-03 deg | 3.829e-03 deg |

Per-shape canonical position max: line 5.00e-05, circle 5.00e-05, figure8 5.00e-05,
helix 6.22e-05, free_form 9.92e-05 m -- every shape inside the 1e-4 m tolerance.

Accepted-scale distribution:

| Statistic | Phase 5 (loose DLS criterion) | Phase 5.1 (strict independent-FK criterion) |
| --- | --- | --- |
| min | 0.1422 | **0.1209** |
| P05 | (not reported) | 0.4375 |
| median | (not reported) | **1.0** |
| mean | 0.8121 | **0.8994** |
| max | 1.0 | 1.0 |
| trajectories requiring shrink | **46 / 120** | **28 / 120** |
| at full scale (1.0) | 74 / 120 | **92 / 120** |

Scale bands: `>=1.0`: 92, `[0.75,1)`: 7, `[0.5,0.75)`: 11, `[0.25,0.5)`: 6, `<0.25`: 4.
Counts below 0.75 / 0.50 / 0.25: **21 / 10 / 4**.

Geometry-search effect: 592 `(alternative, scale)` attempts across the 120 trajectories;
**35 / 120 trajectories ended on a geometry basis other than the v1 default** for their shape.
Winning alternatives: `planexy` 36, `tpl0` 18, `axisxyz_p` 16, `dir+x` 15, `planezx` 10,
`dir-x` 7, `axisxyz_n` 6, `tpl2` 3, `dir-y` 2, `planeyz` 2, `axiszxy_p` 2, `tpl3` 2, `tpl1` 1.

**Interpretation.** The shrink count fell 46 -> 28 and the mean accepted scale rose 0.812 -> 0.899
*while the acceptance criterion became 60x stricter in position and 1000x stricter in
orientation*. The two numbers are therefore not directly comparable: Phase 5's larger minimum
(0.1422) was achieved under a criterion that only required 6 mm reconstruction, whereas every
Phase 5.1 scale is certified at 1e-4 m. The geometric meaningfulness of the benchmark improved and
the evidential standard improved at the same time.

`total_restarts_used_diagnostic = 0`: warm start plus refinement was sufficient at every one of the
288,120 verified waypoints; the deterministic restart ladder was never needed (it remains in place
for robustness).

### Minimum-scale gate status and remaining blocker

`minimum_core_accepted_scale` is deliberately still `null` / `enforced: false` /
`minimum_scale_status: "not_yet_locked_pending_user_decision"` -- the specification does not lock a
minimum geometric scale, and inventing one would either rubber-stamp the current run or fail it
arbitrarily. The run therefore surfaces, rather than silently accepts, the small-scale cases:

**4 trajectories fall below the lowest diagnostic band (0.25), all on the same anchor:**

| trajectory | anchor class | alternative | accepted scale | probes |
| --- | --- | --- | --- | --- |
| `core_circle_fixed_anchor_near_limit_02` | near_limit | planezx | 0.1209 | 42 |
| `core_circle_variable_anchor_near_limit_02` | near_limit | planezx | 0.1209 | 42 |
| `core_figure8_fixed_anchor_near_limit_02` | near_limit | planezx | 0.1969 | 33 |
| `core_figure8_variable_anchor_near_limit_02` | near_limit | planezx | 0.1969 | 33 |

Root cause (measured, not assumed): `anchor_near_limit_02` is **doubly constrained** -- it has the
tightest normalized joint-limit margin of the whole catalog (0.00824) *and* the near_limit class's
lowest `sigma_min` (0.06254, close to the 0.03 near-singular threshold). The two affected shapes
are exactly the two **closed** shapes, which must traverse a complete planar loop and so cannot
exploit a favourable one-way direction the way `line`/`helix`/`free_form` can. For contrast,
`anchor_near_limit_00` has an equally tight margin (0.00854) but healthy conditioning (sigma_min
0.21007) and reaches scale 1.0 on `line`. At 0.1209 the circle radius is ~5.4 mm, which is
strictly reachable and correctly certified but arguably too small to be a meaningful Tier 2-4
tracking benchmark.

Three options, none of which the implementation may choose unilaterally:

1. **Accept and lock `minimum_core_accepted_scale = 0.10`** -- keeps all 120, records that four
   near_limit_02 closed-shape trajectories are deliberately small-amplitude cases.
2. **Lock `minimum_core_accepted_scale = 0.25` and re-select `anchor_near_limit_02`** -- add a
   conditioning floor (e.g. `sigma_min > 0.09`) to the `near_limit` acceptance predicate so a
   near-limit anchor is not simultaneously near-singular, then regenerate anchors + core
   trajectories. Preserves 12 anchors and 120 trajectories.
3. **Lock `minimum_core_accepted_scale = 0.25` and reduce the nominal closed-shape geometry**
   (circle radius / figure-8 amplitudes) so scale 1.0 means a smaller loop globally. Changes the
   benchmark's nominal geometry for all 24 circle and 24 figure-8 trajectories.

Recommendation: **option 2** -- it fixes the cause (an anchor that is near-limit *and*
near-singular, which the Phase 2.5 acceptance predicate permits but the spec never intended as a
compound class) rather than the symptom, and it leaves nominal geometry and all locked counts
untouched.

### Tests

- Targeted Phase 5.1: `pytest tests/test_dataset_v2_core_trajectory_reachability.py -q` -> **25 passed**.
- Full core trajectory: `pytest tests/test_dataset_v2_core_trajectory_generation.py
  tests/test_dataset_v2_core_trajectory_validation.py -q` -> **50 passed**.
- Anchor v2 regression -> **46 passed**. Point-IK/Tier 0/threshold-calibration regression ->
  **87 passed**. Dataset v1 backward compatibility (`test_pipeline_smoke.py`,
  `test_point_dataset.py`, `test_trajectory_files.py`) -> **130 passed**.
- Full suite: `pytest -q` -> **567 passed, 0 failed, 0 skipped, 0 errors** (542 Phase 5 baseline +
  25 new Phase 5.1 tests).

### Confirmations

- Dataset v1: **not modified**. `git diff --name-only` restricted to `assets/`, `benchmarks/`,
  `trajectories/`, `configs/`, `schemas/`, `kinematics/`, `algorithms/`, `generators/`,
  `evaluation/`, root `VERSION`, root `DATASET_MANIFEST.json` returns zero files. HEAD unchanged at
  `d1f017f0cba49e51e05c60a8a2124c343d70bc6d`; nothing committed, nothing pushed.
- Official Dataset v2 core trajectory data: generated and independently validated in a temporary
  directory, then **deleted**. No NPZ/CSV/generated JSON from this phase is committed.
- FK/Jacobian/SO(3)/DLS mathematics: unchanged. Dataset v1's DLS baseline evaluation thresholds:
  unchanged (read only for the validator's "is generation strictly tighter?" assertion, never
  written). No Tier 1-4 DLS baseline evaluation was run. No random challenge trajectories, no
  trials, no PPO/MPDIK/MAPPO work.
- `frozen_test` trajectories were generated and integrity-validated but never used to design the
  generator, choose a tolerance, tune the geometry search, or select a scale.

### Phase 5 final status

**Incomplete pending one user decision.** The scientific hardening itself is done and passing:
all 120 trajectories are certified strictly reachable on both the canonical and the source path
under a criterion fully independent of the DLS baseline, and the validator enforces that
independence. What remains is the locked minimum-scale decision (section above): under the
spec's own rule that a trajectory below a locked minimum scale must be recorded as a blocker
rather than silently accepted, and with no minimum yet locked, the four `anchor_near_limit_02`
closed-shape trajectories at 0.1209/0.1969 are reported as blockers rather than declared
acceptable.

### Recommended next step

Resolve the minimum-scale decision (recommendation: option 2 above -- add a conditioning floor to
the `near_limit` anchor acceptance predicate, lock `minimum_core_accepted_scale = 0.25`, regenerate
anchors and core trajectories). Once that is locked and all 120 clear the gate, Phase 5 closes and
Phase 6 (random challenge trajectories, 90, split 30/30/30) can begin, reusing this phase's strict
reachability engine and geometry-search machinery unchanged.

## Phase 5.2 -- Anchor class isolation, locked minimum scale, frozen-test seed reset

- **Baseline before this phase**: branch `feature/dataset-v2`, commit
  `d1f017f0cba49e51e05c60a8a2124c343d70bc6d`, working tree carrying the uncommitted Phase 5 and
  Phase 5.1 files (13 files, all accounted for and none reverted), `pytest -q` -> 567 passed as of
  the end of Phase 5.1.

### Root cause addressed

Phase 5.1 left four trajectories below any reasonable geometric floor:
`core_{circle,figure8}_{fixed,variable}_anchor_near_limit_02` at scale 0.1209 / 0.1969. The cause
was not the geometry search and not the strict reachability criterion -- it was a **confounded
anchor**. Phase 5.1's acceptance predicates let one anchor satisfy two classes' raw criteria (with
a "prefer clean, fall back to overlapping" preference), and `anchor_near_limit_02` was
simultaneously near a joint limit (normalized margin 0.00824) *and* near-singular
(`sigma_min` 0.06254, inside the moderately-conditioned band). A closed planar loop cannot escape a
doubly-constrained neighbourhood the way an open path can, so its circle and figure-8 collapsed
while its line/helix/free_form did not.

### Locked anchor class isolation (spec section H.2)

Anchor-selection eligibility is now **mutually exclusive by construction** -- machine-readable in
`configs/anchor_config.json:class_eligibility_predicates` with
`anchor_class_isolation_status: "locked"`:

| Class | Predicate |
| --- | --- |
| `regular` | `sigma_min > 0.09` AND `normalized_joint_limit_margin > 0.024991237796029034` |
| `near_limit` | `normalized_joint_limit_margin <= 0.024991237796029034` AND `sigma_min > 0.09` AND `is_near_singular == false` |
| `near_singular` | `sigma_min <= 0.03` AND `normalized_joint_limit_margin > 0.024991237796029034` AND `is_near_limit == false` |

The `0.09` floor reuses the already-calibrated `moderately_conditioned` upper bound rather than
introducing a new number. There is **no overlap fallback**: a class short of its quota is a hard
failure that prints the candidate-availability breakdown.

Scope is strictly anchor selection. Unchanged: the global difficulty definitions in
`configs/difficulty_thresholds.json`, all Point-IK thresholds, and the Point-IK classification
priority (`near_singularity` > `near_joint_limit` > `large_orientation_change` > `far_target` >
`medium_target` > `near_target`). The four diagnostic flags are still computed independently from
the global definitions for every candidate and stored regardless of the class assigned.

### Locked minimum core scale (spec section H.1)

`configs/trajectory_config.json:minimum_scale_gate`:
`minimum_core_accepted_scale = 0.50`, `minimum_scale_status = "locked"`, `enforced = true`,
`minimum_scale_rationale = "Preserve at least half of nominal core trajectory geometry"`.

The gate is a **hard floor on the search ladder**, not a post-hoc filter: the geometry/scale search
never descends below 0.50, so a trajectory that cannot be certified at or above the gate fails
generation outright (naming the anchor, shape, orientation mode, every alternative tried and the
effective floor) rather than being written below the gate and rejected later. The validator
independently fails any on-disk trajectory below 0.50. Nominal geometry is untouched (line 0.12 m,
circle r = 0.045 m, figure-8 0.05/0.03 m, helix r/h = 0.04/0.08 m).

### Frozen-test seed reset (spec section H.3)

`configs/seed_policy.json:frozen_core_seed_revision = 2`, with revision 1 recorded as
`burned_not_shippable` and the reason ("frozen_test content was generated into temporary roots and
observed during Phase 5 / Phase 5.1 while the generator policy was still being tuned").

`frozen_test` anchor split assignment and `frozen_test` core-trajectory path seeds and free-form
templates mix in the revision; `development`/`validation` keep their existing seed namespace/tags.
Documented honestly in `configs/seed_policy.json:frozen_core_seed_policy`: because anchor split
membership comes from a single per-class permutation with exact 2/1/1 quotas, frozen membership is
the complement of development+validation, so bumping the revision necessarily re-permutes all three
splits -- with exact quotas there is no way to redraw frozen membership while holding the others
fixed. This is acceptable here because Phase 5.2 regenerates every anchor and every trajectory
anyway (the class predicate changed), so no split retains any prior content. Core-trajectory path
seeds are per-trajectory, so there only `frozen_test` trajectories mix in the revision.

Regenerated `frozen_test` data was used **only** for integrity, reachability, checksum and count
validation. It was not used to tune the anchor predicate, geometry alternatives, minimum scale, or
reachability tolerance -- all four were locked before this run started.

### Files modified

- `dataset_v2/config_templates.py` -- `ANCHOR_NEAR_LIMIT_MIN_SIGMA_MIN` /
  `ANCHOR_NEAR_SINGULAR_MIN_NORMALIZED_MARGIN` / `MINIMUM_CORE_ACCEPTED_SCALE` /
  `FROZEN_CORE_SEED_REVISION` (+ history); `anchor_config()` gained
  `anchor_class_isolation_status` and `class_eligibility_predicates` (replacing the Phase 4
  `overlap_policy`); `trajectory_config()`'s `minimum_scale_gate` locked at 0.50 and enforced;
  `seed_policy_config()` gained the frozen revision, its history and the frozen seed policy note.
- `dataset_v2/anchor_generation.py` -- new `class_eligibility_masks` (isolated, mutually exclusive)
  and `candidate_availability_report`; `_select_class_candidates` now draws from the isolated pool
  with no fallback and raises with the availability breakdown when short; split-assignment seed
  mixes in the frozen revision; report carries `anchor_class_isolation_status`,
  `frozen_core_seed_revision`, `candidate_availability` and `selection_report_by_class`.
- `dataset_v2/anchor_validation.py` -- validates the *isolated* predicate for each anchor's own
  stored class (a near_limit anchor with `sigma_min <= 0.09`, or a near_singular anchor that is
  also near-limit, now fails), requires `anchor_class_isolation_status == "locked"`, and names the
  offending anchors. The now-unreachable `_classify_single` priority helper was removed.
- `dataset_v2/core_trajectory_generation.py` -- the locked gate floors the search ladder;
  below-gate failures report the effective floor and every alternative tried; `frozen_test`
  trajectories derive path seeds and free-form templates from the revised namespace; the report
  records the frozen revision and seed policy.
- `specs/DLS_DATASET_V2_SPEC.md` -- new sections H.2 (anchor class isolation) and H.3 (frozen-test
  seed reset); H.1's minimum-scale gate moved from `[PROVISIONAL]`/unlocked to `[LOCKED]` at 0.50;
  section G amended with a pointer to H.2 (threshold values unchanged).
- **New tests**: `tests/test_dataset_v2_anchor_class_isolation.py` (21 tests).
- **Modified tests**: `tests/test_dataset_v2_core_trajectory_reachability.py` (the Phase 5.1
  "gate disabled by default" test became "gate locked and enforced", plus two new gate tests).

### Candidate availability under the isolated predicates (full locked pools, 5,000 per sub-pool)

| Quantity | Count |
| --- | --- |
| total candidates | 15,000 |
| raw `is_near_limit` | 5,513 |
| ├ well-conditioned (`sigma_min > 0.09`) -> **eligible near_limit** | **2,204** |
| ├ moderately conditioned (`0.03 < sigma_min <= 0.09`) -> rejected | 2,454 |
| └ also near-singular -> rejected | 855 |
| raw `is_near_singular` | 3,000 |
| ├ margin above threshold -> **eligible near_singular** | **2,145** |
| └ also near-limit -> rejected | 855 |
| **eligible regular** | **4,250** |

Required 6 / 3 / 3 -- ample headroom in every class, so no threshold was relaxed and no pool
resize was needed. The 2,454 rejected moderately-conditioned near-limit candidates are exactly the
population Phase 5.1's `anchor_near_limit_02` (`sigma_min` 0.06254) came from.

### Regenerated anchors (12, isolated classes)

| Split | Anchor | Class | `sigma_min` | normalized margin | controlling joint |
| --- | --- | --- | --- | --- | --- |
| development | `anchor_regular_00` | regular | 0.09047 | 0.082259 | 1 |
| development | `anchor_regular_01` | regular | 0.09211 | 0.070131 | 2 |
| development | `anchor_near_limit_00` | near_limit | 0.12256 | 0.024303 | 1 |
| development | `anchor_near_singular_00` | near_singular | 0.00303 | 0.746419 | 3 |
| validation | `anchor_regular_02` | regular | 0.15550 | 0.041864 | 2 |
| validation | `anchor_regular_03` | regular | 0.13032 | 0.817778 | 0 |
| validation | `anchor_near_limit_01` | near_limit | 0.18864 | 0.009227 | 4 |
| validation | `anchor_near_singular_01` | near_singular | 0.02820 | 0.056049 | 3 |
| frozen_test | `anchor_regular_04` | regular | 0.22975 | 0.097722 | 5 |
| frozen_test | `anchor_regular_05` | regular | 0.25184 | 0.045993 | 4 |
| frozen_test | `anchor_near_limit_02` | near_limit | 0.16824 | 0.007992 | 6 |
| frozen_test | `anchor_near_singular_02` | near_singular | 0.02625 | 0.123237 | 2 |

Counts: 12 total, 6/3/3 by class, 4/4/4 by split, 2/1/1 per class per split -- all exact.
Class ranges: `regular` sigma [0.09047, 0.25184] margin [0.041864, 0.817778], joints {0,1,2,4,5};
`near_limit` sigma [0.12256, 0.18864] margin [0.007992, 0.024303], joints {1,4,6} (three distinct,
as required); `near_singular` sigma [0.00303, 0.02820] margin [0.056049, 0.746419], joints {2,3}.
Every `near_limit` anchor is now comfortably above the 0.09 conditioning floor (the Phase 5.1
offender was 0.06254) and every `near_singular` anchor is comfortably interior in joint space.
Pairwise joint-space diversity (min/mean): regular 7.44/12.32, near_limit 18.01/19.41,
near_singular 8.42/-- ; `near_duplicate_pairs = 0`. Anchor generation ran in **6.7 s** and the
independent anchor validator returned **passed=True** on all 12.

### Core trajectory regeneration -- **5 of 120 FAILED the locked gate**

Full run, temporary root, `--master-seed 42`, no overrides, source waypoint count 2001,
**24.0 min** wall-clock. The generator processed all 120 (no trajectory skipped), then **refused to
write a partial core set** and exited 2. No manifest, generation report, reachability report or
checksum update was produced, so no validator/anti-leakage/checksum pass could be run against a
120-trajectory set -- correct behaviour under the locked count.

**The 5 failing trajectories** (each exhausted every geometry alternative at every scale down to
the 0.50 floor):

| # | Trajectory | Anchor | Anchor class | Split | Alternatives x scales tried |
| --- | --- | --- | --- | --- | --- |
| 16 | `core_figure8_variable_anchor_near_limit_01` | `anchor_near_limit_01` | near_limit | validation | 15 |
| 19 | `core_free_form_fixed_anchor_near_limit_01` | `anchor_near_limit_01` | near_limit | validation | 20 |
| 20 | `core_free_form_variable_anchor_near_limit_01` | `anchor_near_limit_01` | near_limit | validation | 20 |
| 55 | `core_figure8_fixed_anchor_near_singular_02` | `anchor_near_singular_02` | near_singular | frozen_test | 15 |
| 56 | `core_figure8_variable_anchor_near_singular_02` | `anchor_near_singular_02` | near_singular | frozen_test | 15 |

The 115 trajectories that did pass are healthy and show a large improvement over Phase 5.1:

| Statistic | Phase 5.1 (gate off, 120 written) | Phase 5.2 (gate 0.50, 115 passing) |
| --- | --- | --- |
| min accepted scale | 0.1209 | **0.6141** |
| P05 | 0.4375 | 0.6141 |
| median | 1.0 | 1.0 |
| mean | 0.8994 | **0.9700** |
| at scale 1.00 | 92 / 120 | **103 / 115** |
| in [0.75, 1.00) | 7 | 5 |
| in [0.50, 0.75) | 11 | 7 |
| below 0.50 | 10 | **0** |

Strict independent-FK reconstruction across the 115: canonical position max **7.46e-05 m**
(P95 5.00e-05 m), source position max **8.38e-05 m** (P95 6.66e-05 m) -- all inside the 1e-4 m
tolerance, with orientation likewise inside 0.01 deg.

### Diagnosis of the 5 failures

The failures are **not** caused by the anchor class confound Phase 5.2 fixed (that is resolved --
`anchor_near_limit_02`, whose Phase 5.1 predecessor collapsed to scale 0.12, now completes all ten
of its trajectories, free-form at scale 1.00). They are caused by a newly-exposed tension between
three independently locked constraints:

1. `near_limit` requires `normalized_joint_limit_margin <= 0.024991237796029034` (an anchor ~3 deg
   from a joint limit in absolute terms), and `near_singular` requires `sigma_min <= 0.03`.
2. Nominal core geometry is fixed (circle r = 0.045 m, figure-8 0.05/0.03 m, free-form endpoint
   0.10 m).
3. Every trajectory must be certified at `accepted_scale >= 0.50` under a 1e-4 m independent-FK
   criterion.

For `figure8` and `free_form` on the most constrained anchors these cannot all hold at once.
`figure8` is the worst case: it is a **closed** path that must traverse a full lemniscate in a
plane, so unlike `line`/`helix` it cannot be aimed away from the limiting joint or the singular
direction -- 3 of the 5 failures are figure-8, and both figure-8 failures on
`anchor_near_singular_02` are the fixed *and* variable pair, i.e. the position path alone is the
blocker, not the orientation profile. `free_form` fails on `anchor_near_limit_01` in both
orientation modes for the same reason: all four control-point templates place the 0.05 m (at
scale 0.50) endpoint in directions that drive joint 4 into its limit.

Note also an important measurement correction: the three `near_limit` anchors' *normalized* margins
(0.0243 / 0.0092 / 0.0080) look very different, but in absolute terms they are nearly identical
(3.04 deg / 3.32 deg / 2.88 deg) -- the spread is the joint_2/joint_4 half-range asymmetry
documented in `docs/V2_THRESHOLD_CALIBRATION.md`, not a real difference in tightness. So the
discriminator between a near_limit anchor that succeeds everywhere (`anchor_near_limit_00`,
joint 1) and one that fails on two shapes (`anchor_near_limit_01`, joint 4) is **which** joint is
limiting and how that joint must move for a given shape -- not how tight the margin is.

Two of the five failures are on a `frozen_test` anchor. They are reported here because the phase
requires exact failing items to be named, and they were used for **nothing else**: no threshold,
predicate, alternative set, scale gate or tolerance was changed in response to them.

### Options (each requires an explicit user decision; none taken unilaterally)

1. **Widen the geometry-alternative set for the two failing shapes.** `circle`/`figure8` currently
   have 3 plane bases and no traversal-direction (handedness) variant, while `line` and `helix`
   have 6 alternatives each including sign flips; `free_form` has 4 templates. Adding
   handedness variants to the closed shapes (6 instead of 3) and more free-form templates enlarges
   the deterministic search without touching any threshold, gate, count or nominal geometry. This
   is the smallest change and the only one that alters no locked policy -- but the present
   alternative set is exactly what Phase 5.1 specified, so extending it is a spec change.
2. **Add a lower bound to the `near_limit` band and a companion bound for `near_singular`.** e.g.
   require `normalized_joint_limit_margin >= 0.0125` (half the band). This would have excluded
   `anchor_near_limit_01`. It does not help `anchor_near_singular_02`, whose failure is driven by
   `sigma_min`, so it would need a matching `sigma_min` lower bound for the near-singular class --
   which starts to erode what "near-singular" means.
3. **Reduce nominal closed-shape geometry** (circle radius, figure-8 amplitudes) so that scale 1.00
   denotes a smaller loop and 0.50 becomes attainable everywhere. Changes the benchmark's nominal
   geometry for all 24 circle and 24 figure-8 trajectories.
4. **Lower the locked gate** (e.g. 0.35). Explicitly rejected here -- the phase locks it at 0.50
   and forbids relaxation.

Recommendation: **option 1**, and if that is insufficient, option 1 + option 2's `near_limit`
lower bound. Option 1 changes no locked numeric policy (gate, tolerances, thresholds, counts,
nominal geometry all stay exactly as locked) and directly addresses the measured cause: the closed
shapes are the ones failing and they have half the search freedom of the open shapes.

### Tests

- Targeted Phase 5.2: `pytest tests/test_dataset_v2_anchor_class_isolation.py -q` -> **21 passed**.
- Anchor v2 (including the Phase 4 overlap-fallback test rewritten as an isolated-pool test):
  `pytest tests/test_dataset_v2_anchor_generation.py tests/test_dataset_v2_anchor_validation.py -q`
  -> **46 passed**.
- Full suite: `pytest -q` -> **590 passed, 0 failed, 0 skipped, 0 errors** (567 Phase 5.1 baseline
  + 21 new isolation tests + 2 new gate tests, with 1 Phase 4 overlap test rewritten in place).

### Confirmations

- Dataset v1: **not modified**. `git diff --name-only` restricted to `assets/`, `benchmarks/`,
  `trajectories/`, `configs/`, `schemas/`, `kinematics/`, `algorithms/`, `generators/`,
  `evaluation/`, root `VERSION` and root `DATASET_MANIFEST.json` returns zero files. HEAD unchanged
  at `d1f017f0cba49e51e05c60a8a2124c343d70bc6d`; nothing committed, nothing pushed. The Phase 5 and
  Phase 5.1 working-tree changes were preserved throughout (none reverted).
- Point-IK thresholds, difficulty definitions and classification priority: **unchanged**.
  FK/Jacobian/SO(3)/DLS baseline: **unchanged**. Strict generation tolerance: **unchanged**
  (1e-4 m / 0.01 deg). Locked counts (12 anchors, 120 trajectories, 48,000 poses):
  **unchanged** -- the shortfall is reported as a failure, not absorbed by lowering a count.
- Generated data: anchors and 115 trajectories were produced in a temporary root for verification
  and the whole root was **deleted** afterwards. No NPZ/CSV/JSON is committed.
- No random challenge trajectories, no trials, no Tier 1-4 DLS baseline evaluation, no PPO/MPDIK/
  MAPPO work was performed.

### Completion gate

| Gate condition | Result |
| --- | --- |
| 12 anchors generated and validated | **PASS** (validator `passed=True`) |
| class counts 6/3/3 | **PASS** |
| split counts 4/4/4, 2/1/1 per class per split | **PASS** |
| all `near_limit` anchors `sigma_min > 0.09` | **PASS** (0.12256 / 0.18864 / 0.16824) |
| all `near_singular` anchors not near-limit | **PASS** (margins 0.746 / 0.056 / 0.123) |
| candidate availability sufficient without relaxing | **PASS** (2204 / 2145 / 4250 vs 3 / 3 / 6) |
| 120 core trajectories | **FAIL** -- 115 certified, 5 could not reach the locked gate |
| 48,000 canonical poses | **FAIL** (not reached; no partial set written) |
| every accepted scale >= 0.50 | **PASS** for all 115 written (min 0.6141); the 5 failures are precisely those that could not |
| strict source + canonical FK reconstruction | **PASS** on all 115 (canonical max 7.46e-05 m, source max 8.38e-05 m, tolerance 1e-4 m) |
| anti-leakage / checksums / manifest | **NOT REACHED** -- generation aborted before writing them, by design |
| all tests pass | **PASS** |
| Dataset v1 unchanged | **PASS** |

### Phase 5 final status: **INCOMPLETE**

Five of the 120 core trajectories cannot be certified at the locked `minimum_core_accepted_scale =
0.50` under the locked strict reachability criterion with the locked anchor classes and the locked
nominal geometry. Per the phase rules the gate was **not** relaxed, the locked count was **not**
reduced, no trajectory was skipped, and no partial core set was written. The exact failing items,
their anchors, splits and search budgets are listed above, along with four options and a
recommendation -- all of which require an explicit user decision.

What Phase 5.2 did achieve and lock:

- Anchor class isolation is locked and enforced end to end (config, generator, validator, tests);
  the Phase 5.1 confound is gone, and the anchor that previously collapsed to scale 0.12 now
  completes all ten of its trajectories.
- `minimum_core_accepted_scale = 0.50` is locked, enforced as a hard floor on the search ladder and
  independently re-checked by the validator.
- The frozen-test seed namespace is reset to revision 2, with revision 1 recorded as burned.
- Of the 115 trajectories that pass, the accepted-scale distribution improved substantially over
  Phase 5.1 (mean 0.899 -> 0.970, minimum 0.121 -> 0.614, zero below 0.50).

### Recommended next step

Adopt option 1: extend `configs/trajectory_config.json:geometry_alternatives` so the closed shapes
have the same search freedom as the open ones -- add traversal-direction (handedness) variants to
`circle` and `figure8` (3 -> 6 alternatives) and additional deterministic `free_form` templates
(4 -> 8). This changes no locked threshold, gate, tolerance, count or nominal geometry; it only
enlarges the deterministic geometry search that the phase already performs, and it targets the
measured cause (all 5 failures are on the two shapes with the least search freedom). Re-run anchor
+ core generation and re-evaluate the completion gate. If failures persist, combine with a
`near_limit` normalized-margin lower bound (option 2).

## Phase 5.3 -- Generic geometry-alternative expansion for closed shapes and free-form

- **Baseline before this phase**: branch `feature/dataset-v2`, commit
  `d1f017f0cba49e51e05c60a8a2124c343d70bc6d`, working tree carrying the uncommitted Phase 5,
  Phase 5.1 and Phase 5.2 files (17 files, all accounted for, none reverted).

### Audit of the alternative space before this phase

| Shape | Plane/axis bases | Direction signs | Handedness | Start phase / anchor placement | Templates | **Total** |
| --- | --- | --- | --- | --- | --- | --- |
| `line` | 3 axes | **yes** -- both signs of each axis (6 directions) | n/a | fixed (starts at anchor, travels `+dir`) | n/a | **6** |
| `circle` | 3 plane bases | **no** | **no** -- always traversed `+u -> +v` | **no** -- centre always `p0 - r*u` (phase 0) | n/a | **3** |
| `figure8` | 3 plane bases | **no** | **no** -- lobe signs fixed `(+a, +b)`; no major/minor axis swap | **no** | n/a | **3** |
| `helix` | 3 axis bases | **yes** -- `height_sign` in `{+1, -1}` | n/a | fixed | n/a | **6** |
| `free_form` | n/a | n/a | **no** mirror variants | fixed (starts at anchor) | 4 seeds, one generation rule | **4** |
| | | | | | | **22 total** |

**Why `line`/`helix` had more freedom.** Both enumerate a *sign*: `line` covers all six signed local
axes, so the path can be aimed away from whichever joint is near its limit; `helix` enumerates the
travel direction along its axis. Both therefore have an escape route in joint space.

**Why `circle`/`figure8`/`free_form` had less.** `circle` and `figure8` varied only the *plane* the
curve lives in. Within a plane they were always traversed in one sense (`cos u + sin v` for the
circle; `(+a, +b)` lobes for the figure-8), always started at the same phase (circle centre pinned
to `p0 - r*u`), and the figure-8 never swapped which basis axis carried the major amplitude. So for
an anchor whose limiting joint blocks that one traversal sense, there was no alternative to try --
only shrinking. `free_form`'s four templates all came from a single rule (a forward direction
biased toward local `+x` plus seeded lateral offsets) with no mirrored variants and no controlled
spread of departure directions.

**Consistency with the observed failures.** Phase 5.2's *development/validation* failures were
`figure8_variable`, `free_form_fixed` and `free_form_variable` on `anchor_near_limit_01` -- i.e.
exactly the shapes with the least search freedom, and none on `line`/`helix`. This, together with
the plain geometric asymmetry above, is the justification for the expansion; it is a generic
symmetry-completion applied identically to every split. The two `frozen_test` revision-2 failures
(`figure8_fixed`/`figure8_variable` on `anchor_near_singular_02`) are recorded only and were **not**
used to choose or size the new alternative set.

### Expanded alternative set (locked, machine-readable)

| Shape | Before | After | New dimensions |
| --- | --- | --- | --- |
| `line` | 6 | 6 | unchanged |
| `circle` | 3 | **12** | traversal direction (`ccw`/`cw`) x start phase (`0`, `pi`) |
| `figure8` | 3 | **24** | amplitude-sign pairs (handedness + traversal reversal) x major/minor axis swap |
| `helix` | 6 | 6 | unchanged |
| `free_form` | 4 | **8** | six departure directions + two mirrored variants |
| **total** | **22** | **56** | |

`s -> 1-s` on the figure-8 maps exactly to `(-sa, -sb)`, so traversal reversal lives inside the
four sign pairs and is not enumerated twice; a test asserts this identity. No two alternatives of a
shape produce byte-identical geometry, and every alternative places the anchor pose exactly at the
first waypoint (both asserted by tests). For `circle`/`figure8` the variable-orientation rotation
vector is multiplied by the alternative's traversal/handedness sign so the orientation profile
follows the path direction.

### Selection policy

Per alternative, walk the scale schedule from 1.0 down to the 0.50 gate and record its largest
strictly-reachable scale; then take the alternative with the largest accepted scale. The search
never stops at the first alternative that clears the gate — an alternative is abandoned early only
when its remaining schedule can no longer beat the incumbent, which cannot change the winner.
Tie-break: smallest strict position error -> fewest refinement/restart attempts (diagnostic only)
-> `alternative_id` lexical order. DLS runtime, iteration count, baseline success and frozen-test
results are all forbidden signals.

### Full temporary run -- 120/120 generated, all gates pass

Temporary root, `--master-seed 42`, no overrides, source waypoint count 2001. Anchors regenerated
under the unchanged Phase 5.2 isolated predicates (6.7 s, validator `passed=True`, 12 anchors,
6/3/3, 4/4/4, 2/1/1 per class per split). Core generation **50.9 min**; independent validation
**43.9 s** -> `total=120 canonical_poses=48000 passed=True`.

| | development | validation | frozen_test (rev 3) |
| --- | --- | --- | --- |
| trajectories | 40 | 40 | 40 |
| shapes (each) | 8 | 8 | 8 |
| orientation fixed/variable | 20 / 20 | 20 / 20 | 20 / 20 |
| accepted scale min | **1.0000** | **0.6141** | **0.7225** |
| accepted scale P05 | 1.0000 | 0.8382 | 0.8436 |
| accepted scale median | 1.0000 | 1.0000 | 1.0000 |
| accepted scale mean | 1.0000 | 0.9732 | 0.9824 |
| in `[0.50, 0.75)` | 0 | 2 | 2 |
| in `[0.75, 1.00)` | 0 | 2 | 1 |
| at `1.00` | **40** | 36 | 37 |
| strict pos canonical max / P95 | 5.00e-05 / 5.00e-05 | 4.99e-05 / 4.89e-05 | 4.99e-05 / 4.98e-05 |
| strict pos source max / P95 | 5.00e-05 / 5.00e-05 | 9.95e-05 / 5.14e-05 | 7.20e-05 / 5.00e-05 |
| strict ori canonical max / P95 | 4.95e-03 / 4.80e-03 | 5.00e-03 / 5.00e-03 | 5.00e-03 / 4.99e-03 |
| strict ori source max / P95 | 5.00e-03 / 4.99e-03 | 1.00e-02 / 9.98e-03 | 9.90e-03 / 5.00e-03 |
| canonical waypoints reachable | 16000/16000 | 16000/16000 | 16000/16000 |
| source waypoints reachable | 80040/80040 | 80040/80040 | 80040/80040 |
| alternatives attempted | 458 | 647 | 696 |

Overall: min accepted scale **0.6141**, mean **0.9852**, **113/120 at scale 1.00**, only 7
requiring any shrink, **0 below 0.50**. Strict reconstruction stays inside the unchanged 1e-4 m /
0.01 deg tolerances everywhere (canonical position max 5.00e-05 m, source position max 9.95e-05 m).
Closed-path closure for all 48 `circle`/`figure8` trajectories: position error **0.0 m**,
variable-orientation closure **0.0 rad**. `total_restarts_used_diagnostic = 0`.

Selected-alternative family histogram (1,801 attempts across the 120 trajectories;
**114/120 ended on a non-default alternative**):

| Family | Count |
| --- | --- |
| `line_direction` | 24 |
| `circle_cw` | 13 |
| `circle_ccw` | 11 |
| `figure8_right_forward` | 11 |
| `figure8_left_reversed` | 7 |
| `figure8_left_forward` | 6 |
| `free_form_direct` | 18 |
| `free_form_mirrored` | 6 |
| `helix_p` | 12 |
| `helix_n` | 12 |

Both circle traversal directions, three of the four figure-8 sign families, and both free-form
mirror states are genuinely selected — the new dimensions are load-bearing, not decorative.

### Status of the five Phase 5.2 failures

| Trajectory | Split | Phase 5.2 | Phase 5.3 | Winning alternative |
| --- | --- | --- | --- | --- |
| `core_figure8_variable_anchor_near_limit_01` | validation | FAILED | **scale 1.0000** | `figure8_xy_anbp_swap` |
| `core_free_form_fixed_anchor_near_limit_01` | validation | FAILED | **scale 1.0000** | `free_form_ff0` |
| `core_free_form_variable_anchor_near_limit_01` | validation | FAILED | **scale 1.0000** | `free_form_ff0` |
| `core_figure8_fixed_anchor_near_singular_02` | frozen rev 2 | FAILED | **scale 1.0000** | `figure8_yz_apbn_noswap` |
| `core_figure8_variable_anchor_near_singular_02` | frozen rev 2 | FAILED | **scale 1.0000** | `figure8_yz_anbn_noswap` |

All three *validation* failures — the ones that justified the expansion — are resolved at full
nominal scale. The two frozen-revision-2 failures also resolve under revision 3, but that outcome
was neither used to design nor to size the alternative set: the expansion was fixed and locked
before this run, and frozen revision 3 was only integrity/reachability/checksum/count validated.

### Integrity gates

- Independent validator: `passed=True` (120 trajectories, 48,000 canonical poses).
- Checksum manifest: **0 mismatches**.
- Anti-leakage: `collisions_found = 0`, `pass = true` across `trajectory_id`,
  `trajectory_content_hash` and `anchor_id`.
- `DATASET_MANIFEST.json`: `core_trajectories.total = 120`, splits 40/40/40, shapes 24 each,
  orientations 60/60, `canonical_poses_total = 48000`, `generated: true`.
- `frozen_core_seed_revision = 3`; revisions 1 and 2 both recorded `burned_not_shippable`.

## Phase 5.4 -- Deterministic seed fix + feasibility-aware anchor selection

- **Baseline before this phase**: branch `feature/dataset-v2`, commit
  `d1f017f0cba49e51e05c60a8a2124c343d70bc6d`, working tree carrying uncommitted Phase 5-5.3.

### Two defects fixed

**(a) NumPy-version-dependent seed derivation.** Dataset v1's `generators/_common.py::derive_seed`
ends with `np.uint64 % int`, which NumPy 1.x promotes to `float64` (lossy above 2**53) and NumPy
2.x (NEP 50) keeps exact. Every derived seed therefore differed between NumPy majors, producing a
different dataset from the same master seed -- observed directly: NumPy 1.26 gave 0 core-trajectory
failures, NumPy 2.4 gave 2, both from seed 42. This violated the [LOCKED] byte-identical
regeneration rule (spec section E). Fixed with a **Dataset-v2-only** module `dataset_v2/seeds.py`:
SHA-256 over a canonical byte encoding, reduced with pure Python integer arithmetic; no NumPy type
touches the math. Dataset v1's `derive_seed` is left byte-identical (v1 is an immutable baseline).
All six v2 generators migrated; zero legacy imports remain. Frozen revision advanced 3 -> **4**
(revision 3 was generated pre-fix and is now `burned_not_shippable`).

**(b) Feasibility-blind anchor selection.** An anchor's class predicate says nothing about whether
its neighbourhood can support the locked core geometry. Three seed realizations produced 0, 2 and 4
below-gate core trajectories purely from which anchors were drawn. Fixed with feasibility-aware
selection: a candidate may enter the catalog only if all ten locked (shape, orientation)
combinations reach `accepted_scale >= 0.50`.

### Feasibility screening (spec section H.5, `dataset_v2/anchor_feasibility.py`)

- **Two-stage selection** (`configs/anchor_config.json:feasibility_screening`, status `locked`):
  Stage A screens eligible candidates -- in deterministic greedy farthest-point order -- keeping
  only those passing 10/10; Stage B runs the existing diversity selection over the feasible subset
  only. A rejected candidate is replaced from the SAME class pool; classes never mix, no
  combination is skipped, partial feasibility is never accepted.
- **Gate-rung probe.** Feasibility is existential ("is *some* scale >= 0.50 reachable?"), so the
  probe tests the smallest scheduled scale >= the gate -- the cheapest place to confirm a feasible
  candidate. A success there is a real reachable trajectory at a scale >= the gate, so the screen
  cannot falsely accept; the generator maximizes the accepted scale afterwards and can only do
  better. The coarse probe (30 canonical / 151 source waypoints) is a screen, never an
  acceptance: the winner is still generated and validated at full 400/2001 resolution, and the
  independent core-trajectory validator re-checks the final 120 without trusting the screen.
- **Content-keyed cache.** Process-lifetime cache keyed on q (12 dp), model fingerprint,
  geometry-config fingerprint, reachability-config fingerprint, seed algorithm id and probe
  resolution -- never a path, root or timestamp. A config change yields a different key (correct
  invalidation) rather than a stale hit. Turns a repeated `run_anchor_generation` from 127.5 s to
  0.7 s; a production run screens once and is unaffected.
- **Per-anchor evidence** stored in `anchors.npz` (`feasibility_passed`,
  `feasibility_combinations_passed`, `feasibility_worst_accepted_scale`) plus a full screening
  section in the generation report (screened/passed/rejected per class, failure histogram, cache
  hits/misses, config fingerprints, per-anchor feasibility matrix and worst combination).
- **Validator hardening**: anchor validator fails on missing feasibility evidence, fewer than 10
  combinations, verified scale below the gate, geometry/reachability fingerprint mismatch, or
  report evidence that does not match the catalog entry.

### Files

- New: `dataset_v2/seeds.py`, `dataset_v2/anchor_feasibility.py`,
  `tests/test_dataset_v2_seed_determinism.py`, `tests/test_dataset_v2_anchor_feasibility.py`.
- Modified: `dataset_v2/{tier0_generation,threshold_calibration,point_ik_generation,
  anchor_generation,core_trajectory_generation,generation_reachability,core_trajectory_validation,
  anchor_validation,config_templates}.py`; several Phase 5.2/5.3 tests updated for the new
  selected-source label, the frozen revision (now derived from config, not hard-coded), and the
  expanded anchor NPZ fields.

### Full temporary run (deterministic seeds, feasibility screening, deleted after inspection)

- Anchor screening: ~2 min, rejected 3 near_singular candidates failing `figure8_fixed` (x2) and
  `helix_fixed` (x1); all 12 selected pass 10/10 at verified scale 0.5220. Validator `passed=True`.
- Core generation: **120/120, zero failures**, `EXIT=0`. Independent validator: **120 trajectories,
  48,000 canonical poses, passed=True**.
- Counts: split 40/40/40, shapes 24 each, orientations 60/60, 10 per anchor, 400 waypoints each.
- Accepted scale: min **0.6141**, P05 1.0, median 1.0, mean **0.9911**; bands [0.50,0.75)=2,
  [0.75,1.0)=2, ==1.0=**116**; all >= 0.50.
- Strict reconstruction (tolerance 1e-4 m / 0.01 deg): canonical position max 5.00e-05 m
  (P95 4.98e-05), source position max 5.00e-05 m (P95 5.00e-05); canonical orientation max
  5.00e-03 deg, source orientation max 9.95e-03 deg. All 48,000 canonical and 240,120 source
  waypoints reachable.
- Closed-path closure (48 circle/figure8): position 0.0 m, variable-orientation 0.0 rad.
- Integrity: checksum mismatches 0; anti-leakage collisions 0 (pass=true); schema validation of all
  120 records 0 errors; `seed_algorithm_id = dataset_v2/seed/sha256/v1`; frozen revision 4 with
  1/2/3 burned and exactly one active; no legacy derive_seed in dataset_v2/.
- **All 10 previously-failing trajectories now PASS** (8 at scale 1.0, 2 at 0.6141), e.g.
  `core_figure8_variable_anchor_near_limit_01` @1.0 via `figure8_xy_anbn_swap`,
  `core_helix_fixed_anchor_near_limit_01` @1.0 via `helix_xyz_n`.

### Tests (`.venv` = Python 3.11.9, NumPy 2.4.6)

- Phase 5.4 anchor feasibility: **31 passed**. Seed determinism: **16 passed**.
- Anchor generation+validation+isolation: **67 passed**.
- Core trajectory generation+validation+reachability+geometry: **112 passed**.
- Point-IK+Tier0+calibration+scaffold+root-resolution: **127 passed**.
- Dataset v1 backward compatibility: **130 passed**.
- **Full suite: 672 passed, 0 failed, 0 skipped, 0 errors.**

### Confirmations

- Dataset v1: **not modified** (0 paths under assets/benchmarks/trajectories/configs/schemas/
  kinematics/algorithms/generators/evaluation/utils/VERSION/DATASET_MANIFEST.json). HEAD unchanged
  `d1f017f0`; nothing committed or pushed.
- Nothing locked weakened: class predicates, near-limit/near-singular/regular thresholds, the 0.50
  gate, 1e-4 m / 0.01 deg tolerances, nominal geometry, 12/6-3-3/4-4-4, 120/400/48,000, frozen
  revision 4 -- all unchanged. Feasibility screening only *removes* candidates.
- Generated data produced in a temporary root and **deleted** after inspection; none committed.

### Phase 5 final status: **COMPLETE**

Every completion gate passed: deterministic seed derivation; 12 anchors each with 10/10 feasibility
evidence; anchor validator; 120/120 trajectories; 48,000/48,000 canonical poses; all accepted
scales >= 0.50; strict source and canonical reachability; schema; checksums; manifest;
anti-leakage; frozen revision 4; all targeted and regression tests; full pytest; Dataset v1
unchanged.

### Recommended Phase 6

Random challenge trajectory generator (90, split 30/30/30), reusing this phase's strict
reachability engine, geometry-alternative search and feasibility-aware machinery, followed by the
easy/medium/hard trial generator (630 trials).

## Phase 6 -- Random-challenge trajectory generator, validator, CLI

- **Status**: COMPLETE. Random-challenge trajectories only -- no trials, no DLS baseline
  evaluation, no Tier 1-4 evaluation.
- **Baseline before this phase**: branch `feature/dataset-v2`, commit `2ba2241d`, clean working
  tree, `pytest -q` -> **672 passed**, 0 failed/skipped/errored (confirmed by a full run this
  phase; this machine is ~5x slower than Phase 5.4's `.venv`, so the suite took 34 min here).

### Locked challenge generation policy (spec section I.1)

Phase 0 locked only the 90 / 30-30-30 / 400-canonical counts and left the challenge *generation*
policy `[PROVISIONAL]`. Phase 6 locks it, machine-readable in
`configs/random_challenge_config.json` (`status: counts_and_policy_locked_generation_implemented`).
The initial `[PROVISIONAL]` note proposed interpolating between randomly drawn Cartesian control
poses; Phase 6 instead adopts a **smooth joint-space reference family through FK**, which is
strictly stronger on reachability (the property the locked count depends on) and is the reason the
90/90 outcome is a property of the policy rather than luck.

- **Mechanism**: each trajectory starts from an **independent reachable start state** (never one of
  the 12 anchors) and follows a bounded-Fourier joint-space curve
  `q(s)=q_start+offset(s)`, `offset_j(0)=0`, with each joint's amplitude capped at
  `envelope_margin_fraction` (0.90) of that joint's own start joint-limit margin. The reference
  therefore stays inside operational limits with no clipping (C-∞ smooth, bounded curvature); every
  source pose = FK(q(s)) is reachable by construction; orientation is the genuine FK orientation of
  the joint curve. Source (1201 FK samples, > 400) + canonical (400, arc-length-resampled via the
  same `resample_canonical` as core: piecewise-linear position, SO(3) SLERP orientation, exact
  endpoints, sign-continuous quaternions).
- **Six families** (5 per split × 3 = 15 each; 6 × 15 = 90): `smooth_random`, `mixed_curvature`,
  `non_planar`, `large_orientation`, `near_limit_region`, `near_singular_region` -- each with a
  machine-readable region, harmonic set, per-joint amplitude weights, envelope fraction, curvature
  ceiling, coverage floor (where relevant), seed tag and per-split quota.
  `near_limit_region`/`near_singular_region` start states additionally satisfy the Phase 2.5 locked
  `near_joint_limit` (normalized margin <= 0.024991237796029034) / `near_singularity`
  (`sigma_min` <= 0.03) thresholds.
- **Acceptance policy**: no core-style Cartesian scale gate is invented (challenge paths are not
  scaled shapes). Acceptance = start-state validity (+ family start predicate) + within-limits
  bounded envelope + strict independent-FK reachability on all source AND canonical waypoints +
  finite bounded curvature + family coverage floors (`non_planar` min non-planarity 0.02,
  `large_orientation` min angular displacement 0.30 rad).
- **Feasibility-aware diversity selection** (mirrors Phase 5.4): per (family, split) a seeded
  16-candidate pool is coarse-probe screened for strict reachability + coverage floors; the feasible
  subset is diversity-selected (greedy farthest-point over joint-space / workspace centroid /
  arc-length / angular displacement / curvature / non-planarity / start `sigma_min` / start margin,
  reusing `greedy_farthest_point_select`) down to the quota; the selected candidates are re-validated
  at full 400/1201 resolution; a full-validation failure is replaced deterministically from the same
  pool -- never by loosening reachability, counts or the family policy.

### Seed / frozen policy

`master_seed` -> `derive_seed(master_seed, component_tags["random_challenge"]=50)` = challenge
component seed -> per (family, split) `derive_seed(component_seed, family_tag, split_tag[,
frozen_challenge_seed_revision if frozen_test])` -> per candidate `derive_seed(family_split_seed,
7, candidate_index)`. Split isolation is guaranteed by the distinct `split_tag`; frozen isolation
additionally mixes in the new **`frozen_challenge_seed_revision = 1`** (a SEPARATE namespace,
locked before frozen generation). `frozen_core_seed_revision` stays **4** (unchanged). All seeds
use `dataset_v2/seeds.py` (`seed_algorithm_id = dataset_v2/seed/sha256/v1`); no global numpy.random
state. Same-seed regeneration is byte-identical, different seed changes content (both tested; also
verified on NumPy 1.26.4 here despite Phase 5.4's fix being motivated on NumPy 2.x).

### Files added/modified

- **New**: `dataset_v2/challenge_trajectory_generation.py` (generator: region-based independent
  start draw reusing Tier 0's `_group_random_interior`/`_group_mixed_near_limits`/
  `_build_singularity_candidate_pool`; bounded-Fourier joint reference; FK path; `resample_canonical`
  reused from core; curvature/non-planarity diagnostics; coarse feasibility screen + greedy diversity
  select + full re-validation; NPZ/manifest/reachability/diversity/feasibility/anti-leakage reports;
  manifest + checksum update). `dataset_v2/challenge_trajectory_validation.py` (independent
  validator). `pipelines/run_dataset_v2_challenge_trajectory_generation.py` (CLI).
  `tests/test_dataset_v2_challenge_trajectory_generation.py` (34 tests),
  `tests/test_dataset_v2_challenge_trajectory_validation.py` (15 tests).
- **Modified** (extended, not rewritten): `dataset_v2/config_templates.py` (challenge constants,
  full `random_challenge_config()` family policy, `frozen_challenge_seed_revision` in
  `seed_policy_config`); `dataset_v2/schemas.py` (`challenge_trajectory_schema()`, registered ->
  10 schemas); `dataset_v2/manifest.py` (`apply_challenge_trajectory_generation_status`);
  `specs/DLS_DATASET_V2_SPEC.md` (section I.1); `tests/test_dataset_v2_scaffold.py` (schema count
  9 -> 10).

### Full temporary run (master_seed 42, temporary root, deleted after inspection)

Full locked mode, no overrides (source 1201). Generation **390 s**, independent validation **22 s**:

- **Counts**: 90 total, split **30/30/30**, family **15 each**, family×split **5/5/5** -- all exact.
  400 canonical waypoints each -> **36,000 canonical challenge poses**. `full_locked_counts=True`.
- **Strict reachability** (tolerance 1e-4 m / 0.01 deg, independent FK, DLS-baseline-independent):
  every canonical AND source waypoint reachable. Canonical position max **8.03e-05 m** (P95
  4.98e-05), orientation max **0.0078 deg**; source position max **6.16e-05 m** (P95 4.95e-05),
  orientation max **0.0095 deg** -- all inside tolerance. No waypoint skipped.
- **Diversity** (real, load-bearing): arc length **0.067 - 2.577 m**, cumulative angular
  displacement **0.229 - 6.119 rad**, mean curvature **2.21 - 85.4 1/m** (finite, far below the
  20000 ceiling), non-planarity **0.0071 - 0.1336**. `non_planar` family clears its 0.02
  non-planarity floor; `large_orientation` clears its 0.30 rad angular floor.
- **Determinism**: byte-identical NPZ on same-seed reruns; different content on a different seed.
- **Frozen challenge revision 1**; frozen_test path seeds disjoint from development/validation.
- **Integrity**: independent validator `passed=True` (90 / 36,000); checksum mismatches **0**;
  anti-leakage `pass=true`, 0 collisions (trajectory_id / content_hash / canonical & source path
  hash / path_seed / near-duplicate start+path-metrics / no-duplicate-with-core); schema validates a
  representative record. No NPZ/CSV/JSON committed.
- **Combined 210 / 84,000**: verified against a real full core catalog (12 anchors + 120 core
  trajectories regenerated at the locked config in a separate temporary root; core validator
  `passed=True`, 120 / 48,000). With the real 120-row core manifest present, the challenge
  validator's combined check confirms **210 trajectories / 84,000 canonical poses** and **zero**
  challenge/core content-hash collisions (verified this phase).

### Candidate screening summary

Feasibility-aware selection screened 16 candidates per (family, split) = 288 candidates; the
joint-space construction makes candidates reachable by construction, so the feasible subset always
exceeded the quota and no full-validation replacement was needed in the full run (recorded in
`challenge_trajectory_feasibility_report.json`). Diversity selection genuinely reorders the feasible
pool (recorded `diversity_ranked_indices` are not the natural 0..k order).

### Tests

- Targeted challenge: `pytest tests/test_dataset_v2_challenge_trajectory_generation.py
  tests/test_dataset_v2_challenge_trajectory_validation.py -q` -> **49 passed** (34 + 15).
- Regression (scaffold, touched by the schema-count + config changes):
  `pytest tests/test_dataset_v2_scaffold.py -q` -> **27 passed**.
- Full suite: `pytest -q` -> **721 passed** (672 baseline + 49 new), 0 failed/skipped/errored
  (verified this phase; ~33 min on this machine).

### Confirmations

- Dataset v1: **not modified**. Only `dataset_v2/`, `pipelines/`, `specs/`, `docs/`, `tests/` files
  touched; nothing under `assets/`/`benchmarks/`/`trajectories/`(v1)/`configs/`(v1)/`schemas/`(v1)/
  `kinematics/`/`algorithms/`/`generators/`/`evaluation/`, root `VERSION`, root
  `DATASET_MANIFEST.json`. HEAD unchanged `2ba2241d`; nothing committed or pushed.
- Nothing locked weakened: 120 core policy, stable seed algorithm, `frozen_core_seed_revision=4`,
  anchor predicates, strict reachability (1e-4 m / 0.01 deg), core minimum scale gate 0.50, FK /
  Jacobian / SO(3) / DLS / adaptive damping -- all unchanged. No trials generated, no Tier 1-4
  evaluation, no PPO/MPDIK/MAPPO.
- Official permanent Dataset v2 challenge data: **not generated/committed** -- produced in temporary
  roots for verification and deleted.

### Phase 6 final status: **COMPLETE**

90 random-challenge trajectories generated and independently validated at the locked config: exact
90 / 30-30-30 / 15-per-family / 5-per-family-per-split / 36,000 canonical poses; combined
210 / 84,000 with the core catalog; strict source + canonical reachability under a
DLS-baseline-independent 1e-4 m / 0.01 deg criterion; deterministic; frozen challenge revision 1;
validator / schema / checksum / anti-leakage all green; Dataset v1 unchanged.

### Recommended Phase 7

The easy/medium/hard trial generator (630 trials = 3 per trajectory across all 210 core +
random-challenge trajectories, spec section J), reusing the strict reachability engine and the
independent-start construction introduced here; the trial `q_initial` must be constructed only
against each trajectory's first canonical waypoint pose (never a future waypoint's solution, spec
section J).

## Phase 7 -- Combined trajectory catalog, public/protected loaders, trial generator/validator, CLI

- **Status**: COMPLETE. Trials only -- no DLS evaluation, no Tier 1-4 evaluation, no official
  release.
- **Baseline before this phase**: branch `feature/dataset-v2`, commit `ca516d69`, clean working
  tree, `pytest -q` -> **721 passed**, 0 failed/skipped/errored (full run this phase, 44 min on this
  machine). Locked inputs carried in unchanged: stable seed algorithm `dataset_v2/seed/sha256/v1`,
  `frozen_core_seed_revision = 4`, `frozen_challenge_seed_revision = 1`.

### Persistent working dataset root

Phase 7 uses an explicit **persistent working root outside the repository**:
`D:\data\hoang_anh\mpdik_kassow_v2_work` (passed via `--dataset-root`, never hard-coded in source).
It is a **development working dataset**, NOT an official release and never called frozen/final. It
is not in Git and no generated NPZ/CSV/JSON is committed. It is retained at the end of Phase 7 (Phase
8 will reuse it) and carries full config/model/seed fingerprints. The full 210-trajectory input
(scaffold + Tier 0 1000/1000/600 + Point-IK 6,000 + 12 anchors + 120 core / 48,000 canonical poses +
90 challenge / 36,000 canonical poses) was generated at master seed 42 and validated before any trial
was produced; the combined catalog confirms **210 trajectories / 84,000 canonical poses**.

### Locked trial generation policy (spec section J.1)

- **Combined catalog** (`dataset_v2/trajectory_catalog.py`): deterministic 210-row union of the
  core (120) and challenge (90) per-family manifests ->
  `trajectories/combined_trajectory_manifest.csv`, dataset-root-relative paths, hashes,
  model/config/seed fingerprints, frozen-revision fields. An independent validator checks the exact
  union (no duplicate/missing, 70/70/70 split, 120 core / 90 challenge, referenced NPZ present,
  paths relative).
- **Public vs protected loaders** (`dataset_v2/trajectory_loading.py`): the public loader strips all
  protected arrays (`q_reference`, `q_source_reference`, reconstruction errors, reachability flags)
  and provably cannot return `q_reference`; the protected loader (generation/validation/diagnostics
  only) exposes `q_reference`/`q_reference_start`. Tests prove the isolation both at the trajectory
  loader level and at the trial NPZ level (public `trials/<split>.npz` carries no protected key; the
  protected evidence lives under `trials/protected/`).
- **q_initial candidate pool** (`dataset_v2/trial_candidates.py`): per trajectory a deterministic
  1,400-candidate mixture (500 interior + 300 near-limit + 300 singularity-biased + 300 stratified)
  drawn ONLY from the operational joint limits, reusing Tier 0's sampling constructions unchanged.
  Seeds derive from `dataset_v2/seeds.py` under the `trials` component tag (60) + trajectory content
  hash (+ frozen family/trial revisions for frozen_test); no global `numpy.random`, no dependence on
  `q_reference`, no DLS.
- **Difficulty** (`configs/trial_config.json` `difficulty` block, `status: locked`): primary metric
  = combined normalized first-target pose error (position/scale + orientation-geodesic/scale, 50/50);
  non-overlapping easy/medium/hard bands with guard gaps. Calibrated on the 70 **development**
  trajectories only (98,000 candidates, seed 42) -- see `docs/V2_TRIAL_DIFFICULTY_CALIBRATION.md`;
  numbers baked into `dataset_v2/config_templates.py` `TRIAL_*` constants exactly as Phase 2.5's
  calibration. Scales `position 0.93567 m` / `orientation 2.30143 rad`; bands `easy<=0.86012`,
  `medium in [0.92454, 1.04624]`, `hard>=1.11268`; minimum inter-level separation `0.06441`.
- **Selection**: one representative per band, the in-band candidate closest to the band median
  (never the most-trivial), tie-broken by lowest index; a trajectory that cannot populate a band
  fails the run.
- **Frozen trial seed revision** = 1 (SEPARATE namespace from frozen_core=4 / frozen_challenge=1),
  locked before frozen generation.

### Files added/modified

- **New**: `dataset_v2/trajectory_catalog.py`, `dataset_v2/trajectory_loading.py`,
  `dataset_v2/trial_candidates.py`, `dataset_v2/trial_calibration.py`,
  `dataset_v2/trial_generation.py`, `dataset_v2/trial_validation.py`;
  `pipelines/run_dataset_v2_combined_catalog.py`,
  `pipelines/run_dataset_v2_trial_calibration.py`, `pipelines/run_dataset_v2_trial_generation.py`;
  `tests/_dataset_v2_trial_helpers.py`, `tests/test_dataset_v2_trajectory_catalog.py`,
  `tests/test_dataset_v2_trial_generation.py`; `docs/V2_TRIAL_DIFFICULTY_CALIBRATION.md`.
- **Modified** (extended, not rewritten): `dataset_v2/config_templates.py` (trial constants +
  `TRIAL_*` difficulty thresholds + `FROZEN_TRIAL_SEED_REVISION`; full `trial_config()`;
  `frozen_trial_seed_revision` in `seed_policy_config`); `dataset_v2/schemas.py` (full
  `trial_schema()`); `dataset_v2/manifest.py` (`apply_trial_generation_status`);
  `specs/DLS_DATASET_V2_SPEC.md` (section J.1).

### Full persistent run (master seed 42, working root, retained)

- Combined catalog: **210 rows** (120 core + 90 challenge), 70/70/70 split, validator `passed=True`,
  84,000 canonical poses.
- Trial calibration (development only): 70 trajectories, 98,000 candidates; every development
  trajectory populates all three bands (min hard-band count 79/1,400).
- Trial generation: **~2.6 min** -> **630 trials**; split **210/210/210**; difficulty
  **easy/medium/hard = 210/210/210**; **70/70/70 per difficulty per split**; family **core 360 /
  random_challenge 270**; exactly 3 per trajectory, one easy/medium/hard each.
- Independent validation (`--validate-only`): `total=630 ... catalog_passed=True passed=True`.
  Recompute disagreements: FK position error and primary metric both `< 1e-6`. Checksum manifest
  mismatches **0**. Trial anti-leakage `pass=true`, 0 collisions.
- Difficulty separation: primary-metric monotonic (`easy<medium<hard`) for **all 210** trajectories;
  observed minimum separation 0.215 (easy->medium) / 0.159 (medium->hard), both above the configured
  0.06441 floor. Secondary single-axis orderings (not required monotonic): position 208/210,
  orientation 184/210.
- Isolation: public `trials/<split>.npz` carries no protected array; every `q_initial` differs from
  the protected `q_reference_start` and from every `q_reference` waypoint; protected reference hashes
  recompute.
- No NPZ/CSV/JSON committed; working root retained for Phase 8.

### Tests

- New: `tests/test_dataset_v2_trajectory_catalog.py` (8) + `tests/test_dataset_v2_trial_generation.py`
  (19) = **27 tests** (combined-catalog union/validation/duplicate-detection; public loader hides
  `q_reference`; protected loader reads it; CWD independence; 3-per-trajectory + split/difficulty/
  family distribution; independent validator pass; split inheritance; q_initial within limits and
  independent of `q_reference`; public NPZ has no protected arrays while evidence does; NPZ
  `allow_pickle=False`; FK metadata recomputation; monotonicity; difficulty-band classification;
  unique ids/hashes; anti-leakage report; schema validation; determinism same-seed + sensitivity to
  seed; overwrite protection; dry-run writes nothing; Dataset v1 untouched; frozen trial revision
  locked; validator detects an out-of-limit `q_initial`; calibration development-only + deterministic).
- Targeted: `pytest tests/test_dataset_v2_trajectory_catalog.py tests/test_dataset_v2_trial_generation.py -q`
  -> **27 passed** (~20 s). Scaffold/root-resolution regression
  (`tests/test_dataset_v2_scaffold.py tests/test_dataset_root_resolution.py`) -> **37 passed**.
- Full suite: `pytest -q` -> **748 passed** (721 baseline + 27 new), 0 failed/skipped/errored.

### Confirmations

- Dataset v1: **not modified**. `git status --short` restricted to `VERSION`,
  `DATASET_MANIFEST.json`, `benchmarks/`, `trajectories/`, `configs/`, `schemas/`, `kinematics/`,
  `algorithms/`, `generators/`, `evaluation/`, `assets/` returns zero files. Only `dataset_v2/`
  (config_templates/manifest/schemas modified; six new modules), `pipelines/` (three new CLIs),
  `tests/` (three new files), `specs/`, `docs/` touched. HEAD unchanged `ca516d69`; nothing
  committed or pushed.
- Nothing locked weakened: 12 anchors, 120 core trajectories, 90 challenge trajectories, seed
  algorithm, `frozen_core_seed_revision=4`, `frozen_challenge_seed_revision=1`, FK/Jacobian/SO(3)/
  DLS/adaptive damping -- all unchanged. `q_reference` was never used as/for `q_initial`; no DLS was
  run; no Tier 1-4 evaluation; no PPO/MPDIK/MAPPO; frozen_test was only integrity/count/
  classification/monotonicity/checksum validated, never used to calibrate or tune.
- The persistent working root is a **development working dataset**, not an official release.

### Phase 7 final status: **COMPLETE**

Persistent working dataset root exists; combined catalog = 210 / 84,000 canonical poses; 630/630
trials with exact split (210/210/210), difficulty (210/210/210) and per-difficulty-per-split
(70/70/70) counts, and family (core 360 / challenge 270); `q_initial` valid and independent of
`q_reference` generation; difficulty classification and primary-metric monotonicity pass;
public/protected isolation passes; combined-catalog/schema/checksum/manifest pass; anti-leakage
passes; frozen trial revision = 1; all tests pass; Dataset v1 unchanged; nothing committed or
pushed.

### Recommended Phase 8

With the 210-trajectory / 630-trial working dataset locked and validated, Phase 8 can lock the
Dataset v2 **evaluation config** (acceptance thresholds, currently `configs/evaluation_defaults.json`
= `not_yet_defined`) and run the DLS baseline evaluation over Tier 1-4 -- designing and locking that
config against `development`/`validation` only, then running `frozen_test` exactly once after the
config is recorded/checksummed (spec section K frozen-test protocol). No generator, threshold, or
seed policy from Phases 0-7 should change.
