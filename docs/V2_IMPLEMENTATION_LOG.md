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
