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
