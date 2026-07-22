# Dataset v2 Specification — Kassow KR810, Tier 0-4 Kinematics

Status: **Phase 0 (planning/spec only). Nothing in this document has been implemented.**
This spec distinguishes three kinds of statement throughout:

- **[LOCKED]** — fixed by the user's design; must not be changed by any future implementation
  phase without an explicit new decision from the user.
- **[PROVISIONAL]** — an implementation decision proposed here to fill a gap the locked design
  leaves open; may be revised during Phase 1+ without needing to reopen the locked counts/rules.
- **[BLOCKER]** — unresolved; must be decided before the component it touches can be implemented.

## A. Scope and non-scope

**[LOCKED] In scope**: Dataset v2, Tier 0-4 kinematics only, for the Kassow KR810 (7-DOF),
covering FK/Jacobian/singularity validation states (Tier 0), Point-IK samples (Tier 1), core and
random-challenge trajectories with trials (Tier 2-4 inputs).

**[LOCKED] Out of scope** (this dataset, any phase):
- PPO, MPDIK, MAPPO or any learning-based control.
- Dynamics, actuators, controllers, torque control, collision.
- Real-robot / calibrated-TCP calibration, ISO 9283 certification claims.
- Modifying Dataset v1 content or checksums.
- Modifying FK, Jacobian, pose-error, or adaptive-damping formulas (`kinematics/`) or the DLS
  algorithms (`algorithms/`) — Dataset v2 is a data-generation effort against the *existing*
  kinematics/DLS implementation, not a change to it.

## B. Counts (all [LOCKED])

**Tier 0**
- 1,000 FK states
- 1,000 Jacobian states
- 600 singularity states

**Tier 1 — Point-IK**
- 6,000 total samples = 6 difficulty groups × 1,000 samples each:
  `near_target`, `medium_target`, `far_target`, `large_orientation_change`,
  `near_joint_limit`, `near_singularity`.
- Split: development 1,200 / validation 1,200 / frozen_test 3,600.

**Core trajectories**
- 5 shapes: `line`, `circle`, `figure8`, `helix`, `free_form`.
- 2 orientation modes: `fixed`, `variable`.
- 12 anchors: 6 `regular` + 3 `near_limit` + 3 `near_singular`.
- Total: 5 × 2 × 12 = **120 core trajectories**.

**Random challenge trajectories**
- 90 total = development 30 + validation 30 + frozen_test 30.

**Overall**
- 120 + 90 = **210 trajectories total**.
- Each trajectory: exactly 400 canonical waypoints → 210 × 400 = **84,000 canonical target
  poses**.
- Each trajectory: 3 initial states (`easy`, `medium`, `hard`) → 210 × 3 = **630 trials**.

## C. Dataset-root layout (proposed) [PROVISIONAL]

Rooted at an explicit `dataset_v2_root` (never the repo root, never CWD-relative):

```
<dataset_v2_root>/
  VERSION
  DATASET_MANIFEST.json
  configs/
    tier0_config.json
    point_ik_config.json
    anchor_config.json
    trajectory_config.json
    trial_config.json
    seed_policy.json
  schemas/
    fk_state_schema.json
    jacobian_state_schema.json
    singularity_state_schema.json
    point_ik_schema.json
    anchor_schema.json
    trajectory_schema.json
    trial_schema.json
  benchmarks/
    tier0/
      fk_states.npz
      jacobian_states.npz
      singularity_states.npz
    point_ik/
      development.npz
      validation.npz
      frozen_test.npz
      point_ik_manifest.csv
      difficulty_definition.json
  anchors/
    anchor_manifest.csv
    anchors.npz
  trajectories/
    core/
      <trajectory_id>.npz            # canonical, 400 waypoints
      <trajectory_id>_source.npz     # high-resolution source
    random_challenge/
      <trajectory_id>.npz
      <trajectory_id>_source.npz
    trajectory_manifest.csv
  trials/
    trial_manifest.csv
  checksums/
    CHECKSUM_MANIFEST.json           # every generated file's SHA256, dataset-v2-scoped
  reports/
    GENERATION_REPORT.json           # generation-time report, never evaluation results
```

Rules tied to this layout **[LOCKED]**:
- No evaluation results are written under `<dataset_v2_root>` (evaluation output goes elsewhere,
  outside the source dataset tree — mirrors `pipelines/_common.py::ensure_output_structure`
  writing to a caller-supplied `output_dir`, never into `benchmarks/`/`trajectories/`).
- No absolute paths anywhere in configs/manifests/generated JSON — every stored path is relative
  to `dataset_v2_root` (mirrors `generators/_common.py::relative_to_repo`, generalized to
  relative-to-dataset-root).
- No dependence on current working directory — every path resolved from an explicit root object,
  never `Path(".")` or CWD-implicit joins (mirrors `utils/dataset_locator.py`'s
  `Path(__file__)`-anchoring pattern, generalized to take the root as a parameter/config instead
  of hardcoding it to the repo).

## D. Naming conventions [PROVISIONAL unless noted]

- **`sample_id`** (Point-IK): `pik_{split}_{difficulty_group}_{index:05d}`, e.g.
  `pik_development_near_target_00042`. Globally unique across all splits and groups.
- **`anchor_id`**: `anchor_{class}_{index:02d}`, e.g. `anchor_regular_03`, `anchor_near_limit_01`,
  `anchor_near_singular_02`. 12 total, globally unique.
- **`trajectory_id`**:
  - Core: `core_{shape}_{orientation_mode}_{anchor_id}`, e.g.
    `core_circle_variable_anchor_regular_03`.
  - Random challenge: `challenge_{split}_{index:03d}`, e.g. `challenge_frozen_test_017`.
  - Globally unique across core + random challenge.
- **`trial_id`**: `{trajectory_id}_trial_{init_class}`, e.g.
  `core_line_fixed_anchor_regular_00_trial_easy`. Exactly 3 per trajectory (`easy`, `medium`,
  `hard`).
- **`split`** **[LOCKED vocabulary]**: `development` | `validation` | `frozen_test`.
- **trajectory family** **[LOCKED vocabulary]**: `line` | `circle` | `figure8` | `helix` |
  `free_form` (core) or `random_challenge`.
- **orientation mode** **[LOCKED vocabulary]**: `fixed` | `variable`.
- **difficulty group** **[LOCKED vocabulary]**: `near_target` | `medium_target` | `far_target` |
  `large_orientation_change` | `near_joint_limit` | `near_singularity`.
- **initialization class** **[LOCKED vocabulary]**: `easy` | `medium` | `hard`.

## E. Seed policy [LOCKED principles, PROVISIONAL derivation scheme]

- **[LOCKED]** A single master seed anchors the entire v2 generation run.
- **[LOCKED]** No global random state (`numpy.random.seed`/module-level `RandomState`) is ever
  used; every random draw comes from an explicitly derived `np.random.Generator`.
- **[LOCKED]** Regeneration from the same master seed must reproduce byte-identical NPZ content.
- **[PROVISIONAL]** Reuse `generators/_common.py::derive_seed`'s scheme
  (`np.random.SeedSequence([master_seed, *tags])`), with a fixed tag hierarchy:
  `derive_seed(master_seed, component_tag)` for a component seed (e.g. Tier 0, Point-IK, anchors,
  core trajectories, random challenge, trials), then
  `derive_seed(component_seed, split_tag)` for a split seed, then further tags for per-item draws
  (per-sample, per-anchor, per-trajectory, per-trial). Every seed used anywhere in v2 generation
  must be traceable back to `(master_seed, tag_path)` and recorded in
  `reports/GENERATION_REPORT.json`.
- **[LOCKED]** Component/split/item seeds must never overlap in a way that could leak generation
  entropy across `development`/`validation`/`frozen_test` (see section K).

## F. Point-IK generation policy [LOCKED]

- Every target is produced by drawing a valid `q_target` and computing its pose via forward
  kinematics — never a freely chosen Cartesian point assumed reachable (same principle already
  implemented in `generators/generate_point_ik_dataset.py`).
- `q_target` is stored as a **reference/provenance value only**; it must never be used as the
  initial guess for any IK solve that consumes this sample (an evaluator that used `q_target` as
  `q_initial` would trivially "solve" the sample with zero iterations — this must be structurally
  prevented, e.g. by not shipping `q_target` next to a `q_initial` in the same evaluation-facing
  record, or by explicit evaluator-side validation).
- Each sample stores its difficulty covariates (position distance, orientation distance, joint
  distance, `sigma_min` at initial/target, joint-limit margin at initial/target) computed from
  real FK/Jacobian evaluations, not assumed — same principle as v1's
  `_compute_pair_metrics`/`_derive_thresholds`/`_classify_pool`.
- Duplicate protection: no two samples (within or across splits) may share the same
  `(q_initial, q_target)` pair or the same computed content hash (see section N); generation must
  reject/redraw on collision rather than silently keeping duplicates.
- **[LOCKED] Difficulty-group thresholds and priority** (Phase 2.5 calibration —
  `docs/V2_THRESHOLD_CALIBRATION.md`, `configs/difficulty_thresholds.json`): `near_joint_limit` and
  `near_singularity` reuse the same normalized-margin (`0.024991237796029034`) and `sigma_min`
  (`0.03`) thresholds as anchors (section G) and trial covariates, so all three consumers agree on
  what "near the limit"/"near-singular" means. `near_target`/`medium_target`/`far_target`/
  `large_orientation_change` quantile thresholds are **not** calibrated by this phase (out of
  scope — Phase 2.5 covers only the four groups named in its charter) and remain to be re-derived,
  analogous to v1's `_derive_thresholds`, when Point-IK generation is implemented. When a candidate
  pair qualifies for more than one group, priority (highest first) is: `near_singularity` >
  `near_joint_limit` > `large_orientation_change` > `far_target` > `medium_target` >
  `near_target` — unchanged from v1's `generate_point_ik_dataset.py::PRIORITY_ORDER`.

## G. Anchor policy [LOCKED counts/classes/acceptance-threshold values, PROVISIONAL selection procedure]

- **[LOCKED]** 12 anchors total: 6 `regular`, 3 `near_limit`, 3 `near_singular`.
- **[LOCKED]** Split isolation: an anchor's `anchor_id` is fixed to exactly the trajectories that
  use it; anchors themselves are not "split" the way samples/trajectories are, but no
  `anchor_id` may be reused across a `development`/`validation`/`frozen_test` trajectory pair in
  a way that leaks anchor identity as a shortcut signal (i.e. document, per anchor, every
  trajectory/split that consumes it, and keep that mapping visible in `anchor_manifest.csv`).
- **[LOCKED] Acceptance criteria** (numeric thresholds locked by Phase 2.5 calibration —
  `docs/V2_THRESHOLD_CALIBRATION.md`, `configs/difficulty_thresholds.json`):
  - `regular`: `sigma_min` comfortably above the singularity threshold *and* joint-limit margin
    comfortably interior — this reuses v1's `select_anchor` predicate as-is. Calibrated as
    `sigma_min > 0.09` (`moderately_conditioned_upper_bound`) **and**
    `normalized_joint_limit_margin > 0.024991237796029034` (`near_joint_limit` threshold's
    complement).
  - `near_limit`: `normalized_joint_limit_margin <= 0.024991237796029034` (the calibration pool's
    P10 quantile of the per-configuration minimum normalized joint-limit margin — normalized, not
    absolute-rad, because KR810's joint_2/joint_4 operational half-range, ~2.18 rad, is ~2.9x
    smaller than the other five joints' ~6.28 rad; an absolute-rad threshold was empirically found
    to make joint_2/joint_4 the near-limit "controlling joint" ~6x more often than the other five —
    see the calibration doc) while remaining a valid, reachable configuration (never an actual
    limit violation).
  - `near_singular`: `sigma_min <= 0.03`, reused unchanged from v1's shared
    `configs/dls_config.json:singularity_sigma_threshold` (already reused by
    `generators/_trajectory_common.py::select_anchor`'s `ANCHOR_SIGMA_RATIO=3.0` predicate and by
    Dataset v2 Tier 0's own singularity-state classifier — audited in the calibration doc and found
    consistent with all three), while remaining numerically solvable by the existing DLS solver
    (adaptive damping must keep the anchor's own FK/reachability check well-defined).
  - **[PROVISIONAL]**: the concrete *search/selection procedure* that picks 3 diverse `near_limit`
    and 3 diverse `near_singular` anchors satisfying these locked thresholds (workspace + joint-
    space diversity, analogous to `select_anchor`'s search loop) is not yet implemented — only the
    acceptance thresholds are locked here. Anchor generation itself remains a later phase.

## H. Core trajectory policy [LOCKED]

- 5 shapes × 2 orientation modes × 12 anchors = 120 trajectories (section B).
- Every trajectory has both a **high-resolution source representation** (finer time/path
  discretization, used to derive arc-length and angular-displacement metadata precisely) and a
  **canonical representation of exactly 400 waypoints** (resampled/subsampled from the source,
  never independently regenerated, so canonical and source stay geometrically consistent).
- Quaternions stored WXYZ (`kinematics/quaternion_utils.py` convention — reused unchanged).
- Orientation mode `variable` uses SO(3)/SLERP interpolation between anchor and end orientation
  (reuse `generators/generate_orientation_profile.py::variable_orientation_profile` unchanged;
  it already implements SLERP-consistent SO(3) interpolation).
- Closure policy: closed-path shapes (e.g. `circle`, `figure8`) return to their starting pose;
  open-path shapes (`line`, `helix`, `free_form` as applicable) do not. This mirrors v1's
  `closed_path` flag and `orientation_phase_for_shape` handling.

## I. Random challenge policy [LOCKED counts, PROVISIONAL generation policy]

- 90 total, split 30/30/30 (section B).
- **[LOCKED]** Every control pose must be reachable (verified by sequential-DLS reachability
  validation before acceptance — reuse
  `generators/_trajectory_common.py::validate_sequential_reachability`/
  `generate_validated_geometry` unchanged).
- **[PROVISIONAL]** Interpolation policy: random challenge trajectories interpolate between
  randomly drawn, individually-validated control poses using the same quintic time-scaling +
  canonical/high-resolution dual representation as core trajectories (section H), so downstream
  consumers see one consistent trajectory schema regardless of family.
- **[LOCKED]** Random path seeds are split-isolated (a `development` challenge trajectory's path
  seed must never equal, or be derivable in a way that collides with, a `validation`/
  `frozen_test` one — see section K).
- Source/canonical representation: identical dual-representation requirement as core trajectories
  (section H).

## J. Trial policy [LOCKED]

- 3 trials per trajectory: `easy`, `medium`, `hard` initial states → 210 × 3 = 630 trials total.
- Each trial records its difficulty covariates (e.g. distance of `q_initial` from the
  trajectory's first canonical waypoint's IK-consistent configuration, joint-limit margin,
  `sigma_min` at `q_initial`) — same evidentiary principle as Point-IK (section F).
- **[LOCKED]** The initial state for a trial must never be derived from a *future* solution along
  that trajectory (no leaking a downstream waypoint's solved configuration backward into
  `q_initial`); `q_initial` must be constructed independently (e.g. drawn and validated against
  only the trajectory's first target pose, analogous to v1's robustness-trial construction in
  `_trajectory_common.py::build_trials`, but without reusing any later-waypoint solution).

## K. Split and anti-leakage policy [LOCKED]

`development`, `validation`, and `frozen_test` must never share, across any pair of splits:
- `anchor_id`
- Point-IK `sample_id`
- Point-IK content hash (a deterministic hash of `(q_initial, q_target)` or equivalent, distinct
  from `sample_id` so a regenerated sample with a new id but identical content is still caught)
- random path seed (for random challenge trajectories)
- `trajectory_id`
- trajectory hash (content hash of the canonical waypoint sequence)
- `trial_id`

Generation must include an explicit post-generation check that asserts pairwise disjointness of
each of the above sets across the three splits, and generation must fail loudly (not silently
drop/dedupe) on any collision.

**Frozen-test protocol [LOCKED]**:
- `frozen_test` must never be used to design a generator, choose a threshold, tune DLS, or tune
  PPO/MPDIK in any later phase.
- `frozen_test` may only be run after the evaluation config for that run has been locked
  (recorded, immutable, checksummed) — i.e. no iterating on evaluation config against
  `frozen_test` results.

## L. Schema policy [LOCKED principle, PROVISIONAL file list]

- Every generated array/record type gets a corresponding JSON Schema under
  `<dataset_v2_root>/schemas/` (Draft 2020-12, `additionalProperties: false`, mirroring
  `schemas/point_ik_schema.json`/`schemas/trajectory_schema.json`'s existing style).
- Schemas must include every field this spec locks: `split`, anchor class/id, trajectory family,
  orientation mode, arc-length metadata, cumulative angular-displacement metadata, source seed,
  checksum.
- **[PROVISIONAL]** File list: `fk_state_schema.json`, `jacobian_state_schema.json`,
  `singularity_state_schema.json`, `point_ik_schema.json`, `anchor_schema.json`,
  `trajectory_schema.json`, `trial_schema.json` (see layout in section C).

## M. Versioning policy [LOCKED]

- Dataset v2 has its own `VERSION` file and its own `DATASET_MANIFEST.json`, both under
  `<dataset_v2_root>`, independent of the repo-root v1 `VERSION`/`DATASET_MANIFEST.json`.
- `DATASET_MANIFEST.json` (v2) must declare: dataset name/version, scope (Tier 0-4 kinematics
  only, explicit `includes_dynamic_control/ppo/mpdik/mappo: false`), all locked counts from
  section B, split sizes, and pointers to the checksum manifest and generation report.

## N. Checksum policy [LOCKED principle, PROVISIONAL mechanics]

- A dataset-v2-scoped `CHECKSUM_MANIFEST.json` records SHA256 for every generated file (reuse
  `utils/file_checksum.py::sha256_file` and the `build_checksum_entry` pattern from
  `generators/_common.py`, generalized to a v2 root).
- **[LOCKED]** In addition to file-level checksums, every Point-IK sample and every trajectory
  needs a **content hash** (not just a file hash) so anti-leakage checks (section K) can compare
  content, not just filenames/ids. **[PROVISIONAL]** mechanics: SHA256 over a canonical
  (sorted-key, fixed-precision) serialization of the record's numeric content, analogous to
  `generators/_common.py::config_hash` but applied to sample/trajectory data instead of configs.
- Dataset v1's checksum files/manifests are never read from nor written to by any v2 checksum
  logic.

## O. Dataset v1 backward compatibility [LOCKED]

- V1's `VERSION`, `DATASET_MANIFEST.json`, `benchmarks/`, `trajectories/`, and their checksum
  files must remain byte-identical throughout all v2 work.
- V2 code must not import, monkeypatch, or otherwise couple to v1-specific constants
  (`generators/_common.py::REPO_ROOT`, `utils/dataset_locator.py::BENCHMARKS_DIR`, etc.) in a way
  that would break if v1's layout changed independently — new v2 modules should take dataset root
  as an explicit parameter instead of importing v1's hardcoded paths.

## P. Frozen-test protocol

See section K ("Frozen-test protocol"); duplicated reference here per the requested document
structure — the rules live in one place (K) and are not repeated with different wording.

## Q. Acceptance criteria per implementation phase [PROVISIONAL]

- **Tier 0 phase**: exact counts (1000/1000/600) generated, deterministic under fixed seed, NPZ
  loads with `allow_pickle=False`, all values finite.
- **Point-IK phase**: 6,000 samples, 1,000/group, split sizes 1,200/1,200/3,600, zero
  cross-split collisions (section K), every `target_position`/`target_quaternion` matches
  FK(`q_target`) to solver tolerance (mirroring v1's
  `test_fk_of_q_target_matches_stored_target_pose`).
- **Anchor phase**: 12 anchors, 6/3/3 by class, each anchor's acceptance criteria (section G)
  numerically verified and recorded.
- **Core trajectory phase**: 120 trajectories, each with canonical 400 waypoints, a
  high-resolution source, arc-length + angular-displacement metadata, 100% sequential-DLS
  reachability at generation time (reusing v1's validation approach).
- **Random challenge phase**: 90 trajectories, split 30/30/30, same reachability/schema
  guarantees as core.
- **Trial phase**: 630 trials, 3 per trajectory, easy/medium/hard covariates statistically
  distinct (e.g. `hard` trials have measurably larger initial pose/joint distance than `easy`).
- **Anti-leakage phase**: automated check passes with zero collisions across all seven leakage
  dimensions in section K.
- **Freeze phase**: `CHECKSUM_MANIFEST.json` and `GENERATION_REPORT.json` written, `VERSION`/
  `DATASET_MANIFEST.json` finalized, all v2 schemas validate 100% of generated records.

## R. Release-candidate and final-freeze policy [PROVISIONAL]

- A release candidate is a fully generated v2 dataset tree (section C) with every acceptance
  criterion in section Q met and a generation report showing the exact master seed and
  derivation tags used.
- Final freeze = the point at which `frozen_test` is first permitted to be evaluated against a
  locked evaluation config (section K); after final freeze, regenerating `frozen_test` content
  requires an explicit, recorded version bump (new `VERSION`), never an in-place overwrite.

## S. Out-of-scope (restated for completeness) [LOCKED]

PPO, MPDIK, MAPPO, dynamics, actuators, controllers, torque control, collision, real-robot
calibration, calibrated TCP, ISO certification claims. (Same as section A; restated per the
requested document structure.)
