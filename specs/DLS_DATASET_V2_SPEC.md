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
  `docs/V2_THRESHOLD_CALIBRATION.md`, `configs/difficulty_thresholds.json`).
  > **Amended by Phase 5.2 (section H.2):** the per-class criteria below are now additionally
  > required to be **mutually exclusive** — `near_limit` also requires `sigma_min > 0.09`, and
  > `near_singular` also requires `normalized_joint_limit_margin > 0.024991237796029034`. The
  > threshold *values* here are unchanged; only the anchor-selection eligibility is tightened so no
  > anchor can be both near-limit and near-singular. See section H.2 for the rationale and the
  > machine-readable predicates.
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
- **[PROVISIONAL] (Phase 5 implementation)** Source resolution / canonical resampling / reachability
  policy, filling gaps this section left open without changing the locked 120/400/48,000 counts:
  - Source resolution: `configs/trajectory_config.json:source_waypoint_count_nominal` (2001,
    provisional, must stay `> 400`), evaluated at fine `tau`-uniform time steps through the same
    quintic time scaling as the canonical grid.
  - Canonical resampling: arc-length-uniform (not time-uniform) selection of 400 points from the
    source path; position linearly interpolated between the two bracketing source samples;
    orientation resampled via SO(3) geodesic SLERP between the same two bracketing source
    quaternions (`kinematics/rotation_utils.py::so3_log`/`so3_exp`), never linear-interpolate-
    then-normalize; endpoints preserved exactly (arc length 0 and total map to the source's first
    and last samples respectively).
  - `free_form` shape: a deterministic seeded cubic spline (`scipy.interpolate.CubicSpline`,
    chord-length-parameterized) through 5 control points (the anchor position, 3 seeded interior
    offsets, and a seeded endpoint) — open path, control points stored alongside the source
    representation for reproducibility.
  - Reachability validation scope: superseded by section H.1 below (Phase 5.1 validates *both* the
    canonical and the source path).

### H.1 Strict generation reachability, geometry search, and scale gate [LOCKED by Phase 5.1]

**Motivation.** Phase 5 accepted a waypoint as "reachable" whenever
`kinematics/dls_solver.py::solve_dls_until_converged`, driven by Dataset v1's
`configs/dls_config.json`, returned `success=True` — i.e. within that DLS baseline's *own*
evaluation thresholds (0.006 m / 10.0 deg), with no independent check. A dataset whose
reachability is defined by the very solver it will later benchmark cannot measure that solver, and
the reported "max position error = 0.006 m" was an artifact of that threshold rather than a real
kinematic limit. Phase 5.1 removes the circularity.

- **[LOCKED] Independent-FK acceptance.** A target pose counts as reachable only if a reference
  configuration `q_reference` exists such that (a) `q_reference` lies within operational joint
  limits and (b) an *independently recomputed* `FK(q_reference)` reproduces the target position and
  orientation within Dataset v2's own tolerances. The numerical IK engine's `success` flag is never
  sufficient evidence.
- **[LOCKED] Generation tolerances** (`configs/generation_reachability_config.json`, never read
  from `configs/dls_config.json`): position `<= 1e-4 m`, orientation `<= 0.01 deg`. These are 60x
  and 1000x tighter than the DLS baseline's success thresholds respectively. Changing the DLS
  baseline's thresholds must not change which trajectories the generator accepts.
- **[LOCKED] Separation from evaluation.** The DLS implementation may be reused as a generation
  *numerical IK engine* (with Dataset v2's own solver settings, refinement rounds, and
  deterministic restarts), but nothing it produces at generation time is a DLS baseline evaluation
  result, and trajectories are never selected or rejected using DLS iteration count, runtime, or
  coarse-evaluation success.
- **[LOCKED] Both paths validated.** Every canonical waypoint *and* every high-resolution source
  waypoint carries a `q_reference`, a reachability status, and position/orientation reconstruction
  errors. No waypoint is ever skipped, dropped, or silently replaced.
- **[LOCKED] Geometry-alternative search precedes scale reduction.** For each
  `(anchor, shape, orientation_mode)`, a deterministic set of geometry alternatives
  (`configs/trajectory_config.json:geometry_alternatives`) is searched before any scale reduction.
  The largest strictly-reachable scale wins. Alternatives attempted and rejection reasons are
  recorded per trajectory. The search never consults evaluation or frozen-test results.
  *(Alternative set and selection/tie-break rule superseded by section H.4.)*

### H.4 Generic geometry-alternative set and selection policy [LOCKED by Phase 5.3]

**Motivation.** Phase 5.1 gave `line` six signed directions and `helix` six axis/travel-sign
combinations, but left `circle`/`figure8` with only a plane basis (one traversal sense, one start
phase, fixed lobe signs, no major/minor axis swap) and `free_form` with four templates from a
single generation rule. Those three shapes therefore had no escape route when an anchor's limiting
joint blocked their one enumerated traversal — and Phase 5.2's *development/validation* failures
were exactly `figure8` and `free_form`, with none on `line`/`helix`. Phase 5.3 completes the
geometric symmetry generically; it is applied identically to every split and changes no threshold,
gate, tolerance, count or nominal geometry.

- **[LOCKED] Alternative set** (22 → 56 total):
  - `line`: 6 — signed local axes (unchanged).
  - `circle`: **12** = 3 plane bases × 2 traversal directions (`ccw`/`cw`) × 2 start phases
    (`0`, `pi`). The centre is `p0 - r*(cos(phi)*u + sigma*sin(phi)*v)` so the anchor pose is
    exactly the first waypoint for every combination.
  - `figure8`: **24** = 3 plane bases × 4 amplitude-sign pairs × 2 major/minor axis swaps. The four
    sign pairs cover both lobe handedness *and* traversal reversal, because `s -> 1-s` maps exactly
    to `(-sa, -sb)`; reversal is therefore inside the set and is not enumerated twice. `axis_swap`
    exchanges which basis axis carries the major amplitude — the nominal amplitudes 0.05/0.03 m are
    unchanged, only their assignment.
  - `helix`: 6 — axis bases × travel-direction signs (unchanged).
  - `free_form`: **8** locked templates spanning six departure directions (the anchor's signed
    local axes, giving horizontal/vertical/mixed workspace variants) plus two mirrored variants
    whose lateral offsets are reflected. Every template is a smooth cubic spline through the
    configured control points, starts exactly at the anchor pose, and stays inside the same nominal
    envelope and scale policy.
  - No two alternatives of a shape produce byte-identical geometry (asserted by test).
- **[LOCKED] Canonical alternative IDs.** Every alternative carries a stable, lexically-orderable
  `alternative_id` plus structured metadata (`plane_basis_id`, `traversal_direction`, `handedness`,
  `phase_id`, `axis_swap`, `template_id`, `departure_axis`, `mirror`), all recorded in the manifest
  and the geometry-search report — never only as summary text.
- **[LOCKED] Orientation consistency.** For `circle` and `figure8` the variable-orientation
  rotation vector is multiplied by the alternative's traversal/handedness sign, so the orientation
  profile follows the path's traversal sense.
- **[LOCKED] Selection and tie-break** (`configs/trajectory_config.json:
  alternative_selection_policy`): for each alternative, walk the scale schedule from 1.0 downward
  (never below the 0.50 gate) and record the largest scale at which it passes strict reachability;
  then select the alternative with the **largest accepted scale**. The search never stops at the
  first alternative that clears the gate — an alternative is abandoned early only when its
  remaining schedule can no longer beat the best scale found so far, which cannot change the
  winner. Ties are broken, in order, by: smallest strict position reconstruction error; fewest
  refinement/restart attempts (diagnostic only); `alternative_id` lexical order. DLS baseline
  success, solver iteration count, solver runtime and frozen-test results are all forbidden as
  selection signals.
- **[LOCKED by Phase 5.2] Minimum-scale gate.**
  `configs/trajectory_config.json:minimum_scale_gate` locks
  `minimum_core_accepted_scale = 0.50`, `minimum_scale_status: "locked"`, `enforced: true`,
  rationale *"Preserve at least half of nominal core trajectory geometry"*, with diagnostic bands
  `[1.0, 0.75, 0.5, 0.25]`. Every accepted core trajectory must satisfy `accepted_scale >= 0.50`.
  This is a **hard failure**, never a warning: the scale ladder in the geometry search is floored
  at the gate, so a trajectory that cannot reach 0.50 fails generation outright rather than being
  written below it, and the validator independently fails any on-disk trajectory below it. The
  gate is never relaxed to make a run pass, no trajectory is ever skipped to satisfy it, and the
  locked counts (12 anchors, 120 trajectories, 48,000 canonical poses) never change. Nominal
  geometry is unchanged (line 0.12 m, circle r = 0.045 m, figure-8 0.05/0.03 m, helix r/h =
  0.04/0.08 m) — the gate constrains the scale *factor* applied to that geometry, not the geometry.

### H.2 Anchor class isolation [LOCKED by Phase 5.2]

**Motivation.** Phase 5.1's anchor acceptance predicates allowed an anchor to satisfy more than one
class's raw criteria, with a "prefer clean, fall back to overlapping" preference. That admitted
`anchor_near_limit_02`, which was simultaneously near a joint limit (margin 0.00824) *and*
near-singular (`sigma_min` 0.06254). Its two **closed** shapes could only be certified at scale
0.1209 (circle) and 0.1969 (figure-8) — a ~5 mm circle — because a closed planar loop cannot escape
a doubly-constrained neighbourhood the way an open path can. The confound, not the geometry, was
the defect.

- **[LOCKED] Anchor classes are mutually exclusive by construction.** The anchor-selection
  eligibility predicates (`configs/anchor_config.json:class_eligibility_predicates`,
  `anchor_class_isolation_status: "locked"`) are:
  - `regular`: `sigma_min > 0.09` **and** `normalized_joint_limit_margin > 0.024991237796029034`.
  - `near_limit`: `normalized_joint_limit_margin <= 0.024991237796029034` **and**
    `sigma_min > 0.09` **and** `is_near_singular == false`.
  - `near_singular`: `sigma_min <= 0.03` **and**
    `normalized_joint_limit_margin > 0.024991237796029034` **and** `is_near_limit == false`.
  The `0.09` floor reuses the already-calibrated `moderately_conditioned` upper bound rather than
  introducing a new number.
- **[LOCKED] Scope.** These predicates govern **anchor selection only**. The global difficulty-group
  definitions in `configs/difficulty_thresholds.json`, the Point-IK difficulty thresholds, and the
  Point-IK classification priority (`near_singularity` > `near_joint_limit` >
  `large_orientation_change` > `far_target` > `medium_target` > `near_target`) are all unchanged.
  The four diagnostic flags (`is_near_limit`/`is_near_singular`/`is_moderately_conditioned`/
  `is_regular`) continue to be computed independently from the global definitions for every
  candidate and are stored regardless of the selected class.
- **[LOCKED] No fallback.** There is no overlap fallback. A class with fewer eligible candidates
  than its quota is a hard failure reporting the candidate-availability breakdown; thresholds are
  never relaxed and anchors are never duplicated.

### H.3 Frozen-test seed reset [LOCKED by Phase 5.2]

Phase 5 and Phase 5.1 generated `frozen_test` trajectories into temporary roots that were inspected
while generator policy was still being designed (anchor predicate, geometry-alternative search,
strict reachability tolerance, minimum-scale gate were all chosen with that data visible). Under
the frozen-test protocol in section K those trajectories are burned.

- `configs/seed_policy.json:frozen_core_seed_revision = 3` (Phase 5.3). Revision 1 is recorded
  `burned_not_shippable` (observed during Phase 5/5.1 policy design); revision 2 is recorded
  `burned_not_shippable` with reason *"geometry alternative policy expanded after pre-freeze
  validation"*; revision 3 is `active`. Regenerated `frozen_test` data may only be
  integrity/strict-reachability/checksum/count validated — it is never used to tune the alternative
  list, thresholds, scale gate, anchors or tolerances.
- `frozen_test` anchor split assignment and `frozen_test` core-trajectory path seeds and free-form
  templates mix in the revision; `development`/`validation` keep their existing seed namespace.
  Because anchor split membership is drawn from a single per-class permutation with exact 2/1/1
  quotas, frozen membership is the complement of development+validation, so bumping the revision
  necessarily re-permutes all three splits — with exact quotas there is no way to redraw frozen
  membership while holding the others fixed. This is acceptable here because Phase 5.2 regenerates
  every anchor and every trajectory anyway.
- Regenerated `frozen_test` data may only be integrity-validated, reachability-validated,
  checksummed and count-verified. It must never be used to tune the anchor predicate, geometry
  alternatives, minimum scale, or reachability tolerance.

### H.5 Deterministic seed derivation and feasibility-aware anchor selection [LOCKED by Phase 5.4]

**Deterministic seed derivation.** Dataset v2 derives every seed through
`dataset_v2/seeds.py::derive_seed` (algorithm id `dataset_v2/seed/sha256/v1`): SHA-256 over a
canonical byte encoding of `(base_seed, *tags)`, reduced with pure Python integer arithmetic. No
NumPy type participates, so a derived seed is identical across NumPy versions, Python versions and
platforms. This replaces Dataset v1's `generators/_common.py::derive_seed`, whose closing
`np.uint64 % int` promotes to `float64` on NumPy 1.x (lossy above 2**53) and stays exact on NumPy
2.x, making it NumPy-version-dependent. Dataset v1's function is left untouched (v1 is an immutable
baseline). The dataset root records `seed_algorithm_id`, and the frozen-test revision advanced to
**4** because revision 3 was generated before this fix (revisions 1-3 are all
`burned_not_shippable`; exactly one revision is `active`).

**Feasibility-aware anchor selection.** An anchor's class predicate does not guarantee its
neighbourhood can support the locked core geometry; different seed realizations produced 0, 2 and 4
below-gate core trajectories purely from which anchors were drawn. A [LOCKED] feasibility screen
(`configs/anchor_config.json:feasibility_screening`, status `locked`) makes 120/120 a property of
the policy:

- **[LOCKED] All-ten requirement.** A candidate may enter the catalog only if **all ten** locked
  `(shape, orientation_mode)` combinations reach `accepted_scale >= minimum_core_accepted_scale`
  (0.50). Partial feasibility is never accepted.
- **[LOCKED] Two-stage selection.** Stage A screens eligible candidates in deterministic
  greedy-farthest-point order and keeps only those passing 10/10; Stage B runs the existing
  diversity selection over the feasible subset only. A rejected candidate is replaced from the
  **same** eligible class pool -- classes never mix, no combination is skipped, no random retry.
  The catalog stays 6 regular / 3 near_limit / 3 near_singular, 2/1/1 per split.
- **[PROVISIONAL] Staged probe.** The screen runs at a coarse canonical resolution and tests the
  gate rung (smallest scheduled scale `>= 0.50`) -- the cheapest place to confirm feasibility,
  and sound because the generator's later scale-maximization can only exceed a gate-rung success.
  A coarse pass is never an acceptance: the selected anchor's trajectories are still generated and
  validated at full 400-waypoint / full-source resolution, and the independent core-trajectory
  validator re-verifies the final 120 without trusting the screen.
- **[LOCKED] Evidence and validation.** Each anchor stores per-anchor proof that all ten
  combinations passed at a verified scale `>= 0.50`; the anchor validator fails on missing
  evidence, fewer than ten combinations, a below-gate verified scale, or a geometry/reachability
  config-fingerprint mismatch. The feasibility cache key is content-only (q, model fingerprint,
  geometry/reachability config fingerprints, seed algorithm id, probe resolution) -- never a path
  or timestamp -- and is never written into the dataset root.

Nothing else changes: the class predicates, near-limit/near-singular/regular thresholds, the 0.50
gate, the 1e-4 m / 0.01 deg reachability tolerances, the nominal geometry and all locked counts are
untouched. Feasibility screening only *removes* candidates that would have produced a below-gate
trajectory.

## I. Random challenge policy [LOCKED counts, PROVISIONAL generation policy]

- 90 total, split 30/30/30 (section B).
- **[LOCKED]** Every control pose must be reachable (verified by sequential-DLS reachability
  validation before acceptance — reuse
  `generators/_trajectory_common.py::validate_sequential_reachability`/
  `generate_validated_geometry` unchanged).
- **[LOCKED]** Random path seeds are split-isolated (a `development` challenge trajectory's path
  seed must never equal, or be derivable in a way that collides with, a `validation`/
  `frozen_test` one — see section K).
- Source/canonical representation: identical dual-representation requirement as core trajectories
  (section H).

### I.1 Locked generation policy [LOCKED by Phase 6]

Phase 0 left the challenge *generation* policy `[PROVISIONAL]` (only the 90/30-30-30/400 counts were
locked). Phase 6 locks it. The initial `[PROVISIONAL]` note proposed interpolating between randomly
drawn Cartesian control poses; Phase 6 instead adopts a **smooth joint-space reference family
through FK**, which is strictly stronger on the one property the counts depend on — reachability —
and is fully machine-readable in `configs/random_challenge_config.json`
(`status: counts_and_policy_locked_generation_implemented`).

- **[LOCKED] Generation mechanism.** Unlike core trajectories (Cartesian closed-form shapes anchored
  at the 12 locked anchors), each challenge trajectory is generated from an **independent reachable
  start state** (never one of the 12 anchors — avoids leaking anchor identity and yields 90 diverse
  starts) followed by a bounded-Fourier joint-space curve
  `q(s) = q_start + offset(s)`, `offset_j(s) = scale_j·weight_j·Σ_k[a_kj·sin(π·m_k·s) +
  b_kj·(1−cos(π·m_k·s))]`, with `offset_j(0)=0` (so `q(0)=q_start` exactly) and
  `scale_j = min(1, envelope_margin_fraction · margin_j / max_s|weight_j·raw_j(s)|)`. Because each
  joint's amplitude is capped at a fraction of that joint's own start joint-limit margin, the whole
  reference stays inside operational limits with no clipping (C-∞ smooth, bounded curvature, finite
  velocity/acceleration), and every source pose = FK(q(s)) is reachable by construction. This is a
  "known-reachable joint-space reference family converted through FK": no per-waypoint white noise.
  Orientation is the genuine FK orientation of the joint curve (coupled position+orientation
  variation).
- **[LOCKED] Reachability.** The Phase 5.1/5.4 strict, DLS-baseline-independent engine
  (`configs/generation_reachability_config.json`, 1e-4 m / 0.01 deg) verifies **every** source and
  canonical waypoint by independent FK reconstruction; the numerical IK engine's `success` flag is
  never sufficient and no waypoint is ever skipped. (The Phase-0 pointer to v1's
  `validate_sequential_reachability` is superseded by this stricter, independent check — the earlier
  bullet above is kept for historical continuity but the strict engine governs.)
- **[LOCKED] Six challenge families**, 5 per split × 3 splits = 15 each (6 × 15 = 90):
  `smooth_random`, `mixed_curvature`, `non_planar`, `large_orientation`, `near_limit_region`,
  `near_singular_region` — targeting the coverage aspects approved for going beyond the 120 core
  paths (randomized smooth geometry, mixed curvature, non-planar paths, stronger orientation
  variation, near-limit and near-singular neighbourhoods). Each family has a machine-readable region,
  harmonic set, per-joint amplitude weights, envelope fraction, curvature ceiling, coverage floor
  (where relevant), seed tag, and per-split quota. `near_limit_region`/`near_singular_region` start
  states additionally satisfy the Phase 2.5 locked `near_joint_limit`/`near_singularity` thresholds.
- **[LOCKED] Acceptance policy.** No core-style Cartesian scale gate applies (challenge paths are not
  scaled Cartesian shapes). Acceptance = start-state validity (+ family start predicate where
  declared) + within-limits bounded envelope + strict independent-FK reachability on all source and
  canonical waypoints + finite bounded curvature + the family coverage floors
  (`non_planar` min non-planarity, `large_orientation` min angular displacement).
- **[LOCKED] Feasibility-aware diversity selection.** Per (family, split): a seeded candidate pool
  (16) is coarse-probe screened for strict reachability + coverage floors; the feasible subset is
  diversity-selected (greedy farthest-point over joint-space/workspace/arc-length/angular/curvature/
  non-planarity/σ_min/margin features, reusing `greedy_farthest_point_select`) down to the quota;
  the selected candidates are re-validated at full 400/source resolution; a full-validation failure
  is replaced deterministically from the same pool — never by loosening reachability, counts, or the
  family policy.
- **[LOCKED] Frozen-challenge seed revision** (`configs/seed_policy.json:
  frozen_challenge_seed_revision`, initial value **1**) — a **separate** namespace from
  `frozen_core_seed_revision` (which stays **4**, unchanged). `frozen_test` challenge path seeds and
  coefficient draws mix it in; `development`/`validation` use the unrevised namespace. Locked before
  any frozen challenge generation; never rerolled to find an easier frozen set.
- **[LOCKED] Source/canonical dual representation** identical to core: source resolution
  `source_waypoint_count_nominal = 1201` (> 400), canonical = 400 waypoints arc-length-resampled from
  the source via the same `resample_canonical` (piecewise-linear position, SO(3) geodesic SLERP
  orientation, exact endpoints, sign-continuous quaternions). Challenge paths are open (no closure
  requirement).

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
