# Dataset v2 — Threshold Calibration (Phase 2.5)

Status: **complete**. This phase calibrates quantitative definitions for `near_joint_limit`,
`near_singularity`, `moderately_conditioned`, and `regular`, for shared reuse by Tier 1 Point-IK
difficulty groups, the 12 anchor configurations, and trial difficulty covariates. It generates no
official Dataset v2 samples/anchors/trajectories/trials.

- **Baseline commit**: `77674d023156e05ebe7ce1740fb5dd1317d65df1` (`feature/dataset-v2`, clean
  working tree, `pytest -q` → 391 passed before this phase's changes).
- **Calibration seed**: `42` (explicit; derives `generic_pool_seed=574252242500548608` and
  `singularity_pool_seed=1639952255904426752` via `generators/_common.py::derive_seed` — see
  "Seed derivation" below).
- **Candidate pool sizes**: generic = 50,000; singularity-biased = 20,000 (config-driven, see
  `configs/difficulty_thresholds.json:candidate_pool_size`).
- **Runtime**: ~31s wall-clock for the full 50,000 + 20,000 candidate pass (per-candidate FK +
  Jacobian + SVD via the real MuJoCo model), measured with `time` on the reproducibility command
  below.
- **Implementation**: `dataset_v2/threshold_calibration.py` (module) +
  `pipelines/run_dataset_v2_threshold_calibration.py` (CLI). No candidate pool, per-candidate
  array, or NPZ file from this phase was written to disk or committed — only the small scalar
  summary below and the config constants in `dataset_v2/config_templates.py`.

## 1. Candidate pool construction

Two deterministic pools, both drawn from the real KR810 operational joint limits
(`configs/robot_config.json:operational_lower_rad/operational_upper_rad`), never from DLS,
`q_target`, or frozen-test data:

- **Generic pool** (`build_generic_candidate_pool`): uniform over
  `[lower + margin, upper - margin]` per joint, where `margin = 0.01 * half_range` (1% of that
  joint's own half-range), not one flat rad value. This is the population used to derive the
  `near_joint_limit` quantile threshold and to characterize the natural (unbiased) `sigma_min`
  distribution.
- **Singularity-biased pool** (`build_singularity_biased_pool`): reuses
  `dataset_v2/tier0_generation.py::_build_singularity_candidate_pool` unchanged — uniform-interior
  candidates plus an elbow-biased subset (`q[3]` near 0) and a wrist-biased subset (`q[5]` near 0).
  This is the same construction Tier 0 already generates and validates 600 singularity states
  with; reusing it here means the near-singular sufficiency check rides on a proven code path, and
  bias only *proposes* candidates — every classification below still uses the real computed
  `sigma_min`, never the bias label.

Both pools redraw any exact-duplicate row in place (never silently keep or pad duplicates), stay
strictly within the operational limits by construction, and are built entirely in memory —
nothing is written to disk.

**Why a range-proportional margin for the generic pool.** KR810's operational half-ranges are not
uniform: `joint_2`/`joint_4` have half-range ≈2.18 rad, the other five joints ≈6.28 rad (≈2.9x
larger; see `configs/robot_config.json`). An initial attempt used one flat `interior_margin_rad =
0.10` (matching v1 Point-IK's `INTERIOR_MARGIN_RAD`) for every joint. That construction makes it
*structurally impossible* for `joint_2`/`joint_4` to ever reach a normalized margin below
`0.10/2.18 ≈ 0.046`, while the wide-range joints can reach `0.10/6.28 ≈ 0.016` — so a normalized
threshold anywhere below ~0.046 would silently exclude `joint_2`/`joint_4` from ever being the
near-limit "controlling joint," regardless of the threshold's intent. Confirmed empirically: with
the flat-rad pool, `joint_2`/`joint_4` (indices 1, 3) had **zero** occurrences as the controlling
joint under the derived threshold. Switching to a per-joint margin proportional to that joint's
own half-range (1%) fixed this — see the controlling-joint histogram below. This is exactly the
"don't mix rad units across joints with different ranges without normalizing" pitfall the phase
charter warns about, caught by the calibration itself rather than assumed away.

## 2. Formulas

- **Normalized joint-limit margin** (primary near-limit metric):
  `kinematics/joint_limit_utils.py::normalized_joint_limit_margin` /
  `minimum_joint_limit_margin` — per joint, `margin_i = min(q_i - lower_i, upper_i - q_i) /
  half_range_i`; the configuration's value is `min_i(margin_i)`. Reused unchanged from v1 (v1's
  Point-IK generator already uses this same function).
- **Absolute joint-limit margin** (diagnostic only): `min_i(min(q_i - lower_i, upper_i - q_i))` in
  radians, not divided by range. Computed here (no existing repo function returns the
  unnormalized form) purely to demonstrate the unit-mixing risk above; never used for
  classification.
- **`sigma_min`**: `kinematics/singularity_metrics.py::minimum_singular_value` (smallest singular
  value of the 6×7 geometric Jacobian, `kinematics/jacobian.py::geometric_jacobian_world`).
- **`condition_number`**, **`numerical_rank`**, **`manipulability`** (positional Yoshikawa
  measure): computed per candidate for diagnostic completeness (section 3 of the phase charter)
  via `kinematics/singularity_metrics.py` and `kinematics/manipulability.py::
  positional_manipulability`; not used to define any threshold here (per the charter, near-singular
  must depend on `sigma_min`, not condition number alone).
- **FK position**: `kinematics/forward_kinematics.py::forward_kinematics`, computed per candidate
  for workspace-diversity confirmation.

## 3. Distribution summary (generic pool, n=50,000; singularity-biased pool, n=20,000)

| Quantity | min | P01 | P02 | P05 | P10 | median | max |
|---|---|---|---|---|---|---|---|
| normalized joint-limit margin (generic) | 0.010004 | 0.011400 | 0.012788 | 0.017288 | **0.024991** | 0.103940 | 0.773894 |
| absolute joint-limit margin, rad (generic, diagnostic) | 0.021839 | 0.032406 | 0.043361 | 0.071129 | 0.102381 | 0.417559 | 2.150906 |
| `sigma_min` (generic, unbiased) | 0.000227 | 0.006168 | 0.008928 | 0.014545 | 0.023186 | 0.099218 | 0.277080 |
| `sigma_min` (singularity-biased) | 0.000165 | 0.003854 | 0.005383 | 0.008522 | 0.012333 | 0.059939 | 0.270767 |

**Controlling-joint histogram at the near-limit threshold** (which joint achieves the per-candidate
minimum, restricted to candidates classified `near_joint_limit`):

| Joint index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| normalized (n=5,000 classified) | 680 | 711 | 730 | 727 | 746 | 751 | 655 |
| absolute-rad diagnostic (n=5,000 classified) | 253 | **1816** | 298 | **1747** | 313 | 294 | 279 |

The normalized version is roughly even across all seven joints (≈655–751 each, vs. an even split
of ≈714); the absolute-rad diagnostic is dominated by joints 1 and 3 (`joint_2`, `joint_4`, the
narrow-range joints) at ≈6x the rate of the other five — direct empirical confirmation that an
absolute-rad threshold would over-represent the narrow-range joints, and that the normalized
definition is the more stable, joint-range-independent choice.

**Notable finding**: the all-zero joint configuration (`q = 0`, the arm's naive "home" pose) is an
*exact* singularity for this model (`sigma_min ≈ 9.3e-19`, confirmed by direct computation) —
consistent with Tier 0's `zero_or_home` FK group perturbing away from zero rather than assuming it
regular, and a reminder that "home" must never be assumed well-conditioned for this arm.

## 4. Near-joint-limit definition

Both an absolute (rad) and a normalized threshold were evaluated (phase charter section 4). The
normalized definition is selected as primary: it is stable across joints with different ranges (the
controlling-joint histogram above), while the absolute definition would silently bias which joint
"counts" as near its limit. The absolute value is retained in config as a diagnostic only.

- **Selected**: `normalized_joint_limit_margin <= 0.024991237796029034` (the generic pool's own
  P10 quantile).
- Below this threshold: 5,000 of 50,000 generic-pool candidates (10.0%, by construction of a P10
  quantile) — comfortably covers Point-IK's 1,000-per-group need and anchors' 3-candidate need
  with room for workspace/joint-space diversity selection.
- **Status**: `locked`.

## 5. Near-singularity definition (v1 threshold audit)

v1's `singularity_sigma_threshold = 0.03` (`configs/dls_config.json`) has a clear, already-reused
source: `generators/_trajectory_common.py::select_anchor` already gates its "regular anchor"
search on `sigma_min >= threshold * ANCHOR_SIGMA_RATIO` (`ANCHOR_SIGMA_RATIO = 3.0` → 0.09), and
Dataset v2's own Tier 0 singularity-state generator (`dataset_v2/tier0_generation.py`) already
classifies its 600 states against this exact threshold with the same `3.0` multiplier for
`moderately_conditioned_upper_bound`. This calibration's own generic-pool `sigma_min` distribution
places `0.03` between the pool's P05 (0.014545) and P10 (0.023186)/median (0.099218) — i.e., in
the pool's genuine lower tail, not an arbitrary cut — and the singularity-biased pool shows 6,038
of 20,000 candidates (30.2%) at or below it, so it is neither vacuous nor near-universal.

- **Selected**: reuse v1's `0.03` unchanged. **Status**: `locked`.
- **Source**: `configs/dls_config.json:singularity_sigma_threshold` (v1, unmodified).

## 6. Moderately-conditioned and regular

Non-overlapping, exhaustive tri-state on `sigma_min` (matches Tier 0's own three-group
singularity-state split exactly):

- `near_singular`: `sigma_min <= 0.03`
- `moderately_conditioned`: `0.03 < sigma_min <= 0.09` (`moderately_conditioned_upper_bound =
  0.03 * 3.0`, same `3.0` multiplier as v1's `select_anchor`/Tier 0)
- `regular` (sigma axis): `sigma_min > 0.09`

Independently, on the margin axis: `near_joint_limit` (`margin <= 0.024991...`) vs. not. A
candidate is a **regular anchor** (spec section 6) only if both hold: `sigma_min > 0.09` **and**
`normalized_joint_limit_margin > 0.024991237796029034`. In the generic pool, 24,736 of 50,000
candidates (49.5%) satisfy both simultaneously — far more than the 6 regular anchors needed, with
room for workspace-diversity selection.

## 7. Classification priority

Locked, unchanged from v1's `generators/generate_point_ik_dataset.py::PRIORITY_ORDER` (highest
first): `near_singularity` > `near_joint_limit` > `large_orientation_change` > `far_target` >
`medium_target` > `near_target`. Reused as-is after confirming it is still logically consistent:
the two groups calibrated in this phase (`near_singularity`, `near_joint_limit`) are also the two
rarest/tightest-tailed classes (P10-level quantiles), so claiming their eligible candidates first
before the broader position/orientation-based groups draw from what remains is the same rationale
v1 already applied.

## 8. Expected candidate availability (at locked pool sizes)

| Group / consumer | Count in pool | Need | Margin |
|---|---|---|---|
| Point-IK `near_joint_limit` (generic pool, P10 by construction) | 5,000 | 1,000 | 5x |
| Point-IK `near_singularity` (generic pool, unbiased) | 6,892 | 1,000 | 6.9x |
| Point-IK `near_singularity` (singularity-biased pool) | 6,038 / 20,000 | 1,000 | 6x |
| Anchor `near_limit` (3 needed, diversity required) | 5,000 candidates to select from | 3 | ample |
| Anchor `near_singular` (3 needed, diversity required) | 6,038 candidates to select from | 3 | ample |
| Anchor `regular` (6 needed) | 24,736 candidates (both conditions) | 6 | ample |

## 9. Rationale summary

- Every threshold traces to a real computed quantity (FK/Jacobian/joint-limit), never DLS
  convergence, DLS error, `q_target`-as-solution, or frozen-test data.
- `near_joint_limit` is a *newly derived* quantile (P10 of this phase's own generic pool,
  0.024991237796029034) — distinct from v1 Point-IK's historical `0.006685...`, because v1's value
  was the P10 of a **pair-minimum** (`min(initial_margin, target_margin)` over two draws), while
  this phase calibrates the **single-configuration** marginal distribution needed by anchors/trials
  (which have no "pair"). A pair-minimum distribution is always ≤ its single-configuration
  counterpart at the same quantile level, so applying this single-configuration threshold to future
  Point-IK pairs will yield *at least* as many eligible pairs as this phase's counts show, never
  fewer.
- `near_singularity` reuses v1's existing, already-cross-referenced threshold rather than inventing
  a new number, per the phase charter's stated preference.
- `moderately_conditioned`/`regular` reuse the same `3.0` multiplier already load-bearing in two
  other places in this codebase (`select_anchor`, Tier 0), so all threshold-consuming code paths
  agree.

## 10. Limitations

- Only `near_joint_limit`, `near_singularity`, `moderately_conditioned`, and `regular` are
  calibrated here. `near_target`/`medium_target`/`far_target`/`large_orientation_change` quantile
  thresholds are out of this phase's charter and remain `unresolved` for Point-IK generation (to be
  re-derived analogous to v1's `_derive_thresholds` when that phase runs).
- The anchor **selection procedure** (searching for 3 diverse `near_limit` + 3 diverse
  `near_singular` anchors satisfying these thresholds, analogous to `select_anchor`'s search loop)
  is not implemented by this phase — only the acceptance thresholds are locked.
- Calibration was run once at seed 42 with the config-declared pool sizes; determinism and
  cross-seed variation are covered by automated tests
  (`tests/test_dataset_v2_threshold_calibration.py`), but no multi-seed stability study across many
  seeds was performed (not required by the phase charter).

## 11. Reproducibility command

```
python -m pipelines.run_dataset_v2_threshold_calibration --seed 42
```

(Add `--report-json PATH` to also write the scalar summary as JSON to a caller-supplied path —
never into a Dataset v2 root or the repository. No candidate pool or per-candidate array is ever
written to disk.)
