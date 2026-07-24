# Dataset v2 — Trial difficulty-threshold calibration (Phase 7)

Calibration of the easy/medium/hard bands for the Phase 7 trial generator. Reproduce with:

```
python -m pipelines.run_dataset_v2_trial_calibration --dataset-root <working_root> --master-seed 42
```

against a working root that already holds the 210 trajectories (12 anchors + 120 core + 90
challenge) generated at master seed 42.

## Primary metric (LOCKED)

`primary = 0.5 * (initial_position_error_m / position_scale_m) + 0.5 * (initial_orientation_error_rad / orientation_scale_rad)`

- `initial_position_error_m` — Euclidean distance from `FK(q_initial)` to the trajectory's **first**
  canonical target position.
- `initial_orientation_error_rad` — the **SO(3) geodesic angle** from `FK(q_initial)` to the first
  canonical target quaternion (never an Euler-angle difference).
- The two terms are each divided by a calibration scale (the pooled median of the raw error) so
  neither unit dominates, then combined 50/50.

This is a **combined normalized first-target pose error**, not normalized joint distance: the KR810
is redundant, so many joint configurations map to nearly the same pose and joint distance alone
would be a poor difficulty signal (task/spec section 9). Normalized joint distance to the protected
`q_reference_start` is still computed and stored, but only as a **diagnostic** covariate — it never
defines the class.

## Procedure (development trajectories only)

1. For each of the 70 **development** trajectories, build the same deterministic q_initial
   candidate pool the trial generator uses (`dataset_v2/trial_candidates.py`): a 1,400-candidate
   mixture (500 interior + 300 near-limit + 300 singularity-biased + 300 stratified) drawn **only**
   from the operational joint limits — never from `q_reference`.
2. Compute FK covariates and the raw position/orientation error of each candidate to the
   trajectory's first canonical target pose.
3. Pool all 98,000 development candidates. The calibration **scales** are the pooled medians of the
   raw position and orientation errors.
4. The **bands** are non-overlapping percentile bands of the combined metric with guard gaps
   (easy ≤ P30, medium ∈ [P40, P60], hard ≥ P70). The guard gaps (P30–P40, P60–P70) guarantee
   `easy < medium < hard` by construction and a positive minimum inter-level separation.

Validation and frozen_test trajectories are **never** used to choose the thresholds (spec section J
frozen-test protocol).

## Results (master seed 42, LOCKED into `dataset_v2/config_templates.py`)

| Quantity | Value |
| --- | --- |
| development trajectories | 70 |
| pooled candidates | 98,000 |
| `position_scale_m` (pooled median) | 0.9356705199958848 |
| `orientation_scale_rad` (pooled median) | 2.3014333912872837 |
| `easy_upper` (P30) | 0.8601239140675875 |
| `medium_lower` (P40) | 0.9245379046213449 |
| `medium_upper` (P60) | 1.0462350595875294 |
| `hard_lower` (P70) | 1.1126783448821171 |
| minimum inter-level separation | 0.0644139905537574 |

Raw error ranges (pooled): position 0.0220 – 1.9127 m (median 0.9357, P95 1.5075); orientation
0.0452 – 3.1416 rad (median 2.3014, P95 3.0616).

Pooled band candidate counts: easy 29,400 / medium 19,600 / hard 29,400. **Every** development
trajectory populates all three bands — the tightest per-trajectory band count is 79 (hard), 196
(medium), 215 (easy) out of 1,400, so no band is ever starved.

## How it was locked

The numbers above are baked into `dataset_v2/config_templates.py` as the `TRIAL_*` constants and
surfaced through `configs/trial_config.json`'s `difficulty` block (`status: "locked"`), exactly as
Phase 2.5's threshold calibration was. The policy — primary metric, weights, scales, bands,
inclusivity (band bounds inclusive; guard gaps exclusive), classification priority (primary-band
membership), and minimum separation — was locked **before** any validation or frozen_test trial was
generated, and is never changed after observing frozen results.

## Full-run outcome (working root, master seed 42)

Applying the locked bands to all 210 trajectories produced exactly 630 trials with all bands
populated on the first attempt (no trajectory failed for want of a candidate). Per-trajectory the
selected `easy < medium < hard` holds for all 210 trajectories on the primary metric, with observed
minimum separations 0.215 (easy→medium) and 0.159 (medium→hard) — both well above the configured
0.0644 floor. Position-error ordering held for 208/210 and orientation-error ordering for 184/210
selected triples; the primary combined metric is monotonic for all 210 (secondary single-axis
orderings are not required to be monotonic — spec section 11).
