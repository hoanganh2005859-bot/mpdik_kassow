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
