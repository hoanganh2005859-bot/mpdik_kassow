"""Dataset v2 DLS evaluation harness (Phase 8A).

This package is the *evaluation* counterpart to ``dataset_v2/`` (generation). It is kept strictly
separate from Dataset v1's ``evaluation/`` package (which it reuses for pure metric math, never
modifies) and from the v1 ``pipelines/run_tier*`` CLIs (which are hard-wired to Dataset v1 paths).

Design invariants enforced across the package:

* The evaluator only ever reads a **public** evaluation root (target poses + ``q_initial`` +
  public metadata). ``q_reference`` / ``q_reference_start`` / ``q_target_reference`` and every
  other protected reconstruction array are physically absent from that root and are additionally
  rejected at runtime by :mod:`evaluation_v2.protected_guard`.
* Every path is resolved against an explicit root (dataset root, public root, protected root,
  evaluation-output root) -- no CWD assumptions, no absolute paths baked into any output.
* ``frozen_test`` is never touched by anything in this package's development/validation flow.
* Nothing here imports or runs PPO / MPDIK / MAPPO / dynamics / controllers.
"""
