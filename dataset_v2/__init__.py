"""Dataset v2 scaffolding: dataset-root locator, config/schema templates, checksum manifest.

Everything here is parameterized by an explicit ``dataset_root`` (never the repo root, never
CWD-implicit) and produces no fabricated data -- see ``specs/DLS_DATASET_V2_SPEC.md`` for the
locked design this scaffolding follows. Phase 1 does not generate Tier 0-4 content; see
``dataset_v2.scaffold`` for what is actually created on disk.
"""
