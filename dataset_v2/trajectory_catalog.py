"""Dataset v2 combined trajectory catalog (Phase 7).

Phases 5/5.x and 6 each wrote their own manifest under ``<dataset_v2_root>/trajectories/``:
``core_trajectory_manifest.csv`` (120 core trajectories) and
``challenge_trajectory_manifest.csv`` (90 random-challenge trajectories). This module builds the
single deterministic *combined* catalog the trial generator (Phase 7) consumes -- exactly 210 rows,
the disjoint union of the two, with a unified column set and dataset-root-relative canonical/source
paths (never absolute).

The catalog is the one place trial generation resolves a trajectory to its on-disk NPZ files and
public metadata; it never stores or exposes ``q_reference`` (that stays a protected array inside the
canonical NPZ, reachable only through ``dataset_v2/trajectory_loading.py``'s protected loader).

This module never modifies Dataset v1, never runs DLS, and never uses global ``numpy.random``
state.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dataset_v2.core_trajectory_generation import MANIFEST_NAME as CORE_MANIFEST_NAME
from dataset_v2.challenge_trajectory_generation import MANIFEST_NAME as CHALLENGE_MANIFEST_NAME
from dataset_v2.core_trajectory_generation import _atomic_write_csv, _atomic_write_json
from dataset_v2.locator import DatasetV2Paths, require_dataset_v2_root
from utils.config_loader import load_json_config

COMBINED_MANIFEST_NAME = "combined_trajectory_manifest.csv"
COMBINED_CATALOG_REPORT_NAME = "combined_trajectory_catalog_report.json"

CORE_FAMILY = "core"
CHALLENGE_FAMILY = "random_challenge"

EXPECTED_TOTAL = 210
EXPECTED_CORE = 120
EXPECTED_CHALLENGE = 90
EXPECTED_PER_SPLIT = 70
CANONICAL_WAYPOINTS = 400

SPLITS = ("development", "validation", "frozen_test")

# Unified column set (spec section C / task section 4). ``shape``/``orientation_mode``/``anchor_id``/
# ``anchor_class`` apply to core rows; ``challenge_family`` applies to random-challenge rows; the
# non-applicable columns are left as the empty string (never a fabricated value).
COMBINED_COLUMNS = [
    "trajectory_id",
    "family",
    "split",
    "shape",
    "orientation_mode",
    "anchor_id",
    "anchor_class",
    "challenge_family",
    "canonical_path",
    "source_path",
    "canonical_waypoint_count",
    "source_waypoint_count",
    "content_hash",
    "sha256",
    "source_sha256",
    "source_seed",
    "path_seed",
    "frozen_seed_revision",
    "frozen_seed_namespace",
    "model_fingerprint",
    "config_fingerprint",
]


@dataclass
class CombinedCatalogReport:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    total: int = 0
    family_counts: Dict[str, int] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)


def _read_manifest_rows(manifest_path: Path) -> List[dict]:
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _split_dir_name(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"unknown split '{split}'")
    return split


def _canonical_rel_path(split: str, trajectory_id: str) -> str:
    return f"trajectories/{_split_dir_name(split)}/{trajectory_id}.npz"


def _source_rel_path(split: str, trajectory_id: str) -> str:
    return f"trajectories/{_split_dir_name(split)}/{trajectory_id}_source.npz"


def _core_row_to_combined(row: dict, frozen_core_revision: int) -> dict:
    split = row["split"]
    frozen = int(frozen_core_revision) if split == "frozen_test" else 0
    return {
        "trajectory_id": row["trajectory_id"],
        "family": CORE_FAMILY,
        "split": split,
        "shape": row["shape"],
        "orientation_mode": row["orientation_mode"],
        "anchor_id": row["anchor_id"],
        "anchor_class": row["anchor_class"],
        "challenge_family": "",
        "canonical_path": _canonical_rel_path(split, row["trajectory_id"]),
        "source_path": _source_rel_path(split, row["trajectory_id"]),
        "canonical_waypoint_count": row["canonical_waypoint_count"],
        "source_waypoint_count": row["source_waypoint_count"],
        "content_hash": row["content_hash"],
        "sha256": row["sha256"],
        "source_sha256": row["source_sha256"],
        "source_seed": row["source_seed"],
        "path_seed": "",
        "frozen_seed_revision": frozen,
        "frozen_seed_namespace": "frozen_core_seed_revision" if frozen else "",
        "model_fingerprint": row["model_fingerprint"],
        "config_fingerprint": row["config_fingerprint"],
    }


def _challenge_row_to_combined(row: dict) -> dict:
    split = row["split"]
    frozen = int(row.get("frozen_challenge_seed_revision", 0) or 0)
    return {
        "trajectory_id": row["trajectory_id"],
        "family": CHALLENGE_FAMILY,
        "split": split,
        "shape": "",
        "orientation_mode": "",
        "anchor_id": "",
        "anchor_class": "",
        "challenge_family": row["challenge_family"],
        "canonical_path": _canonical_rel_path(split, row["trajectory_id"]),
        "source_path": _source_rel_path(split, row["trajectory_id"]),
        "canonical_waypoint_count": row["canonical_waypoint_count"],
        "source_waypoint_count": row["source_waypoint_count"],
        "content_hash": row["content_hash"],
        "sha256": row["sha256"],
        "source_sha256": row["source_sha256"],
        "source_seed": row["source_seed"],
        "path_seed": row["path_seed"],
        "frozen_seed_revision": frozen if split == "frozen_test" else 0,
        "frozen_seed_namespace": "frozen_challenge_seed_revision" if (frozen and split == "frozen_test") else "",
        "model_fingerprint": row["model_fingerprint"],
        "config_fingerprint": row["config_fingerprint"],
    }


def build_combined_rows(paths: DatasetV2Paths) -> List[dict]:
    """Read the two per-family manifests and return the combined rows (sorted by trajectory_id).

    Raises ``FileNotFoundError`` if either per-family manifest is missing -- the combined catalog is
    never built from a partial trajectory set.
    """
    core_manifest = paths.trajectories_dir / CORE_MANIFEST_NAME
    challenge_manifest = paths.trajectories_dir / CHALLENGE_MANIFEST_NAME
    if not core_manifest.is_file():
        raise FileNotFoundError(
            f"core trajectory manifest not found: {core_manifest}; generate core trajectories "
            "before building the combined catalog."
        )
    if not challenge_manifest.is_file():
        raise FileNotFoundError(
            f"challenge trajectory manifest not found: {challenge_manifest}; generate random-challenge "
            "trajectories before building the combined catalog."
        )

    seed_policy = load_json_config(paths.configs_dir / "seed_policy.json")
    frozen_core_revision = int(seed_policy["frozen_core_seed_revision"])

    core_rows = [_core_row_to_combined(r, frozen_core_revision) for r in _read_manifest_rows(core_manifest)]
    challenge_rows = [_challenge_row_to_combined(r) for r in _read_manifest_rows(challenge_manifest)]
    combined = core_rows + challenge_rows
    combined.sort(key=lambda r: r["trajectory_id"])
    return combined


def _rows_to_matrix(rows: List[dict]) -> List[list]:
    return [[row[col] for col in COMBINED_COLUMNS] for row in rows]


def build_combined_catalog(dataset_root, overwrite: bool = False, full_counts: bool = True) -> Path:
    """Build ``trajectories/combined_trajectory_manifest.csv`` (deterministic union of the two
    per-family manifests) and a companion report; returns the manifest path.

    Refuses to overwrite an existing combined manifest unless ``overwrite=True``. Validates the
    union (no duplicate/missing; exact 210 rows when ``full_counts``) before writing -- never
    writes a partial catalog. ``full_counts=False`` (tests/smoke) relaxes only the 210-row
    expectation, never the duplicate/union integrity checks.
    """
    paths = require_dataset_v2_root(dataset_root)
    manifest_path = paths.trajectories_dir / COMBINED_MANIFEST_NAME
    if manifest_path.is_file() and not overwrite:
        raise FileExistsError(
            f"combined trajectory catalog already exists ({manifest_path}); pass overwrite=True "
            "(--overwrite on the CLI) to rebuild it."
        )

    rows = build_combined_rows(paths)

    # Validate the union before writing -- a partial catalog is never persisted.
    ids = [r["trajectory_id"] for r in rows]
    hashes = [r["content_hash"] for r in rows]
    problems: List[str] = []
    if full_counts and len(rows) != EXPECTED_TOTAL:
        problems.append(f"combined catalog has {len(rows)} rows, expected {EXPECTED_TOTAL}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate trajectory_id in combined catalog")
    if len(set(hashes)) != len(hashes):
        problems.append("duplicate content_hash in combined catalog")
    if problems:
        raise ValueError("; ".join(problems))

    _atomic_write_csv(manifest_path, COMBINED_COLUMNS, _rows_to_matrix(rows))

    family_counts: Dict[str, int] = {CORE_FAMILY: 0, CHALLENGE_FAMILY: 0}
    split_counts: Dict[str, int] = {s: 0 for s in SPLITS}
    for r in rows:
        family_counts[r["family"]] = family_counts.get(r["family"], 0) + 1
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1

    report = {
        "combined_manifest": COMBINED_MANIFEST_NAME,
        "total": len(rows),
        "family_counts": family_counts,
        "split_counts": split_counts,
        "canonical_waypoints_per_trajectory": CANONICAL_WAYPOINTS,
        "canonical_poses_total": len(rows) * CANONICAL_WAYPOINTS,
        "columns": COMBINED_COLUMNS,
        "note": (
            "Deterministic union of core_trajectory_manifest.csv (120) and "
            "challenge_trajectory_manifest.csv (90). q_reference is never stored here; it remains a "
            "protected array inside the canonical NPZ, exposed only through the protected loader."
        ),
    }
    _atomic_write_json(paths.trajectories_dir / COMBINED_CATALOG_REPORT_NAME, report)
    return manifest_path


def load_combined_catalog(dataset_root) -> List[dict]:
    """Read the on-disk combined catalog into a list of dict rows (ordered by trajectory_id)."""
    paths = require_dataset_v2_root(dataset_root)
    manifest_path = paths.trajectories_dir / COMBINED_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"combined trajectory catalog not found: {manifest_path}; build it first via "
            "dataset_v2.trajectory_catalog.build_combined_catalog."
        )
    return _read_manifest_rows(manifest_path)


def validate_combined_catalog(dataset_root, full_counts: bool = True) -> CombinedCatalogReport:
    """Independently validate the combined catalog against the two per-family manifests.

    Checks: exact 210 union; exact core/challenge/ split counts; no duplicate trajectory_id or
    content_hash; the union is exactly the two per-family id sets (no missing, no extra); every
    referenced canonical/source NPZ exists; and every path is dataset-root-relative (never
    absolute). Never runs DLS.
    """
    paths = require_dataset_v2_root(dataset_root)
    reasons: List[str] = []

    manifest_path = paths.trajectories_dir / COMBINED_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"combined trajectory catalog not found: {manifest_path}")
    combined = _read_manifest_rows(manifest_path)

    core_manifest = paths.trajectories_dir / CORE_MANIFEST_NAME
    challenge_manifest = paths.trajectories_dir / CHALLENGE_MANIFEST_NAME
    core_rows = _read_manifest_rows(core_manifest) if core_manifest.is_file() else []
    challenge_rows = _read_manifest_rows(challenge_manifest) if challenge_manifest.is_file() else []

    ids = [r["trajectory_id"] for r in combined]
    hashes = [r["content_hash"] for r in combined]
    if len(set(ids)) != len(ids):
        reasons.append("duplicate trajectory_id in combined catalog")
    if len(set(hashes)) != len(hashes):
        reasons.append("duplicate content_hash in combined catalog")

    # Exact union against the two per-family manifests.
    core_ids = {r["trajectory_id"] for r in core_rows}
    challenge_ids = {r["trajectory_id"] for r in challenge_rows}
    combined_ids = set(ids)
    missing_core = core_ids - combined_ids
    missing_challenge = challenge_ids - combined_ids
    extra = combined_ids - (core_ids | challenge_ids)
    if missing_core:
        reasons.append(f"combined catalog missing {len(missing_core)} core trajectory id(s): {sorted(missing_core)[:5]}")
    if missing_challenge:
        reasons.append(f"combined catalog missing {len(missing_challenge)} challenge trajectory id(s): {sorted(missing_challenge)[:5]}")
    if extra:
        reasons.append(f"combined catalog has {len(extra)} id(s) not in any per-family manifest: {sorted(extra)[:5]}")
    if core_ids & challenge_ids:
        reasons.append("a trajectory_id appears in BOTH the core and challenge manifests")

    family_counts: Dict[str, int] = {CORE_FAMILY: 0, CHALLENGE_FAMILY: 0}
    split_counts: Dict[str, int] = {s: 0 for s in SPLITS}
    for r in combined:
        family_counts[r["family"]] = family_counts.get(r["family"], 0) + 1
        if r["split"] in split_counts:
            split_counts[r["split"]] += 1

    if full_counts:
        if len(combined) != EXPECTED_TOTAL:
            reasons.append(f"combined catalog has {len(combined)} rows, expected {EXPECTED_TOTAL}")
        if family_counts.get(CORE_FAMILY, 0) != EXPECTED_CORE:
            reasons.append(f"core family count {family_counts.get(CORE_FAMILY, 0)} != {EXPECTED_CORE}")
        if family_counts.get(CHALLENGE_FAMILY, 0) != EXPECTED_CHALLENGE:
            reasons.append(f"challenge family count {family_counts.get(CHALLENGE_FAMILY, 0)} != {EXPECTED_CHALLENGE}")
        for split in SPLITS:
            if split_counts[split] != EXPECTED_PER_SPLIT:
                reasons.append(f"split '{split}' count {split_counts[split]} != {EXPECTED_PER_SPLIT}")

    # Path integrity: relative + files exist.
    for r in combined:
        for col in ("canonical_path", "source_path"):
            rel = r[col]
            if Path(rel).is_absolute() or ":" in rel or rel.startswith("/") or rel.startswith("\\"):
                reasons.append(f"{r['trajectory_id']}: {col} is not dataset-root-relative ({rel})")
                continue
            if not (paths.root / rel).is_file():
                reasons.append(f"{r['trajectory_id']}: {col} does not exist on disk ({rel})")

    return CombinedCatalogReport(
        passed=len(reasons) == 0,
        reasons=reasons,
        total=len(combined),
        family_counts=family_counts,
        split_counts=split_counts,
    )
