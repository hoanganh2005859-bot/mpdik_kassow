"""Dataset v2 checksum/fingerprint scaffolding (spec section N).

Reuses ``utils.file_checksum.sha256_file`` (already streaming, already used by v1) unchanged --
this module only adds the v2-scoped manifest shape and verification logic. Never reads from or
writes to any Dataset v1 checksum file.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from dataset_v2.locator import dataset_v2_paths
from utils.file_checksum import sha256_file

_CHECKSUMMABLE_CATEGORIES = ("source_config_fingerprint", "generated_data_checksum", "release_archive_checksum")

# Directories that may hold *generated* Dataset v2 content (as opposed to the source/config
# scaffold). Fingerprinted into the "generated_data_checksum" category once populated; at
# scaffold time these hold only a ".gitkeep" placeholder and contribute no entries.
_GENERATED_DATA_DIR_FIELDS = (
    "tier0_validation_dir",
    "tier1_point_ik_dir",
    "anchors_dir",
    "trajectories_development_dir",
    "trajectories_validation_dir",
    "trajectories_frozen_test_dir",
    "trials_dir",
)


@dataclass(frozen=True)
class ChecksumMismatch:
    filename: str
    expected_sha256: str
    actual_sha256: str


def build_file_entry(path: Path, dataset_root: Path) -> dict:
    resolved = Path(path)
    relative = resolved.resolve().relative_to(Path(dataset_root).resolve()).as_posix()
    return {
        "filename": relative,
        "sha256": sha256_file(resolved),
        "file_size_bytes": resolved.stat().st_size,
    }


def build_source_config_fingerprint(dataset_root: Union[str, Path]) -> List[dict]:
    """SHA256 entries for every scaffold file that actually exists on disk (VERSION,
    DATASET_MANIFEST.json, configs/*.json, schemas/*.json) -- never a fabricated entry for a file
    that doesn't exist, and never the checksum manifest file itself (would create a self-reference
    loop).
    """
    paths = dataset_v2_paths(dataset_root)
    candidates = [paths.version_file, paths.manifest_file]
    if paths.configs_dir.is_dir():
        candidates.extend(sorted(paths.configs_dir.glob("*.json")))
    if paths.schemas_dir.is_dir():
        candidates.extend(sorted(paths.schemas_dir.glob("*.json")))

    entries = []
    for candidate in candidates:
        if candidate == paths.checksum_manifest_file:
            continue
        if candidate.is_file():
            entries.append(build_file_entry(candidate, paths.root))
    entries.sort(key=lambda e: e["filename"])
    return entries


def build_generated_data_fingerprint(dataset_root: Union[str, Path]) -> List[dict]:
    """SHA256 entries for every generated-data file that actually exists on disk under any of
    Dataset v2's generation output directories (Tier 0 validation states, Point-IK, anchors,
    trajectories, trials). Empty for a scaffold-only root (those directories hold only
    ``.gitkeep``); never a fabricated entry for a file that doesn't exist.
    """
    paths = dataset_v2_paths(dataset_root)
    entries = []
    for field_name in _GENERATED_DATA_DIR_FIELDS:
        directory = getattr(paths, field_name)
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.rglob("*")):
            if candidate.is_file() and candidate.name != ".gitkeep":
                entries.append(build_file_entry(candidate, paths.root))
    entries.sort(key=lambda e: e["filename"])
    return entries


def build_checksum_manifest(dataset_root: Union[str, Path]) -> dict:
    """Checksum manifest over the current on-disk state of ``dataset_root``: the source/config
    fingerprint category always reflects the scaffold, and the generated-data category reflects
    whatever Tier 0-4 generation has actually produced so far (empty at scaffold time, not
    fabricated).
    """
    generated_entries = build_generated_data_fingerprint(dataset_root)
    status = "scaffold_only_no_generated_data" if not generated_entries else "partial_generation_in_progress"
    return {
        "dataset_root_relative": True,
        "categories": {
            "source_config_fingerprint": build_source_config_fingerprint(dataset_root),
            "generated_data_checksum": generated_entries,
            "release_archive_checksum": [],
        },
        "status": status,
    }


def verify_checksum_manifest(dataset_root: Union[str, Path]) -> List[ChecksumMismatch]:
    """Recompute SHA256 for every entry in checksums/CHECKSUM_MANIFEST.json and report mismatches.

    Returns an empty list if every entry matches. Does not raise on mismatch -- callers decide
    whether a mismatch is fatal; this only detects and reports.
    """
    paths = dataset_v2_paths(dataset_root)
    manifest = json.loads(paths.checksum_manifest_file.read_text(encoding="utf-8"))

    mismatches: List[ChecksumMismatch] = []
    for category in _CHECKSUMMABLE_CATEGORIES:
        for entry in manifest["categories"].get(category, []):
            file_path = paths.root / entry["filename"]
            if not file_path.is_file():
                mismatches.append(ChecksumMismatch(entry["filename"], entry["sha256"], "<file not found>"))
                continue
            actual = sha256_file(file_path)
            if actual != entry["sha256"]:
                mismatches.append(ChecksumMismatch(entry["filename"], entry["sha256"], actual))
    return mismatches


def content_hash_of_record(record: Dict) -> str:
    """Stable SHA256 hash of a JSON-serializable record's numeric/content fields.

    Sorted-key, fixed-precision serialization, analogous to ``generators/_common.py::config_hash``
    but intended for sample/trajectory *content* (spec section N) rather than configs -- used by
    future anti-leakage checks (section K) to compare content, not just filenames/ids.
    """

    def _round_floats(value):
        if isinstance(value, float):
            return round(value, 12)
        if isinstance(value, dict):
            return {k: _round_floats(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_round_floats(v) for v in value]
        return value

    encoded = json.dumps(_round_floats(record), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
