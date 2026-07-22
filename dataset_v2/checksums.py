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


def build_checksum_manifest(dataset_root: Union[str, Path]) -> dict:
    """Development-stage checksum manifest: only the source/config fingerprint category is
    populated (Phase 1 generates no data, so generated_data_checksum/release_archive_checksum stay
    empty, not fabricated).
    """
    return {
        "dataset_root_relative": True,
        "categories": {
            "source_config_fingerprint": build_source_config_fingerprint(dataset_root),
            "generated_data_checksum": [],
            "release_archive_checksum": [],
        },
        "status": "scaffold_only_no_generated_data",
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
