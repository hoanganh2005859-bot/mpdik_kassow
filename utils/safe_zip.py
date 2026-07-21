"""Stdlib-only safe ZIP extraction and root-manifest peeking for the Kaggle release ZIP.

Rejects path traversal, absolute member paths, symlinks, and oversized archives before writing
anything to disk. This is the tested reference implementation; the Kaggle notebook's dataset-ZIP
fallback (notebooks/KR810_Tier0_Tier4_Kaggle_Template.ipynb, "Locate dataset" cell) inlines an
equivalent, stdlib-only copy of this logic because it must run before the project's own `utils`
package -- which lives inside the very archive being validated -- is importable.
"""

import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_MAX_MEMBERS = 20000
DEFAULT_MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:/")


class UnsafeZipError(Exception):
    """Raised when a ZIP archive fails a safety check (traversal, absolute path, symlink, size)."""


def is_unsafe_member_name(name: str) -> bool:
    """True if ``name`` is an absolute path or contains a '..' path-traversal segment."""
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    if _WINDOWS_ABSOLUTE_RE.match(normalized):
        return True
    if ".." in Path(normalized).parts:
        return True
    return False


def is_symlink_member(info: zipfile.ZipInfo) -> bool:
    """True if ``info`` encodes a Unix symlink in its external_attr (Windows-created ZIPs never set this)."""
    mode = (info.external_attr >> 16) & 0xFFFF
    return bool(mode) and (mode & 0o170000) == 0o120000


def safe_extract_zip(
    zip_path: Path,
    destination: Path,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> None:
    """Safely extract ``zip_path`` into ``destination`` (created if needed).

    Raises UnsafeZipError -- before writing anything -- if any member is an absolute path,
    contains '..', would resolve outside ``destination``, is a symlink, or if the archive
    exceeds ``max_members`` / ``max_uncompressed_bytes`` (a basic zip-bomb guard).
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > max_members:
            raise UnsafeZipError(f"archive has {len(infos)} entries, exceeds limit {max_members}")

        total_uncompressed = sum(info.file_size for info in infos)
        if total_uncompressed > max_uncompressed_bytes:
            raise UnsafeZipError(
                f"archive uncompressed size {total_uncompressed} exceeds limit {max_uncompressed_bytes}"
            )

        planned = []
        for info in infos:
            if is_unsafe_member_name(info.filename):
                raise UnsafeZipError(f"rejected unsafe member path: {info.filename!r}")
            if is_symlink_member(info):
                raise UnsafeZipError(f"rejected symlink member: {info.filename!r}")

            target = (destination / info.filename).resolve()
            if target != destination and destination not in target.parents:
                raise UnsafeZipError(f"member resolves outside destination: {info.filename!r}")
            planned.append((info, target))

        for info, target in planned:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, open(target, "wb") as handle:
                shutil.copyfileobj(source, handle)


def read_root_manifest(zip_path: Path, manifest_name: str = "DATASET_MANIFEST.json") -> Optional[dict]:
    """Return the parsed root-level ``manifest_name`` JSON from ``zip_path``, or None if absent/invalid."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if manifest_name not in set(zf.namelist()):
                return None
            with zf.open(manifest_name) as handle:
                return json.loads(handle.read().decode("utf-8"))
    except (zipfile.BadZipFile, OSError, json.JSONDecodeError, KeyError):
        return None


def zip_contains_all(zip_path: Path, required_names: Iterable[str]) -> bool:
    """True iff every path in ``required_names`` is present in ``zip_path``'s namelist."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
    except (zipfile.BadZipFile, OSError):
        return False
    return all(name in names for name in required_names)
