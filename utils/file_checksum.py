"""Computes SHA256 checksums for asset and dataset files."""

import hashlib
from pathlib import Path
from typing import Union

_CHUNK_SIZE = 1 << 20


def sha256_file(path: Union[str, Path]) -> str:
    """Compute the SHA256 hex digest of a file's contents, streaming in fixed-size chunks."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
