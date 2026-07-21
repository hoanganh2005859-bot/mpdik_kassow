"""Reads and writes benchmark/trajectory NPZ archives with schema-consistent array layouts.

Thin wrappers around numpy's own NPZ I/O that enforce this project's conventions: never use
``allow_pickle=True`` (no object-dtype arrays anywhere in this dataset), and never write
non-finite floating point values silently.
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np


def load_npz(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load an NPZ archive with ``allow_pickle=False`` and return a plain dict of arrays.

    Returning a plain dict (rather than the lazy ``NpzFile``) makes the loaded arrays safe to
    use after the underlying file handle would otherwise be closed, and lets callers treat the
    result like any other in-memory mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def save_npz(path: Union[str, Path], arrays: Dict[str, np.ndarray], overwrite: bool = True) -> Path:
    """Save ``arrays`` to ``path`` as an NPZ archive, rejecting object dtype and non-finite values.

    Writes atomically (temp file + os.replace).
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)

    clean_arrays = {}
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        if arr.dtype == object:
            raise TypeError(f"array '{name}' has object dtype; not allowed in NPZ output")
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            raise ValueError(f"array '{name}' contains NaN/Inf values")
        clean_arrays[name] = arr

    tmp_path = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp_path, **clean_arrays)
    tmp_path.replace(path)
    return path
