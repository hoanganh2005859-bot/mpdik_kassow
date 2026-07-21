"""Shared helpers for generators/: model/config loading, RNG derivation, and NPZ/checksum I/O.

Internal to the generators/ package; not part of the public kinematics/utils API surface.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from kinematics.model_loader import ModelContext, load_model_context
from utils.config_loader import load_json_config
from utils.file_checksum import sha256_file

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

GENERATOR_VERSION = "1.0.0"

_MODEL_CONTEXT: Optional[ModelContext] = None


def get_model_context() -> ModelContext:
    """Return a process-wide cached ModelContext (compiling the MuJoCo model is expensive)."""
    global _MODEL_CONTEXT
    if _MODEL_CONTEXT is None:
        _MODEL_CONTEXT = load_model_context()
    return _MODEL_CONTEXT


def load_benchmark_config() -> dict:
    return load_json_config(CONFIGS_DIR / "benchmark_config.json")


def load_dls_config() -> dict:
    return load_json_config(CONFIGS_DIR / "dls_config.json")


def load_trajectory_config() -> dict:
    return load_json_config(CONFIGS_DIR / "trajectory_config.json")


def load_robot_config() -> dict:
    return load_json_config(CONFIGS_DIR / "robot_config.json")


def derive_seed(base_seed: int, *tags: int) -> int:
    """Deterministically derive a child seed from a base seed and integer tags.

    Built on numpy's SeedSequence so derived seeds are well separated and reproducible
    across runs/platforms, independent of Python's hash randomization.
    """
    entropy = [int(base_seed) & 0xFFFFFFFF] + [int(t) & 0xFFFFFFFF for t in tags]
    seq = np.random.SeedSequence(entropy)
    return int(seq.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def rng_from(base_seed: int, *tags: int) -> np.random.Generator:
    """Deterministic numpy Generator derived from a base seed and integer tags."""
    return np.random.default_rng(derive_seed(base_seed, *tags))


def relative_to_repo(path) -> str:
    """POSIX-style path relative to the repo root, safe to store in manifests (no absolute path)."""
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


def ensure_output_path(path, overwrite: bool) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to regenerate it")
    return path


def save_npz(path, overwrite: bool, arrays: Dict[str, np.ndarray]) -> Path:
    """Save arrays to an .npz archive, rejecting object dtype and non-finite float values upfront."""
    path = ensure_output_path(path, overwrite)
    clean_arrays = {}
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        if arr.dtype == object:
            raise TypeError(f"array '{name}' has object dtype; not allowed in NPZ output")
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            raise ValueError(f"array '{name}' contains NaN/Inf values")
        clean_arrays[name] = arr
    np.savez(path, **clean_arrays)
    return path


def config_hash(config: dict) -> str:
    """Stable SHA256 hash of a JSON-serializable config dict (sorted keys, compact separators)."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_checksum_entry(npz_path, arrays: Dict[str, np.ndarray], generation_seed: int, extra: Optional[dict] = None) -> dict:
    """Build a checksum/manifest metadata entry for a generated NPZ file."""
    npz_path = Path(npz_path)
    sample_count = int(next(iter(arrays.values())).shape[0]) if arrays else 0
    entry = {
        "filename": relative_to_repo(npz_path),
        "sha256": sha256_file(npz_path),
        "file_size_bytes": npz_path.stat().st_size,
        "sample_count": sample_count,
        "array_names": sorted(arrays.keys()),
        "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
        "array_dtypes": {k: str(v.dtype) for k, v in arrays.items()},
        "generation_seed": int(generation_seed),
    }
    if extra:
        entry.update(extra)
    return entry


def write_json(path, overwrite: bool, payload: dict) -> Path:
    path = ensure_output_path(path, overwrite)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path
