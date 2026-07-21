"""Seeds random number generators and records environment/version metadata for reproducible runs."""

import platform
import sys
from typing import Any, Dict

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Return a fresh, deterministic numpy Generator seeded from ``seed``.

    This project never uses the legacy global ``numpy.random`` state (see the generators/
    package, which always derives an explicit ``np.random.default_rng``); this helper exists so
    algorithm/evaluation callers that need a seeded generator follow the same convention.
    """
    return np.random.default_rng(seed)


def environment_metadata() -> Dict[str, Any]:
    """Collect environment/version metadata worth recording alongside a run's results."""
    metadata: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    try:
        import mujoco

        metadata["mujoco_version"] = mujoco.__version__
    except ImportError:
        metadata["mujoco_version"] = None
    return metadata
