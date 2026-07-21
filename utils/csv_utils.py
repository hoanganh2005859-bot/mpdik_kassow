"""Reads and writes the trajectory manifest, trial, and custom trajectory CSV formats.

Also provides generic, dependency-free helpers used by algorithms/ and evaluation/ to write
pandas DataFrames to CSV with a stable column order and to coerce numpy scalar types (which are
not JSON-serializable) into plain Python types before writing CSV or JSON output.
"""

from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd


def json_safe_scalar(value: Any) -> Any:
    """Coerce a numpy scalar/bool/array-of-one into a plain JSON/CSV-safe Python type.

    Leaves ``None`` and already-plain Python types unchanged. Never returns an object-dtype
    numpy value. Raises TypeError for multi-element arrays (those must be flattened into
    separate columns by the caller, not embedded as a single cell).
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise TypeError(
                f"cannot store a {value.size}-element array as a single scalar cell; flatten it into separate columns first"
            )
        return json_safe_scalar(value.reshape(()).item())
    return value


def write_dataframe_csv(df: pd.DataFrame, path: Union[str, Path], overwrite: bool = True) -> Path:
    """Write ``df`` to ``path`` as CSV, preserving the DataFrame's own column order.

    Writes atomically (temp file + os.replace) so a crash mid-write never leaves a truncated
    CSV at ``path``. Raises FileExistsError if ``path`` exists and ``overwrite`` is False.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(path.name + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)
    return path


def read_dataframe_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read a CSV previously written by ``write_dataframe_csv`` back into a DataFrame."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)
