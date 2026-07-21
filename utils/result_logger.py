"""Writes per-tier run results and summaries to a caller-specified output directory.

Never chooses an output location on its own (no implicit writes into the dataset input
directories); every function here requires an explicit path from the caller.
"""

import json
from pathlib import Path
from typing import Any, Union

import pandas as pd

from utils.csv_utils import write_dataframe_csv


def write_result_csv(df: pd.DataFrame, path: Union[str, Path], overwrite: bool = True) -> Path:
    """Write a results DataFrame to ``path`` as CSV (stable column order, atomic write)."""
    return write_dataframe_csv(df, path, overwrite=overwrite)


def write_result_json(payload: Any, path: Union[str, Path], overwrite: bool = True) -> Path:
    """Write a JSON-serializable ``payload`` to ``path``, atomically, with stable key order."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path
