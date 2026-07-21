"""Writes per-tier run results and summaries to a caller-specified output directory.

Never chooses an output location on its own (no implicit writes into the dataset input
directories); every function here requires an explicit path from the caller.

Also configures console/file logging for pipeline runs: short console messages (no per-waypoint
prints), an optional log file under the run's output directory, and a helper to silence tqdm
progress bars without touching call sites (they already accept ``show_progress``).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from utils.csv_utils import write_dataframe_csv

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


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


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Configure the root logger for a pipeline run: short console output plus an optional file.

    Only records the relative ``log_file`` path (never an absolute source-machine path beyond
    what the caller explicitly supplies for the run's own output directory).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())
    root_logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger
