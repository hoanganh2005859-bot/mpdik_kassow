"""Loads and merges JSON configuration files from configs/ with experiment preset overrides."""

import json
from pathlib import Path
from typing import Union

from utils.exceptions import ModelConfigurationError


def load_json_config(path: Union[str, Path]) -> dict:
    """Load a single JSON config file from disk.

    Args:
        path: Path (str or pathlib.Path) to a JSON file. Relative paths are
            resolved against the current working directory, not hardcoded.

    Returns:
        The parsed JSON content as a plain dict.

    Raises:
        ModelConfigurationError: if the file does not exist or is not valid JSON.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ModelConfigurationError(f"config file not found: {config_path}")

    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelConfigurationError(f"could not read config file: {config_path}") from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelConfigurationError(f"invalid JSON in config file: {config_path} ({exc})") from exc
