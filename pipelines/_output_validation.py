"""Internal utilities to validate Tier 0-4 pipeline output (required files, JSON/CSV integrity).

Used both by run_tier0_to_tier4.py's --resume logic (decide whether a previous tier's output can
be reused) and by tests/manual verification after a run. Never treats a numerical metric that is
legitimately unavailable (e.g. Tier 4 acceleration utilization with no configured acceleration
limits) as a validation failure -- only checks that its ``status`` field says so explicitly.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class OutputValidationError(Exception):
    """Raised when a pipeline output file fails a structural validation check."""


def validate_json_file(path: Path, required_keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Parse ``path`` as JSON and check that ``required_keys`` are present at the top level."""
    path = Path(path)
    if not path.is_file():
        raise OutputValidationError(f"required JSON file missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OutputValidationError(f"invalid JSON in {path}: {exc}") from exc

    if required_keys:
        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise OutputValidationError(f"{path} missing required keys: {missing}")
    return payload


def validate_csv_file(
    path: Path,
    required_columns: Optional[Sequence[str]] = None,
    allow_empty: bool = False,
) -> "pd.DataFrame":  # noqa: F821 - typing only
    """Read ``path`` as CSV and check header columns / non-emptiness."""
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise OutputValidationError(f"required CSV file missing: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        if not allow_empty:
            raise OutputValidationError(f"{path} has no result rows")
        df = pd.DataFrame()

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise OutputValidationError(f"{path} missing required columns: {missing}")

    if not allow_empty and df.shape[0] == 0:
        raise OutputValidationError(f"{path} has no result rows")

    return df


def validate_no_nan_in_identifiers(df: "pd.DataFrame", identifier_columns: Sequence[str], path: Path) -> None:
    """Check that identifier columns (ids, categories, methods) never contain NaN/null."""
    for col in identifier_columns:
        if col in df.columns and df[col].isna().any():
            raise OutputValidationError(f"{path} column '{col}' contains null identifier values")


def validate_tier_completed(tier_state: Optional[Dict[str, Any]]) -> bool:
    """True iff a stored tier-state record (from run_manifest.json) reports status == 'completed'."""
    return bool(tier_state is not None and tier_state.get("status") == "completed")


def collect_required_files(tier_dir: Path, filenames: Sequence[str]) -> List[Path]:
    """Return the list of required output file paths under ``tier_dir`` (existence not checked)."""
    return [Path(tier_dir) / name for name in filenames]
