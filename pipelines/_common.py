"""Shared orchestration helpers for pipelines/run_tier0_kinematics.py ... run_tier0_to_tier4.py.

Centralizes: config loading/merging (resolved_config.json), output-directory structure creation
and safety-checked overwrite, sample/trial selection for the smoke/full presets, and checksum
helpers used by run_manifest.json. Nothing here writes into assets/, benchmarks/, or
trajectories/ -- every write target is an explicit output path supplied by the caller.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils.config_loader import load_json_config
from utils.dataset_locator import CONFIGS_DIR, REPO_ROOT
from utils.file_checksum import sha256_file

CONFIG_FILES = {
    "robot_config": CONFIGS_DIR / "robot_config.json",
    "frame_config": CONFIGS_DIR / "frame_config.json",
    "dls_config": CONFIGS_DIR / "dls_config.json",
    "benchmark_config": CONFIGS_DIR / "benchmark_config.json",
    "trajectory_config": CONFIGS_DIR / "trajectory_config.json",
    "evaluation_config": CONFIGS_DIR / "evaluation_config.json",
}
EXPERIMENT_PRESETS_PATH = CONFIGS_DIR / "experiment_presets.json"

VALID_METHODS = ("warm_start", "cold_start")
VALID_TRIAL_CATEGORIES = ("repeatability", "robustness", "all")

# Safety allowlist for --overwrite: the output directory must resolve underneath one of these,
# and must never equal the directory itself, so an operator can never accidentally point
# --output at the repo root or a dataset input directory.
_PROTECTED_DIRS = [
    REPO_ROOT,
    REPO_ROOT / "assets",
    REPO_ROOT / "benchmarks",
    REPO_ROOT / "trajectories",
    REPO_ROOT / "kr810",
    REPO_ROOT / "configs",
    REPO_ROOT / "schemas",
    REPO_ROOT / "algorithms",
    REPO_ROOT / "evaluation",
    REPO_ROOT / "kinematics",
    REPO_ROOT / "generators",
    REPO_ROOT / "pipelines",
    REPO_ROOT / "utils",
    REPO_ROOT / "tests",
]


def load_all_configs() -> Dict[str, dict]:
    """Load every configs/*.json file used by the pipeline."""
    return {name: load_json_config(path) for name, path in CONFIG_FILES.items()}


def load_experiment_presets() -> Dict[str, dict]:
    """Load configs/experiment_presets.json."""
    return load_json_config(EXPERIMENT_PRESETS_PATH)


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "all":
        return None
    return int(value)


def _as_optional_str_list(value: Any) -> Optional[List[str]]:
    if value is None or value == "all":
        return None
    return list(value)


@dataclass
class ResolvedSettings:
    """The flattened, effective settings pipelines actually consume (CLI > preset > config default)."""

    preset: str
    seed: int
    point_sample_limit: Optional[int]
    validation_fk_samples: Optional[int]
    validation_jacobian_samples: Optional[int]
    validation_singularity_samples: Optional[int]
    selected_trajectories: Optional[List[str]]
    methods: List[str]
    trial_category: str
    trial_limit: Optional[int]
    waypoint_limit: Optional[int]
    make_plots: bool


def build_resolved_config(preset: str, cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge configs/*.json + the selected preset + CLI overrides into one resolved config.

    Priority: CLI override > preset > config file default. Never mutates the source config
    files; returns a fresh dict every call.
    """
    presets = load_experiment_presets()
    if preset not in presets:
        raise ValueError(f"unknown preset '{preset}', expected one of {sorted(presets)}")
    preset_values = presets[preset]
    configs = load_all_configs()

    def resolved(cli_key: str, preset_key: str, default: Any) -> Any:
        if cli_overrides.get(cli_key) is not None:
            return cli_overrides[cli_key]
        if preset_key in preset_values:
            return preset_values[preset_key]
        return default

    seed = int(cli_overrides.get("seed") or configs["benchmark_config"].get("random_seed", 42))

    methods = cli_overrides.get("methods")
    if methods is None:
        methods = list(VALID_METHODS)
    for m in methods:
        if m not in VALID_METHODS:
            raise ValueError(f"unknown method '{m}', expected one of {VALID_METHODS}")

    trial_category = cli_overrides.get("trial_category") or "all"
    if trial_category not in VALID_TRIAL_CATEGORIES:
        raise ValueError(f"unknown trial_category '{trial_category}', expected one of {VALID_TRIAL_CATEGORIES}")

    settings = ResolvedSettings(
        preset=preset,
        seed=seed,
        point_sample_limit=_as_optional_int(resolved("point_sample_limit", "point_sample_limit", None)),
        validation_fk_samples=_as_optional_int(resolved("validation_fk_samples", "validation_fk_samples", None)),
        validation_jacobian_samples=_as_optional_int(
            resolved("validation_jacobian_samples", "validation_jacobian_samples", None)
        ),
        validation_singularity_samples=_as_optional_int(
            resolved("validation_singularity_samples", "validation_singularity_samples", None)
        ),
        selected_trajectories=_as_optional_str_list(
            resolved("trajectory_ids", "selected_trajectories", None)
        ),
        methods=methods,
        trial_category=trial_category,
        trial_limit=_as_optional_int(resolved("trial_limit", "trajectory_trial_limit", None)),
        waypoint_limit=_as_optional_int(resolved("waypoint_limit", "waypoint_limit", None)),
        make_plots=not bool(cli_overrides.get("no_plots", False)),
    )

    return {
        "preset": preset,
        "robot_config": configs["robot_config"],
        "frame_config": configs["frame_config"],
        "dls_config": configs["dls_config"],
        "benchmark_config": configs["benchmark_config"],
        "trajectory_config": configs["trajectory_config"],
        "evaluation_config": configs["evaluation_config"],
        "experiment_preset": preset_values,
        "cli_overrides": {k: v for k, v in cli_overrides.items() if v is not None},
        "effective": settings.__dict__,
    }


# --------------------------------------------------------------------------------------------
# Output directory structure
# --------------------------------------------------------------------------------------------

TIER_DIR_NAMES = {
    "tier0": "tier0_kinematics",
    "tier1": "tier1_point_dls",
    "tier2": "tier2_sequential_dls",
    "tier3": "tier3_trajectory_tracking",
    "tier4": "tier4_joint_feasibility",
}


def tier_dir(output_dir: Path, tier: str) -> Path:
    return Path(output_dir) / TIER_DIR_NAMES[tier]


def ensure_output_structure(output_dir: Path) -> Dict[str, Path]:
    """Create the Tier 0-4 output directory tree (figures/ subdirs included) under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"root": output_dir}
    for tier, dirname in TIER_DIR_NAMES.items():
        tier_path = output_dir / dirname
        tier_path.mkdir(parents=True, exist_ok=True)
        (tier_path / "figures").mkdir(parents=True, exist_ok=True)
        paths[tier] = tier_path
    return paths


def is_safe_to_remove(output_dir: Path) -> bool:
    """True iff ``output_dir`` is safe for --overwrite to recursively delete.

    Refuses the repo root itself, any of its top-level source/dataset directories, and any
    ancestor of those directories (which would delete them as a side effect).
    """
    resolved = Path(output_dir).resolve()
    for protected in _PROTECTED_DIRS:
        protected = protected.resolve()
        if resolved == protected:
            return False
        if protected in resolved.parents:
            continue
        if resolved in protected.parents:
            return False
    return True


# --------------------------------------------------------------------------------------------
# Selection helpers
# --------------------------------------------------------------------------------------------


def select_stratified_point_sample_ids(
    sample_ids: np.ndarray, difficulty_ids: np.ndarray, limit: Optional[int]
) -> Optional[List[int]]:
    """Deterministically select up to ``limit`` point-IK sample ids, spread across difficulty groups.

    Returns None if ``limit`` is None (caller should then use every sample). Otherwise takes an
    even share from each observed difficulty group (in ascending sample_id order within each
    group), then truncates to exactly ``limit`` total, so every run of the same benchmark with
    the same limit selects the same ids.
    """
    if limit is None:
        return None

    unique_groups = sorted(int(g) for g in np.unique(difficulty_ids))
    n_groups = len(unique_groups)
    per_group = -(-limit // n_groups)  # ceil

    selected: List[int] = []
    for group in unique_groups:
        group_sample_ids = sorted(int(s) for s in sample_ids[difficulty_ids == group])
        selected.extend(group_sample_ids[:per_group])

    selected = sorted(selected)[:limit]
    return selected


def select_trials(
    trials_df: pd.DataFrame,
    trajectory_ids: Optional[Sequence[str]],
    trial_category: str,
    trial_limit: Optional[int],
) -> pd.DataFrame:
    """Filter/select trajectory_trials.csv rows for Tier 2, per CLI/preset selection rules.

    Selection is deterministic (sorted by trial_id). When ``trial_category == "all"`` and a
    ``trial_limit`` is given, coverage is guaranteed per selected trajectory first: one
    repeatability and one robustness trial for each distinct ``trajectory_id`` among the
    filtered candidates (in trajectory_id order) are kept before filling any remaining slots in
    sorted trial_id order -- so a multi-trajectory selection (e.g. the smoke preset's
    line_fixed_orientation + circle_fixed_orientation) exercises every selected trajectory
    rather than letting the first trajectory alphabetically consume the whole trial_limit.
    """
    df = trials_df
    if trajectory_ids is not None:
        df = df[df["trajectory_id"].isin(list(trajectory_ids))]
    if trial_category != "all":
        df = df[df["trial_category"] == trial_category]

    df = df.sort_values("trial_id").reset_index(drop=True)

    if trial_limit is None or len(df) <= trial_limit:
        return df

    if trial_category == "all":
        picked_idx: List[int] = []
        for trajectory_id in sorted(df["trajectory_id"].unique()):
            for category in sorted(df["trial_category"].unique()):
                first_idx = df.index[(df["trajectory_id"] == trajectory_id) & (df["trial_category"] == category)]
                if len(first_idx) > 0:
                    picked_idx.append(int(first_idx[0]))
        for idx in df.index:
            if len(picked_idx) >= trial_limit:
                break
            if idx not in picked_idx:
                picked_idx.append(int(idx))
        picked_idx = sorted(picked_idx)[:trial_limit]
        return df.loc[picked_idx].reset_index(drop=True)

    return df.iloc[:trial_limit].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Checksums
# --------------------------------------------------------------------------------------------


def relative_to_repo(path: Path) -> str:
    """Render ``path`` relative to the repo root when possible (never an absolute source path)."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def compute_dataset_checksums(selected_trajectory_ids: Optional[Sequence[str]]) -> Dict[str, Any]:
    """SHA256 checksums for the model, the point-IK benchmark, and the selected trajectory files."""
    from utils.dataset_locator import MODEL_PATH, POINT_IK_BENCHMARK_PATH, TRAJECTORY_MANIFEST_PATH

    checksums: Dict[str, Any] = {
        "model_sha256": sha256_file(MODEL_PATH),
        "point_benchmark_sha256": sha256_file(POINT_IK_BENCHMARK_PATH),
        "trajectory_file_checksums": {},
    }

    manifest = pd.read_csv(TRAJECTORY_MANIFEST_PATH)
    if selected_trajectory_ids is not None:
        manifest = manifest[manifest["trajectory_id"].isin(list(selected_trajectory_ids))]

    for _, row in manifest.iterrows():
        file_path = REPO_ROOT / row["file_path"]
        checksums["trajectory_file_checksums"][row["trajectory_id"]] = sha256_file(file_path)

    return checksums


def canonical_signature(payload: Dict[str, Any]) -> str:
    """A stable string signature of a JSON-serializable dict, for resume input-change detection."""
    import hashlib

    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
