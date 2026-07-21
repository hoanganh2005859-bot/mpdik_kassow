"""Resolves repository-relative paths for assets, benchmarks, and trajectories across local/Kaggle runs.

Every path returned here is derived from this file's own location via ``pathlib.Path`` (never a
hardcoded ``D:\\``, ``C:\\``, ``/home/...``, or Kaggle input slug), so the same code resolves
correctly on Windows, Linux, and Kaggle notebooks.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = REPO_ROOT / "assets"
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
TRAJECTORIES_DIR = REPO_ROOT / "trajectories"
CONFIGS_DIR = REPO_ROOT / "configs"
SCHEMAS_DIR = REPO_ROOT / "schemas"

MODEL_PATH = ASSETS_DIR / "kr810.xml"
POINT_IK_BENCHMARK_PATH = BENCHMARKS_DIR / "point_ik" / "point_ik_v1.npz"
FK_VALIDATION_PATH = BENCHMARKS_DIR / "validation" / "fk_test_states.npz"
JACOBIAN_VALIDATION_PATH = BENCHMARKS_DIR / "validation" / "jacobian_test_states.npz"
SINGULARITY_VALIDATION_PATH = BENCHMARKS_DIR / "validation" / "singularity_test_states.npz"
TRAJECTORY_MANIFEST_PATH = TRAJECTORIES_DIR / "trajectory_manifest.csv"
TRAJECTORY_TRIALS_PATH = TRAJECTORIES_DIR / "trajectory_trials.csv"


def repo_root() -> Path:
    """Absolute path to the repository root, derived from this file's location."""
    return REPO_ROOT


def resolve_repo_path(relative_path: str) -> Path:
    """Resolve a repo-relative path (e.g. 'assets/kr810.xml') against the repository root."""
    return REPO_ROOT / relative_path
