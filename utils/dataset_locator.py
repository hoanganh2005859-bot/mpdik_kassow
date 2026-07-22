"""Resolves repository-relative paths for assets, benchmarks, and trajectories across local/Kaggle runs.

Every path returned here is derived from this file's own location via ``pathlib.Path`` (never a
hardcoded ``D:\\``, ``C:\\``, ``/home/...``, or Kaggle input slug), so the same code resolves
correctly on Windows, Linux, and Kaggle notebooks.

``resolve_dataset_root``/``dataset_paths_for`` below are the one central, reusable mechanism for
resolving an *explicit* dataset root (never CWD-implicit, never hardcoded) -- used both to keep
Dataset v1's default (root-less) behavior working unchanged, and as the building block Dataset
v2's own locator (``dataset_v2.locator``) is built on. Do not duplicate this normalization logic
elsewhere; extend it here instead.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from utils.exceptions import ModelConfigurationError

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DatasetPaths:
    """The Dataset v1 directory/file layout, resolved against an arbitrary root.

    Field set mirrors this module's own v1 constants (``ASSETS_DIR``, ``BENCHMARKS_DIR``, ...) so
    the same layout can be resolved against any root, not only ``REPO_ROOT``.
    """

    root: Path
    assets_dir: Path
    benchmarks_dir: Path
    trajectories_dir: Path
    configs_dir: Path
    schemas_dir: Path
    model_path: Path
    point_ik_benchmark_path: Path
    fk_validation_path: Path
    jacobian_validation_path: Path
    singularity_validation_path: Path
    trajectory_manifest_path: Path
    trajectory_trials_path: Path


def dataset_paths_for(root: Union[str, Path]) -> DatasetPaths:
    """Build the Dataset v1 path bundle rooted at ``root`` (does not require ``root`` to exist)."""
    root = Path(root)
    benchmarks_dir = root / "benchmarks"
    trajectories_dir = root / "trajectories"
    return DatasetPaths(
        root=root,
        assets_dir=root / "assets",
        benchmarks_dir=benchmarks_dir,
        trajectories_dir=trajectories_dir,
        configs_dir=root / "configs",
        schemas_dir=root / "schemas",
        model_path=root / "assets" / "kr810.xml",
        point_ik_benchmark_path=benchmarks_dir / "point_ik" / "point_ik_v1.npz",
        fk_validation_path=benchmarks_dir / "validation" / "fk_test_states.npz",
        jacobian_validation_path=benchmarks_dir / "validation" / "jacobian_test_states.npz",
        singularity_validation_path=benchmarks_dir / "validation" / "singularity_test_states.npz",
        trajectory_manifest_path=trajectories_dir / "trajectory_manifest.csv",
        trajectory_trials_path=trajectories_dir / "trajectory_trials.csv",
    )


_DEFAULT_PATHS = dataset_paths_for(REPO_ROOT)

ASSETS_DIR = _DEFAULT_PATHS.assets_dir
BENCHMARKS_DIR = _DEFAULT_PATHS.benchmarks_dir
TRAJECTORIES_DIR = _DEFAULT_PATHS.trajectories_dir
CONFIGS_DIR = _DEFAULT_PATHS.configs_dir
SCHEMAS_DIR = _DEFAULT_PATHS.schemas_dir

MODEL_PATH = _DEFAULT_PATHS.model_path
POINT_IK_BENCHMARK_PATH = _DEFAULT_PATHS.point_ik_benchmark_path
FK_VALIDATION_PATH = _DEFAULT_PATHS.fk_validation_path
JACOBIAN_VALIDATION_PATH = _DEFAULT_PATHS.jacobian_validation_path
SINGULARITY_VALIDATION_PATH = _DEFAULT_PATHS.singularity_validation_path
TRAJECTORY_MANIFEST_PATH = _DEFAULT_PATHS.trajectory_manifest_path
TRAJECTORY_TRIALS_PATH = _DEFAULT_PATHS.trajectory_trials_path


def repo_root() -> Path:
    """Absolute path to the repository root, derived from this file's location."""
    return REPO_ROOT


def resolve_repo_path(relative_path: str) -> Path:
    """Resolve a repo-relative path (e.g. 'assets/kr810.xml') against the repository root."""
    return REPO_ROOT / relative_path


def resolve_dataset_root(
    dataset_root: Optional[Union[str, Path]] = None, *, require_exists: bool = True
) -> Path:
    """Resolve an explicit dataset root, or fall back to ``REPO_ROOT`` (Dataset v1's unchanged
    default) when ``dataset_root`` is ``None``.

    Never resolves against the current working directory implicitly: a relative
    ``dataset_root`` is anchored to ``Path.cwd()`` explicitly and then normalized, so the result
    is always an absolute, real path regardless of caller CWD. Raises
    ``ModelConfigurationError`` with an actionable message if ``require_exists`` and the root does
    not exist or is not a directory.
    """
    if dataset_root is None:
        return REPO_ROOT

    root = Path(dataset_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()

    if require_exists:
        if not root.exists():
            raise ModelConfigurationError(
                f"dataset root does not exist: {root} "
                "(pass an existing directory, or create it first)"
            )
        if not root.is_dir():
            raise ModelConfigurationError(f"dataset root is not a directory: {root}")

    return root
