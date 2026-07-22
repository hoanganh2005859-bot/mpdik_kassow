"""Dataset v2 analogue of ``utils/dataset_locator.py``: paths parameterized by an explicit,
caller-supplied dataset root instead of this file's own location.

Dataset v2 has no repo-root default (unlike Dataset v1's ``REPO_ROOT``-anchored constants) -- a
``dataset_root`` is always required, per ``specs/DLS_DATASET_V2_SPEC.md`` section C ("never the
repo root, never CWD-relative"). Root *normalization* (expanding, anchoring a relative path to
Path.cwd() explicitly, resolving symlinks) is delegated to
``utils.dataset_locator.resolve_dataset_root`` so there is exactly one implementation of that
logic in the repository; this module only adds the v2-specific subpath layout on top.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from utils.dataset_locator import resolve_dataset_root
from utils.exceptions import ModelConfigurationError

VERSION_FILENAME = "VERSION"
MANIFEST_FILENAME = "DATASET_MANIFEST.json"
CHECKSUM_MANIFEST_FILENAME = "CHECKSUM_MANIFEST.json"


@dataclass(frozen=True)
class DatasetV2Paths:
    """The Dataset v2 scaffold layout (see ``specs/DLS_DATASET_V2_SPEC.md`` section C), resolved
    against one explicit ``root``. Directories are not guaranteed to exist -- ``dataset_v2_paths``
    is used both to plan a scaffold before creating it and to address an existing one.
    """

    root: Path
    version_file: Path
    manifest_file: Path
    configs_dir: Path
    schemas_dir: Path
    checksums_dir: Path
    checksum_manifest_file: Path
    tier0_validation_dir: Path
    tier1_point_ik_dir: Path
    anchors_dir: Path
    trajectories_dir: Path
    trajectories_development_dir: Path
    trajectories_validation_dir: Path
    trajectories_frozen_test_dir: Path
    trials_dir: Path
    references_dir: Path
    reports_dir: Path


def dataset_v2_paths(dataset_root: Union[str, Path]) -> DatasetV2Paths:
    """Build the Dataset v2 path bundle rooted at an explicit ``dataset_root``.

    ``dataset_root`` must not be ``None`` -- Dataset v2 never implies a root from CWD or the repo
    root. Does not require the root (or any subpath) to already exist.
    """
    if dataset_root is None:
        raise ModelConfigurationError(
            "Dataset v2 requires an explicit dataset_root; it is never implied from the current "
            "working directory or the repository root."
        )
    root = resolve_dataset_root(dataset_root, require_exists=False)
    trajectories_dir = root / "trajectories"
    checksums_dir = root / "checksums"
    return DatasetV2Paths(
        root=root,
        version_file=root / VERSION_FILENAME,
        manifest_file=root / MANIFEST_FILENAME,
        configs_dir=root / "configs",
        schemas_dir=root / "schemas",
        checksums_dir=checksums_dir,
        checksum_manifest_file=checksums_dir / CHECKSUM_MANIFEST_FILENAME,
        tier0_validation_dir=root / "tier0_validation",
        tier1_point_ik_dir=root / "tier1_point_ik",
        anchors_dir=root / "anchors",
        trajectories_dir=trajectories_dir,
        trajectories_development_dir=trajectories_dir / "development",
        trajectories_validation_dir=trajectories_dir / "validation",
        trajectories_frozen_test_dir=trajectories_dir / "frozen_test",
        trials_dir=root / "trials",
        references_dir=root / "references",
        reports_dir=root / "reports",
    )


def require_dataset_v2_root(dataset_root: Union[str, Path]) -> DatasetV2Paths:
    """Resolve ``dataset_root`` and validate it actually holds a Dataset v2 scaffold/dataset.

    Raises ``ModelConfigurationError`` with an actionable message if the root does not exist, is
    not a directory, or is missing ``DATASET_MANIFEST.json`` (the minimal signal that this is a
    Dataset v2 root at all).
    """
    if dataset_root is None:
        raise ModelConfigurationError(
            "Dataset v2 requires an explicit dataset_root; it is never implied from the current "
            "working directory or the repository root."
        )
    root = resolve_dataset_root(dataset_root, require_exists=True)
    paths = dataset_v2_paths(root)
    if not paths.manifest_file.is_file():
        raise ModelConfigurationError(
            f"dataset-v2 root is missing {MANIFEST_FILENAME}: {paths.manifest_file}. "
            "This directory does not look like a Dataset v2 root/scaffold; create one first via "
            "dataset_v2.scaffold.create_dataset_v2_scaffold(dataset_root=...)."
        )
    return paths


def relative_to_dataset_v2_root(path: Union[str, Path], dataset_root: Union[str, Path]) -> str:
    """POSIX-style path relative to ``dataset_root``, safe to store in v2 manifests/checksums."""
    root = resolve_dataset_root(dataset_root, require_exists=False)
    return Path(path).resolve().relative_to(root).as_posix()
