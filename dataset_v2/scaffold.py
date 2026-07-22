"""Creates the Dataset v2 directory/config/schema/checksum scaffold at an explicit dataset root.

Phase 1 only: this writes structure, config/schema templates, and a checksum manifest over that
scaffold -- it never writes NPZ/CSV data, never invents sample counts, and never claims the
dataset is generated or frozen (``DATASET_MANIFEST.json["generated"]``/``["frozen"]`` are always
``False`` here).
"""

import json
from pathlib import Path
from typing import Union

from dataset_v2 import config_templates, schemas as schema_templates
from dataset_v2.checksums import build_checksum_manifest
from dataset_v2.locator import DatasetV2Paths, dataset_v2_paths
from dataset_v2.manifest import build_dataset_manifest
from utils.dataset_locator import REPO_ROOT
from utils.exceptions import ModelConfigurationError

DATASET_VERSION = config_templates.DATASET_VERSION

# Directories that must exist under the root even before any file is written into them (spec
# section C layout, Phase-1 scope per docs/V2_IMPLEMENTATION_LOG.md). Kept empty (a .gitkeep only)
# until a later generation phase actually populates them -- no placeholder NPZ/CSV here.
_EMPTY_SCAFFOLD_DIRS = (
    "tier0_validation_dir",
    "tier1_point_ik_dir",
    "anchors_dir",
    "trajectories_development_dir",
    "trajectories_validation_dir",
    "trajectories_frozen_test_dir",
    "trials_dir",
    "references_dir",
    "reports_dir",
)

_V1_PROTECTED_DIRS = [
    REPO_ROOT,
    REPO_ROOT / "assets",
    REPO_ROOT / "benchmarks",
    REPO_ROOT / "trajectories",
    REPO_ROOT / "configs",
    REPO_ROOT / "schemas",
    REPO_ROOT / "kinematics",
    REPO_ROOT / "algorithms",
    REPO_ROOT / "generators",
    REPO_ROOT / "evaluation",
    REPO_ROOT / "pipelines",
    REPO_ROOT / "utils",
    REPO_ROOT / "tests",
]


def _reject_v1_paths(root: Path) -> None:
    resolved = root.resolve()
    for protected in _V1_PROTECTED_DIRS:
        protected = protected.resolve()
        if resolved == protected or protected in resolved.parents or resolved in protected.parents:
            raise ModelConfigurationError(
                f"refusing to scaffold Dataset v2 at/under a Dataset v1 path: {resolved} "
                f"(conflicts with {protected}); choose a dataset root outside the repository's "
                "own source/dataset directories."
            )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def create_dataset_v2_scaffold(
    dataset_root: Union[str, Path], master_seed: int, overwrite: bool = False
) -> DatasetV2Paths:
    """Create (or refresh, with ``overwrite=True``) the Dataset v2 scaffold at ``dataset_root``.

    ``master_seed`` is recorded verbatim in ``configs/seed_policy.json`` for traceability -- it is
    not used to generate anything in Phase 1 (there is no generation logic yet).
    """
    paths = dataset_v2_paths(dataset_root)
    _reject_v1_paths(paths.root)

    if paths.manifest_file.is_file() and not overwrite:
        raise FileExistsError(
            f"Dataset v2 scaffold already exists at {paths.root} "
            f"({paths.manifest_file} present); pass overwrite=True to regenerate it."
        )

    paths.root.mkdir(parents=True, exist_ok=True)
    paths.configs_dir.mkdir(parents=True, exist_ok=True)
    paths.schemas_dir.mkdir(parents=True, exist_ok=True)
    paths.checksums_dir.mkdir(parents=True, exist_ok=True)
    for field_name in _EMPTY_SCAFFOLD_DIRS:
        directory = getattr(paths, field_name)
        directory.mkdir(parents=True, exist_ok=True)
        gitkeep = directory / ".gitkeep"
        if not any(directory.iterdir()):
            gitkeep.write_text("", encoding="utf-8")

    paths.version_file.write_text(DATASET_VERSION + "\n", encoding="utf-8")

    for filename, config in config_templates.all_configs(master_seed).items():
        _write_json(paths.configs_dir / filename, config)

    for filename, schema in schema_templates.all_schemas().items():
        _write_json(paths.schemas_dir / filename, schema)

    # DATASET_MANIFEST.json must exist before the checksum manifest is built (it fingerprints it).
    _write_json(paths.manifest_file, build_dataset_manifest())

    checksum_manifest = build_checksum_manifest(paths.root)
    _write_json(paths.checksum_manifest_file, checksum_manifest)

    return paths
