"""Path bundles for the three Phase 8A roots, each resolved against an explicit caller-supplied
location (never CWD, never the repo root, never a hard-coded absolute path):

* **Public evaluation root** -- the ONLY thing the DLS evaluator reads. Contains stripped
  point-IK / trial / trajectory NPZs (target poses + ``q_initial`` + public metadata) and the
  public catalog/manifest. Provably free of any protected reference array.
* **Protected validation root** -- reconstruction evidence (``q_reference`` /
  ``q_target_reference`` + content hashes) kept OUTSIDE the public root, used only to
  independently confirm isolation (no public ``q_initial`` equals a reference solution, no public
  NPZ carries a protected key). Never read by the evaluator.
* **Evaluation output root** -- where run outputs (per-tier CSV/JSON, manifests, checkpoints, the
  lock bundle) are written. Never the dataset root, never the public/protected roots.

Root normalization is delegated to :func:`utils.dataset_locator.resolve_dataset_root` so there is
one implementation of that logic in the repo.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from utils.dataset_locator import resolve_dataset_root
from utils.exceptions import ModelConfigurationError

#: Splits the Phase 8A development/validation flow is allowed to touch. ``frozen_test`` is
#: deliberately absent -- it is never exported to the public root and never evaluated here.
EVAL_SPLITS = ("development", "validation")

PUBLIC_MANIFEST_FILENAME = "PUBLIC_EXPORT_MANIFEST.json"
PROTECTED_MANIFEST_FILENAME = "PROTECTED_VALIDATION_MANIFEST.json"


@dataclass(frozen=True)
class PublicEvalPaths:
    """Layout of the public evaluation root."""

    root: Path
    manifest_file: Path
    configs_dir: Path
    schemas_dir: Path
    point_ik_dir: Path
    trials_dir: Path
    trajectories_dir: Path

    def point_ik_split_file(self, split: str) -> Path:
        return self.point_ik_dir / f"{split}.npz"

    def trials_split_file(self, split: str) -> Path:
        return self.trials_dir / f"{split}.npz"

    def combined_manifest_file(self) -> Path:
        return self.trajectories_dir / "public_trajectory_manifest.csv"

    def trajectory_split_dir(self, split: str) -> Path:
        return self.trajectories_dir / split


@dataclass(frozen=True)
class ProtectedValidationPaths:
    """Layout of the protected validation root (isolation evidence only)."""

    root: Path
    manifest_file: Path
    point_ik_dir: Path
    trajectories_dir: Path

    def point_ik_split_file(self, split: str) -> Path:
        return self.point_ik_dir / f"{split}.npz"

    def trajectory_reference_file(self, split: str) -> Path:
        return self.trajectories_dir / f"{split}_q_reference_hashes.json"


@dataclass(frozen=True)
class EvalOutputPaths:
    """Layout of an evaluation output root for one (config, run) directory."""

    root: Path

    def config_run_dir(self, run_name: str) -> Path:
        return self.root / run_name


def _resolve(root: Union[str, Path], *, require_exists: bool) -> Path:
    if root is None:
        raise ModelConfigurationError(
            "an explicit root is required; it is never implied from the current working "
            "directory or the repository root."
        )
    return resolve_dataset_root(root, require_exists=require_exists)


def public_eval_paths(public_root: Union[str, Path], *, require_exists: bool = False) -> PublicEvalPaths:
    root = _resolve(public_root, require_exists=require_exists)
    return PublicEvalPaths(
        root=root,
        manifest_file=root / PUBLIC_MANIFEST_FILENAME,
        configs_dir=root / "configs",
        schemas_dir=root / "schemas",
        point_ik_dir=root / "tier1_point_ik",
        trials_dir=root / "trials",
        trajectories_dir=root / "trajectories",
    )


def protected_validation_paths(
    protected_root: Union[str, Path], *, require_exists: bool = False
) -> ProtectedValidationPaths:
    root = _resolve(protected_root, require_exists=require_exists)
    return ProtectedValidationPaths(
        root=root,
        manifest_file=root / PROTECTED_MANIFEST_FILENAME,
        point_ik_dir=root / "tier1_point_ik",
        trajectories_dir=root / "trajectories",
    )


def eval_output_paths(output_root: Union[str, Path], *, require_exists: bool = False) -> EvalOutputPaths:
    root = _resolve(output_root, require_exists=require_exists)
    return EvalOutputPaths(root=root)


def require_public_eval_root(public_root: Union[str, Path]) -> PublicEvalPaths:
    """Resolve the public root and confirm it holds a public export (manifest present)."""
    paths = public_eval_paths(public_root, require_exists=True)
    if not paths.manifest_file.is_file():
        raise ModelConfigurationError(
            f"public evaluation root is missing {PUBLIC_MANIFEST_FILENAME}: {paths.manifest_file}. "
            "Build it first via pipelines.run_dataset_v2_public_export."
        )
    return paths
