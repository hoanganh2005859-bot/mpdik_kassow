"""Dataset v2 public and protected trajectory loaders (Phase 7, task section 5).

Trajectory canonical NPZ files carry BOTH public geometry (target poses, timing, arc-length /
angular-displacement metadata) AND protected reconstruction evidence -- notably ``q_reference``, the
DLS reference configuration solved at generation time. Tier 2-4 evaluation must never see
``q_reference`` (it would leak a reference IK solution the solver under test is meant to find), so
access is split into two explicit levels:

* :func:`load_public_trajectory` returns ONLY the public fields an evaluator may use (id / split /
  family, target poses, waypoint / timing / path metadata, public geometry metadata). It provably
  cannot return ``q_reference`` or any other protected array.
* :func:`load_protected_trajectory` additionally exposes ``q_reference`` and the reference-start
  configuration; it is for generation, validation, reachability evidence, and trial difficulty
  diagnostics only -- never wired into any public/evaluation API.

Neither loader runs DLS; neither uses global ``numpy.random`` state.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from dataset_v2.locator import require_dataset_v2_root
from dataset_v2.trajectory_catalog import load_combined_catalog
from utils.npz_utils import load_npz

#: Arrays inside a trajectory NPZ that the public loader must NEVER expose. Anything not listed
#: here is considered public geometry/timing metadata.
PROTECTED_ARRAY_KEYS = frozenset(
    {
        "q_reference",
        "q_source_reference",
        "position_reconstruction_error_m",
        "orientation_reconstruction_error_rad",
        "waypoint_reachable",
    }
)

#: Scalar metadata fields the public loader may surface (drawn from the combined catalog row).
PUBLIC_METADATA_FIELDS = (
    "trajectory_id",
    "split",
    "family",
    "shape",
    "orientation_mode",
    "anchor_id",
    "anchor_class",
    "challenge_family",
    "canonical_waypoint_count",
    "source_waypoint_count",
)


@dataclass
class PublicTrajectory:
    """Everything an evaluator may see about a trajectory -- and nothing protected."""

    trajectory_id: str
    split: str
    family: str
    metadata: Dict[str, str] = field(default_factory=dict)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def target_position(self) -> np.ndarray:
        return self.arrays["target_position"]

    @property
    def target_quaternion(self) -> np.ndarray:
        return self.arrays["target_quaternion"]

    @property
    def first_target_position(self) -> np.ndarray:
        return self.arrays["target_position"][0]

    @property
    def first_target_quaternion_wxyz(self) -> np.ndarray:
        return self.arrays["target_quaternion"][0]

    def public_field_names(self) -> List[str]:
        return sorted(self.arrays.keys())


@dataclass
class ProtectedTrajectory:
    """Internal-only view: public geometry PLUS the protected reference evidence."""

    trajectory_id: str
    split: str
    family: str
    metadata: Dict[str, str] = field(default_factory=dict)
    canonical: Dict[str, np.ndarray] = field(default_factory=dict)
    source: Optional[Dict[str, np.ndarray]] = None

    @property
    def q_reference(self) -> np.ndarray:
        return self.canonical["q_reference"]

    @property
    def q_reference_start(self) -> np.ndarray:
        """The protected reference configuration at canonical waypoint 0."""
        return self.canonical["q_reference"][0]

    @property
    def first_target_position(self) -> np.ndarray:
        return self.canonical["target_position"][0]

    @property
    def first_target_quaternion_wxyz(self) -> np.ndarray:
        return self.canonical["target_quaternion"][0]


def _catalog_index(dataset_root) -> Dict[str, dict]:
    return {row["trajectory_id"]: row for row in load_combined_catalog(dataset_root)}


def _resolve_row(dataset_root, trajectory_id: str, catalog_row: Optional[dict]) -> dict:
    if catalog_row is not None:
        return catalog_row
    index = _catalog_index(dataset_root)
    if trajectory_id not in index:
        raise KeyError(f"trajectory_id '{trajectory_id}' not found in the combined catalog")
    return index[trajectory_id]


def load_public_trajectory(dataset_root, trajectory_id: str, catalog_row: Optional[dict] = None) -> PublicTrajectory:
    """Load a trajectory's PUBLIC representation only -- no ``q_reference``, no reconstruction
    evidence, no reachability flags. Protected arrays are stripped before the object is built, so
    there is no code path by which a caller can reach them through this function.
    """
    paths = require_dataset_v2_root(dataset_root)
    row = _resolve_row(paths.root, trajectory_id, catalog_row)
    canonical_path = paths.root / row["canonical_path"]
    raw = load_npz(canonical_path)

    public_arrays = {name: arr for name, arr in raw.items() if name not in PROTECTED_ARRAY_KEYS}
    # Defensive: guarantee nothing protected slipped through.
    leaked = set(public_arrays) & PROTECTED_ARRAY_KEYS
    if leaked:
        raise AssertionError(f"public trajectory loader would leak protected arrays: {sorted(leaked)}")

    metadata = {field: row.get(field, "") for field in PUBLIC_METADATA_FIELDS}
    return PublicTrajectory(
        trajectory_id=row["trajectory_id"],
        split=row["split"],
        family=row["family"],
        metadata=metadata,
        arrays=public_arrays,
    )


def load_protected_trajectory(
    dataset_root, trajectory_id: str, catalog_row: Optional[dict] = None, include_source: bool = False
) -> ProtectedTrajectory:
    """Load a trajectory's full representation, INCLUDING the protected ``q_reference`` arrays.

    For generation / validation / reachability-evidence / trial-difficulty-diagnostic use only.
    Never expose the returned object through a public/evaluation API.
    """
    paths = require_dataset_v2_root(dataset_root)
    row = _resolve_row(paths.root, trajectory_id, catalog_row)
    canonical = load_npz(paths.root / row["canonical_path"])
    source = load_npz(paths.root / row["source_path"]) if include_source else None

    metadata = {field: row.get(field, "") for field in PUBLIC_METADATA_FIELDS}
    return ProtectedTrajectory(
        trajectory_id=row["trajectory_id"],
        split=row["split"],
        family=row["family"],
        metadata=metadata,
        canonical=canonical,
        source=source,
    )
