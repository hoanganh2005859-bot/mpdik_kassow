"""Shared q_initial candidate construction + covariates for Dataset v2 trials (Phase 7).

Both the difficulty calibration (``dataset_v2/trial_calibration.py``) and the trial generator
(``dataset_v2/trial_generation.py``) build q_initial candidate pools the same way, so the
construction lives here once.

``q_initial`` is drawn ONLY from the operational joint limits, as a deterministic mixture of four
sub-pools (task section 7): global operational-interior samples, limit-aware samples,
singularity-aware diagnostic samples, and stratified joint-space samples. It is never derived from,
or perturbed around, a trajectory's ``q_reference`` (which this module never reads), never a DLS
solution, and never a frozen evaluation result. Every candidate is classified only AFTER FK is
computed and compared to a trajectory's first target pose.

Reuses ``kinematics/`` and the already-verified Tier 0 sampling constructions unchanged; never runs
DLS; never touches global ``numpy.random`` state.
"""

import zlib
from typing import Dict, List, Tuple

import numpy as np

from dataset_v2.anchor_generation import _compute_candidate_metrics
from dataset_v2.seeds import derive_seed
from dataset_v2.tier0_generation import (
    _build_singularity_candidate_pool,
    _group_mixed_near_limits,
    _group_random_interior,
)
from kinematics.model_loader import ModelContext
from kinematics.quaternion_utils import quaternion_geodesic_angle

# Per-trajectory pool seed tag (under the trials component seed). Sub-pool tags below sit under the
# per-trajectory pool seed. Difficulty tag derives a per-trial source_seed under the pool seed.
POOL_TAG = 10
SUBPOOL_INTERIOR_TAG = 1
SUBPOOL_NEAR_LIMIT_TAG = 2
SUBPOOL_SINGULAR_TAG = 3
SUBPOOL_STRATIFIED_TAG = 4


def trial_pool_seed(
    component_seed: int,
    content_hash: str,
    family: str,
    split: str,
    frozen_core_revision: int,
    frozen_challenge_revision: int,
    frozen_trial_revision: int,
) -> int:
    """Deterministic per-trajectory candidate-pool seed.

    Non-frozen trajectories key only on the trajectory content hash. ``frozen_test`` trajectories
    additionally mix in the trajectory's own frozen family revision (core or challenge) AND the
    frozen trial revision -- a separate namespace so frozen trials can never coincide with
    development/validation content or with anything observed while thresholds were calibrated.
    """
    tags = [POOL_TAG, zlib.crc32(content_hash.encode("utf-8"))]
    if split == "frozen_test":
        family_revision = frozen_core_revision if family == "core" else frozen_challenge_revision
        tags.extend([int(family_revision), int(frozen_trial_revision)])
    return derive_seed(component_seed, *tags)


def trial_source_seed(pool_seed: int, init_class_tag: int) -> int:
    """Per-trial source seed recorded in the manifest (derives from the pool seed + difficulty)."""
    return derive_seed(pool_seed, init_class_tag)

# Tier 0 near-limit / singularity sampling-policy margins (single source of truth; same values the
# challenge generator reuses via its own `_tier0_sampling_policy`).
_INTERIOR_MARGIN_RAD = 0.15
_NEAR_LIMIT_MARGIN_RAD = 0.05
_NEAR_LIMIT_BAND_RAD = 0.05


def _stratified_interior(
    rng: np.random.Generator, nq: int, count: int, lower: np.ndarray, upper: np.ndarray, margin: np.ndarray
) -> np.ndarray:
    """Per-joint stratified joint-space samples: split each joint's interior range into ``count``
    equal strata, draw one uniform point per stratum, then shuffle each joint's strata order
    independently -- good coverage without a grid's regularity."""
    lo = lower + margin
    hi = upper - margin
    q = np.empty((count, nq), dtype=np.float64)
    for j in range(nq):
        edges = np.linspace(lo[j], hi[j], count + 1)
        offsets = rng.uniform(0.0, 1.0, size=count)
        samples = edges[:-1] + offsets * (edges[1:] - edges[:-1])
        rng.shuffle(samples)
        q[:, j] = samples
    return q


def build_candidate_pool(
    model_context: ModelContext,
    pool_seed: int,
    interior_samples: int,
    near_limit_samples: int,
    singular_samples: int,
    stratified_samples: int,
    interior_margin_fraction: float,
    stratified_margin_fraction: float,
) -> Tuple[np.ndarray, List[str]]:
    """Deterministic q_initial candidate pool for one trajectory, drawn only from joint limits.

    Returns ``(q_batch [N, nq], source_pool_labels)``. Each sub-pool gets its own derived seed from
    ``pool_seed`` so the mixture is reproducible and independent of the others.
    """
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    nq = model_context.nq
    span = upper - lower

    batches: List[np.ndarray] = []
    labels: List[str] = []

    if interior_samples > 0:
        rng = np.random.default_rng(derive_seed(pool_seed, SUBPOOL_INTERIOR_TAG))
        margin = interior_margin_fraction * span
        q = _group_random_interior(rng, nq, interior_samples, lower, upper, margin)
        batches.append(q)
        labels.extend(["interior"] * q.shape[0])

    if near_limit_samples > 0:
        rng = np.random.default_rng(derive_seed(pool_seed, SUBPOOL_NEAR_LIMIT_TAG))
        q = _group_mixed_near_limits(
            rng, nq, near_limit_samples, lower, upper, _NEAR_LIMIT_MARGIN_RAD, _NEAR_LIMIT_BAND_RAD, _INTERIOR_MARGIN_RAD
        )
        batches.append(q)
        labels.extend(["near_limit"] * q.shape[0])

    if singular_samples > 0:
        rng = np.random.default_rng(derive_seed(pool_seed, SUBPOOL_SINGULAR_TAG))
        q = _build_singularity_candidate_pool(rng, nq, lower, upper, singular_samples, _INTERIOR_MARGIN_RAD)
        batches.append(q)
        labels.extend(["singular"] * q.shape[0])

    if stratified_samples > 0:
        rng = np.random.default_rng(derive_seed(pool_seed, SUBPOOL_STRATIFIED_TAG))
        margin = stratified_margin_fraction * span
        q = _stratified_interior(rng, nq, stratified_samples, lower, upper, margin)
        batches.append(q)
        labels.extend(["stratified"] * q.shape[0])

    if not batches:
        raise ValueError("trial candidate pool is empty; all sub-pool sizes are zero")

    q_batch = np.concatenate(batches, axis=0)
    return q_batch, labels


def pose_errors(
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    first_target_position: np.ndarray,
    first_target_quaternion_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-candidate Cartesian position error (m) and SO(3) geodesic orientation error (rad) to a
    single first target pose. Orientation uses the geodesic angle, never an Euler difference."""
    n = positions.shape[0]
    position_error = np.linalg.norm(positions - first_target_position[None, :], axis=1)
    orientation_error = np.empty(n, dtype=np.float64)
    for i in range(n):
        orientation_error[i] = quaternion_geodesic_angle(quaternions_wxyz[i], first_target_quaternion_wxyz)
    return position_error, orientation_error


def primary_metric(
    position_error: np.ndarray,
    orientation_error: np.ndarray,
    position_scale_m: float,
    orientation_scale_rad: float,
    position_weight: float,
    orientation_weight: float,
) -> np.ndarray:
    """Combined normalized first-target pose error (the locked primary difficulty metric)."""
    return position_weight * (position_error / position_scale_m) + orientation_weight * (
        orientation_error / orientation_scale_rad
    )


def compute_candidate_covariates(model_context: ModelContext, q_batch: np.ndarray) -> Dict[str, np.ndarray]:
    """FK/Jacobian covariates for a q_initial batch (reuses the anchor generator's implementation,
    which computes real FK position/quaternion, sigma_min/max, condition number, numerical rank,
    manipulability, normalized/absolute joint-limit margin, and controlling joint). No DLS."""
    return _compute_candidate_metrics(model_context, q_batch)
