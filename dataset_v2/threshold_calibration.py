"""Phase 2.5 threshold calibration for Dataset v2 difficulty definitions.

Derives, from deterministic pools of real KR810 joint configurations, empirical distributions of
two real quantities -- the normalized joint-limit margin
(``kinematics/joint_limit_utils.py::minimum_joint_limit_margin``) and the geometric Jacobian's
smallest singular value ``sigma_min``
(``kinematics/singularity_metrics.py::minimum_singular_value``) -- so that ``near_joint_limit``,
``near_singularity``, ``moderately_conditioned``, and ``regular`` can be defined quantitatively and
reused consistently across Tier 1 Point-IK difficulty groups, the 12 anchor configurations, and
trial difficulty covariates.

This module only classifies/reports. It never generates official Dataset v2 samples, never writes
a candidate pool to disk, and never touches ``numpy``'s global random state -- every draw comes
from an explicitly derived ``np.random.Generator`` (``generators/_common.py::derive_seed``).
Classification never uses DLS convergence, DLS final error, ``q_target``-as-solution, or
frozen-test data; it depends only on real FK/Jacobian/joint-limit quantities computed here.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from generators._common import derive_seed, get_model_context, load_dls_config
from kinematics.jacobian import geometric_jacobian_world
from kinematics.joint_limit_utils import minimum_joint_limit_margin, normalized_joint_limit_margin
from kinematics.manipulability import positional_manipulability
from kinematics.model_loader import ModelContext
from kinematics.singularity_metrics import condition_number, numerical_rank, singular_values

# Reuses Tier 0's exact uniform + elbow(q4~0) + wrist(q6~0) bias construction so the calibration's
# near-singular sufficiency check draws on the same candidate-pool logic Tier 0 already generated
# and validated in Phase 2 (dataset_v2/tier0_generation.py). Bias only *proposes* candidates --
# classification below always uses a real computed sigma_min, never the bias label itself.
from dataset_v2.tier0_generation import _build_singularity_candidate_pool

# Calibration-only seed tag. Distinct from every official component tag in
# config_templates.SEED_COMPONENT_TAGS (10/20/30/40/50/60) so a calibration run can never collide
# with or be mistaken for an official Dataset v2 generation seed.
CALIBRATION_COMPONENT_TAG = 990
GENERIC_POOL_TAG = 1
SINGULARITY_POOL_TAG = 2

DEFAULT_GENERIC_POOL_SIZE = 50000
DEFAULT_SINGULARITY_POOL_SIZE = 20000
# Tier 0's candidate-pool builder (reused below for the singularity-biased pool) takes a flat
# absolute-rad interior margin; kept here only for that call site.
DEFAULT_INTERIOR_MARGIN_RAD = 0.10
# The *generic* pool (used to derive the near_joint_limit quantile) instead uses a margin
# proportional to each joint's own half-range (1% of half-range), not one flat rad value. KR810's
# operational half-ranges differ by ~2.9x between joint_2/joint_4 (~2.18 rad) and the other five
# joints (~6.28 rad, see configs/robot_config.json); a flat absolute rad margin would let the
# wide-range joints sample much closer to their bound (in normalized terms) than the narrow-range
# ones ever could, structurally excluding joint_2/joint_4 from ever being the controlling
# near-limit joint. A range-proportional margin keeps every joint's achievable normalized-margin
# span comparable, per spec section 3's "không trộn đơn vị rad giữa các joint có range khác nhau
# mà không chuẩn hóa".
DEFAULT_INTERIOR_MARGIN_FRACTION = 0.01

QUANTILE_LEVELS = {"min": 0.0, "p01": 0.01, "p02": 0.02, "p05": 0.05, "p10": 0.10, "median": 0.50}

# Reused unchanged from v1's anchor-search predicate (generators/_trajectory_common.py::
# select_anchor, ANCHOR_SIGMA_RATIO = 3.0) and already used by Tier 0's own singularity-state
# classifier (dataset_v2/tier0_generation.py, singularity_moderate_upper_multiplier = 3.0) --
# reusing it here keeps all three call sites' regular/moderate boundary identical.
MODERATE_UPPER_MULTIPLIER = 3.0


def _redraw_duplicates(rng, builder, count, max_attempts=25):
    q = np.asarray(builder(count), dtype=np.float64)
    for _ in range(max_attempts):
        _, first_idx = np.unique(q, axis=0, return_index=True)
        if first_idx.shape[0] == q.shape[0]:
            return q
        dup_mask = np.ones(q.shape[0], dtype=bool)
        dup_mask[first_idx] = False
        n_dup = int(dup_mask.sum())
        q[dup_mask] = builder(n_dup)
    raise ValueError("could not eliminate duplicate joint states in calibration candidate pool after max redraw attempts")


def build_generic_candidate_pool(
    rng: np.random.Generator,
    model_context: ModelContext,
    pool_size: int,
    interior_margin_fraction: float = DEFAULT_INTERIOR_MARGIN_FRACTION,
) -> np.ndarray:
    """Uniform-interior candidate pool over the real operational joint limits.

    Sampling margin is a *fraction of each joint's own half-range*, not one flat rad value -- see
    ``DEFAULT_INTERIOR_MARGIN_FRACTION`` for why a flat rad margin would bias which joint can ever
    be the near-limit controlling joint. This is the "generic" population used to derive
    quantile-based thresholds (near_joint_limit) and to characterize the natural/unbiased
    distribution of sigma_min, condition number, and manipulability -- deterministic given
    ``rng``, duplicate-free by construction.
    """
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    half_range = (upper - lower) / 2.0
    margin = interior_margin_fraction * half_range
    nq = model_context.nq

    def builder(count):
        return rng.uniform(lower + margin, upper - margin, size=(count, nq))

    return _redraw_duplicates(rng, builder, pool_size)


def build_singularity_biased_pool(
    rng: np.random.Generator,
    model_context: ModelContext,
    pool_size: int,
    interior_margin_rad: float = DEFAULT_INTERIOR_MARGIN_RAD,
) -> np.ndarray:
    """Uniform + elbow/wrist-biased candidate pool (Tier 0's construction, reused unchanged).

    Used only to verify near-singular/moderately-conditioned candidate *sufficiency*; the
    near_joint_limit / regular quantile thresholds themselves are derived from the generic pool.
    """
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    nq = model_context.nq
    q = _build_singularity_candidate_pool(rng, nq, lower, upper, pool_size, interior_margin_rad)
    _, first_idx = np.unique(q, axis=0, return_index=True)
    if first_idx.shape[0] != q.shape[0]:
        q = q[np.sort(first_idx)]
    return q


def _absolute_min_margin_rad(q_row: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    distance = np.minimum(q_row - lower, upper - q_row)
    controlling_joint = int(np.argmin(distance))
    return float(distance[controlling_joint]), controlling_joint


@dataclass
class CandidatePoolMetrics:
    q_samples: np.ndarray
    normalized_min_margin: np.ndarray
    normalized_margin_controlling_joint: np.ndarray
    absolute_min_margin_rad: np.ndarray
    absolute_margin_controlling_joint: np.ndarray
    sigma_min: np.ndarray
    sigma_max: np.ndarray
    condition_number: np.ndarray
    numerical_rank: np.ndarray
    manipulability: np.ndarray
    fk_position: np.ndarray


def compute_candidate_metrics(model_context: ModelContext, q_samples: np.ndarray) -> CandidatePoolMetrics:
    """Compute every real, FK/Jacobian/joint-limit-derived quantity per candidate.

    Never uses DLS, never assumes a label -- every field is a direct computation against the
    existing (unmodified) ``kinematics/`` implementation.
    """
    from kinematics.forward_kinematics import forward_kinematics

    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    n = q_samples.shape[0]

    normalized_margin = np.empty(n, dtype=np.float64)
    normalized_controlling = np.empty(n, dtype=np.int32)
    absolute_margin = np.empty(n, dtype=np.float64)
    absolute_controlling = np.empty(n, dtype=np.int32)
    sigma_min = np.empty(n, dtype=np.float64)
    sigma_max = np.empty(n, dtype=np.float64)
    cond = np.empty(n, dtype=np.float64)
    rank = np.empty(n, dtype=np.int32)
    manip = np.empty(n, dtype=np.float64)
    fk_position = np.empty((n, 3), dtype=np.float64)

    data = model_context.new_data()
    for i in range(n):
        q = q_samples[i]
        per_joint_normalized = normalized_joint_limit_margin(q, lower, upper)
        normalized_controlling[i] = int(np.argmin(per_joint_normalized))
        normalized_margin[i] = float(per_joint_normalized[normalized_controlling[i]])
        absolute_margin[i], absolute_controlling[i] = _absolute_min_margin_rad(q, lower, upper)

        J = geometric_jacobian_world(model_context, q, data=data)
        sv = singular_values(J)
        sigma_min[i] = float(sv[-1])
        sigma_max[i] = float(sv[0])
        cond[i] = condition_number(J)
        rank[i] = numerical_rank(J)
        manip[i] = positional_manipulability(J)

        fk = forward_kinematics(model_context, q, data=data)
        fk_position[i] = fk.position

    return CandidatePoolMetrics(
        q_samples=q_samples,
        normalized_min_margin=normalized_margin,
        normalized_margin_controlling_joint=normalized_controlling,
        absolute_min_margin_rad=absolute_margin,
        absolute_margin_controlling_joint=absolute_controlling,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        condition_number=cond,
        numerical_rank=rank,
        manipulability=manip,
        fk_position=fk_position,
    )


def summarize_distribution(values: np.ndarray) -> Dict[str, float]:
    summary = {name: float(np.quantile(values, level)) for name, level in QUANTILE_LEVELS.items()}
    summary["max"] = float(np.max(values))
    summary["count"] = int(values.shape[0])
    return summary


def controlling_joint_histogram(controlling_joint: np.ndarray, nq: int) -> Dict[int, int]:
    return {j: int(np.sum(controlling_joint == j)) for j in range(nq)}


def count_below(values: np.ndarray, threshold: float) -> int:
    return int(np.sum(values <= threshold))


def count_in_range(values: np.ndarray, low: float, high: float) -> int:
    return int(np.sum((values > low) & (values <= high)))


def count_above(values: np.ndarray, threshold: float) -> int:
    return int(np.sum(values > threshold))


@dataclass
class ThresholdCalibrationResult:
    calibration_seed: int
    generic_pool_seed: int
    singularity_pool_seed: int
    generic_pool_size: int
    singularity_pool_size: int
    generic_metrics: CandidatePoolMetrics
    singularity_metrics: CandidatePoolMetrics
    normalized_margin_distribution: Dict[str, float]
    absolute_margin_distribution: Dict[str, float]
    sigma_min_distribution_generic: Dict[str, float]
    sigma_min_distribution_biased: Dict[str, float]
    near_joint_limit_normalized_threshold: float
    near_joint_limit_absolute_threshold_rad_diagnostic: float
    near_singularity_sigma_threshold: float
    near_singularity_threshold_source: str
    moderately_conditioned_upper_bound: float
    regular_min_sigma_min: float
    regular_min_normalized_margin: float
    candidate_counts: Dict[str, int]
    controlling_joint_normalized: Dict[int, int]
    controlling_joint_absolute: Dict[int, int]


def calibrate(
    calibration_seed: int,
    generic_pool_size: int = DEFAULT_GENERIC_POOL_SIZE,
    singularity_pool_size: int = DEFAULT_SINGULARITY_POOL_SIZE,
    interior_margin_fraction: float = DEFAULT_INTERIOR_MARGIN_FRACTION,
    singularity_pool_interior_margin_rad: float = DEFAULT_INTERIOR_MARGIN_RAD,
    near_joint_limit_quantile: float = 0.10,
    model_context: Optional[ModelContext] = None,
) -> ThresholdCalibrationResult:
    """Run the full Phase 2.5 calibration pass and return every distribution/threshold needed
    for ``docs/V2_THRESHOLD_CALIBRATION.md`` and ``configs/difficulty_thresholds.json``.
    """
    model_context = model_context if model_context is not None else get_model_context()

    calibration_component_seed = derive_seed(calibration_seed, CALIBRATION_COMPONENT_TAG)
    generic_pool_seed = derive_seed(calibration_component_seed, GENERIC_POOL_TAG)
    singularity_pool_seed = derive_seed(calibration_component_seed, SINGULARITY_POOL_TAG)

    generic_rng = np.random.default_rng(generic_pool_seed)
    singularity_rng = np.random.default_rng(singularity_pool_seed)

    generic_q = build_generic_candidate_pool(generic_rng, model_context, generic_pool_size, interior_margin_fraction)
    singularity_q = build_singularity_biased_pool(
        singularity_rng, model_context, singularity_pool_size, singularity_pool_interior_margin_rad
    )

    generic_metrics = compute_candidate_metrics(model_context, generic_q)
    singularity_metrics = compute_candidate_metrics(model_context, singularity_q)

    normalized_margin_distribution = summarize_distribution(generic_metrics.normalized_min_margin)
    absolute_margin_distribution = summarize_distribution(generic_metrics.absolute_min_margin_rad)
    sigma_min_distribution_generic = summarize_distribution(generic_metrics.sigma_min)
    sigma_min_distribution_biased = summarize_distribution(singularity_metrics.sigma_min)

    near_joint_limit_normalized_threshold = float(
        np.quantile(generic_metrics.normalized_min_margin, near_joint_limit_quantile)
    )
    near_joint_limit_absolute_threshold_rad_diagnostic = float(
        np.quantile(generic_metrics.absolute_min_margin_rad, near_joint_limit_quantile)
    )

    dls_config = load_dls_config()
    near_singularity_sigma_threshold = float(dls_config["singularity_sigma_threshold"])
    near_singularity_threshold_source = (
        "repo root configs/dls_config.json:singularity_sigma_threshold (v1 shared DLS config, "
        "reused unchanged; already reused by generators/_trajectory_common.py::select_anchor's "
        "ANCHOR_SIGMA_RATIO=3.0 predicate and by dataset_v2/tier0_generation.py's Tier 0 "
        "singularity-state classifier)"
    )
    moderately_conditioned_upper_bound = near_singularity_sigma_threshold * MODERATE_UPPER_MULTIPLIER
    regular_min_sigma_min = moderately_conditioned_upper_bound
    regular_min_normalized_margin = near_joint_limit_normalized_threshold

    candidate_counts = {
        "generic_pool_size": int(generic_metrics.normalized_min_margin.shape[0]),
        "singularity_pool_size": int(singularity_metrics.sigma_min.shape[0]),
        "near_joint_limit_in_generic_pool": count_below(generic_metrics.normalized_min_margin, near_joint_limit_normalized_threshold),
        "near_singularity_in_generic_pool": count_below(generic_metrics.sigma_min, near_singularity_sigma_threshold),
        "near_singularity_in_biased_pool": count_below(singularity_metrics.sigma_min, near_singularity_sigma_threshold),
        "moderately_conditioned_in_biased_pool": count_in_range(
            singularity_metrics.sigma_min, near_singularity_sigma_threshold, moderately_conditioned_upper_bound
        ),
        "regular_sigma_in_biased_pool": count_above(singularity_metrics.sigma_min, moderately_conditioned_upper_bound),
        "regular_combined_in_generic_pool": int(
            np.sum(
                (generic_metrics.normalized_min_margin > regular_min_normalized_margin)
                & (generic_metrics.sigma_min > regular_min_sigma_min)
            )
        ),
    }

    controlling_joint_normalized = controlling_joint_histogram(
        generic_metrics.normalized_margin_controlling_joint[
            generic_metrics.normalized_min_margin <= near_joint_limit_normalized_threshold
        ],
        model_context.nq,
    )
    controlling_joint_absolute = controlling_joint_histogram(
        generic_metrics.absolute_margin_controlling_joint[
            generic_metrics.absolute_min_margin_rad <= near_joint_limit_absolute_threshold_rad_diagnostic
        ],
        model_context.nq,
    )

    return ThresholdCalibrationResult(
        calibration_seed=calibration_seed,
        generic_pool_seed=generic_pool_seed,
        singularity_pool_seed=singularity_pool_seed,
        generic_pool_size=generic_pool_size,
        singularity_pool_size=singularity_pool_size,
        generic_metrics=generic_metrics,
        singularity_metrics=singularity_metrics,
        normalized_margin_distribution=normalized_margin_distribution,
        absolute_margin_distribution=absolute_margin_distribution,
        sigma_min_distribution_generic=sigma_min_distribution_generic,
        sigma_min_distribution_biased=sigma_min_distribution_biased,
        near_joint_limit_normalized_threshold=near_joint_limit_normalized_threshold,
        near_joint_limit_absolute_threshold_rad_diagnostic=near_joint_limit_absolute_threshold_rad_diagnostic,
        near_singularity_sigma_threshold=near_singularity_sigma_threshold,
        near_singularity_threshold_source=near_singularity_threshold_source,
        moderately_conditioned_upper_bound=moderately_conditioned_upper_bound,
        regular_min_sigma_min=regular_min_sigma_min,
        regular_min_normalized_margin=regular_min_normalized_margin,
        candidate_counts=candidate_counts,
        controlling_joint_normalized=controlling_joint_normalized,
        controlling_joint_absolute=controlling_joint_absolute,
    )


CLASSIFICATION_PRIORITY = [
    "near_singularity",
    "near_joint_limit",
    "large_orientation_change",
    "far_target",
    "medium_target",
    "near_target",
]


def classify_single(
    normalized_margin: float,
    sigma_min: float,
    near_joint_limit_threshold: float,
    near_singularity_threshold: float,
    moderately_conditioned_upper_bound: float,
) -> Dict[str, bool]:
    """Classify one configuration's margin/singularity state into non-overlapping bins.

    ``near_joint_limit`` and the sigma_min-based tri-state are independent axes; a "regular
    anchor" (spec section 6) additionally requires both "not near limit" and "regular sigma".
    """
    is_near_joint_limit = normalized_margin <= near_joint_limit_threshold
    is_near_singular = sigma_min <= near_singularity_threshold
    is_moderately_conditioned = (sigma_min > near_singularity_threshold) and (sigma_min <= moderately_conditioned_upper_bound)
    is_regular_sigma = sigma_min > moderately_conditioned_upper_bound
    is_regular = is_regular_sigma and not is_near_joint_limit
    return {
        "near_joint_limit": bool(is_near_joint_limit),
        "near_singular": bool(is_near_singular),
        "moderately_conditioned": bool(is_moderately_conditioned),
        "regular_sigma": bool(is_regular_sigma),
        "regular": bool(is_regular),
    }
