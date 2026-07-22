"""Independent validator for Dataset v2 Tier 0 (FK/Jacobian/singularity validation states).

Reuses ``evaluation/kinematics_validation.py`` (the already-verified Tier 0 v1 validation pass)
for the actual FK/Jacobian/singularity checks -- this module only adds v2-specific structural
checks (exact counts, group counts, duplicate detection, classification-vs-threshold consistency)
on top, and never reimplements FK/Jacobian/singularity math.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from dataset_v2.locator import require_dataset_v2_root
from dataset_v2.tier0_generation import (
    FK_GROUPS,
    FK_NPZ_NAME,
    JACOBIAN_GROUPS,
    JACOBIAN_NPZ_NAME,
    SINGULARITY_GROUPS,
    SINGULARITY_METADATA_NAME,
    SINGULARITY_NPZ_NAME,
    load_singularity_threshold,
)
from evaluation.kinematics_validation import (
    DEFAULT_JACOBIAN_RELATIVE_ERROR_THRESHOLD,
    validate_fk_states,
    validate_jacobian_states,
)
from kinematics.model_loader import ModelContext, load_model_context
from utils.npz_utils import load_npz


@dataclass
class Tier0ValidationReport:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    fk_count: int = 0
    fk_group_counts: Dict[str, int] = field(default_factory=dict)
    jacobian_count: int = 0
    jacobian_group_counts: Dict[str, int] = field(default_factory=dict)
    singularity_count: int = 0
    singularity_group_counts: Dict[str, int] = field(default_factory=dict)
    max_jacobian_relative_error: float = float("nan")


def _check_duplicates(q_samples: np.ndarray, label: str, reasons: List[str]) -> None:
    _, first_idx = np.unique(q_samples, axis=0, return_index=True)
    n_dup = q_samples.shape[0] - first_idx.shape[0]
    if n_dup > 0:
        reasons.append(f"{label}: {n_dup} duplicate joint state(s) found")


def _check_limits(q_samples: np.ndarray, lower, upper, label: str, reasons: List[str]) -> None:
    if not np.all(np.isfinite(q_samples)):
        reasons.append(f"{label}: non-finite joint values found")
        return
    if np.any(q_samples < lower) or np.any(q_samples > upper):
        reasons.append(f"{label}: one or more states violate operational joint limits")


def _check_group_counts(
    group_id_arr: np.ndarray, groups: Dict[int, str], expected: Optional[Dict[str, int]], label: str, reasons: List[str]
) -> Dict[str, int]:
    counts = {name: int(np.sum(group_id_arr == gid)) for gid, name in groups.items()}
    if expected is not None:
        for name, expected_count in expected.items():
            if counts.get(name, 0) != expected_count:
                reasons.append(f"{label}: group '{name}' has {counts.get(name, 0)} state(s), expected {expected_count}")
    return counts


def validate_fk_npz(
    model_context: ModelContext,
    arrays: Dict[str, np.ndarray],
    expected_total: Optional[int] = None,
    expected_group_counts: Optional[Dict[str, int]] = None,
    rotation_tolerance: float = 1e-6,
):
    reasons: List[str] = []
    q = arrays["q_samples"]
    total = int(q.shape[0])
    if expected_total is not None and total != expected_total:
        reasons.append(f"FK: expected {expected_total} states, found {total}")

    group_counts = _check_group_counts(arrays["group_id"], FK_GROUPS, expected_group_counts, "FK", reasons)
    _check_limits(q, model_context.operational_lower_rad, model_context.operational_upper_rad, "FK", reasons)
    _check_duplicates(q, "FK", reasons)

    results = validate_fk_states(model_context, q, arrays["sample_id"], arrays["group_id"], rotation_tolerance=rotation_tolerance)
    failures = [r for r in results if not r.passed]
    if failures:
        reasons.append(f"FK: {len(failures)} sample(s) failed FK validation (rotation/finite/determinism)")

    return total, group_counts, reasons


def validate_jacobian_npz(
    model_context: ModelContext,
    arrays: Dict[str, np.ndarray],
    expected_total: Optional[int] = None,
    expected_group_counts: Optional[Dict[str, int]] = None,
    relative_error_threshold: float = DEFAULT_JACOBIAN_RELATIVE_ERROR_THRESHOLD,
):
    reasons: List[str] = []
    q = arrays["q_samples"]
    total = int(q.shape[0])
    if expected_total is not None and total != expected_total:
        reasons.append(f"Jacobian: expected {expected_total} states, found {total}")

    group_counts = _check_group_counts(arrays["group_id"], JACOBIAN_GROUPS, expected_group_counts, "Jacobian", reasons)
    _check_limits(q, model_context.operational_lower_rad, model_context.operational_upper_rad, "Jacobian", reasons)
    _check_duplicates(q, "Jacobian", reasons)

    results = validate_jacobian_states(
        model_context,
        q,
        arrays["sample_id"],
        arrays["group_id"],
        finite_difference_epsilon=arrays["finite_difference_epsilon"],
        relative_error_threshold=relative_error_threshold,
    )
    rel_errors = np.array([r.relative_error for r in results], dtype=np.float64)
    max_rel_error = float(np.max(rel_errors)) if rel_errors.size else float("nan")
    failures = [r for r in results if not r.passed]
    if failures:
        reasons.append(
            f"Jacobian: {len(failures)} sample(s) exceeded relative-error threshold "
            f"{relative_error_threshold:.1e} (max={max_rel_error:.3e})"
        )

    if "sigma_min" in arrays:
        recomputed_sigma_min = np.array([r.sigma_min for r in results], dtype=np.float64)
        mismatch = np.abs(arrays["sigma_min"] - recomputed_sigma_min) > 1e-9
        if np.any(mismatch):
            reasons.append(f"Jacobian: stored sigma_min does not match recomputed value for {int(np.sum(mismatch))} sample(s)")

    return total, group_counts, reasons, max_rel_error


def validate_singularity_npz(
    arrays: Dict[str, np.ndarray],
    expected_total: Optional[int] = None,
    expected_group_counts: Optional[Dict[str, int]] = None,
    threshold: Optional[float] = None,
    moderate_upper: Optional[float] = None,
):
    reasons: List[str] = []
    q = arrays["q_samples"]
    total = int(q.shape[0])
    if expected_total is not None and total != expected_total:
        reasons.append(f"Singularity: expected {expected_total} states, found {total}")

    group_counts = _check_group_counts(arrays["group_id"], SINGULARITY_GROUPS, expected_group_counts, "Singularity", reasons)
    _check_duplicates(q, "Singularity", reasons)

    sigma_min = arrays["sigma_min"]
    condition = arrays["condition_number"]
    if not np.all(np.isfinite(sigma_min)) or np.any(sigma_min < 0.0):
        reasons.append("Singularity: sigma_min must be finite and non-negative")
    if np.any(np.isnan(condition)):
        reasons.append("Singularity: condition_number must never be NaN (inf is allowed for a singular Jacobian)")

    if threshold is not None and moderate_upper is not None:
        group_id = arrays["group_id"]
        name_to_id = {name: gid for gid, name in SINGULARITY_GROUPS.items()}
        near_id = name_to_id["near_singular"]
        moderate_id = name_to_id["moderately_conditioned"]
        regular_id = name_to_id["regular"]

        near_sigma = sigma_min[group_id == near_id]
        if near_sigma.size and np.any(near_sigma > threshold):
            reasons.append("Singularity: a 'near_singular' sample has sigma_min above the singularity threshold")

        moderate_sigma = sigma_min[group_id == moderate_id]
        if moderate_sigma.size and np.any((moderate_sigma <= threshold) | (moderate_sigma > moderate_upper)):
            reasons.append("Singularity: a 'moderately_conditioned' sample falls outside the moderate sigma_min band")

        regular_sigma = sigma_min[group_id == regular_id]
        if regular_sigma.size and np.any(regular_sigma <= moderate_upper):
            reasons.append("Singularity: a 'regular' sample does not have sufficient sigma_min margin")

    return total, group_counts, reasons


def validate_tier0(
    dataset_root,
    model_context: Optional[ModelContext] = None,
    expected_group_counts: Optional[Dict[str, Dict[str, int]]] = None,
    expected_totals: Optional[Dict[str, int]] = None,
    full_counts: bool = True,
    relative_error_threshold: float = DEFAULT_JACOBIAN_RELATIVE_ERROR_THRESHOLD,
) -> Tier0ValidationReport:
    """Validate the on-disk Tier 0 v2 NPZ files under ``dataset_root``.

    ``full_counts=True`` (default) checks against the locked 1000/1000/600 counts and their
    even 200/200/200-per-group splits. Pass ``full_counts=False`` for reduced fixtures; ``
    expected_totals`` (keys ``"fk"``/``"jacobian"``/``"singularity"``) and/or
    ``expected_group_counts`` (same keys, each a group-name -> count dict) then let the caller
    still assert exact counts even off the locked full mode. Either is optional and independently
    skippable (``None`` skips that particular check).
    """
    paths = require_dataset_v2_root(dataset_root)
    tier0_dir = paths.tier0_validation_dir
    model_context = model_context if model_context is not None else load_model_context()

    reasons: List[str] = []

    fk_arrays = load_npz(tier0_dir / FK_NPZ_NAME)
    jacobian_arrays = load_npz(tier0_dir / JACOBIAN_NPZ_NAME)
    singularity_arrays = load_npz(tier0_dir / SINGULARITY_NPZ_NAME)

    if full_counts:
        expected_fk_total, expected_jacobian_total, expected_singularity_total = 1000, 1000, 600
        expected_fk_groups = {name: 200 for name in FK_GROUPS.values()}
        expected_jacobian_groups = {name: 200 for name in JACOBIAN_GROUPS.values()}
        expected_singularity_groups = {name: 200 for name in SINGULARITY_GROUPS.values()}
    else:
        expected_fk_total = (expected_totals or {}).get("fk")
        expected_jacobian_total = (expected_totals or {}).get("jacobian")
        expected_singularity_total = (expected_totals or {}).get("singularity")
        expected_fk_groups = (expected_group_counts or {}).get("fk")
        expected_jacobian_groups = (expected_group_counts or {}).get("jacobian")
        expected_singularity_groups = (expected_group_counts or {}).get("singularity")

    fk_total, fk_group_counts, fk_reasons = validate_fk_npz(model_context, fk_arrays, expected_fk_total, expected_fk_groups)
    reasons.extend(fk_reasons)

    jacobian_total, jacobian_group_counts, jacobian_reasons, max_rel_error = validate_jacobian_npz(
        model_context, jacobian_arrays, expected_jacobian_total, expected_jacobian_groups, relative_error_threshold
    )
    reasons.extend(jacobian_reasons)

    metadata_path = tier0_dir / SINGULARITY_METADATA_NAME
    if metadata_path.is_file():
        singularity_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        threshold = float(singularity_metadata["singularity_threshold"])
        moderate_upper = float(singularity_metadata["moderately_conditioned_upper_bound"])
    else:
        threshold, _source = load_singularity_threshold()
        moderate_upper = None

    singularity_total, singularity_group_counts, singularity_reasons = validate_singularity_npz(
        singularity_arrays, expected_singularity_total, expected_singularity_groups, threshold, moderate_upper
    )
    reasons.extend(singularity_reasons)

    return Tier0ValidationReport(
        passed=len(reasons) == 0,
        reasons=reasons,
        fk_count=fk_total,
        fk_group_counts=fk_group_counts,
        jacobian_count=jacobian_total,
        jacobian_group_counts=jacobian_group_counts,
        singularity_count=singularity_total,
        singularity_group_counts=singularity_group_counts,
        max_jacobian_relative_error=max_rel_error,
    )
