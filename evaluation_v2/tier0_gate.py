"""Tier 0 kinematics gate for the Dataset v2 evaluation (task section 8).

Runs the FK / Jacobian / singularity validation over the dataset's Tier 0 states (which are plain
kinematic sanity configurations, not protected reference solutions) and applies the gate:
non-finite FK == 0, invalid rotations == 0, non-finite Jacobian == 0, and max Jacobian relative
error <= 1e-4. If the gate fails, the caller must not run Tier 1-4.

Reuses ``evaluation/kinematics_validation.py`` unchanged for the numerics.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from dataset_v2.locator import require_dataset_v2_root
from evaluation.kinematics_validation import (
    DEFAULT_JACOBIAN_RELATIVE_ERROR_THRESHOLD,
    compute_gate_result,
    validate_fk_states,
    validate_jacobian_states,
    validate_singularity_states,
)
from kinematics.model_loader import ModelContext, load_model_context
from utils.npz_utils import load_npz

JACOBIAN_RELATIVE_ERROR_GATE = DEFAULT_JACOBIAN_RELATIVE_ERROR_THRESHOLD  # 1e-4


def _load_singularity_threshold(meta_path: Path) -> float:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return float(meta.get("singularity_threshold", 0.03))


def run_tier0_gate(
    dataset_root,
    *,
    model_context: Optional[ModelContext] = None,
    fk_limit: Optional[int] = None,
    jacobian_limit: Optional[int] = None,
    singularity_limit: Optional[int] = None,
    output_file: Optional[Path] = None,
):
    """Compute the Tier 0 gate over the dataset's Tier 0 states.

    ``*_limit`` truncate the state sets for smoke tests only. Returns the ``Tier0GateResult``.
    """
    ds = require_dataset_v2_root(dataset_root)
    model_context = model_context or load_model_context()
    t0 = ds.tier0_validation_dir

    fk = load_npz(t0 / "fk_test_states_v2.npz")
    jac = load_npz(t0 / "jacobian_test_states_v2.npz")
    sing = load_npz(t0 / "singularity_test_states_v2.npz")
    sing_threshold = _load_singularity_threshold(t0 / "singularity_test_states_v2_metadata.json")

    def _slice(arr, n):
        return arr if n is None else arr[:n]

    fk_results = validate_fk_states(
        model_context,
        _slice(fk["q_samples"], fk_limit),
        _slice(fk["sample_id"], fk_limit),
        _slice(fk["group_id"], fk_limit),
    )
    jac_results = validate_jacobian_states(
        model_context,
        _slice(jac["q_samples"], jacobian_limit),
        _slice(jac["sample_id"], jacobian_limit),
        _slice(jac["group_id"], jacobian_limit),
        finite_difference_epsilon=_slice(jac["finite_difference_epsilon"], jacobian_limit),
        relative_error_threshold=JACOBIAN_RELATIVE_ERROR_GATE,
    )
    sing_results = validate_singularity_states(
        model_context,
        _slice(sing["q_samples"], singularity_limit),
        _slice(sing["sample_id"], singularity_limit),
        _slice(sing["group_id"], singularity_limit),
        near_singular_threshold=sing_threshold,
    )

    gate = compute_gate_result(
        fk_results, jac_results, sing_results, relative_error_threshold=JACOBIAN_RELATIVE_ERROR_GATE
    )

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(gate)
        payload["jacobian_relative_error_gate"] = JACOBIAN_RELATIVE_ERROR_GATE
        tmp = output_file.with_name(output_file.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(output_file)

    return gate
