"""Evaluation lock bundle assembly (task section 12).

After a selected config has been confirmed once on validation, this builds the lock bundle: the
resolved config, its SHA256, the code / dataset / public / protected / environment fingerprints,
the exact command, and the expected (but NOT executed) frozen-test workload. ``lock_status`` is
always ``candidate_locked_pending_commit`` -- committing is a separate, explicit step.

The expected frozen workload is stated for transparency only; nothing here runs frozen_test.
"""

import json
from pathlib import Path

from evaluation_v2 import fingerprints
from evaluation_v2.candidate_configs import CandidateConfig, pre_registration_record

# Expected frozen-test workload (task section 12). NOT executed here.
EXPECTED_FROZEN_WORKLOAD = {
    "point_ik_solves": 3600,
    "trajectory_waypoint_solves": 210 * 2 * 400,  # 168000
    "total_dls_calls": 3600 + 210 * 2 * 400,       # 171600
    "note": "expected frozen_test workload; frozen_test is NOT run in Phase 8A",
}


def build_lock_bundle(
    selected_candidate: CandidateConfig,
    *,
    dataset_root,
    public_root,
    protected_root,
    development_summary: dict,
    validation_summary: dict,
    proposed_frozen_command: str,
) -> dict:
    """Assemble the full evaluation lock bundle dict for a selected candidate."""
    resolved = {
        "candidate": selected_candidate.to_record(),
        "reporting_and_convergence": pre_registration_record()["reporting_thresholds"],
    }
    config_sha256 = fingerprints.config_fingerprint(resolved)

    dataset_fp = fingerprints.directory_fingerprint(Path(dataset_root))
    public_fp = fingerprints.directory_fingerprint(Path(public_root))
    protected_fp = fingerprints.directory_fingerprint(Path(protected_root))

    return {
        "lock_status": "candidate_locked_pending_commit",
        "selected_candidate_id": selected_candidate.candidate_id,
        "resolved_evaluation_config": resolved,
        "config_sha256": config_sha256,
        "code_fingerprint": fingerprints.code_fingerprint(),
        "dataset_fingerprint": dataset_fp["sha256"],
        "public_bundle_fingerprint": public_fp["sha256"],
        "protected_bundle_fingerprint": protected_fp["sha256"],
        "environment_fingerprint": fingerprints.environment_fingerprint(),
        "development_summary": development_summary,
        "validation_summary": validation_summary,
        "expected_frozen_workload": EXPECTED_FROZEN_WORKLOAD,
        "proposed_frozen_command": proposed_frozen_command,
    }


def write_lock_bundle(output_path: Path, bundle: dict) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    tmp.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(output_path)
