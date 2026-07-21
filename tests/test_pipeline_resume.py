"""Integration test for --resume behavior of pipelines.run_tier0_to_tier4.

Checks three things end to end (using each tier's own output file mtime as the "was this tier
recomputed" signal, since a valid --resume reuses the file on disk untouched while a rerun always
rewrites it via the atomic temp-file-then-replace path in utils.result_logger):

1. A second run with identical arguments and --resume skips every tier (all mtimes unchanged).
2. Deleting a tier's required output file forces that tier (and its downstream dependents,
   Tier 3/4 depending on Tier 2) to recompute, while unaffected upstream tiers still reuse their
   existing output.
3. Changing a CLI argument that only affects Tier 2's input signature forces Tier 2 (and its
   dependents) to recompute while Tier 0/Tier 1 -- whose signatures do not depend on it -- are
   still reused, and the waypoint row count never duplicates across the rerun.
"""

import pandas as pd

from pipelines.run_tier0_to_tier4 import main

_BASE_ARGS = [
    "--preset", "smoke",
    "--point-sample-limit", "12",
    "--trajectory-ids", "line_fixed_orientation",
    "--trial-limit", "2",
    "--waypoint-limit", "25",
    "--no-plots",
]

_MTIME_PROBES = {
    "tier0": "tier0_kinematics/summary.json",
    "tier1": "tier1_point_dls/metrics_overall.json",
    "tier2": "tier2_sequential_dls/waypoint_results.csv",
    "tier3": "tier3_trajectory_tracking/trajectory_metrics.csv",
    "tier4": "tier4_joint_feasibility/smoothness_metrics.csv",
}


def _mtimes(output_dir):
    return {tier: (output_dir / relpath).stat().st_mtime_ns for tier, relpath in _MTIME_PROBES.items()}


def test_resume_skips_every_tier_when_nothing_changed(tmp_path):
    output_dir = tmp_path / "run"

    exit_code = main(["--output", str(output_dir)] + _BASE_ARGS)
    assert exit_code == 0
    first_mtimes = _mtimes(output_dir)
    first_row_count = len(pd.read_csv(output_dir / "tier2_sequential_dls" / "waypoint_results.csv"))

    exit_code = main(["--output", str(output_dir), "--resume"] + _BASE_ARGS)
    assert exit_code == 0
    second_mtimes = _mtimes(output_dir)
    second_row_count = len(pd.read_csv(output_dir / "tier2_sequential_dls" / "waypoint_results.csv"))

    assert first_mtimes == second_mtimes, "an unchanged --resume run must not recompute any tier"
    assert first_row_count == second_row_count


def test_resume_recomputes_only_tiers_with_missing_required_output(tmp_path):
    output_dir = tmp_path / "run"
    main(["--output", str(output_dir)] + _BASE_ARGS)
    first_mtimes = _mtimes(output_dir)

    # Simulate corrupted/incomplete Tier 2 output: a required file is missing.
    (output_dir / "tier2_sequential_dls" / "trajectory_trial_summaries.csv").unlink()

    exit_code = main(["--output", str(output_dir), "--resume"] + _BASE_ARGS)
    assert exit_code == 0
    second_mtimes = _mtimes(output_dir)

    assert second_mtimes["tier0"] == first_mtimes["tier0"], "tier0 output was valid and must be reused"
    assert second_mtimes["tier1"] == first_mtimes["tier1"], "tier1 output was valid and must be reused"
    assert second_mtimes["tier2"] != first_mtimes["tier2"], "tier2 was missing a required file and must rerun"
    assert second_mtimes["tier3"] != first_mtimes["tier3"], "tier3 depends on tier2 and must rerun"
    assert second_mtimes["tier4"] != first_mtimes["tier4"], "tier4 depends on tier2 and must rerun"

    waypoint_df = pd.read_csv(output_dir / "tier2_sequential_dls" / "waypoint_results.csv")
    assert waypoint_df.duplicated(subset=["trial_id", "method", "waypoint_id"]).sum() == 0


def test_resume_recomputes_tier2_and_dependents_when_cli_input_changes(tmp_path):
    output_dir = tmp_path / "run"
    main(["--output", str(output_dir)] + _BASE_ARGS)
    first_mtimes = _mtimes(output_dir)

    changed_args = [
        "--preset", "smoke",
        "--point-sample-limit", "12",
        "--trajectory-ids", "line_fixed_orientation",
        "--trial-limit", "2",
        "--waypoint-limit", "20",  # changed from 25 -> alters tier2's resolved input signature
        "--no-plots",
    ]
    exit_code = main(["--output", str(output_dir), "--resume"] + changed_args)
    assert exit_code == 0
    second_mtimes = _mtimes(output_dir)

    assert second_mtimes["tier0"] == first_mtimes["tier0"]
    assert second_mtimes["tier1"] == first_mtimes["tier1"]
    assert second_mtimes["tier2"] != first_mtimes["tier2"]
    assert second_mtimes["tier3"] != first_mtimes["tier3"]
    assert second_mtimes["tier4"] != first_mtimes["tier4"]

    waypoint_df = pd.read_csv(output_dir / "tier2_sequential_dls" / "waypoint_results.csv")
    assert (waypoint_df["waypoint_id"] < 20).all()
    assert waypoint_df.duplicated(subset=["trial_id", "method", "waypoint_id"]).sum() == 0
