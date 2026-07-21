"""Integration test for the full Tier 0-4 smoke pipeline (pipelines.run_tier0_to_tier4).

Runs a real (but deliberately tiny) smoke pipeline end to end into ``tmp_path`` -- never into the
repository's own results/ or dataset directories -- and checks: successful exit, every required
tier output file exists and is non-empty where expected, FINAL_SUMMARY.json parses with all five
tiers reporting a status, both warm-start and cold-start results are present, and nothing was
written into assets/, benchmarks/, or trajectories/.
"""

import json
from pathlib import Path

import pandas as pd

from pipelines.run_tier0_to_tier4 import main
from utils.dataset_locator import ASSETS_DIR, BENCHMARKS_DIR, REPO_ROOT, TRAJECTORIES_DIR

_WATCHED_DATASET_DIRS = [ASSETS_DIR, BENCHMARKS_DIR, TRAJECTORIES_DIR]


def _snapshot(dirs):
    snapshot = {}
    for directory in dirs:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path.relative_to(REPO_ROOT))] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def test_smoke_pipeline_end_to_end(tmp_path):
    output_dir = tmp_path / "smoke_run"
    before = _snapshot(_WATCHED_DATASET_DIRS)

    exit_code = main([
        "--preset", "smoke",
        "--output", str(output_dir),
        "--point-sample-limit", "12",
        "--trajectory-ids", "line_fixed_orientation",
        "--trial-limit", "2",
        "--waypoint-limit", "25",
        "--no-plots",
    ])

    after = _snapshot(_WATCHED_DATASET_DIRS)

    assert exit_code == 0
    assert before == after, "smoke run must never modify assets/, benchmarks/, or trajectories/"

    # Every written path must be inside tmp_path.
    for path in output_dir.rglob("*"):
        assert tmp_path in path.parents

    for name in ("FINAL_SUMMARY.json", "run_manifest.json", "resolved_config.json"):
        assert (output_dir / name).is_file(), f"missing {name}"

    for tier_dir in (
        "tier0_kinematics", "tier1_point_dls", "tier2_sequential_dls",
        "tier3_trajectory_tracking", "tier4_joint_feasibility",
    ):
        assert (output_dir / tier_dir).is_dir(), f"missing {tier_dir}/"

    final_summary = json.loads((output_dir / "FINAL_SUMMARY.json").read_text(encoding="utf-8"))
    assert final_summary["overall_status"] == "completed"
    for tier in ("tier0", "tier1", "tier2", "tier3", "tier4"):
        assert "status" in final_summary[tier]
        assert final_summary[tier]["status"] != "not_run"

    point_results = pd.read_csv(output_dir / "tier1_point_dls" / "point_results.csv")
    assert len(point_results) == 12

    waypoint_results = pd.read_csv(output_dir / "tier2_sequential_dls" / "waypoint_results.csv")
    assert len(waypoint_results) > 0
    assert set(waypoint_results["method"].unique()) == {"warm_start", "cold_start"}

    trial_summaries = pd.read_csv(output_dir / "tier2_sequential_dls" / "trajectory_trial_summaries.csv")
    assert set(trial_summaries["method"].unique()) == {"warm_start", "cold_start"}

    tracking_metrics = pd.read_csv(output_dir / "tier3_trajectory_tracking" / "trajectory_metrics.csv")
    assert len(tracking_metrics) > 0

    smoothness_metrics = pd.read_csv(output_dir / "tier4_joint_feasibility" / "smoothness_metrics.csv")
    assert len(smoothness_metrics) > 0
