"""Integration test for the Tier 0 pipeline (pipelines.run_tier0_kinematics).

Runs Tier 0 on a small subset of the validation benchmarks (not the full 1000/200/300 sample
sets) and checks the mandatory gate output: required files exist, the summary JSON parses, the
gate passes against the real KR810 model, and the Jacobian relative error stays within the
project's 1e-4 gate threshold.
"""

import json

from pipelines.run_tier0_kinematics import run_tier0


def test_tier0_runs_on_subset_and_writes_expected_files(tmp_path):
    output_dir = tmp_path / "tier0_kinematics"

    result = run_tier0(
        fk_sample_limit=15,
        jacobian_sample_limit=6,
        singularity_sample_limit=6,
        output_dir=output_dir,
        make_plots=False,
    )

    assert (output_dir / "fk_validation.csv").is_file()
    assert (output_dir / "jacobian_validation.csv").is_file()
    assert (output_dir / "singularity_validation.csv").is_file()
    assert (output_dir / "summary.json").is_file()

    assert len(result["fk_results"]) == 15
    assert len(result["jacobian_results"]) == 6
    assert len(result["singularity_results"]) == 6


def test_tier0_summary_json_parses_and_gate_passes(tmp_path):
    output_dir = tmp_path / "tier0_kinematics"
    run_tier0(
        fk_sample_limit=15,
        jacobian_sample_limit=6,
        singularity_sample_limit=6,
        output_dir=output_dir,
        make_plots=False,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["gate_pass"] is True
    assert summary["gate_reasons"] == []
    assert summary["max_jacobian_relative_error"] <= 1e-4
    assert summary["fk_sample_count"] == 15
    assert summary["jacobian_sample_count"] == 6
    assert summary["singularity_sample_count"] == 6


def test_tier0_gate_result_matches_summary(tmp_path):
    output_dir = tmp_path / "tier0_kinematics"
    result = run_tier0(
        fk_sample_limit=10,
        jacobian_sample_limit=5,
        singularity_sample_limit=5,
        output_dir=output_dir,
        make_plots=False,
    )

    assert result["gate"].gate_pass is True
    assert result["gate"].max_jacobian_relative_error <= 1e-4
    assert result["gate"].fk_rotation_failures == 0
