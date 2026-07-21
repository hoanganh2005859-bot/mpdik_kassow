"""Combined entry point that runs Tier 0 through Tier 4 in sequence for a given experiment preset.

Enforces the Tier 0-4 dependency chain (see module docstrings of the individual
``pipelines.run_tier*`` modules for what each tier itself guarantees):

- Tier 0 is a mandatory gate. If it fails (non-finite FK/Jacobian output, invalid rotations, or
  a Jacobian relative error above threshold), the run stops before Tier 1 -- Tier 1-4 are
  recorded as ``not_run``.
- Tier 1 always runs to completion and never gates the rest of the pipeline: a low point-IK
  success rate is recorded (``acceptance_status``) but Tier 2-4 still run against the same
  model/config.
- Tier 3 and Tier 4 consume Tier 2's ``waypoint_results.csv`` directly (no re-solving); if Tier 2
  produced zero usable waypoint results (e.g. every selected trial failed fatally), Tier 3/4 are
  recorded as ``skipped`` with a reason rather than crashing.

Usage:
    python -m pipelines.run_tier0_to_tier4 --preset smoke --output results/smoke_run
    python -m pipelines.run_tier0_to_tier4 --preset full --output results/full_run
"""

import argparse
import logging
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from pipelines import _common
from pipelines._output_validation import OutputValidationError, validate_csv_file, validate_json_file
from pipelines.run_tier0_kinematics import run_tier0
from pipelines.run_tier1_point_dls import run_tier1
from pipelines.run_tier2_sequential_dls import run_tier2
from pipelines.run_tier3_trajectory_tracking import run_tier3
from pipelines.run_tier4_joint_feasibility import run_tier4
from kinematics.dls_solver import load_dls_config
from kinematics.model_loader import load_model_context
from utils.reproducibility import environment_metadata
from utils.result_logger import configure_logging, write_result_json

logger = logging.getLogger(__name__)

_TIER_REQUIRED_FILES = {
    "tier0": ["fk_validation.csv", "jacobian_validation.csv", "singularity_validation.csv", "summary.json"],
    "tier1": [
        "point_results.csv", "metrics_overall.json", "metrics_by_difficulty.csv",
        "failure_reasons.csv", "failure_cases.json",
    ],
    "tier2": ["waypoint_results.csv", "trajectory_trial_summaries.csv", "warm_vs_cold.csv", "failure_cases.json"],
    "tier3": ["trajectory_metrics.csv", "cross_track_metrics.csv", "iso9283_metrics.csv", "confidence_intervals.csv"],
    "tier4": [
        "smoothness_metrics.csv", "joint_feasibility_metrics.csv",
        "singularity_path_metrics.csv", "runtime_metrics.csv",
    ],
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Tier 0-4 (kinematics validation through joint feasibility).")
    parser.add_argument("--preset", choices=["smoke", "full"], required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated: warm_start,cold_start.")
    parser.add_argument("--trajectory-ids", type=str, default=None, help="Comma-separated trajectory_id list.")
    parser.add_argument("--trial-category", choices=["repeatability", "robustness", "all"], default=None)
    parser.add_argument("--trial-limit", type=int, default=None)
    parser.add_argument("--point-sample-limit", type=int, default=None)
    parser.add_argument("--waypoint-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _prepare_output_dir(output_dir: Path, overwrite: bool, resume: bool) -> None:
    if output_dir.exists():
        if resume:
            return
        if overwrite:
            if not _common.is_safe_to_remove(output_dir):
                raise RuntimeError(f"refusing to remove protected path: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"output directory '{output_dir}' already exists; pass --overwrite or --resume"
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_previous_manifest(output_dir: Path) -> Optional[dict]:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return validate_json_file(manifest_path)
    except OutputValidationError:
        return None


def _tier_output_valid(tier: str, tier_dir: Path) -> bool:
    try:
        for filename in _TIER_REQUIRED_FILES[tier]:
            path = tier_dir / filename
            if filename.endswith(".json"):
                validate_json_file(path)
            else:
                validate_csv_file(path, allow_empty=True)
        return True
    except OutputValidationError:
        return False


def _can_reuse_tier(
    tier: str, signature: str, previous_manifest: Optional[dict], tier_dir: Path
) -> bool:
    if previous_manifest is None:
        return False
    tier_state = previous_manifest.get("tiers", {}).get(tier)
    if tier_state is None or tier_state.get("status") != "completed":
        return False
    if tier_state.get("input_signature") != signature:
        return False
    return _tier_output_valid(tier, tier_dir)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    """pd.read_csv that tolerates a header-only or fully empty CSV (returns an empty DataFrame)."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _acceptance_entry(name: str, tier: str, value: Any, threshold: Any, passed: bool, unit: str) -> dict:
    return {
        "name": name, "tier": tier, "value": value, "threshold": threshold,
        "passed": bool(passed), "unit": unit, "source": "project_criterion",
    }


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.overwrite and args.resume:
        print("error: --overwrite and --resume are mutually exclusive", file=sys.stderr)
        return 2

    output_dir = Path(args.output)
    try:
        _prepare_output_dir(output_dir, args.overwrite, args.resume)
    except (FileExistsError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    configure_logging(args.log_level, log_file=output_dir / "run.log")
    paths = _common.ensure_output_structure(output_dir)

    cli_overrides = {
        "seed": args.seed,
        "methods": args.methods.split(",") if args.methods else None,
        "trajectory_ids": args.trajectory_ids.split(",") if args.trajectory_ids else None,
        "trial_category": args.trial_category,
        "trial_limit": args.trial_limit,
        "point_sample_limit": args.point_sample_limit,
        "waypoint_limit": args.waypoint_limit,
        "no_plots": args.no_plots,
    }
    resolved_config = _common.build_resolved_config(args.preset, cli_overrides)
    effective = resolved_config["effective"]
    write_result_json(resolved_config, output_dir / "resolved_config.json")

    checksums = _common.compute_dataset_checksums(effective["selected_trajectories"])

    previous_manifest = _load_previous_manifest(output_dir) if args.resume else None

    start_time = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    run_id = start_time.strftime("run-%Y%m%dT%H%M%SZ")

    tiers_state: Dict[str, dict] = {}
    warnings: list = []
    fatal_error: Optional[str] = None
    tier0_summary = None
    tier1_summary = None
    tier2_summaries_df: Optional[pd.DataFrame] = None
    tier3_tracking_df = tier3_cross_track_df = tier3_iso_df = None
    tier4_smoothness_df = tier4_feasibility_df = tier4_singularity_df = tier4_runtime_df = None

    try:
        model_context = load_model_context()
        dls_config = load_dls_config()

        # ---------------- Tier 0 (mandatory gate) ----------------
        tier0_signature = _common.canonical_signature({
            "dls_config": dls_config,
            "fk_samples": effective["validation_fk_samples"],
            "jacobian_samples": effective["validation_jacobian_samples"],
            "singularity_samples": effective["validation_singularity_samples"],
            "model_sha256": checksums["model_sha256"],
        })
        if _can_reuse_tier("tier0", tier0_signature, previous_manifest, paths["tier0"]):
            logger.info("tier0: reusing valid existing output (--resume)")
            tier0_summary = validate_json_file(paths["tier0"] / "summary.json")
            gate_pass = bool(tier0_summary["gate_pass"])
            tiers_state["tier0"] = previous_manifest["tiers"]["tier0"]
        else:
            tier0_out = run_tier0(
                model_context=model_context,
                dls_config=dls_config,
                fk_sample_limit=effective["validation_fk_samples"],
                jacobian_sample_limit=effective["validation_jacobian_samples"],
                singularity_sample_limit=effective["validation_singularity_samples"],
                output_dir=paths["tier0"],
                make_plots=effective["make_plots"],
            )
            tier0_summary = tier0_out["summary"]
            gate_pass = tier0_out["gate"].gate_pass
            tiers_state["tier0"] = {
                "status": "completed", "input_signature": tier0_signature,
                "gate_pass": gate_pass, "sample_count": tier0_summary["fk_sample_count"],
            }

        if not gate_pass:
            fatal_error = "Tier 0 gate failed: " + "; ".join(tier0_summary.get("gate_reasons", []))
            logger.error(fatal_error)
            for tier in ("tier1", "tier2", "tier3", "tier4"):
                tiers_state[tier] = {"status": "not_run", "reason": "Tier 0 gate failed"}
        else:
            # ---------------- Tier 1 ----------------
            tier1_signature = _common.canonical_signature({
                "dls_config": dls_config,
                "point_sample_limit": effective["point_sample_limit"],
                "model_sha256": checksums["model_sha256"],
                "point_benchmark_sha256": checksums["point_benchmark_sha256"],
            })
            if _can_reuse_tier("tier1", tier1_signature, previous_manifest, paths["tier1"]):
                logger.info("tier1: reusing valid existing output (--resume)")
                tier1_summary = validate_json_file(paths["tier1"] / "metrics_overall.json")
                tiers_state["tier1"] = previous_manifest["tiers"]["tier1"]
            else:
                tier1_result = run_tier1(
                    model_context=model_context,
                    dls_config=dls_config,
                    sample_limit=effective["point_sample_limit"],
                    output_dir=paths["tier1"],
                    make_plots=effective["make_plots"],
                )
                tier1_summary = tier1_result["overall_summary"]
                tiers_state["tier1"] = {
                    "status": "completed", "input_signature": tier1_signature,
                    "acceptance_status": tier1_summary["acceptance_status"],
                    "sample_count": tier1_summary["sample_count"],
                }

            # ---------------- Tier 2 ----------------
            tier2_signature = _common.canonical_signature({
                "dls_config": dls_config,
                "methods": effective["methods"],
                "selected_trajectories": effective["selected_trajectories"],
                "trial_category": effective["trial_category"],
                "trial_limit": effective["trial_limit"],
                "waypoint_limit": effective["waypoint_limit"],
                "model_sha256": checksums["model_sha256"],
                "trajectory_file_checksums": checksums["trajectory_file_checksums"],
            })
            tier2_reused = _can_reuse_tier("tier2", tier2_signature, previous_manifest, paths["tier2"])
            if tier2_reused:
                logger.info("tier2: reusing valid existing output (--resume)")
                tier2_waypoint_df = validate_csv_file(paths["tier2"] / "waypoint_results.csv", allow_empty=True)
                tier2_summaries_df = _read_csv_or_empty(paths["tier2"] / "trajectory_trial_summaries.csv")
                tiers_state["tier2"] = previous_manifest["tiers"]["tier2"]
            else:
                tier2_result = run_tier2(
                    model_context=model_context,
                    dls_config=dls_config,
                    trajectory_ids=effective["selected_trajectories"],
                    trial_category=effective["trial_category"],
                    methods=effective["methods"],
                    trial_limit=effective["trial_limit"],
                    waypoint_limit=effective["waypoint_limit"],
                    output_dir=paths["tier2"],
                    make_plots=effective["make_plots"],
                )
                tier2_waypoint_df = tier2_result["waypoint_df"]
                tier2_summaries_df = tier2_result["summaries_df"]
                tiers_state["tier2"] = {
                    "status": "completed", "input_signature": tier2_signature,
                    "sample_count": len(tier2_waypoint_df),
                }

            # ---------------- Tier 3 ----------------
            if tier2_waypoint_df.empty:
                warnings.append("Tier 2 produced no waypoint results; Tier 3 skipped.")
                tiers_state["tier3"] = {"status": "skipped", "reason": "no Tier 2 waypoint results"}
            else:
                tier3_signature = _common.canonical_signature({"tier2_signature": tier2_signature, "seed": effective["seed"]})
                if tier2_reused and _can_reuse_tier("tier3", tier3_signature, previous_manifest, paths["tier3"]):
                    logger.info("tier3: reusing valid existing output (--resume)")
                    tier3_tracking_df = _read_csv_or_empty(paths["tier3"] / "trajectory_metrics.csv")
                    tier3_cross_track_df = _read_csv_or_empty(paths["tier3"] / "cross_track_metrics.csv")
                    tier3_iso_df = _read_csv_or_empty(paths["tier3"] / "iso9283_metrics.csv")
                    tiers_state["tier3"] = previous_manifest["tiers"]["tier3"]
                else:
                    tier3_result = run_tier3(
                        tier2_waypoint_df,
                        output_dir=paths["tier3"],
                        confidence_level=resolved_config["evaluation_config"]["confidence_level"],
                        bootstrap_resamples=resolved_config["evaluation_config"]["bootstrap_resamples"],
                        seed=effective["seed"],
                        make_plots=effective["make_plots"],
                    )
                    tier3_tracking_df = tier3_result["tracking_df"]
                    tier3_cross_track_df = tier3_result["cross_track_df"]
                    tier3_iso_df = tier3_result["iso9283_df"]
                    tiers_state["tier3"] = {
                        "status": "completed", "input_signature": tier3_signature,
                        "sample_count": len(tier3_tracking_df),
                    }
                    if tier3_iso_df.empty:
                        warnings.append(
                            "Tier 3 ISO9283-inspired metrics unavailable: no selected "
                            "(trajectory_id, method, speed_scale) group had >= 2 repeatability "
                            "repeats; use a larger --trial-limit to populate iso9283_metrics.csv."
                        )

            # ---------------- Tier 4 ----------------
            if tier2_waypoint_df.empty:
                warnings.append("Tier 2 produced no waypoint results; Tier 4 skipped.")
                tiers_state["tier4"] = {"status": "skipped", "reason": "no Tier 2 waypoint results"}
            else:
                tier4_signature = _common.canonical_signature({"tier2_signature": tier2_signature})
                if tier2_reused and _can_reuse_tier("tier4", tier4_signature, previous_manifest, paths["tier4"]):
                    logger.info("tier4: reusing valid existing output (--resume)")
                    tier4_smoothness_df = _read_csv_or_empty(paths["tier4"] / "smoothness_metrics.csv")
                    tier4_feasibility_df = _read_csv_or_empty(paths["tier4"] / "joint_feasibility_metrics.csv")
                    tier4_singularity_df = _read_csv_or_empty(paths["tier4"] / "singularity_path_metrics.csv")
                    tier4_runtime_df = _read_csv_or_empty(paths["tier4"] / "runtime_metrics.csv")
                    tiers_state["tier4"] = previous_manifest["tiers"]["tier4"]
                else:
                    tier4_result = run_tier4(tier2_waypoint_df, output_dir=paths["tier4"], make_plots=effective["make_plots"])
                    tier4_smoothness_df = tier4_result["smoothness_df"]
                    tier4_feasibility_df = tier4_result["feasibility_df"]
                    tier4_singularity_df = tier4_result["singularity_df"]
                    tier4_runtime_df = tier4_result["runtime_df"]
                    tiers_state["tier4"] = {
                        "status": "completed", "input_signature": tier4_signature,
                        "sample_count": len(tier4_smoothness_df),
                    }

    except Exception as exc:  # pragma: no cover - defensive top-level guard
        fatal_error = f"{type(exc).__name__}: {exc}"
        logger.error("pipeline failed: %s\n%s", fatal_error, traceback.format_exc())

    overall_status = "completed" if fatal_error is None else "failed"

    final_summary = _build_final_summary(
        run_id, args.preset, tier0_summary, tier1_summary,
        tier2_summaries_df, tier3_tracking_df, tier3_cross_track_df, tier3_iso_df,
        tier4_smoothness_df, tier4_feasibility_df, tier4_singularity_df, tier4_runtime_df,
        resolved_config, overall_status, warnings, fatal_error,
    )
    write_result_json(final_summary, output_dir / "FINAL_SUMMARY.json")

    end_time = datetime.now(timezone.utc)
    run_manifest = {
        "run_id": run_id,
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": end_time.isoformat(),
        "duration_s": time.perf_counter() - start_perf,
        "project_version": "1.0.0",
        "dataset_manifest_version": "1.0.0",
        "preset": args.preset,
        "seed": effective["seed"],
        "command_line": " ".join(sys.argv),
        "output_path": _common.relative_to_repo(output_dir),
        "environment": environment_metadata(),
        "checksums": checksums,
        "selected_point_samples": effective["point_sample_limit"],
        "selected_trajectories": effective["selected_trajectories"] or "all",
        "selected_methods": effective["methods"],
        "tiers": tiers_state,
        "overall_status": overall_status,
        "warnings": warnings,
        "fatal_error": fatal_error,
    }
    write_result_json(run_manifest, output_dir / "run_manifest.json")

    return 0 if overall_status == "completed" else 1


def _build_final_summary(
    run_id, preset, tier0_summary, tier1_summary,
    tier2_summaries_df, tier3_tracking_df, tier3_cross_track_df, tier3_iso_df,
    tier4_smoothness_df, tier4_feasibility_df, tier4_singularity_df, tier4_runtime_df,
    resolved_config, overall_status, warnings, fatal_error,
) -> dict:
    evaluation_config = resolved_config["evaluation_config"]
    acceptance_criteria = []

    tier0_block = {"status": "not_run", "sample_count": 0}
    if tier0_summary is not None:
        tier0_block = {
            "status": "passed" if tier0_summary["gate_pass"] else "failed",
            "gate_pass": tier0_summary["gate_pass"],
            "max_jacobian_relative_error": tier0_summary["max_jacobian_relative_error"],
            "minimum_sigma_min": tier0_summary["minimum_sigma_min"],
            "sample_count": tier0_summary["fk_sample_count"],
            "notes": tier0_summary.get("note", ""),
        }

    tier1_block = {"status": "not_run", "sample_count": 0}
    if tier1_summary is not None:
        s = tier1_summary
        tier1_block = {
            "status": s["acceptance_status"],
            "execution_status": s["execution_status"],
            "sample_count": s["sample_count"],
            "success_rate": s["success_rate"],
            "success_rate_wilson_ci": s["success_rate_wilson_ci"],
            "position_rmse_mm": s["position_rmse_m"] * 1000.0,
            "position_p95_mm": s["position_p95_m"] * 1000.0,
            "position_max_mm": s["position_max_m"] * 1000.0,
            "orientation_rmse_deg": s["orientation_rmse_deg"],
            "orientation_p95_deg": s["orientation_p95_deg"],
            "orientation_max_deg": s["orientation_max_deg"],
            "mean_iterations": s["mean_iterations"],
            "p95_runtime_ms": s["p95_solve_time_ms"],
            "failure_counts": s["failure_reason_counts"],
        }
        acceptance_criteria.append({**s["acceptance_criterion"], "tier": "tier1"})

    tier2_block = {"status": "not_run"}
    if tier2_summaries_df is not None and not tier2_summaries_df.empty:
        tier2_block = {"status": "completed"}
        for method in ("warm_start", "cold_start"):
            method_summaries = tier2_summaries_df[tier2_summaries_df["method"] == method]
            if method_summaries.empty:
                continue
            waypoint_success_rate = float(method_summaries["waypoint_success_rate"].mean())
            full_completion_rate = float(method_summaries["full_trajectory_completed"].mean())
            tier2_block[method] = {
                "trial_count": int(len(method_summaries)),
                "waypoint_success_rate": waypoint_success_rate,
                "full_trajectory_completion_rate": full_completion_rate,
                "maximum_failure_streak": int(method_summaries["maximum_failure_streak"].max()),
                "recovery_rate": float(method_summaries["recovery_rate"].mean(skipna=True)),
                "mean_iterations": float(method_summaries["mean_iterations"].mean()),
                "p95_runtime_ms": float(method_summaries["p95_solve_time_ms"].mean()),
            }
            acceptance_criteria.append(_acceptance_entry(
                f"tier2_{method}_waypoint_success_rate", "tier2", waypoint_success_rate,
                evaluation_config["minimum_waypoint_success_rate"],
                waypoint_success_rate >= evaluation_config["minimum_waypoint_success_rate"], "fraction",
            ))
            acceptance_criteria.append(_acceptance_entry(
                f"tier2_{method}_trajectory_completion_rate", "tier2", full_completion_rate,
                evaluation_config["required_trajectory_completion_rate"],
                full_completion_rate >= evaluation_config["required_trajectory_completion_rate"], "fraction",
            ))

    tier3_block = {"status": "not_run"}
    if tier3_tracking_df is not None and not tier3_tracking_df.empty:
        tier3_block = {"status": "completed"}
        tracking_df = tier3_tracking_df
        cross_track_df = tier3_cross_track_df
        iso_df = tier3_iso_df
        for method in ("warm_start", "cold_start"):
            method_tracking = tracking_df[tracking_df["method"] == method]
            method_cross = cross_track_df[cross_track_df["method"] == method]
            if method_tracking.empty:
                continue
            position_rmse_mm = float(method_tracking["position_rmse_m"].mean() * 1000.0)
            position_p95_mm = float(method_tracking["position_p95_m"].mean() * 1000.0)
            position_max_mm = float(method_tracking["position_max_m"].max() * 1000.0)
            orientation_p95_deg = float(method_tracking["orientation_p95_deg"].mean())
            method_iso = iso_df[iso_df["method"] == method] if not iso_df.empty else iso_df
            tier3_block[method] = {
                "position_rmse_mm": position_rmse_mm,
                "position_p95_mm": position_p95_mm,
                "position_max_mm": position_max_mm,
                "orientation_rmse_deg": float(method_tracking["orientation_rmse_deg"].mean()),
                "orientation_p95_deg": orientation_p95_deg,
                "cross_track_rmse_mm": float(method_cross["cross_track_rmse_m"].mean() * 1000.0) if not method_cross.empty else None,
                "cross_track_p95_mm": float(method_cross["cross_track_p95_m"].mean() * 1000.0) if not method_cross.empty else None,
                "atp_mm": float(method_iso["atp_m"].mean() * 1000.0) if not method_iso.empty else None,
                "rtp_mm": float(method_iso["rtp_m"].mean() * 1000.0) if not method_iso.empty else None,
            }
            acceptance_criteria.append(_acceptance_entry(
                f"tier3_{method}_position_rmse_mm", "tier3", position_rmse_mm,
                evaluation_config["kinematic_position_rmse_target_mm"],
                position_rmse_mm <= evaluation_config["kinematic_position_rmse_target_mm"], "mm",
            ))
            acceptance_criteria.append(_acceptance_entry(
                f"tier3_{method}_position_p95_mm", "tier3", position_p95_mm,
                evaluation_config["kinematic_position_p95_target_mm"],
                position_p95_mm <= evaluation_config["kinematic_position_p95_target_mm"], "mm",
            ))
            acceptance_criteria.append(_acceptance_entry(
                f"tier3_{method}_orientation_p95_deg", "tier3", orientation_p95_deg,
                evaluation_config["orientation_p95_target_deg"],
                orientation_p95_deg <= evaluation_config["orientation_p95_target_deg"], "deg",
            ))

    tier4_block = {"status": "not_run"}
    if tier4_smoothness_df is not None and not tier4_smoothness_df.empty:
        tier4_block = {"status": "completed"}
        smoothness_df = tier4_smoothness_df
        feasibility_df = tier4_feasibility_df
        singularity_df = tier4_singularity_df
        runtime_df = tier4_runtime_df
        for method in ("warm_start", "cold_start"):
            m_smooth = smoothness_df[smoothness_df["method"] == method]
            m_feas = feasibility_df[feasibility_df["method"] == method]
            m_sing = singularity_df[singularity_df["method"] == method]
            m_runtime = runtime_df[runtime_df["method"] == method]
            if m_smooth.empty:
                continue
            acceleration_statuses = set(m_feas["acceleration_status"].unique()) if not m_feas.empty else set()
            tier4_block[method] = {
                "max_joint_jump_rad": float(m_smooth["max_joint_jump_rad"].max()),
                "rms_jerk_rad_s3": float(m_smooth["global_rms_jerk_rad_s3"].dropna().mean()) if m_smooth["global_rms_jerk_rad_s3"].notna().any() else None,
                "min_joint_limit_margin": float(m_feas["minimum_normalized_joint_limit_margin"].dropna().min()) if not m_feas.empty and m_feas["minimum_normalized_joint_limit_margin"].notna().any() else None,
                "velocity_violation_rate": float((m_feas["velocity_violation_count"].fillna(0) > 0).mean()) if not m_feas.empty else None,
                "minimum_sigma_min": float(m_sing["minimum_sigma_min"].min()) if not m_sing.empty else None,
                "near_singular_fraction": float(m_sing["near_singular_fraction"].mean()) if not m_sing.empty else None,
                "deadline_miss_rate": float(m_runtime["deadline_miss_rate"].mean()) if not m_runtime.empty else None,
                "acceleration_status": "available" if acceleration_statuses == {"available"} else "unavailable",
            }

    return {
        "run_id": run_id,
        "preset": preset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tier0": tier0_block,
        "tier1": tier1_block,
        "tier2": tier2_block,
        "tier3": tier3_block,
        "tier4": tier4_block,
        "acceptance_criteria": acceptance_criteria,
        "overall_status": overall_status,
        "warnings": warnings,
        "fatal_error": fatal_error,
        "note": (
            "Acceptance criteria are project-defined thresholds for this dataset's own evaluation "
            "pipeline (see configs/evaluation_config.json). They are not ISO 9283 certification "
            "limits, and Tier 0-4 covers kinematic evaluation only (no dynamics/controller)."
        ),
    }


if __name__ == "__main__":
    sys.exit(main())
