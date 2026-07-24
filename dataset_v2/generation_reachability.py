"""Strict, generation-time reachability verification for Dataset v2 (Phase 5.1).

The rule this module exists to enforce: **a target pose is "reachable" only if a reference joint
configuration exists whose INDEPENDENTLY recomputed forward kinematics reproduces that pose within
Dataset v2's own strict tolerances.** A numerical IK engine's `success` flag is never sufficient
evidence, and no Dataset v1 DLS *baseline evaluation* threshold ever participates in the decision
-- a dataset whose reachability is defined by the very solver it will later benchmark cannot
measure that solver.

The DLS implementation in ``kinematics/dls_solver.py`` is reused unchanged here purely as a
numerical IK *engine*, driven by Dataset v2's own generation solver settings
(``configs/generation_reachability_config.json``). Nothing this module produces is a DLS baseline
evaluation result, and nothing here may be reported as one.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataset_v2.seeds import derive_seed
from kinematics.dls_solver import solve_dls_until_converged
from kinematics.forward_kinematics import forward_kinematics
from kinematics.model_loader import ModelContext
from kinematics.quaternion_utils import quaternion_wxyz_to_matrix
from kinematics.rotation_utils import rotation_geodesic_angle
from utils.config_loader import load_json_config

GENERATION_REACHABILITY_CONFIG_NAME = "generation_reachability_config.json"


@dataclass(frozen=True)
class GenerationReachabilitySettings:
    position_tolerance_m: float
    orientation_tolerance_deg: float
    solver_config: dict
    max_refinement_rounds: int
    max_restarts: int
    restart_perturbation_rad: float

    @property
    def orientation_tolerance_rad(self) -> float:
        return float(np.radians(self.orientation_tolerance_deg))


def load_generation_reachability_settings(paths) -> GenerationReachabilitySettings:
    """Load Dataset v2's own strict reachability policy.

    Reads only ``configs/generation_reachability_config.json`` -- never ``configs/dls_config.json``.
    """
    config = load_json_config(paths.configs_dir / GENERATION_REACHABILITY_CONFIG_NAME)
    if config.get("independence", {}).get("reads_dls_evaluation_thresholds", True):
        raise ValueError(
            f"{GENERATION_REACHABILITY_CONFIG_NAME} declares reads_dls_evaluation_thresholds=true; "
            "Dataset v2 generation reachability must be independent of the DLS baseline evaluation "
            "thresholds it will later be used to measure."
        )
    solver = dict(config["generation_solver"])
    solver.pop("engine", None)
    solver.pop("solver_threshold_note", None)
    return GenerationReachabilitySettings(
        position_tolerance_m=float(config["position_reconstruction_tolerance_m"]),
        orientation_tolerance_deg=float(config["orientation_reconstruction_tolerance_deg"]),
        solver_config=solver,
        max_refinement_rounds=int(config["refinement_policy"]["max_refinement_rounds"]),
        max_restarts=int(config["restart_policy"]["max_restarts"]),
        restart_perturbation_rad=float(config["restart_policy"]["restart_perturbation_rad"]),
    )


def probe_settings_from(paths, full: GenerationReachabilitySettings) -> GenerationReachabilitySettings:
    """Cheap, strictly-weaker settings for the (geometry alternative, scale) search.

    Identical tolerances -- only the effort budget shrinks. Because it is weaker it can only
    under-estimate reachability, never accept something the full policy would reject; whatever the
    search selects is re-validated end-to-end with ``full`` before being written.
    """
    config = load_json_config(paths.configs_dir / GENERATION_REACHABILITY_CONFIG_NAME)
    probe = config["search_probe_policy"]
    return GenerationReachabilitySettings(
        position_tolerance_m=full.position_tolerance_m,
        orientation_tolerance_deg=full.orientation_tolerance_deg,
        solver_config=full.solver_config,
        max_refinement_rounds=int(probe["max_refinement_rounds"]),
        max_restarts=int(probe["max_restarts"]),
        restart_perturbation_rad=full.restart_perturbation_rad,
    )


def fk_reconstruction_error(
    model_context: ModelContext, q: np.ndarray, target_position: np.ndarray, target_rotation: np.ndarray, data=None
) -> Tuple[float, float]:
    """Independently recompute FK(q) and report (position error [m], orientation error [rad]).

    This is the *only* evidence that decides reachability -- deliberately separate from whatever
    the numerical IK engine reported about its own convergence.
    """
    fk = forward_kinematics(model_context, q, data=data)
    position_error_m = float(np.linalg.norm(fk.position - target_position))
    orientation_error_rad = float(rotation_geodesic_angle(fk.rotation_matrix, target_rotation))
    return position_error_m, orientation_error_rad


def _within_limits(model_context: ModelContext, q: np.ndarray) -> bool:
    return bool(
        np.all(q >= model_context.operational_lower_rad - 1e-12) and np.all(q <= model_context.operational_upper_rad + 1e-12)
    )


def _solve_with_refinement(
    model_context: ModelContext,
    settings: GenerationReachabilitySettings,
    q_start: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    data=None,
) -> Tuple[np.ndarray, float, float, bool, int]:
    """Warm-start solve, then re-enter the solver until independent FK reconstruction passes.

    Re-entering resets the solver's internal stagnation window, which otherwise halts descent well
    above the strict generation tolerance. Never relaxes the tolerance.
    """
    q = np.asarray(q_start, dtype=np.float64).copy()
    position_error_m = np.inf
    orientation_error_rad = np.inf

    for round_index in range(settings.max_refinement_rounds):
        result = solve_dls_until_converged(model_context, q, target_position, target_rotation, config=settings.solver_config)
        q = result.q_solution
        position_error_m, orientation_error_rad = fk_reconstruction_error(
            model_context, q, target_position, target_rotation, data=data
        )
        passed = (
            position_error_m <= settings.position_tolerance_m
            and orientation_error_rad <= settings.orientation_tolerance_rad
            and _within_limits(model_context, q)
        )
        if passed:
            return q, position_error_m, orientation_error_rad, True, round_index + 1

    return q, position_error_m, orientation_error_rad, False, settings.max_refinement_rounds


def solve_reference_configuration(
    model_context: ModelContext,
    settings: GenerationReachabilitySettings,
    q_warm_start: np.ndarray,
    q_anchor: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    restart_seed: int,
    data=None,
) -> dict:
    """Find a q_reference for one target pose, verified by independent FK reconstruction.

    Tries, in deterministic order: warm start from ``q_warm_start`` (+refinement), then the
    anchor configuration, then seeded perturbations of the warm start. Returns a dict with
    ``reachable`` reflecting the independent FK check only.
    """
    target_rotation = quaternion_wxyz_to_matrix(target_quaternion)

    candidates: List[np.ndarray] = [np.asarray(q_warm_start, dtype=np.float64)]
    if settings.max_restarts >= 1:
        candidates.append(np.asarray(q_anchor, dtype=np.float64))
    if settings.max_restarts >= 2:
        rng = np.random.default_rng(restart_seed)
        lower = model_context.operational_lower_rad
        upper = model_context.operational_upper_rad
        for _ in range(settings.max_restarts - 1):
            perturbation = rng.uniform(-settings.restart_perturbation_rad, settings.restart_perturbation_rad, size=model_context.nq)
            candidates.append(np.clip(np.asarray(q_warm_start, dtype=np.float64) + perturbation, lower, upper))

    best = None
    for attempt_index, q_start in enumerate(candidates):
        q, position_error_m, orientation_error_rad, passed, rounds = _solve_with_refinement(
            model_context, settings, q_start, target_position, target_rotation, data=data
        )
        record = {
            "q_reference": q,
            "position_error_m": position_error_m,
            "orientation_error_rad": orientation_error_rad,
            "reachable": passed,
            "attempts_used": attempt_index + 1,
            "refinement_rounds": rounds,
        }
        if passed:
            return record
        if best is None or position_error_m < best["position_error_m"]:
            best = record

    return best


def validate_path_strict(
    model_context: ModelContext,
    settings: GenerationReachabilitySettings,
    q_anchor: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    path_seed: int,
    stop_on_first_failure: bool = False,
    data=None,
) -> dict:
    """Sequentially find and independently verify a q_reference for every waypoint on a path.

    ``stop_on_first_failure`` is a pure search optimization used while probing candidate
    (geometry alternative, scale) pairs -- it never causes a waypoint to be skipped or dropped
    from an *accepted* trajectory, because an accepted path is always re-validated end to end.
    """
    n = positions.shape[0]
    data = data if data is not None else model_context.new_data()

    q_reference = np.zeros((n, model_context.nq), dtype=np.float64)
    reachable = np.zeros(n, dtype=bool)
    position_errors_m = np.zeros(n, dtype=np.float64)
    orientation_errors_rad = np.zeros(n, dtype=np.float64)
    restarts_used = np.zeros(n, dtype=np.int32)

    q_warm = np.asarray(q_anchor, dtype=np.float64).copy()
    first_failure_index = -1

    for i in range(n):
        record = solve_reference_configuration(
            model_context,
            settings,
            q_warm_start=q_warm,
            q_anchor=q_anchor,
            target_position=positions[i],
            target_quaternion=quaternions[i],
            restart_seed=derive_seed(path_seed, i),
            data=data,
        )
        q_reference[i] = record["q_reference"]
        reachable[i] = record["reachable"]
        position_errors_m[i] = record["position_error_m"]
        orientation_errors_rad[i] = record["orientation_error_rad"]
        restarts_used[i] = record["attempts_used"] - 1
        q_warm = record["q_reference"]

        if not record["reachable"]:
            if first_failure_index < 0:
                first_failure_index = i
            if stop_on_first_failure:
                return {
                    "all_reachable": False,
                    "first_failure_index": first_failure_index,
                    "waypoints_checked": i + 1,
                    "q_reference": q_reference[: i + 1],
                    "reachable": reachable[: i + 1],
                    "position_errors_m": position_errors_m[: i + 1],
                    "orientation_errors_rad": orientation_errors_rad[: i + 1],
                    "restarts_used": restarts_used[: i + 1],
                }

    return {
        "all_reachable": bool(np.all(reachable)),
        "first_failure_index": first_failure_index,
        "waypoints_checked": n,
        "q_reference": q_reference,
        "reachable": reachable,
        "position_errors_m": position_errors_m,
        "orientation_errors_rad": orientation_errors_rad,
        "restarts_used": restarts_used,
    }


def error_distribution(position_errors_m: np.ndarray, orientation_errors_rad: np.ndarray) -> dict:
    """Max/P95/median summary of a path's independent FK reconstruction errors."""
    return {
        "position_max_m": float(np.max(position_errors_m)),
        "position_p95_m": float(np.percentile(position_errors_m, 95)),
        "position_median_m": float(np.median(position_errors_m)),
        "orientation_max_deg": float(np.degrees(np.max(orientation_errors_rad))),
        "orientation_p95_deg": float(np.degrees(np.percentile(orientation_errors_rad, 95))),
        "orientation_median_deg": float(np.degrees(np.median(orientation_errors_rad))),
    }
