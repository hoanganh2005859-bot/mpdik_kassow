"""Feasibility screening for Dataset v2 anchor candidates (Phase 5.4).

Why this exists
---------------
An anchor's class predicate (``regular``/``near_limit``/``near_singular``) describes the anchor's
own kinematic state. It says nothing about whether the *neighbourhood* of that anchor can support
the locked core trajectory geometry. Three independent seed realizations produced 0, 2 and 4 core
trajectories that could not reach ``minimum_core_accepted_scale`` -- purely as a function of which
anchors happened to be drawn.

This module adds a screen: a candidate may only enter the 12-anchor catalog if **all ten** locked
(shape, orientation_mode) combinations can be certified at ``accepted_scale >= 0.50`` under the
same locked geometry alternatives, scale schedule and strict reachability policy the real
generator uses. Nothing is weakened -- the screen only *removes* candidates that would have
produced a below-gate trajectory.

Coarse probe vs. full verification
----------------------------------
Screening every eligible candidate at full 400-waypoint resolution is not affordable, so the
screen runs at a coarse canonical resolution. The coarse probe is explicitly **not** an
acceptance: whatever it selects is still generated and validated at full resolution, and the
independent core trajectory validator re-checks the final 120 without consulting this module. A
coarse pass that later fails full generation surfaces as a normal generation failure.

Feasibility here means "some alternative reaches the gate", not "the best alternative" -- the real
generator maximizes the accepted scale afterwards, and a maximum can only be >= any particular
success, so a candidate that passes the screen cannot be pushed below the gate by that maximization.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataset_v2.config_templates import CORE_SHAPES, ORIENTATION_MODES
from dataset_v2.seeds import SEED_ALGORITHM_ID, derive_seed

FEASIBILITY_PROBE_TAG = 77

#: Process-lifetime feasibility cache. The key is purely content-derived (candidate ``q``, model
#: fingerprint, geometry-config fingerprint, reachability-config fingerprint, seed algorithm id and
#: probe resolution -- never a path, a dataset root or a timestamp), so an entry is valid for any
#: root whose configs hash the same, and any config change produces a different key rather than a
#: stale hit. A production run screens once and gains nothing; a test session that generates anchors
#: repeatedly reuses the identical probes instead of recomputing them every call.
_PROCESS_FEASIBILITY_CACHE: Dict[str, dict] = {}


def process_feasibility_cache() -> Dict[str, dict]:
    """The shared, content-keyed feasibility cache for this process."""
    return _PROCESS_FEASIBILITY_CACHE


def clear_process_feasibility_cache() -> None:
    """Drop every cached verdict (used by tests that need a cold measurement)."""
    _PROCESS_FEASIBILITY_CACHE.clear()


@dataclass(frozen=True)
class FeasibilitySettings:
    enabled: bool
    required_combinations: int
    minimum_passing_combinations: int
    minimum_scale: float
    coarse_canonical_waypoints: int
    coarse_source_waypoints: int
    max_attempts_per_combination: int
    screening_budget_per_class: Dict[str, int]
    cache_enabled: bool


@dataclass
class FeasibilityStats:
    """Screening bookkeeping, reported verbatim in the anchor generation report."""

    screened: Dict[str, int] = field(default_factory=dict)
    passed: Dict[str, int] = field(default_factory=dict)
    rejected: Dict[str, int] = field(default_factory=dict)
    failure_histogram: Dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    runtime_seconds: float = 0.0


def load_feasibility_settings(paths) -> FeasibilitySettings:
    from utils.config_loader import load_json_config

    anchor_config = load_json_config(paths.configs_dir / "anchor_config.json")
    screening = anchor_config["feasibility_screening"]
    if screening.get("status") != "locked":
        raise ValueError("configs/anchor_config.json:feasibility_screening.status must be 'locked'")
    if screening.get("partial_feasibility_accepted", False):
        raise ValueError("partial anchor feasibility must never be accepted")
    return FeasibilitySettings(
        enabled=bool(screening["enabled"]),
        required_combinations=int(screening["required_combinations"]),
        minimum_passing_combinations=int(screening["minimum_passing_combinations"]),
        minimum_scale=float(screening["minimum_scale"]),
        coarse_canonical_waypoints=int(screening["coarse_probe_canonical_waypoints"]),
        coarse_source_waypoints=int(screening["coarse_probe_source_waypoints"]),
        max_attempts_per_combination=int(screening["max_attempts_per_combination"]),
        screening_budget_per_class={k: int(v) for k, v in screening["screening_budget_per_class"].items()},
        cache_enabled=bool(screening["cache_policy"]["enabled"]),
    )


def locked_combinations() -> List[Tuple[str, str]]:
    """The ten locked (shape, orientation_mode) combinations, in deterministic order."""
    return [(shape, mode) for shape in CORE_SHAPES for mode in ORIENTATION_MODES]


def feasibility_cache_key(
    q: np.ndarray,
    model_fingerprint: str,
    geometry_config_fingerprint: str,
    reachability_config_fingerprint: str,
    settings: FeasibilitySettings,
) -> str:
    """Deterministic cache key -- content only, never an absolute path or a timestamp."""
    payload = {
        "q": [round(float(v), 12) for v in np.asarray(q, dtype=np.float64)],
        "model_fingerprint": model_fingerprint,
        "geometry_config_fingerprint": geometry_config_fingerprint,
        "reachability_config_fingerprint": reachability_config_fingerprint,
        "seed_algorithm_id": SEED_ALGORITHM_ID,
        "coarse_canonical_waypoints": settings.coarse_canonical_waypoints,
        "coarse_source_waypoints": settings.coarse_source_waypoints,
        "minimum_scale": round(float(settings.minimum_scale), 12),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_fingerprints(paths) -> Tuple[str, str]:
    """(geometry_config_fingerprint, reachability_config_fingerprint) for cache invalidation."""
    from utils.config_loader import load_json_config

    trajectory_config = load_json_config(paths.configs_dir / "trajectory_config.json")
    reachability_config = load_json_config(paths.configs_dir / "generation_reachability_config.json")
    geometry_payload = {
        "geometry": trajectory_config["geometry"],
        "geometry_alternatives": trajectory_config["geometry_alternatives"],
        "scale_reduction_policy": trajectory_config["scale_reduction_policy"],
        "minimum_scale_gate": trajectory_config["minimum_scale_gate"],
        "orientation_rotation_angle_rad": trajectory_config["orientation_rotation_angle_rad"],
        "orientation_rotation_axis": trajectory_config["orientation_rotation_axis"],
        "duration_s": trajectory_config["duration_s"],
    }
    geometry_fp = hashlib.sha256(json.dumps(geometry_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    reachability_fp = hashlib.sha256(json.dumps(reachability_config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return geometry_fp, reachability_fp


def probe_combination_feasible(
    model_context,
    q_anchor: np.ndarray,
    shape: str,
    orientation_mode: str,
    settings: FeasibilitySettings,
    core_settings,
    probe_reach_settings,
    path_seed: int,
) -> dict:
    """Can this anchor support one (shape, orientation_mode) at >= the minimum scale?

    The question is existential -- "is *some* scale >= the gate reachable?" -- so the probe tests
    the **gate rung**: the smallest rung of the locked scale schedule that is still >= the minimum
    scale. Smaller geometry stays closer to the anchor and is the easiest rung to satisfy, so this
    is where a feasible candidate is cheapest to confirm.

    Soundness: a success here is a genuine reachable trajectory at a scale >= the gate, so the
    screen cannot falsely accept. The real generator afterwards *maximizes* the accepted scale over
    the same alternatives, and a maximum is always >= this particular success, so it cannot drop
    below the gate either. A failure here is treated as infeasible; that is conservative (a
    candidate might in principle succeed only at a larger rung), and a conservative rejection is
    safe because the candidate is simply replaced from the same class pool.
    """
    # Imported lazily: core_trajectory_generation imports anchor_generation, which imports this
    # module, so a module-level import would close a cycle.
    from dataset_v2.core_trajectory_generation import (
        SHAPE_CLOSED_PATH,
        build_source_positions,
        enumerate_geometry_alternatives,
        resample_canonical,
    )
    from dataset_v2.generation_reachability import validate_path_strict
    from generators._trajectory_common import build_time_and_path_parameter, orientation_arrays

    alternatives = enumerate_geometry_alternatives(shape, core_settings.geometry_alternatives)
    closed_path = SHAPE_CLOSED_PATH[shape]
    from kinematics.forward_kinematics import forward_kinematics

    fk_anchor = forward_kinematics(model_context, q_anchor)

    time_s, tau, s, _a, _b, _c = build_time_and_path_parameter(settings.coarse_source_waypoints, core_settings.duration_s)

    # the gate rung: smallest scale on the locked schedule that is still >= the minimum scale
    schedule = []
    scale = 1.0
    for _ in range(core_settings.max_shrink_attempts):
        if scale < settings.minimum_scale:
            break
        schedule.append(scale)
        scale *= core_settings.shrink_factor
    if not schedule:
        return {"feasible": False, "attempts": 0, "reason": "empty scale schedule", "best_scale": None}
    gate_scale = schedule[-1]

    attempts = 0
    scales_tried: List[float] = [gate_scale]
    for _outer in (0,):
        scale = gate_scale
        for alternative in alternatives:
            if attempts >= settings.max_attempts_per_combination:
                return {
                    "feasible": False,
                    "attempts": attempts,
                    "reason": f"probe budget {settings.max_attempts_per_combination} exhausted",
                    "best_scale": None,
                }
            attempts += 1

            ff_offsets = None
            if shape == "free_form":
                from dataset_v2.core_trajectory_generation import _free_form_unit_offsets

                ff_seed = derive_seed(path_seed, FEASIBILITY_PROBE_TAG, alternative["seed_offset"])
                ff_offsets = _free_form_unit_offsets(
                    np.random.default_rng(ff_seed),
                    int(core_settings.geometry["free_form"]["control_point_count"]),
                    mirror=alternative["mirror"],
                )
            positions, _params = build_source_positions(shape, fk_anchor, scale, s, core_settings.geometry, alternative, ff_offsets)
            rotation_vector = (
                core_settings.orientation_rotation_angle_rad
                * scale
                * float(alternative.get("orientation_sign", 1.0))
                * core_settings.orientation_rotation_axis
                if orientation_mode == "variable"
                else None
            )
            quaternions = orientation_arrays(
                orientation_mode, fk_anchor.rotation_matrix, rotation_vector, s, closed_path, settings.coarse_source_waypoints
            )
            canonical = resample_canonical(time_s, tau, positions, quaternions, settings.coarse_canonical_waypoints)
            result = validate_path_strict(
                model_context,
                probe_reach_settings,
                q_anchor,
                canonical["target_position"],
                canonical["target_quaternion"],
                path_seed=path_seed,
                stop_on_first_failure=True,
            )
            if result["all_reachable"]:
                return {
                    "feasible": True,
                    "attempts": attempts,
                    "reason": None,
                    "best_scale": scale,
                    "alternative_id": alternative["alternative_id"],
                }

    return {
        "feasible": False,
        "attempts": attempts,
        "reason": (
            f"none of the {len(alternatives)} geometry alternatives is strictly reachable at the "
            f"gate scale {gate_scale:.4f} (>= minimum {settings.minimum_scale})"
        ),
        "best_scale": None,
    }


def probe_anchor_feasibility(
    model_context,
    q_anchor: np.ndarray,
    settings: FeasibilitySettings,
    core_settings,
    probe_reach_settings,
    master_seed: int,
    cache: Optional[Dict[str, dict]] = None,
    cache_key: Optional[str] = None,
    stats: Optional[FeasibilityStats] = None,
) -> dict:
    """Screen one candidate against all ten locked combinations.

    Feasible only when **every** combination reaches the minimum scale -- partial feasibility is
    never accepted. Stops at the first failing combination (the verdict cannot change).
    """
    if cache is not None and cache_key is not None and cache_key in cache:
        if stats is not None:
            stats.cache_hits += 1
        return cache[cache_key]
    if stats is not None:
        stats.cache_misses += 1

    combinations = locked_combinations()
    matrix: Dict[str, dict] = {}
    feasible = True
    worst_scale = None
    first_failure = None

    for shape, mode in combinations:
        label = f"{shape}_{mode}"
        path_seed = derive_seed(master_seed, FEASIBILITY_PROBE_TAG, _label_tag(label))
        outcome = probe_combination_feasible(
            model_context, q_anchor, shape, mode, settings, core_settings, probe_reach_settings, path_seed
        )
        matrix[label] = outcome
        if outcome["feasible"]:
            worst_scale = outcome["best_scale"] if worst_scale is None else min(worst_scale, outcome["best_scale"])
        else:
            feasible = False
            first_failure = label
            break  # verdict already decided; remaining combinations cannot rescue it

    verdict = {
        "feasible": feasible,
        "combinations_tested": len(matrix),
        "combinations_required": settings.required_combinations,
        "combinations_passed": sum(1 for v in matrix.values() if v["feasible"]),
        "first_failing_combination": first_failure,
        "worst_accepted_scale": worst_scale,
        "matrix": matrix,
    }
    if cache is not None and cache_key is not None:
        cache[cache_key] = verdict
    return verdict


def _label_tag(label: str) -> int:
    """Deterministic integer tag for a combination label (``zlib.crc32``, never Python ``hash``)."""
    import zlib

    return zlib.crc32(label.encode("utf-8"))
