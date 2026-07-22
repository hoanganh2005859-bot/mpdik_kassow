"""Builders for Dataset v2's config scaffold (``<dataset_v2_root>/configs/*.json``).

Every count/name below is copied from the [LOCKED] sections of ``specs/DLS_DATASET_V2_SPEC.md``
(section B counts, section D naming, section E/K seed and split policy). Anything the spec marks
[BLOCKER]/unresolved (the ``near_limit``/``near_singular`` anchor thresholds, section G) is
represented here as an explicit ``"status": "unresolved"`` value, never a guessed number -- Phase
1 must not invent thresholds the spec has not fixed.

These functions return plain dicts (JSON-serializable, no absolute paths, no dataset_root baked
in); ``dataset_v2.scaffold`` is what writes them to disk.
"""

from typing import Dict

DATASET_SCHEMA_VERSION = "2.0.0"
DATASET_VERSION = "2.0.0-dev"

DIFFICULTY_GROUPS = [
    "near_target",
    "medium_target",
    "far_target",
    "large_orientation_change",
    "near_joint_limit",
    "near_singularity",
]

CORE_SHAPES = ["line", "circle", "figure8", "helix", "free_form"]
ORIENTATION_MODES = ["fixed", "variable"]
SPLITS = ["development", "validation", "frozen_test"]
INIT_CLASSES = ["easy", "medium", "hard"]

# Component/split seed-derivation tags -- fixed, documented tag hierarchy per spec section E.
SEED_COMPONENT_TAGS = {
    "tier0": 10,
    "point_ik": 20,
    "anchors": 30,
    "core_trajectories": 40,
    "random_challenge": 50,
    "trials": 60,
}
SEED_SPLIT_TAGS = {"development": 1, "validation": 2, "frozen_test": 3}


def dataset_config() -> dict:
    return {
        "dataset_name": "kassow-kr810-dataset-v2-tier0-tier4-kinematics",
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "status": "scaffold",
        "generated": False,
        "frozen": False,
        "scope": {
            "tiers_included": [
                "tier0_kinematics_validation",
                "tier1_point_ik",
                "tier2_anchors_and_core_trajectories",
                "tier3_random_challenge_trajectories",
                "tier4_trials",
            ],
            "includes_dynamic_control": False,
            "includes_ppo": False,
            "includes_mpdik": False,
            "includes_mappo": False,
        },
        "notes": [
            "Phase 1 scaffold only: no Tier 0-4 data has been generated for Dataset v2.",
            "See specs/DLS_DATASET_V2_SPEC.md for the full locked design this scaffold follows.",
        ],
    }


def robot_config() -> dict:
    return {
        "name": "Kassow KR810",
        "degrees_of_freedom": 7,
        "simulation_engine": "MuJoCo",
        "note": (
            "Dataset v2 does not duplicate robot assets under its own root; generation reuses "
            "the repository's existing assets/kr810.xml and kinematics/ model loader unchanged."
        ),
    }


def seed_policy_config(master_seed: int) -> dict:
    return {
        "master_seed": int(master_seed),
        "derivation_scheme": (
            "np.random.SeedSequence([master_seed, *tags]) via "
            "generators/_common.py::derive_seed/rng_from, reused unchanged for Dataset v2"
        ),
        "component_tags": dict(SEED_COMPONENT_TAGS),
        "split_tags": dict(SEED_SPLIT_TAGS),
        "rules": [
            "No global numpy.random state is ever used; every random draw comes from an "
            "explicitly derived np.random.Generator.",
            "Every seed used anywhere in v2 generation must be traceable back to "
            "(master_seed, tag_path) and recorded in reports/GENERATION_REPORT.json.",
            "Component/split/item seeds must never overlap in a way that leaks generation "
            "entropy across development/validation/frozen_test.",
        ],
        "status": "policy_defined_not_yet_used_for_generation",
    }


def tier0_config() -> dict:
    return {
        "fk_state_count": 1000,
        "jacobian_state_count": 1000,
        "singularity_state_count": 600,
        "status": "counts_locked_generation_not_implemented",
    }


def point_ik_config() -> dict:
    samples_per_group = 1000
    split_sizes = {"development": 1200, "validation": 1200, "frozen_test": 3600}
    return {
        "difficulty_groups": list(DIFFICULTY_GROUPS),
        "samples_per_group": samples_per_group,
        "total_samples": samples_per_group * len(DIFFICULTY_GROUPS),
        "split_sizes": split_sizes,
        "sample_id_pattern": "pik_{split}_{difficulty_group}_{index:05d}",
        "q_target_usage_policy": (
            "q_target is a reference/provenance value only; it must never be used as q_initial "
            "for any IK solve evaluating this sample."
        ),
        "status": "counts_locked_generation_not_implemented",
    }


def anchor_config() -> dict:
    return {
        "classes": {"regular": 6, "near_limit": 3, "near_singular": 3},
        "total": 12,
        "anchor_id_pattern": "anchor_{class}_{index:02d}",
        "acceptance_criteria": {
            "regular": (
                "sigma_min comfortably above the singularity threshold and joint-limit margin "
                "comfortably interior (reuses v1's generators/_trajectory_common.py::select_anchor "
                "predicate as-is)."
            ),
            "near_limit": {
                "status": "unresolved",
                "note": (
                    "Exact joint-limit-margin threshold not yet fixed by "
                    "specs/DLS_DATASET_V2_SPEC.md (section G [BLOCKER]); must be decided before "
                    "anchor generation can run. Not invented here."
                ),
            },
            "near_singular": {
                "status": "unresolved",
                "note": (
                    "Exact sigma_min threshold not yet fixed by specs/DLS_DATASET_V2_SPEC.md "
                    "(section G [BLOCKER]); must be decided before anchor generation can run. "
                    "Not invented here."
                ),
            },
        },
        "status": "counts_locked_thresholds_unresolved_generation_not_implemented",
    }


def trajectory_config() -> dict:
    return {
        "shapes": list(CORE_SHAPES),
        "orientation_modes": list(ORIENTATION_MODES),
        "anchor_count": 12,
        "total_core_trajectories": len(CORE_SHAPES) * len(ORIENTATION_MODES) * 12,
        "canonical_waypoints_per_trajectory": 400,
        "trajectory_id_pattern": "core_{shape}_{orientation_mode}_{anchor_id}",
        "dual_representation": (
            "canonical (400 waypoints) + high-resolution source; canonical is "
            "resampled/subsampled from source, never independently regenerated"
        ),
        "quaternion_order": "wxyz",
        "status": "counts_locked_generation_not_implemented",
    }


def random_challenge_config() -> dict:
    return {
        "total": 90,
        "split_sizes": {"development": 30, "validation": 30, "frozen_test": 30},
        "trajectory_id_pattern": "challenge_{split}_{index:03d}",
        "reachability_policy": (
            "every control pose validated via sequential-DLS reachability "
            "(generators/_trajectory_common.py::validate_sequential_reachability) before acceptance"
        ),
        "status": "counts_locked_generation_not_implemented",
    }


def trial_config() -> dict:
    return {
        "init_classes": list(INIT_CLASSES),
        "trials_per_trajectory": 3,
        "total_trajectories": 210,
        "total_trials": 630,
        "trial_id_pattern": "{trajectory_id}_trial_{init_class}",
        "status": "counts_locked_generation_not_implemented",
    }


def split_policy_config() -> dict:
    return {
        "splits": list(SPLITS),
        "anti_leakage_dimensions": [
            "anchor_id",
            "point_ik_sample_id",
            "point_ik_content_hash",
            "random_path_seed",
            "trajectory_id",
            "trajectory_content_hash",
            "trial_id",
        ],
        "frozen_test_protocol": [
            "frozen_test must never be used to design a generator, choose a threshold, tune "
            "DLS, or tune PPO/MPDIK in any later phase.",
            "frozen_test may only be run after the evaluation config for that run has been "
            "locked (recorded, immutable, checksummed) -- no iterating on evaluation config "
            "against frozen_test results.",
        ],
        "status": "policy_defined_not_yet_enforced_by_generation",
    }


def evaluation_defaults_config() -> dict:
    return {
        "status": "not_yet_defined",
        "note": (
            "Dataset v2 evaluation acceptance thresholds are not specified by "
            "specs/DLS_DATASET_V2_SPEC.md and must not be invented here; populate this file in "
            "a later phase once thresholds are locked, following the frozen-test protocol in "
            "configs/split_policy.json."
        ),
    }


def all_configs(master_seed: int) -> Dict[str, dict]:
    """Every config-scaffold file, keyed by its filename under ``configs/``."""
    return {
        "dataset_config.json": dataset_config(),
        "robot_config.json": robot_config(),
        "seed_policy.json": seed_policy_config(master_seed),
        "tier0_config.json": tier0_config(),
        "point_ik_config.json": point_ik_config(),
        "anchor_config.json": anchor_config(),
        "trajectory_config.json": trajectory_config(),
        "random_challenge_config.json": random_challenge_config(),
        "trial_config.json": trial_config(),
        "split_policy.json": split_policy_config(),
        "evaluation_defaults.json": evaluation_defaults_config(),
    }
