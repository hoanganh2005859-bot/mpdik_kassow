"""Builds Dataset v2's root ``DATASET_MANIFEST.json`` (spec section M).

Declares scope, all [LOCKED] counts from spec section B, split sizes, and pointers to the
checksum manifest / generation report -- but never claims data has been generated or frozen; that
only becomes true once an actual generation phase runs (Phase 1 does not).
"""

from dataset_v2.config_templates import (
    CORE_SHAPES,
    DATASET_VERSION,
    DIFFICULTY_GROUPS,
    ORIENTATION_MODES,
)


def build_dataset_manifest() -> dict:
    point_ik_samples_per_group = 1000
    point_ik_split_sizes = {"development": 1200, "validation": 1200, "frozen_test": 3600}
    core_total = len(CORE_SHAPES) * len(ORIENTATION_MODES) * 12
    random_challenge_split_sizes = {"development": 30, "validation": 30, "frozen_test": 30}
    random_challenge_total = sum(random_challenge_split_sizes.values())
    trajectories_total = core_total + random_challenge_total
    canonical_waypoints_per_trajectory = 400

    return {
        "dataset_name": "kassow-kr810-dataset-v2-tier0-tier4-kinematics",
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
        "counts": {
            "tier0": {"fk_states": 1000, "jacobian_states": 1000, "singularity_states": 600},
            "point_ik": {
                "total_samples": point_ik_samples_per_group * len(DIFFICULTY_GROUPS),
                "samples_per_group": point_ik_samples_per_group,
                "groups": len(DIFFICULTY_GROUPS),
                "split_sizes": point_ik_split_sizes,
            },
            "anchors": {"total": 12, "regular": 6, "near_limit": 3, "near_singular": 3},
            "core_trajectories": {
                "total": core_total,
                "shapes": len(CORE_SHAPES),
                "orientation_modes": len(ORIENTATION_MODES),
                "anchors": 12,
            },
            "random_challenge_trajectories": {
                "total": random_challenge_total,
                "split_sizes": random_challenge_split_sizes,
            },
            "trajectories_total": trajectories_total,
            "canonical_waypoints_per_trajectory": canonical_waypoints_per_trajectory,
            "canonical_poses_total": trajectories_total * canonical_waypoints_per_trajectory,
            "trials_total": trajectories_total * 3,
        },
        "pointers": {
            "checksum_manifest": "checksums/CHECKSUM_MANIFEST.json",
            "generation_report": "reports/GENERATION_REPORT.json",
        },
        "notes": [
            "Phase 1 scaffold only: no Tier 0-4 data has been generated for Dataset v2.",
            "See docs/V2_IMPLEMENTATION_LOG.md and specs/DLS_DATASET_V2_SPEC.md for status and design.",
        ],
    }


def apply_tier0_generation_status(
    manifest: dict,
    fk_count: int,
    jacobian_count: int,
    singularity_count: int,
    fk_group_counts: dict,
    jacobian_group_counts: dict,
    singularity_group_counts: dict,
    full_locked_counts: bool,
) -> dict:
    """Return a copy of ``manifest`` with ``counts.tier0`` updated to the *actual* generated
    counts (Phase 2). Never mutates dataset-wide ``generated``/``frozen``/``status`` -- Tier 0
    being generated does not mean Tier 1-4 are, so those flags stay whatever the caller already
    has recorded (see spec section M: the whole-dataset flags only flip once every tier is done).
    """
    updated = dict(manifest)
    counts = dict(updated.get("counts", {}))
    counts["tier0"] = {
        "fk_states": int(fk_count),
        "jacobian_states": int(jacobian_count),
        "singularity_states": int(singularity_count),
        "generated": True,
        "full_locked_counts": bool(full_locked_counts),
        "fk_group_counts": dict(fk_group_counts),
        "jacobian_group_counts": dict(jacobian_group_counts),
        "singularity_group_counts": dict(singularity_group_counts),
    }
    updated["counts"] = counts
    return updated


def apply_point_ik_generation_status(
    manifest: dict,
    total_samples: int,
    group_counts: dict,
    split_counts: dict,
    group_split_counts: dict,
    full_locked_counts: bool,
) -> dict:
    """Return a copy of ``manifest`` with ``counts.point_ik`` updated to the *actual* generated
    counts (Phase 3). Never mutates dataset-wide ``generated``/``frozen``/``status`` -- Point-IK
    being generated does not mean Tier 0/2-4 are (see ``apply_tier0_generation_status``).
    """
    updated = dict(manifest)
    counts = dict(updated.get("counts", {}))
    counts["point_ik"] = {
        **dict(counts.get("point_ik", {})),
        "total_samples": int(total_samples),
        "generated": True,
        "full_locked_counts": bool(full_locked_counts),
        "group_counts": dict(group_counts),
        "split_counts": dict(split_counts),
        "group_split_counts": {name: dict(value) for name, value in group_split_counts.items()},
    }
    updated["counts"] = counts
    return updated
