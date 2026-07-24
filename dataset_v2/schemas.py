"""Builders for Dataset v2's JSON Schema scaffold (``<dataset_v2_root>/schemas/*.json``).

Draft 2020-12, ``additionalProperties: false`` on every record-shaped schema, mirroring the style
of the repo's existing ``schemas/point_ik_schema.json``/``schemas/trajectory_schema.json`` (see
``specs/DLS_DATASET_V2_SPEC.md`` section L). These schemas validate the *metadata/manifest record*
shape locked/provisioned by the spec; they do not need to describe generated NPZ array payloads
in full since Phase 1 generates no data (section L: "extendable without breaking backward
compatibility").
"""

from typing import Dict

SHA256_PATTERN = "^[a-f0-9]{64}$"
SPLIT_ENUM = ["development", "validation", "frozen_test"]
DIFFICULTY_ENUM = [
    "near_target",
    "medium_target",
    "far_target",
    "large_orientation_change",
    "near_joint_limit",
    "near_singularity",
]
ORIENTATION_MODE_ENUM = ["fixed", "variable"]
TRAJECTORY_FAMILY_ENUM = ["line", "circle", "figure8", "helix", "free_form", "random_challenge"]
ANCHOR_CLASS_ENUM = ["regular", "near_limit", "near_singular"]
INIT_CLASS_ENUM = ["easy", "medium", "hard"]

_QUATERNION_WXYZ = {
    "type": "array",
    "description": "Unit quaternion [w, x, y, z] (wxyz order), world frame.",
    "items": {"type": "number"},
    "minItems": 4,
    "maxItems": 4,
}


def dataset_manifest_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "dataset_manifest_schema.json",
        "title": "Dataset v2 Manifest",
        "description": "Root DATASET_MANIFEST.json for a Dataset v2 root.",
        "type": "object",
        "required": ["dataset_name", "dataset_version", "status", "generated", "frozen", "scope"],
        "properties": {
            "dataset_name": {"type": "string", "minLength": 1},
            "dataset_version": {"type": "string", "minLength": 1},
            "status": {"type": "string", "minLength": 1},
            "generated": {
                "type": "boolean",
                "description": "Must be false until Tier 0-4 data has actually been generated.",
            },
            "frozen": {
                "type": "boolean",
                "description": "Must be false until frozen_test has been locked per the frozen-test protocol.",
            },
            "scope": {
                "type": "object",
                "required": [
                    "tiers_included",
                    "includes_dynamic_control",
                    "includes_ppo",
                    "includes_mpdik",
                    "includes_mappo",
                ],
                "properties": {
                    "tiers_included": {"type": "array", "items": {"type": "string"}},
                    "includes_dynamic_control": {"type": "boolean", "const": False},
                    "includes_ppo": {"type": "boolean", "const": False},
                    "includes_mpdik": {"type": "boolean", "const": False},
                    "includes_mappo": {"type": "boolean", "const": False},
                },
                "additionalProperties": False,
            },
            "counts": {"type": "object"},
            "pointers": {"type": "object"},
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }


def generation_config_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "generation_config_schema.json",
        "title": "Dataset v2 Seed/Generation Config",
        "description": "configs/seed_policy.json: the master seed and derivation-tag hierarchy for Dataset v2 generation.",
        "type": "object",
        "required": ["master_seed", "derivation_scheme", "component_tags", "split_tags", "status"],
        "properties": {
            "master_seed": {"type": "integer"},
            "derivation_scheme": {"type": "string", "minLength": 1},
            "component_tags": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
            },
            "split_tags": {
                "type": "object",
                "propertyNames": {"enum": SPLIT_ENUM},
                "additionalProperties": {"type": "integer"},
            },
            "rules": {"type": "array", "items": {"type": "string"}},
            "status": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }


def anchor_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "anchor_schema.json",
        "title": "Dataset v2 Anchor Catalog Record",
        "description": "One row of anchors/anchor_manifest.csv: a single anchor configuration, its class/split, and its recomputable covariates (Phase 4).",
        "type": "object",
        "required": [
            "anchor_id",
            "split",
            "anchor_class",
            "q",
            "position",
            "quaternion_wxyz",
            "sigma_min",
            "sigma_max",
            "condition_number",
            "numerical_rank",
            "minimum_normalized_limit_margin",
            "minimum_absolute_limit_margin_rad",
            "controlling_joint_index",
            "is_near_limit",
            "is_near_singular",
            "is_moderately_conditioned",
            "is_regular",
            "source_seed",
            "content_hash",
        ],
        "properties": {
            "anchor_id": {"type": "string", "pattern": "^anchor_(regular|near_limit|near_singular)_[0-9]{2}$"},
            "split": {"type": "string", "enum": SPLIT_ENUM},
            "anchor_class": {"type": "string", "enum": ANCHOR_CLASS_ENUM},
            "q": {
                "type": "array",
                "description": "Anchor joint configuration [q1..q7] in radians.",
                "items": {"type": "number"},
                "minItems": 7,
                "maxItems": 7,
            },
            "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "quaternion_wxyz": _QUATERNION_WXYZ,
            "sigma_min": {"type": "number", "minimum": 0},
            "sigma_max": {"type": "number", "minimum": 0},
            "condition_number": {"type": "number", "minimum": 0},
            "numerical_rank": {"type": "integer", "minimum": 0},
            "manipulability": {"type": "number", "minimum": 0},
            "minimum_normalized_limit_margin": {"type": "number"},
            "minimum_absolute_limit_margin_rad": {"type": "number"},
            "controlling_joint_index": {"type": "integer", "minimum": 0, "maximum": 6},
            "is_near_limit": {"type": "boolean"},
            "is_near_singular": {"type": "boolean"},
            "is_moderately_conditioned": {"type": "boolean"},
            "is_regular": {"type": "boolean"},
            "source_pool": {"type": "string"},
            "source_seed": {"type": "integer"},
            "content_hash": {"type": "string", "pattern": SHA256_PATTERN},
            "model_fingerprint": {"type": "string"},
            "config_fingerprint": {"type": "string"},
        },
        "additionalProperties": False,
    }


def tier0_state_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "tier0_state_schema.json",
        "title": "Dataset v2 Tier 0 State/Checksum Record",
        "description": "A single Tier 0 validation state (FK, Jacobian, or singularity) plus its checksum-manifest metadata.",
        "type": "object",
        "required": ["state_type", "sample_id", "q_sample", "source_seed"],
        "properties": {
            "state_type": {"type": "string", "enum": ["fk", "jacobian", "singularity"]},
            "sample_id": {"type": "integer", "minimum": 0},
            "q_sample": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 7,
                "maxItems": 7,
            },
            "source_seed": {"type": "integer"},
            "sample_count": {"type": "integer", "minimum": 0},
            "sha256": {"type": "string", "pattern": SHA256_PATTERN},
        },
        "additionalProperties": False,
    }


def point_ik_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "point_ik_schema.json",
        "title": "Dataset v2 Point-IK Sample",
        "description": (
            "A single Tier 1 point-IK sample record (Phase 3). q_target_reference is a "
            "reference/provenance value only -- FK(q_target_reference) produces "
            "target_position/target_quaternion, but q_target_reference must never be used as "
            "q_initial for any IK solve evaluating this sample."
        ),
        "type": "object",
        "required": [
            "sample_id",
            "split",
            "difficulty_group",
            "q_initial",
            "q_target_reference",
            "initial_position",
            "initial_quaternion",
            "target_position",
            "target_quaternion",
            "position_distance_m",
            "orientation_distance_rad",
            "joint_distance_rad",
            "initial_sigma_min",
            "target_sigma_min",
            "initial_sigma_max",
            "target_sigma_max",
            "initial_condition_number",
            "target_condition_number",
            "minimum_initial_limit_margin_normalized",
            "minimum_target_limit_margin_normalized",
            "source_seed",
            "content_hash",
        ],
        "properties": {
            "sample_id": {"type": "string", "pattern": "^pik_(development|validation|frozen_test)_[a-z_]+_[0-9]{5}$"},
            "split": {"type": "string", "enum": SPLIT_ENUM},
            "difficulty_group": {"type": "string", "enum": DIFFICULTY_ENUM},
            "q_initial": {"type": "array", "items": {"type": "number"}, "minItems": 7, "maxItems": 7},
            "q_target_reference": {"type": "array", "items": {"type": "number"}, "minItems": 7, "maxItems": 7},
            "initial_position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "initial_quaternion": _QUATERNION_WXYZ,
            "target_position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "target_quaternion": _QUATERNION_WXYZ,
            "position_distance_m": {"type": "number", "minimum": 0},
            "orientation_distance_rad": {"type": "number", "minimum": 0},
            "joint_distance_rad": {"type": "number", "minimum": 0},
            "initial_sigma_min": {"type": "number", "minimum": 0},
            "target_sigma_min": {"type": "number", "minimum": 0},
            "initial_sigma_max": {"type": "number", "minimum": 0},
            "target_sigma_max": {"type": "number", "minimum": 0},
            "initial_condition_number": {"type": "number", "minimum": 0},
            "target_condition_number": {"type": "number", "minimum": 0},
            "minimum_initial_limit_margin_normalized": {"type": "number"},
            "minimum_target_limit_margin_normalized": {"type": "number"},
            "minimum_initial_limit_margin_rad": {"type": "number", "description": "Diagnostic only, never used for classification."},
            "minimum_target_limit_margin_rad": {"type": "number", "description": "Diagnostic only, never used for classification."},
            "source_seed": {"type": "integer"},
            "content_hash": {"type": "string", "pattern": SHA256_PATTERN},
        },
        "additionalProperties": False,
    }


def trajectory_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "trajectory_schema.json",
        "title": "Dataset v2 Core Trajectory Manifest Record",
        "description": (
            "One core_trajectory_manifest.csv row (Phase 5): family/split/shape/orientation-mode, "
            "anchor inheritance, dual-representation counts, arc-length/angular-displacement "
            "metadata, closure/reachability status, and checksum."
        ),
        "type": "object",
        "required": [
            "trajectory_id",
            "family",
            "split",
            "shape",
            "orientation_mode",
            "anchor_id",
            "anchor_class",
            "source_seed",
            "source_waypoint_count",
            "canonical_waypoint_count",
            "quaternion_convention",
            "duration_s",
            "canonical_control_period_s",
            "arc_length_m",
            "cumulative_angular_displacement_rad",
            "closed_path",
            "reachability_status",
            "reachability_tolerance_position_m",
            "reachability_tolerance_orientation_deg",
            "geometry_alternative_id",
            "accepted_scale",
            "generation_status",
            "content_hash",
            "sha256",
        ],
        "properties": {
            "trajectory_id": {
                "type": "string",
                "pattern": "^core_(line|circle|figure8|helix|free_form)_(fixed|variable)_anchor_(regular|near_limit|near_singular)_[0-9]{2}$",
            },
            "family": {"type": "string", "const": "core"},
            "split": {"type": "string", "enum": SPLIT_ENUM},
            "shape": {"type": "string", "enum": ["line", "circle", "figure8", "helix", "free_form"]},
            "orientation_mode": {"type": "string", "enum": ORIENTATION_MODE_ENUM},
            "anchor_id": {"type": "string"},
            "anchor_class": {"type": "string", "enum": ANCHOR_CLASS_ENUM},
            "anchor_content_hash": {"type": "string"},
            "source_seed": {"type": "integer"},
            "source_waypoint_count": {"type": "integer", "minimum": 401},
            "canonical_waypoint_count": {"type": "integer", "const": 400},
            "quaternion_convention": {"type": "string", "const": "wxyz"},
            "duration_s": {"type": "number", "exclusiveMinimum": 0},
            "canonical_control_period_s": {"type": "number", "exclusiveMinimum": 0},
            "nominal_scale": {"type": "number"},
            "accepted_scale": {"type": "number", "exclusiveMinimum": 0},
            "scale_band": {"type": "string", "minLength": 1},
            "geometry_alternative_id": {"type": "string", "minLength": 1},
            "geometry_alternatives_attempted": {"type": "integer", "minimum": 1},
            "scale_reduction_reason": {"type": "string"},
            "geometry_parameters_json": {"type": "string"},
            "reachability_tolerance_position_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.006},
            "reachability_tolerance_orientation_deg": {"type": "number", "exclusiveMinimum": 0, "maximum": 10.0},
            "canonical_position_reconstruction_max_m": {"type": "number", "minimum": 0},
            "canonical_orientation_reconstruction_max_deg": {"type": "number", "minimum": 0},
            "source_position_reconstruction_max_m": {"type": "number", "minimum": 0},
            "source_orientation_reconstruction_max_deg": {"type": "number", "minimum": 0},
            "canonical_waypoints_reachable": {"type": "integer", "minimum": 0},
            "source_waypoints_reachable": {"type": "integer", "minimum": 0},
            "arc_length_m": {"type": "number", "minimum": 0},
            "cumulative_angular_displacement_rad": {"type": "number", "minimum": 0},
            "closed_path": {"type": "boolean"},
            "closure_position_error_m": {"type": "number", "minimum": 0},
            "closure_orientation_error_rad": {"type": "number", "minimum": 0},
            "reachability_status": {"type": "string", "enum": ["validated", "incomplete"]},
            "reachability_success_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "generation_status": {"type": "string", "enum": ["development"]},
            "model_fingerprint": {"type": "string"},
            "config_fingerprint": {"type": "string"},
            "content_hash": {"type": "string", "pattern": SHA256_PATTERN},
            "sha256": {"type": "string", "pattern": SHA256_PATTERN},
            "source_sha256": {"type": "string", "pattern": SHA256_PATTERN},
        },
        "additionalProperties": False,
    }


CHALLENGE_FAMILY_ENUM = [
    "smooth_random",
    "mixed_curvature",
    "non_planar",
    "large_orientation",
    "near_limit_region",
    "near_singular_region",
]


def challenge_trajectory_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "challenge_trajectory_schema.json",
        "title": "Dataset v2 Random Challenge Trajectory Manifest Record",
        "description": (
            "One challenge_trajectory_manifest.csv row (Phase 6): a random-challenge trajectory "
            "generated from a smooth seeded joint-space reference family through FK. Records "
            "family/split, independent-reachable-start metadata, dual-representation counts, "
            "geometry diagnostics (arc length, angular displacement, curvature, non-planarity), "
            "strict reachability evidence, and checksums."
        ),
        "type": "object",
        "required": [
            "trajectory_id",
            "family",
            "challenge_family",
            "split",
            "source_seed",
            "path_seed",
            "source_waypoint_count",
            "canonical_waypoint_count",
            "quaternion_convention",
            "duration_s",
            "canonical_control_period_s",
            "start_content_hash",
            "arc_length_m",
            "cumulative_angular_displacement_rad",
            "mean_curvature_1_per_m",
            "max_curvature_1_per_m",
            "non_planarity",
            "reachability_status",
            "reachability_tolerance_position_m",
            "reachability_tolerance_orientation_deg",
            "canonical_waypoints_reachable",
            "source_waypoints_reachable",
            "generation_status",
            "content_hash",
            "sha256",
            "source_sha256",
        ],
        "properties": {
            "trajectory_id": {"type": "string", "pattern": "^challenge_(development|validation|frozen_test)_[0-9]{3}$"},
            "family": {"type": "string", "const": "random_challenge"},
            "challenge_family": {"type": "string", "enum": CHALLENGE_FAMILY_ENUM},
            "split": {"type": "string", "enum": SPLIT_ENUM},
            "family_candidate_index": {"type": "integer", "minimum": 0},
            "source_seed": {"type": "integer"},
            "path_seed": {"type": "integer"},
            "frozen_challenge_seed_revision": {"type": "integer", "minimum": 1},
            "source_waypoint_count": {"type": "integer", "minimum": 401},
            "canonical_waypoint_count": {"type": "integer", "const": 400},
            "quaternion_convention": {"type": "string", "const": "wxyz"},
            "duration_s": {"type": "number", "exclusiveMinimum": 0},
            "canonical_control_period_s": {"type": "number", "exclusiveMinimum": 0},
            "start_position": {"type": "string"},
            "start_sigma_min": {"type": "number", "minimum": 0},
            "start_sigma_max": {"type": "number", "minimum": 0},
            "start_condition_number": {"type": "number", "minimum": 0},
            "start_normalized_limit_margin": {"type": "number"},
            "start_absolute_limit_margin_rad": {"type": "number"},
            "start_controlling_joint_index": {"type": "integer", "minimum": 0, "maximum": 6},
            "start_content_hash": {"type": "string", "pattern": SHA256_PATTERN},
            "envelope_margin_fraction": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
            "harmonics_json": {"type": "string"},
            "geometry_parameters_json": {"type": "string"},
            "arc_length_m": {"type": "number", "minimum": 0},
            "cumulative_angular_displacement_rad": {"type": "number", "minimum": 0},
            "mean_curvature_1_per_m": {"type": "number", "minimum": 0},
            "max_curvature_1_per_m": {"type": "number", "minimum": 0},
            "non_planarity": {"type": "number", "minimum": 0},
            "reachability_status": {"type": "string", "enum": ["validated", "incomplete"]},
            "reachability_tolerance_position_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.006},
            "reachability_tolerance_orientation_deg": {"type": "number", "exclusiveMinimum": 0, "maximum": 10.0},
            "canonical_position_reconstruction_max_m": {"type": "number", "minimum": 0},
            "canonical_orientation_reconstruction_max_deg": {"type": "number", "minimum": 0},
            "source_position_reconstruction_max_m": {"type": "number", "minimum": 0},
            "source_orientation_reconstruction_max_deg": {"type": "number", "minimum": 0},
            "canonical_waypoints_reachable": {"type": "integer", "minimum": 0},
            "source_waypoints_reachable": {"type": "integer", "minimum": 0},
            "generation_status": {"type": "string", "enum": ["development"]},
            "model_fingerprint": {"type": "string"},
            "config_fingerprint": {"type": "string"},
            "content_hash": {"type": "string", "pattern": SHA256_PATTERN},
            "sha256": {"type": "string", "pattern": SHA256_PATTERN},
            "source_sha256": {"type": "string", "pattern": SHA256_PATTERN},
        },
        "additionalProperties": False,
    }


def trial_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "trial_schema.json",
        "title": "Dataset v2 Trial Manifest Record",
        "description": "One trials/trial_manifest.csv row: an easy/medium/hard initial-state trial for one trajectory.",
        "type": "object",
        "required": ["trial_id", "trajectory_id", "init_class", "q_initial", "source_seed"],
        "properties": {
            "trial_id": {"type": "string", "pattern": "^.+_trial_(easy|medium|hard)$"},
            "trajectory_id": {"type": "string", "minLength": 1},
            "init_class": {"type": "string", "enum": INIT_CLASS_ENUM},
            "q_initial": {"type": "array", "items": {"type": "number"}, "minItems": 7, "maxItems": 7},
            "initial_position_distance_m": {"type": "number", "minimum": 0},
            "initial_limit_margin": {"type": "number"},
            "initial_sigma_min": {"type": "number", "minimum": 0},
            "source_seed": {"type": "integer"},
        },
        "additionalProperties": False,
    }


def validation_report_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "validation_report_schema.json",
        "title": "Dataset v2 Anti-Leakage / Validation Report",
        "description": "Post-generation report asserting pairwise disjointness of the section-K anti-leakage identifier sets across development/validation/frozen_test.",
        "type": "object",
        "required": ["dimensions_checked", "collisions_found", "pass"],
        "properties": {
            "dimensions_checked": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "anchor_id",
                        "point_ik_sample_id",
                        "point_ik_content_hash",
                        "random_path_seed",
                        "trajectory_id",
                        "trajectory_content_hash",
                        "trial_id",
                    ],
                },
            },
            "collisions_found": {"type": "integer", "minimum": 0},
            "collision_details": {"type": "array", "items": {"type": "object"}},
            "pass": {"type": "boolean"},
        },
        "additionalProperties": False,
    }


def checksum_manifest_schema() -> dict:
    file_entry = {
        "type": "object",
        "required": ["filename", "sha256", "file_size_bytes"],
        "properties": {
            "filename": {
                "type": "string",
                "description": "Path relative to the dataset_v2 root (never absolute).",
                "minLength": 1,
            },
            "sha256": {"type": "string", "pattern": SHA256_PATTERN},
            "file_size_bytes": {"type": "integer", "minimum": 0},
            "sample_count": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": True,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "checksum_manifest_schema.json",
        "title": "Dataset v2 Checksum Manifest",
        "description": "checksums/CHECKSUM_MANIFEST.json: SHA256 of every generated file, split into source/config fingerprint, generated-data checksum, and release-archive checksum categories (spec section N).",
        "type": "object",
        "required": ["categories", "status"],
        "properties": {
            "dataset_root_relative": {"type": "boolean"},
            "categories": {
                "type": "object",
                "required": ["source_config_fingerprint", "generated_data_checksum", "release_archive_checksum"],
                "properties": {
                    "source_config_fingerprint": {"type": "array", "items": file_entry},
                    "generated_data_checksum": {"type": "array", "items": file_entry},
                    "release_archive_checksum": {"type": "array", "items": file_entry},
                },
                "additionalProperties": False,
            },
            "status": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }


def all_schemas() -> Dict[str, dict]:
    """Every schema-scaffold file, keyed by its filename under ``schemas/``."""
    return {
        "dataset_manifest_schema.json": dataset_manifest_schema(),
        "generation_config_schema.json": generation_config_schema(),
        "anchor_schema.json": anchor_schema(),
        "tier0_state_schema.json": tier0_state_schema(),
        "point_ik_schema.json": point_ik_schema(),
        "trajectory_schema.json": trajectory_schema(),
        "challenge_trajectory_schema.json": challenge_trajectory_schema(),
        "trial_schema.json": trial_schema(),
        "validation_report_schema.json": validation_report_schema(),
        "checksum_manifest_schema.json": checksum_manifest_schema(),
    }
