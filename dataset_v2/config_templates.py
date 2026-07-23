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

# Phase 2.5 threshold calibration results (docs/V2_THRESHOLD_CALIBRATION.md), locked for reuse
# across Tier 1 Point-IK difficulty groups, the 12 anchor configurations, and trial difficulty
# covariates. Derived from a 50,000-candidate generic pool (calibration seed 42) via
# dataset_v2.threshold_calibration.calibrate; near_joint_limit uses the pool's own P10 quantile of
# the normalized joint-limit margin (kinematics/joint_limit_utils.py::minimum_joint_limit_margin);
# near_singularity reuses v1's configs/dls_config.json:singularity_sigma_threshold unchanged
# (already reused by generators/_trajectory_common.py::select_anchor and Tier 0's own singularity
# classifier); moderately_conditioned_upper_bound reuses the same 3.0 multiplier both of those
# already use.
CALIBRATION_SEED = 42
CALIBRATION_GENERIC_POOL_SIZE = 50000
CALIBRATION_SINGULARITY_POOL_SIZE = 20000
NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD = 0.024991237796029034
NEAR_JOINT_LIMIT_ABSOLUTE_MARGIN_RAD_DIAGNOSTIC = 0.10238099902013219
NEAR_SINGULARITY_SIGMA_MIN_THRESHOLD = 0.03
MODERATELY_CONDITIONED_UPPER_BOUND = 0.09
MODERATE_UPPER_MULTIPLIER = 3.0
REGULAR_MIN_SIGMA_MIN = MODERATELY_CONDITIONED_UPPER_BOUND
REGULAR_MIN_NORMALIZED_MARGIN = NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD
CLASSIFICATION_PRIORITY_HIGHEST_FIRST = [
    "near_singularity",
    "near_joint_limit",
    "large_orientation_change",
    "far_target",
    "medium_target",
    "near_target",
]


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
        "fk_groups": [
            "zero_or_home",
            "random_interior",
            "near_operational_lower_limit",
            "near_operational_upper_limit",
            "mixed_near_limits",
        ],
        "jacobian_groups": [
            "regular",
            "near_lower_limit",
            "near_upper_limit",
            "mixed_near_limits",
            "low_sigma",
        ],
        "singularity_groups": ["regular", "moderately_conditioned", "near_singular"],
        "sampling_policy": {
            "interior_margin_rad": 0.15,
            "near_limit_margin_rad": 0.05,
            "near_limit_band_rad": 0.05,
            "home_perturbation_rad": 0.10,
            "finite_difference_epsilon": 1e-6,
            "jacobian_low_sigma_candidate_pool_multiplier": 20,
            "jacobian_low_sigma_candidate_pool_min": 2000,
            "singularity_candidate_pool_size": 20000,
            "singularity_moderate_upper_multiplier": 3.0,
        },
        "singularity_threshold_source": (
            "repo root configs/dls_config.json:singularity_sigma_threshold (v1's shared DLS "
            "config, reused unchanged and not duplicated into Dataset v2)"
        ),
        "status": "counts_locked_generation_implemented",
    }


def point_ik_config() -> dict:
    samples_per_group = 1000
    split_sizes_per_group = {"development": 200, "validation": 200, "frozen_test": 600}
    split_sizes = {"development": 1200, "validation": 1200, "frozen_test": 3600}
    return {
        "difficulty_groups": list(DIFFICULTY_GROUPS),
        "samples_per_group": samples_per_group,
        "total_samples": samples_per_group * len(DIFFICULTY_GROUPS),
        "split_sizes": split_sizes,
        "split_sizes_per_group": split_sizes_per_group,
        "sample_id_pattern": "pik_{split}_{difficulty_group}_{index:05d}",
        "q_target_usage_policy": (
            "q_target_reference is a reference/provenance value only (proves reachability via "
            "FK and enables joint-space/redundancy analysis); it must never be used as q_initial "
            "for any IK solve evaluating this sample."
        ),
        "difficulty_threshold_config_file": "difficulty_thresholds.json",
        "near_joint_limit_and_near_singularity_status": (
            "thresholds locked in configs/difficulty_thresholds.json (Phase 2.5, "
            "docs/V2_THRESHOLD_CALIBRATION.md), applied as the pair-min of "
            "(minimum_initial_limit_margin_normalized, minimum_target_limit_margin_normalized) and "
            "(initial_sigma_min, target_sigma_min) respectively -- same single-configuration "
            "threshold value reused unchanged, not re-derived per pair."
        ),
        "pair_pool_policy": {
            "pool_size_default": 150000,
            "pool_size_formula": (
                "max(samples_per_group * n_groups * 25, 30000) -- same ratio "
                "generators/generate_point_ik_dataset.py uses for v1's 30000-candidate pool"
            ),
            "interior_margin_fraction": 0.01,
            "interior_margin_note": (
                "q_initial is sampled with a margin proportional to each joint's own half-range "
                "(1% of half-range), not v1's flat 0.10 rad -- this avoids the joint_2/joint_4 "
                "margin-sampling bias documented in docs/V2_THRESHOLD_CALIBRATION.md section 1 "
                "(a flat-rad margin structurally under-represents the narrow-range joints among "
                "near-joint-limit candidates). q_target_reference is then q_initial perturbed by a "
                "random unit direction scaled by a log-uniform magnitude and clipped back into the "
                "operational limits -- reused unchanged from v1's generic-pool construction."
            ),
            "magnitude_log_min": -2.0,
            "magnitude_log_max": 0.5,
            "position_low_quantile": 1.0 / 3.0,
            "position_high_quantile": 2.0 / 3.0,
            "orientation_top_quantile": 0.85,
            "quantile_note": (
                "position 33rd/66th percentile and orientation 85th percentile boundaries reused "
                "unchanged from v1's generators/generate_point_ik_dataset.py (well-defined, "
                "non-overlapping, no unintended gaps); computed fresh at generation time from this "
                "phase's own candidate pool (pool_size_default above), never copied from v1's "
                "stored quantile values."
            ),
        },
        "diversity_selection_policy": {
            "method": "stratified deterministic selection over quantile-binned covariates",
            "covariates": [
                "initial_joint_space_radius_rad",
                "target_workspace_radius_m",
                "orientation_distance_rad",
                "position_distance_m",
                "pair_sigma_min",
                "pair_limit_margin_normalized",
            ],
            "bins_per_covariate": 4,
            "selection_procedure": (
                "each eligible candidate pair is assigned a composite stratum key from "
                "quantile-binning each covariate (bin edges computed over that difficulty group's "
                "own eligible candidate pool); a seeded permutation (derived from the master seed) "
                "sets the draw order within each stratum; the group's exact quota (1000) is drawn "
                "round-robin across occupied strata until met or the pool is exhausted (an "
                "exhausted pool raises an actionable error reporting the group and candidate count "
                "rather than relaxing a threshold or duplicating a sample)."
            ),
            "diversity_note": (
                "prevents concentrating a difficulty group's 1000 samples in one joint-space or "
                "Cartesian-workspace region; never uses DLS/solver outcome; never changes a "
                "locked/derived threshold; see spec section 6."
            ),
        },
        "status": "counts_locked_generation_implemented",
    }


def difficulty_threshold_config() -> dict:
    """Phase 2.5 calibrated thresholds shared by Point-IK, anchors, and trial covariates.

    See docs/V2_THRESHOLD_CALIBRATION.md for the full derivation, distribution report, and
    reproducibility command. Values here are locked (supported by the calibration run recorded in
    that document), not guessed -- generation itself (Point-IK/anchor/trial phases) is still not
    implemented.
    """
    return {
        "status": "locked",
        "calibration_source": "docs/V2_THRESHOLD_CALIBRATION.md",
        "calibration_seed": CALIBRATION_SEED,
        "candidate_pool_size": {
            "generic": CALIBRATION_GENERIC_POOL_SIZE,
            "singularity_biased": CALIBRATION_SINGULARITY_POOL_SIZE,
        },
        "near_joint_limit": {
            "metric": "normalized_joint_limit_margin",
            "metric_source": "kinematics/joint_limit_utils.py::minimum_joint_limit_margin",
            "definition": "normalized",
            "threshold_normalized": NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD,
            "threshold_percentile": 10,
            "rationale": (
                "P10 of the generic candidate pool's per-configuration normalized minimum "
                "joint-limit margin; normalized (not absolute-rad) because KR810's operational "
                "half-ranges differ ~2.9x between joint_2/joint_4 (~2.18 rad) and the other five "
                "joints (~6.28 rad) -- an absolute-rad threshold was found to make joint_2/joint_4 "
                "control the ranking ~6x more often than the other joints (see calibration doc)."
            ),
            "absolute_threshold_rad_diagnostic": NEAR_JOINT_LIMIT_ABSOLUTE_MARGIN_RAD_DIAGNOSTIC,
            "absolute_threshold_status": "diagnostic_only_not_used_for_classification",
        },
        "near_singularity": {
            "metric": "sigma_min",
            "metric_source": "kinematics/singularity_metrics.py::minimum_singular_value",
            "threshold_sigma_min": NEAR_SINGULARITY_SIGMA_MIN_THRESHOLD,
            "threshold_source": (
                "reused unchanged from repo root configs/dls_config.json:"
                "singularity_sigma_threshold (v1 shared DLS config); already reused by "
                "generators/_trajectory_common.py::select_anchor (ANCHOR_SIGMA_RATIO=3.0) and by "
                "dataset_v2/tier0_generation.py's Tier 0 singularity-state classifier"
            ),
        },
        "moderately_conditioned": {
            "upper_bound_sigma_min": MODERATELY_CONDITIONED_UPPER_BOUND,
            "multiplier": MODERATE_UPPER_MULTIPLIER,
            "multiplier_source": (
                "same 3.0 multiplier already used by select_anchor's ANCHOR_SIGMA_RATIO and by "
                "Tier 0's singularity_moderate_upper_multiplier"
            ),
        },
        "regular": {
            "min_sigma_min": REGULAR_MIN_SIGMA_MIN,
            "min_normalized_joint_limit_margin": REGULAR_MIN_NORMALIZED_MARGIN,
            "note": "regular = sigma_min > moderately_conditioned upper bound AND normalized margin > near_joint_limit threshold (non-overlapping complement of the other two classes).",
        },
        "classification_priority_highest_first": list(CLASSIFICATION_PRIORITY_HIGHEST_FIRST),
        "classification_priority_note": (
            "A candidate qualifying for more than one difficulty group is assigned to the "
            "highest-priority group in classification_priority_highest_first; matches v1's "
            "generators/generate_point_ik_dataset.py::PRIORITY_ORDER unchanged."
        ),
    }


ANCHOR_CLASS_PRIORITY_HIGHEST_FIRST = ["near_singular", "near_limit", "regular"]


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
                "status": "locked",
                "threshold_normalized_joint_limit_margin": NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD,
                "note": (
                    "normalized_joint_limit_margin <= threshold_normalized_joint_limit_margin "
                    "(see configs/difficulty_thresholds.json, docs/V2_THRESHOLD_CALIBRATION.md), "
                    "while remaining a valid, reachable configuration (never an actual limit "
                    "violation)."
                ),
            },
            "near_singular": {
                "status": "locked",
                "threshold_sigma_min": NEAR_SINGULARITY_SIGMA_MIN_THRESHOLD,
                "note": (
                    "sigma_min <= threshold_sigma_min (see configs/difficulty_thresholds.json, "
                    "docs/V2_THRESHOLD_CALIBRATION.md), while remaining numerically solvable by "
                    "the existing DLS solver."
                ),
            },
        },
        "classification_priority_highest_first": list(ANCHOR_CLASS_PRIORITY_HIGHEST_FIRST),
        "classification_priority_note": (
            "A candidate satisfying more than one class's eligibility criteria (e.g. both "
            "near_singular and near_limit) is reported with its primary_class set to the "
            "highest-priority eligible class; both diagnostic flags (is_near_limit/"
            "is_near_singular) are always stored regardless of the primary_class assignment."
        ),
        "split_assignment": {
            "splits": list(SPLITS),
            "counts_per_class_per_split": {
                "regular": {"development": 2, "validation": 2, "frozen_test": 2},
                "near_limit": {"development": 1, "validation": 1, "frozen_test": 1},
                "near_singular": {"development": 1, "validation": 1, "frozen_test": 1},
            },
            "method": (
                "each class's diversity-selected anchors are assigned to splits via a seeded "
                "deterministic permutation, sliced into the per-split counts above -- never a "
                "post-hoc random split that could unbalance a class across splits."
            ),
        },
        "candidate_pool_policy": {
            "regular_pool_size_default": 5000,
            "near_limit_biased_pool_size_default": 5000,
            "singularity_biased_pool_size_default": 5000,
            "regular_interior_margin_fraction": 0.01,
            "regular_pool_note": (
                "uniform-interior pool (dataset_v2/tier0_generation.py::_group_random_interior), "
                "margin proportional to each joint's own half-range (1% of half-range), same "
                "rationale as Point-IK/threshold-calibration's generic pool -- avoids the "
                "joint_2/joint_4 margin-sampling bias documented in "
                "docs/V2_THRESHOLD_CALIBRATION.md section 1."
            ),
            "near_limit_biased_pool_note": (
                "reuses dataset_v2/tier0_generation.py::_group_mixed_near_limits unchanged "
                "(each joint independently randomized near-lower/near-upper/interior, giving "
                "natural controlling-joint diversity); margin/band/interior_margin_rad sourced "
                "from configs/tier0_config.json:sampling_policy (single source, never copied)."
            ),
            "singularity_biased_pool_note": (
                "reuses dataset_v2/tier0_generation.py::_build_singularity_candidate_pool "
                "unchanged (uniform + elbow(q4~0) + wrist(q6~0) biased subsets); "
                "interior_margin_rad sourced from configs/tier0_config.json:sampling_policy. Bias "
                "only proposes candidates -- classification always uses the real computed "
                "sigma_min, never the bias label."
            ),
        },
        "overlap_policy": {
            "near_limit_preference": (
                "prefer candidates with is_near_singular=false (clean) when selecting the 3 "
                "near_limit anchors; fall back to the full near_limit-eligible pool (including "
                "overlap with near_singular) only if the clean subset has fewer than 3 "
                "candidates; never relaxes the near_joint_limit threshold, never duplicates."
            ),
            "near_singular_preference": (
                "prefer candidates whose normalized joint-limit margin is above the "
                "near_joint_limit threshold (clean, i.e. not also near_limit) when selecting the "
                "3 near_singular anchors; same fallback rule as near_limit."
            ),
            "report_fields": ["clean_count", "overlap_count", "selected_source"],
        },
        "diversity_selection_policy": {
            "method": "deterministic greedy farthest-point (max-min) selection over a normalized composite feature vector",
            "feature_groups": [
                "joint_space (7d, min-max normalized by operational range)",
                "workspace_position (3d, min-max normalized over the eligible candidate pool's bounding box)",
                "orientation_log_vector (3d, so3_log(R)/pi)",
                "sigma_min (1d, min-max normalized over the eligible pool)",
                "normalized_joint_limit_margin (1d, min-max normalized over the eligible pool)",
                "controlling_joint_one_hot (7d, near_limit class only, extra-weighted to actively spread controlling joints)",
            ],
            "weighting": (
                "each feature group's sub-vector is divided by sqrt(its own dimensionality) "
                "before concatenation, so groups with more raw dimensions do not dominate "
                "Euclidean distance; the near_limit class's controlling_joint_one_hot group is "
                "additionally scaled by controlling_joint_emphasis."
            ),
            "controlling_joint_emphasis": 2.0,
            "initial_point_selection": (
                "the first selected point is the candidate farthest (in normalized feature "
                "space) from the eligible pool's centroid; ties broken by a seeded permutation "
                "rank (deterministic, not arbitrary)."
            ),
            "tie_breaking": (
                "at every selection step, ties in the max-min diversity score are broken by the "
                "same seeded permutation rank derived from that class's selection seed -- never "
                "unseeded/arbitrary."
            ),
        },
        "near_duplicate_tolerance": {
            "joint_space_rad": 0.05,
            "position_m": 0.005,
            "orientation_rad": 0.01,
            "policy": (
                "if any two selected anchors (regardless of class) fall within all three "
                "tolerances simultaneously and are assigned to different splits, generation "
                "fails loudly (spec section 6 anti-leakage); the tolerance itself is never "
                "silently relaxed."
            ),
        },
        "status": "counts_and_thresholds_locked_generation_implemented",
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
        "difficulty_thresholds.json": difficulty_threshold_config(),
        "anchor_config.json": anchor_config(),
        "trajectory_config.json": trajectory_config(),
        "random_challenge_config.json": random_challenge_config(),
        "trial_config.json": trial_config(),
        "split_policy.json": split_policy_config(),
        "evaluation_defaults.json": evaluation_defaults_config(),
    }
