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

from dataset_v2.seeds import SEED_ALGORITHM_ID

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

# --- Frozen-test seed revision (spec section K frozen-test protocol) ---------------------------
# Phase 5/5.1 generated frozen_test trajectories into temporary roots that were inspected while the
# generator policy was still being designed (anchor predicate, geometry-alternative search,
# reachability tolerance, minimum-scale gate were all tuned with that data visible). Those frozen
# trajectories are therefore burned and must never be shipped as the official frozen_test split.
# Bumping this revision moves frozen_test onto a fresh seed namespace so no regenerated frozen
# content can coincide with anything observed during design.
FROZEN_CORE_SEED_REVISION = 4
FROZEN_CORE_SEED_REVISION_HISTORY = [
    {
        "revision": 1,
        "status": "burned_not_shippable",
        "reason": (
            "frozen_test content was generated into temporary roots and observed during Phase 5 / "
            "Phase 5.1 while the generator policy was still being tuned (anchor class predicate, "
            "geometry-alternative search, strict reachability tolerance, minimum-scale gate)."
        ),
    },
    {
        "revision": 2,
        "status": "burned_not_shippable",
        "reason": "geometry alternative policy expanded after pre-freeze validation",
    },
    {
        "revision": 3,
        "status": "burned_not_shippable",
        "reason": (
            "generated before the Dataset v2 deterministic seed fix; its seeds came from the "
            "NumPy-version-dependent v1 derivation (np.uint64 % int promotes to float64 on NumPy "
            "1.x, stays exact on NumPy 2.x), so revision 3 content is not reproducible across "
            "environments."
        ),
    },
    {
        "revision": 4,
        "status": "active",
        "reason": (
            "fresh frozen_test namespace introduced alongside the Dataset v2 deterministic seed "
            "derivation (dataset_v2/seeds.py, SHA-256 over canonical bytes with pure Python "
            "integer arithmetic). Regenerated frozen_test data is only integrity / strict "
            "reachability / checksum / count validated -- it is never used to tune the alternative "
            "list, thresholds, scale gate, anchors or tolerances."
        ),
    },
]

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
            "dataset_v2/seeds.py::derive_seed(master_seed, *tags) -- SHA-256 over a canonical byte "
            "encoding, reduced with pure Python integer arithmetic, so every derived seed is "
            "independent of the installed NumPy version. Dataset v1's "
            "generators/_common.py::derive_seed is left untouched (it finishes with "
            "np.uint64 % int, which promotes to float64 on NumPy 1.x and stays exact on NumPy 2.x, "
            "making its output NumPy-version-dependent)."
        ),
        "seed_algorithm_id": SEED_ALGORITHM_ID,
        "component_tags": dict(SEED_COMPONENT_TAGS),
        "split_tags": dict(SEED_SPLIT_TAGS),
        "frozen_core_seed_revision": FROZEN_CORE_SEED_REVISION,
        "frozen_core_seed_revision_history": [dict(entry) for entry in FROZEN_CORE_SEED_REVISION_HISTORY],
        "frozen_core_seed_policy": (
            "frozen_test-specific seeds (anchor split assignment and core-trajectory path seeds) "
            "mix in frozen_core_seed_revision; development/validation keep their existing seed "
            "namespace/tags. Note that anchor split *membership* is drawn from a single per-class "
            "permutation with exact 2/1/1 quotas, so bumping the frozen revision necessarily "
            "re-permutes that assignment for all three splits -- with exact quotas there is no way "
            "to redraw frozen membership while holding development/validation membership fixed. "
            "This is acceptable and intended here because Phase 5.2 regenerates every anchor and "
            "every trajectory anyway (the anchor class predicate changed), so no split retains "
            "any prior content. Core-trajectory path seeds, by contrast, are per-trajectory and "
            "only frozen_test trajectories mix in the revision."
        ),
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

# --- Phase 5.2 anchor class isolation ---------------------------------------------------------
# Anchor-selection-only conditioning floor for the near_limit class. Reuses the already-calibrated
# MODERATELY_CONDITIONED_UPPER_BOUND (0.09) rather than introducing a new number: a near_limit
# anchor must be *only* near a joint limit, never simultaneously near-singular. Phase 5.1 measured
# the cost of allowing the compound case -- anchor_near_limit_02 (margin 0.00824 AND sigma_min
# 0.06254) forced its closed-shape trajectories down to scale 0.12/0.20. This constant changes the
# anchor-class eligibility predicate ONLY; the global difficulty-group definitions in
# difficulty_thresholds.json and the Point-IK classification priority are untouched.
ANCHOR_NEAR_LIMIT_MIN_SIGMA_MIN = MODERATELY_CONDITIONED_UPPER_BOUND
ANCHOR_NEAR_SINGULAR_MIN_NORMALIZED_MARGIN = NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD

# --- Phase 5.1 strict generation reachability (independent of the DLS baseline) ----------------
# These tolerances define what "this target pose has a valid reference solution" means at
# GENERATION time. They are deliberately NOT read from configs/dls_config.json (Dataset v1's DLS
# baseline evaluation config) -- a dataset whose reachability is defined by the very solver it
# will later benchmark cannot measure that solver. They are ~60x (position) and ~1000x
# (orientation) tighter than the DLS baseline's success thresholds (0.006 m / 10.0 deg).
GENERATION_POSITION_TOLERANCE_M = 1e-4
GENERATION_ORIENTATION_TOLERANCE_DEG = 0.01
SCALE_DIAGNOSTIC_BANDS = [1.0, 0.75, 0.5, 0.25]

# Phase 5.2 [LOCKED]: minimum geometric scale a core trajectory may be accepted at. Rationale:
# preserve at least half of the nominal core trajectory geometry, so a "circle" benchmark stays a
# recognisably-sized circle rather than a few-millimetre loop.
MINIMUM_CORE_ACCEPTED_SCALE = 0.50


def generation_reachability_config() -> dict:
    """Strict, generation-only reachability policy (Phase 5.1, spec section H.1).

    Acceptance of a generated target pose depends on an *independent FK reconstruction* check
    against the tolerances below -- never on the numerical IK engine's own ``success`` flag, and
    never on any Dataset v1 DLS baseline evaluation threshold.
    """
    return {
        "status": "locked",
        "independence": {
            "reads_dls_evaluation_thresholds": False,
            "note": (
                "position/orientation tolerances and the generation solver settings below are "
                "defined entirely within Dataset v2 and are never read from configs/dls_config.json "
                "(Dataset v1's DLS baseline evaluation config). Changing the DLS baseline's "
                "success thresholds must not change which trajectories this generator accepts -- "
                "asserted by tests/test_dataset_v2_core_trajectory_reachability.py."
            ),
            "dls_baseline_position_threshold_m_for_reference_only": 0.006,
            "dls_baseline_orientation_threshold_deg_for_reference_only": 10.0,
        },
        "position_reconstruction_tolerance_m": GENERATION_POSITION_TOLERANCE_M,
        "orientation_reconstruction_tolerance_deg": GENERATION_ORIENTATION_TOLERANCE_DEG,
        "acceptance_rule": (
            "a canonical/source waypoint is 'reachable' if and only if a q_reference exists that "
            "(a) lies within operational joint limits and (b) whose INDEPENDENTLY recomputed "
            "FK(q_reference) reproduces the target position and orientation within the tolerances "
            "above. The numerical IK engine's success flag is never sufficient."
        ),
        "generation_solver": {
            "engine": (
                "kinematics/dls_solver.py::solve_dls_until_converged, used as a generation-time "
                "numerical IK engine only -- this is NOT a DLS baseline evaluation result and must "
                "never be reported as one. The solver math is reused unchanged; only the config "
                "values below (which are Dataset v2's own) differ from v1's."
            ),
            "max_iterations": 400,
            "position_success_threshold_m": GENERATION_POSITION_TOLERANCE_M * 0.5,
            "orientation_success_threshold_deg": GENERATION_ORIENTATION_TOLERANCE_DEG * 0.5,
            "position_weight": 1.0,
            "orientation_weight": 1.0,
            "damping_mode": "adaptive",
            "lambda_default": 0.01,
            "lambda_min": 1e-6,
            "lambda_max": 0.05,
            "step_scale": 1.0,
            "max_joint_step_rad": 0.1,
            "joint_limit_avoidance": False,
            "null_space_gain": 0.0,
            "clip_to_operational_limits": True,
            "singularity_sigma_threshold": 0.03,
            "solver_threshold_note": (
                "the solver's own success thresholds are set to half the acceptance tolerances so "
                "the independent FK check has headroom; acceptance is still decided only by the "
                "independent FK check. lambda_min/lambda_max are tighter than v1's baseline so the "
                "generation engine can converge to 1e-4 m; joint_limit_avoidance/null_space_gain "
                "are disabled so the reference solution is a pure pose solution."
            ),
        },
        "refinement_policy": {
            "max_refinement_rounds": 6,
            "note": (
                "after each solve, FK(q_solution) is checked independently; if it misses the "
                "tolerance the solver is re-entered warm-started from that solution (a fresh call "
                "resets the solver's stagnation window, which otherwise stops descent well above "
                "1e-4 m). Never relaxes the tolerance."
            ),
        },
        "search_probe_policy": {
            "max_refinement_rounds": 2,
            "max_restarts": 0,
            "note": (
                "cheap probe settings used ONLY while searching over (geometry alternative, scale) "
                "candidates. The probe is strictly WEAKER than the full acceptance policy (fewer "
                "refinement rounds, no restarts), so it can only ever under-estimate reachability "
                "-- it can never accept a waypoint the full policy would reject. Whichever "
                "candidate the search selects is then re-validated end-to-end with the FULL policy "
                "on both the canonical and the source path before the trajectory is written; that "
                "full re-validation, not the probe, is what the accepted dataset guarantees. "
                "Rationale: a single unreachable waypoint costs refinement_rounds x (1 + restarts) "
                "solver calls, which dominates search time at near-singular anchors."
            ),
        },
        "restart_policy": {
            "max_restarts": 3,
            "restart_perturbation_rad": 0.05,
            "note": (
                "if warm-start + refinement still misses the tolerance, deterministic restarts are "
                "tried: first from the trajectory's anchor configuration, then from seeded "
                "perturbations of the previous waypoint's reference (seed derived from the master "
                "seed + trajectory tag + waypoint index -- never unseeded, never global RNG). A "
                "waypoint that still fails makes the whole (geometry alternative, scale) attempt "
                "fail; waypoints are never skipped or dropped."
            ),
        },
        "path_validation_scope": {
            "canonical_path": "every one of the 400 canonical waypoints",
            "source_path": (
                "every one of the high-resolution source waypoints; validated once for the "
                "finally-accepted (geometry alternative, scale) so the shipped source "
                "representation carries the same strict guarantee as the canonical one"
            ),
        },
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
                "status": "locked",
                "threshold_normalized_joint_limit_margin": NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD,
                "min_sigma_min_exclusive": ANCHOR_NEAR_LIMIT_MIN_SIGMA_MIN,
                "note": (
                    "normalized_joint_limit_margin <= threshold_normalized_joint_limit_margin "
                    "(see configs/difficulty_thresholds.json, docs/V2_THRESHOLD_CALIBRATION.md) "
                    "AND sigma_min > min_sigma_min_exclusive (Phase 5.2 class isolation), while "
                    "remaining a valid, reachable configuration (never an actual limit violation)."
                ),
            },
            "near_singular": {
                "status": "locked",
                "threshold_sigma_min": NEAR_SINGULARITY_SIGMA_MIN_THRESHOLD,
                "min_normalized_joint_limit_margin_exclusive": ANCHOR_NEAR_SINGULAR_MIN_NORMALIZED_MARGIN,
                "note": (
                    "sigma_min <= threshold_sigma_min (see configs/difficulty_thresholds.json, "
                    "docs/V2_THRESHOLD_CALIBRATION.md) AND normalized_joint_limit_margin > "
                    "min_normalized_joint_limit_margin_exclusive (Phase 5.2 class isolation), "
                    "while remaining numerically solvable by the existing DLS solver."
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
        "anchor_class_isolation_status": "locked",
        "class_eligibility_predicates": {
            "policy": (
                "Phase 5.2 [LOCKED]: anchor classes are mutually exclusive by construction. An "
                "anchor may be near a joint limit OR near-singular, never both -- the compound "
                "case is what forced anchor_near_limit_02's closed-shape trajectories to scale "
                "0.12/0.20 in Phase 5.1. These predicates govern ANCHOR SELECTION ONLY; the "
                "global difficulty definitions in difficulty_thresholds.json and the Point-IK "
                "classification priority are unchanged, and the four diagnostic flags "
                "(is_near_limit/is_near_singular/is_moderately_conditioned/is_regular) are still "
                "computed independently from those global definitions for every candidate."
            ),
            "regular": {
                "min_sigma_min_exclusive": REGULAR_MIN_SIGMA_MIN,
                "min_normalized_joint_limit_margin_exclusive": REGULAR_MIN_NORMALIZED_MARGIN,
                "rule": "sigma_min > 0.09 AND normalized_joint_limit_margin > 0.024991237796029034",
            },
            "near_limit": {
                "max_normalized_joint_limit_margin_inclusive": NEAR_JOINT_LIMIT_NORMALIZED_MARGIN_THRESHOLD,
                "min_sigma_min_exclusive": ANCHOR_NEAR_LIMIT_MIN_SIGMA_MIN,
                "require_not_near_singular": True,
                "rule": (
                    "normalized_joint_limit_margin <= 0.024991237796029034 AND sigma_min > 0.09 "
                    "AND is_near_singular == false"
                ),
                "rationale": (
                    "a near_limit anchor must be well-conditioned so the class isolates the "
                    "joint-limit factor; sigma_min > 0.09 reuses the calibrated "
                    "moderately_conditioned upper bound and implies is_near_singular == false "
                    "(which is asserted explicitly regardless)."
                ),
            },
            "near_singular": {
                "max_sigma_min_inclusive": NEAR_SINGULARITY_SIGMA_MIN_THRESHOLD,
                "min_normalized_joint_limit_margin_exclusive": ANCHOR_NEAR_SINGULAR_MIN_NORMALIZED_MARGIN,
                "require_not_near_limit": True,
                "rule": (
                    "sigma_min <= 0.03 AND normalized_joint_limit_margin > 0.024991237796029034 "
                    "AND is_near_limit == false"
                ),
                "rationale": (
                    "a near_singular anchor must be comfortably interior in joint space so the "
                    "class isolates the conditioning factor; the margin condition implies "
                    "is_near_limit == false (asserted explicitly regardless)."
                ),
            },
            "no_fallback": (
                "there is no overlap fallback. If a class has fewer eligible candidates than its "
                "quota, generation fails loudly with the candidate-availability breakdown; "
                "thresholds are never relaxed and anchors are never duplicated."
            ),
            "report_fields": [
                "eligible_count",
                "near_limit_well_conditioned_count",
                "near_limit_moderately_conditioned_count",
                "near_limit_near_singular_overlap_count",
                "near_singular_clean_count",
                "near_singular_near_limit_overlap_count",
                "regular_count",
            ],
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
        "feasibility_screening": {
            "status": "locked",
            "enabled": True,
            "rationale": (
                "Phase 5.4: anchor eligibility (class predicate) says nothing about whether the "
                "anchor's neighbourhood can actually support the locked core trajectory geometry. "
                "Three independent seed realizations produced 0, 2 and 4 core trajectories that "
                "could not reach minimum_core_accepted_scale, purely depending on which anchors "
                "were drawn. Feasibility screening makes the 120/120 outcome a property of the "
                "policy instead of luck, without weakening any locked threshold, gate, tolerance, "
                "geometry or count."
            ),
            "required_combinations": 10,
            "minimum_passing_combinations": 10,
            "partial_feasibility_accepted": False,
            "minimum_scale": MINIMUM_CORE_ACCEPTED_SCALE,
            "combinations": [
                f"{shape}_{mode}" for shape in CORE_SHAPES for mode in ORIENTATION_MODES
            ],
            "coarse_probe_canonical_waypoints": 30,
            "coarse_probe_source_waypoints": 151,
            "full_verification_canonical_waypoints": 400,
            "full_verification_note": (
                "the coarse probe is a screen, never an acceptance: a candidate that passes it "
                "must still pass full 400-waypoint canonical + full source generation before it "
                "enters the catalog, and the core trajectory validator re-verifies the final 120 "
                "independently without trusting the screen."
            ),
            "max_attempts_per_combination": 30,
            "screening_budget_per_class": {"regular": 24, "near_limit": 16, "near_singular": 16},
            "screening_order": (
                "candidates are screened in deterministic greedy farthest-point order over the "
                "eligible pool, so the screened subset is itself diverse and the order does not "
                "depend on any random retry"
            ),
            "rejection_policy": (
                "a candidate failing any one of the 10 combinations at the minimum scale is "
                "rejected outright and replaced by the next feasible candidate from the SAME "
                "eligible class pool -- candidates never move between classes, no combination is "
                "skipped, and partial feasibility is never accepted"
            ),
            "selection_order": ["class_predicate", "feasibility_screen", "diversity_selection", "split_assignment"],
            "cache_policy": {
                "enabled": True,
                "key_fields": [
                    "q (rounded to 12 decimals)",
                    "model_fingerprint",
                    "geometry_config_fingerprint",
                    "reachability_config_fingerprint",
                    "seed_algorithm_id",
                    "coarse probe resolution + minimum scale",
                ],
                "key_excludes": ["absolute paths", "timestamps", "dataset root location"],
                "scope": "in-process only; never written into the dataset root and never committed",
            },
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
        "total_canonical_poses": len(CORE_SHAPES) * len(ORIENTATION_MODES) * 12 * 400,
        "trajectory_id_pattern": "core_{shape}_{orientation_mode}_{anchor_id}",
        "dual_representation": (
            "canonical (400 waypoints) + high-resolution source; canonical is "
            "resampled/subsampled from source, never independently regenerated"
        ),
        "quaternion_order": "wxyz",
        "duration_s": 10.0,
        "time_scaling": "quintic",
        "source_waypoint_count_nominal": 2001,
        "source_waypoint_count_note": (
            "[PROVISIONAL] (spec section 5): not locked by specs/DLS_DATASET_V2_SPEC.md; chosen "
            "as ~5x the canonical count (>>400) so arc-length canonical resampling has a "
            "fine-grained source path to interpolate between; overridable via "
            "--source-waypoints for tests/smoke only, never for the locked 400-canonical-waypoint "
            "output."
        ),
        "orientation_rotation_angle_rad": 0.35,
        "orientation_rotation_axis": [0.0, 0.0, 1.0],
        "orientation_rotation_note": (
            "reused unchanged from v1's generators/generate_line_trajectory.py (and sibling "
            "shape generators) ROTATION_ANGLE_RAD/ROTATION_AXIS convention; scaled by the same "
            "scale factor as position geometry during the reachability scale search."
        ),
        "geometry": {
            "line": {"nominal_length_m": 0.12, "closed_path": False},
            "circle": {"nominal_radius_m": 0.045, "closed_path": True},
            "figure8": {"nominal_amplitude_a_m": 0.05, "nominal_amplitude_b_m": 0.03, "closed_path": True},
            "helix": {"nominal_radius_m": 0.04, "nominal_height_m": 0.08, "closed_path": False},
            "free_form": {
                "control_point_count": 5,
                "endpoint_distance_m": 0.10,
                "lateral_amplitude_m": 0.02,
                "closed_path": False,
                "spline_method": (
                    "scipy.interpolate.CubicSpline (not-a-knot), chord-length-parameterized "
                    "control points"
                ),
                "note": (
                    "control points are the anchor position (fixed start) plus deterministic "
                    "seeded lateral offsets at interior points and a seeded endpoint direction -- "
                    "different per anchor (seeded by anchor index under the core_trajectories "
                    "component seed) but fully deterministic/reproducible; never plain "
                    "per-waypoint random noise -- curvature comes from a smooth cubic spline "
                    "through a handful of control points."
                ),
            },
        },
        "closure_tolerance_position_m": 1e-6,
        "closure_tolerance_orientation_rad": 1e-6,
        "canonical_resampling_policy": {
            "method": "arc-length-uniform resampling of the high-resolution source path",
            "position_interpolation": (
                "piecewise-linear interpolation between the two bracketing source waypoints, "
                "weighted by fractional arc-length position"
            ),
            "orientation_interpolation": (
                "SO(3) geodesic SLERP between the two bracketing source quaternions "
                "(kinematics/rotation_utils.py so3_log/so3_exp) -- never linear-interpolate-then-"
                "normalize"
            ),
            "endpoint_policy": (
                "first/last canonical waypoints are exactly the first/last source waypoints "
                "(arc length 0 and total)"
            ),
            "source_parameter_mapping": (
                "each canonical waypoint stores the interpolated source tau (normalized "
                "time/path parameter) it was resampled at"
            ),
        },
        "scale_reduction_policy": {
            "shrink_factor": 0.85,
            "max_shrink_attempts": 15,
            "min_scale": 0.05,
            "note": (
                "reused unchanged from v1's generators/_trajectory_common.py SHRINK_FACTOR/"
                "MAX_SHRINK_ATTEMPTS/MIN_SCALE. Scale is only reduced AFTER every geometry "
                "alternative in geometry_alternatives has been tried at the larger scale (Phase "
                "5.1) -- shrinking is the last resort, not the first. Every attempt's outcome and "
                "rejection reason is recorded, never silently dropped."
            ),
        },
        "geometry_alternatives": {
            "policy": (
                "Phase 5.3 [LOCKED]: for each (anchor, shape, orientation_mode), EVERY alternative "
                "below is evaluated and the one achieving the LARGEST strictly-reachable scale "
                "wins. Alternatives change only the deterministic geometric basis "
                "(plane/axis/traversal/handedness/phase/template), never the nominal geometry "
                "magnitudes, and the same set is applied identically to development, validation "
                "and frozen_test. Phase 5.1 gave `line` six signed directions and `helix` six "
                "axis/travel-sign combinations, but left `circle`/`figure8` with only a plane "
                "basis (one traversal sense, one start phase, fixed lobe signs) and `free_form` "
                "with four same-rule templates. That asymmetry -- not the geometry and not the "
                "reachability criterion -- is why Phase 5.2's validation failures were exactly "
                "figure8 and free_form. Phase 5.3 completes the symmetry generically."
            ),
            "total_alternatives": {"line": 6, "circle": 12, "figure8": 24, "helix": 6, "free_form": 8},
            "line": {
                "direction_axes": ["+x", "-x", "+y", "-y", "+z", "-z"],
                "note": "anchor end-effector local axes (world frame); v1 used '+x' only.",
            },
            "circle": {
                "plane_bases": [["x", "y"], ["y", "z"], ["z", "x"]],
                "traversal_directions": ["ccw", "cw"],
                "start_phases_rad": [0.0, 3.141592653589793],
                "note": (
                    "p(s) = c + r*cos(2*pi*s + phi)*u + sigma*r*sin(2*pi*s + phi)*v with "
                    "c = p0 - r*(cos(phi)*u + sigma*sin(phi)*v), so the anchor pose is exactly the "
                    "first waypoint for every alternative. sigma = +1 (ccw) / -1 (cw) reverses "
                    "traversal; phi in {0, pi} places the circle centre on either side of the "
                    "anchor. Further phases are omitted because they duplicate the geometry "
                    "reachable by the other plane bases. 3 x 2 x 2 = 12."
                ),
            },
            "figure8": {
                "plane_bases": [["x", "y"], ["y", "z"], ["z", "x"]],
                "amplitude_signs": [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
                "axis_swaps": [False, True],
                "note": (
                    "p(s) = c + sa*A*sin(2*pi*s)*e1 + sb*B*sin(4*pi*s)*e2 with c = p0 (both sine "
                    "terms vanish at s=0, so the anchor pose is exactly the first waypoint). The "
                    "four (sa, sb) sign pairs cover both lobe handedness AND traversal reversal: "
                    "s -> 1-s maps to (-sa, -sb) exactly, so reversal is already inside this set "
                    "and is not enumerated twice. axis_swap exchanges which basis axis carries the "
                    "major amplitude A vs the minor B -- the nominal amplitudes 0.05/0.03 m are "
                    "unchanged, only their assignment. 3 x 4 x 2 = 24."
                ),
            },
            "helix": {
                "axis_bases": [["x", "y", "z"], ["y", "z", "x"], ["z", "x", "y"]],
                "height_signs": [1.0, -1.0],
                "note": "(u, v, w) basis plus travel direction along w; v1 used ('x','y','z'), +1 only.",
            },
            "free_form": {
                "templates": [
                    {"template_id": "ff0", "departure_axis": "+x", "mirror": False, "seed_offset": 0},
                    {"template_id": "ff1", "departure_axis": "-x", "mirror": False, "seed_offset": 1},
                    {"template_id": "ff2", "departure_axis": "+y", "mirror": False, "seed_offset": 2},
                    {"template_id": "ff3", "departure_axis": "-y", "mirror": False, "seed_offset": 3},
                    {"template_id": "ff4", "departure_axis": "+z", "mirror": False, "seed_offset": 4},
                    {"template_id": "ff5", "departure_axis": "-z", "mirror": False, "seed_offset": 5},
                    {"template_id": "ff6", "departure_axis": "+x", "mirror": True, "seed_offset": 6},
                    {"template_id": "ff7", "departure_axis": "+y", "mirror": True, "seed_offset": 7},
                ],
                "note": (
                    "8 locked templates (was 4). Each is a smooth cubic spline through the "
                    "configured number of control points, always starting exactly at the anchor "
                    "pose, always inside the same nominal envelope (endpoint_distance_m, "
                    "lateral_amplitude_m) and the same scale policy -- never per-waypoint noise. "
                    "The templates span six departure directions (the anchor's signed local axes, "
                    "giving horizontal / vertical / mixed workspace variants) plus two mirrored "
                    "variants whose lateral offsets are reflected. Control points come from a seed "
                    "derived from (master seed, core component tag, anchor id, seed_offset) and, "
                    "for frozen_test, the frozen seed revision. No template is specific to any "
                    "anchor."
                ),
            },
        },
        "alternative_selection_policy": {
            "status": "locked",
            "procedure": (
                "For each (anchor, shape, orientation_mode): enumerate every alternative from the "
                "locked set above; for each alternative walk the scale schedule from 1.0 downward "
                "(never below minimum_core_accepted_scale) and record the largest scale at which "
                "the alternative passes strict source/canonical reachability. Select the "
                "alternative with the LARGEST accepted scale. The search never stops at the first "
                "alternative that clears the gate -- an alternative is only abandoned early when "
                "its remaining schedule can no longer beat the best scale found so far, which "
                "cannot change the winner."
            ),
            "tie_break_order": [
                "largest accepted scale",
                "smallest strict position reconstruction error",
                "fewest refinement/restart attempts (diagnostic only)",
                "alternative_id lexical order",
            ],
            "forbidden_signals": [
                "DLS baseline evaluation success",
                "DLS solver iteration count",
                "DLS solver runtime",
                "frozen_test evaluation results",
            ],
            "rejection_reasons_recorded": True,
        },
        "minimum_scale_gate": {
            "minimum_core_accepted_scale": MINIMUM_CORE_ACCEPTED_SCALE,
            "minimum_scale_status": "locked",
            "minimum_scale_rationale": "Preserve at least half of nominal core trajectory geometry",
            "enforced": True,
            "diagnostic_bands": list(SCALE_DIAGNOSTIC_BANDS),
            "note": (
                "Phase 5.2 [LOCKED]: every accepted core trajectory must satisfy accepted_scale "
                ">= 0.50. This is a hard failure, never a warning: the generator refuses to write "
                "a core set containing a below-gate trajectory and the validator fails any "
                "on-disk trajectory below it. The gate is never relaxed to make a run pass, no "
                "trajectory is ever skipped to satisfy it, and the locked counts (12 anchors, 120 "
                "trajectories) never change. Nominal geometry is unchanged (line 0.12 m, circle "
                "r=0.045 m, figure8 0.05/0.03 m, helix r/h 0.04/0.08 m) -- the gate constrains the "
                "accepted scale factor applied to that geometry, not the geometry itself."
            ),
        },
        "reachability_policy": {
            "method": (
                "sequential warm-started DLS (generators/_trajectory_common.py-equivalent "
                "logic, unchanged kinematics/DLS math) over the 400-waypoint canonical path, "
                "starting from the anchor's own q"
            ),
            "fk_reconstruction_check": (
                "each waypoint's stored position/orientation reconstruction error is "
                "solve_dls_until_converged's own FK(q_solution)-vs-target error (an independent "
                "FK evaluation, never assumed)"
            ),
            "q_reference_storage": (
                "stored per canonical waypoint (spec section 8 permits this); "
                "reference/provenance only, never implies a DLS baseline evaluation result"
            ),
            "no_waypoint_dropped": True,
        },
        "status": "counts_locked_generation_implemented",
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
        "generation_reachability_config.json": generation_reachability_config(),
        "random_challenge_config.json": random_challenge_config(),
        "trial_config.json": trial_config(),
        "split_policy.json": split_policy_config(),
        "evaluation_defaults.json": evaluation_defaults_config(),
    }
