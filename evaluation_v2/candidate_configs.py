"""Pre-registered finite candidate set of DLS evaluation configurations, plus the fixed
convergence / reporting thresholds shared by all of them (task section 5).

Locked policy decisions encoded here:

* **Reporting thresholds** are three fixed accuracy tiers, applied *post hoc* to each solve's
  achieved ``position_error_m`` / ``orientation_error_deg`` -- they never change what the solver
  does:
    - coarse  : 6 mm / 5 deg
    - standard: 3 mm / 2 deg   (the primary success metric for candidate selection)
    - strict  : 1 mm / 1 deg
  The dataset **generation** tolerance (1e-4 m / 0.01 deg) is explicitly NOT used as a reporting
  threshold, per task section 5.

* **Convergence tolerance** (the solver's internal early-stop success test) is fixed at the
  strict reporting tier (1 mm / 1 deg) for every candidate, so "converged" == "achieved strict".
  Coarse/standard then measure graceful degradation of solves that stagnated or hit the iteration
  cap just short of strict. This is a solver mechanic, deliberately kept separate from -- but
  numerically aligned with -- the strict reporting tier, and separate from the generation
  tolerance.

* The candidates vary only in solver mechanics that are legitimate accuracy/robustness/runtime
  trade-offs: iteration budget, damping scheme, and joint-limit handling. Everything protected
  (targets, trials, seeds) is untouched.

Candidate selection consumes this set on the ``development`` split only; exactly one candidate is
then locked (see :mod:`evaluation_v2.selection`). This module invents no acceptance threshold --
it only defines the candidate mechanics and the reporting/convergence tiers.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List

# --------------------------------------------------------------------------------------------
# Fixed reporting + convergence thresholds (identical across all candidates).
# --------------------------------------------------------------------------------------------

#: Post-hoc reporting tiers: name -> (position_m, orientation_deg).
REPORTING_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "coarse": {"position_m": 0.006, "orientation_deg": 5.0},
    "standard": {"position_m": 0.003, "orientation_deg": 2.0},
    "strict": {"position_m": 0.001, "orientation_deg": 1.0},
}

#: The tier used as the primary success metric for candidate selection.
PRIMARY_REPORTING_TIER = "standard"

#: Solver early-stop convergence tolerance shared by every candidate (== strict reporting tier).
CONVERGENCE_POSITION_M = 0.001
CONVERGENCE_ORIENTATION_DEG = 1.0

#: Generation tolerance -- recorded ONLY to assert it is never used as a reporting threshold.
GENERATION_TOLERANCE_POSITION_M = 1e-4
GENERATION_TOLERANCE_ORIENTATION_DEG = 0.01

#: DLS mechanics common to every candidate (weights, singularity threshold). Damping and
#: iteration/joint-limit fields are set per candidate.
_COMMON_MECHANICS = {
    "position_weight": 1.0,
    "orientation_weight": 0.2,
    "singularity_sigma_threshold": 0.03,
    "lambda_min": 0.0001,
    "lambda_default": 0.01,
}


@dataclass(frozen=True)
class CandidateConfig:
    """One pre-registered candidate DLS configuration.

    Fields map directly onto the keys read by
    :func:`kinematics.dls_solver.solve_dls_until_converged`, except the reporting thresholds,
    which are applied afterward by the metric layer.
    """

    candidate_id: str
    description: str
    max_iterations: int
    damping_mode: str  # "adaptive" | "fixed"
    lambda_max: float  # adaptive upper damping (ignored when damping_mode == "fixed")
    lambda_default: float  # fixed damping (used when damping_mode == "fixed")
    step_scale: float
    max_joint_step_rad: float
    clip_to_operational_limits: bool
    joint_limit_avoidance: bool
    null_space_gain: float
    convergence_position_m: float = CONVERGENCE_POSITION_M
    convergence_orientation_deg: float = CONVERGENCE_ORIENTATION_DEG
    reporting_thresholds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in REPORTING_THRESHOLDS.items()}
    )

    def solver_config(self) -> dict:
        """Build the full config dict consumed by ``solve_dls_until_converged`` for this candidate.

        The solver's success test uses the candidate's convergence tolerance; the coarse/standard/
        strict reporting is computed separately by the metric layer from the achieved errors.
        """
        cfg = dict(_COMMON_MECHANICS)
        cfg.update(
            {
                "max_iterations": int(self.max_iterations),
                "position_success_threshold_m": float(self.convergence_position_m),
                "orientation_success_threshold_deg": float(self.convergence_orientation_deg),
                "damping_mode": self.damping_mode,
                "lambda_max": float(self.lambda_max),
                "lambda_default": float(self.lambda_default),
                "step_scale": float(self.step_scale),
                "max_joint_step_rad": float(self.max_joint_step_rad),
                "clip_to_operational_limits": bool(self.clip_to_operational_limits),
                "joint_limit_avoidance": bool(self.joint_limit_avoidance),
                "null_space_gain": float(self.null_space_gain),
            }
        )
        return cfg

    def to_record(self) -> dict:
        """JSON-serializable record of this candidate (for pre-registration / lock bundle)."""
        record = asdict(self)
        record["solver_config"] = self.solver_config()
        return record


def candidate_set() -> List[CandidateConfig]:
    """Return the pre-registered, ordered finite candidate set.

    Ordering is stable and used as the deterministic tie-breaker of last resort in selection.
    """
    return [
        CandidateConfig(
            candidate_id="cand_A_adaptive_baseline",
            description=(
                "Adaptive damping, 150-iteration budget, unit step, 0.1 rad step cap, clipping + "
                "null-space joint-limit avoidance. Mirrors the v1 DLS mechanics but with the tight "
                "strict-tier convergence tolerance; the low-runtime reference candidate."
            ),
            max_iterations=150,
            damping_mode="adaptive",
            lambda_max=0.2,
            lambda_default=0.01,
            step_scale=1.0,
            max_joint_step_rad=0.1,
            clip_to_operational_limits=True,
            joint_limit_avoidance=True,
            null_space_gain=0.02,
        ),
        CandidateConfig(
            candidate_id="cand_B_adaptive_deep",
            description=(
                "Candidate A with a deeper 300-iteration budget: trades runtime for a higher "
                "fraction of solves reaching the strict convergence tolerance."
            ),
            max_iterations=300,
            damping_mode="adaptive",
            lambda_max=0.2,
            lambda_default=0.01,
            step_scale=1.0,
            max_joint_step_rad=0.1,
            clip_to_operational_limits=True,
            joint_limit_avoidance=True,
            null_space_gain=0.02,
        ),
        CandidateConfig(
            candidate_id="cand_C_adaptive_lowdamp",
            description=(
                "Deep budget with a lower maximum adaptive damping (lambda_max=0.1): faster "
                "convergence in well-conditioned regions, more sensitive near singularities."
            ),
            max_iterations=300,
            damping_mode="adaptive",
            lambda_max=0.1,
            lambda_default=0.01,
            step_scale=1.0,
            max_joint_step_rad=0.1,
            clip_to_operational_limits=True,
            joint_limit_avoidance=True,
            null_space_gain=0.02,
        ),
        CandidateConfig(
            candidate_id="cand_D_pure_dls",
            description=(
                "Deep budget, adaptive damping, clipping ON but null-space joint-limit avoidance "
                "OFF: isolates the contribution of the null-space centering term."
            ),
            max_iterations=300,
            damping_mode="adaptive",
            lambda_max=0.2,
            lambda_default=0.01,
            step_scale=1.0,
            max_joint_step_rad=0.1,
            clip_to_operational_limits=True,
            joint_limit_avoidance=False,
            null_space_gain=0.0,
        ),
        CandidateConfig(
            candidate_id="cand_E_fixed_damp",
            description=(
                "Deep budget with fixed damping (lambda=0.01) instead of adaptive: a simpler, "
                "single-parameter damping baseline for comparison."
            ),
            max_iterations=300,
            damping_mode="fixed",
            lambda_max=0.2,
            lambda_default=0.01,
            step_scale=1.0,
            max_joint_step_rad=0.1,
            clip_to_operational_limits=True,
            joint_limit_avoidance=True,
            null_space_gain=0.02,
        ),
    ]


def candidate_by_id(candidate_id: str) -> CandidateConfig:
    for cand in candidate_set():
        if cand.candidate_id == candidate_id:
            return cand
    known = [c.candidate_id for c in candidate_set()]
    raise KeyError(f"unknown candidate_id '{candidate_id}'; known candidates: {known}")


def pre_registration_record() -> dict:
    """Full pre-registration document: reporting/convergence policy + every candidate."""
    return {
        "reporting_thresholds": {k: dict(v) for k, v in REPORTING_THRESHOLDS.items()},
        "primary_reporting_tier": PRIMARY_REPORTING_TIER,
        "convergence_tolerance": {
            "position_m": CONVERGENCE_POSITION_M,
            "orientation_deg": CONVERGENCE_ORIENTATION_DEG,
            "note": "solver early-stop tolerance; equals the strict reporting tier",
        },
        "generation_tolerance_not_used_for_reporting": {
            "position_m": GENERATION_TOLERANCE_POSITION_M,
            "orientation_deg": GENERATION_TOLERANCE_ORIENTATION_DEG,
        },
        "candidates": [c.to_record() for c in candidate_set()],
    }
