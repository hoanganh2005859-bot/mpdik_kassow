"""Post-hoc reporting-tier success classification shared by Tier 1 and Tier 2.

A solve's achieved ``position_error_m`` / ``orientation_error_deg`` is compared to each reporting
tier (coarse / standard / strict). This never influences the solver -- it is measurement only.
"""

from typing import Dict

from evaluation_v2.candidate_configs import REPORTING_THRESHOLDS

REPORTING_TIERS = ("coarse", "standard", "strict")


def reporting_success(
    position_error_m: float,
    orientation_error_deg: float,
    thresholds: Dict[str, Dict[str, float]] = None,
) -> Dict[str, bool]:
    """Return ``{tier: bool}`` success at each reporting tier for one achieved error pair."""
    thresholds = thresholds or REPORTING_THRESHOLDS
    out = {}
    for tier in REPORTING_TIERS:
        t = thresholds[tier]
        out[tier] = bool(
            position_error_m <= t["position_m"] and orientation_error_deg <= t["orientation_deg"]
        )
    return out


def success_columns(position_error_m: float, orientation_error_deg: float, thresholds=None) -> dict:
    """Flatten reporting-tier success into ``success_<tier>`` boolean columns."""
    s = reporting_success(position_error_m, orientation_error_deg, thresholds)
    return {f"success_{tier}": s[tier] for tier in REPORTING_TIERS}
