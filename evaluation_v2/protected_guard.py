"""Runtime guard that keeps protected reference fields out of the DLS evaluator.

Dataset v2 trajectory / point-IK NPZs carry BOTH public geometry (target poses, ``q_initial``,
timing) AND protected reconstruction evidence -- notably the reference joint solutions
``q_reference`` (trajectories) and ``q_target_reference`` (point-IK). A DLS evaluator that saw any
of those would be handed (part of) the very IK answer it is supposed to search for, invalidating
the benchmark.

Two lines of defence, both active in Phase 8A:

1. **Physical isolation** -- :mod:`evaluation_v2.public_export` writes a public evaluation root
   that never contains any protected array on disk, so the evaluator's inputs cannot carry them.
2. **Runtime guard (this module)** -- every mapping/record that crosses into the evaluator is
   passed through :func:`assert_no_protected_fields`, which raises immediately if a protected key
   is present. This catches a caller that (accidentally or maliciously) points the evaluator at a
   full dataset-root NPZ instead of the public export.

The guard is intentionally *name-based and conservative*: any field whose name matches a protected
key (exactly, or by a ``q_reference`` / ``reconstruction`` / ``q_target_reference`` /
``waypoint_reachable`` substring) is rejected. Public fields never use those names.
"""

from typing import Iterable, Mapping

from utils.exceptions import ModelConfigurationError

#: Exact protected array/column names that must never reach the evaluator.
PROTECTED_FIELD_NAMES = frozenset(
    {
        "q_reference",
        "q_reference_start",
        "q_source_reference",
        "q_target_reference",
        "position_reconstruction_error_m",
        "orientation_reconstruction_error_rad",
        "waypoint_reachable",
    }
)

#: Substrings that also mark a field as protected, catching renamed/prefixed variants.
_PROTECTED_SUBSTRINGS = (
    "q_reference",
    "q_source_reference",
    "q_target_reference",
    "reconstruction_error",
    "waypoint_reachable",
)


def is_protected_field(name: str) -> bool:
    """Return True if ``name`` denotes a protected reference field."""
    if name in PROTECTED_FIELD_NAMES:
        return True
    lowered = str(name).lower()
    return any(token in lowered for token in _PROTECTED_SUBSTRINGS)


def find_protected_fields(names: Iterable[str]) -> list:
    """Return the sorted subset of ``names`` that are protected."""
    return sorted({name for name in names if is_protected_field(name)})


def assert_no_protected_fields(mapping: Mapping, context: str) -> None:
    """Raise ``ModelConfigurationError`` if ``mapping`` exposes any protected reference field.

    Args:
        mapping: Any name->value mapping about to be handed to the evaluator (an NPZ dict, a
            catalog row, a config, ...).
        context: Human-readable description of what is being checked, quoted back in the error.
    """
    leaked = find_protected_fields(mapping.keys())
    if leaked:
        raise ModelConfigurationError(
            f"protected reference field(s) reached the DLS evaluator via {context}: {leaked}. "
            "The evaluator must be given only the PUBLIC evaluation export (target poses, "
            "q_initial, public metadata); q_reference / q_target_reference and other "
            "reconstruction evidence are never valid evaluator inputs."
        )
