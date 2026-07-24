"""Golden seed-vector tests for Dataset v2's NumPy-version-independent seed derivation.

The defect this guards against: Dataset v1's ``generators/_common.py::derive_seed`` finishes with
``np.uint64 % python_int``. NumPy 1.x promotes that to ``float64`` and loses precision; NumPy 2.x
(NEP 50) keeps it exact in ``uint64``. Every derived seed therefore changed between NumPy majors,
silently producing a different dataset from the same master seed -- violating the [LOCKED] rule in
``specs/DLS_DATASET_V2_SPEC.md`` section E.

The golden vectors below are hard-coded expected values. They must hold on every NumPy version,
Python version and platform. If a change to ``dataset_v2/seeds.py`` alters them, that is a dataset
identity change and must be an explicit, recorded decision -- not an accident.

Dataset v1's ``derive_seed`` is deliberately left untouched (``CLAUDE.md``: v1 is an immutable
regression baseline), so this file also asserts that v1's function still exists and is still used
by v1 code.
"""

import hashlib

import numpy as np
import pytest

from dataset_v2.seeds import (
    SEED_ALGORITHM_ID,
    SEED_MODULUS,
    canonical_seed_payload,
    derive_seed,
    rng_from,
)

# --- Golden vectors -----------------------------------------------------------------------------
# Computed from the locked algorithm: sha256(b"dataset_v2/seed/sha256/v1|<base>|<tag>|...") -> first
# 8 bytes big-endian -> % (2**63 - 1). Pure Python integer arithmetic, no NumPy involvement.
GOLDEN_SEEDS = {
    (42,): 4903147646735054,
    (42, 10): 6676141216150053868,   # tier0 component
    (42, 20): 752829479224538130,    # point_ik component
    (42, 30): 3188872229139199479,   # anchors component
    (42, 40): 7450263331184291184,   # core_trajectories component
    (42, 50): 1083787435012209096,   # random_challenge component
    (42, 60): 675199010644122859,    # trials component
    (0,): 6460735921946825104,
    (-1, 7): 8661457445530808986,
    (2**62, 123456789): 8980338938972063083,
}


def _expected(base, *tags):
    payload = ("dataset_v2/seed/sha256/v1|" + "|".join(str(v) for v in (base, *tags))).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


# --- 1. The algorithm is exactly what the module documents --------------------------------------


def test_algorithm_id_is_locked():
    assert SEED_ALGORITHM_ID == "dataset_v2/seed/sha256/v1"
    assert SEED_MODULUS == 2**63 - 1


def test_canonical_payload_encoding_is_pinned():
    assert canonical_seed_payload(42) == b"dataset_v2/seed/sha256/v1|42"
    assert canonical_seed_payload(42, 10, 3) == b"dataset_v2/seed/sha256/v1|42|10|3"
    assert canonical_seed_payload(-1, 0) == b"dataset_v2/seed/sha256/v1|-1|0"


def test_derive_seed_matches_the_documented_algorithm():
    """Recompute the algorithm independently, from its written description."""
    for args in [(42,), (42, 10), (42, 40, 3), (0,), (-1, 7), (2**62, 123456789), (7, 0, 0, 0)]:
        assert derive_seed(*args) == _expected(*args), args


def test_golden_seed_vectors_are_stable():
    """Hard-coded expectations that must never drift across NumPy/Python/platform.

    These are the actual seeds Dataset v2 generation depends on (the component tags 10/20/30/40/
    50/60 are Tier 0, Point-IK, anchors, core trajectories, random challenge and trials). A change
    here means the dataset's identity changed.
    """
    assert GOLDEN_SEEDS, "golden vector table must not be empty"
    for args, expected_seed in GOLDEN_SEEDS.items():
        value = derive_seed(*args)
        assert value == expected_seed, f"golden seed drift for {args}: {value} != {expected_seed}"
        assert value == _expected(*args)
        assert isinstance(value, int) and not isinstance(value, np.integer)
        assert 0 <= value < SEED_MODULUS


# --- 2. No NumPy type participates in the arithmetic --------------------------------------------


def test_result_is_a_pure_python_int():
    value = derive_seed(42, 40)
    assert type(value) is int


def test_numpy_integer_inputs_give_identical_results():
    """A NumPy integer tag must derive the same seed as the equivalent Python int -- otherwise the
    result would depend on incidental call-site dtypes."""
    assert derive_seed(np.int64(42), np.int32(40)) == derive_seed(42, 40)
    assert derive_seed(42, np.uint64(30)) == derive_seed(42, 30)


def test_large_values_are_not_truncated_by_float_promotion():
    """The exact failure mode of the v1 derivation: a value above 2**53 losing precision once it is
    promoted through float64."""
    big = 2**62 + 12345
    assert derive_seed(big) == _expected(big)
    # a 1-unit change in a >2**53 input must change the seed (float64 could not represent it)
    assert derive_seed(big) != derive_seed(big + 1)


def test_non_integer_inputs_are_rejected():
    for bad in (1.5, "42", None, [1]):
        with pytest.raises(TypeError):
            derive_seed(bad)
    with pytest.raises(TypeError):
        derive_seed(42, 1.0)
    # bool would silently collide with 0/1
    with pytest.raises(TypeError):
        derive_seed(True)


# --- 3. Derivation properties -------------------------------------------------------------------


def test_distinct_tag_paths_give_distinct_seeds():
    seeds = {
        derive_seed(42, 10),
        derive_seed(42, 20),
        derive_seed(42, 30),
        derive_seed(42, 40),
        derive_seed(42, 50),
        derive_seed(42, 60),
    }
    assert len(seeds) == 6


def test_tag_order_and_arity_matter():
    assert derive_seed(42, 1, 2) != derive_seed(42, 2, 1)
    assert derive_seed(42, 1) != derive_seed(42, 1, 0)
    # the separator must prevent concatenation collisions: (1, 23) vs (12, 3)
    assert derive_seed(42, 1, 23) != derive_seed(42, 12, 3)


def test_derivation_is_repeatable_within_a_process():
    assert derive_seed(42, 40, 7) == derive_seed(42, 40, 7)


def test_rng_from_is_seeded_by_derive_seed():
    expected = np.random.default_rng(derive_seed(42, 40)).random(5)
    assert np.array_equal(rng_from(42, 40).random(5), expected)


# --- 4. Every Dataset v2 component uses the new derivation ---------------------------------------


V2_SEEDED_MODULES = [
    "dataset_v2.tier0_generation",
    "dataset_v2.threshold_calibration",
    "dataset_v2.point_ik_generation",
    "dataset_v2.anchor_generation",
    "dataset_v2.core_trajectory_generation",
    "dataset_v2.generation_reachability",
]


def test_all_v2_generators_bind_the_v2_derive_seed():
    """Covers Tier 0, calibration, Point-IK, anchors + split assignment, core trajectories,
    geometry alternatives and free-form templates -- they all seed through these modules."""
    import importlib

    import dataset_v2.seeds as v2_seeds

    for module_name in V2_SEEDED_MODULES:
        module = importlib.import_module(module_name)
        assert hasattr(module, "derive_seed"), f"{module_name} does not bind derive_seed"
        assert module.derive_seed is v2_seeds.derive_seed, (
            f"{module_name} still binds a derive_seed that is not dataset_v2.seeds.derive_seed"
        )


def test_no_dataset_v2_module_imports_the_legacy_derive_seed():
    import ast
    import pathlib

    package = pathlib.Path(__file__).resolve().parent.parent / "dataset_v2"
    offenders = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "generators._common":
                if any(alias.name == "derive_seed" for alias in node.names):
                    offenders.append(path.name)
    assert not offenders, f"Dataset v2 modules still importing the legacy derive_seed: {offenders}"


# --- 5. Dataset v1 is untouched -------------------------------------------------------------------


def test_dataset_v1_derive_seed_still_exists_and_is_separate():
    """v1's derivation must remain in place, unmodified, and must NOT be the v2 one."""
    from generators import _common as v1_common
    import dataset_v2.seeds as v2_seeds

    assert hasattr(v1_common, "derive_seed")
    assert v1_common.derive_seed is not v2_seeds.derive_seed
    # v1 still uses its own function internally
    assert v1_common.rng_from(1, 2) is not None


def test_v1_and_v2_derivations_are_independent():
    """They are allowed (and expected) to disagree -- v2 fixed the promotion bug that v1 keeps for
    backward compatibility with its already-checksummed data."""
    from generators._common import derive_seed as v1_derive_seed

    v1_value = v1_derive_seed(42, 40)
    v2_value = derive_seed(42, 40)
    assert isinstance(v1_value, int) and isinstance(v2_value, int)
    assert v1_value != v2_value
