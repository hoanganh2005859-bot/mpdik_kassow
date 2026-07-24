"""Dataset v2's own deterministic seed derivation -- NumPy-version-independent.

Why this module exists
----------------------
Dataset v1's ``generators/_common.py::derive_seed`` ends with::

    int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))

The final ``np.uint64 % python_int`` is **not stable across NumPy majors**:

* NumPy 1.x promotes the operands to ``float64`` (NEP 50 predecessor rules), silently losing
  precision -- ``772962124644224780 -> 772962124644224768``.
* NumPy 2.x (NEP 50) keeps the value in ``uint64`` and returns it exactly.

Every derived seed therefore differs between NumPy majors, which cascades into different candidate
pools, anchors, splits and trajectories. That violates the [LOCKED] rule in
``specs/DLS_DATASET_V2_SPEC.md`` section E: "Regeneration from the same master seed must reproduce
byte-identical NPZ content."

This module replaces that derivation for Dataset v2 only. Dataset v1's ``derive_seed`` is left
exactly as it is (``CLAUDE.md``: Dataset v1 is an immutable regression baseline), so v1's
generators and their already-checksummed outputs are unaffected.

Algorithm
---------
Pure Python integer arithmetic over a SHA-256 digest of a canonical byte encoding:

1. every input is normalised to a Python ``int``;
2. the inputs are joined into a canonical, unambiguous byte string
   ``b"<domain>|<base>|<tag0>|<tag1>|..."``;
3. SHA-256 of those bytes;
4. the leading 8 bytes are read big-endian into a Python ``int``;
5. reduced modulo ``2**63 - 1`` using Python's arbitrary-precision integers.

No NumPy type participates in the arithmetic, so the result depends only on the input integers --
identical across NumPy versions, Python versions, platforms and byte orders.
"""

import hashlib
from typing import Union

import numpy as np

#: Bumping this string changes every derived seed; treat it as part of the dataset's identity.
SEED_ALGORITHM_ID = "dataset_v2/seed/sha256/v1"

_DOMAIN_SEPARATOR = SEED_ALGORITHM_ID.encode("ascii")
_FIELD_SEPARATOR = b"|"

#: Seeds are reduced into ``[0, 2**63 - 2]`` -- the same range Dataset v1's derivation targeted, so
#: downstream consumers see the same magnitude of value.
SEED_MODULUS = 2**63 - 1

IntLike = Union[int, "np.integer"]


def _as_int(value: IntLike, what: str) -> int:
    """Normalise ``value`` to a Python ``int``, rejecting anything lossy or ambiguous.

    ``bool`` is rejected because ``True``/``1`` would silently collide; floats are rejected because
    they make the canonical encoding platform-dependent.
    """
    if isinstance(value, bool):
        raise TypeError(f"{what} must be an integer, not a bool (got {value!r})")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(f"{what} must be an integer, got {type(value).__name__} ({value!r})")


def canonical_seed_payload(base_seed: IntLike, *tags: IntLike) -> bytes:
    """The exact bytes hashed by :func:`derive_seed` (exposed so tests can pin the encoding)."""
    parts = [str(_as_int(base_seed, "base_seed"))]
    parts.extend(str(_as_int(tag, f"tag[{i}]")) for i, tag in enumerate(tags))
    return _DOMAIN_SEPARATOR + _FIELD_SEPARATOR + _FIELD_SEPARATOR.join(p.encode("ascii") for p in parts)


def derive_seed(base_seed: IntLike, *tags: IntLike) -> int:
    """Deterministically derive a child seed from a base seed and integer tags.

    Drop-in replacement for ``generators/_common.py::derive_seed`` for Dataset v2 use, but computed
    entirely with Python integers over a SHA-256 digest so the result is independent of the
    installed NumPy version.
    """
    digest = hashlib.sha256(canonical_seed_payload(base_seed, *tags)).digest()
    return int.from_bytes(digest[:8], "big") % SEED_MODULUS


def rng_from(base_seed: IntLike, *tags: IntLike) -> np.random.Generator:
    """A ``np.random.Generator`` seeded from :func:`derive_seed`.

    ``np.random.default_rng(<python int>)`` seeds PCG64 through ``SeedSequence``, whose stream is
    guaranteed stable across NumPy versions -- the instability fixed here was only in the *seed
    arithmetic*, never in the bit generator itself.
    """
    return np.random.default_rng(derive_seed(base_seed, *tags))
