"""Manipulability index computation (e.g. Yoshikawa measure) from the end-effector Jacobian.

Caveats:
- The full 6x7 geometric Jacobian mixes position (meter) and orientation (radian)
  units. The Yoshikawa measure computed from the full Jacobian (``yoshikawa_manipulability``)
  is therefore scale-dependent and must not be compared directly across robots or used
  as an absolute scientific metric unless the Jacobian rows have been normalized/nondimensionalized
  first (see ``normalized_jacobian``).
- ``positional_manipulability`` uses only the position sub-block J_v (3x7, meters/radian)
  and is a meaningful, unit-consistent measure on its own.
"""

import numpy as np

from utils.exceptions import NumericalKinematicsError

_DEFAULT_SCALE_FLOOR = 1e-12


def yoshikawa_manipulability(J: np.ndarray) -> float:
    """Yoshikawa manipulability measure w = sqrt(det(J @ J.T)) of the full geometric Jacobian.

    Warning: mixes position (m) and orientation (rad) units; not directly comparable
    across robots/scalings without normalization.
    """
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2:
        raise NumericalKinematicsError(f"expected a 2D Jacobian, got shape {J.shape}")
    gram = J @ J.T
    det = np.linalg.det(gram)
    return float(np.sqrt(max(det, 0.0)))


def positional_manipulability(J: np.ndarray) -> float:
    """Yoshikawa manipulability measure of the position-only sub-block J_v (first 3 rows).

    Unit-consistent (meters/radian only); safe to use as a standalone positional metric.
    """
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2 or J.shape[0] < 3:
        raise NumericalKinematicsError(f"expected a Jacobian with >= 3 rows, got shape {J.shape}")
    J_v = J[:3, :]
    gram = J_v @ J_v.T
    det = np.linalg.det(gram)
    return float(np.sqrt(max(det, 0.0)))


def normalized_jacobian(J: np.ndarray, length_scale: float) -> np.ndarray:
    """Nondimensionalize the orientation rows of a 6-row Jacobian by ``length_scale`` (meters).

    Divides the last 3 rows (orientation, rad) by ``length_scale`` so all rows share
    comparable units (1/length-scale-normalized), enabling a scale-consistent Yoshikawa
    measure. Position rows (first 3) are left unchanged.
    """
    J = np.asarray(J, dtype=np.float64)
    if J.shape[0] != 6:
        raise NumericalKinematicsError(f"expected a 6-row Jacobian, got shape {J.shape}")
    if length_scale <= _DEFAULT_SCALE_FLOOR:
        raise NumericalKinematicsError("length_scale must be positive and well above zero")
    J_normalized = J.copy()
    J_normalized[3:, :] = J_normalized[3:, :] * length_scale
    return J_normalized
