"""Loads and validates the compiled KR810 MuJoCo model referenced by configs/robot_config.json.

The returned ModelContext holds the compiled mujoco.MjModel plus resolved joint/site
indices, but does not hand out a single shared mutable MjData for use across solvers.
Callers must request a fresh MjData via ``new_data()`` (or an independent copy via
``copy_data()``) whenever they need one, so that concurrent solvers never mutate the
same simulation state.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import mujoco
import numpy as np

from utils.config_loader import load_json_config
from utils.exceptions import InvalidJointVectorError, ModelConfigurationError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "assets" / "kr810.xml"
DEFAULT_ROBOT_CONFIG_PATH = REPO_ROOT / "configs" / "robot_config.json"


@dataclass(frozen=True)
class ModelContext:
    """Immutable bundle of a compiled MjModel plus resolved joint/site metadata."""

    model: mujoco.MjModel
    model_path: Path
    joint_names: tuple
    joint_ids: tuple
    qpos_addresses: tuple
    dof_addresses: tuple
    ee_site_id: int
    ee_site_name: str
    operational_lower_rad: np.ndarray
    operational_upper_rad: np.ndarray
    velocity_limits_rad_s: np.ndarray
    nq: int
    nv: int

    def new_data(self) -> mujoco.MjData:
        """Create a brand-new, independent MjData instance for this model."""
        return mujoco.MjData(self.model)

    def copy_data(self, data: mujoco.MjData) -> mujoco.MjData:
        """Create an independent deep copy of an existing MjData (does not alias state)."""
        data_copy = mujoco.MjData(self.model)
        mujoco.mj_copyData(data_copy, self.model, data)
        return data_copy

    def validate_q(self, q: np.ndarray) -> np.ndarray:
        """Validate a joint vector's shape and finiteness. Returns q as a float64 ndarray.

        Does not clip or otherwise silently modify out-of-range values.
        """
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape != (self.nq,):
            raise InvalidJointVectorError(
                f"expected q shape ({self.nq},), got {q_arr.shape}"
            )
        if not np.all(np.isfinite(q_arr)):
            raise InvalidJointVectorError("q contains non-finite values (NaN or Inf)")
        return q_arr

    def set_qpos(self, data: mujoco.MjData, q: np.ndarray) -> None:
        """Write a validated joint vector into the correct qpos addresses of ``data``."""
        q_arr = self.validate_q(q)
        for addr, value in zip(self.qpos_addresses, q_arr):
            data.qpos[addr] = value

    def forward(self, data: mujoco.MjData) -> None:
        """Run mujoco.mj_forward on the given data using this context's model."""
        mujoco.mj_forward(self.model, data)


def _resolve_joint_metadata(model: mujoco.MjModel, joint_names: list) -> tuple:
    joint_ids = []
    qpos_addresses = []
    dof_addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ModelConfigurationError(f"joint '{name}' not found in compiled model")
        joint_type = model.jnt_type[joint_id]
        if joint_type != mujoco.mjtJoint.mjJNT_HINGE and joint_type != mujoco.mjtJoint.mjJNT_SLIDE:
            raise ModelConfigurationError(
                f"joint '{name}' has unexpected joint type {joint_type} (expected hinge/slide)"
            )
        joint_ids.append(joint_id)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    return tuple(joint_ids), tuple(qpos_addresses), tuple(dof_addresses)


def load_model_context(
    model_path: Optional[Union[str, Path]] = None,
    robot_config_path: Optional[Union[str, Path]] = None,
) -> ModelContext:
    """Load the KR810 MuJoCo model and cross-validate it against configs/robot_config.json.

    Args:
        model_path: Path to the MJCF model. Defaults to assets/kr810.xml under the repo root.
        robot_config_path: Path to the robot config JSON. Defaults to configs/robot_config.json.

    Returns:
        A populated, immutable ModelContext.

    Raises:
        ModelConfigurationError: on any mismatch between the compiled model and the
            expected robot configuration (joint count, joint names/order, ee_site, etc).
    """
    resolved_model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    resolved_config_path = (
        Path(robot_config_path) if robot_config_path is not None else DEFAULT_ROBOT_CONFIG_PATH
    )

    if not resolved_model_path.is_file():
        raise ModelConfigurationError(f"model file not found: {resolved_model_path}")

    config = load_json_config(resolved_config_path)

    expected_joint_names = list(config["joint_order"])
    expected_nq = int(config["nq"])
    expected_nv = int(config["nv"])
    ee_site_name = str(config["end_effector_site"])

    model = mujoco.MjModel.from_xml_path(str(resolved_model_path))

    if model.nq != expected_nq:
        raise ModelConfigurationError(
            f"model nq={model.nq} does not match config nq={expected_nq}"
        )
    if model.nv != expected_nv:
        raise ModelConfigurationError(
            f"model nv={model.nv} does not match config nv={expected_nv}"
        )

    joint_ids, qpos_addresses, dof_addresses = _resolve_joint_metadata(model, expected_joint_names)

    if len(set(qpos_addresses)) != len(qpos_addresses) or len(set(dof_addresses)) != len(dof_addresses):
        raise ModelConfigurationError("resolved qpos/dof addresses are not unique")

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    if ee_site_id < 0:
        raise ModelConfigurationError(f"end-effector site '{ee_site_name}' not found in compiled model")

    operational_lower = np.asarray(config["operational_lower_rad"], dtype=np.float64)
    operational_upper = np.asarray(config["operational_upper_rad"], dtype=np.float64)
    velocity_limits = np.asarray(config["velocity_limits_rad_s"], dtype=np.float64)

    if operational_lower.shape != (expected_nq,) or operational_upper.shape != (expected_nq,):
        raise ModelConfigurationError("operational limit arrays do not match nq in shape")
    if velocity_limits.shape != (expected_nq,):
        raise ModelConfigurationError("velocity limit array does not match nq in shape")
    if not np.all(operational_upper >= operational_lower):
        raise ModelConfigurationError("operational_upper_rad must be >= operational_lower_rad elementwise")

    return ModelContext(
        model=model,
        model_path=resolved_model_path,
        joint_names=tuple(expected_joint_names),
        joint_ids=joint_ids,
        qpos_addresses=qpos_addresses,
        dof_addresses=dof_addresses,
        ee_site_id=int(ee_site_id),
        ee_site_name=ee_site_name,
        operational_lower_rad=operational_lower,
        operational_upper_rad=operational_upper,
        velocity_limits_rad_s=velocity_limits,
        nq=expected_nq,
        nv=expected_nv,
    )
