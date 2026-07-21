"""Tests for nq/nv/nu, joint ordering, and end-effector site behavior on the compiled model."""

import json
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MJCF_PATH = REPO_ROOT / "assets" / "kr810.xml"
CONFIGS_DIR = REPO_ROOT / "configs"

EXPECTED_JOINT_ORDER = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]


def _load_model():
    return mujoco.MjModel.from_xml_path(str(MJCF_PATH))


def test_nq_equals_seven():
    model = _load_model()
    assert model.nq == 7


def test_nv_equals_seven():
    model = _load_model()
    assert model.nv == 7


def test_nu_zero_is_accepted():
    model = _load_model()
    assert model.nu == 0


def test_movable_joint_order_matches_joint_1_to_7():
    model = _load_model()
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    assert names == EXPECTED_JOINT_ORDER


def test_mj_forward_runs_with_zero_qpos():
    model = _load_model()
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.qpos).all()


def test_ee_site_pose_finite_at_zero_qpos():
    model = _load_model()
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    assert site_id >= 0
    assert np.isfinite(data.site_xpos[site_id]).all()
    assert np.isfinite(data.site_xmat[site_id]).all()


def test_robot_config_json_parses_and_matches_model():
    data = json.loads((CONFIGS_DIR / "robot_config.json").read_text(encoding="utf-8"))
    assert data["nq"] == 7
    assert data["nv"] == 7
    assert data["joint_order"] == EXPECTED_JOINT_ORDER
    assert data["asset_status"] == "integrated"
    assert data["end_effector_site"] == "ee_site"


def test_frame_config_json_parses_and_is_resolved():
    data = json.loads((CONFIGS_DIR / "frame_config.json").read_text(encoding="utf-8"))
    assert data["status"] == "resolved"
    assert data["end_effector_site"] == "ee_site"


def test_dataset_manifest_json_parses_and_reports_integrated_assets():
    data = json.loads((REPO_ROOT / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    assert data["assets"]["status"] == "integrated"
    assert data["assets"]["n_joints"] == 7
