"""Tests for asset presence, mesh resolution, and independent loading of assets/kr810.xml."""

import hashlib
import json
import re
from pathlib import Path

import mujoco
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
URDF_PATH = ASSETS_DIR / "kr810.urdf"
MJCF_PATH = ASSETS_DIR / "kr810.xml"
MESH_DIR = ASSETS_DIR / "meshes" / "a810"
METADATA_PATH = ASSETS_DIR / "model_metadata.json"
LEGACY_DIR = REPO_ROOT / "kr810"

EXPECTED_MESHES = [
    "a810_Base.stl",
    "a810_Link2.stl",
    "a810_Link3.stl",
    "a810_Link4.stl",
    "a810_Link5.stl",
    "a810_Link6.stl",
    "a810_Link7.stl",
    "a810_ToolIO.stl",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_urdf_file_exists():
    assert URDF_PATH.is_file(), f"missing {URDF_PATH}"


def test_mjcf_file_exists():
    assert MJCF_PATH.is_file(), f"missing {MJCF_PATH}"


def test_exactly_eight_referenced_meshes_present():
    stl_files = sorted(p.name for p in MESH_DIR.glob("*.stl"))
    assert stl_files == sorted(EXPECTED_MESHES)
    assert len(stl_files) == 8


def test_all_mesh_paths_referenced_in_mjcf_resolve():
    xml_text = MJCF_PATH.read_text(encoding="utf-8")
    mesh_files = re.findall(r'<mesh[^>]*\bfile="([^"]+)"', xml_text)
    assert len(mesh_files) == 8
    for mesh_file in mesh_files:
        resolved = MESH_DIR / mesh_file
        assert resolved.is_file(), f"mesh reference did not resolve: {mesh_file}"


def test_mjcf_loads_with_mujoco():
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    assert model.nbody > 0


def test_ee_site_exists_on_end_effector_body():
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    assert site_id >= 0, "ee_site not found in compiled model"
    body_id = model.site_bodyid[site_id]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    assert body_name == "end_effector"


def test_no_absolute_source_paths_in_mjcf():
    xml_text = MJCF_PATH.read_text(encoding="utf-8")
    for forbidden in (r"C:\\", r"D:\\", "/home/", "/d/"):
        assert forbidden not in xml_text, f"found forbidden absolute path marker: {forbidden}"


def test_model_metadata_json_parses():
    data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert data["robot_name"] == "Kassow KR810"
    assert len(data["meshes"]) == 8


def test_destination_urdf_hash_matches_source_recorded_hash():
    data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert _sha256(URDF_PATH) == data["source_urdf_sha256"]
    assert _sha256(URDF_PATH) == data["destination_urdf_sha256"]


def test_legacy_kr810_folder_untouched():
    assert LEGACY_DIR.is_dir(), "legacy kr810/ folder should still exist"
    legacy_urdf = LEGACY_DIR / "urdf" / "kr810_description.urdf.xacro"
    assert legacy_urdf.is_file(), "legacy xacro should remain in place"
    legacy_mesh = LEGACY_DIR / "meshes" / "a810" / "a810_Base.stl"
    assert legacy_mesh.is_file(), "legacy mesh should remain in place"
