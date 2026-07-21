"""Integration tests for scripts/package_kaggle_release.py (the Stage 8 Kaggle release packager).

Packages once per test session into tmp_path (never into the repository's own dist/), then
checks: the ZIP has DATASET_MANIFEST.json at its root with no outer wrapper folder, contains
every required file and excludes every forbidden one, has no absolute/traversal member paths,
every JSON member parses, the NPZ/model files it contains actually load after extraction, the
release notebook is a clean nbformat v4 copy, and RELEASE_MANIFEST.json's recorded SHA256 values
match the real files. Also exercises utils.safe_zip's extraction safety checks directly (valid
ZIP, path traversal, absolute path, oversized/too-many-member archives).
"""

import json
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import nbformat as nbf
import pytest

from scripts.package_kaggle_release import (
    _EXCLUDE_DIR_NAMES,
    _EXCLUDE_FILE_SUFFIXES,
    _REQUIRED_STAGED_MEMBERS,
    main as package_main,
)
from utils.dataset_locator import REPO_ROOT
from utils.file_checksum import sha256_file
from utils.safe_zip import UnsafeZipError, safe_extract_zip


@pytest.fixture(scope="module")
def packaged_release(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("kaggle_release_dist")
    exit_code = package_main(["--output-dir", str(output_dir), "--overwrite"])
    assert exit_code == 0
    return output_dir


@pytest.fixture(scope="module")
def dataset_zip_path(packaged_release):
    zips = list(packaged_release.glob("KR810_Tier0_Tier4_Kaggle_Dataset_v*.zip"))
    assert len(zips) == 1, f"expected exactly one dataset ZIP, found {zips}"
    return zips[0]


@pytest.fixture(scope="module")
def notebook_release_path(packaged_release):
    notebooks = list(packaged_release.glob("KR810_Tier0_Tier4_Kaggle_Template_v*.ipynb"))
    assert len(notebooks) == 1, f"expected exactly one release notebook, found {notebooks}"
    return notebooks[0]


@pytest.fixture(scope="module")
def extracted_dataset(tmp_path_factory, dataset_zip_path):
    destination = tmp_path_factory.mktemp("kaggle_release_extracted")
    safe_extract_zip(dataset_zip_path, destination)
    return destination


def test_packaging_script_exists():
    assert (REPO_ROOT / "scripts" / "package_kaggle_release.py").is_file()


def test_package_creates_nonempty_zip_and_manifest(packaged_release, dataset_zip_path):
    assert dataset_zip_path.is_file()
    assert dataset_zip_path.stat().st_size > 0
    assert (packaged_release / "RELEASE_MANIFEST.json").is_file()


def test_zip_manifest_at_root_with_no_outer_folder(dataset_zip_path):
    with zipfile.ZipFile(dataset_zip_path) as zf:
        names = zf.namelist()
    assert "DATASET_MANIFEST.json" in names
    top_level_entries = {name.split("/", 1)[0] for name in names}
    assert "assets" in top_level_entries
    assert "pipelines" in top_level_entries
    # If an outer wrapper folder existed, DATASET_MANIFEST.json would be nested (e.g.
    # "Kassow_KR810_Trajectory_Tier0_Tier4/DATASET_MANIFEST.json") rather than a root entry.
    assert all("/" not in name or not name.startswith("Kassow") for name in names)


def test_zip_contains_required_members(dataset_zip_path):
    with zipfile.ZipFile(dataset_zip_path) as zf:
        names = set(zf.namelist())
    for required in _REQUIRED_STAGED_MEMBERS + ["configs", "schemas", "kinematics", "algorithms", "evaluation"]:
        assert any(name == required or name.startswith(required + "/") for name in names), (
            f"missing required member/prefix: {required}"
        )


def test_zip_excludes_forbidden_paths(dataset_zip_path):
    with zipfile.ZipFile(dataset_zip_path) as zf:
        names = zf.namelist()
    forbidden_prefixes = tuple(f"{name}/" for name in _EXCLUDE_DIR_NAMES)
    for name in names:
        assert not name.startswith(forbidden_prefixes), f"forbidden path leaked into ZIP: {name}"
        assert "/__pycache__/" not in name and "/.git/" not in name and "/.pytest_cache/" not in name
        assert not any(name.endswith(suffix) for suffix in _EXCLUDE_FILE_SUFFIXES), f"forbidden suffix: {name}"
    assert not any(name.startswith("notebooks/") for name in names)


def test_zip_has_no_absolute_or_traversal_members(dataset_zip_path):
    with zipfile.ZipFile(dataset_zip_path) as zf:
        for info in zf.infolist():
            assert not info.filename.startswith("/"), f"absolute member path: {info.filename}"
            assert ".." not in Path(info.filename).parts, f"traversal member path: {info.filename}"


def test_all_json_members_parse(dataset_zip_path):
    with zipfile.ZipFile(dataset_zip_path) as zf:
        json_names = [n for n in zf.namelist() if n.endswith(".json")]
        assert len(json_names) > 0
        for name in json_names:
            json.loads(zf.read(name).decode("utf-8"))


def test_extracted_npz_files_load_without_pickle(extracted_dataset):
    point_ik_path = extracted_dataset / "benchmarks" / "point_ik" / "point_ik_v1.npz"
    with np.load(point_ik_path, allow_pickle=False) as data:
        assert "q_target" in data
        assert data["q_target"].shape[0] == 1200

    manifest_rows = (extracted_dataset / "trajectories" / "trajectory_manifest.csv").read_text(encoding="utf-8")
    trajectory_npz_paths = [
        line.split(",")[2] for line in manifest_rows.splitlines()[1:] if line.strip()
    ]
    assert len(trajectory_npz_paths) == 8
    for relative_path in trajectory_npz_paths:
        with np.load(extracted_dataset / relative_path, allow_pickle=False) as data:
            assert len(data.files) > 0


def test_extracted_model_loads_with_mujoco(extracted_dataset):
    model = mujoco.MjModel.from_xml_path(str(extracted_dataset / "assets" / "kr810.xml"))
    assert model.nq == 7
    assert model.nv == 7


def test_release_notebook_is_clean_nbformat_v4(notebook_release_path):
    nb = nbf.read(notebook_release_path, as_version=4)
    nbf.validate(nb)
    assert nb.nbformat == 4
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
            compile(cell.source, str(notebook_release_path), "exec")


def test_release_manifest_sha256_matches_actual_files(packaged_release, dataset_zip_path, notebook_release_path):
    manifest = json.loads((packaged_release / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["dataset_zip_sha256"] == sha256_file(dataset_zip_path)
    assert manifest["dataset_zip_size_bytes"] == dataset_zip_path.stat().st_size
    assert manifest["notebook_sha256"] == sha256_file(notebook_release_path)
    assert manifest["notebook_size_bytes"] == notebook_release_path.stat().st_size
    assert manifest["packaging_validation_status"] == "passed"


def test_safe_extract_accepts_valid_zip(tmp_path):
    zip_path = tmp_path / "valid.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("DATASET_MANIFEST.json", "{}")
        zf.writestr("assets/kr810.xml", "<mujoco/>")

    destination = tmp_path / "extracted"
    safe_extract_zip(zip_path, destination)

    assert (destination / "DATASET_MANIFEST.json").read_text(encoding="utf-8") == "{}"
    assert (destination / "assets" / "kr810.xml").is_file()


def test_safe_extract_rejects_path_traversal(tmp_path):
    zip_path = tmp_path / "traversal.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../evil.txt", "malicious")

    with pytest.raises(UnsafeZipError):
        safe_extract_zip(zip_path, tmp_path / "extracted")
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_rejects_absolute_path(tmp_path):
    zip_path = tmp_path / "absolute.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("/etc/passwd", "malicious")

    with pytest.raises(UnsafeZipError):
        safe_extract_zip(zip_path, tmp_path / "extracted")


def test_safe_extract_rejects_archives_exceeding_configured_limits(tmp_path):
    too_many_zip = tmp_path / "too_many.zip"
    with zipfile.ZipFile(too_many_zip, "w") as zf:
        for i in range(5):
            zf.writestr(f"file_{i}.txt", "x")
    with pytest.raises(UnsafeZipError):
        safe_extract_zip(too_many_zip, tmp_path / "extracted_many", max_members=3)

    too_large_zip = tmp_path / "too_large.zip"
    with zipfile.ZipFile(too_large_zip, "w") as zf:
        zf.writestr("big.bin", b"x" * 1000)
    with pytest.raises(UnsafeZipError):
        safe_extract_zip(too_large_zip, tmp_path / "extracted_large", max_uncompressed_bytes=10)
