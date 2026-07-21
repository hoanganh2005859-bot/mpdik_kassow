"""Packages the Kaggle Dataset ZIP, a release copy of the notebook, and a release manifest.

Usage:
    python -m scripts.package_kaggle_release --output-dir dist --overwrite

Builds, in order: a staging copy of an explicit allowlist of repository root entries (never a
raw "everything in git" copy), the Kaggle Dataset ZIP from that staging copy, a sanitized release
copy of notebooks/KR810_Tier0_Tier4_Kaggle_Template.ipynb, and dist/RELEASE_MANIFEST.json /
dist/SHA256SUMS.txt describing both artifacts. Does not modify any source file, retrain anything,
or regenerate the benchmark/trajectory data; it only repackages what Stage 1-7 already produced.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import nbformat as nbf

from utils.dataset_locator import REPO_ROOT
from utils.file_checksum import sha256_file
from utils.safe_zip import is_unsafe_member_name

NOTEBOOK_SOURCE = REPO_ROOT / "notebooks" / "KR810_Tier0_Tier4_Kaggle_Template.ipynb"
UPLOAD_GUIDE_SOURCE = Path(__file__).resolve().parent / "KAGGLE_UPLOAD_AND_RUN_GUIDE.md"
STAGING_ROOT = REPO_ROOT / "temporary_results" / "kaggle_dataset_staging"

# Explicit allowlist: only these root entries are ever copied into the Kaggle Dataset staging
# tree. notebooks/, kr810/ (legacy), .git/, .venv/, results/, dist/, etc. are never in this list,
# so they can never end up in the release ZIP regardless of what else exists in the repo.
ALLOWLIST_ROOT_FILES = ["README.md", "DATASET_MANIFEST.json", "VERSION", "requirements.txt", "pyproject.toml"]
ALLOWLIST_ROOT_DIRS = [
    "assets", "configs", "kinematics", "benchmarks", "trajectories", "generators",
    "algorithms", "evaluation", "pipelines", "utils", "tests", "schemas",
]

_EXCLUDE_DIR_NAMES = {
    ".git", ".github", ".venv", "venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
    "temporary_results", "results", "smoke_results", "dist", "build", ".vscode", ".idea", "kr810",
}
_EXCLUDE_FILE_SUFFIXES = {".pyc", ".pyo", ".log", ".zip"}

_REQUIRED_STAGED_MEMBERS = [
    "DATASET_MANIFEST.json",
    "requirements.txt",
    "assets/kr810.xml",
    "benchmarks/point_ik/point_ik_v1.npz",
    "pipelines/run_tier0_to_tier4.py",
    "tests/test_asset_loading.py",
]

_KNOWN_LIMITATIONS = [
    "Tier 0-4 covers kinematic evaluation only (no actuator dynamics, no torque-level controller).",
    "No PPO, MPDIK, or MAPPO is included; this release is IK/kinematic-trajectory data only.",
    "ee_site is a MuJoCo end-effector reference frame, not a calibrated tool-center-point (TCP).",
    "Acceptance thresholds in configs/evaluation_config.json are project-defined criteria, not an "
    "ISO 9283 certification.",
    "A smoke-preset run is a fast sanity check, not this dataset's official research result.",
]


class PackagingError(Exception):
    """Raised when the staged tree or the produced ZIP/notebook fails a packaging validation check."""


def _is_safe_staging_path(path: Path) -> bool:
    """True iff ``path`` is REPO_ROOT/temporary_results itself or nested inside it."""
    resolved = path.resolve()
    temporary_results = (REPO_ROOT / "temporary_results").resolve()
    return resolved == temporary_results or temporary_results in resolved.parents


def _ignore_for_staging(directory: str, names: List[str]) -> set:
    ignored = set()
    for name in names:
        full = Path(directory) / name
        if full.is_symlink():
            ignored.add(name)
        elif full.is_dir() and (name in _EXCLUDE_DIR_NAMES or name.endswith(".egg-info")):
            ignored.add(name)
        elif full.is_file() and full.suffix in _EXCLUDE_FILE_SUFFIXES:
            ignored.add(name)
    return ignored


def stage_project(repo_root: Path, staging_root: Path) -> None:
    """Copy the explicit root allowlist from ``repo_root`` into a fresh ``staging_root``."""
    if not _is_safe_staging_path(staging_root):
        raise PackagingError(f"refusing to stage into unsafe path: {staging_root}")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    for filename in ALLOWLIST_ROOT_FILES:
        source = repo_root / filename
        if not source.is_file():
            raise PackagingError(f"required root file missing from repository: {filename}")
        shutil.copy2(source, staging_root / filename)

    for dirname in ALLOWLIST_ROOT_DIRS:
        source = repo_root / dirname
        if not source.is_dir():
            raise PackagingError(f"required root directory missing from repository: {dirname}")
        shutil.copytree(source, staging_root / dirname, ignore=_ignore_for_staging)


def validate_staging(staging_root: Path) -> None:
    """Structural checks on the staged tree before it is zipped."""
    for relative in _REQUIRED_STAGED_MEMBERS:
        if not (staging_root / relative).is_file():
            raise PackagingError(f"staged tree missing required file: {relative}")

    for path in staging_root.rglob("*"):
        if path.is_symlink():
            raise PackagingError(f"staged tree contains a symlink, refusing to package: {path}")
        if path.is_dir() and (path.name in _EXCLUDE_DIR_NAMES or path.name.endswith(".egg-info")):
            raise PackagingError(f"staged tree contains an excluded directory: {path}")
        if path.is_file() and path.suffix in _EXCLUDE_FILE_SUFFIXES:
            raise PackagingError(f"staged tree contains an excluded file: {path}")

    manifest = json.loads((staging_root / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    if "dataset_name" not in manifest:
        raise PackagingError("staged DATASET_MANIFEST.json missing 'dataset_name'")


def build_zip(staging_root: Path, zip_path: Path) -> None:
    """Deterministically zip ``staging_root`` (sorted, POSIX arcnames) into ``zip_path``."""
    if zip_path.exists():
        zip_path.unlink()

    relative_files = sorted(
        p.relative_to(staging_root).as_posix()
        for p in staging_root.rglob("*")
        if p.is_file() and not p.is_symlink()
    )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for relpath in relative_files:
            zf.write(staging_root / relpath, arcname=relpath)


def validate_zip(zip_path: Path) -> None:
    """Re-open the produced ZIP and check root layout, required members, and exclusions."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    if "DATASET_MANIFEST.json" not in names:
        raise PackagingError("DATASET_MANIFEST.json is not at the ZIP root (an outer folder may have been added)")

    for relative in _REQUIRED_STAGED_MEMBERS:
        if relative not in names:
            raise PackagingError(f"ZIP missing required member: {relative}")

    for name in names:
        if is_unsafe_member_name(name):
            raise PackagingError(f"ZIP contains an unsafe member path: {name!r}")
        lowered = name.lower()
        if lowered.startswith("kr810/") or "/kr810/" in lowered:
            raise PackagingError(f"ZIP contains the legacy kr810/ folder: {name!r}")
        for excluded_dir in _EXCLUDE_DIR_NAMES - {"kr810"}:
            if f"{excluded_dir}/" in name or name.startswith(f"{excluded_dir}/"):
                raise PackagingError(f"ZIP contains an excluded path: {name!r}")
        if any(name.endswith(suffix) for suffix in _EXCLUDE_FILE_SUFFIXES):
            raise PackagingError(f"ZIP contains an excluded file: {name!r}")


def release_notebook_copy(source_notebook: Path, destination_notebook: Path) -> None:
    """Copy ``source_notebook`` to ``destination_notebook``, stripping any outputs/execution counts."""
    nb = nbf.read(source_notebook, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    nbf.write(nb, destination_notebook)


def validate_notebook_release(notebook_path: Path) -> None:
    nb = nbf.read(notebook_path, as_version=4)
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            if cell.get("outputs", []) != []:
                raise PackagingError(f"release notebook has stale outputs: {notebook_path}")
            if cell.get("execution_count") is not None:
                raise PackagingError(f"release notebook has a stale execution_count: {notebook_path}")
            compile(cell.source, str(notebook_path), "exec")


def _git(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_release_info() -> Dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "source_git_commit": commit,
        "dirty_worktree": (len(status) > 0) if status is not None else None,
    }


def count_pytest_tests(repo_root: Path) -> Optional[int]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+) tests? collected", result.stdout)
    return int(match.group(1)) if match else None


def count_trajectory_files(staging_root: Path) -> int:
    with open(staging_root / "trajectories" / "trajectory_manifest.csv", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _read_manifest_int(manifest: dict, *keys: str) -> Optional[int]:
    node: Any = manifest
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return int(node) if isinstance(node, (int, float)) else None


def build_release_manifest(
    version: str,
    staging_root: Path,
    dataset_zip_path: Path,
    notebook_path: Path,
) -> Dict[str, Any]:
    manifest = json.loads((staging_root / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))

    dataset_zip_sha256 = sha256_file(dataset_zip_path)
    dataset_zip_size = dataset_zip_path.stat().st_size
    notebook_sha256 = sha256_file(notebook_path)
    notebook_size = notebook_path.stat().st_size

    release = {
        "release_name": "Kassow KR810 Tier0-4 Kaggle Release",
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **git_release_info(),
        "dataset_zip_filename": dataset_zip_path.name,
        "dataset_zip_sha256": dataset_zip_sha256,
        "dataset_zip_size_bytes": dataset_zip_size,
        "notebook_filename": notebook_path.name,
        "notebook_sha256": notebook_sha256,
        "notebook_size_bytes": notebook_size,
        "project_manifest_sha256": sha256_file(staging_root / "DATASET_MANIFEST.json"),
        "model_sha256": sha256_file(staging_root / "assets" / "kr810.xml"),
        "point_benchmark_sha256": sha256_file(staging_root / "benchmarks" / "point_ik" / "point_ik_v1.npz"),
        "trajectory_file_count": count_trajectory_files(staging_root),
        "point_sample_count": _read_manifest_int(manifest, "generation_summary", "point_ik_sample_count"),
        "test_count": count_pytest_tests(REPO_ROOT),
        "packaging_validation_status": "passed",
        "exclusions": sorted(_EXCLUDE_DIR_NAMES) + sorted(_EXCLUDE_FILE_SUFFIXES) + ["notebooks/ (released separately)"],
        "known_limitations": _KNOWN_LIMITATIONS,
    }
    return release


def _write_checksums_file(path: Path, entries: List[Path]) -> None:
    lines = [f"{sha256_file(entry)}  {entry.name}" for entry in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    version = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_zip_path = output_dir / f"KR810_Tier0_Tier4_Kaggle_Dataset_v{version}.zip"
    notebook_dest_path = output_dir / f"KR810_Tier0_Tier4_Kaggle_Template_v{version}.ipynb"
    release_manifest_path = output_dir / "RELEASE_MANIFEST.json"
    checksums_path = output_dir / "SHA256SUMS.txt"
    guide_dest_path = output_dir / "KAGGLE_UPLOAD_AND_RUN_GUIDE.md"

    if not args.overwrite:
        existing = [
            p for p in (dataset_zip_path, notebook_dest_path, release_manifest_path, checksums_path, guide_dest_path)
            if p.exists()
        ]
        if existing:
            listing = ", ".join(str(p) for p in existing)
            print(f"error: output files already exist ({listing}); pass --overwrite to replace them", file=sys.stderr)
            return 2

    if not NOTEBOOK_SOURCE.is_file():
        print(f"error: notebook source not found: {NOTEBOOK_SOURCE}", file=sys.stderr)
        return 2
    if not UPLOAD_GUIDE_SOURCE.is_file():
        print(f"error: upload guide source not found: {UPLOAD_GUIDE_SOURCE}", file=sys.stderr)
        return 2

    try:
        stage_project(REPO_ROOT, STAGING_ROOT)
        validate_staging(STAGING_ROOT)
        build_zip(STAGING_ROOT, dataset_zip_path)
        validate_zip(dataset_zip_path)

        release_notebook_copy(NOTEBOOK_SOURCE, notebook_dest_path)
        validate_notebook_release(notebook_dest_path)

        shutil.copy2(UPLOAD_GUIDE_SOURCE, guide_dest_path)

        manifest = build_release_manifest(version, STAGING_ROOT, dataset_zip_path, notebook_dest_path)
        release_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        _write_checksums_file(checksums_path, [dataset_zip_path, notebook_dest_path, release_manifest_path, guide_dest_path])
    finally:
        if STAGING_ROOT.exists():
            shutil.rmtree(STAGING_ROOT)

    print(f"Dataset ZIP : {dataset_zip_path} ({dataset_zip_path.stat().st_size} bytes)")
    print(f"Notebook    : {notebook_dest_path}")
    print(f"Guide       : {guide_dest_path}")
    print(f"Manifest    : {release_manifest_path}")
    print(f"Checksums   : {checksums_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
