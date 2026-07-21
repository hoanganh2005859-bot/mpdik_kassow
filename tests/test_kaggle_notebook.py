"""Static validation for notebooks/KR810_Tier0_Tier4_Kaggle_Template.ipynb.

Checks structure and integration hygiene without executing the notebook (execution requires a
Kaggle-like /kaggle/input mount, or a full local pipeline run -- see notebooks/README.md for the
local integration validation procedure). Every check here is a static, repository-only check:
nbformat validity, no stale outputs/execution counts, every code cell compiles, no notebook magic,
no hardcoded absolute Windows paths, no hardcoded Kaggle Dataset slug, generic /kaggle/input and
/kaggle/working usage, no forbidden imports/rendering, and no copied-in DLS/FK/Jacobian algorithm
code (the notebook must only call into kinematics/algorithms/evaluation/pipelines).
"""

import json
import re
from pathlib import Path

import nbformat as nbf

from utils.dataset_locator import REPO_ROOT

NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "KR810_Tier0_Tier4_Kaggle_Template.ipynb"

_FORBIDDEN_IMPORT_NAMES = ("torch", "stable_baselines3", "gymnasium")
_FORBIDDEN_RENDER_PATTERNS = ("mujoco.viewer", "mj_render", "MjViewer", "launch_passive")
_FORBIDDEN_ALGORITHM_PATTERNS = (
    "def forward_kinematics", "def compute_jacobian", "def damped_least_squares", "def dls_step",
)
_WINDOWS_ABS_PATH_PATTERN = re.compile(r"[A-Za-z]:[\\/](?:Users|Kassow|Windows|Program Files)")
_HARDCODED_KAGGLE_SLUG_PATTERN = re.compile(r"/kaggle/input/[A-Za-z0-9][\w.-]+(?:/|['\"])")


def _load_notebook():
    assert NOTEBOOK_PATH.is_file(), f"notebook not found at {NOTEBOOK_PATH}"
    return nbf.read(NOTEBOOK_PATH, as_version=4)


def _code_cell_sources(nb):
    return [cell.source for cell in nb.cells if cell.cell_type == "code"]


def _all_source_text(nb):
    return "\n".join(cell.source for cell in nb.cells)


def test_notebook_exists():
    assert NOTEBOOK_PATH.is_file()


def test_notebook_is_nbformat_v4():
    nb = _load_notebook()
    nbf.validate(nb)
    assert nb.nbformat == 4


def test_notebook_has_expected_cell_count_and_key_sections():
    nb = _load_notebook()
    assert 17 <= len(nb.cells) <= 20, f"expected 17-20 cells, got {len(nb.cells)}"

    full_text = _all_source_text(nb).lower()
    for marker in (
        "user configuration",
        "environment diagnostics",
        "locate dataset",
        "copy project to working",
        "dependency check",
        "import path",
        "validate dataset",
        "resolve pipeline arguments",
        "precheck",
        "run pipeline",
        "final_summary",
        "warm_vs_cold",
        "tier 3",
        "tier 4",
        "package",
    ):
        assert marker in full_text, f"expected section marker {marker!r} not found in notebook"


def test_no_execution_outputs():
    nb = _load_notebook()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert cell.get("outputs", []) == [], f"code cell {i} has stale outputs"


def test_no_execution_counts():
    nb = _load_notebook()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert cell.get("execution_count") is None, f"code cell {i} has a stale execution_count"


def test_every_code_cell_parses():
    nb = _load_notebook()
    for i, source in enumerate(_code_cell_sources(nb)):
        compile(source, f"<cell {i}>", "exec")


def test_no_notebook_magics():
    nb = _load_notebook()
    for i, source in enumerate(_code_cell_sources(nb)):
        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("%"), f"cell {i} uses notebook magic: {line!r}"
            assert not stripped.startswith("!"), f"cell {i} uses shell-escape magic: {line!r}"


def test_no_hardcoded_windows_absolute_paths():
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    match = _WINDOWS_ABS_PATH_PATTERN.search(full_text)
    assert match is None, f"found a hardcoded Windows absolute path: {match.group(0)!r}"


def test_no_hardcoded_kaggle_dataset_slug():
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    match = _HARDCODED_KAGGLE_SLUG_PATTERN.search(full_text)
    assert match is None, f"found a hardcoded /kaggle/input/<slug> path: {match.group(0)!r}"


def test_has_generic_kaggle_input_discovery():
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    assert "/kaggle/input" in full_text
    assert "iterdir" in full_text or "glob" in full_text


def test_has_kaggle_working_output():
    nb = _load_notebook()
    assert "/kaggle/working" in _all_source_text(nb)


def test_has_dataset_manifest_discovery():
    nb = _load_notebook()
    assert "DATASET_MANIFEST.json" in _all_source_text(nb)


def test_has_user_configuration_cell():
    nb = _load_notebook()
    code_sources = _code_cell_sources(nb)
    config_cell = next((s for s in code_sources if "PRESET" in s and "RUN_NAME" in s), None)
    assert config_cell is not None, "no code cell defines PRESET/RUN_NAME user configuration"
    for var in ("PRESET", "SEED", "METHODS", "TRIAL_LIMIT", "POINT_SAMPLE_LIMIT", "WAYPOINT_LIMIT", "RESUME", "OVERWRITE"):
        assert var in config_cell, f"user configuration cell missing {var}"


def test_has_smoke_and_full_preset_support():
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    assert '"smoke"' in full_text
    assert '"full"' in full_text


def test_has_zip_packaging():
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    assert "zipfile" in full_text
    assert "ZipFile" in full_text


def test_no_forbidden_ml_imports():
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    for name in _FORBIDDEN_IMPORT_NAMES:
        assert name not in full_text, f"notebook references forbidden dependency: {name}"


def test_no_mujoco_rendering():
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    for pattern in _FORBIDDEN_RENDER_PATTERNS:
        assert pattern not in full_text, f"notebook appears to render MuJoCo: {pattern}"


def test_never_writes_into_kaggle_input():
    nb = _load_notebook()
    for i, source in enumerate(_code_cell_sources(nb)):
        for line in source.splitlines():
            if "/kaggle/input" not in line:
                continue
            assert not re.search(r"open\([^)]*/kaggle/input[^)]*,\s*[\"']w", line), (
                f"cell {i} appears to open a /kaggle/input path for writing: {line!r}"
            )
            assert "shutil.rmtree" not in line, f"cell {i} appears to delete under /kaggle/input: {line!r}"
            assert not re.search(r"\.write_text\(", line) or "read_text" in line, (
                f"cell {i} may write into /kaggle/input: {line!r}"
            )


def test_no_copied_dls_or_fk_algorithm_code():
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    for pattern in _FORBIDDEN_ALGORITHM_PATTERNS:
        assert pattern not in full_text, f"notebook appears to reimplement an algorithm: {pattern}"

    # The notebook must invoke the pipeline module rather than defining its own tier logic.
    assert "pipelines.run_tier0_to_tier4" in full_text
    assert "import kinematics" in full_text or "from kinematics" in full_text


def test_dataset_manifest_matches_repo():
    manifest = json.loads((REPO_ROOT / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    assert manifest["dataset_name"] in full_text


def test_has_dataset_zip_override_variable():
    nb = _load_notebook()
    code_sources = _code_cell_sources(nb)
    config_cell = next((s for s in code_sources if "PRESET" in s and "RUN_NAME" in s), None)
    assert config_cell is not None
    assert "DATASET_ROOT_OVERRIDE" in config_cell
    assert "DATASET_ZIP_OVERRIDE" in config_cell


def test_has_zip_fallback_discovery():
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    assert "*.zip" in full_text or ".glob(" in full_text
    assert "_zip_root_manifest" in full_text or "_resolve_via_zip_fallback" in full_text


def test_has_safe_extraction_with_checks():
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    assert "_safe_extract_zip" in full_text
    assert "_is_unsafe_member_name" in full_text
    assert "_is_symlink_member" in full_text
    # Basic zip-bomb guard: a member-count and/or uncompressed-size limit must be enforced.
    assert "_MAX_ZIP_MEMBERS" in full_text
    assert "_MAX_ZIP_UNCOMPRESSED_BYTES" in full_text


def test_never_calls_extractall_unchecked():
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    assert "extractall" not in full_text, "notebook must extract member-by-member with safety checks, not extractall()"


def test_extraction_path_is_under_kaggle_working():
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    assert "_kr810_dataset_extracted" in full_text
    assert '"/kaggle/working/_kr810_dataset_extracted"' in full_text


def test_expanded_dataset_discovery_is_tried_before_zip_fallback():
    nb = _load_notebook()
    locate_cell = next(
        s for s in _code_cell_sources(nb)
        if "_looks_like_project_root" in s and "_resolve_via_zip_fallback" in s
    )
    expanded_check_index = locate_cell.index("if len(candidates) == 0:")
    zip_fallback_call_index = locate_cell.index("_resolve_via_zip_fallback(kaggle_input)")
    assert expanded_check_index < zip_fallback_call_index, (
        "ZIP fallback must only be attempted after the expanded-dataset candidate search finds none"
    )


def test_no_hardcoded_release_zip_filename():
    nb = _load_notebook()
    full_text = _all_source_text(nb)
    assert "Kaggle_Dataset_v" not in full_text, "notebook must discover the dataset ZIP by content, not by a fixed release filename"
