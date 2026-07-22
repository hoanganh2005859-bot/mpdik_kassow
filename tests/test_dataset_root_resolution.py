"""Backward-compatibility tests for the dataset-root resolution mechanism added in Phase 1.

Proves: Dataset v1 still loads by default and via an explicit dataset root pointing at the repo
root, resolution never depends on CWD, invalid roots raise actionable errors, and no absolute
path assumption was introduced. See utils/dataset_locator.py::resolve_dataset_root/dataset_paths_for.
"""

import os

import numpy as np
import pytest

from utils.dataset_locator import (
    ASSETS_DIR,
    BENCHMARKS_DIR,
    CONFIGS_DIR,
    MODEL_PATH,
    POINT_IK_BENCHMARK_PATH,
    REPO_ROOT,
    SCHEMAS_DIR,
    TRAJECTORIES_DIR,
    dataset_paths_for,
    resolve_dataset_root,
)
from utils.exceptions import ModelConfigurationError


def test_default_resolution_matches_repo_root():
    assert resolve_dataset_root(None) == REPO_ROOT


def test_default_resolution_is_unchanged_v1_behavior():
    # Dataset v1 loads today via the bare module constants; confirm those constants still point
    # at real, existing v1 content with no code path change.
    assert ASSETS_DIR.is_dir()
    assert BENCHMARKS_DIR.is_dir()
    assert TRAJECTORIES_DIR.is_dir()
    assert MODEL_PATH.is_file()
    assert POINT_IK_BENCHMARK_PATH.is_file()


def test_explicit_repo_root_resolves_identically_to_default():
    explicit = resolve_dataset_root(str(REPO_ROOT))
    assert explicit == REPO_ROOT == resolve_dataset_root(None)


def test_dataset_paths_for_explicit_root_matches_v1_constants():
    paths = dataset_paths_for(resolve_dataset_root(str(REPO_ROOT)))
    assert paths.assets_dir == ASSETS_DIR
    assert paths.benchmarks_dir == BENCHMARKS_DIR
    assert paths.trajectories_dir == TRAJECTORIES_DIR
    assert paths.configs_dir == CONFIGS_DIR
    assert paths.schemas_dir == SCHEMAS_DIR
    assert paths.model_path == MODEL_PATH
    assert paths.point_ik_benchmark_path == POINT_IK_BENCHMARK_PATH


def test_dataset_v1_loads_via_explicit_dataset_root_path_bundle():
    paths = dataset_paths_for(resolve_dataset_root(str(REPO_ROOT)))
    with np.load(paths.point_ik_benchmark_path, allow_pickle=False) as data:
        assert data["sample_id"].shape[0] == 1200


def test_resolution_independent_of_current_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_dataset_root(str(REPO_ROOT)) == REPO_ROOT
    assert os.getcwd() == str(tmp_path)


def test_invalid_dataset_root_raises_actionable_error(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ModelConfigurationError, match="does not exist"):
        resolve_dataset_root(missing)


def test_dataset_root_must_be_a_directory(tmp_path):
    a_file = tmp_path / "not_a_dir.txt"
    a_file.write_text("x", encoding="utf-8")
    with pytest.raises(ModelConfigurationError, match="not a directory"):
        resolve_dataset_root(a_file)


def test_require_exists_false_allows_nonexistent_root(tmp_path):
    missing = tmp_path / "future_root"
    resolved = resolve_dataset_root(missing, require_exists=False)
    assert resolved == missing.resolve()


def test_no_absolute_path_literals_introduced_in_module():
    # dataset_locator.py must derive everything from Path(__file__)/an explicit root, never a
    # hardcoded platform path literal (mirrors the module's own docstring guarantee). Only code
    # lines are checked -- the docstring itself mentions these strings as examples of what NOT
    # to do, so it is deliberately excluded.
    import ast
    import inspect

    import utils.dataset_locator as mod

    source = inspect.getsource(mod)
    tree = ast.parse(source)
    code_only_lines = set(range(1, len(source.splitlines()) + 1))
    docstring_node = tree.body[0]
    if isinstance(docstring_node, ast.Expr) and isinstance(docstring_node.value, ast.Constant):
        code_only_lines -= set(range(docstring_node.lineno, docstring_node.end_lineno + 1))

    code_lines = [line for i, line in enumerate(source.splitlines(), start=1) if i in code_only_lines]
    code_only_source = "\n".join(code_lines)
    assert "C:\\\\" not in code_only_source
    assert "/home/" not in code_only_source
