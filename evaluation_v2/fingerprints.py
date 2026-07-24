"""Deterministic fingerprints used to gate resume and to fill the evaluation lock bundle.

* **config fingerprint** -- SHA256 of the canonical JSON of a resolved config dict.
* **code fingerprint** -- SHA256 over the source bytes of the evaluation-relevant modules
  (``evaluation_v2/``, the reused ``algorithms/`` and ``kinematics/`` DLS math, and the reused
  ``evaluation/`` metric modules). Any edit to solver/metric/harness code changes it.
* **dataset / public / protected fingerprints** -- SHA256 over a sorted (relative-path, file
  sha256) list of the files under a root, so a changed input is detected.
* **environment fingerprint** -- Python / NumPy / SciPy / MuJoCo / platform, so a run is
  reproducible-context-tagged and a resume onto a different environment is visible.

Nothing here mutates state or touches global RNG.
"""

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Iterable, Union

from utils.file_checksum import sha256_file

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Source trees whose bytes define the evaluation code fingerprint.
_CODE_DIRS = ("evaluation_v2", "algorithms", "kinematics", "evaluation")


def canonical_json(obj) -> str:
    """Deterministic JSON string: sorted keys, compact separators, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_fingerprint(config: dict) -> str:
    """SHA256 of the canonical JSON of a config dict."""
    return sha256_of_text(canonical_json(config))


def _iter_py_files(dirs: Iterable[str]) -> Iterable[Path]:
    for d in dirs:
        base = REPO_ROOT / d
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


def code_fingerprint() -> str:
    """SHA256 over the sorted (relative-path, sha256) list of evaluation-relevant source files."""
    hasher = hashlib.sha256()
    for path in _iter_py_files(_CODE_DIRS):
        rel = path.relative_to(REPO_ROOT).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(sha256_file(path).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def directory_fingerprint(root: Union[str, Path], *, skip_names: Iterable[str] = ()) -> dict:
    """Fingerprint every file under ``root``.

    Returns a dict with the aggregate ``sha256``, the ``file_count``, and a sorted
    ``files`` list of ``{"path": <relative posix>, "sha256": ...}`` entries. ``skip_names`` is a
    set of basenames to ignore (e.g. a manifest that embeds this very fingerprint).
    """
    root = Path(root).resolve()
    skip = set(skip_names)
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in skip:
            continue
        rel = path.relative_to(root).as_posix()
        entries.append({"path": rel, "sha256": sha256_file(path)})
    hasher = hashlib.sha256()
    for entry in entries:
        hasher.update(entry["path"].encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(entry["sha256"].encode("ascii"))
        hasher.update(b"\n")
    return {"sha256": hasher.hexdigest(), "file_count": len(entries), "files": entries}


def environment_fingerprint() -> dict:
    """Reproducibility-context fingerprint of the running interpreter / libraries / platform."""
    versions = {}
    for mod in ("numpy", "scipy", "mujoco", "pandas"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover - a missing optional dep should not crash fingerprints
            versions[mod] = None
    payload = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        **versions,
    }
    payload["sha256"] = sha256_of_text(canonical_json(payload))
    return payload
