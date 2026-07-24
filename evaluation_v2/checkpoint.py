"""Deterministic checkpoint / resume for the large Phase 8A DLS workload.

The workload is decomposed into **shards** -- the smallest unit of work whose output is written
exactly once, atomically:

* Tier 1 point-IK: one shard per split (or per chunk of samples).
* Tier 2 trajectory: one shard per ``(split, method, trial_id)`` -- an interruption loses at most
  one trial's 400 waypoints.

Each shard is a pandas DataFrame written to ``checkpoint/shards/<shard_key>.csv`` via
temp-file + ``os.replace`` (atomic on Windows and POSIX). A ``checkpoint/shard_index.json`` records
every completed shard's SHA256 and row count. Resume:

* reloads the index, and treats a shard as done only if its CSV is present AND its SHA256 matches
  the index (a half-written / corrupted shard is redone, never trusted);
* refuses to resume if the run's config / code / dataset / public-bundle fingerprints differ from
  the ones recorded in ``checkpoint/run_lock.json`` (a changed input must not be silently mixed
  into an old run).

Final per-tier CSVs are assembled by concatenating completed shards in a deterministic key order,
so the output never contains duplicate rows regardless of how many resumes occurred.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from evaluation_v2 import fingerprints
from utils.file_checksum import sha256_file


class ResumeMismatchError(RuntimeError):
    """Raised when a resume is attempted against a run with different fingerprints."""


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


@dataclass
class RunFingerprints:
    """The four fingerprints that must match for a resume to be accepted."""

    config: str
    code: str
    dataset: str
    public_bundle: str

    def to_dict(self) -> dict:
        return {
            "config_fingerprint": self.config,
            "code_fingerprint": self.code,
            "dataset_fingerprint": self.dataset,
            "public_bundle_fingerprint": self.public_bundle,
        }


class CheckpointManager:
    """Owns a single config-run directory's checkpoint state."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoint"
        self.shards_dir = self.checkpoint_dir / "shards"
        self.index_file = self.checkpoint_dir / "shard_index.json"
        self.lock_file = self.checkpoint_dir / "run_lock.json"
        self._index: Dict[str, dict] = {}

    # ---- lifecycle -------------------------------------------------------------------------
    def begin(self, fingerprints_: RunFingerprints, *, resume: bool, overwrite: bool) -> None:
        """Create or validate the run directory and its lock file."""
        if self.lock_file.is_file():
            existing = json.loads(self.lock_file.read_text(encoding="utf-8"))
            if overwrite and not resume:
                self._reset()
            elif resume:
                self._validate_lock(existing, fingerprints_)
                self._load_index()
            else:
                raise FileExistsError(
                    f"run directory {self.run_dir} already has a checkpoint lock; pass resume=True "
                    "to continue it or overwrite=True to discard it."
                )
        else:
            if resume and not overwrite:
                # Fresh resume of a not-yet-started run is allowed (nothing to validate).
                pass
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        if not self.lock_file.is_file():
            lock = {"fingerprints": fingerprints_.to_dict(), "status": "in_progress"}
            _atomic_write_text(self.lock_file, json.dumps(lock, indent=2, sort_keys=True))
            self._index = {}
            self._save_index()

    def _reset(self) -> None:
        import shutil

        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        self._index = {}

    def _validate_lock(self, existing: dict, fingerprints_: RunFingerprints) -> None:
        recorded = existing.get("fingerprints", {})
        wanted = fingerprints_.to_dict()
        mismatched = {k: (recorded.get(k), wanted[k]) for k in wanted if recorded.get(k) != wanted[k]}
        if mismatched:
            raise ResumeMismatchError(
                f"cannot resume {self.run_dir}: fingerprint mismatch {sorted(mismatched)}. "
                "Config, code, dataset, or public bundle changed since the run started."
            )

    # ---- shard index -----------------------------------------------------------------------
    def _load_index(self) -> None:
        if self.index_file.is_file():
            self._index = json.loads(self.index_file.read_text(encoding="utf-8"))
        else:
            self._index = {}

    def _save_index(self) -> None:
        _atomic_write_text(self.index_file, json.dumps(self._index, indent=2, sort_keys=True))

    def shard_file(self, shard_key: str) -> Path:
        safe = shard_key.replace("/", "__")
        return self.shards_dir / f"{safe}.csv"

    def has_shard(self, shard_key: str) -> bool:
        """True only if the shard is recorded AND its on-disk CSV still matches the recorded hash."""
        entry = self._index.get(shard_key)
        if entry is None:
            return False
        path = self.shard_file(shard_key)
        if not path.is_file():
            return False
        return sha256_file(path) == entry["sha256"]

    def write_shard(self, shard_key: str, df: pd.DataFrame) -> None:
        path = self.shard_file(shard_key)
        _atomic_write_csv(path, df)
        self._index[shard_key] = {"sha256": sha256_file(path), "rows": int(len(df))}
        self._save_index()

    def read_shard(self, shard_key: str) -> pd.DataFrame:
        return pd.read_csv(self.shard_file(shard_key))

    def completed_shard_keys(self) -> List[str]:
        return sorted(k for k in self._index if self.has_shard(k))

    def assemble(self, shard_keys: List[str]) -> pd.DataFrame:
        """Concatenate the given shards (in the given order) into one DataFrame, no duplicates."""
        frames = [self.read_shard(k) for k in shard_keys if self.has_shard(k)]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def mark_complete(self) -> None:
        if self.lock_file.is_file():
            lock = json.loads(self.lock_file.read_text(encoding="utf-8"))
            lock["status"] = "complete"
            _atomic_write_text(self.lock_file, json.dumps(lock, indent=2, sort_keys=True))


def build_run_fingerprints(
    resolved_config: dict, dataset_fingerprint: str, public_bundle_fingerprint: str
) -> RunFingerprints:
    return RunFingerprints(
        config=fingerprints.config_fingerprint(resolved_config),
        code=fingerprints.code_fingerprint(),
        dataset=dataset_fingerprint,
        public_bundle=public_bundle_fingerprint,
    )
