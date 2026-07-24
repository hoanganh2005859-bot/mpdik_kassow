"""Build the public evaluation root and the protected validation root from a Dataset v2 root.

The public root is the ONLY input the DLS evaluator ever reads. It is written by *whitelisting*
public fields (never by blacklisting protected ones), so a field is public only if it is named
here explicitly:

* Point-IK (per split): ``q_initial`` + target pose + public covariates of ``q_initial``. The
  reference solution ``q_target_reference`` and every covariate derived from it
  (``target_sigma_*``, ``target_condition_number``, ``joint_distance_rad``,
  ``minimum_target_limit_margin_*``) are excluded.
* Trials (per split): ``q_initial`` + trial/trajectory identity + difficulty. (The trial NPZ
  already carries no reference array; the whitelist keeps it minimal and finite-safe.)
* Trajectories (per split): public geometry/timing only -- every array in
  :data:`dataset_v2.trajectory_loading.PROTECTED_ARRAY_KEYS` is dropped.

Only ``development`` and ``validation`` are exported. ``frozen_test`` is never read or written.

The protected root holds the stripped-out reference evidence (``q_target_reference`` for point-IK,
per-trajectory ``q_reference`` content hashes) OUTSIDE the public root, purely so an independent
test can prove no public ``q_initial`` coincides with a reference solution. The evaluator never
reads it.
"""

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from dataset_v2.locator import require_dataset_v2_root
from dataset_v2.trajectory_catalog import load_combined_catalog
from dataset_v2.trajectory_loading import PROTECTED_ARRAY_KEYS, load_protected_trajectory
from evaluation_v2 import fingerprints
from evaluation_v2.locator import (
    EVAL_SPLITS,
    protected_validation_paths,
    public_eval_paths,
)
from evaluation_v2.protected_guard import assert_no_protected_fields, find_protected_fields
from utils.file_checksum import sha256_file
from utils.npz_utils import load_npz, save_npz

# Whitelisted public Point-IK arrays (finite-safe: no condition-number columns, which can be inf).
PUBLIC_POINT_IK_KEYS = (
    "sample_id",
    "split_id",
    "difficulty_id",
    "q_initial",
    "initial_position",
    "initial_quaternion_wxyz",
    "target_position",
    "target_quaternion_wxyz",
    "position_distance_m",
    "orientation_distance_rad",
    "initial_sigma_min",
    "minimum_initial_limit_margin_normalized",
    "source_seed",
    "content_hash",
)

# Reference-solution-derived Point-IK arrays kept OUT of the public root.
PROTECTED_POINT_IK_KEYS = (
    "q_target_reference",
    "joint_distance_rad",
    "target_sigma_min",
    "target_sigma_max",
    "target_condition_number",
    "minimum_target_limit_margin_normalized",
    "minimum_target_limit_margin_rad",
)

# Whitelisted public trial arrays.
PUBLIC_TRIAL_KEYS = (
    "trial_id",
    "trajectory_id",
    "trajectory_family",
    "split",
    "difficulty",
    "difficulty_id",
    "q_initial",
    "first_target_position",
    "first_target_quaternion_wxyz",
    "content_hash",
)

# Public trajectory manifest columns (source_* / seed columns dropped: source NPZ is protected).
PUBLIC_MANIFEST_COLUMNS = (
    "trajectory_id",
    "family",
    "split",
    "shape",
    "orientation_mode",
    "anchor_id",
    "anchor_class",
    "challenge_family",
    "canonical_waypoint_count",
    "source_waypoint_count",
    "content_hash",
    "public_canonical_path",
    "public_sha256",
)


def _select_public_arrays(raw: Dict[str, np.ndarray], keys) -> Dict[str, np.ndarray]:
    selected = {k: raw[k] for k in keys if k in raw}
    leaked = find_protected_fields(selected.keys())
    if leaked:  # pragma: no cover - defensive; whitelist excludes these by construction
        raise AssertionError(f"public whitelist unexpectedly included protected fields: {leaked}")
    return selected


def _write_json(path: Path, obj: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _write_manifest_csv(path: Path, columns, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    tmp.replace(path)


def export_public_and_protected(
    dataset_root,
    public_root,
    protected_root,
    *,
    overwrite: bool = False,
) -> dict:
    """Export the public evaluation root and protected validation root for development+validation.

    Returns a summary dict (also written to both roots' manifests).
    """
    ds = require_dataset_v2_root(dataset_root)
    public = public_eval_paths(public_root, require_exists=False)
    protected = protected_validation_paths(protected_root, require_exists=False)

    for root_paths, label in ((public, "public"), (protected, "protected")):
        if root_paths.root.exists() and any(root_paths.root.iterdir()) and not overwrite:
            raise FileExistsError(
                f"{label} root {root_paths.root} already exists and is non-empty; pass overwrite=True."
            )
    if public.root.resolve() == ds.root.resolve() or protected.root.resolve() == ds.root.resolve():
        raise ValueError("public/protected roots must be distinct from the dataset root.")

    public.point_ik_dir.mkdir(parents=True, exist_ok=True)
    public.trials_dir.mkdir(parents=True, exist_ok=True)
    public.trajectories_dir.mkdir(parents=True, exist_ok=True)
    protected.point_ik_dir.mkdir(parents=True, exist_ok=True)
    protected.trajectories_dir.mkdir(parents=True, exist_ok=True)

    summary = {"splits": list(EVAL_SPLITS), "point_ik": {}, "trials": {}, "trajectories": {}}

    # ----- Point-IK -----
    for split in EVAL_SPLITS:
        raw = load_npz(ds.tier1_point_ik_dir / f"{split}.npz")
        public_arrays = _select_public_arrays(raw, PUBLIC_POINT_IK_KEYS)
        save_npz(public.point_ik_split_file(split), public_arrays, overwrite=True)
        # Protected evidence: the reference solution + its id, kept out of the public root.
        prot = {k: raw[k] for k in PROTECTED_POINT_IK_KEYS if k in raw}
        prot["sample_id"] = raw["sample_id"]
        save_npz(protected.point_ik_split_file(split), prot, overwrite=True)
        summary["point_ik"][split] = int(raw["sample_id"].shape[0])

    # ----- Trials -----
    for split in EVAL_SPLITS:
        raw = load_npz(ds.trials_dir / f"{split}.npz")
        assert_no_protected_fields(raw, f"trial NPZ {split} (source)")
        public_arrays = _select_public_arrays(raw, PUBLIC_TRIAL_KEYS)
        save_npz(public.trials_split_file(split), public_arrays, overwrite=True)
        summary["trials"][split] = int(raw["trial_id"].shape[0])

    # ----- Trajectories -----
    catalog = load_combined_catalog(ds.root)
    manifest_rows_by_split: Dict[str, List[dict]] = {s: [] for s in EVAL_SPLITS}
    ref_hashes_by_split: Dict[str, Dict[str, str]] = {s: {} for s in EVAL_SPLITS}
    n_traj = {s: 0 for s in EVAL_SPLITS}
    for row in catalog:
        split = row["split"]
        if split not in EVAL_SPLITS:
            continue  # never export frozen_test
        traj = load_protected_trajectory(ds.root, row["trajectory_id"], catalog_row=row)
        public_canonical = {k: v for k, v in traj.canonical.items() if k not in PROTECTED_ARRAY_KEYS}
        leaked = find_protected_fields(public_canonical.keys())
        if leaked:  # pragma: no cover - defensive
            raise AssertionError(f"public trajectory would leak {leaked}")
        out_dir = public.trajectory_split_dir(split)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{row['trajectory_id']}.npz"
        save_npz(out_file, public_canonical, overwrite=True)
        rel = out_file.relative_to(public.root).as_posix()
        manifest_rows_by_split[split].append(
            {
                **{c: row.get(c, "") for c in PUBLIC_MANIFEST_COLUMNS if c in row},
                "public_canonical_path": rel,
                "public_sha256": sha256_file(out_file),
            }
        )
        # Protected evidence: a content hash of q_reference (never the array itself in the public root).
        q_ref = np.asarray(traj.canonical["q_reference"], dtype=np.float64)
        ref_hashes_by_split[split][row["trajectory_id"]] = fingerprints.sha256_of_text(
            fingerprints.canonical_json(np.round(q_ref, 12).tolist())
        )
        n_traj[split] += 1

    # Public per-split manifests + a combined public manifest.
    combined_rows: List[dict] = []
    for split in EVAL_SPLITS:
        rows = sorted(manifest_rows_by_split[split], key=lambda r: r["trajectory_id"])
        _write_manifest_csv(
            public.trajectory_split_dir(split) / "public_manifest.csv", PUBLIC_MANIFEST_COLUMNS, rows
        )
        combined_rows.extend(rows)
        summary["trajectories"][split] = n_traj[split]
    _write_manifest_csv(public.combined_manifest_file(), PUBLIC_MANIFEST_COLUMNS, sorted(combined_rows, key=lambda r: r["trajectory_id"]))

    # Protected reference-hash files.
    for split in EVAL_SPLITS:
        _write_json(
            protected.trajectory_reference_file(split),
            {"split": split, "q_reference_hashes": ref_hashes_by_split[split]},
        )

    # ----- Manifests with fingerprints -----
    dataset_fp = fingerprints.directory_fingerprint(ds.root)
    public_manifest = {
        "source_dataset_root_fingerprint": dataset_fp["sha256"],
        "source_dataset_file_count": dataset_fp["file_count"],
        "splits_exported": list(EVAL_SPLITS),
        "frozen_test_exported": False,
        "counts": summary,
        "public_point_ik_keys": list(PUBLIC_POINT_IK_KEYS),
        "public_trial_keys": list(PUBLIC_TRIAL_KEYS),
        "protected_keys_excluded": sorted(set(PROTECTED_POINT_IK_KEYS) | set(PROTECTED_ARRAY_KEYS)),
        "code_fingerprint": fingerprints.code_fingerprint(),
        "environment": fingerprints.environment_fingerprint(),
    }
    _write_json(public.manifest_file, public_manifest)
    # Public bundle fingerprint (over everything except the manifest that embeds it).
    public_fp = fingerprints.directory_fingerprint(public.root, skip_names={public.manifest_file.name})
    public_manifest["public_bundle_fingerprint"] = public_fp["sha256"]
    public_manifest["public_bundle_file_count"] = public_fp["file_count"]
    _write_json(public.manifest_file, public_manifest)

    protected_fp = fingerprints.directory_fingerprint(protected.root, skip_names={protected.manifest_file.name})
    protected_manifest = {
        "purpose": "isolation evidence only; never read by the evaluator",
        "splits": list(EVAL_SPLITS),
        "protected_bundle_fingerprint": protected_fp["sha256"],
        "protected_bundle_file_count": protected_fp["file_count"],
    }
    _write_json(protected.manifest_file, protected_manifest)

    summary["public_bundle_fingerprint"] = public_manifest["public_bundle_fingerprint"]
    summary["protected_bundle_fingerprint"] = protected_fp["sha256"]
    return summary
