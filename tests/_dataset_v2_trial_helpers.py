"""Shared fast fixture builder for Dataset v2 Phase 7 (trial) tests.

Fabricates a minimal Dataset v2 root with a handful of *static* trajectories built from FK only
(no DLS reachability solving), so trial-generation/validation/catalog/loader tests run in seconds.
Each fabricated trajectory's canonical/source NPZ carries the same public + protected array layout
the real generators produce (including ``q_reference``), with ``FK(q_reference[0])`` equal to the
first target pose by construction.
"""

import csv

import numpy as np

from dataset_v2.challenge_trajectory_generation import MANIFEST_COLUMNS as CHALLENGE_COLS
from dataset_v2.challenge_trajectory_generation import MANIFEST_NAME as CHALLENGE_MANIFEST
from dataset_v2.checksums import content_hash_of_record
from dataset_v2.core_trajectory_generation import MANIFEST_COLUMNS as CORE_COLS
from dataset_v2.core_trajectory_generation import MANIFEST_NAME as CORE_MANIFEST
from dataset_v2.locator import dataset_v2_paths
from dataset_v2.scaffold import create_dataset_v2_scaffold
from dataset_v2.trajectory_catalog import build_combined_catalog
from kinematics.forward_kinematics import forward_kinematics
from kinematics.quaternion_utils import canonicalize_quaternion_wxyz
from utils.dataset_locator import MODEL_PATH as V1_MODEL_PATH
from utils.file_checksum import sha256_file
from utils.npz_utils import save_npz

CANON = 8
SOURCE = 8
MODEL_FINGERPRINT = sha256_file(V1_MODEL_PATH)


def _split_dir(paths, split):
    return {
        "development": paths.trajectories_development_dir,
        "validation": paths.trajectories_validation_dir,
        "frozen_test": paths.trajectories_frozen_test_dir,
    }[split]


def _write_trajectory(paths, model_context, split, trajectory_id, q):
    fk = forward_kinematics(model_context, q)
    quat = canonicalize_quaternion_wxyz(fk.quaternion_wxyz)
    pos = fk.position

    canonical = {
        "waypoint_id": np.arange(CANON, dtype=np.int64),
        "time_s": np.linspace(0.0, 1.0, CANON),
        "source_parameter_u": np.linspace(0.0, 1.0, CANON),
        "cumulative_arc_length_m": np.zeros(CANON),
        "target_position": np.tile(pos, (CANON, 1)),
        "target_quaternion": np.tile(quat, (CANON, 1)),
        "cumulative_angular_displacement_rad": np.zeros(CANON),
        "q_reference": np.tile(q, (CANON, 1)),
        "position_reconstruction_error_m": np.zeros(CANON),
        "orientation_reconstruction_error_rad": np.zeros(CANON),
        "waypoint_reachable": np.ones(CANON, dtype=bool),
    }
    source = {
        "waypoint_id": np.arange(SOURCE, dtype=np.int64),
        "time_s": np.linspace(0.0, 1.0, SOURCE),
        "tau": np.linspace(0.0, 1.0, SOURCE),
        "target_position": np.tile(pos, (SOURCE, 1)),
        "target_quaternion": np.tile(quat, (SOURCE, 1)),
        "q_reference": np.tile(q, (SOURCE, 1)),
        "position_reconstruction_error_m": np.zeros(SOURCE),
        "orientation_reconstruction_error_rad": np.zeros(SOURCE),
        "waypoint_reachable": np.ones(SOURCE, dtype=bool),
    }
    split_dir = _split_dir(paths, split)
    canon_path = split_dir / f"{trajectory_id}.npz"
    source_path = split_dir / f"{trajectory_id}_source.npz"
    save_npz(canon_path, canonical, overwrite=True)
    save_npz(source_path, source, overwrite=True)
    content_hash = content_hash_of_record({"trajectory_id": trajectory_id, "q": [round(float(v), 12) for v in q]})
    return content_hash, sha256_file(canon_path), sha256_file(source_path)


def build_fixture(root, model_context, master_seed=42, n_core=6, n_challenge=6, splits=("development", "validation", "frozen_test")):
    """Scaffold a Dataset v2 root and fabricate ``n_core`` core + ``n_challenge`` challenge static
    trajectories spread evenly across ``splits``, then build the combined catalog. Returns the list
    of (trajectory_id, family, split) tuples."""
    create_dataset_v2_scaffold(root, master_seed=master_seed)
    paths = dataset_v2_paths(root)
    rng = np.random.default_rng(master_seed)
    lower = model_context.operational_lower_rad
    upper = model_context.operational_upper_rad
    margin = 0.1 * (upper - lower)

    core_rows = []
    challenge_rows = []
    catalog_entries = []

    for i in range(n_core):
        split = splits[i % len(splits)]
        tid = f"core_line_fixed_anchor_regular_{i:02d}"
        q = rng.uniform(lower + margin, upper - margin)
        content_hash, sha, source_sha = _write_trajectory(paths, model_context, split, tid, q)
        row = {c: "" for c in CORE_COLS}
        row.update(
            {
                "trajectory_id": tid,
                "family": "core",
                "split": split,
                "shape": "line",
                "orientation_mode": "fixed",
                "anchor_id": f"anchor_regular_{i:02d}",
                "anchor_class": "regular",
                "source_seed": 111,
                "source_waypoint_count": SOURCE,
                "canonical_waypoint_count": CANON,
                "quaternion_convention": "wxyz",
                "content_hash": content_hash,
                "sha256": sha,
                "source_sha256": source_sha,
                "model_fingerprint": MODEL_FINGERPRINT,
                "config_fingerprint": "fixture",
            }
        )
        core_rows.append(row)
        catalog_entries.append((tid, "core", split))

    for i in range(n_challenge):
        split = splits[i % len(splits)]
        tid = f"challenge_{split}_{i:03d}"
        q = rng.uniform(lower + margin, upper - margin)
        content_hash, sha, source_sha = _write_trajectory(paths, model_context, split, tid, q)
        row = {c: "" for c in CHALLENGE_COLS}
        row.update(
            {
                "trajectory_id": tid,
                "family": "random_challenge",
                "challenge_family": "smooth_random",
                "split": split,
                "source_seed": 222,
                "path_seed": 3330 + i,
                "frozen_challenge_seed_revision": 1 if split == "frozen_test" else 0,
                "source_waypoint_count": SOURCE,
                "canonical_waypoint_count": CANON,
                "quaternion_convention": "wxyz",
                "content_hash": content_hash,
                "sha256": sha,
                "source_sha256": source_sha,
                "model_fingerprint": MODEL_FINGERPRINT,
                "config_fingerprint": "fixture",
            }
        )
        challenge_rows.append(row)
        catalog_entries.append((tid, "random_challenge", split))

    _write_manifest(paths.trajectories_dir / CORE_MANIFEST, CORE_COLS, core_rows)
    _write_manifest(paths.trajectories_dir / CHALLENGE_MANIFEST, CHALLENGE_COLS, challenge_rows)
    build_combined_catalog(root, overwrite=True, full_counts=False)
    return catalog_entries


def _write_manifest(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
