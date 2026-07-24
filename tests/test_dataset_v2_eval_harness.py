"""Dataset-dependent tests for the Dataset v2 evaluation harness: public/protected isolation and
tier smokes (Tier 0 gate, point-IK, warm/cold trajectory, resume). These build a small public
export from the persistent Dataset v2 working root and run tiny-limit solves.

Skipped automatically if the working root is not present (e.g. CI without the dataset). The root
may be overridden with the ``MPDIK_V2_WORK_ROOT`` environment variable.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from dataset_v2.locator import dataset_v2_paths

WORK_ROOT = Path(os.environ.get("MPDIK_V2_WORK_ROOT", r"D:\data\hoang_anh\mpdik_kassow_v2_work"))

pytestmark = pytest.mark.skipif(
    not (WORK_ROOT / "DATASET_MANIFEST.json").is_file(),
    reason=f"Dataset v2 working root not available at {WORK_ROOT}",
)


@pytest.fixture(scope="module")
def model_context():
    from kinematics.model_loader import load_model_context

    return load_model_context()


@pytest.fixture(scope="module")
def exported_roots(tmp_path_factory):
    from evaluation_v2.public_export import export_public_and_protected

    base = tmp_path_factory.mktemp("eval_export")
    public_root = base / "public"
    protected_root = base / "protected"
    export_public_and_protected(WORK_ROOT, public_root, protected_root, overwrite=True)
    return public_root, protected_root


# ---- isolation ----------------------------------------------------------------------------
def test_public_point_ik_has_no_reference_solution(exported_roots):
    from evaluation_v2.protected_guard import find_protected_fields
    from utils.npz_utils import load_npz

    public_root, _ = exported_roots
    for split in ("development", "validation"):
        arrays = load_npz(public_root / "tier1_point_ik" / f"{split}.npz")
        assert find_protected_fields(arrays.keys()) == []
        assert "q_target_reference" not in arrays


def test_public_trajectories_have_no_q_reference(exported_roots):
    from evaluation_v2.protected_guard import find_protected_fields
    from utils.npz_utils import load_npz

    public_root, _ = exported_roots
    traj_files = list((public_root / "trajectories").rglob("*.npz"))
    assert traj_files
    for f in traj_files[:5]:
        arrays = load_npz(f)
        assert find_protected_fields(arrays.keys()) == []
        assert "q_reference" not in arrays


def test_protected_root_holds_reference_evidence_outside_public(exported_roots):
    from utils.npz_utils import load_npz

    public_root, protected_root = exported_roots
    prot = load_npz(protected_root / "tier1_point_ik" / "development.npz")
    assert "q_target_reference" in prot
    # And that evidence is nowhere under the public root.
    assert not any("protected" in p.name for p in public_root.rglob("*.npz"))


def test_public_qinitial_never_equals_reference_solution(exported_roots):
    from utils.npz_utils import load_npz

    public_root, protected_root = exported_roots
    pub = load_npz(public_root / "tier1_point_ik" / "development.npz")
    prot = load_npz(protected_root / "tier1_point_ik" / "development.npz")
    # Row-aligned by sample_id order (both written from the same source order).
    assert np.array_equal(pub["sample_id"], prot["sample_id"])
    diffs = np.abs(pub["q_initial"] - prot["q_target_reference"]).max(axis=1)
    assert np.all(diffs > 1e-9)


def test_evaluator_runtime_guard_rejects_protected_npz(tmp_path, model_context):
    from evaluation_v2.candidate_configs import candidate_by_id
    from evaluation_v2.point_eval import evaluate_point_ik_split
    from utils.exceptions import ModelConfigurationError
    from utils.npz_utils import save_npz

    bad = tmp_path / "bad.npz"
    save_npz(bad, {
        "sample_id": np.array([0]), "difficulty_id": np.array([0]),
        "q_initial": np.zeros((1, 7)), "target_position": np.zeros((1, 3)),
        "target_quaternion_wxyz": np.tile([1.0, 0, 0, 0], (1, 1)),
        "q_target_reference": np.zeros((1, 7)),  # protected leak
    })
    with pytest.raises(ModelConfigurationError):
        evaluate_point_ik_split(bad, candidate_by_id("cand_A_adaptive_baseline"), model_context=model_context)


# ---- tier smokes --------------------------------------------------------------------------
def test_tier0_gate_smoke(model_context):
    from evaluation_v2.tier0_gate import run_tier0_gate

    gate = run_tier0_gate(
        WORK_ROOT, model_context=model_context, fk_limit=20, jacobian_limit=20, singularity_limit=10
    )
    assert gate.gate_pass
    assert gate.max_jacobian_relative_error <= 1e-4


def test_point_ik_smoke(exported_roots, model_context):
    from evaluation_v2.candidate_configs import candidate_by_id
    from evaluation_v2.point_eval import evaluate_point_ik_split

    public_root, _ = exported_roots
    df = evaluate_point_ik_split(
        public_root / "tier1_point_ik" / "development.npz",
        candidate_by_id("cand_A_adaptive_baseline"),
        model_context=model_context, sample_limit=3,
    )
    assert len(df) == 3
    for col in ("converged", "success_coarse", "success_standard", "success_strict", "position_error_m"):
        assert col in df.columns
    assert "q_target_reference" not in " ".join(df.columns)


def test_warm_and_cold_smoke(exported_roots, model_context):
    import csv

    from evaluation_v2.candidate_configs import candidate_by_id
    from evaluation_v2.trajectory_eval import evaluate_trajectory_trial
    from utils.npz_utils import load_npz

    public_root, _ = exported_roots
    trials = load_npz(public_root / "trials" / "development.npz")
    trial_idx = 0
    trajectory_id = str(trials["trajectory_id"][trial_idx])
    with open(public_root / "trajectories" / "public_trajectory_manifest.csv", newline="", encoding="utf-8") as h:
        manifest = {r["trajectory_id"]: r for r in csv.DictReader(h)}
    canonical = public_root / manifest[trajectory_id]["public_canonical_path"]

    common = dict(
        public_canonical_file=canonical,
        trial_id=str(trials["trial_id"][trial_idx]),
        trajectory_id=trajectory_id,
        trajectory_family=str(trials["trajectory_family"][trial_idx]),
        difficulty=str(trials["difficulty"][trial_idx]),
        split="development",
        q_initial=np.asarray(trials["q_initial"][trial_idx], dtype=np.float64),
        candidate=candidate_by_id("cand_A_adaptive_baseline"),
        model_context=model_context,
        waypoint_limit=6,
    )
    warm = evaluate_trajectory_trial(method="warm_start", **common)
    cold = evaluate_trajectory_trial(method="cold_start", **common)
    assert len(warm) == 6 and len(cold) == 6
    # Cold-start seeds every waypoint from the same q_initial (recovery flag never set).
    assert not cold["recovered_after_previous_failure"].any()
    # Both methods used the same targets.
    assert np.allclose(warm[["target_position_x"]].to_numpy(), cold[["target_position_x"]].to_numpy())


def test_resume_smoke_reuses_shards_no_duplicates(tmp_path, exported_roots, model_context):
    from evaluation_v2.orchestrator import run_evaluation

    public_root, _ = exported_roots
    out = tmp_path / "eval_out"
    kwargs = dict(
        splits=("development",), run_name="resume_smoke",
        point_sample_limit=3, trial_limit=1, waypoint_limit=6, model_context=model_context,
    )
    m1 = run_evaluation(WORK_ROOT, public_root, out, _cand(), overwrite=True, **kwargs)
    assert m1["overall_status"] == "completed"
    wp1 = (out / "resume_smoke" / "tier2_sequential_dls" / "waypoint_results.csv").read_text(encoding="utf-8")

    # Record shard mtimes, then resume: shards must not be recomputed.
    shard_dir = out / "resume_smoke" / "checkpoint" / "shards"
    before = {p.name: p.stat().st_mtime_ns for p in shard_dir.glob("*.csv")}
    m2 = run_evaluation(WORK_ROOT, public_root, out, _cand(), resume=True, **kwargs)
    after = {p.name: p.stat().st_mtime_ns for p in shard_dir.glob("*.csv")}
    assert before == after  # no shard rewritten on resume
    assert m2["overall_status"] == "completed"
    wp2 = (out / "resume_smoke" / "tier2_sequential_dls" / "waypoint_results.csv").read_text(encoding="utf-8")
    assert wp1 == wp2  # identical output, no duplicate rows


def test_evaluation_never_reads_frozen_test(exported_roots):
    # frozen_test must not appear in the public export at all.
    public_root, _ = exported_roots
    assert not (public_root / "tier1_point_ik" / "frozen_test.npz").exists()
    assert not (public_root / "trials" / "frozen_test.npz").exists()
    assert not (public_root / "trajectories" / "frozen_test").exists()


def _cand():
    from evaluation_v2.candidate_configs import candidate_by_id

    return candidate_by_id("cand_A_adaptive_baseline")
