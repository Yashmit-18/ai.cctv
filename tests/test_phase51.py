"""Phase 51 tests -- calibration infrastructure hardening & validation harness.

Phase 50 delivered the calibration tool, the five production gates, and the
real-camera validation runbooks.  Phase 51 hardens that infrastructure so that:

* data-integrity problems (roster gaps, empty pose dirs, corrupt crops,
  unsupported files, cross-employee duplicates, mixed crop sizes) are REPORTED
  and never silently dropped;
* ``REAL LABELED DATA`` requires a validated manifest (PNG existence is NO
  evidence of provenance -- SIMULATED artifacts can never become REAL);
* calibration is deterministic (golden dry-runs reproduce bit-for-bit);
* the recommender refuses (``eligible=False`` + explicit reason) on a genuine
  matrix of degenerate inputs instead of silently picking a closest point;
* synthetic artifacts cannot pass the production ``real_calibration_data`` gate;
* the feed gate detects black/frozen/low-FPS/abrupt-freeze feeds and the
  validation-runner choke point never proceeds on an invalid feed;
* model/checkpoint identity is explicit and fail-closed (no silent fallback,
  no variant mislabel).

Status labels stay strict: everything here is CODE VERIFIED on synthetic
inputs.  No real camera feed exists, so no REAL/EXECUTED claims are made.

Phone + side/back "harness" tests are code-level invariants (deterministic
fake clock), matching the Phase 50 validation runbooks.
"""

import hashlib
import json
import time as _realtime
from pathlib import Path

import numpy as np
import pytest

import src.calibration as C
import src.tracker as T
from src.reid import AppearanceExtractor
from src.tracker import MultiTracker

MODEL_X05 = Path("models/osnet_x0_5_msmt17.pth")
ORTH_X = np.array([1.0, 0.0, 0.0])
ORTH_Y = np.array([0.0, 1.0, 0.0])
ORTH_Z = np.array([0.0, 0.0, 1.0])

POSES = ("front", "left", "right", "45-degree", "partial", "back")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _crop(root: Path, emp: str, pose: str, name: str, value: int,
          size: int = 16):
    d = root / emp / pose
    d.mkdir(parents=True, exist_ok=True)
    import cv2 as cv
    cv.imwrite(str(d / name), np.full((size, size, 3), value, dtype=np.uint8))


def _manifest(files: dict, status: str = "REAL LABELED DATA",
              version: int = 1, capture: dict | None = None) -> dict:
    out = {"manifest_version": version, "dataset_status": status,
           "files": files}
    if capture is not None:
        out["capture"] = capture
    return out


def _embs(base, n: int, noise: float = 0.05, seed: int = 7) -> list:
    rng = np.random.default_rng(seed)
    return [np.array(base, dtype=np.float64)
            + rng.normal(0.0, noise, len(base)) for _ in range(n)]


def _ready_dir(root: Path, status: str = "REAL LABELED DATA"):
    _crop(root, "EMP001", "front", "a.png", 128)
    _crop(root, "EMP001", "front", "b.png", 129)
    _crop(root, "EMP002", "front", "a.png", 180)
    _crop(root, "EMP002", "front", "b.png", 181)
    files = {
        "EMP001/front/a.png": {"employee_id": "EMP001", "pose": "front"},
        "EMP001/front/b.png": {"employee_id": "EMP001", "pose": "front"},
        "EMP002/front/a.png": {"employee_id": "EMP002", "pose": "front"},
        "EMP002/front/b.png": {"employee_id": "EMP002", "pose": "front"},
    }
    (root / C.MANIFEST_NAME).write_text(
        json.dumps(_manifest(files, status=status)), encoding="utf-8")


def _frame(val: int) -> np.ndarray:
    return np.full((64, 64, 3), val, dtype=np.uint8)


class FakeTime:
    """Deterministic wall clock shared by the whole tracker module."""
    now = 10_000.0

    @classmethod
    def time(cls) -> float:
        return cls.now

    @classmethod
    def strftime(cls, fmt, tt=None):
        return _realtime.strftime(fmt, tt or _realtime.localtime(cls.now))

    @classmethod
    def localtime(cls, tt=None):
        return _realtime.localtime(tt or cls.now)


@pytest.fixture
def fake_clock(monkeypatch):
    FakeTime.now = 10_000.0
    monkeypatch.setattr(T, "time", FakeTime)
    return FakeTime


@pytest.fixture
def recorder(monkeypatch):
    seen = []

    def fake_log(conn, ts, emp, state, dur, source=""):
        seen.append((ts, emp, state, dur, source))

    monkeypatch.setattr(T, "log_interval", fake_log)
    return seen


@pytest.fixture
def instant_commit(monkeypatch):
    monkeypatch.setattr(T, "SMOOTHING_BUFFER_SEC", 0.0)
    monkeypatch.setattr(T, "IDENTITY_STABILITY_FRAMES", 1)
    monkeypatch.setattr(T, "AWAY_AFTER_SEC", 3.0)
    monkeypatch.setattr(T, "PHONE_AFTER_SEC", 5.0)
    monkeypatch.setattr(T, "PHONE_GAP_GRACE_SEC", 2.0)


def _present() -> dict:
    return {"present": True, "phone_detected": False}


def _phone() -> dict:
    return {"present": True, "phone_detected": True}


def _absent() -> dict:
    return {"present": False, "phone_detected": False}


# ---------------------------------------------------------------------------
# A. Data integrity -- roster, poses, corrupt, unsupported, duplicates, sizes
# ---------------------------------------------------------------------------

def test_scan_reports_missing_and_empty_poses(tmp_path):
    root = tmp_path / "cal"
    _crop(root, "EMP001", "front", "a.png", 60)
    (root / "EMP002" / "back").mkdir(parents=True)
    survey = C.scan_data_dir(root, ("EMP001", "EMP002"))
    assert survey["missing_poses"]["EMP001"] == [
        "left", "right", "45-degree", "partial", "back"]
    assert "back" in survey["empty_poses"]["EMP002"]
    assert survey["employees_missing"] == ["EMP002"]  # no usable samples


def test_scan_reports_corrupt_crop_not_silently_dropped(tmp_path):
    root = tmp_path / "cal"
    d = root / "EMP001" / "front"
    d.mkdir(parents=True)
    _crop(root, "EMP001", "front", "a.png", 128)
    (d / "broken.png").write_bytes(b"this is not a png at all")
    survey = C.scan_data_dir(root)
    assert survey["files_discovered_raw"] == 2
    assert any(f["path"].endswith("broken.png")
               for f in survey["unreadable_files"])
    assert len(survey["samples"]["EMP001"]["front"]) == 1
    assert "all_files" in survey and len(survey["all_files"]) == 1


def test_scan_reports_unsupported_file(tmp_path):
    root = tmp_path / "cal"
    _crop(root, "EMP001", "front", "a.png", 128)
    (root / "EMP001" / "front" / "notes.txt").write_text("ignored", encoding="utf-8")
    survey = C.scan_data_dir(root)
    assert any(f["path"].endswith("notes.txt")
               for f in survey["unsupported_files"])


def test_scan_dedup_rejects_cross_pose_duplicate(tmp_path):
    root = tmp_path / "cal"
    _crop(root, "EMP001", "front", "a.png", 128)
    _crop(root, "EMP001", "back", "a.png", 128)
    survey = C.scan_data_dir(root)
    assert len([f for f in survey["duplicates"]
                if f["duplicate_of"].endswith("front/a.png")]) == 1
    assert "back" not in survey["samples"]["EMP001"]


def test_scan_dedup_rejects_cross_employee_duplicate(tmp_path):
    root = tmp_path / "cal"
    _crop(root, "EMP001", "front", "a.png", 128)
    _crop(root, "EMP002", "front", "a.png", 128)
    survey = C.scan_data_dir(root)
    assert len(survey["duplicates"]) == 1
    assert "EMP002" not in survey["samples"]  # only the duplicate existed


def test_scan_records_image_sizes(tmp_path):
    root = tmp_path / "cal"
    _crop(root, "EMP001", "front", "a.png", 128, size=16)
    _crop(root, "EMP001", "front", "b.png", 129, size=32)
    survey = C.scan_data_dir(root)
    assert survey["image_sizes"] == {"16x16": 1, "32x32": 1}


def test_no_file_silently_disappears(tmp_path):
    root = tmp_path / "cal"
    d = root / "EMP001" / "front"
    d.mkdir(parents=True)
    _crop(root, "EMP001", "front", "a.png", 128)
    (d / "a.png",)
    (d / "b.png").write_bytes(b"corrupt")
    (d / "c.png").write_bytes(b"corrupt again")
    survey = C.scan_data_dir(root)
    accounted = (len(survey["all_files"]) + len(survey["duplicates"])
                 + len(survey["unreadable_files"]))
    assert survey["files_discovered_raw"] == accounted == 3
    assert accounted == len(survey["samples"]["EMP001"]["front"]) \
        + len(survey["duplicates"]) + len(survey["unreadable_files"])


def test_build_report_counts_corrupt_crops_as_failed(tmp_path):
    root = tmp_path / "cal"
    (root / "EMP001" / "front").mkdir(parents=True)
    _crop(root, "EMP001", "front", "a.png", 128)
    (root / "EMP001" / "front" / "b.png").write_bytes(b"corrupt")
    (root / "EMP001" / "front" / "c.png").write_bytes(b"corrupt")
    survey = C.scan_data_dir(root)
    rep = C.build_report(extractor=AppearanceExtractor(model_path=""),
                         data_dir=root, survey=survey)
    assert rep.samples_total == 1 and rep.samples_embedded == 1
    assert len(rep.failed_files) == 0          # corrupts removed at scan stage
    assert len(survey["unreadable_files"]) == 2


def test_dataset_fingerprint_stable_and_content_aware(tmp_path):
    root = tmp_path / "cal"
    _crop(root, "EMP001", "front", "a.png", 128)
    f1 = C.dataset_fingerprint(root)
    f2 = C.dataset_fingerprint(root)
    assert f1 == f2
    assert len(f1) == 64
    _crop(root, "EMP001", "front", "b.png", 200)
    assert C.dataset_fingerprint(root) != f1


# ---------------------------------------------------------------------------
# B. Manifest -- provenance is REQUIRED for REAL, never inferred from files
# ---------------------------------------------------------------------------

def test_manifest_missing_returns_none(tmp_path):
    assert C.load_manifest(tmp_path / "nope") is None


def test_manifest_corrupt_json_returns_none(tmp_path):
    (tmp_path / C.MANIFEST_NAME).write_text("not json", encoding="utf-8")
    assert C.load_manifest(tmp_path) is None


def test_manifest_wrong_version_returns_none(tmp_path):
    (tmp_path / C.MANIFEST_NAME).write_text(
        json.dumps(_manifest({}, version=999)), encoding="utf-8")
    assert C.load_manifest(tmp_path) is None


def test_validate_manifest_ok_with_full_coverage(tmp_path):
    root = tmp_path / "cal"
    _ready_dir(root)
    samples = C.scan_data_dir(root)["samples"]
    man = C.validate_manifest(C.load_manifest(root), samples)
    assert man["ok"] is True
    assert man["status"] == "REAL LABELED DATA"
    assert man["missing"] == [] and man["mismatches"] == []


def test_validate_manifest_fails_when_file_missing(tmp_path):
    root = tmp_path / "cal"
    _crop(root, "EMP001", "front", "a.png", 128)
    (root / C.MANIFEST_NAME).write_text(
        json.dumps(_manifest({})), encoding="utf-8")
    samples = C.scan_data_dir(root)["samples"]
    man = C.validate_manifest(C.load_manifest(root), samples)
    assert man["ok"] is False
    assert man["missing"] == ["EMP001/front/a.png"]


def test_validate_manifest_fails_on_employee_mismatch(tmp_path):
    root = tmp_path / "cal"
    _crop(root, "EMP001", "front", "a.png", 128)
    files = {"EMP001/front/a.png":
             {"employee_id": "EMP999", "pose": "front"}}
    (root / C.MANIFEST_NAME).write_text(
        json.dumps(_manifest(files)), encoding="utf-8")
    samples = C.scan_data_dir(root)["samples"]
    man = C.validate_manifest(C.load_manifest(root), samples)
    assert man["ok"] is False
    assert "EMP001/front/a.png" in man["mismatches"]


def test_validate_manifest_reports_unreferenced_records(tmp_path):
    root = tmp_path / "cal"
    _crop(root, "EMP001", "front", "a.png", 128)
    files = {"EMP001/front/a.png":
             {"employee_id": "EMP001", "pose": "front"},
             "EMP002/front/ghost.png": {"employee_id": "EMP002", "pose": "front"}}
    (root / C.MANIFEST_NAME).write_text(
        json.dumps(_manifest(files)), encoding="utf-8")
    samples = C.scan_data_dir(root)["samples"]
    man = C.validate_manifest(C.load_manifest(root), samples)
    assert man["records_unreferenced"] == 1


def test_simulated_manifest_never_real(tmp_path):
    root = tmp_path / "cal"
    _ready_dir(root, status="SIMULATED")
    samples = C.scan_data_dir(root)["samples"]
    man = C.validate_manifest(C.load_manifest(root), samples)
    assert man["ok"] is False
    assert man["status"] == "SIMULATED"


def test_manifest_capture_metadata_survives_validation(tmp_path):
    root = tmp_path / "cal"
    _ready_dir(root)
    (root / C.MANIFEST_NAME).write_text(
        json.dumps(_manifest({"EMP001/front/a.png": {"employee_id": "EMP001",
                                                     "pose": "front"},
                              "EMP001/front/b.png": {"employee_id": "EMP001",
                                                     "pose": "front"},
                              "EMP002/front/a.png": {"employee_id": "EMP002",
                                                     "pose": "front"},
                              "EMP002/front/b.png": {"employee_id": "EMP002",
                                                     "pose": "front"}},
                             capture={"camera": 0, "fps": 30,
                                      "geometry": "front"})),
        encoding="utf-8")
    samples = C.scan_data_dir(root)["samples"]
    man = C.validate_manifest(C.load_manifest(root), samples)
    assert man["capture"]["camera"] == 0


# ---------------------------------------------------------------------------
# C. Determinism
# ---------------------------------------------------------------------------

def test_dry_run_deterministic_identical(tmp_path):
    r1 = C.run_calibration(dry_run_synthetic=True,
                           out_file=str(tmp_path / "r1.json"))
    r2 = C.run_calibration(dry_run_synthetic=True,
                           out_file=str(tmp_path / "r2.json"))
    assert r1.dataset_status == "SIMULATED"
    assert r1.label.startswith("SIMULATED")
    assert r1.recommended == r2.recommended
    assert r1.same.p50 == r2.same.p50
    assert r1.diff.p50 == r2.diff.p50
    assert r1.margins.p50 == r2.margins.p50
    assert r1.samples_embedded == r2.samples_embedded


def test_operating_point_matrix_deterministic():
    by_emp = {"EMP001": _embs(ORTH_X, 4, seed=11),
              "EMP002": _embs(ORTH_Y, 4, seed=21)}
    a = C.scan_operating_points(by_emp)
    b = C.scan_operating_points(by_emp)
    assert [(p.threshold, p.margin, p.far, p.frr) for p in a] == \
        [(p.threshold, p.margin, p.far, p.frr) for p in b]


# ---------------------------------------------------------------------------
# D. Recommender fixture matrix (A..M) -- refuse, never silently pick
# ---------------------------------------------------------------------------

def test_fixture_a_no_embeddings_refuses():
    rec = C.recommend({})
    assert rec["eligible"] is False and rec["threshold"] is None
    assert rec["reason"]


def test_fixture_b_single_employee_refuses():
    rec = C.recommend({"EMP001": _embs(ORTH_X, 8, seed=3)})
    assert rec["eligible"] is False
    assert "employee" in rec["reason"].lower()


def test_fixture_c_well_separated_eligible():
    by_emp = {"EMP001": _embs(ORTH_X, 4, noise=0.05, seed=11),
              "EMP002": _embs(ORTH_Y, 4, noise=0.05, seed=21)}
    rec = C.recommend(by_emp)
    assert rec["eligible"] is True
    assert rec["threshold"] is not None and rec["margin"] is not None


def test_fixture_d_confusable_dead_model_refuses():
    by_emp = {"EMP001": _embs(ORTH_X, 4, noise=0.08, seed=5),
              "EMP002": _embs(ORTH_X, 4, noise=0.08, seed=5)}
    rec = C.recommend(by_emp)
    assert rec["eligible"] is False
    assert rec["threshold"] is None


def test_fixture_e_overlapping_refuses_or_any_safe_point():
    by_emp = {"EMP001": _embs(ORTH_X, 4, noise=0.08, seed=5),
              "EMP002": _embs(ORTH_X + 0.02 * ORTH_Y, 4,
                              noise=0.08, seed=5)}
    rec = C.recommend(by_emp, frr_ceiling=0.30)
    assert rec["eligible"] is False


def test_fixture_f_zero_norm_vectors_refused_without_crash():
    zeros = [np.zeros(3) for _ in range(4)]
    by_emp = {"EMP001": list(zeros), "EMP002": list(zeros)}
    rec = C.recommend(by_emp)
    assert rec["eligible"] is False
    assert rec["threshold"] is None


def test_fixture_g_sample_shortage_refuses():
    by_emp = {"EMP001": _embs(ORTH_X, 4, seed=1),
              "EMP002": _embs(ORTH_Y, 3, seed=2)}
    rec = C.recommend(by_emp)
    assert rec["eligible"] is False
    assert "insufficient samples" in rec["reason"]


def test_fixture_h_refusal_always_carries_explicit_reason():
    cases = [
        {},
        {"EMP001": _embs(ORTH_X, 8, seed=1)},
        {"EMP001": _embs(ORTH_X, 2, seed=1),
         "EMP002": _embs(ORTH_Y, 3, seed=2)},
        {"EMP001": _embs(ORTH_X, 4, noise=0.15, seed=4),
         "EMP002": _embs(ORTH_X, 4, noise=0.15, seed=4)},
    ]
    for by_emp in cases:
        rec = C.recommend(by_emp)
        assert rec["eligible"] is False
        assert isinstance(rec["reason"], str) and rec["reason"]
        assert rec["threshold"] is None and rec["margin"] is None


def test_fixture_i_empty_employee_value_ignored():
    by_emp = {"EMP001": [], "EMP002": _embs(ORTH_Y, 8, seed=2)}
    rec = C.recommend(by_emp)
    assert rec["eligible"] is False          # effectively one employee


def test_fixture_j_tight_frr_ceiling_still_refuses_degenerate():
    by_emp = {"EMP001": _embs(ORTH_X, 4, noise=0.08, seed=5),
              "EMP002": _embs(ORTH_X, 4, noise=0.08, seed=5)}
    rec = C.recommend(by_emp, frr_ceiling=0.20, safety_far_max=0.01)
    assert rec["eligible"] is False
    assert rec["threshold"] is None
    assert "FAR" in rec["reason"] and "FRR" in rec["reason"]


def test_fixture_k_zero_far_tolerance_refuses_confusable():
    by_emp = {"EMP001": _embs(ORTH_X + 0.03 * ORTH_Y, 4,
                              noise=0.08, seed=6),
              "EMP002": _embs(ORTH_X, 4, noise=0.08, seed=6)}
    rec = C.recommend(by_emp, safety_far_max=0.0)
    assert rec["eligible"] is False


def test_fixture_l_scaling_does_not_change_decisions():
    by_emp_a = {"EMP001": _embs(ORTH_X, 4, seed=13),
                "EMP002": _embs(ORTH_Y, 4, seed=23)}
    by_emp_b = {"EMP001": [3.0 * v for v in by_emp_a["EMP001"]],
                "EMP002": [3.0 * v for v in by_emp_a["EMP002"]]}
    assert C.recommend(by_emp_a) == C.recommend(by_emp_b)


def test_fixture_m_three_employees_eligible_sane_point():
    by_emp = {"EMP001": _embs(ORTH_X, 4, seed=31),
              "EMP002": _embs(ORTH_Y, 4, seed=32),
              "EMP003": _embs(ORTH_Z, 4, seed=33)}
    rec = C.recommend(by_emp)
    assert rec["eligible"] is True
    assert rec["threshold"] > 0.0
    assert rec["margin"] is not None and rec["margin"] >= 0.0


# ---------------------------------------------------------------------------
# E. Production gate -- synthetic can never pass real-data gate
# ---------------------------------------------------------------------------

def test_gate_all_true_passes_without_report():
    g = C.production_gate(real_calibration_data=True,
                          thresholds_calibrated=True,
                          margin_calibrated=True,
                          identity_safety_tests=True,
                          real_camera_validation=True)
    assert g.all_gates is True and g.reasons == []
    assert all(g.gates.values())


def test_gate_default_all_false_lists_every_reason():
    g = C.production_gate()
    assert g.all_gates is False
    assert g.reasons == ["real_calibration_data", "thresholds_calibrated",
                         "margin_calibrated", "identity_safety_tests",
                         "real_camera_validation"]


@pytest.mark.parametrize("missing", [
    "real_calibration_data", "thresholds_calibrated",
    "margin_calibrated", "identity_safety_tests", "real_camera_validation"])
def test_gate_any_missing_fails(missing):
    kwargs = {"real_calibration_data": True, "thresholds_calibrated": True,
              "margin_calibrated": True, "identity_safety_tests": True,
              "real_camera_validation": True}
    kwargs[missing] = False
    g = C.production_gate(**kwargs)
    assert g.all_gates is False
    assert missing in g.reasons


def test_synthetic_artifact_cannot_pass_real_data_gate(tmp_path):
    rep = C.run_calibration(dry_run_synthetic=True,
                            out_file=str(tmp_path / "r.json"))
    assert rep.dataset_status == "SIMULATED"
    g = C.production_gate(real_calibration_data=True,
                          thresholds_calibrated=True,
                          margin_calibrated=True,
                          identity_safety_tests=True,
                          real_camera_validation=True,
                          report=rep)
    assert g.all_gates is False
    assert "real_calibration_data" in g.reasons


def test_gate_real_report_allows_pass(tmp_path):
    root = tmp_path / "cal"
    _ready_dir(root)
    rep = C.build_report(extractor=AppearanceExtractor(model_path=""),
                         data_dir=root)
    assert rep.dataset_status == "REAL LABELED DATA"
    g = C.production_gate(real_calibration_data=True,
                          thresholds_calibrated=True,
                          margin_calibrated=True,
                          identity_safety_tests=True,
                          real_camera_validation=True,
                          report=rep)
    assert g.all_gates is True


def test_gate_camera_validation_still_required_with_real_report(tmp_path):
    root = tmp_path / "cal"
    _ready_dir(root)
    rep = C.build_report(extractor=AppearanceExtractor(model_path=""),
                         data_dir=root)
    g = C.production_gate(real_calibration_data=True,
                          thresholds_calibrated=True,
                          margin_calibrated=True,
                          identity_safety_tests=True,
                          real_camera_validation=False,
                          report=rep)
    assert g.all_gates is False
    assert "real_camera_validation" in g.reasons


# ---------------------------------------------------------------------------
# F. Feed gate -- 9 scenarios + validation-runner choke point
# ---------------------------------------------------------------------------

def test_feed_gate_accepts_bright_changing_fps():
    frames = [_frame(160 + i % 2) for i in range(10)]
    stats = C.feed_usable_stats(frames, capture_fps=22.0, require_fps=15.0)
    assert stats["usable"] is True
    assert stats["bright"] is True and stats["changing"] is True


def test_feed_gate_black_frame_rejected():
    frames = [_frame(i % 2) for i in range(10)]
    stats = C.feed_usable_stats(frames, capture_fps=22.0, require_fps=15.0)
    assert stats["usable"] is False
    assert stats["bright"] is False


def test_feed_gate_frozen_frame_rejected():
    frames = [_frame(140) for _ in range(10)]
    stats = C.feed_usable_stats(frames, capture_fps=22.0, require_fps=15.0)
    assert stats["usable"] is False
    assert stats["changing"] is False


def test_feed_gate_abrupt_freeze_tail_rejected():
    frames = [_frame(120 + i) for i in range(5)] + [_frame(125) for _ in range(8)]
    stats = C.feed_usable_stats(frames, capture_fps=22.0, require_fps=15.0)
    assert stats["usable"] is False
    assert stats["changed_ratio"] < 0.5


def test_feed_gate_low_fps_rejected():
    frames = [_frame(160 + i % 2) for i in range(10)]
    stats = C.feed_usable_stats(frames, capture_fps=5.0, require_fps=15.0)
    assert stats["usable"] is False
    assert stats["fps_ok"] is False


def test_feed_gate_insufficient_frames_rejected():
    frames = [_frame(100), _frame(200)]
    stats = C.feed_usable_stats(frames, capture_fps=22.0, require_fps=15.0,
                                min_frames=3)
    assert stats["usable"] is False
    assert stats["insufficient_frames"] is True


def test_feed_gate_empty_list_rejected():
    stats = C.feed_usable_stats([], capture_fps=22.0)
    assert stats["usable"] is False
    assert stats["insufficient_frames"] is True


def test_feed_gate_single_frame_rejected():
    stats = C.feed_usable_stats([_frame(140)], capture_fps=22.0)
    assert stats["usable"] is False


def test_feed_gate_majority_ratio_boundary_accepted():
    frames = [_frame(120), _frame(121), _frame(122), _frame(122), _frame(122)]
    stats = C.feed_usable_stats(frames, capture_fps=22.0, require_fps=15.0)
    assert stats["changed_ratio"] == 0.5
    assert stats["changing"] is True
    assert stats["usable"] is True


def test_runner_verdict_blocks_black_feed():
    frames = [_frame(2) for _ in range(6)]
    v = C.validation_runner_verdict(frames, capture_fps=22.0)
    assert v["proceed"] is False
    assert v["state"] == "NOT EXECUTED / INVALID FEED"
    assert "BLACK" in v["reason"]


def test_runner_verdict_blocks_frozen_feed():
    frames = [_frame(140) for _ in range(6)]
    v = C.validation_runner_verdict(frames, capture_fps=22.0)
    assert v["proceed"] is False
    assert "FROZEN" in v["reason"]


def test_runner_verdict_allows_bright_changing():
    frames = [_frame(160 + i % 2) for i in range(10)]
    v = C.validation_runner_verdict(frames, capture_fps=22.0)
    assert v["proceed"] is True
    assert v["state"] == "EXECUTABLE (FEED OK)"


# ---------------------------------------------------------------------------
# G. Model / checkpoint integrity -- explicit, fail-closed
# ---------------------------------------------------------------------------

def test_checkpoint_identity_none_for_missing_file():
    assert C.checkpoint_identity(str(Path("models/nope.pth"))) is None


def test_checkpoint_identity_real_osnet_x05():
    if not MODEL_X05.is_file():
        pytest.skip("osnet_x0_5 checkpoint not present")
    ident = C.checkpoint_identity(str(MODEL_X05))
    assert ident["checkpoint"] == "osnet_x0_5_msmt17.pth"
    assert ident["family"] == "osnet"
    assert ident["bytes"] > 0
    assert len(ident["sha256"]) == 64


def test_fail_closed_missing_checkpoint_raises():
    ex = AppearanceExtractor(model_path="")
    with pytest.raises(FileNotFoundError):
        C._ensure_model_fail_closed("models/does_not_exist.pth", ex)


def test_fail_closed_wrong_model_raises():
    ex = AppearanceExtractor(model_path="")
    assert ex.model_source == "built-in"
    with pytest.raises(RuntimeError):
        C._ensure_model_fail_closed(str(MODEL_X05), ex)


def test_fail_closed_empty_model_path_allows_builtin():
    ex = AppearanceExtractor(model_path="")
    C._ensure_model_fail_closed("", ex)      # production default unchanged


@pytest.mark.skipif(not MODEL_X05.is_file(),
                    reason="osnet_x0_5 checkpoint not present")
def test_extractor_records_checkpoint_variant():
    ex = AppearanceExtractor(model_path=str(MODEL_X05))
    assert ex.model_source == "osnet"
    assert ex.checkpoint == "osnet_x0_5_msmt17.pth"
    assert ex.model_variant == "osnet_x0_5"
    assert ex.feature_dim == 512


# ---------------------------------------------------------------------------
# H. Phone harness -- 12 code invariants (deterministic fake clock)
# ---------------------------------------------------------------------------

def test_phone_wall_clock_not_frame_count(fake_clock, recorder,
                                          instant_commit):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1
    for _ in range(6):                       # only 6 frames, 1s apart
        t.process(_phone())
        FakeTime.now += 1
    assert t.committed_state == "ON_PHONE"   # 5 real wall-clock seconds


def test_phone_under_threshold_not_committed(fake_clock, instant_commit):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    FakeTime.now += 3.9
    t.process(_phone())
    assert t.committed_state == "ACTIVE"
    assert t.current_state == "ACTIVE"


def test_phone_state_not_flashed_before_threshold(fake_clock, instant_commit):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    FakeTime.now += 2.0
    t.process(_phone())
    assert t.committed_state == "ACTIVE"
    assert t.current_state == "ACTIVE"       # dashboard never flashes ON_PHONE


def test_phone_episode_gap_under_grace_kept(fake_clock, recorder,
                                            instant_commit):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    FakeTime.now += 1.0
    t.process(_present())                    # brief no-phone frame
    FakeTime.now += 0.9
    t.process(_phone())
    FakeTime.now += 3.2
    t.process(_phone())                      # cumulative >5s
    assert t.committed_state == "ON_PHONE"


def test_phone_episode_gap_over_grace_restarts(fake_clock, recorder,
                                               instant_commit):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    FakeTime.now += 3.0
    t.process(_present())                    # gap > PHONE_GAP_GRACE_SEC
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    FakeTime.now += 5.5
    t.process(_phone())                      # fresh episode, crosses again
    assert t.committed_state == "ON_PHONE"


def test_phone_exactly_one_alert_per_episode(tmp_db, fake_clock,
                                             instant_commit):
    from src.database import query_security_events
    t = T.EmployeeTracker(tmp_db, "EMP001", source="cam1")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    FakeTime.now += 5.0
    t.process(_phone())
    assert len(query_security_events(tmp_db, event_type="PHONE_USE")) == 1


def test_phone_alert_does_not_repeat_while_sustained(tmp_db, fake_clock,
                                                     instant_commit):
    from src.database import query_security_events
    t = T.EmployeeTracker(tmp_db, "EMP001", source="cam1")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    FakeTime.now += 5.0
    t.process(_phone())
    for _ in range(20):
        t.process(_phone())
        FakeTime.now += 1
    assert len(query_security_events(tmp_db, event_type="PHONE_USE")) == 1


def test_phone_absence_closes_run_to_away(fake_clock, recorder,
                                          instant_commit):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    FakeTime.now += 5.0
    t.process(_phone())
    assert t.committed_state == "ON_PHONE"
    FakeTime.now += 3.1                       # past AWAY_AFTER_SEC grace
    t.process(_absent())
    assert t.committed_state == "AWAY"
    assert t._phone_since is None


def test_phone_offline_freezes_state(fake_clock, recorder, instant_commit):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())                          # episode opens at 10001
    FakeTime.now += 20
    t.process(_present(), camera_online=False)   # long dead-camera gap
    assert t.committed_state == "ACTIVE"         # wall time advanced nothing
    assert t.current_state == "ACTIVE"
    assert t._phone_since == pytest.approx(10_001.0)  # run kept, not started


def test_phone_state_independent_per_employee(fake_clock, instant_commit):
    t = MultiTracker(None)
    t.process_batch([
        {"cam": "c1", "emp_id": "EMP001", "phone": False},
        {"cam": "c1", "emp_id": "EMP002", "phone": True},
    ], True)
    FakeTime.now += 6
    t.process_batch([
        {"cam": "c1", "emp_id": "EMP001", "phone": False},
        {"cam": "c1", "emp_id": "EMP002", "phone": True},
    ], True)
    states = t.live_states()
    assert states.get("EMP001") == "ACTIVE"
    assert states.get("EMP002") == "ON_PHONE"


def test_phone_attributed_to_owning_employee_only(tmp_db, fake_clock,
                                                  instant_commit):
    from src.database import query_security_events
    t = MultiTracker(tmp_db)
    # Runbook Scenario A: identity established first, then phone use starts.
    t.process_batch([{"cam": "c1", "emp_id": "EMP001", "phone": False}], True)
    FakeTime.now += 1
    for _ in range(6):
        t.process_batch([
            {"cam": "c1", "emp_id": "EMP001", "phone": True},
            {"cam": "c1", "emp_id": "Unknown", "phone": True},
        ], True)
        FakeTime.now += 1
    events = query_security_events(tmp_db, event_type="PHONE_USE")
    assert len(events) == 1
    assert events[0]["employee_id"] == "EMP001"


def test_phone_duration_is_wall_clock_float(fake_clock, instant_commit):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    assert isinstance(t._phone_since, float)
    FakeTime.now += 3.0
    t.process(_phone())
    assert t._phone_last == pytest.approx(FakeTime.now)