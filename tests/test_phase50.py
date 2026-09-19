"""Phase 50 regression tests: calibration tooling + production safety.

Covers the offline threshold-margin calibration utility (never mutates
config), the OSNet production gate, cadence/feed preconditions, and the
identity-safety invariants of the existing tracker hierarchy (face authority,
trusted-identity protection, no single-frame adoptions, no oscillation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src import calibration as C
from src.calibration import (CalibrationReport, Distribution, GateResult,
                             build_report, cadence_recommendation,
                             collect_employee_embeddings,
                             feed_usable_stats, pairwise_similarities,
                             production_gate, recommend,
                             recommended_config, run_calibration,
                             scan_operating_points, scan_data_dir,
                             embed_samples, synthetic_employees)
from src.domain import UNKNOWN_ID
from src.tracker import Track


def _synthetic_embs(employees, seed=1, dim=64) -> dict:
    """Deterministic well-separated per-employee clusters: each employee gets
    a distinct base vector plus small jitter (models appearance similarity)."""
    rng = np.random.default_rng(seed)
    out = {}
    for e in employees:
        base = rng.normal(0, 1, dim).astype(np.float32)
        jitter = rng.normal(0, 0.08, (4, dim)).astype(np.float32)
        out[e] = [base + j for j in jitter]
    return out


def _confusable_embs(employees, rng_seed=3, dim=16) -> dict:
    """Two employees drawn from the SAME pair of embeddings: genuinely
    indistinguishable at feature level (own-gal avg == cross-gal avg)."""
    rng = np.random.default_rng(rng_seed)
    v1 = rng.normal(0, 1, dim).astype(np.float32)
    v2 = rng.normal(0, 1, dim).astype(np.float32)
    return {e: [v1 if i % 2 == 0 else v2 for i in range(8)]
            for e in employees}


# ---------------------------------------------------------------------------
# Data discovery + embedding
# ---------------------------------------------------------------------------

def test_scan_data_dir_missing_is_empty(tmp_path):
    survey = scan_data_dir(tmp_path / "nope")
    assert survey["samples"] == {}
    assert survey["employees_seen"] == []


def test_scan_data_dir_discovers_posed_crops(tmp_path):
    import cv2 as cv
    root = tmp_path / "calibration"
    (root / "EMP001" / "front").mkdir(parents=True)
    (root / "EMP001" / "back").mkdir(parents=True)
    (root / "EMP002" / "front").mkdir(parents=True)
    cv.imwrite(str(root / "EMP001" / "front" / "a.png"),
               np.full((16, 16, 3), 128, dtype=np.uint8))
    cv.imwrite(str(root / "EMP001" / "front" / "b.png"),
               np.full((16, 16, 3), 129, dtype=np.uint8))
    cv.imwrite(str(root / "EMP001" / "back" / "c.png"),
               np.full((16, 16, 3), 130, dtype=np.uint8))
    cv.imwrite(str(root / "EMP002" / "front" / "d.png"),
               np.full((16, 16, 3), 131, dtype=np.uint8))
    survey = scan_data_dir(root)
    assert survey["employees_seen"] == ["EMP001", "EMP002"]
    assert len(survey["samples"]["EMP001"]["front"]) == 2
    assert len(survey["samples"]["EMP001"]["back"]) == 1


def test_embed_samples_reports_missing_poses(tmp_path):
    import cv2 as cv
    from src.reid import AppearanceExtractor
    root = tmp_path / "cal"
    (root / "EMP001" / "front").mkdir(parents=True)
    cv.imwrite(str(root / "EMP001" / "front" / "a.png"),
               np.full((16, 16, 3), 140, dtype=np.uint8))
    samples = scan_data_dir(root)["samples"]
    per_emp, stats = embed_samples(samples, AppearanceExtractor(model_path=""))
    assert stats["samples_embedded"] == 1
    assert stats["missing_poses"]["EMP001"] == [
        "left", "right", "45-degree", "partial", "back"]


def test_collect_employee_embeddings_flattens():
    per = {"EMP001": {"front": [np.ones(4)], "back": [np.zeros(4)]}}
    out = collect_employee_embeddings(per)
    assert len(out["EMP001"]) == 2


# ---------------------------------------------------------------------------
# Similarity / margin math
# ---------------------------------------------------------------------------

def test_pairwise_similarities_separated():
    one_hot = {"EMP001": [np.array([1., 0., 0.])],
               "EMP002": [np.array([0., 1., 0.])]}
    same, diff = pairwise_similarities(one_hot)
    assert same.size == 0          # single sample per employee
    assert diff.size == 1
    assert diff[0] == pytest.approx(0.0, abs=1e-6)


def test_best_vs_second_margins_sensible():
    by = {"EMP001": [np.array([1., 0., 0.]), np.array([0.9, 0.1, 0.])],
          "EMP002": [np.array([0., 1., 0.])]}
    m = C.best_vs_second_margins(by)
    assert m.size == 3
    assert np.all(m >= 0.0)


# ---------------------------------------------------------------------------
# Operating points + recommendation
# ---------------------------------------------------------------------------

def test_scan_operating_points_grid_and_order():
    by = _synthetic_embs(["EMP001", "EMP002"])
    pts = scan_operating_points(by)
    assert len(pts) == 27 * 6
    first, second = pts[0], pts[1]
    assert first.threshold == pytest.approx(0.30)
    assert second.margin == pytest.approx(0.02)


def test_recommend_meets_far_cap_on_separated_embs():
    by = _synthetic_embs(["EMP001", "EMP002", "EMP004"])
    rec = recommend(by, safety_far_max=0.01)
    assert rec["eligible"] is True
    assert rec["far_at_recommendation"] <= 0.01
    assert rec["threshold"] is not None and rec["margin"] is not None


def test_recommend_declares_ineligible_on_confusable_embs():
    by = _confusable_embs(["EMP001", "EMP002", "EMP004"])
    rec = recommend(by, safety_far_max=0.01)
    assert rec["eligible"] is False
    assert rec["threshold"] is None


# ---------------------------------------------------------------------------
# Report + missing employees + no config mutation
# ---------------------------------------------------------------------------

def test_build_report_no_data_marks_not_validated(tmp_path):
    from src.reid import AppearanceExtractor
    rep = build_report(extractor=AppearanceExtractor(model_path=""),
                       data_dir=tmp_path / "empty")
    assert rep.label.startswith("NOT VALIDATED")
    assert rep.same is None and rep.diff is None
    assert rep.employees_missing == ["EMP001", "EMP002", "EMP003", "EMP004"]


def test_build_report_reports_missing_employee(tmp_path):
    import cv2 as cv
    from src.reid import AppearanceExtractor
    root = tmp_path / "cal"
    (root / "EMP001" / "front").mkdir(parents=True)
    (root / "EMP002" / "front").mkdir(parents=True)
    cv.imwrite(str(root / "EMP001" / "front" / "a.png"),
               np.full((16, 16, 3), 100, dtype=np.uint8))
    cv.imwrite(str(root / "EMP002" / "front" / "a.png"),
               np.full((16, 16, 3), 101, dtype=np.uint8))
    rep = build_report(extractor=AppearanceExtractor(model_path=""),
                       data_dir=root,
                       expected_employees=("EMP001", "EMP002", "EMP003",
                                           "EMP004"))
    assert rep.employees_seen == ["EMP001", "EMP002"]
    assert rep.employees_missing == ["EMP003", "EMP004"]


def test_build_report_real_labeled_data_is_code_verified(tmp_path):
    import cv2 as cv
    import json
    from src.reid import AppearanceExtractor
    root = tmp_path / "cal"
    (root / "EMP001" / "front").mkdir(parents=True)
    (root / "EMP002" / "front").mkdir(parents=True)
    cv.imwrite(str(root / "EMP001" / "front" / "a.png"),
               np.full((16, 16, 3), 90, dtype=np.uint8))
    cv.imwrite(str(root / "EMP002" / "front" / "a.png"),
               np.full((16, 16, 3), 190, dtype=np.uint8))
    manifest = {
        "manifest_version": 1,
        "dataset_status": "REAL LABELED DATA",
        "capture": {"camera": "test-cam", "date": "2026-09-19",
                    "operator": "test"},
        "files": {
            "EMP001/front/a.png": {"employee_id": "EMP001", "pose": "front"},
            "EMP002/front/a.png": {"employee_id": "EMP002", "pose": "front"},
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    rep = build_report(extractor=AppearanceExtractor(model_path=""),
                       data_dir=root)
    assert rep.dataset_status == "REAL LABELED DATA"
    assert rep.manifest_validation["ok"] is True
    assert rep.label == "REAL LABELED DATA -- CODE VERIFIED"
    assert rep.samples_embedded == 2
    assert rep.same is not None and rep.diff is not None
    assert rep.recommended is not None
    assert rep.config_unchanged is True


def test_build_report_without_manifest_is_provenance_unproven(tmp_path):
    import cv2 as cv
    from src.reid import AppearanceExtractor
    root = tmp_path / "cal"
    (root / "EMP001" / "front").mkdir(parents=True)
    (root / "EMP002" / "front").mkdir(parents=True)
    cv.imwrite(str(root / "EMP001" / "front" / "a.png"),
               np.full((16, 16, 3), 91, dtype=np.uint8))
    cv.imwrite(str(root / "EMP002" / "front" / "a.png"),
               np.full((16, 16, 3), 191, dtype=np.uint8))
    rep = build_report(extractor=AppearanceExtractor(model_path=""),
                       data_dir=root)
    assert rep.manifest_status == "MISSING"
    assert rep.dataset_status == "FILES PRESENT -- PROVENANCE NOT PROVEN"
    assert rep.manifest_validation["ok"] is False
    assert rep.label == ("FILES PRESENT -- PROVENANCE NOT PROVEN "
                         "-- NOT VALIDATED")


def test_build_report_with_wrong_status_manifest_not_real(tmp_path):
    import cv2 as cv
    import json
    from src.reid import AppearanceExtractor
    root = tmp_path / "cal"
    (root / "EMP001" / "front").mkdir(parents=True)
    (root / "EMP002" / "front").mkdir(parents=True)
    cv.imwrite(str(root / "EMP001" / "front" / "a.png"),
               np.full((16, 16, 3), 92, dtype=np.uint8))
    cv.imwrite(str(root / "EMP002" / "front" / "a.png"),
               np.full((16, 16, 3), 192, dtype=np.uint8))
    manifest = {
        "manifest_version": 1,
        "dataset_status": "SIMULATED",
        "files": {
            "EMP001/front/a.png": {"employee_id": "EMP001", "pose": "front"},
            "EMP002/front/a.png": {"employee_id": "EMP002", "pose": "front"},
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    rep = build_report(extractor=AppearanceExtractor(model_path=""),
                       data_dir=root)
    assert rep.manifest_validation["ok"] is False
    assert rep.dataset_status == "FILES PRESENT -- PROVENANCE NOT PROVEN"
    assert rep.label.startswith("FILES PRESENT")


def test_calibration_never_mutates_config(monkeypatch, tmp_path):
    before_t = float(config.REID_STRONG_THRESHOLD)
    before_m = float(config.REID_MARGIN_MIN)
    rec = {"eligible": True, "threshold": 0.70, "margin": 0.05,
           "far_at_recommendation": 0.0, "frr_at_recommendation": 0.0,
           "same_n": 10, "diff_n": 10, "safety_far_max": 0.01}
    proposed = recommended_config("osnet", rec)
    assert proposed["REID_STRONG_THRESHOLD"] == 0.70
    # run the full pipeline (no-data) -- config must be untouched
    run_calibration(data_dir=tmp_path / "empty",
                    out_file=str(tmp_path / "report.json"))
    assert float(config.REID_STRONG_THRESHOLD) == before_t
    assert float(config.REID_MARGIN_MIN) == before_m


def test_recommended_config_floors_weak_suggestion():
    proposed = recommended_config("osnet", {
        "eligible": True, "threshold": 0.30, "margin": 0.0,
        "far_at_recommendation": 0.0, "frr_at_recommendation": 0.0,
        "same_n": 5, "diff_n": 5, "safety_far_max": 0.01})
    assert proposed["REID_STRONG_THRESHOLD"] == 0.50
    assert proposed["REID_MARGIN_MIN"] == 0.0


def test_synthetic_employees_deterministic_and_labelled():
    a = synthetic_employees()
    b = synthetic_employees()
    assert a.keys() == b.keys()
    assert all(x.shape == y.shape
               for emp in a for pose in a[emp]
               for x, y in zip(a[emp][pose], b[emp][pose]))
    # runs through the pipeline and stays labelled SIMULATED
    rep = run_calibration(dry_run_synthetic=True,
                          out_file=str(Path(__import__("tempfile").mkdtemp())
                                        / "r.json"))
    assert isinstance(rep, CalibrationReport)
    assert "SIMULATED" in rep.label


# ---------------------------------------------------------------------------
# Production gate + cadence + feed preconditions
# ---------------------------------------------------------------------------

def test_production_gate_closed_by_default():
    g = production_gate()
    assert g.all_gates is False
    assert set(g.reasons) == {"real_calibration_data", "thresholds_calibrated",
                              "margin_calibrated", "identity_safety_tests",
                              "real_camera_validation"}


def test_production_gate_all_open():
    g = production_gate(real_calibration_data=True,
                        thresholds_calibrated=True, margin_calibrated=True,
                        identity_safety_tests=True, real_camera_validation=True)
    assert g.all_gates is True
    assert g.reasons == []


def test_production_gate_partial_keeps_blocking():
    g = production_gate(thresholds_calibrated=True, margin_calibrated=True,
                        identity_safety_tests=True)
    assert g.all_gates is False
    assert "real_calibration_data" in g.reasons
    assert "real_camera_validation" in g.reasons


def test_cadence_builtin_every_cycle():
    c = cadence_recommendation("built-in", 11.3)
    assert c["cadence"] == "every-cycle"


def test_cadence_osnet_policy():
    c = cadence_recommendation("osnet", 77.5)
    assert c["cadence"] == "unresolved / face-unavailable / periodic refresh"


def test_feed_gate_rejects_black_frozen():
    black = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
    s = feed_usable_stats(black)
    assert s["usable"] is False
    assert s["bright"] is False and s["changing"] is False


def test_feed_gate_accepts_bright_changing():
    rng = np.random.default_rng(7)
    frames = [(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8) +
               np.uint8(i * 3)) % 255
              for i in range(4)]
    s = feed_usable_stats(frames, capture_fps=22.0, require_fps=15.0)
    assert s["usable"] is True
    assert s["bright"] is True and s["changing"] is True and s["fps_ok"]


def test_feed_gate_fps_guard_blocks_low_fps():
    rng = np.random.default_rng(8)
    frames = [rng.normal(128, 40, (48, 64, 3)).clip(0, 255).astype(np.uint8)
              for _ in range(4)]
    s = feed_usable_stats(frames, capture_fps=5.0, require_fps=15.0)
    assert s["usable"] is False
    assert s["fps_ok"] is False


# ---------------------------------------------------------------------------
# Identity safety (tracker hierarchy, unchanged by Phase 50)
# ---------------------------------------------------------------------------

def _track(*, app_adopt=2, adopt=1, switch=3, now=1000.0):
    return Track("t1", "cam1", (0, 0, 20, 20), 0.9, 640, 480, now,
                 adopt_frames=adopt, switch_frames=switch,
                 appear_adopt_frames=app_adopt)


def test_strong_face_beats_weak_reid():
    t = _track()
    t.register_vote("EMP001", 0.95, 1000.0)
    t.register_vote("EMP001", 0.95, 1001.0)
    assert (t.identity, t.identity_source) == ("EMP001", "face")
    t.register_appearance_vote("EMP002", 0.90, 1002.0)
    t.register_appearance_vote("EMP002", 0.90, 1003.0)
    assert (t.identity, t.identity_source) == ("EMP001", "face")


def test_trusted_identity_protected_from_appearance():
    t = _track()
    t.register_vote("EMP004", 0.96, 1000.0)
    t.register_vote("EMP004", 0.96, 1001.0)
    assert t.identity == "EMP004" and t.identity_source == "face"
    t.register_appearance_vote("EMP001", 0.94, 1002.0)
    assert t.identity == "EMP004"


def test_trusted_emp001_immune_to_ambiguous_reid():
    t = _track()
    t.register_vote("EMP001", 0.97, 1000.0)
    t.register_vote("EMP001", 0.97, 1001.0)
    t.register_appearance_vote("EMP002", 0.88, 1002.0)
    t.register_appearance_vote("EMP002", 0.88, 1003.0)
    t.register_appearance_vote("EMP002", 0.88, 1004.0)
    assert t.identity == "EMP001" and t.identity_source == "face"


def test_weak_reid_never_reaches_tracker():
    t = _track()
    t.register_vote(UNKNOWN_ID, 0.60, 1000.0)
    assert t.identity == UNKNOWN_ID
    # weak (sub-threshold) ReID must not be passed as a strong appearance vote
    t.update((0, 0, 20, 20), 0.9, False, 1001.0,
             reid_emp="EMP002", reid_score=0.50, reid_margin=0.05,
             reid_strong=False)
    assert t.identity == UNKNOWN_ID
    assert not t._app_runs


def test_strong_reid_without_face_adopts_appearance():
    t = _track(app_adopt=1)
    t.register_appearance_vote("EMP001", 0.95, 1000.0)
    assert t.identity == "EMP001"
    assert t.identity_source == "appearance"


def test_single_frame_reid_does_not_adopt_instantly():
    t = _track(app_adopt=3)
    t.register_appearance_vote("EMP001", 0.95, 1000.0)
    # a single appearance frame never adopts an identity; the track is only
    # marked unresolved (Unknown), exactly the production behaviour
    assert t.identity == UNKNOWN_ID


def test_no_identity_oscillation():
    t = _track(app_adopt=1, switch=3)
    for i in range(3):
        t.register_appearance_vote("EMP001", 0.95, 1000.0 + i)
    assert t.identity == "EMP001"
    # EMP001 leaves; EMP002 needs its own consecutive run and EMP001 dropping
    for i in range(3):
        t.register_appearance_vote("EMP002", 0.96, 1000.0 + 10 + i)
    assert t.identity == "EMP002"
    # a single stray EMP001 vote must not flip back -> no oscillation
    t.register_appearance_vote("EMP001", 0.95, 1000.0 + 20)
    assert t.identity == "EMP002"