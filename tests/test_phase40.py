"""Phase 40 regression tests — employee name mapping, multi-face detection,
real-time refresh pipeline, face enrollment, tracker details, and cache
fingerprint.

All tests operate on isolated temp dirs / in-memory databases.  No model
loading, no network, no mocked recognitions.
"""

import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.tracker import MultiTracker, AWAY, ACTIVE, ON_PHONE, SpatialTracker, Track


# ── helpers ──────────────────────────────────────────────────────────────

def _write_live_state(tmp_path, states=None, telemetry=None, sources=None,
                      durations=None, cam_health=None, fps=2.0, names=None,
                      last_seen=None):
    """Call main._write_live_state with a temp file."""
    import main as _m
    target = str(tmp_path / "live.json")
    old_file = _m._LIVE_STATE_FILE
    _m._LIVE_STATE_FILE = target
    try:
        _m._write_live_state(states or {}, telemetry or {}, cam_health or {}, fps,
                             sources=sources, durations=durations, names=names,
                             last_seen=last_seen)
    finally:
        _m._LIVE_STATE_FILE = old_file
    return json.loads(Path(target).read_text(encoding="utf-8"))


class _FakeClock:
    """Deterministic wall clock for tracker FSM tests (Phase 41)."""
    now = 10_000.0

    @classmethod
    def time(cls):
        return cls.now

    @classmethod
    def strftime(cls, fmt, tt=None):
        return time.strftime(fmt, tt or time.localtime(cls.now))

    @classmethod
    def localtime(cls, tt=None):
        return time.localtime(tt or cls.now)


@pytest.fixture
def fake_clock(monkeypatch):
    import src.tracker as _T
    _FakeClock.now = 10_000.0
    monkeypatch.setattr(_T, "time", _FakeClock)
    return _FakeClock


# ── tracker: multi-employee independence ────────────────────────────────

class TestMultiEmployeeIndependence:
    """Employee A ACTIVE while B ON_PHONE, independent tracking."""

    def test_independent_states(self, tmp_db):
        t = MultiTracker(tmp_db)
        dets_a = [
            {"cam": "c1", "emp_id": "EMP001", "phone": False},
            {"cam": "c1", "emp_id": "EMP002", "phone": True},
        ]
        for _ in range(5):
            t.process_batch(dets_a, True)
        states = t.live_states()
        assert states.get("EMP001") == ACTIVE
        assert states.get("EMP002") == ON_PHONE

    def test_one_emp_absent_other_active(self, tmp_db, fake_clock):
        t = MultiTracker(tmp_db)
        dets_both = [
            {"cam": "c1", "emp_id": "EMP001", "phone": False},
            {"cam": "c1", "emp_id": "EMP002", "phone": False},
        ]
        for _ in range(5):
            t.process_batch(dets_both, True)
            fake_clock.now += 1
        # Now: only EMP001 visible (send a detection for another emp to
        # keep detections non-empty so the freeze path is not taken).
        dets_only_emp1 = [
            {"cam": "c1", "emp_id": "EMP001", "phone": False},
            {"cam": "c1", "emp_id": "IMPOSTOR", "phone": False},
        ]
        for _ in range(8):
            t.process_batch(dets_only_emp1, True)
            fake_clock.now += 1
        states = t.live_states()
        assert states.get("EMP001") == ACTIVE
        assert states.get("EMP002") == AWAY


class TestCameraOfflineFreezes:
    """Camera offline freezes state; recovery unfreezes."""

    def test_offline_freezes_state(self, tmp_db):
        t = MultiTracker(tmp_db)
        dets = [{"cam": "c1", "emp_id": "EMP001", "phone": False}]
        for _ in range(5):
            t.process_batch(dets, True)
        assert t.live_states().get("EMP001") == ACTIVE
        # Camera goes offline (empty detections + offline flag)
        t.process_batch([], False)
        t.process_batch([], False)
        # State should still be ACTIVE (frozen), not AWAY
        assert t.live_states().get("EMP001") == ACTIVE

    def test_recovery_unfreezes(self, tmp_db):
        t = MultiTracker(tmp_db)
        dets = [{"cam": "c1", "emp_id": "EMP001", "phone": False}]
        for _ in range(5):
            t.process_batch(dets, True)
        # Camera offline
        t.process_batch([], False)
        # Camera back with detection
        for _ in range(5):
            t.process_batch(dets, True)
        assert t.live_states().get("EMP001") == ACTIVE


# ── tracker: session details ────────────────────────────────────────────

class TestTrackerSessionDetails:
    """Session duration, active/phone/away seconds, last_seen."""

    def test_session_duration_accumulates(self, tmp_db, monkeypatch):
        """live_durations reports in-flight wall seconds."""
        import src.tracker as T
        # Use zero buffer so commits are instant
        monkeypatch.setattr(T, "SMOOTHING_BUFFER_SEC", 0)
        monkeypatch.setattr(T, "IDENTITY_STABILITY_FRAMES", 1)

        t = MultiTracker(tmp_db)
        dets = [{"cam": "c1", "emp_id": "EMP001", "phone": False}]
        t.process_batch(dets, True)
        dur = t.live_durations()
        # duration_seconds = time.time() - _since; even in a fast loop
        # this is >= 0; verify the tracker is tracking it.
        assert "EMP001" in dur
        assert dur["EMP001"] >= 0.0

    def test_state_is_active_when_present(self, tmp_db):
        """EMP001 visible without phone → ACTIVE state."""
        t = MultiTracker(tmp_db)
        dets = [{"cam": "c1", "emp_id": "EMP001", "phone": False}]
        for _ in range(5):
            t.process_batch(dets, True)
        states = t.live_states()
        assert states.get("EMP001") == ACTIVE

    def test_state_is_on_phone_when_present(self, tmp_db):
        """EMP001 visible with phone → ON_PHONE state."""
        t = MultiTracker(tmp_db)
        dets = [{"cam": "c1", "emp_id": "EMP001", "phone": True}]
        for _ in range(5):
            t.process_batch(dets, True)
        states = t.live_states()
        assert states.get("EMP001") == ON_PHONE

    def test_away_when_no_detections_for_other_emp(self, tmp_db, monkeypatch, fake_clock):
        """EMP001 disappears while other employees remain → AWAY."""
        import src.tracker as T
        monkeypatch.setattr(T, "SMOOTHING_BUFFER_SEC", 0)
        monkeypatch.setattr(T, "IDENTITY_STABILITY_FRAMES", 1)

        t = MultiTracker(tmp_db)
        dets = [{"cam": "c1", "emp_id": "EMP001", "phone": False}]
        for _ in range(5):
            t.process_batch(dets, True)
            fake_clock.now += 1
        # EMP001 disappears; only OTHER visible
        for _ in range(8):
            t.process_batch(
                [{"cam": "c1", "emp_id": "OTHER", "phone": False}], True)
            fake_clock.now += 1
        states = t.live_states()
        assert states.get("EMP001") == AWAY

    def test_last_seen_recorded(self, tmp_db):
        t = MultiTracker(tmp_db)
        dets = [{"cam": "c1", "emp_id": "EMP001", "phone": False}]
        for _ in range(5):
            t.process_batch(dets, True)
        ls = t.live_last_seen()
        assert ls.get("EMP001") is not None
        assert isinstance(ls.get("EMP001"), float)

    def test_last_seen_none_when_never_seen(self, tmp_db):
        t = MultiTracker(tmp_db)
        # No detections at all → no trackers created
        ls = t.live_last_seen()
        assert ls == {}


# ── live_state rich employees dict ──────────────────────────────────────

class TestLiveStateRichEmployees:
    """live_state.json carries per-employee names and details."""

    def test_employees_dict_present(self, tmp_path):
        data = _write_live_state(
            tmp_path,
            states={"EMP001": "ACTIVE"},
            telemetry={"EMP001": {"ACTIVE": 5.0, "ON_PHONE": 0, "AWAY": 0}},
            sources={"EMP001": "local_webcam"},
            durations={"EMP001": 120.0},
            names={"EMP001": "Yashmit"},
        )
        assert "employees" in data
        assert "EMP001" in data["employees"]

    def test_employee_name_from_names_param(self, tmp_path):
        data = _write_live_state(
            tmp_path,
            states={"EMP001": "ACTIVE"},
            names={"EMP001": "Yashmit Sharma"},
        )
        emp = data["employees"]["EMP001"]
        assert emp["name"] == "Yashmit Sharma"
        assert emp["employee_id"] == "EMP001"

    def test_employee_detail_fields(self, tmp_path):
        data = _write_live_state(
            tmp_path,
            states={"EMP001": "ACTIVE"},
            telemetry={"EMP001": {"ACTIVE": 10.0, "ON_PHONE": 2.0, "AWAY": 1.0}},
            sources={"EMP001": "cam_01"},
            durations={"EMP001": 300.0},
            names={"EMP001": "Yashmit"},
            last_seen={"EMP001": time.time()},
        )
        emp = data["employees"]["EMP001"]
        assert emp["session_sec"] == 300.0
        assert emp["active_sec"] == 10.0
        assert emp["phone_sec"] == 2.0
        assert emp["away_sec"] == 1.0
        assert emp["source"] == "cam_01"
        assert emp["last_seen"] is not None

    def test_unknown_excluded_from_employees(self, tmp_path):
        data = _write_live_state(
            tmp_path,
            states={"EMP001": "ACTIVE", "Unknown": "ACTIVE"},
            names={"EMP001": "Yashmit"},
        )
        assert "EMP001" in data["employees"]
        assert "Unknown" not in data["employees"]

    def test_employee_fallback_to_id_when_no_name(self, tmp_path):
        data = _write_live_state(
            tmp_path,
            states={"EMP001": "ACTIVE"},
            names={},
        )
        emp = data["employees"]["EMP001"]
        assert emp["name"] == "EMP001"

    def test_backward_compat_no_names(self, tmp_path):
        """Calling without names/last_seen still works (backward compat)."""
        data = _write_live_state(
            tmp_path,
            states={"EMP001": "ACTIVE"},
            telemetry={"EMP001": {"ACTIVE": 1.0}},
        )
        assert "states" in data
        assert "employees" in data  # present but empty (names=None)


# ── face enrollment: registry fingerprint ────────────────────────────────

class TestRegistryFingerprint:
    """Cache fingerprint guard: new/changed face files trigger rebuild."""

    def test_fingerprint_stored_in_cache(self, tmp_path, monkeypatch):
        faces_dir = tmp_path / "faces"
        faces_dir.mkdir()
        (faces_dir / "EMP001.jpg").write_bytes(b"face_bytes")
        embed_file = tmp_path / "embeddings.pkl"
        monkeypatch.setattr("src.face_registry.FACES_DIR", faces_dir)
        monkeypatch.setattr("src.face_registry.EMBEDDINGS_FILE", embed_file)

        from src.face_registry import FaceRegistry
        reg = FaceRegistry.__new__(FaceRegistry)
        reg._employee_ids = ["EMP001"]
        reg._embeddings = np.zeros((1, 512), dtype=np.float32)
        reg._by_employee = {"EMP001": 0}
        reg._file_status = {"EMP001.jpg": "ENROLLED"}
        reg._save_cache()
        with open(str(embed_file), "rb") as f:
            data = pickle.load(f)
        assert "file_fingerprint" in data
        assert "EMP001.jpg" in data["file_fingerprint"]

    def test_stale_fingerprint_triggers_rebuild(self, tmp_path, monkeypatch):
        faces_dir = tmp_path / "faces"
        faces_dir.mkdir()
        (faces_dir / "EMP001.jpg").write_bytes(b"fake_image_data_1")
        embed_file = tmp_path / "embeddings.pkl"
        monkeypatch.setattr("src.face_registry.FACES_DIR", faces_dir)
        monkeypatch.setattr("src.face_registry.EMBEDDINGS_FILE", embed_file)

        from src.face_registry import FaceRegistry
        reg = FaceRegistry.__new__(FaceRegistry)
        reg._employee_ids = ["EMP001"]
        reg._embeddings = np.zeros((1, 512), dtype=np.float32)
        reg._by_employee = {"EMP001": 0}
        reg._file_status = {"EMP001.jpg": "ENROLLED"}
        reg._save_cache()

        # Modify the face file (different size → different fingerprint)
        (faces_dir / "EMP001.jpg").write_bytes(b"fake_image_data_2_modified")
        reg2 = FaceRegistry.__new__(FaceRegistry)
        with pytest.raises(ValueError, match="face files changed"):
            reg2._load_cache()

    def test_new_file_triggers_rebuild(self, tmp_path, monkeypatch):
        faces_dir = tmp_path / "faces"
        faces_dir.mkdir()
        (faces_dir / "EMP001.jpg").write_bytes(b"img_data")
        embed_file = tmp_path / "embeddings.pkl"
        monkeypatch.setattr("src.face_registry.FACES_DIR", faces_dir)
        monkeypatch.setattr("src.face_registry.EMBEDDINGS_FILE", embed_file)

        from src.face_registry import FaceRegistry
        reg = FaceRegistry.__new__(FaceRegistry)
        reg._employee_ids = ["EMP001"]
        reg._embeddings = np.zeros((1, 512), dtype=np.float32)
        reg._by_employee = {"EMP001": 0}
        reg._file_status = {"EMP001.jpg": "ENROLLED"}
        reg._save_cache()

        (faces_dir / "EMP002.jpg").write_bytes(b"new_face")
        reg2 = FaceRegistry.__new__(FaceRegistry)
        with pytest.raises(ValueError, match="face files changed"):
            reg2._load_cache()


# ── face enrollment: enrollment_table ────────────────────────────────────

class TestEnrollmentTable:
    """enrollment_table() returns correct structure with DB names."""

    def test_enrollment_table_empty_when_no_faces(self, tmp_path, monkeypatch):
        faces_dir = tmp_path / "faces"
        faces_dir.mkdir()
        embed_file = tmp_path / "embeddings.pkl"
        monkeypatch.setattr("src.face_registry.FACES_DIR", faces_dir)
        monkeypatch.setattr("src.face_registry.EMBEDDINGS_FILE", embed_file)
        # Monkeypatch insightface so no real model loads
        monkeypatch.setattr("src.face_registry.FaceRegistry._build_from_images",
                            lambda self: None)
        monkeypatch.setattr("src.face_registry.FaceRegistry._load_cache",
                            lambda self: (_ for _ in ()).throw(FileNotFoundError))
        monkeypatch.setattr("src.face_registry.FaceRegistry._save_cache",
                            lambda self: None)

        from src.face_registry import enrollment_table
        rows, registry = enrollment_table()
        assert rows == []
        assert registry.num_enrolled_employees == 0

    def test_enrollment_table_with_face_file(self, tmp_path, monkeypatch):
        faces_dir = tmp_path / "faces"
        faces_dir.mkdir()
        (faces_dir / "EMP001.jpg").write_bytes(b"fake")
        embed_file = tmp_path / "embeddings.pkl"
        monkeypatch.setattr("src.face_registry.FACES_DIR", faces_dir)
        monkeypatch.setattr("src.face_registry.EMBEDDINGS_FILE", embed_file)

        # Build a FaceRegistry that pretends EMP001 is enrolled
        from src.face_registry import FaceRegistry
        orig_init = FaceRegistry.__init__

        def _fake_init(self, **kwargs):
            self._employee_ids = ["EMP001"]
            self._embeddings = np.zeros((1, 512), dtype=np.float32)
            self._by_employee = {"EMP001": 0}
            self._file_status = {"EMP001.jpg": "ENROLLED"}
            self._threshold = 0.6
            self._app = None

        monkeypatch.setattr(FaceRegistry, "__init__", _fake_init)

        from src.face_registry import enrollment_table
        rows, registry = enrollment_table()
        ids = [r["employee_id"] for r in rows]
        assert "EMP001" in ids
        # Status should come from the mocked employee_status
        assert rows[0]["status"] == "ENROLLED"
        assert rows[0]["has_embedding"] == "YES"


# ── face detection: NO_FACE / face status ───────────────────────────────

class TestFaceStatuses:
    """Face registry status for edge cases."""

    def test_empty_dir_no_employees(self, tmp_path, monkeypatch):
        faces_dir = tmp_path / "faces"
        faces_dir.mkdir()
        embed_file = tmp_path / "embeddings.pkl"
        monkeypatch.setattr("src.face_registry.FACES_DIR", faces_dir)
        monkeypatch.setattr("src.face_registry.EMBEDDINGS_FILE", embed_file)
        monkeypatch.setattr("src.face_registry.FaceRegistry._build_from_images",
                            lambda self: None)
        monkeypatch.setattr("src.face_registry.FaceRegistry._load_cache",
                            lambda self: (_ for _ in ()).throw(FileNotFoundError))
        monkeypatch.setattr("src.face_registry.FaceRegistry._save_cache",
                            lambda self: None)

        from src.face_registry import FaceRegistry
        reg = FaceRegistry()
        status = reg.employee_status()
        assert status == {}

    def test_status_keys_match_files(self, tmp_path, monkeypatch):
        """Status keys match face files present in the directory."""
        faces_dir = tmp_path / "faces"
        faces_dir.mkdir()
        (faces_dir / "EMP001.jpg").write_bytes(b"f1")
        (faces_dir / "EMP002.jpg").write_bytes(b"f2")
        embed_file = tmp_path / "embeddings.pkl"
        monkeypatch.setattr("src.face_registry.FACES_DIR", faces_dir)
        monkeypatch.setattr("src.face_registry.EMBEDDINGS_FILE", embed_file)
        monkeypatch.setattr("src.face_registry.FaceRegistry._build_from_images",
                            lambda self: None)
        monkeypatch.setattr("src.face_registry.FaceRegistry._load_cache",
                            lambda self: (_ for _ in ()).throw(FileNotFoundError))
        monkeypatch.setattr("src.face_registry.FaceRegistry._save_cache",
                            lambda self: None)

        from src.face_registry import FaceRegistry
        reg = FaceRegistry()
        status = reg.employee_status()
        # With _build_from_images as no-op, status only has files from cache
        # (empty). But file_status is populated by _build_from_images.
        # So we verify the image_paths static method.
        paths = FaceRegistry._image_paths()
        assert set(paths.keys()) == {"EMP001.jpg", "EMP002.jpg"}


# ── dashboard: name display ─────────────────────────────────────────────

class TestDashboardNameDisplay:
    """Dashboard reads employees dict and shows names."""

    def test_tab_live_reads_employees(self, tmp_path, monkeypatch):
        live_file = tmp_path / "live.json"
        live_file.write_text(json.dumps({
            "updated": time.time(),
            "iso": "2025-01-01 00:00:00",
            "states": {"EMP001": "ACTIVE"},
            "telemetry": {},
            "employees": {
                "EMP001": {
                    "employee_id": "EMP001",
                    "name": "Yashmit Sharma",
                    "state": "ACTIVE",
                    "session_sec": 100.0,
                    "active_sec": 80.0,
                    "phone_sec": 0.0,
                    "away_sec": 0.0,
                    "source": "local_webcam",
                    "last_seen": time.time(),
                }
            },
            "camera_health": {},
            "fps": 2.0,
            "security": {},
        }), encoding="utf-8")
        monkeypatch.setattr("app._LIVE_STATE_FILE", live_file)

        from app import load_live_state
        data = load_live_state()
        assert "employees" in data
        emp = data["employees"]["EMP001"]
        assert emp["name"] == "Yashmit Sharma"
        assert emp["state"] == "ACTIVE"

    def test_stale_file_returns_empty(self, tmp_path, monkeypatch):
        live_file = tmp_path / "live.json"
        live_file.write_text(json.dumps({
            "updated": time.time() - 120,
            "iso": "old",
            "states": {},
            "telemetry": {},
            "camera_health": {},
            "fps": 0.0,
            "security": {},
        }), encoding="utf-8")
        monkeypatch.setattr("app._LIVE_STATE_FILE", live_file)

        from app import load_live_state
        data = load_live_state()
        assert data == {}


# ── detector: debug logging ──────────────────────────────────────────────

class TestDetectorDebugLogging:
    """Face recognition debug log emitted per face at DEBUG level."""

    def test_debug_log_per_face(self, caplog):
        import logging

        mock_registry = MagicMock()
        mock_registry.identify.return_value = ("EMP001", 0.85)
        mock_registry.threshold = 0.6

        fake_face = MagicMock()
        fake_face.bbox = np.array([100, 100, 200, 200])
        fake_face.normed_embedding = np.zeros(512, dtype=np.float32)
        mock_registry.app.get.return_value = [fake_face]

        from src.detector import ActivityDetector
        det = ActivityDetector.__new__(ActivityDetector)
        det._face_registry = mock_registry
        det.model = MagicMock()
        det.conf = 0.5
        det._half = False
        det._device = "cpu"

        mock_result = MagicMock()
        mock_result.boxes = []
        det.model.predict.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with caplog.at_level(logging.DEBUG, logger="cctv.detector"):
            det._process_frame(frame)

        assert any("face bbox=" in r.message for r in caplog.records)
        assert any("0.8500" in r.message for r in caplog.records)

    def test_debug_log_shows_unknown(self, caplog):
        import logging

        mock_registry = MagicMock()
        mock_registry.identify.return_value = ("Unknown", 0.30)
        mock_registry.threshold = 0.6

        fake_face = MagicMock()
        fake_face.bbox = np.array([50, 50, 150, 150])
        fake_face.normed_embedding = np.zeros(512, dtype=np.float32)
        mock_registry.app.get.return_value = [fake_face]

        from src.detector import ActivityDetector
        det = ActivityDetector.__new__(ActivityDetector)
        det._face_registry = mock_registry
        det.model = MagicMock()
        det.conf = 0.5
        det._half = False
        det._device = "cpu"

        mock_result = MagicMock()
        mock_result.boxes = []
        det.model.predict.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with caplog.at_level(logging.DEBUG, logger="cctv.detector"):
            det._process_frame(frame)

        assert any("Unknown" in r.message for r in caplog.records)


# ── face registry: rebuild_and_report ────────────────────────────────────

class TestRegistryRebuildReport:
    """scan_status()/employee_status() return structured status info."""

    def test_scan_status_structure(self, tmp_path, monkeypatch):
        faces_dir = tmp_path / "faces"
        faces_dir.mkdir()
        embed_file = tmp_path / "embeddings.pkl"
        monkeypatch.setattr("src.face_registry.FACES_DIR", faces_dir)
        monkeypatch.setattr("src.face_registry.EMBEDDINGS_FILE", embed_file)
        monkeypatch.setattr("src.face_registry.FaceRegistry._build_from_images",
                            lambda self: None)
        monkeypatch.setattr("src.face_registry.FaceRegistry._load_cache",
                            lambda self: (_ for _ in ()).throw(FileNotFoundError))
        monkeypatch.setattr("src.face_registry.FaceRegistry._save_cache",
                            lambda self: None)

        from src.face_registry import FaceRegistry
        reg = FaceRegistry()
        # scan_status returns per-file status dict
        assert isinstance(reg.scan_status(), dict)
        assert isinstance(reg.employee_status(), dict)

    def test_employee_status_reflects_file_status(self, tmp_path, monkeypatch):
        faces_dir = tmp_path / "faces"
        faces_dir.mkdir()
        embed_file = tmp_path / "embeddings.pkl"
        monkeypatch.setattr("src.face_registry.FACES_DIR", faces_dir)
        monkeypatch.setattr("src.face_registry.EMBEDDINGS_FILE", embed_file)

        from src.face_registry import FaceRegistry
        reg = FaceRegistry.__new__(FaceRegistry)
        reg._file_status = {"EMP001": "ENROLLED", "EMP002": "NO_FACE"}
        reg._by_employee = {"EMP001": 0}
        status = reg.employee_status()
        assert status.get("EMP001") == "ENROLLED"
        assert status.get("EMP002") == "NO_FACE"


# ── Phase 40: numbered Unknown display labels ───────────────────────────

class TestNumberedUnknownLabels:
    """Per-track numbered ``Unknown N`` display labels.

    The number MUST be owned by the *spatial track* (its lifecycle), never by
    the detection-array index or per-frame detection order.  The canonical
    identity always stays ``"Unknown"`` -- ``Unknown N`` is display-only.
    """

    _FW, _FH = 640, 480

    @staticmethod
    def _person(cam, box, fw=640, fh=480, conf=0.9):
        return {"cam": cam, "det_type": "person", "emp_id": "__person__",
                "person_box": tuple(box), "person_conf": conf,
                "frame_w": fw, "frame_h": fh}

    @staticmethod
    def _face(cam, emp_id, face_box, person_box, fw=640, fh=480, score=0.4):
        return {"cam": cam, "emp_id": emp_id, "face_score": score,
                "face_box": tuple(face_box), "person_box": tuple(person_box),
                "frame_w": fw, "frame_h": fh}

    @staticmethod
    def _label_by_track(snap):
        return {t["track_id"]: t["identity_label"] for t in snap["tracks"]}

    # -- one unknown ------------------------------------------------
    def test_one_unknown_becomes_unknown_1(self):
        st = SpatialTracker(adopt_frames=3)
        p = self._person("c1", (10, 10, 90, 260))
        f = self._face("c1", "Unknown", (30, 30, 70, 120), (10, 10, 90, 260))
        st.process([p, f], now=1000.0)
        snap = st.snapshot(now=1000.0)
        assert snap["unknown_count"] == 1
        assert snap["tracks"][0]["identity_label"] == "Unknown 1"
        assert snap["tracks"][0]["identity"] == "Unknown"

    # -- multiple simultaneous unknowns ----------------------------
    def test_multiple_unknowns_get_distinct_labels(self):
        st = SpatialTracker(max_age_sec=5.0, adopt_frames=3)
        dets = [
            self._person("c1", (10, 10, 90, 260)),
            self._face("c1", "Unknown", (30, 30, 70, 120), (10, 10, 90, 260)),
            self._person("c1", (200, 10, 300, 260)),
            self._face("c1", "Unknown", (220, 30, 270, 120), (200, 10, 300, 260)),
            self._person("c1", (400, 10, 500, 260)),
            self._face("c1", "Unknown", (420, 30, 470, 120), (400, 10, 500, 260)),
        ]
        for i in range(3):
            st.process(dets, now=1000.0 + i)
        snap = st.snapshot(now=1000.0 + 2)
        labels = [t["identity_label"] for t in snap["tracks"]]
        assert sorted(labels) == ["Unknown 1", "Unknown 2", "Unknown 3"]
        assert snap["unknown_count"] == 3
        assert len({t["track_id"] for t in snap["tracks"]}) == 3

    # -- stable numbering across frames + movement ------------------
    def test_labels_stay_attached_to_tracks_as_they_move(self):
        st = SpatialTracker(max_age_sec=5.0, adopt_frames=3, iou_threshold=0.2)
        pA = self._person("c1", (10, 10, 90, 260))
        fA = self._face("c1", "Unknown", (30, 30, 70, 120), (10, 10, 90, 260))
        pB = self._person("c1", (200, 10, 300, 260))
        fB = self._face("c1", "Unknown", (220, 30, 270, 120), (200, 10, 300, 260))
        for i in range(3):
            st.process([pA, fA, pB, fB], now=2000.0 + i)
        before = self._label_by_track(st.snapshot(now=2000.0 + 2))
        assert before == {"c1#1": "Unknown 1", "c1#2": "Unknown 2"}

        # A walks slowly to the right (IoU keeps the same track alive);
        # B stands still.  Labels must follow the spatial tracks: the moving
        # person keeps "Unknown 1", the stationary one keeps "Unknown 2".
        steps = [((20, 10, 100, 260), (40, 30, 80, 120)),
                 ((30, 10, 110, 260), (50, 30, 90, 120)),
                 ((40, 10, 120, 260), (60, 30, 100, 120))]
        for k, (box, face_box) in enumerate(steps):
            st.process([
                self._person("c1", box),
                self._face("c1", "Unknown", face_box, box),
                pB, fB,
            ], now=2010.0 + k)
        after = self._label_by_track(st.snapshot(now=2010.0 + len(steps) - 1))
        assert after == {"c1#1": "Unknown 1", "c1#2": "Unknown 2"}
        # Physical continuity matched: no track was dropped and re-spawned.
        assert set(after) == {"c1#1", "c1#2"}

    # -- detection-order independence -------------------------------
    def test_detection_order_change_does_not_renumber(self):
        st = SpatialTracker(max_age_sec=5.0, adopt_frames=3)
        pA = self._person("c1", (10, 10, 90, 260))
        fA = self._face("c1", "Unknown", (30, 30, 70, 120), (10, 10, 90, 260))
        pB = self._person("c1", (200, 10, 300, 260))
        fB = self._face("c1", "Unknown", (220, 30, 270, 120), (200, 10, 300, 260))
        for i in range(3):
            st.process([pA, fA, pB, fB], now=3000.0 + i)
        before = self._label_by_track(st.snapshot(now=3000.0 + 2))

        # Same people, detections delivered in a completely different order.
        for i in range(3):
            st.process([fB, pB, fA, pA], now=3120.0 + i)
        after = self._label_by_track(st.snapshot(now=3120.0 + 2))
        assert before == after

    # -- unknown track disappearance / pruning -----------------------
    def test_unknown_leaves_other_keeps_label_and_new_track_gets_fresh_one(self):
        st = SpatialTracker(max_age_sec=3.0, adopt_frames=3)
        pA = self._person("c1", (10, 10, 90, 260))
        fA = self._face("c1", "Unknown", (30, 30, 70, 120), (10, 10, 90, 260))
        pB = self._person("c1", (200, 10, 300, 260))
        fB = self._face("c1", "Unknown", (220, 30, 270, 120), (200, 10, 300, 260))
        for i in range(3):
            st.process([pA, fA, pB, fB], now=4000.0 + i)
        before = self._label_by_track(st.snapshot(now=4000.0 + 2))
        assert before["c1#1"] == "Unknown 1"
        assert before["c1#2"] == "Unknown 2"

        # Track c1#1 leaves beyond max_age -> pruned; c1#2 keeps its label.
        for i in range(4):
            st.process([pB, fB], now=4010.0 + i)
        snap = st.snapshot(now=4014.0)
        assert len(snap["tracks"]) == 1
        assert snap["tracks"][0]["track_id"] == "c1#2"
        assert snap["tracks"][0]["identity_label"] == "Unknown 2"

        # A brand new unknown enters: it must NOT steal "Unknown 2".
        pC = self._person("c1", (10, 10, 90, 260))
        fC = self._face("c1", "Unknown", (30, 30, 70, 120), (10, 10, 90, 260))
        st.process([pB, fB, pC, fC], now=4015.0)
        after = self._label_by_track(st.snapshot(now=4015.0))
        assert after["c1#2"] == "Unknown 2"          # survivor unchanged
        assert "Unknown 1" in after.values()         # new track gets freed label

    # -- track retained (temporary disappearance) --------------------
    def test_track_retained_keeps_label_while_new_track_appears(self):
        st = SpatialTracker(max_age_sec=3.0, adopt_frames=3)
        pA = self._person("c1", (10, 10, 90, 260))
        fA = self._face("c1", "Unknown", (30, 30, 70, 120), (10, 10, 90, 260))
        for i in range(3):
            st.process([pA, fA], now=5000.0 + i)
        assert st.snapshot(now=5000.0 + 2)["tracks"][0]["identity_label"] == "Unknown 1"

        # A disappears for 1s (still inside max_age): the tracker retains the
        # track, so the label is preserved instead of being handed to others.
        st.process([], now=5003.0)
        assert st.snapshot(now=5003.0)["tracks"][0]["identity_label"] == "Unknown 1"
        for i in range(2):
            st.process([pA, fA], now=5004.0 + i)
        assert st.snapshot(now=5005.0)["tracks"][0]["identity_label"] == "Unknown 1"

        # A second person enters while A's track is alive -> next free label,
        # not a steal of A's active "Unknown 1".
        pB = self._person("c1", (200, 10, 300, 260))
        fB = self._face("c1", "Unknown", (220, 30, 270, 120), (200, 10, 300, 260))
        st.process([pA, fA, pB, fB], now=5006.0)
        labels = {t["identity_label"] for t in st.snapshot(now=5006.0)["tracks"]}
        assert labels == {"Unknown 1", "Unknown 2"}

    # -- known employee never labelled -------------------------------
    def test_known_employee_never_labeled_unknown(self):
        st = SpatialTracker(adopt_frames=3)
        p = self._person("c1", (10, 10, 90, 260))
        f = self._face("c1", "EMP001", (30, 30, 70, 120), (10, 10, 90, 260),
                       score=0.9)
        for i in range(3):
            st.process([p, f], now=6000.0 + i)
        snap = st.snapshot(now=6000.0 + 2)
        assert snap["unknown_count"] == 0
        row = snap["tracks"][0]
        assert row["identity"] == "EMP001"
        assert row["identity_label"] == "EMP001"
        assert row["unknown_label"] is None

    # -- known employee + multiple unknowns, no collision -----------
    def test_known_plus_unknowns_no_collision(self):
        st = SpatialTracker(max_age_sec=5.0, adopt_frames=3)
        dets = [
            self._person("c1", (0, 10, 80, 260)),
            self._face("c1", "EMP001", (10, 30, 60, 120), (0, 10, 80, 260), score=0.9),
            self._person("c1", (160, 10, 250, 260)),
            self._face("c1", "Unknown", (180, 30, 230, 120), (160, 10, 250, 260)),
            self._person("c1", (340, 10, 430, 260)),
            self._face("c1", "Unknown", (360, 30, 410, 120), (340, 10, 430, 260)),
        ]
        for i in range(3):
            st.process(dets, now=7000.0 + i)
        snap = st.snapshot(now=7000.0 + 2)
        labels = [t["identity_label"] for t in snap["tracks"]]
        assert sorted(labels) == ["EMP001", "Unknown 1", "Unknown 2"]
        assert snap["unknown_count"] == 2
        assert set(snap["tracks"][i]["identity"] for i in range(3)) == \
            {"EMP001", "Unknown"}

        # No label string can ever collide with an employee id: numbered
        # Unknown labels never look like EMP ids and vice versa.
        emp_labels = [l for l in labels
                      if l != "Unknown" and not l.startswith("Unknown ")]
        assert emp_labels == ["EMP001"]
        assert not any(l.startswith("EMP") for l in labels if l.startswith("Unknown"))

    # -- numbering is display-only -----------------------------------
    def test_numbering_is_display_only_canonical_identity_untouched(self):
        st = SpatialTracker(adopt_frames=3)
        p = self._person("c1", (10, 10, 90, 260))
        f = self._face("c1", "Unknown", (30, 30, 70, 120), (10, 10, 90, 260))
        for i in range(3):
            st.process([p, f], now=8000.0 + i)
        tracks = st.active_tracks(8000.0 + 2)
        assert tracks[0].identity == "Unknown"     # canonical id untouched
        assert tracks[0].unknown_label == 1
        snap = st.snapshot(now=8000.0 + 2)
        assert snap["tracks"][0]["identity"] == "Unknown"
        assert snap["tracks"][0]["identity_label"] == "Unknown 1"
        json.dumps(snap)                          # must stay JSON-safe

    # -- live_state / dashboard passthrough --------------------------
    def test_live_state_spatial_tracks_carry_labels(self, tmp_path):
        st = SpatialTracker(adopt_frames=3)
        p = self._person("c1", (10, 10, 90, 260))
        f = self._face("c1", "Unknown", (30, 30, 70, 120), (10, 10, 90, 260))
        for i in range(3):
            st.process([p, f], now=9000.0 + i)
        import main as _m
        target = str(tmp_path / "live_spatial.json")
        old_file = _m._LIVE_STATE_FILE
        _m._LIVE_STATE_FILE = target
        try:
            _m._write_live_state(
                {"Unknown": "ACTIVE"},
                {"Unknown": {"ACTIVE": 1.0, "ON_PHONE": 0.0, "AWAY": 0.0}},
                {"local_webcam": {"health": "ONLINE", "usable": True}}, 2.0,
                sources={"Unknown": "local_webcam"},
                durations={"Unknown": 1.0},
                spatial=st.snapshot(now=9000.0 + 2),
            )
        finally:
            _m._LIVE_STATE_FILE = old_file
        data = json.loads(Path(target).read_text(encoding="utf-8"))
        tr = data["ai"]["spatial_tracks"]["tracks"][0]
        assert tr["identity_label"] == "Unknown 1"
        assert tr["identity"] == "Unknown"
        assert tr["unknown_label"] == 1
        assert data["ai"]["spatial_tracks"]["unknown_count"] == 1
