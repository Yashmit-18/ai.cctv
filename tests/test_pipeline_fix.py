"""Regression tests for the live-pipeline fixes (recognition-chain audit).

Covers:
* dashboard status cells contain plain text, never raw Streamlit markdown
* ``NOT OBSERVED`` (never a blank/fabricated state) for never-observed employees
* stale-snapshot surfacing via ``live_state_age``
* tiny insightface noise faces are skipped (no phantom Unknown increments)
* structured FACE_DEBUG / DETECTION_DEBUG lines behind the config switch
* ``--source auto`` camera-index auto-detection (first usable / non-dark device)

No models, no real webcams, no network.  Frames are synthetic numpy arrays.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app import _status_badge, enrollment_status_frame, load_live_state
from app import live_state_age
from src.detector import ActivityDetector


# ── dashboard: plain-text status cells ───────────────────────────────────

class TestPlainTextStatuses:
    """The reported artifact ':#e74c3c[●] **AWAY**' must never recur."""

    def test_status_badge_is_plain_text(self):
        for state, expected in [
            ("ACTIVE", "ACTIVE"),
            ("ON_PHONE", "ON PHONE"),
            ("AWAY", "AWAY"),
            ("", "NOT OBSERVED"),
            ("NOT_OBSERVED", "NOT OBSERVED"),
            (None, "NOT OBSERVED"),
        ]:
            badge = _status_badge(state)
            assert badge == expected
            assert ":" not in badge and "*" not in badge and "[" not in badge

    def test_enrollment_status_frame_is_plain_text(self):
        df = enrollment_status_frame(
            {"EMP001": "ENROLLED", "EMP002": "NO_FACE", "EMP003": "INVALID_IMAGE"}
        )
        assert list(df["Status"]) == ["ENROLLED", "NO FACE", "INVALID IMAGE"]
        assert all(":" not in s and "**" not in s for s in df["Status"])


# ── live_state: NOT OBSERVED semantics ───────────────────────────────────

class TestLiveStateNotObserved:
    def _write(self, tmp_path, **kwargs):
        import main as m
        target = str(tmp_path / "live.json")
        old = m._LIVE_STATE_FILE
        m._LIVE_STATE_FILE = target
        try:
            m._write_live_state(
                kwargs.get("states", {}),
                kwargs.get("telemetry", {}),
                kwargs.get("cam_health", {}),
                kwargs.get("fps", 2.0),
                sources=kwargs.get("sources"),
                durations=kwargs.get("durations"),
                names=kwargs.get("names"),
                last_seen=kwargs.get("last_seen"),
                last_detection_sec=kwargs.get("last_detection_sec"),
            )
        finally:
            m._LIVE_STATE_FILE = old
        return json.loads(Path(target).read_text(encoding="utf-8"))

    def test_never_observed_reports_not_observed(self, tmp_path):
        data = self._write(
            tmp_path,
            states={"EMP001": "ACTIVE"},
            names={"EMP001": "Yashmit", "EMP002": "Rahul"},
            last_seen={"EMP001": time.time()},
        )
        assert data["employees"]["EMP001"]["state"] == "ACTIVE"
        assert data["employees"]["EMP002"]["state"] == "NOT_OBSERVED"

    def test_observed_then_departed_is_away(self, tmp_path):
        data = self._write(
            tmp_path,
            states={},  # EMP001 left the tracker's live snapshot
            names={"EMP001": "Yashmit"},
            last_seen={"EMP001": time.time() - 30},
        )
        assert data["employees"]["EMP001"]["state"] == "AWAY"

    def test_last_detection_sec_carried(self, tmp_path):
        data = self._write(
            tmp_path,
            states={"EMP001": "ACTIVE"},
            names={"EMP001": "Yashmit"},
            last_seen={"EMP001": time.time()},
            last_detection_sec=42.0,
        )
        assert data["last_detection_sec"] == 42.0


# ── stale-snapshot surfacing ─────────────────────────────────────────────

class TestStaleSurfacing:
    def test_stale_state_age_is_visible(self, tmp_path, monkeypatch):
        live_file = tmp_path / "live.json"
        live_file.write_text(json.dumps({
            "updated": time.time() - 300,
            "states": {}, "telemetry": {}, "camera_health": {}, "fps": 0.0,
        }), encoding="utf-8")
        monkeypatch.setattr("app._LIVE_STATE_FILE", live_file)
        assert load_live_state() == {}
        age = live_state_age()
        assert age is not None and age >= 300

    def test_no_file_no_age(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app._LIVE_STATE_FILE", tmp_path / "nope.json")
        assert live_state_age() is None


# ── detector: noise-face guard + debug surface ───────────────────────────

class TestDetectorNoiseGuard:
    def _detector(self):
        det = ActivityDetector.__new__(ActivityDetector)
        det._face_registry = MagicMock()
        det._face_registry.threshold = 0.6
        det.model = MagicMock()
        det.conf = 0.5
        det._half = False
        det._device = "cpu"
        return det

    def test_tiny_face_skipped(self, caplog, monkeypatch):
        import logging
        import src.detector as d

        fake = MagicMock()
        fake.bbox = np.array([100, 100, 150, 158])  # area 2900 < LIVE_MIN_FACE_AREA
        fake.normed_embedding = np.zeros(512, dtype=np.float32)

        det = self._detector()
        det._face_registry.app.get.return_value = [fake]
        det._face_registry.identify.return_value = ("Unknown", 0.2)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with caplog.at_level(logging.DEBUG, logger="cctv.detector"):
            status, fboxes, flabels, phones, present = det._process_frame(frame)

        det._face_registry.identify.assert_not_called()
        assert fboxes == [] and flabels == []
        assert "Unknown" not in status
        assert any("skipped (smaller than LIVE_MIN_FACE_AREA" in r.message
                   for r in caplog.records)

    def test_live_guard_does_not_block_real_faces(self, caplog):
        import logging

        fake = MagicMock()
        fake.bbox = np.array([240, 140, 460, 480])  # area 74800
        fake.normed_embedding = np.zeros(512, dtype=np.float32)

        det = self._detector()
        det._face_registry.app.get.return_value = [fake]
        det._face_registry.identify.return_value = ("EMP001", 0.62)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with caplog.at_level(logging.DEBUG, logger="cctv.detector"):
            status, fboxes, flabels, phones, present = det._process_frame(frame)

        det._face_registry.identify.assert_called_once()
        assert flabels == [("EMP001", 0.62)]
        assert status["EMP001"]["present"] is True

    def test_face_debug_structured_line(self, caplog, monkeypatch):
        import logging
        import src.detector as d

        monkeypatch.setattr(d, "FACE_DEBUG", True)
        monkeypatch.setattr(d, "DETECTION_DEBUG", True)
        fake = MagicMock()
        fake.bbox = np.array([100, 100, 260, 280])
        fake.normed_embedding = np.zeros(512, dtype=np.float32)

        det = self._detector()
        det._face_registry.app.get.return_value = [fake]
        det._face_registry.identify.return_value = ("EMP001", 0.81)
        det._face_registry.identify_k.return_value = [("EMP001", 0.81),
                                                      ("Unknown", 0.2)]
        det._face_registry.employee_name.side_effect = lambda e: \
            "Yashmit Sharma" if e == "EMP001" else ""

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with caplog.at_level(logging.INFO, logger="cctv.detector"):
            det._process_frame(frame, cam_id="local_webcam")

        assert any("FACE_DEBUG camera=local_webcam" in r.message
                   and "candidate=EMP001" in r.message
                   and "name=Yashmit Sharma" in r.message
                   and "similarity=0.8100" in r.message
                   and "runner_up=Unknown" in r.message
                   and "decision=MATCH" in r.message for r in caplog.records)
        assert any("DETECTION_DEBUG camera=local_webcam" in r.message
                   for r in caplog.records)


# ── --source auto camera selection ───────────────────────────────────────

class TestAutoCameraIndex:
    def test_auto_picks_first_bright_device(self, monkeypatch):
        import main as m

        class FakeCap:
            def __init__(self, mean):
                self._mean = mean

            def isOpened(self):
                return True

            def read(self):
                frame = np.full((240, 320, 3), self._mean, dtype=np.uint8)
                return True, frame

            def release(self):
                pass

        monkeypatch.setattr(m.cv2, "VideoCapture",
                            lambda idx: FakeCap(200 if idx == 1 else 3))
        assert m._auto_detect_camera_index() == 1

    def test_auto_raises_when_all_dark(self, monkeypatch):
        import main as m

        class FakeCap:
            def isOpened(self):
                return True

            def read(self):
                return True, np.zeros((100, 100, 3), dtype=np.uint8)

            def release(self):
                pass

        monkeypatch.setattr(m.cv2, "VideoCapture", lambda idx: FakeCap())
        with pytest.raises(RuntimeError):
            m._auto_detect_camera_index(max_probe=3)

    def test_auto_skips_unopened_indices(self, monkeypatch):
        import main as m

        class FakeCap:
            def __init__(self, idx):
                self._idx = idx

            def isOpened(self):
                return self._idx != 1

            def read(self):
                return True, np.full((100, 100, 3), 100, dtype=np.uint8)

            def release(self):
                pass

        monkeypatch.setattr(m.cv2, "VideoCapture", FakeCap)
        assert m._auto_detect_camera_index() == 0


# ── config knobs sanity ──────────────────────────────────────────────────

def test_new_config_knobs_present_and_valid():
    import config
    assert config.LIVE_MIN_FACE_AREA >= 0
    assert config.BLACK_FRAME_MEAN >= 0
    assert isinstance(config.FACE_DEBUG, bool)