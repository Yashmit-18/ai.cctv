"""Tests for webcam-first daemon behaviour (Phase 2).

Pure logic only -- no real webcam is opened here.  ``main`` is imported for
its small helper functions; nothing starts the daemon or touches hardware.
"""

import json
import time

from main import (
    _local_camera_index,
    _local_source_message,
    _wait_for_initial_frames,
    _write_live_state,
)


def test_local_camera_index_mapping():
    assert _local_camera_index("webcam") == 0
    assert _local_camera_index("0") == 0
    assert _local_camera_index("1") == 1
    assert _local_camera_index("2") == 2
    assert _local_camera_index("laptop") == 0
    assert _local_camera_index("USB-CAM") == 0
    assert _local_camera_index("") == 0


def test_local_source_message_is_actionable():
    msg = _local_source_message("webcam")
    assert "Unable to open webcam" in msg
    assert "--source 1" in msg


class _NoFrames:
    online_count = 0
    active_camera_ids = []

    def get_latest_batch(self):
        return {}


class _Connected:
    online_count = 0
    active_camera_ids = ["local_webcam"]
    frame = object()

    def get_latest_batch(self):
        return {"local_webcam": self.frame}


class _FrameOnly:
    online_count = 1
    active_camera_ids = []
    frame = object()

    def get_latest_batch(self):
        return {"local_webcam": self.frame}


def test_wait_for_initial_frames_timeout(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    assert _wait_for_initial_frames(_NoFrames(), timeout=0.01) is False


def test_wait_for_initial_frames_succeeds_on_connect(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    assert _wait_for_initial_frames(_Connected(), timeout=0.01) is True


def test_wait_for_initial_frames_succeeds_on_first_frame(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    assert _wait_for_initial_frames(_FrameOnly(), timeout=0.01) is True


def test_wait_for_initial_frames_ignores_connected_but_frame_dry(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    # connected flag present but no frame AND no active backend -> keep waiting
    class _Odd:
        online_count = 1
        active_camera_ids = []

        def get_latest_batch(self):
            return {"cam1": None}

    assert _wait_for_initial_frames(_Odd(), timeout=0.01) is False


def test_write_live_state_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("main._LIVE_STATE_FILE", str(tmp_path / "live.json"))
    _write_live_state(
        {"EMP001": "ACTIVE"},
        {"EMP001": {"ACTIVE": 1.0}},
        {"local_webcam": {"health": "ONLINE"}},
        2.0,
        sources={"EMP001": "local_webcam"},
        durations={"EMP001": 30.0},
    )
    data = json.loads((tmp_path / "live.json").read_text(encoding="utf-8"))
    assert data["states"] == {"EMP001": "ACTIVE"}
    assert data["sources"] == {"EMP001": "local_webcam"}
    assert data["durations"] == {"EMP001": 30.0}
    assert data["camera_health"]["local_webcam"]["health"] == "ONLINE"
    assert data["fps"] == 2.0
    assert "updated" in data and "iso" in data