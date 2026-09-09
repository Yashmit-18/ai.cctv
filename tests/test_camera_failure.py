"""Camera failure / local fast-path behaviour (Phase 3, 15, 21).

Uses a fake ``cv2.VideoCapture`` so no real device is required.  Verifies
that a failing source reports health correctly, never blocks, stops safely,
and that local (non-RTSP) sources take the direct read path rather than the
per-read helper thread.
"""

import time

import numpy as np

import src.camera as camera
from src.camera import VideoCapture


class _FakeCapDead:
    """A capture backend that is never open."""

    def isOpened(self):
        return False

    def read(self):
        return False, None

    def grab(self):
        return False

    def set(self, prop, value):  # noqa: A002
        return False

    def release(self):
        pass


class _FakeCapLive:
    """A capture backend that always yields a fresh frame."""

    def __init__(self):
        self._frame = np.zeros((24, 24), dtype=np.uint8) + 100

    def isOpened(self):
        return True

    def read(self):
        return True, self._frame.copy()

    def grab(self):
        return True

    def set(self, prop, value):  # noqa: A002
        return True

    def release(self):
        pass


def _patch_cap(monkeypatch, fake):
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *a, **k: fake)


def test_invalid_source_never_online(monkeypatch):
    _patch_cap(monkeypatch, _FakeCapDead())
    monkeypatch.setattr(time, "sleep", lambda _: None)
    cap = VideoCapture(source=99, frame_skip=1)
    assert cap.health == "OFFLINE"
    cap.start()
    try:
        deadline = time.time() + 2
        while cap.reconnect_count == 0 and time.time() < deadline:
            pass
        time.sleep(0.01)
        assert cap.reconnect_count >= 1
        assert cap.is_connected is False
        assert cap.health != "ONLINE"
        assert cap.health in ("RECONNECTING", "OFFLINE")
        assert cap.read() is None
    finally:
        cap.stop()
    assert cap.health in ("RECONNECTING", "OFFLINE")


def test_local_source_uses_direct_read_no_helper_thread(monkeypatch):
    _patch_cap(monkeypatch, _FakeCapLive())
    # The timed helpers must never run for a local (webcam) source.
    monkeypatch.setattr(VideoCapture, "_timed_read",
                        lambda self: (_ for _ in ()).throw(AssertionError("timed read used")))
    monkeypatch.setattr(VideoCapture, "_timed_read_grab",
                        lambda self: (_ for _ in ()).throw(AssertionError("timed grab used")))
    cap = VideoCapture(source=0, frame_skip=1)
    cap.start()
    try:
        deadline = time.time() + 3
        while cap.frames_read == 0 and time.time() < deadline:
            time.sleep(0.02)
        assert cap.frames_read > 0
        assert cap.is_connected is True
        assert cap.health == "ONLINE"
        assert cap.read() is not None
    finally:
        cap.stop()


def test_local_detection():
    assert VideoCapture(source=0)._is_local() is True
    assert VideoCapture(source="webcam")._is_local() is True
    assert VideoCapture(source="1")._is_local() is True
    assert VideoCapture(source="rtsp://x/y")._is_local() is False
    assert VideoCapture(source="video.mp4")._is_local() is False


def test_stop_is_safe_before_start():
    cap = VideoCapture(source=0)
    cap.stop()  # must not raise


def test_unopened_health_is_offline():
    assert VideoCapture(source=999).health == "OFFLINE"


def test_rtsp_credentials_are_never_logged(monkeypatch, caplog):
    # Regression: raw RTSP URLs (user:password@) must never reach the log.
    _patch_cap(monkeypatch, _FakeCapDead())
    cap = VideoCapture(source="rtsp://admin:s3cr3tpw@192.168.1.50:554/live")
    ok = cap._open_stream()
    assert ok is False
    assert "s3cr3tpw" not in caplog.text
    assert "***" in caplog.text


def test_health_info_carries_fps(monkeypatch):
    # Per-camera FPS must be part of the health snapshot used by the dashboard.
    _patch_cap(monkeypatch, _FakeCapLive())
    cap = VideoCapture(source=0, frame_skip=1)
    info = cap.health_info()
    assert "fps" in info
    assert info["health"] == "OFFLINE"  # unopened until start()


def test_camera_manager_redacts_rtsp_credentials(monkeypatch):
    # The dashboard-facing health API redacts RTSP credentials at the boundary.
    from src.camera_manager import MultiCameraManager

    def _stub_capture(source):
        class _Cap:
            def __init__(self):
                self.source = source

            def start(self):
                return self

            def health_info(self):
                return {"source": str(self.source), "connected": False,
                        "health": "OFFLINE", "frames_read": 0, "fps": 0.0,
                        "reconnects": 0, "last_frame": 0}

            def stop(self):
                pass

        return _Cap()

    monkeypatch.setattr("src.camera_manager.VideoCapture", _stub_capture)
    mgr = MultiCameraManager(cameras={"cam_99": "rtsp://admin:hunter2@192.168.1.99:554/live"})
    mgr.start()
    try:
        health = mgr.camera_health()
    finally:
        mgr.stop()
    assert health["cam_99"]["health"] == "OFFLINE"
    assert "hunter2" not in health["cam_99"]["source"]
    assert "***" in health["cam_99"]["source"]


class _RecoveringCap:
    """A fake capture whose availability is driven by a shared state flag.

    Each reconnect attempt gets a *fresh* instance (mirroring ``cv2``), and
    the device "comes back" once the shared flag flips to live.
    """

    def __init__(self, state):
        self._state = state
        self._frame = np.zeros((24, 24), dtype=np.uint8) + 100

    def isOpened(self):
        return self._state["live"]

    def read(self):
        ok = self._state["live"]
        return (ok, self._frame.copy()) if ok else (False, None)

    def set(self, prop, value):  # noqa: A002
        return True

    def release(self):
        pass


def test_recovery_after_outage_reaches_online(monkeypatch):
    """27.8: a source that fails then comes back must reach ONLINE again."""
    state = {"live": False}
    monkeypatch.setattr(camera.cv2, "VideoCapture",
                        lambda *a, **k: _RecoveringCap(state))
    monkeypatch.setattr(camera.time, "sleep", lambda _: None)
    cap = VideoCapture(source=7, frame_skip=1)
    cap.start()
    try:
        # Dead phase: reader should hit the reconnect loop (reconnect_count>0).
        deadline = time.time() + 3
        while cap.reconnect_count == 0 and time.time() < deadline:
            time.sleep(0.02)
        assert cap.reconnect_count >= 1
        assert cap.health != "ONLINE"

        # Device comes back; the reconnect loop should bring us ONLINE with
        # new frames flowing (recovery, not a one-shot fluke).
        state["live"] = True
        deadline = time.time() + 5
        while time.time() < deadline:
            if cap.health == "ONLINE" and cap.frames_read > 0:
                break
            time.sleep(0.02)
        assert cap.health == "ONLINE"
        assert cap.frames_read > 0
    finally:
        cap.stop()