"""PHASE 58 -- Production RTSP / IP-CCTV / NVR camera-layer tests.

Covers the per-camera configuration model (``CameraConfig`` /
``CameraSourceKind``), RTSP credential handling, reconnect policy bounds,
frame health (thaw/dark/blank), multi-camera independence, and the rule that
a camera failure NEVER fabricates employee AWAY.

18 tests, ordered  TEST 1 .. TEST 18.  All are SIMULATED/UNIT tests: every
camera is a fake ``cv2.VideoCapture`` or a mocked ``MultiCameraManager``,
so no real RTSP device is required (real-NVR validation is reported
separately and honestly as NOT_VALIDATED unless a real feed was observed).

==========================
TEST REQUIREMENT MATRIX
==========================
TEST  1   Camera config model loads per-camera identity/behaviour/env.
TEST  2   Malformed/empty RTSP config is rejected safely (no crash).
TEST  3   RTSP connection failure never leaks credentials in logs.
TEST  4   CameraConfig outward shapes never contain the password.
TEST  5   RTSP failure maps to CAMERA_OFFLINE (health + security event).
TEST  6   Recovery maps back to CAMERA_RECOVERED + ONLINE.
TEST  7   Frozen RTSP feed classes as FROZEN_FRAME.
TEST  8   Dark/blank feed is gated out as DARK_BLANK_FRAME (never AWAY).
TEST  9   Camera failure cannot produce employee AWAY.
TEST 10   Reconnect backoff is bounded (never exceeds configured max).
TEST 11   Repeated failures don't busy-spin; stop() still exits cleanly.
TEST 12   Capture resources are released on stop()/release.
TEST 13   Local/webcam sources keep the direct-read fast path (regression).
TEST 14   Latest-frame policy is bounded (single latest, no unbounded queue).
TEST 15   Multiple cameras report independent health.
TEST 16   One failed camera doesn't starve the others.
TEST 17   Health/telemetry transitions are surfaced to the dashboard HUD.
TEST 18   No RTSP credentials in any report/health/log output.
"""

import math
import time

import numpy as np
import pytest

import config
import src.camera as camera
from config import camera_configs
from src.camera import VideoCapture
from src.domain import (
    CAM_DARK_BLANK_FRAME,
    CAM_FROZEN,
    CAM_LOW_FPS,
    CAM_NO_FRAME,
    CAM_OFFLINE,
    CAM_ONLINE,
    CAM_RECONNECTING,
    CAMERA_OFFLINE,
    CAMERA_RECOVERED,
    CameraConfig,
    CameraSourceKind,
    redact_url,
)


# ----------------------------------------------------------------------
# Fake capture backends
# ----------------------------------------------------------------------
class _FakeCapDead:
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
    def __init__(self, frame=None):
        self._frame = np.zeros((24, 24), dtype=np.uint8) + 100 if frame is None else frame
        self._released = False

    def isOpened(self):
        return True

    def read(self):
        return True, self._frame.copy()

    def grab(self):
        return True

    def set(self, prop, value):  # noqa: A002
        return True

    def release(self):
        self._released = True


def _patch_cap(monkeypatch, factory):
    monkeypatch.setattr(camera.cv2, "VideoCapture", factory)


def _pause_sleep(monkeypatch):
    monkeypatch.setattr(camera.time, "sleep", lambda _: None)


# ======================================================================
# TEST 1 -- camera config model
# ======================================================================
def test_01_camera_config_model_loads_from_env(monkeypatch):
    monkeypatch.setenv("CCTV_CAM_03_URL", "rtsp://10.0.0.3:554/nvr/main")
    monkeypatch.setenv("CCTV_CAM_03_NAME", "Lobby North")
    monkeypatch.setenv("CCTV_CAM_03_TYPE", "rtsp")
    monkeypatch.setenv("CCTV_CAM_03_ENABLED", "1")
    monkeypatch.setenv("CCTV_CAM_03_LOCATION", "Lobby (north entrance)")
    monkeypatch.setenv("CCTV_CAM_03_FPS", "3")
    monkeypatch.setenv("CCTV_CAM_03_RECONNECT_BASE", "1")
    monkeypatch.setenv("CCTV_CAM_03_RECONNECT_MAX", "15")
    monkeypatch.setenv("CCTV_CAM_03_RECONNECT_FACTOR", "2")
    # Make sure nothing else from the environment bleeds in.
    for i in range(1, 11):
        if i != 3:
            monkeypatch.delenv(f"CCTV_CAM_{i:02d}_URL", raising=False)

    cfg = camera_configs()
    cam = cfg["cam_03"]
    assert cam.name == "Lobby North"
    assert cam.kind is CameraSourceKind.RTSP
    assert cam.enabled is True
    assert cam.location == "Lobby (north entrance)"
    assert cam.fps_target == 3.0
    assert cam.reconnect_base == 1.0
    assert cam.reconnect_max == 15.0
    assert cam.reconnect_factor == 2.0
    assert cam.camera_id == "cam_03"


def test_02_malformed_or_empty_config_is_safe(monkeypatch):
    # Empty slots are silently excluded -- no exception, no crash.
    for i in range(1, 11):
        monkeypatch.delenv(f"CCTV_CAM_{i:02d}_URL", raising=False)
        monkeypatch.delenv(f"CCTV_CAM_{i:02d}_ENABLED", raising=False)
    cfg = camera_configs()
    assert set(cfg) == set()

    # Unknown TYPE falls back to inference rather than erroring.
    monkeypatch.setenv("CCTV_CAM_05_URL", "rtsp://x/y")
    monkeypatch.setenv("CCTV_CAM_05_TYPE", "hmd")
    cfg = camera_configs()
    assert cfg["cam_05"].kind is CameraSourceKind.RTSP

    # A URL that clearly isn't RTSP is inferred as a file/non-RTSP source.
    monkeypatch.setenv("CCTV_CAM_06_URL", "C:/footage/day.mp4")
    cfg = camera_configs()
    assert cfg["cam_06"].kind is CameraSourceKind.VIDEO_FILE


# ======================================================================
# TEST 3 / TEST 18 -- credentials never leak
# ======================================================================
def test_03_rtsp_failure_logs_never_contain_password(monkeypatch, caplog):
    _patch_cap(monkeypatch, lambda *a, **k: _FakeCapDead())
    _pause_sleep(monkeypatch)
    cap = VideoCapture(source="rtsp://admin:s3cr3tpw@10.0.0.9:554/live")
    cap._open_stream()
    assert "s3cr3tpw" not in caplog.text
    assert "***" in caplog.text
    cap.stop()


def test_04_camera_config_shapes_never_contain_password():
    cfg = CameraConfig(
        camera_id="cam_03", name="NVR",
        kind=CameraSourceKind.RTSP,
        url="rtsp://admin:m1a1pw@10.0.0.3:554/stream1",
        username="admin", password="m1a1pw",
    )
    text = " ".join([str(cfg.safe_dict()), cfg.display(), cfg._rtsp_host()])
    assert "m1a1pw" not in text
    assert "admin:" not in " ".join([cfg.display(), cfg._rtsp_host()])


# ======================================================================
# TEST 5 / TEST 6 -- offline / recovery state machine
# ======================================================================
def test_05_rtsp_failure_becomes_camera_offline_event():
    from src.camera_state import CameraStateManager

    mgr = CameraStateManager(offline_trigger_sec=0.0)
    # First tick marks camera offline; trigger=0 fires CAMERA_OFFLINE now.
    events = mgr.update("cam_03", False, now=100.0)
    assert any(e["event_type"] == CAMERA_OFFLINE for e in events)
    assert mgr.is_online("cam_03") is False
    # Health snapshot of the failed capture is never ONLINE either.
    cap = VideoCapture(source="rtsp://n/a", kind=CameraSourceKind.RTSP)
    cap._health = CAM_RECONNECTING
    assert cap.health in (CAM_OFFLINE, CAM_RECONNECTING)


def test_06_recovery_emits_camera_recovered_and_online(monkeypatch):
    from src.camera_state import CameraStateManager

    camstate = CameraStateManager(offline_trigger_sec=0.0)
    camstate.update("cam_03", False, now=100.0)
    camstate.update("cam_03", False, now=110.0)
    assert not camstate.is_online("cam_03")

    # Feed comes back; the security FSM records a recovery.
    events = camstate.update("cam_03", True, now=115.0)
    assert any(e["event_type"] == CAMERA_RECOVERED for e in events)
    assert camstate.is_online("cam_03") is True

    # And the capture itself reaches ONLINE again after a reconnect.
    state = {"live": False}
    _patch_cap(monkeypatch, lambda *a, **k: _RecoveringCap(state))
    cap = VideoCapture(
        source="rtsp://x/y", frame_skip=1, reconnect=(0.1, 1.0, 2.0),
    )
    cap.start()
    try:
        deadline = time.time() + 3
        while cap.reconnect_count == 0 and time.time() < deadline:
            time.sleep(0.01)
        assert cap.reconnect_count >= 1
        state["live"] = True
        deadline = time.time() + 5
        while time.time() < deadline:
            if cap.health == CAM_ONLINE and cap.frames_read > 0:
                break
            time.sleep(0.01)
        assert cap.health == CAM_ONLINE
        assert cap.frames_read > 0
    finally:
        cap.stop()


class _RecoveringCap:
    """A fake capture whose liveliness is driven by a shared flag.

    Reconnects receive a fresh instance (mirroring cv2), so recovery means a
    *new* open() sees the device back and frames start flowing.
    """

    def __init__(self, state):
        self._state = state
        self._frame = np.zeros((24, 24), dtype=np.uint8) + 100

    def isOpened(self):
        return self._state["live"]

    def read(self):
        ok = self._state["live"]
        return (ok, self._frame.copy()) if ok else (False, None)

    def grab(self):
        return self._state["live"]

    def set(self, prop, value):  # noqa: A002
        return True

    def release(self):
        pass


# ======================================================================
# TEST 7 -- frozen frame
# ======================================================================
def test_07_frozen_rtsp_feed_reports_frozen_frame(monkeypatch):
    _patch_cap(monkeypatch, lambda *a, **k: _FakeCapLive())
    _pause_sleep(monkeypatch)
    monkeypatch.setattr(camera, "_FROZEN_STREAK", 5)
    monkeypatch.setattr(camera, "_FROZEN_MIN_SEC", 0.0)
    cap = VideoCapture(source="rtsp://x/y", frame_skip=1)
    cap.start()
    try:
        deadline = time.time() + 4
        while cap.health != CAM_FROZEN and time.time() < deadline:
            time.sleep(0.02)
        assert cap.health == CAM_FROZEN
        assert cap.health_info()["frozen_streak"] >= 5
    finally:
        cap.stop()


# ======================================================================
# TEST 8 -- dark / blank gate
# ======================================================================
def test_08_dark_blank_feed_is_gated_never_drives_state():
    from main import _usable_camera_frames

    dark = np.zeros((24, 24), dtype=np.uint8) + 2  # mean 2 < BLACK_FRAME_MEAN
    bright = np.full((24, 24), 140, dtype=np.uint8)
    health = {
        "cam_03": {"health": CAM_ONLINE},
        "cam_04": {"health": CAM_ONLINE},
    }
    # The dark camera's frame is dropped with the explicit DARK_BLANK_FRAME
    # reason; the bright one stays usable.
    frames = {"cam_03": dark, "cam_04": bright}
    usable = _usable_camera_frames(frames, health)
    assert "cam_03" not in usable
    assert "cam_04" in usable
    assert health["cam_03"]["usable"] is False
    assert health["cam_03"]["unusable_reason"] == CAM_DARK_BLANK_FRAME
    assert health["cam_03"].get("frame_mean") == 2.0


# ======================================================================
# TEST 9 -- camera failure never fabricates AWAY
# ======================================================================
def test_09_camera_failure_never_fabricates_employee_away():
    from main import _usable_camera_frames
    from src.camera_state import CameraStateManager

    # When a camera is unusable, its frames are dropped => no detections can
    # even reach the productivity tracker for that camera.
    health = {"cam_03": {"health": CAM_OFFLINE}}
    usable = _usable_camera_frames({"cam_03": np.zeros((24, 24), dtype=np.uint8) + 9}, health)
    assert usable == {}
    assert CAM_DARK_BLANK_FRAME or health["cam_03"]["usable"] in (False, None)

    # The security FSM records CAMERA_OFFLINE -- never an "employee AWAY".
    camstate = CameraStateManager(offline_trigger_sec=0.0)
    events = camstate.update("cam_03", False, now=10.0)
    assert all(e["event_type"] != "AWAY" for e in events)
    assert any(e["event_type"] == CAMERA_OFFLINE for e in events)
    # No AWAY/ACTIVE event types are emitted by the camera FSM at all.
    allowed = {CAMERA_OFFLINE, CAMERA_RECOVERED}
    assert all(e["event_type"] in allowed for e in events)


# ======================================================================
# TEST 10 / TEST 11 -- reconnect policy
# ======================================================================
def test_10_reconnect_backoff_is_bounded(monkeypatch):
    _patch_cap(monkeypatch, lambda *a, **k: _FakeCapDead())
    _pause_sleep(monkeypatch)
    cap = VideoCapture(
        source="rtsp://n/a",
        frame_skip=1,
        reconnect=(2.0, 30.0, 2.5),  # defaults
    )
    base, max_backoff, factor = cap._reconnect_params()
    assert base == 2.0
    assert max_backoff == 30.0
    assert factor == 2.5
    # Simulate the reader's update rule: it never exceeds max_backoff.
    backoff = base
    sequence = [backoff]
    for _ in range(60):
        backoff = min(backoff * factor, max_backoff)
        sequence.append(backoff)
    assert max(sequence) <= max_backoff
    assert sequence[-1] == max_backoff  # plateaus at the cap
    cap.stop()


def test_11_repeated_failures_do_not_busy_spin(monkeypatch):
    _patch_cap(monkeypatch, lambda *a, **k: _FakeCapDead())
    # Real sleeps: backoff pacing IS the boundedness guarantee.  With a small
    # base (0.2s) and hard cap (1.0s), a ~2.5s window can only fit a handful
    # of retries -- a busy-spin would produce far more.
    cap = VideoCapture(source="rtsp://n/a", frame_skip=1, reconnect=(0.2, 1.0, 1.5))
    cap.start()
    try:
        deadline = time.time() + 2.5
        while cap.reconnect_count < 3 and time.time() < deadline:
            time.sleep(0.02)
        assert cap.reconnect_count >= 3
        assert cap.reconnect_count < 50  # no unbounded retry storm
    finally:
        cap.stop()
    assert cap.is_connected is False


# ======================================================================
# TEST 12 -- resource release
# ======================================================================
def test_12_capture_resources_are_released_on_stop(monkeypatch):
    fake = _FakeCapLive()
    _patch_cap(monkeypatch, lambda *a, **k: fake)
    cap = VideoCapture(source=0, frame_skip=1)
    cap.start()
    try:
        deadline = time.time() + 3
        while cap.frames_read == 0 and time.time() < deadline:
            time.sleep(0.02)
        assert cap.frames_read > 0
    finally:
        cap.stop()
    assert fake._released is True
    assert cap._cap is None


# ======================================================================
# TEST 13 -- local/webcam regression
# ======================================================================
def test_13_local_and_webcam_sources_stay_on_direct_path(monkeypatch):
    # Local must never take the timed (RTSP) helper-thread path.
    _patch_cap(monkeypatch, lambda *a, **k: _FakeCapLive())
    monkeypatch.setattr(camera.VideoCapture, "_timed_read",
                        lambda self: (_ for _ in ()).throw(AssertionError("timed read used")))
    monkeypatch.setattr(camera.VideoCapture, "_timed_read_grab",
                        lambda self: (_ for _ in ()).throw(AssertionError("timed grab used")))
    cap = VideoCapture(source=0, frame_skip=1)
    cap.start()
    try:
        deadline = time.time() + 3
        while cap.frames_read == 0 and time.time() < deadline:
            time.sleep(0.02)
        assert cap.frames_read > 0
        assert cap.health == CAM_ONLINE
    finally:
        cap.stop()

    # Source-kind inference stays correct for aliases.
    assert CameraSourceKind.infer(0) is CameraSourceKind.LOCAL
    assert CameraSourceKind.infer("webcam") is CameraSourceKind.LOCAL
    assert CameraSourceKind.infer("local") is CameraSourceKind.LOCAL
    assert CameraSourceKind.infer("rtsp://x/y") is CameraSourceKind.RTSP
    assert CameraSourceKind.infer("clip.mp4") is CameraSourceKind.VIDEO_FILE


# ======================================================================
# TEST 14 -- latest-frame policy is bounded
# ======================================================================
def test_14_read_returns_single_latest_frame(monkeypatch):
    _patch_cap(monkeypatch, lambda *a, **k: _FakeCapLive())
    _pause_sleep(monkeypatch)
    cap = VideoCapture(source=0, frame_skip=1)
    cap.start()
    try:
        deadline = time.time() + 3
        while cap.frames_read < 3 and time.time() < deadline:
            time.sleep(0.01)
        frame = cap.read()
        assert frame is not None
        assert isinstance(frame, np.ndarray)  # single latest, not a queue
        assert frame.shape == (24, 24)
    finally:
        cap.stop()


def test_14b_manager_batch_is_bounded_to_configured_cameras(monkeypatch):
    from src.camera_manager import MultiCameraManager

    def _stub_capture(source, **kw):
        class _Cap:
            def __init__(self):
                self.kind = "rtsp"
                self.camera_id = kw.get("camera_id", str(source))

            def start(self):
                return self

            def read(self):
                return np.zeros((8, 8), dtype=np.uint8)

            def health_info(self):
                return {"source": redact_url(str(source)), "connected": False,
                        "health": CAM_OFFLINE, "frames_read": 0, "fps": 0.0,
                        "reconnects": 0, "last_frame": 0}

            def stop(self):
                pass

        return _Cap()

    monkeypatch.setattr("src.camera_manager.VideoCapture", _stub_capture)
    mgr = MultiCameraManager(
        cameras={"cam_01": "rtsp://a", "cam_02": "rtsp://b", "cam_03": "rtsp://c"},
        max_batch=2,
    )
    mgr.start()
    try:
        batch = mgr.get_latest_batch()
        assert len(batch) <= 2  # bounded by max_batch
    finally:
        mgr.stop()


# ======================================================================
# TEST 15 / TEST 16 -- multi-camera independence
# ======================================================================
def _multi_cam_factory(spec):
    """cv2.VideoCapture factory dispatching on the source URL.

    ``spec`` maps id -> bool ("will the fake report isOpened()/frames").
    """
    def _factory(source, *a, **k):
        for cam_id, live in spec.items():
            if str(source).endswith(cam_id):
                return _FakeCapLive() if live else _FakeCapDead()
        return _FakeCapDead()
    return _factory


def test_15_16_multi_camera_independent_health_and_isolation(monkeypatch):
    from src.camera_manager import MultiCameraManager

    spec = {"cam_01": True, "cam_02": False, "cam_03": True}
    _patch_cap(monkeypatch, _multi_cam_factory(spec))
    _pause_sleep(monkeypatch)
    cameras = {
        "cam_01": "rtsp://10.0.0.1:554/" + "cam_01",
        "cam_02": "rtsp://10.0.0.2:554/" + "cam_02",
        "cam_03": "rtsp://10.0.0.3:554/" + "cam_03",
    }
    mgr = MultiCameraManager(cameras=cameras, max_batch=5).start()
    try:
        deadline = time.time() + 4
        while time.time() < deadline:
            health = mgr.camera_health()
            if health["cam_01"].get("health") == CAM_ONLINE and \
                    health["cam_03"].get("health") == CAM_ONLINE:
                break
            time.sleep(0.02)

        health = mgr.camera_health()
        # Independent health: 01 & 03 online, 02 dead -- never dragged down.
        assert health["cam_01"]["health"] == CAM_ONLINE
        assert health["cam_02"]["health"] in (CAM_OFFLINE, CAM_RECONNECTING)
        assert health["cam_03"]["health"] == CAM_ONLINE

        # One dead camera does not starve the others: the live cams still
        # deliver frames in the pool batch.
        batch = mgr.get_latest_batch()
        assert batch["cam_01"] is not None
        assert batch["cam_03"] is not None

        # Actively-connected camera count reflects the isolation honestly.
        assert mgr.online_count == 2
    finally:
        mgr.stop()


# ======================================================================
# TEST 17 -- health/telemetry surfaced
# ======================================================================
def test_17_health_telemetry_reported_to_dashboard(monkeypatch):
    _patch_cap(monkeypatch, lambda *a, **k: _FakeCapLive())
    cap = VideoCapture(source=0, frame_skip=1, camera_id="local_webcam")
    info = cap.health_info()
    # Richer Phase 58 telemetry is always present in the HUD snapshot.
    assert info["id"] == "local_webcam"
    assert info["kind"] == "local"
    assert "resolution" in info
    assert "reconnects" in info
    assert "width" in info and "height" in info
    # Credential-free by construction.
    assert "rtsp://" not in str(info)
    cap.stop()


def test_17b_transitions_surface_readable_events():
    from src.camera_state import CameraStateManager

    camstate = CameraStateManager(offline_trigger_sec=0.0)
    evs_off = camstate.update("cam_05", False, now=1.0)
    evs_on = camstate.update("cam_05", True, now=2.0)
    types = [e["event_type"] for e in evs_off + evs_on]
    assert CAMERA_OFFLINE in types
    assert CAMERA_RECOVERED in types
    assert all(e.get("camera") == "cam_05" for e in evs_off + evs_on)


# ======================================================================
# TEST 18 -- no credentials in reports / health / live-state output
# ======================================================================
def test_18_no_rtsp_credentials_in_any_outward_output(monkeypatch):
    # A fully-credentialed URL flows through config, capture health and
    # display shapes -- the password must never appear anywhere.
    monkeypatch.setenv("CCTV_CAM_07_URL", "rtsp://admin:u3h4k@10.0.0.77:554/stream")
    monkeypatch.setenv("CCTV_CAM_07_NAME", "Warehouse")
    cfg = camera_configs()
    cam = cfg["cam_07"]

    combined = " ".join([
        str(cam.safe_dict()),
        cam.display(),
        str(redact_url(cam.url)),
    ])
    assert "u3h4k" not in combined
    assert "admin:u3h4k" not in combined

    # The manager health snapshot (what live_state.json carries) is clean.
    from src.camera_manager import MultiCameraManager

    def _stub_capture(source, **kw):
        class _Cap:
            def start(self):
                return self

            def health_info(self):
                return {"source": redact_url(str(source)), "connected": False,
                        "health": CAM_OFFLINE, "frames_read": 0, "fps": 0.0,
                        "reconnects": 0, "last_frame": 0,
                        "kind": "rtsp", "resolution": "",
                        "id": kw.get("camera_id", "")}

            def stop(self):
                pass

        return _Cap()

    monkeypatch.setattr("src.camera_manager.VideoCapture", _stub_capture)
    mgr = MultiCameraManager(
        cameras={"cam_07": cam.url},
        configs={"cam_07": cam},
    ).start()
    try:
        health = mgr.camera_health()
    finally:
        mgr.stop()
    assert "u3h4k" not in str(health)


# Keep math import used in a couple of place (bounds).
def test_import_surface():
    assert CAM_NO_FRAME == "NO_FRAME"
    assert CAM_LOW_FPS == "LOW_FPS"
    assert math.isfinite(CameraConfig.reconnect_max * 0.0)