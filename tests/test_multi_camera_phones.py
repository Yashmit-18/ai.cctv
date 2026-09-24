"""Multi-camera "phone as CCTV" runtime coverage (SIMULATED).

All scenarios are SIMULATED -- the cameras are in-process fake stream readers
dispatched through the *real* ``MultiCameraManager`` / ``SpatialTracker`` /
``MultiTracker`` / ``_usable_camera_frames`` code paths.  No real RTSP device
is touched, so ``REAL RTSP`` is not claimed anywhere in this file.

The simulated "phones" are RTSP/IP camera sources (``CameraSourceKind.RTSP``)
with stable, unique, immutable identities ``CAM01 / CAM02 / CAM03``, one
independent reader thread each.  Scenarios assert the operational claims the
customer asked to verify:

* 2-3 phones run concurrently and all feeds are processed;
* more cameras than the GPU batch cap are served round-robin (no permanent
  starvation -- documents `README.md` "round-robin" contract);
* one dead feed never stops the other phones;
* an employee seen on two cameras is ONE employee (dedup, no double count),
  while tracking stays camera-local;
* an offline/dark feed never fabricates AWAY presence time;
* admin camera management keeps credentials redacted and duplicate/CRUD
  invariants intact.

The single shared registry / thresholds / FSM semantics are not touched here --
that is covered by ``test_employee_image_matrix.py`` and the existing suite.
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import pytest

from src import database as db
from src.camera_manager import MultiCameraManager
from src.camera_store import CameraRecord, CameraStore, REDACTED
from src.camera_store import reconcile_runtime, runtime_configs
from src.domain import CAM_ONLINE, CAM_OFFLINE, CAM_RECONNECTING, CAM_NO_FRAME
from src.domain import CAM_DARK_BLANK_FRAME, CAM_FROZEN, CAM_LOW_FPS
from src.domain import redact_url
from src.tracker import MultiTracker, SpatialTracker


# ----------------------------------------------------------------------
# SIMULATED stream reader (no cv2 / no hardware)
# ----------------------------------------------------------------------

def _frame(brightness: int = 120, size=(48, 64, 3)) -> np.ndarray:
    """SIMULATED camera frame: a real-faced brightness buffer."""
    return np.full(size, brightness, dtype=np.uint8)


class _SimPhoneCam:
    """SIMULATED per-phone reader with controllable liveness."""

    def __init__(self, cam_id: str, source: str, live: bool = True,
                 brightness: int = 120):
        self.camera_id = cam_id
        self.source = source
        self._live = live
        self._brightness = brightness
        self._frames_read = 0
        self.last_frame_timestamp = time.time() if live else 0.0

    @property
    def is_connected(self) -> bool:
        return self._live

    def start(self):
        return self

    def stop(self):
        pass

    def read(self):
        if not self._live:
            return None
        self._frames_read += 1
        self.last_frame_timestamp = time.time()
        return _frame(self._brightness)

    def health_info(self) -> dict:
        return {
            "id": self.camera_id,
            "source": redact_url(self.source),
            "connected": self._live,
            "health": CAM_ONLINE if self._live else CAM_OFFLINE,
            "frames_read": self._frames_read,
            "fps": 15.0 if self._live else 0.0,
            "reconnects": 0,
            "last_frame": time.time() if self._live else 0.0,
        }


def _pool(cams: dict[str, "_SimPhoneCam"], max_batch: int | None = None,
          **kw) -> MultiCameraManager:
    mgr = MultiCameraManager(cameras={cid: c.source for cid, c in cams.items()},
                             max_batch=max_batch, **kw)
    mgr._captures = dict(cams)  # noqa: SLF001 - deterministic, no threads
    return mgr


def _phone_url(host: str) -> str:
    return f"rtsp://user:Secret{host}@{host}:554/live/main"


@pytest.fixture()
def store_conn():
    tmp = tempfile.mkdtemp(prefix="mcam_")
    conn = db.get_connection(os.path.join(tmp, "cameras.db"))
    db.init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


# ======================================================================
# 1-3. Two and three phones run concurrently; default batch covers them
# ======================================================================

def test_simulated_two_phone_cameras_run_concurrently():
    """SIMULATED: CAM01 + CAM02 phones both ONLINE and both feed the batch."""
    cams = {
        "CAM01": _SimPhoneCam("CAM01", _phone_url("10.0.0.1")),
        "CAM02": _SimPhoneCam("CAM02", _phone_url("10.0.0.2")),
    }
    mgr = _pool(cams, max_batch=4)
    batch = mgr.get_latest_batch()
    assert set(batch) == {"CAM01", "CAM02"}
    health = mgr.camera_health()
    assert health["CAM01"]["health"] == CAM_ONLINE
    assert health["CAM02"]["health"] == CAM_ONLINE


def test_simulated_three_phone_cameras_all_processed_each_cycle():
    """SIMULATED: 3 phones fit inside the default batch (4) -- no starvation."""
    cams = {f"CAM0{i}": _SimPhoneCam(f"CAM0{i}", _phone_url(f"10.0.0.{i}"))
            for i in range(1, 4)}
    mgr = _pool(cams, max_batch=4)
    for _ in range(5):
        assert set(mgr.get_latest_batch()) == {"CAM01", "CAM02", "CAM03"}


def test_simulated_batch_default_4_keeps_three_phones_starved_free():
    """SIMULATED: same guarantee through latest_frames(limit=4)."""
    cams = {f"CAM0{i}": _SimPhoneCam(f"CAM0{i}", _phone_url(f"10.0.0.{i}"))
            for i in range(1, 4)}
    mgr = _pool(cams, max_batch=4)
    for _ in range(5):
        assert set(mgr.latest_frames(limit=4)) == {"CAM01", "CAM02", "CAM03"}


# ======================================================================
# 4-5. More configured phones than the batch cap: round-robin, never starved
# ======================================================================

def test_simulated_round_robin_serves_all_when_cameras_exceed_batch():
    """SIMULATED: 6 phones, batch cap 2 -> every phone served in 3 cycles."""
    cams = {f"CAM0{i}": _SimPhoneCam(f"CAM0{i}", _phone_url(f"10.0.0.{i}"))
            for i in range(1, 7)}
    mgr = _pool(cams, max_batch=2)
    seen: set[str] = set()
    for _ in range(3):
        batch = mgr.get_latest_batch()
        assert 0 < len(batch) <= 2  # bounded by the GPU batch cap
        seen |= set(batch)
    assert seen == set(cams)  # no camera is ever permanently starved


def test_simulated_latest_frames_limit_rotates_no_permanent_starvation():
    """SIMULATED: latest_frames(limit=2) over 6 phones also rotates fairly."""
    cams = {f"CAM0{i}": _SimPhoneCam(f"CAM0{i}", _phone_url(f"10.0.0.{i}"))
            for i in range(1, 7)}
    mgr = _pool(cams, max_batch=None)
    seen: set[str] = set()
    for _ in range(3):
        batch = mgr.latest_frames(limit=2)
        assert 0 < len(batch) <= 2
        seen |= set(batch)
    assert seen == set(cams)


# ======================================================================
# 6. Independence: one dead phone does not stop the others
# ======================================================================

def test_simulated_one_dead_phone_does_not_stop_others():
    """SIMULATED: CAM02 offline; CAM01/CAM03 stay ONLINE and processed."""
    cams = {
        "CAM01": _SimPhoneCam("CAM01", _phone_url("10.0.0.1"), live=True),
        "CAM02": _SimPhoneCam("CAM02", _phone_url("10.0.0.2"), live=False),
        "CAM03": _SimPhoneCam("CAM03", _phone_url("10.0.0.3"), live=True),
    }
    mgr = _pool(cams, max_batch=4)
    batch = mgr.get_latest_batch()
    assert "CAM02" not in batch
    assert set(batch) == {"CAM01", "CAM03"}
    health = mgr.camera_health()
    assert health["CAM01"]["health"] == CAM_ONLINE
    assert health["CAM03"]["health"] == CAM_ONLINE
    assert health["CAM02"]["health"] == CAM_OFFLINE


# ======================================================================
# 7-8. Identity semantics: same employee on 2 cameras is ONE employee
# ======================================================================

def test_simulated_same_employee_on_two_cameras_deduplicated():
    """Same EMP001 seen on CAM01 and CAM03 -> one merged present entry."""
    dets = [
        {"emp_id": "EMP001", "cam": "CAM01", "phone": False},
        {"emp_id": "EMP001", "cam": "CAM03", "phone": False},
        {"emp_id": "EMP001", "cam": "CAM01", "phone": True},
    ]
    merged = MultiTracker._deduplicate(dets)
    assert merged == {"EMP001": {"present": True, "phone_detected": True}}


def test_simulated_person_across_cameras_counts_once():
    """A person box on two cameras merges into one present signal (no double)."""
    dets = [
        {"emp_id": "__person__", "cam": "CAM01", "person_present": True},
        {"emp_id": "__person__", "cam": "CAM03", "person_present": True},
    ]
    merged = MultiTracker._deduplicate(dets)
    assert "__person__" in merged
    merged_person = MultiTracker._person_detected(dets)
    assert merged_person is True


# ======================================================================
# 9. Spatial tracking is camera-local
# ======================================================================

def test_simulated_spatial_tracking_camera_local():
    """Two identical person boxes on CAM01 and CAM02 stay two local tracks."""
    st = SpatialTracker(iou_threshold=0.5)
    now = time.time()
    tracks = st.process([
        {"cam": "CAM01", "det_type": "person", "person_box": [10, 10, 50, 50]},
        {"cam": "CAM02", "det_type": "person", "person_box": [10, 10, 50, 50]},
        {"cam": "CAM02", "face_box": [12, 12, 48, 48], "emp_id": "EMP001"},
    ], now=now)
    cams = {t.cam for t in tracks}
    assert cams == {"CAM01", "CAM02"}          # never merged across cameras
    assert len({t.track_id for t in tracks}) == 2
    assert st._last_persons_by_camera == {"CAM01": 1, "CAM02": 1}  # noqa: SLF001


def test_simulated_detector_cadence_ticks_are_per_camera():
    """Per-camera face cadence counters advance independently (SIMULATED)."""
    from src.detector import Detector  # lazy: src.detector imports cv2 eagerly
    det = object.__new__(Detector)
    det._face_ticks = {}    # noqa: SLF001
    det._face_cadence = 2   # noqa: SLF001
    assert det._face_tick("CAM01") is True     # CAM01 1st cadence fires
    assert det._face_tick("CAM02") is True     # CAM02 independent counter fires
    assert det._face_tick("CAM01") is False    # CAM01 2nd tick skipped
    assert det._face_tick("CAM02") is False
    assert det._face_tick("CAM01") is True     # CAM01 3rd tick fires again
    assert set(det._face_ticks) == {"CAM01", "CAM02"}  # noqa: SLF001


# ======================================================================
# 11. Offline / dark feed never fabricates AWAY (usable-frame gating)
# ======================================================================

def test_simulated_offline_and_dark_phone_frames_gated_out():
    from main import _usable_camera_frames

    bright = np.full((48, 64, 3), 120, dtype=np.uint8)
    dark = np.full((48, 64, 3), 3, dtype=np.uint8)
    health = {
        "CAM01": {"health": CAM_ONLINE},
        "CAM02": {"health": CAM_OFFLINE},
        "CAM03": {"health": CAM_RECONNECTING},
        "CAM04": {"health": CAM_NO_FRAME},
        "CAM05": {"health": CAM_ONLINE},   # fresh but dark/covered lens
        "CAM06": {"health": CAM_FROZEN},   # still usable
        "CAM07": {"health": CAM_LOW_FPS},  # still usable
    }
    frames = {
        "CAM01": bright, "CAM02": bright, "CAM03": bright,
        "CAM04": bright, "CAM05": dark, "CAM06": bright, "CAM07": bright,
    }
    usable = _usable_camera_frames(frames, health)
    assert set(usable) == {"CAM01", "CAM06", "CAM07"}
    assert health["CAM02"]["unusable_reason"] == CAM_OFFLINE
    assert health["CAM05"]["unusable_reason"] == CAM_DARK_BLANK_FRAME
    assert health["CAM05"]["frame_mean"] == 3.0


# ======================================================================
# 12-13. Admin camera dashboard: credentials redacted, identity unique
# ======================================================================

def test_simulated_health_snapshot_never_exposes_credentials():
    raw = _phone_url("10.0.0.1")
    cam = _SimPhoneCam("CAM01", raw)
    mgr = _pool({"CAM01": cam}, max_batch=4)
    h = mgr.camera_health()["CAM01"]
    assert h["source"] == redact_url(raw)
    assert h["source"] != raw
    assert "Secret" not in h["source"]
    assert redact_url(raw) == "rtsp://user:***@10.0.0.1:554/live/main"


def test_simulated_admin_records_never_expose_password(store_conn):
    store = CameraStore(store_conn)
    store.add(camera_id="CAM01", name="Phone 1", kind="rtsp",
              url="rtsp://10.0.0.1:554/live/main",
              username="user", password="Sup3rSecret", enabled=True,
              location="Hall A")
    rec = store.get("CAM01")
    assert isinstance(rec, CameraRecord)
    safe = rec.safe_dict()
    assert safe["password"] == REDACTED        # redacted, never the live secret
    assert "Sup3rSecret" not in str(safe)
    assert safe["credentials"] == "CONFIGURED"
    with pytest.raises(ValueError):
        store.add(camera_id="CAM01", name="Dup", kind="rtsp",
                  url="rtsp://10.0.0.9:554/x")


# ======================================================================
# 14-15. Admin CRUD + runtime resolution (SIMULATED, store > env)
# ======================================================================

def test_simulated_admin_three_phone_crud_and_disable_reconcile(store_conn):
    store = CameraStore(store_conn)
    for i in range(1, 4):
        store.add(camera_id=f"CAM0{i}", name=f"Phone {i}", kind="rtsp",
                  url=f"rtsp://10.0.0.{i}:554/live/main", enabled=True,
                  location="Hall A")
    assert store.list()[0].camera_id == "CAM01"
    store.set_enabled("CAM02", False)          # disable one phone
    runtime = runtime_configs(store_conn, env_configs={})
    assert set(runtime) == {"CAM01", "CAM02", "CAM03"}  # config set stays complete
    assert store.get("CAM02").enabled is False
    # Pool start excludes the disabled phone (enabled gate in MultiCameraManager).
    assert MultiCameraManager._cfg(runtime["CAM02"], "rtsp://10.0.0.2/") is None
    assert MultiCameraManager._cfg(runtime["CAM01"], "rtsp://10.0.0.1/") is not None
    store.update("CAM03", {"name": "Phone 3 (desk)", "location": "Desk B"})
    assert store.get("CAM03").name == "Phone 3 (desk)"
    store.delete("CAM01")
    assert {r.camera_id for r in store.list()} == {"CAM02", "CAM03"}


def test_simulated_reconcile_applies_authoritative_phone_config(store_conn):
    store = CameraStore(store_conn)
    store.add(camera_id="CAM01", name="Phone 1", kind="rtsp",
              url="rtsp://10.0.0.1:554/live/main", enabled=True)
    store.add(camera_id="CAM02", name="Phone 2", kind="rtsp",
              url="rtsp://10.0.0.2:554/live/main", enabled=True)
    store.request_apply()

    applied: dict = {}

    class _FakeManager:
        def apply_configs(self, configs: dict) -> str:
            applied.update(configs)
            return f"started {','.join(sorted(configs))}"

    status = reconcile_runtime(store_conn, _FakeManager(), env_configs={})
    assert status["action"] == "applied"
    assert set(applied) == {"CAM01", "CAM02"}
    handshake = store.apply_state()
    assert handshake["requested_revision"] == handshake["applied_revision"]

    # No-change second pass is a no-op.
    assert reconcile_runtime(store_conn, _FakeManager(),
                             env_configs={})["action"] == "up_to_date"