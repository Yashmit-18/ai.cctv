"""Phase 65 -- Streamlit Cloud stability + premium SOC UI.

Covers the cloud crash fixed by the lazy camera-runtime boundary:

* ``src.camera_store`` and its pure store operations import and run without
  ``cv2`` (simulating Streamlit Cloud, where the GUI OpenCV build is broken);
* ``test_camera_connection`` reports ``RUNTIME_UNAVAILABLE`` (never a crash,
  never a leaked credential) when the capture runtime cannot be imported;
* the real local capture path still works when the runtime IS available;
* the Cameras page renders in cloud mode and its Test-Connection handler shows
  the cloud-unavailable message instead of raising;
* every navigation page renders without an exception when ``cv2`` is blocked.

The simulation blocks ``import cv2`` at the interpreter level (the Cloud image
fails inside OpenCV with a missing ``libGL``) and evicts already-imported
camera runtime modules so the lazy import genuinely re-runs.
"""

from __future__ import annotations

import builtins
import os
import sys
import tempfile

import pytest

from src import camera_store as cs
from src.camera_store import CameraStore, test_camera_connection as tc_probe
from src import database as db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")


# ----------------------------------------------------------------------
# Cloud simulation helpers
# ----------------------------------------------------------------------

_RUNTIME_MODULES = ("cv2", "src.camera", "src.camera_manager")


def _pop_runtime_modules() -> dict:
    """Temporarily remove already-imported camera runtime modules."""
    saved = {}
    for key in _RUNTIME_MODULES:
        if key in sys.modules:
            saved[key] = sys.modules.pop(key)
    return saved


def _restore_runtime_modules(saved: dict) -> None:
    for key, value in saved.items():
        sys.modules[key] = value


def _block_cv2(monkeypatch) -> None:
    """Block any future ``import cv2`` (raises ImportError, like Cloud)."""
    real_import = builtins.__import__

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cv2" or name.startswith("cv2."):
            raise ImportError(
                "simulated cloud: OpenCV unavailable (GUI build needs libGL)")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked)


def _enter_cloud(monkeypatch) -> dict:
    saved = _pop_runtime_modules()
    _block_cv2(monkeypatch)
    return saved


def _fake_capture(runtime_ok: bool = True):
    """Minimal VideoCapture stand-in (same seams as Phase 58/61 tests)."""

    class _FakeCap:
        def __init__(self, source, kind=None, camera_id=None,
                     reconnect=(2.0, 30.0, 2.5)):
            self.source = source
            self.kind = kind
            self.camera_id = camera_id

        @property
        def is_connected(self):
            return runtime_ok

        def start(self):
            return self

        def stop(self):
            return None

        def health_info(self):
            if not runtime_ok:
                return {"connected": False, "health": "OFFLINE",
                        "resolution": "", "width": 0, "height": 0,
                        "fps": 0.0, "frames_read": 0, "reconnects": 0}
            return {"connected": True, "health": "ONLINE", "resolution": "1280x720",
                    "width": 1280, "height": 720, "fps": 15.0,
                    "frames_read": 1, "reconnects": 0}

    return _FakeCap


def _db_conn(path):
    conn = db.get_connection(path)
    db.init_db(conn)
    return conn


# ----------------------------------------------------------------------
# Store operations never need the capture runtime
# ----------------------------------------------------------------------

def test_camera_store_pure_ops_never_touch_cv2(monkeypatch):
    saved = _enter_cloud(monkeypatch)
    try:
        assert cs.REDACTED == "********"
        assert cs._RUNTIME_UNAVAILABLE == "RUNTIME_UNAVAILABLE"
        store_cls = cs.CameraStore
        assert store_cls is CameraStore
    finally:
        _restore_runtime_modules(saved)


def test_camera_store_crud_works_with_cv2_blocked(monkeypatch):
    saved = _enter_cloud(monkeypatch)
    try:
        tmp = tempfile.mkdtemp(prefix="p65_")
        conn = _db_conn(os.path.join(tmp, "cameras.db"))
        try:
            store = CameraStore(conn)
            rec = store.add(camera_id="CLOUD01", name="Cloud Cam",
                            kind="rtsp", url="rtsp://10.0.0.9/live",
                            username="u", password="pw", location="Hall B")
            assert rec.camera_id == "CLOUD01"
            assert store.list()[0].camera_id == "CLOUD01"
            updated = store.update("CLOUD01", {"name": "Cloud Cam v2"})
            assert updated.name == "Cloud Cam v2"
            assert updated.password == "pw"
            safe = updated.safe_dict()
            assert safe["credentials"] == "CONFIGURED"
            assert safe["password"] == cs.REDACTED
            assert "pw" not in str(safe)
            assert "pw" not in str(updated)       # repr is credential-free
        finally:
            conn.close()
    finally:
        _restore_runtime_modules(saved)


# ----------------------------------------------------------------------
# Test Connection: cloud mode vs real local runtime
# ----------------------------------------------------------------------

def test_test_connection_reports_runtime_unavailable_in_cloud(monkeypatch):
    saved = _enter_cloud(monkeypatch)
    try:
        res = tc_probe("rtsp://10.0.0.9/live", kind="rtsp",
                       username="u", password="pw", timeout=1.0)
        assert res["ok"] is False
        assert res["connected"] is False
        assert res["health"] == "RUNTIME_UNAVAILABLE"
        assert "camera runtime unavailable" in res["error"]
        assert res["url"].startswith("rtsp://")
        assert "pw" not in str(res)               # credential never leaked
        assert "u:pw" not in res["url"]
    finally:
        _restore_runtime_modules(saved)


def test_test_connection_uses_real_capture_path_when_runtime_available(monkeypatch):
    def _runtime():
        return _fake_capture(runtime_ok=True), None

    monkeypatch.setattr(cs, "load_camera_runtime", _runtime)
    res = tc_probe("rtsp://10.0.0.9/live", kind="rtsp",
                   username="u", password="pw", timeout=2.0)
    assert res["ok"] is True
    assert res["connected"] is True
    assert res["health"] == "ONLINE"
    assert res["fps"] == 15.0
    assert "u:pw" not in res["url"]
    assert "pw" not in str(res)


def test_test_connection_failed_probe_never_leaks_credentials(monkeypatch):
    def _runtime():
        return _fake_capture(runtime_ok=False), None

    monkeypatch.setattr(cs, "load_camera_runtime", _runtime)
    res = tc_probe("rtsp://10.0.0.9/live", kind="rtsp",
                   username="u", password="pw", timeout=2.0)
    assert res["ok"] is False
    assert res["connected"] is False
    assert res["health"] == "OFFLINE"
    assert "pw" not in str(res)


# ----------------------------------------------------------------------
# Streamlit Cloud simulation through the live app
# ----------------------------------------------------------------------

def _app_streamlit():
    st = pytest.importorskip("streamlit.testing.v1").AppTest
    return st.from_file(APP, default_timeout=60)


def test_cloud_cameras_page_renders_and_reports_test_unavailable(monkeypatch):
    saved = _enter_cloud(monkeypatch)
    try:
        at = _app_streamlit()
        at.run()
        assert not at.exception, at.exception

        nav = at.sidebar.radio[0]
        nav.set_value(option := [o for o in nav.options if "Cameras" in o][0])
        at.run()
        assert not at.exception, at.exception
        subs = " ".join(str(el.value) for el in at.subheader)
        assert "Camera list" in subs
        assert "Add camera" in subs
        labels = {str(el.label) for el in at.metric}
        assert {"Total Cameras", "Online", "Offline", "Disabled"} <= labels

        # Fill the add form and hit Test Connection -> cloud-unavailable path.
        set_url = set_id = False
        for w in at.text_input:
            if w.label == "Camera ID":
                w.set_value("CLOUDTEST")
                set_id = True
            elif w.label == "URL / source":
                w.set_value("rtsp://10.0.0.9/live")
                set_url = True
        assert set_url and set_id
        btns = [b for b in at.button if str(b.label) == "Test Connection"]
        assert btns
        btns[0].click()
        at.run()
        assert not at.exception, at.exception
        body_text = "\n".join(str(el.value) for el in at.error)
        assert "TEST CONNECTION UNAVAILABLE IN CLOUD" in body_text
    finally:
        _restore_runtime_modules(saved)


def test_all_navigation_pages_render_in_cloud_mode(monkeypatch):
    saved = _enter_cloud(monkeypatch)
    try:
        at = _app_streamlit()
        at.run()
        assert not at.exception, at.exception
        options = list(at.sidebar.radio[0].options)
        assert len(options) == 10
        for label in options:
            nav = at.sidebar.radio[0]
            nav.set_value(label)
            at.run()
            assert not at.exception, at.exception
    finally:
        _restore_runtime_modules(saved)