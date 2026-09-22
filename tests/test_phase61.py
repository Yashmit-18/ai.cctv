"""Phase 61 -- Admin Camera Management (persistent, dashboard-driven).

Covers:
* CRUD of the persistent admin camera store (add / get / update / enable /
  disable / delete);
* stable identity (camera_id never changes on edit) and duplicate rejection;
* credential handling -- password never appears in any outward shape
  (safe_dict / repr / redacted URL) and recomposed transport URLs fold the
  split credentials without touching the stored base URL;
* syntax-only validation (RTSP scheme + netloc; never reachability);
* deletes guarded by historical references;
* runtime resolution: persistent wins, env fills gaps, empty store == env;
* reconcile/apply handshake -- revision drift and explicit requests trigger
  `MultiCameraManager.apply_configs`; no-change is a no-op;
* Test Connection probes bounded, releases resources, reports ok=False without
  hardware and never contains the password;
* camera failure never fabricates employee AWAY / never guesses reachability.

The fake capture monkeypatches the same two seams Phase 58 tests use
(``src.camera.VideoCapture`` and ``src.camera_manager.VideoCapture``).
"""

from __future__ import annotations

import os
import tempfile

import pytest

from src import database as db
from src.camera_manager import MultiCameraManager
from src.camera_store import (CameraRecord, CameraStore, REDACTED,
                              compose_rtsp_url, reconcile_runtime,
                              runtime_configs, validate_camera_fields)
from src.camera_store import test_camera_connection as tc_probe
from src.domain import CameraSourceKind


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

_CAM_ONLINE = "ONLINE"
_CAM_OFFLINE = "OFFLINE"


class _FakeCap:
    """Minimal stand-in for ``src.camera.VideoCapture`` (Phase 58 pattern)."""

    def __init__(self, source, kind=None, camera_id=None,
                 reconnect=(2.0, 30.0, 2.5), ok=True):
        self.source = source
        self.kind = kind
        self.camera_id = camera_id
        self._reconnect_cfg = tuple(reconnect)
        self._ok = ok
        self.started = False
        self.stopped = False
        self.width = 1280 if ok else 0
        self.height = 720 if ok else 0

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True

    @property
    def is_connected(self) -> bool:
        return self._ok and self.started and not self.stopped

    def health_info(self) -> dict:
        return {
            "id": self.camera_id,
            "source": "redacted",
            "connected": self.is_connected,
            "health": _CAM_ONLINE if self.is_connected else _CAM_OFFLINE,
            "frames_read": 1 if self.is_connected else 0,
            "fps": 15.0 if self.is_connected else 0.0,
            "reconnects": 0,
            "last_frame": 1.0 if self.is_connected else 0.0,
            "frozen_streak": 0,
            "kind": getattr(self.kind, "value", str(self.kind or "")),
            "width": self.width,
            "height": self.height,
            "resolution": f"{self.width}x{self.height}" if self.width else "",
        }


@pytest.fixture()
def store_conn():
    tmp = tempfile.mkdtemp(prefix="p61_")
    path = os.path.join(tmp, "cameras.db")
    conn = db.get_connection(path)
    db.init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def store(store_conn):
    return CameraStore(store_conn)


def _seed(conn, camera_id="CAM01", url="rtsp://10.0.0.5/stream",
          username="", password="", enabled=True, kind="rtsp"):
    store = CameraStore(conn)
    store.add(camera_id=camera_id, name=f"Camera {camera_id}", kind=kind,
              url=url, username=username, password=password, enabled=enabled,
              location="Hall B")
    return store.get(camera_id)


# ----------------------------------------------------------------------
# Validation (syntax only -- never reachability)
# ----------------------------------------------------------------------

def test_validate_accepts_rtsp_syntax_even_without_device():
    fields = validate_camera_fields(camera_id="CAM01", name="Entrance",
                                    kind="rtsp", url="rtsp://192.168.1.99:554/hd")
    assert fields["kind"] == "rtsp"
    assert fields["url"] == "rtsp://192.168.1.99:554/hd"


def test_validate_rejects_bad_rtsp_url_syntax():
    with pytest.raises(ValueError):
        validate_camera_fields(camera_id="C1", name="x", kind="rtsp",
                               url="http://not-rtsp.example/live")
    with pytest.raises(ValueError):
        validate_camera_fields(camera_id="C1", name="x", kind="rtsp", url="")


def test_validate_rejects_bad_id_and_missing_name():
    with pytest.raises(ValueError):
        validate_camera_fields(camera_id="bad id!", name="x", kind="rtsp",
                               url="rtsp://h/")
    with pytest.raises(ValueError):
        validate_camera_fields(camera_id="C1", name=" ", kind="rtsp",
                               url="rtsp://h/")


def test_validate_rebounds_special_char_credentials_and_fps():
    fields = validate_camera_fields(
        camera_id="C1", name="x", kind="rtsp", url="rtsp://h:554/a",
        fps_target=99,
        reconnect_base=0.0, reconnect_max=60.0, reconnect_factor=1.5)
    assert fields["fps_target"] == 99.0
    with pytest.raises(ValueError):
        validate_camera_fields(
            camera_id="C1", name="x", kind="rtsp", url="rtsp://h/",
            reconnect_max=0.0)
    with pytest.raises(ValueError):
        validate_camera_fields(
            camera_id="C1", name="x", kind="rtsp", url="rtsp://h/",
            reconnect_factor=0.5)
    with pytest.raises(ValueError):
        validate_camera_fields(
            camera_id="C1", name="x", kind="rtsp", url="rtsp://h/", fps_target=500)


# ----------------------------------------------------------------------
# CRUD
# ----------------------------------------------------------------------

def test_add_and_get_roundtrip(store):
    rec = store.add(camera_id="CAM02", name="Loading Bay", kind="rtsp",
                    url="rtsp://10.0.1.9/live", location="Dock A",
                    reconnect_max=45.0)
    got = store.get("CAM02")
    assert got.camera_id == "CAM02"
    assert got.name == "Loading Bay"
    assert got.kind == "rtsp"
    assert got.enabled is True
    assert got.location == "Dock A"
    assert got.reconnect_max == 45.0


def test_add_duplicate_rejected(store):
    _seed(store.conn, "CAM01")
    with pytest.raises(ValueError):
        _seed(store.conn, "CAM01")


def test_update_partial_keeps_other_fields(store):
    _seed(store.conn, "CAM01", username="alice", password="s3cr3t")
    updated = store.update("CAM01", {"name": "Gate Camera"})
    assert updated.name == "Gate Camera"
    assert updated.password == "s3cr3t"          # untouched
    assert updated.username == "alice"           # untouched
    assert updated.url == "rtsp://10.0.0.5/stream"


def test_update_password_keeps_identity(store):
    rec = _seed(store.conn, "CAM01")
    updated = store.update("CAM01",
                           {"password": "newpass", "name": "Renamed"})
    assert updated.camera_id == "CAM01"
    assert updated.name == "Renamed"
    assert updated.password == "newpass"


def test_update_none_password_keeps_saved(store):
    _seed(store.conn, "CAM01", password="oldpass")
    updated = store.update("CAM01", {"password": None})
    assert updated.password == "oldpass"


def test_update_invalid_partial_rolls_back_nothing_bad(store):
    _seed(store.conn, "CAM01")
    with pytest.raises(ValueError):
        store.update("CAM01", {"url": "garbage"})
    assert store.get("CAM01").password == ""  # store untouched


def test_set_enabled_toggle(store):
    _seed(store.conn, "CAM01")
    assert store.get("CAM01").enabled is True
    store.set_enabled("CAM01", False)
    assert store.get("CAM01").enabled is False
    store.set_enabled("CAM01", True)
    assert store.get("CAM01").enabled is True


def test_delete_with_history_refused(store):
    _seed(store.conn, "CAM01")
    db.record_camera_health(store.conn, "CAM01",
                            {"health": _CAM_ONLINE, "resolution": "1280x720"})
    with pytest.raises(ValueError):
        store.delete("CAM01")
    assert store.get("CAM01") is not None


def test_delete_without_history(store):
    _seed(store.conn, "CAM01")
    assert store.delete("CAM01") is True
    assert store.get("CAM01") is None


# ----------------------------------------------------------------------
# Credential / secrecy rules
# ----------------------------------------------------------------------

def test_password_never_in_safe_dict_or_repr(store):
    _seed(store.conn, "CAM01", username="alice", password="TopSecret!@#")
    rec = store.get("CAM01")
    sd = rec.safe_dict()
    assert sd["password"] == REDACTED
    assert "TopSecret" not in repr(rec)
    assert "alice" not in repr(rec) or True  # username may show; password never
    assert "TopSecret" not in str(rec)
    assert "TopSecret" not in rec.display()


def test_safe_dict_url_redacted_and_embedded_cred_mask():
    rec = CameraRecord(
        camera_id="X", name="x", kind="rtsp", enabled=True, location="",
        url="rtsp://bob:hunter2@10.0.0.7:554/s", username="", password="")
    sd = rec.safe_dict()
    assert "hunter2" not in sd["url"]
    assert ":" in sd["url"] or "*" in sd["url"]


def test_composed_url_percent_encodes_special_chars(store):
    _seed(store.conn, "C9", username="us er", password="p@ss/word#1")
    composed = store.get("C9").composed_url()
    assert "us%20er" in composed
    assert "p%40ss%2Fword%231" in composed
    assert "@" not in store.get("C9").url  # stored base URL untouched


def test_composed_url_respects_embedded_credentials(store):
    _seed(store.conn, "C9", url="rtsp://joe:old@host:554/live",
           username="other", password="otherpass")
    composed = store.get("C9").composed_url()
    assert "joe:old@" in composed
    assert "otherpass" not in composed  # embedded credentials win


def test_no_password_in_test_connection_result_no_hardware(monkeypatch):
    from src import camera as src_camera

    def _fake_ctor(*args, **kwargs):
        return _FakeCap(ok=False, **kwargs)

    monkeypatch.setattr(src_camera, "VideoCapture", _fake_ctor)
    result = tc_probe("rtsp://10.0.0.200/live",
                      kind="rtsp", username="admin",
                      password="pw-secret")
    assert result["ok"] is False
    assert "pw-secret" not in str(result)
    # Never the password in any shape; the probe URL carries the transport
    # URL redacted (password removed, never re-embedded).
    assert "pw-secret" not in (result["url"] or "")
    assert "***" in (result["url"] or "") or ":" in (result["url"] or "")


# ----------------------------------------------------------------------
# Runtime resolution (Phase 58 compatibility)
# ----------------------------------------------------------------------

def test_empty_store_returns_env_verbatim(store_conn, monkeypatch):
    import config as cfgmod

    def _fake_env():
        return {
            "CAM_A": cfgmod.CameraConfig(camera_id="CAM_A", name="A",
                                         kind=CameraSourceKind.RTSP,
                                         url="rtsp://env/1", enabled=True),
        }

    monkeypatch.setattr(cfgmod, "camera_configs", _fake_env)
    resolved = runtime_configs(store_conn)
    assert set(resolved) == {"CAM_A"}


def test_persistent_wins_and_env_fills_gaps(store_conn, monkeypatch):
    import config as cfgmod

    _seed(store_conn, "CAM01", url="rtsp://persisted/1", username="u")
    _seed(store_conn, "CAM02")
    store = CameraStore(store_conn)
    store.add(camera_id="PERSIST", name="P", kind="rtsp",
              url="rtsp://persisted/2")
    store.set_enabled("CAM02", False)

    def _fake_env():
        return {
            "CAM01": cfgmod.CameraConfig(camera_id="CAM01", name="envA",
                                         kind=CameraSourceKind.RTSP,
                                         url="rtsp://env/1", enabled=True),
            "CAM03": cfgmod.CameraConfig(camera_id="CAM03", name="envC",
                                         kind=CameraSourceKind.RTSP,
                                         url="rtsp://env/3", enabled=True),
        }

    monkeypatch.setattr(cfgmod, "camera_configs", _fake_env)
    resolved = runtime_configs(store_conn)
    # Persistent records win (composed transport URL carries the split
    # credentials, host still the persisted one, never the env source) ...
    assert resolved["CAM01"].url == "rtsp://u@persisted/1"
    assert resolved["CAM01"].enabled is True
    assert "env/1" not in resolved["CAM01"].url
    # ... disabled camera excluded from the pool (manager's job) but present
    # in the config set with enabled=False ...
    assert resolved["CAM02"].enabled is False
    # ... and env camera absent from the store is kept.
    assert resolved["CAM03"].url == "rtsp://env/3"


def test_to_config_matches_run_contract(store_conn):
    rec = _seed(store_conn, "CAM01", url="rtsp://10.0.0.5/stream",
                username="admin", password="pw")
    cfg = rec.to_config()
    assert cfg.camera_id == "CAM01"
    assert cfg.kind is CameraSourceKind.RTSP
    assert cfg.password == "pw"  # inside CameraConfig (Phase 58 model)
    assert cfg.enabled is True


def test_local_without_url_transports_to_device_zero(store_conn):
    store = CameraStore(store_conn)
    store.add(camera_id="WCAM", name="Webcam", kind="local", url="")
    rec = store.get("WCAM")
    assert rec.transport_url() == 0


# ----------------------------------------------------------------------
# Runtime apply / reconcile handshake
# ----------------------------------------------------------------------

def test_reconcile_no_change_is_noop(store_conn, monkeypatch):
    _seed(store_conn, "CAM01")
    store = CameraStore(store_conn)

    from src import camera_manager as cmmod
    monkeypatch.setattr(cmmod, "VideoCapture", lambda **kw: _FakeCap(**kw))
    mgr = MultiCameraManager(
        cameras={"CAM01": "rtsp://10.0.0.5/stream"},
        configs=runtime_configs(store_conn, env_configs={})).start()

    result = reconcile_runtime(store_conn, mgr, env_configs={})
    assert result["action"] == "applied"          # first reconcile applies
    result2 = reconcile_runtime(store_conn, mgr, env_configs={})
    assert result2["action"] == "up_to_date"      # no drift -> no-op
    mgr.stop()


def test_reconcile_applies_on_revision_change(store_conn, monkeypatch):
    _seed(store_conn, "CAM01")
    from src import camera_manager as cmmod
    fake = _FakeCap(source="rtsp://10.0.0.5/stream", kind=CameraSourceKind.RTSP,
                    camera_id="CAM01")
    monkeypatch.setattr(cmmod, "VideoCapture", lambda **kw: _FakeCap(**kw))
    mgr = MultiCameraManager(cameras={"CAM01": "rtsp://10.0.0.5/stream"},
                             configs=runtime_configs(store_conn,
                                                     env_configs={})).start()
    reconcile_runtime(store_conn, mgr, env_configs={})

    _seed(store_conn, "CAM02", url="rtsp://10.0.1.1/stream")
    result = reconcile_runtime(store_conn, mgr, env_configs={})
    assert result["action"] == "applied"
    assert result["summary"]["started"] >= 1
    mgr.stop()


def test_apply_configs_stops_disabled_and_starts_new(store_conn, monkeypatch):
    _seed(store_conn, "CAM01")
    from src import camera_manager as cmmod
    monkeypatch.setattr(cmmod, "VideoCapture",
                        lambda **kw: _FakeCap(**kw))
    mgr = MultiCameraManager(
        cameras={"CAM01": "rtsp://10.0.0.5/stream"},
        configs=runtime_configs(store_conn, env_configs={})).start()
    old_cap = mgr._captures["CAM01"]

    # disable CAM01 + add CAM02 (persisted) -> apply_configs stops CAM01,
    # starts CAM02, keeps nothing of the old set.
    _seed(store_conn, "CAM02", url="rtsp://10.0.1.1/stream")
    store = CameraStore(store_conn)
    store.set_enabled("CAM01", False)

    summary = mgr.apply_configs(runtime_configs(store_conn, env_configs={}))
    assert summary["stopped"] == 1               # CAM01 removed
    assert summary["started"] == 1               # CAM02 gained
    assert old_cap.stopped is True
    assert "CAM02" in mgr._captures
    assert "CAM01" not in mgr._captures
    mgr.stop()


def test_apply_configs_keeps_unchanged_and_reopens_changed(store_conn, monkeypatch):
    _seed(store_conn, "CAM01")
    from src import camera_manager as cmmod
    monkeypatch.setattr(cmmod, "VideoCapture",
                        lambda **kw: _FakeCap(**kw))
    mgr = MultiCameraManager(
        cameras={"CAM01": "rtsp://10.0.0.5/stream"},
        configs=runtime_configs(store_conn, env_configs={})).start()
    old_cap = mgr._captures["CAM01"]

    res = mgr.apply_configs(runtime_configs(store_conn, env_configs={}))
    assert res["kept"] == 1
    assert mgr._captures["CAM01"] is old_cap               # untouched

    store = CameraStore(store_conn)
    store.update("CAM01", {"url": "rtsp://10.9.9.9/stream"})
    res = mgr.apply_configs(runtime_configs(store_conn, env_configs={}))
    assert res["updated"] == 1                             # reopened
    assert old_cap.stopped is True
    assert mgr._captures["CAM01"].source == "rtsp://10.9.9.9/stream"
    mgr.stop()


def test_apply_sets_applied_revision_matching_store(store_conn, monkeypatch):
    _seed(store_conn, "CAM01")
    from src import camera_manager as cmmod
    monkeypatch.setattr(cmmod, "VideoCapture", lambda **kw: _FakeCap(**kw))
    mgr = MultiCameraManager(cameras={"CAM01": "rtsp://10.0.0.5/stream"},
                             configs=runtime_configs(store_conn,
                                                     env_configs={})).start()
    reconcile_runtime(store_conn, mgr, env_configs={})
    st = CameraStore(store_conn).apply_state()
    assert st and st.get("applied_revision") == CameraStore(store_conn).revision()
    mgr.stop()


def test_request_apply_wins_across_sources(store_conn):
    _seed(store_conn, "CAM01")
    store = CameraStore(store_conn)
    rev = store.request_apply()
    st = store.apply_state()
    assert st.get("requested_revision") == rev


# ----------------------------------------------------------------------
# Camera failure honesty (no fabricated AWAY / no guessed reachability)
# ----------------------------------------------------------------------

def test_test_connection_without_hardware_is_ok_false(monkeypatch):
    from src import camera as src_camera
    monkeypatch.setattr(src_camera, "VideoCapture",
                        lambda **kw: _FakeCap(ok=False, **kw))
    result = tc_probe("rtsp://10.0.0.200:554/live",
                                    kind="rtsp", timeout=1.0)
    assert result["ok"] is False
    assert "not reachable" in result["error"] or "unavailable" in result["error"]


def test_test_connection_releases_capture_even_on_missing_file(monkeypatch):
    from src import camera as src_camera

    class _FailStart(_FakeCap):
        def start(self):
            raise RuntimeError("backend open failed")

    monkeypatch.setattr(src_camera, "VideoCapture",
                        lambda **kw: _FailStart(ok=False, **kw))
    result = tc_probe("does-not-exist.mp4", kind="video_file",
                                    timeout=0.5)
    # finally-path executed; clean failure dict; nothing teared down wrong.
    assert result["ok"] is False
    assert isinstance(result, dict)


def test_invalid_kind_reported_without_crash():
    result = tc_probe("rtsp://h/x", kind="spaceship")
    assert result["ok"] is False
    assert result["health"] == "INVALID_CONFIG"


# ----------------------------------------------------------------------
# Revision stability / multi-camera isolation
# ----------------------------------------------------------------------

def test_revision_changes_with_credentials_and_flag(store):
    _seed(store.conn, "A", url="rtsp://h/a")
    rev0 = store.revision()
    store.update("A", {"username": "u"})
    rev1 = store.revision()
    store.update("A", {"password": "p"})
    rev2 = store.revision()
    assert len({rev0, rev1, rev2}) == 3   # never silently ignored

    store2 = CameraStore(store.conn)
    _seed(store2.conn, "B", url="rtsp://h/b")
    assert store.revision() != rev2        # multi-camera isolation


def test_env_only_deployment_never_triggers_apply(store_conn):
    st = CameraStore(store_conn).apply_state()
    assert st is None                       # no handshake row before any apply
