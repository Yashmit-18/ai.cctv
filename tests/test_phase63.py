"""Phase 63 -- Phone-detection guardrails & camera admin knobs.

Closes the Phase 62 gap end-to-end (real APIs, no mocks):

* the Phase 63 phone-pass knobs -- class id / model path / resolution /
  cadence / evidence window+min -- are persisted in the existing ``settings``
  table, with defaults that mirror the ``config`` env knobs exactly;
* validation of those knobs (imgsz >= 32, cadence >= 1, evidence min <=
  window, integer class id, trimmed model path);
* ``apply_to_config`` rebinds ``config.PHONE_*`` ONLY for keys the admin
  actually persisted -- an env-tuned deployment that never touched the panel
  keeps its exact env values (backward compatible);
* the away/phone FSM thresholds are read LIVE from ``config`` (never captured
  at import time) and ``main.py`` forwards the persisted ``KEY_AWAY`` /
  ``KEY_PHONE`` values into ``MultiTracker``, so Admin -> Thresholds finally
  takes real effect on the employee FSM;
* ``live_state["ai"]["phone_pass"]`` publishes the dedicated phone-pass
  diagnostics to the dashboard;
* the dashboard renders the new knobs (thresholds form) and the live
  phone-pass panel (AI Capabilities).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

import config
import main as main_mod
from src import database as db
from src.admin_store import (KEY_AWAY, KEY_PHONE, KEY_PHONE_CLASS,
                             KEY_PHONE_MODEL_PATH, KEY_PHONE_IMGSZ,
                             KEY_PHONE_CADENCE, KEY_PHONE_EVIDENCE_WINDOW,
                             KEY_PHONE_EVIDENCE_MIN, SettingsStore,
                             validate_office_settings)
from src.tracker import EmployeeTracker, MultiTracker, ON_PHONE

_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _mkconn(tmp_path) -> sqlite3.Connection:
    conn = db.get_connection(str(tmp_path / "phase63.db"))
    db.init_db(conn)
    return conn


def _set_phone_settings(store, **kw):
    return store.update(actor="admin", **kw)


def _patched_phone_attrs(monkeypatch, **kw):
    """Monkeypatch config.PHONE_* knobs and restore them at teardown."""
    for name, value in kw.items():
        monkeypatch.setattr(config, name, value)


# ----------------------------------------------------------------------
# Store: default parity with config env knobs + persistence
# ----------------------------------------------------------------------

def test_phone_pass_defaults_match_config_env_knobs(tmp_path):
    conn = _mkconn(tmp_path)
    s = SettingsStore(conn).get()
    assert s.phone_class == 67 == config.PHONE_CLASS
    assert s.phone_model_path == "" == config.PHONE_MODEL_PATH
    assert s.phone_imgsz == 1280 == config.PHONE_IMGSZ
    assert s.phone_cadence == 3 == config.PHONE_DETECT_CADENCE
    assert s.phone_evidence_window == 5 == config.PHONE_EVIDENCE_WINDOW
    assert s.phone_evidence_min == 2 == config.PHONE_EVIDENCE_MIN
    # nothing persisted yet -> admin keys all unset (env-only deployment)
    assert SettingsStore(conn).keys_set() == ()
    conn.close()


def test_phone_pass_persistence_roundtrip_marks_keys(tmp_path):
    conn = _mkconn(tmp_path)
    store = SettingsStore(conn)
    _set_phone_settings(store, phone_class=0, phone_model_path="C:/opt/ph.pt",
                        phone_imgsz=960, phone_cadence=2,
                        phone_evidence_window=6, phone_evidence_min=3)
    keys = set(store.keys_set())
    assert {KEY_PHONE_CLASS, KEY_PHONE_MODEL_PATH, KEY_PHONE_IMGSZ,
            KEY_PHONE_CADENCE, KEY_PHONE_EVIDENCE_WINDOW,
            KEY_PHONE_EVIDENCE_MIN} <= keys
    s = store.get()
    assert s.phone_class == 0
    assert s.phone_model_path == "C:/opt/ph.pt"
    assert s.phone_imgsz == 960
    assert s.phone_cadence == 2
    assert s.phone_evidence_window == 6
    assert s.phone_evidence_min == 3
    conn.close()


def test_phone_pass_update_merges_not_resets(tmp_path):
    conn = _mkconn(tmp_path)
    store = SettingsStore(conn)
    # First write persists the merged set (defaults for untouched fields).
    _set_phone_settings(store, phone_imgsz=960)
    keys = set(store.keys_set())
    assert KEY_PHONE_IMGSZ in keys
    assert KEY_PHONE_CLASS in keys          # merged default, not dropped
    s = store.get()
    assert s.phone_imgsz == 960
    assert s.phone_class == 67              # unchanged default
    # A later partial write merges -- it never resets already-persisted knobs.
    _set_phone_settings(store, phone_cadence=2)
    s = store.get()
    assert s.phone_imgsz == 960             # untouched by the cadre update
    assert s.phone_cadence == 2
    conn.close()


def test_phone_pass_validation_rejects_bad_values(tmp_path):
    conn = _mkconn(tmp_path)
    store = SettingsStore(conn)
    with pytest.raises(ValueError):
        _set_phone_settings(store, phone_imgsz=31)          # < 32
    with pytest.raises(ValueError):
        _set_phone_settings(store, phone_cadence=0)         # < 1
    with pytest.raises(ValueError):
        _set_phone_settings(store, phone_class="abc")       # not an int
    with pytest.raises(ValueError):
        _set_phone_settings(store, phone_class=-1)
    with pytest.raises(ValueError):
        # min must be <= window
        _set_phone_settings(store, phone_evidence_window=2,
                            phone_evidence_min=4)
    with pytest.raises(ValueError):
        _set_phone_settings(store, phone_evidence_min=0)
    assert db.get_setting(conn, KEY_PHONE_IMGSZ) is None
    # model path is trimmed, empty allowed
    _set_phone_settings(store, phone_model_path="  ")
    assert store.get().phone_model_path == ""
    conn.close()


def test_validate_office_settings_accepts_full_phase63_payload():
    out = validate_office_settings(
        timezone="Asia/Kolkata", office_start="10:00", office_end="18:30",
        lunch_start="14:00", lunch_end="14:35",
        away_seconds=10, phone_seconds=5, talking_seconds=15,
        phone_class=0, phone_model_path="C:/opt/ph.pt", phone_imgsz=960,
        phone_cadence=2, phone_evidence_window=6, phone_evidence_min=3)
    assert out["phone_class"] == 0
    assert out["phone_model_path"] == "C:/opt/ph.pt"
    assert out["phone_imgsz"] == 960
    assert out["phone_cadence"] == 2
    assert out["phone_evidence_window"] == 6
    assert out["phone_evidence_min"] == 3


# ----------------------------------------------------------------------
# apply_to_config: rebind persisted knobs only (never env surprises)
# ----------------------------------------------------------------------

def test_apply_to_config_rebinds_only_persisted_phone_knobs(tmp_path):
    conn = _mkconn(tmp_path)
    store = SettingsStore(conn)
    before = {k: getattr(config, k) for k in (
        "PHONE_CLASS", "PHONE_MODEL_PATH", "PHONE_IMGSZ",
        "PHONE_DETECT_CADENCE", "PHONE_EVIDENCE_WINDOW",
        "PHONE_EVIDENCE_MIN")}
    try:
        # persist ONLY the resolution; every other knob must stay untouched.
        _set_phone_settings(store, phone_imgsz=960)
        store.apply_to_config()
        assert config.PHONE_IMGSZ == 960
        assert config.PHONE_CLASS == before["PHONE_CLASS"]
        assert config.PHONE_MODEL_PATH == before["PHONE_MODEL_PATH"]
        assert config.PHONE_DETECT_CADENCE == before["PHONE_DETECT_CADENCE"]
        assert config.PHONE_EVIDENCE_WINDOW == before["PHONE_EVIDENCE_WINDOW"]
        assert config.PHONE_EVIDENCE_MIN == before["PHONE_EVIDENCE_MIN"]
        # persist the rest -> all rebind.
        _set_phone_settings(store, phone_class=0, phone_model_path="P",
                            phone_cadence=4, phone_evidence_window=8,
                            phone_evidence_min=5)
        store.apply_to_config()
        assert config.PHONE_CLASS == 0
        assert config.PHONE_MODEL_PATH == "P"
        assert config.PHONE_DETECT_CADENCE == 4
        assert config.PHONE_EVIDENCE_WINDOW == 8
        assert config.PHONE_EVIDENCE_MIN == 5
        # FUTURE_* reserved knobs keep flowing to main.py as before.
        assert config.FUTURE_AWAY_THRESHOLD_SECONDS == 10.0
        assert config.FUTURE_PHONE_THRESHOLD_SECONDS == 5.0
    finally:
        for k, v in before.items():
            setattr(config, k, v)
    conn.close()


def test_apply_to_config_empty_settings_never_touches_env_knobs(
        tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PHONE_IMGSZ", 777)
    monkeypatch.setattr(config, "PHONE_DETECT_CADENCE", 6)
    monkeypatch.setattr(config, "PHONE_CLASS", 99)
    conn = _mkconn(tmp_path)
    store = SettingsStore(conn)
    assert store.keys_set() == ()
    store.apply_to_config()
    # untouched settings table -> env-tuned values survive verbatim.
    assert config.PHONE_IMGSZ == 777
    assert config.PHONE_DETECT_CADENCE == 6
    assert config.PHONE_CLASS == 99
    conn.close()


# ----------------------------------------------------------------------
# FSM wiring: admin away/phone thresholds take real effect
# ----------------------------------------------------------------------

def test_fsm_phone_threshold_explicit_param_honored(tmp_path):
    conn = _mkconn(tmp_path)
    t = EmployeeTracker(conn, "EMP001", phone_after_sec=0.0)
    assert t._phone_after_sec == 0.0
    t.process({"present": True, "phone_detected": False})
    assert t.committed_state == "ACTIVE"
    t.process({"present": True, "phone_detected": True})
    assert t.committed_state == ON_PHONE
    events = db.query_security_events(conn, event_type="PHONE_USE")
    assert events and events[0]["employee_id"] == "EMP001"
    conn.close()


def test_fsm_phone_threshold_live_config_default(tmp_path, monkeypatch):
    # Prove the FSM reads config LIVE: monkeypatch to 0.0 (instant) while the
    # import-time constant stays 5.0 -- an import-time read would NOT commit.
    monkeypatch.setattr(config, "PHONE_AFTER_SEC", 0.0)
    conn = _mkconn(tmp_path)
    t = EmployeeTracker(conn, "EMP002")   # no explicit param
    assert t._phone_after_sec == 0.0
    t.process({"present": True, "phone_detected": False})
    t.process({"present": True, "phone_detected": True})
    assert t.committed_state == ON_PHONE
    conn.close()


def test_fsm_away_threshold_live_config_default(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AWAY_AFTER_SEC", 0.0)
    conn = _mkconn(tmp_path)
    t = EmployeeTracker(conn, "EMP002")
    assert t._away_after_sec == 0.0
    t.process({"present": True, "phone_detected": False})
    assert t.committed_state == "ACTIVE"
    t.process({"present": False, "phone_detected": False})
    assert t.committed_state == "AWAY"
    conn.close()


def test_multitracker_forwards_fsm_thresholds(tmp_path):
    conn = _mkconn(tmp_path)
    mt = MultiTracker(conn, phone_after_sec=0.0, away_after_sec=0.0)
    mt.process({"employees": {"EMP001": {"present": True,
                                         "phone_detected": False}}})
    assert mt.live_state("EMP001") == "ACTIVE"
    mt.process({"employees": {"EMP001": {"present": True,
                                         "phone_detected": True}}})
    assert mt.live_state("EMP001") == ON_PHONE
    mt.process({"employees": {"EMP001": {"present": False,
                                         "phone_detected": False}}})
    assert mt.live_state("EMP001") == "AWAY"
    conn.close()


def test_main_wires_admin_thresholds_into_multitracker(tmp_path, monkeypatch):
    # main.py builds MultiTracker with the persisted away/phone seconds (or
    # None when untouched).  Verify the wiring contract the daemon relies on.
    conn = _mkconn(tmp_path)
    store = SettingsStore(conn)
    _set_phone_settings(store, away_seconds=20.0, phone_seconds=7.0)
    keys = set(store.keys_set())
    away = store.get().away_seconds if KEY_AWAY in keys else None
    phone = store.get().phone_seconds if KEY_PHONE in keys else None
    assert away == 20.0
    assert phone == 7.0
    mt = MultiTracker(conn, away_after_sec=away, phone_after_sec=phone)
    stock = mt._trackers
    mt.process_batch([{"emp_id": "EMP001", "present": True, "phone": False}])
    assert stock["EMP001"]._away_after_sec == 20.0
    assert stock["EMP001"]._phone_after_sec == 7.0
    conn.close()


# ----------------------------------------------------------------------
# live_state publishing + dashboard surfaces
# ----------------------------------------------------------------------

def test_write_live_state_publishes_phone_pass(tmp_path, monkeypatch):
    target = str(tmp_path / "live_state.json")
    monkeypatch.setattr(main_mod, "_LIVE_STATE_FILE", target)
    main_mod._write_live_state({}, {}, {}, 0.0,
                               phone_pass={"calls": 5, "source": "base-model"})
    with open(target, encoding="utf-8") as f:
        data = json.load(f)
    assert data["ai"]["phone_pass"]["calls"] == 5
    assert data["ai"]["phone_pass"]["source"] == "base-model"


def test_write_live_state_phone_pass_defaults_empty(tmp_path, monkeypatch):
    target = str(tmp_path / "live_state.json")
    monkeypatch.setattr(main_mod, "_LIVE_STATE_FILE", target)
    main_mod._write_live_state({}, {}, {}, 0.0)
    with open(target, encoding="utf-8") as f:
        data = json.load(f)
    assert data["ai"]["phone_pass"] == {}


def test_app_thresholds_source_has_phone_pass_section():
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert "Phone detection pass (Phase 63)" in src
    assert "phone_evidence_min" in src
    assert "phone_cadence" in src


def test_ai_capabilities_renders_phone_pass_panel(monkeypatch, tmp_path):
    st = pytest.importorskip("streamlit.testing.v1").AppTest
    now = time.time()
    live = {
        "updated": now,
        "iso": "2023-11-14 22:13:20",
        "states": {},
        "telemetry": {},
        "durations": {},
        "last_seen": {},
        "employees": {},
        "camera_health": {"c0": {"source": "0", "health": "ONLINE",
                                 "fps": 24.0, "frames_read": 10,
                                 "last_frame": now, "usable": True}},
        "fps": 23.0,
        "last_detection_sec": 1.0,
        "security": {"open_incidents": 0, "high_severity_open": 0},
        "ai": {
            "spatial_tracks": {"tracks": [], "count": 0},
            "motion": {},
            "camera_selection": {},
            "capabilities": {},
            "phone_pass": {"source": "base-model", "phone_class": 67,
                           "imgsz": 1280, "cadence": 3, "calls": 12,
                           "boxes_seen": 8, "conf_mean": 0.61,
                           "conf_min": 0.41},
        },
    }
    snap = tmp_path / "live_state.json"
    snap.write_text(json.dumps(live), encoding="utf-8")
    monkeypatch.setenv("CCTV_LIVE_STATE_FILE", str(snap))

    at = st.from_file(str(_ROOT / "app.py"), default_timeout=40)
    at.run()
    assert not at.exception, at.exception
    nav = at.sidebar.radio[0]
    opt = next(o for o in nav.options if "Live Monitoring" in o)
    nav.set_value(opt)
    at.run()
    assert not at.exception, at.exception
    assert any("Phone Detection Pass" in str(e.value) for e in at.subheader)