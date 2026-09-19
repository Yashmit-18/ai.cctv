"""Phase 42 -- premium dashboard UI/UX design-system regression tests.

Locks in the Phase 42 semantics that the visual redesign must never break:

  * status views are textual + tonal, never color-only;
  * "Unknown" is neutral (violet) -- never red, never an employee;
  * KPIs come from the live snapshot / security snapshot only (no invented
    numbers when the daemon is not running);
  * camera sources are sanitised so RTSP credentials can never surface;
  * the app shell renders from a real-shaped (synthetic) snapshot without
    exceptions and reflects the snapshot's active/away/on-phone counts.
"""

import time

import pytest

import app

st = pytest.importorskip("streamlit.testing.v1").AppTest

_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
_APP = _ROOT / "app.py"


# ----------------------------------------------------------------------
# Pure helpers: status/camera views
# ----------------------------------------------------------------------

def test_view_state_unknown_is_neutral_never_red():
    v = app._view_state("Unknown")
    assert v["label"].upper().startswith("UNKNOWN")
    assert v["tone"] == "violet"
    v2 = app._view_state("UNKNOWN")
    assert v2["tone"] == "violet"


def test_view_state_tones_are_semantic():
    assert app._view_state("ACTIVE")["tone"] == "green"
    assert app._view_state("ON_PHONE")["tone"] == "orange"
    assert app._view_state("AWAY")["tone"] == "gray"
    assert app._view_state("NOT_OBSERVED")["tone"] == "gray"
    # Unknown/garbage never throws and always returns a label.
    assert app._view_state(None)["label"] != ""
    assert app._view_state(123)["label"] != ""


def test_view_camera_tones():
    assert app._view_camera("ONLINE")["tone"] == "green"
    assert app._view_camera("OFFLINE")["tone"] == "red"
    assert app._view_camera("NO_FRAME")["tone"] == "red"
    assert app._view_camera("LOW_FPS")["tone"] == "orange"
    assert app._view_camera("FROZEN_FRAME")["tone"] == "violet"


def test_safe_cam_source_never_leaks_credentials():
    assert app._safe_cam_source("rtsp://user:secret@host/stream") == "RTSP (hidden)"
    assert app._safe_cam_source("0") == "0"
    assert app._safe_cam_source("") == "—"
    assert app._safe_cam_source(None) == "—"


# ----------------------------------------------------------------------
# KPIs: real data only
# ----------------------------------------------------------------------

def _fixture_live() -> dict:
    now = time.time()
    return {
        "updated": now,
        "iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "states": {"EMP001": "ACTIVE", "EMP002": "AWAY", "EMP003": "ON_PHONE",
                   "Unknown": "ACTIVE"},
        "telemetry": {"EMP001": {"ACTIVE": 10.0, "ON_PHONE": 0.0, "AWAY": 0.0},
                      "EMP002": {"ACTIVE": 0.0, "ON_PHONE": 0.0, "AWAY": 5.0},
                      "EMP003": {"ACTIVE": 2.0, "ON_PHONE": 30.0, "AWAY": 0.0}},
        "durations": {"EMP001": 10.0, "EMP002": 8.0, "EMP003": 12.0,
                      "Unknown": 6.0},
        "last_seen": {"EMP001": now - 2, "EMP002": now - 40, "EMP003": now - 3,
                      "Unknown": now - 1},
        "employees": {
            "EMP001": {"employee_id": "EMP001", "name": "Yashmit Sharma",
                       "state": "ACTIVE", "session_sec": 10.0, "active_sec": 10.0,
                       "phone_sec": 0.0, "away_sec": 0.0, "source": "local_webcam",
                       "last_seen": now - 2},
            "EMP002": {"employee_id": "EMP002", "name": "Sourabh", "state": "AWAY",
                       "session_sec": 8.0, "active_sec": 3.0, "phone_sec": 0.0,
                       "away_sec": 5.0, "source": "local_webcam", "last_seen": now - 40},
            "EMP003": {"employee_id": "EMP003", "name": "Piyush", "state": "ON_PHONE",
                       "session_sec": 12.0, "active_sec": 2.0, "phone_sec": 30.0,
                       "away_sec": 0.0, "source": "local_webcam", "last_seen": now - 3},
            "EMP004": {"employee_id": "EMP004", "name": "Vishal", "state": "NOT_OBSERVED",
                       "session_sec": 0.0, "active_sec": 0.0, "phone_sec": 0.0,
                       "away_sec": 0.0, "source": "", "last_seen": None},
        },
        "camera_health": {"local_webcam": {"source": "0", "health": "ONLINE",
                                           "fps": 25.0, "frames_read": 500,
                                           "last_frame": now, "usable": True}},
        "fps": 24.5,
        "last_detection_sec": 5,
        "security": {"open_incidents": 2, "high_severity_open": 1},
        "ai": {"spatial_tracks": {"tracks": [], "count": 0},
               "motion": {}, "camera_selection": {}, "capabilities": {}},
    }


def test_live_kpis_use_only_real_data():
    live = _fixture_live()
    import pandas as pd
    k = app._live_kpis(live, pd.DataFrame())
    assert k["active"] == 1          # EMP001 only (ACTIVE)
    assert k["away"] == 1            # EMP002 only (AWAY)
    assert k["on_phone"] == 1        # EMP003 only (ON_PHONE)
    assert k["unknown"] == 1         # the "Unknown" key (never an employee)
    assert k["cameras_online"] == 1
    assert k["cameras_total"] == 1
    assert k["open_incidents"] == 2

    # No snapshot -> live-derived KPIs stay 0 / absent, never invented.
    k2 = app._live_kpis({}, pd.DataFrame())
    assert k2["active"] == 0 and k2["away"] == 0 and k2["on_phone"] == 0
    assert k2["cameras_total"] == 0
    assert k2["open_incidents"] is None


def test_snapshot_status_classification(monkeypatch):
    monkeypatch.setattr(app, "live_state_age", lambda: None)
    assert app._snapshot_status({})["kind"] == "offline"

    monkeypatch.setattr(app, "live_state_age", lambda: 120.0)
    s = app._snapshot_status({})
    assert s["kind"] == "stale" and s["age"] == 120.0

    s = app._snapshot_status(_fixture_live())
    assert s["kind"] == "live" and s["age"] < 60


# ----------------------------------------------------------------------
# App shell render with a real-shaped snapshot (AppTest, no browser)
# ----------------------------------------------------------------------

def _app():
    at = st.from_file(str(_APP), default_timeout=40)
    return at


def test_dashboard_renders_snapshot_derived_kpis(monkeypatch, tmp_path):
    # Feed an isolated fixture snapshot through the CCTV_LIVE_STATE_FILE seam
    # so the production data/live_state.json is never touched.
    live = _fixture_live()
    snap_file = tmp_path / "live_state.json"
    snap_file.write_text(
        __import__("json").dumps(live), encoding="utf-8")
    monkeypatch.setenv("CCTV_LIVE_STATE_FILE", str(snap_file))

    at = _app()
    at.run()
    assert not at.exception, at.exception

    headers = " ".join(str(el.value) for el in at.header)
    assert "Live Overview" in headers
    assert len(at.metric) >= 8

    values = {str(el.label): str(el.value) for el in at.metric}
    assert values.get("Cameras Online") == "1/1"
    assert values.get("People Active") == "1"
    assert values.get("On Phone") == "1"
    assert values.get("Open Incidents") == "2"

    # The premium shell renders its self-contained components (topbar, live
    # strip, footer) as st.html blocks.
    assert len(at.get("html")) >= 3

    # Switch to Live Monitoring: camera + people cards also stream.
    nav = at.sidebar.radio[0]
    nav.set_value(option := [o for o in nav.options if "Live Monitoring" in o][0])
    at.run()
    assert not at.exception, at.exception
    assert len(at.get("html")) >= 5  # topbar, strip, camera cards, people cards, footer


def test_dashboard_no_snapshot_still_boots(monkeypatch, tmp_path):
    # Point the seam at a directory without any snapshot.
    monkeypatch.setenv("CCTV_LIVE_STATE_FILE", str(tmp_path / "absent.json"))
    at = _app()
    at.run()
    assert not at.exception, at.exception