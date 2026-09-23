"""Streamlit Web Dashboard for the CCTV Employee Productivity Tracker.

Production dashboard (Phases 9-11).  Reads live snapshots written by the
daemon (``data/live_state.json``) and analytics from the read-only SQLite
connection.  All productivity numbers come from :mod:`src.analytics` /
:mod:`src.productivity` so the report, daemon and dashboard agree.

Security (Phase 11)
-------------------
A lightweight admin/viewer gate is provided via environment variables
(``CCTV_DASH_AUTH``, ``CCTV_DASH_USER``, ``CCTV_DASH_PASS``,
``CCTV_DASH_VIEWER_PASS``).  For production, deploy behind a reverse proxy /
VPN -- this is a configuration boundary, not a substitute for it.

Run
---
    streamlit run app.py
"""

import html
import json
import logging
import os
import pickle
import re
import sqlite3
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    DASH_ADMIN_PASS,
    DASH_AUTH_ENABLED,
    DASH_DAY_METRICS_TTL_SEC,
    DASH_FAIL_CLOSED,
    DASH_RANGE_METRICS_TTL_SEC,
    DASH_REFRESH_SEC,
    DASH_ROLE_VIEWER,
    DASH_SESSION_MINUTES,
    DASH_USERNAME,
    DASH_VIEWER_PASS,
    DB_PATH,
    FACES_DIR,
    PROJECT_ROOT,
    REPORT_OUTPUT_DIR,
    WORK_SCHEDULE,
)
from src.analytics import employee_day_metrics, employee_range_metrics
from src.domain import CAM_DARK_BLANK_FRAME
from src.employees import EmployeeStore
from src.incidents import IncidentEngine
from src.notifier import last_email_status
from src.security_events import EventStore, SecurityEvent
from src.security_events import severity_to_index
from src.seat_chairs import ChairStore
from src.seat_zones import SeatZoneStore
from src.zones import ZoneStore

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
# Where the daemon's live snapshot lives.  ``CCTV_LIVE_STATE_FILE`` is a
# test seam: when set (e.g. in the test harness) the dashboard reads the
# snapshot from that path instead, so AppTests can feed isolated fixture
# snapshots without ever touching the production file.  The daemon itself is
# not affected -- it always writes the real ``data/live_state.json``.
_LIVE_STATE_FILE = (Path(os.environ["CCTV_LIVE_STATE_FILE"])
                    if os.environ.get("CCTV_LIVE_STATE_FILE") else
                    Path(PROJECT_ROOT) / "data" / "live_state.json")

st.set_page_config(
    page_title="AI CCTV Intelligence — Security Operations Center",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Read-only SQLite helpers (safe against background daemon writes)
# ----------------------------------------------------------------------

_LAST_DB_ERROR = ""
# Serialises first-run schema bootstrap.  Streamlit Cloud runs each session in
# its own thread, so two concurrent first loads could otherwise initialise the
# same fresh database simultaneously and transiently fail on a busy lock.
_DB_BOOTSTRAP_LOCK = threading.Lock()


def _format_db_error(db_path: str, stage: str, exc: Exception) -> str:
    path = Path(db_path)
    return (
        f"stage={stage}; exception={type(exc).__name__}: {exc}; "
        f"database_path={path}; cwd={Path.cwd()}; "
        f"parent_exists={path.parent.exists()}; file_exists={path.exists()}"
    )


def _probe_schema(db_file: Path) -> bool:
    """Return True when the canonical app schema still needs to be created.

    A missing file, a zero-byte file, or a file without the ``employees`` /
    ``activity_logs`` tables all report True.  A readable, fully initialised
    database reports False.  A file that exists but cannot be read (corrupt /
    not a SQLite database) is surfaced as a ``sqlite3.DatabaseError`` so the
    caller can report the real problem instead of silently hiding it.
    """
    if not db_file.exists():
        return True
    probe = sqlite3.connect(str(db_file), timeout=2)
    try:
        tables = {
            row[0] for row in probe.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        return not {"employees", "activity_logs"} <= tables
    finally:
        probe.close()


class _ReadonlyConnection(sqlite3.Connection):
    """Session-scoped read-only SQLite handle (Phase 64D).

    ``close()`` is a no-op for handles marked ``_cctv_shared``: the shared
    handle is opened once per Streamlit session and reused across every page
    navigation, so the dashboard no longer opens and probes a fresh SQLite
    connection on every render.  Unshared handles (the per-call fallback used
    by cache helpers and by callers outside a Streamlit runtime) close
    normally and never leak.
    """

    def close(self) -> None:
        if not getattr(self, "_cctv_shared", False):
            super().close()


def _open_readonly_connection(db_path: str = DB_PATH) -> sqlite3.Connection | None:
    """Open ONE fresh read-only connection (used by short-lived callers).

    A plain connection with ``PRAGMA query_only`` is used instead of a
    ``file:...?mode=ro&immutable=1`` URI: read-only URI connections do not
    reliably see rows still held in the live WAL on Windows (``immutable=1``
    skips the WAL and only reads the occasionally-checkpointed main file),
    which made live metrics disagree with :mod:`src.analytics`.  Query-only
    semantics still hard-block any write through this handle.

    If the database file does not yet exist (fresh deployment / first launch),
    create the parent directory and initialize the canonical SQLite schema once
    before opening the read-only handle.  This keeps the dashboard from
    displaying a spurious "Database unavailable." state on a newly booted
    cloud instance while preserving the normal read-only access pattern.

    Callers that keep the handle across renders are expected to use
    :func:`get_readonly_connection` instead, which shares ONE instance per
    Streamlit session and never requires the caller to close it.
    """
    global _LAST_DB_ERROR
    _LAST_DB_ERROR = ""
    db_file = Path(db_path)
    try:
        needs_schema = _probe_schema(db_file)
    except Exception as exc:
        _LAST_DB_ERROR = _format_db_error(str(db_file), "probe", exc)
        logging.getLogger("cctv.dashboard").warning(_LAST_DB_ERROR)
        return None
    if needs_schema:
        with _DB_BOOTSTRAP_LOCK:
            try:
                needs_schema = _probe_schema(db_file)
            except Exception as exc:
                _LAST_DB_ERROR = _format_db_error(str(db_file), "probe", exc)
                logging.getLogger("cctv.dashboard").warning(_LAST_DB_ERROR)
                return None
            if needs_schema:
                try:
                    from src import database as db
                    conn = db.get_connection(str(db_file))
                    try:
                        db.init_db(conn)
                    finally:
                        conn.close()
                except Exception as exc:
                    _LAST_DB_ERROR = _format_db_error(
                        str(db_file), "schema_init", exc)
                    logging.getLogger("cctv.dashboard").warning(_LAST_DB_ERROR)
                    return None
    try:
        conn = sqlite3.connect(str(db_file), factory=_ReadonlyConnection,
                               check_same_thread=False, timeout=2)
        conn.execute("PRAGMA query_only=ON")
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:
        _LAST_DB_ERROR = _format_db_error(str(db_file), "readonly_connect", exc)
        logging.getLogger("cctv.dashboard").warning(_LAST_DB_ERROR)
        return None


def get_readonly_connection(db_path: str = DB_PATH) -> sqlite3.Connection | None:
    """Return the session's shared read-only handle.

    The handle is opened once per Streamlit session (first render) and reused
    by every page and every navigation, which removes the per-render open /
    schema-probe / close cycle that dominated dashboard page turns on slow
    filesystems (e.g. Streamlit Cloud GFFS).  Read-only handles are safe for a
    single Streamlit session because a session only ever runs one script turn
    on one thread; the daemon's writes still arrive through separate write
    connections and are visible through WAL.  Callers never close this handle
    (``close()`` is a no-op on shared handles).

    Outside a running Streamlit runtime (unit tests, tooling) this falls back
    to a fresh per-call connection with the same bootstrap semantics.
    """
    if not st.runtime.exists():
        return _open_readonly_connection(db_path)
    key = f"_cctv_readonly_conn_{db_path}"
    conn = st.session_state.get(key)
    if conn is None:
        conn = _open_readonly_connection(db_path)
        if conn is not None:
            conn._cctv_shared = True
            try:
                st.session_state[key] = conn
            except Exception:  # pragma: no cover - defensive
                conn._cctv_shared = False
                conn.close()
                return _open_readonly_connection(db_path)
    return conn


@st.cache_data(ttl=DASH_DAY_METRICS_TTL_SEC, show_spinner=False)
def cached_day_metrics(day_str: str) -> pd.DataFrame:
    """Cached per-employee metrics for a day (TTL tolerates daemon writes)."""
    return _day_metrics_unp(day_str)


def _day_metrics_unp(day_str: str) -> pd.DataFrame:
    conn = _open_readonly_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        rows = employee_day_metrics(conn, date.fromisoformat(day_str))
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    return _to_frame(rows)


@st.cache_data(ttl=DASH_RANGE_METRICS_TTL_SEC, show_spinner=False)
def cached_range_metrics(start_str: str, end_str: str) -> pd.DataFrame:
    return _range_metrics_unp(start_str, end_str)


def _range_metrics_unp(start_str: str, end_str: str) -> pd.DataFrame:
    conn = _open_readonly_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        by_day = employee_range_metrics(conn, date.fromisoformat(start_str),
                                        date.fromisoformat(end_str))
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    frames = []
    for d, rows in sorted(by_day.items()):
        df = _to_frame(rows)
        if not df.empty:
            df["date"] = d.isoformat()
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["productive_pct"] = pd.to_numeric(df["productive_pct"], errors="coerce")
    return df


def _read_live_file() -> dict | None:
    """Raw read of the daemon snapshot; None if absent/unreadable."""
    try:
        if not _LIVE_STATE_FILE.exists():
            return None
        with open(_LIVE_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_live_state() -> dict:
    """Read the daemon's live snapshot; returns {} if stale/absent.

    ``live_state_age()`` still reports the true age of a stale file so the
    dashboard can distinguish "daemon stopped / data stale" from "no data".
    """
    data = _read_live_file()
    if not data:
        return {}
    if time.time() - data.get("updated", 0) > 60:
        return {}
    return data


def live_state_age() -> float | None:
    """Seconds since the snapshot was written; None when no snapshot exists."""
    data = _read_live_file()
    if not data:
        return None
    return time.time() - data.get("updated", 0)


def _open_write_conn():
    from src.database import get_connection
    return get_connection(DB_PATH)


# ----------------------------------------------------------------------
# Face-enrollment helpers (dashboard onboarding, Phase 28)
# ----------------------------------------------------------------------
_ENROLL_TEXT = {
    "ENROLLED": "Face enrolled and ready for recognition.",
    "MULTIPLE_FACES": "More than one face in the photo; the largest is used.",
    "NO_FACE": "No face detected in the image.",
    "LOW_QUALITY": "Face too small or unusable -- upload a clearer photo.",
    "INVALID_IMAGE": "Image could not be read; try another format.",
}
_ENROLL_COLOR = {
    "ENROLLED": "#2ecc71",
    "MULTIPLE_FACES": "#e67e22",
    "NO_FACE": "#e74c3c",
    "LOW_QUALITY": "#e67e22",
    "INVALID_IMAGE": "#e74c3c",
}


def _faces_dir() -> Path:
    return Path(FACES_DIR)


_SAFE_EMP_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_employee_id(eid: str) -> str:
    """Return a filesystem-safe employee id, or ``''`` if unsafe.

    Prevents path traversal: the id is embedded in filenames (``data/faces/``).
    We allow only letters, digits, ``_`` and ``-`` so no ``..``, separators, or
    absolute paths can escape the intended directory (B14).
    """
    eid = str(eid).strip()
    if not eid or not _SAFE_EMP_ID_RE.match(eid):
        return ""
    return eid


def save_enrollment_image(emp_id: str, filename: str, data: bytes) -> tuple[bool, str]:
    """Save an uploaded photo as ``data/faces/<EMP_ID><ext>``.

    Returns ``(ok, message)``.  Never touches biometric data -- only the raw
    image file, which the face registry consumes on rebuild.  The employee id
    is sanitised so it cannot traverse out of ``data/faces/`` (B14).
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".bmp"):
        return False, f"Unsupported image type '{suffix or '(none)'}'. Use JPG/PNG/BMP."
    eid = _safe_employee_id(emp_id)
    if not eid or eid.lower() in ("unknown", "__person__"):
        return False, (f"'{emp_id}' is not a valid employee id "
                       "(letters, digits, _ and - only).")
    target = _faces_dir() / f"{eid}{suffix}"
    try:
        _faces_dir().mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    except OSError as exc:
        return False, f"Could not write image: {exc}"
    return True, f"Saved {target.name} into {_faces_dir()}"


def cached_enrollment_status() -> dict:
    """Per-employee enrollment status straight from the embeddings cache.

    Read-only and fast (no face model load); falls back to {} when the model
    has never been built.  The cheap path the dashboard shows between full
    validations.
    """
    from config import EMBEDDINGS_FILE
    path = Path(EMBEDDINGS_FILE)
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return dict(data.get("file_status", {}) or {})
    except Exception:
        return {}


def enrollment_status_frame(status_map: dict) -> pd.DataFrame:
    """Render per-employee enrollment status as a readable dataframe."""
    return pd.DataFrame([
        {
            "Employee ID": eid,
            "Status": str(sts).replace("_", " "),
            "What this means": _ENROLL_TEXT.get(sts, sts),
        }
        for eid, sts in sorted(status_map.items())
    ])


# ======================================================================
# Authentication (Phase 11)
# ======================================================================

def _auth_enabled() -> bool:
    return DASH_AUTH_ENABLED and bool(DASH_ADMIN_PASS or DASH_VIEWER_PASS)


def _auth_fail_closed() -> bool:
    """Production guard: auth intended but no password set -> block access."""
    return (DASH_FAIL_CLOSED and DASH_AUTH_ENABLED
            and not bool(DASH_ADMIN_PASS or DASH_VIEWER_PASS))


def do_auth() -> str:
    """Return the role ('admin' / 'viewer') after login, or '' if not authed."""
    # Production fail-closed must gate BEFORE the _auth_enabled() shortcut:
    # when no password is configured _auth_enabled() is False and, without this
    # guard, the dashboard silently opened as admin despite DASH_FAIL_CLOSED=1.
    if _auth_fail_closed():
        st.error(
            "Dashboard authentication is enabled but no admin/viewer password "
            "is configured. Access is blocked (fail-closed). Set "
            "CCTV_DASH_PASS (and optionally CCTV_DASH_VIEWER_PASS) in your "
            "environment / .env to enable the dashboard. For local development "
            "only, set CCTV_DASH_AUTH=0."
        )
        return ""
    if not _auth_enabled():
        return "admin"
    if "cctv_role" in st.session_state:
        if DASH_SESSION_MINUTES > 0:
            ts = st.session_state.get("cctv_login_ts")
            if ts is None or (time.time() - ts) > DASH_SESSION_MINUTES * 60:
                st.session_state.pop("cctv_role", None)
                st.session_state.pop("cctv_login_ts", None)
                st.warning("Session expired. Please sign in again.")
            else:
                return st.session_state["cctv_role"]
        else:
            return st.session_state["cctv_role"]
    st.markdown("### Login")
    st.caption("Admin: full control (employees, enrollment). "
               "Viewer: monitor, analytics and report downloads only.")
    with st.form("login"):
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in")
    if submit:
        role = _authenticate(user, pw)
        if role:
            st.session_state["cctv_role"] = role
            st.session_state["cctv_login_ts"] = time.time()
            st.rerun()
        else:
            st.error("Invalid credentials.")
    return ""


def _authenticate(user: str, pw: str) -> str:
    """Validate dashboard credentials; returns 'admin' / 'viewer' / ''.

    Admin is only granted against a non-empty configured admin password: an
    empty ``CCTV_DASH_PASS`` must never let a blank password escalate to admin
    (previously, with only ``CCTV_DASH_VIEWER_PASS`` set, ``user="admin"`` and
    ``pw=""`` matched ``pw == DASH_ADMIN_PASS`` and granted admin).
    """
    if DASH_ADMIN_PASS and user == DASH_USERNAME and pw == DASH_ADMIN_PASS:
        return "admin"
    if DASH_VIEWER_PASS and pw == DASH_VIEWER_PASS:
        return "viewer"
    return ""


# ======================================================================
# Reusable rendering helpers
# ======================================================================

def _status_badge(state: str) -> str:
    """Plain-text live status for table cells.

    Deliberately NOT raw Streamlit markdown: placing ':#e74c3c[●] **AWAY**'
    into a DataFrame cell renders the literal syntax (dashboard artifact that
    was reported).  Reads as e.g. 'ACTIVE', 'ON PHONE', 'AWAY',
    'NOT OBSERVED'.
    """
    label = str(state).strip() if state is not None else ""
    if not label or label.upper() in ("NOT_OBSERVED", "NOT OBSERVED"):
        return "NOT OBSERVED"
    return label.replace("_", " ")


def _live_overview_metrics(df_today: pd.DataFrame, live: dict) -> dict:
    # "Recognised Today" must count employees with real observation, not
    # every roster row (NOT OBSERVED rows inflate the headline number).
    recognised = 0
    if not df_today.empty:
        if "status" in df_today.columns:
            recognised = int((df_today["status"].fillna("") == "OBSERVED").sum())
        else:
            recognised = len(df_today)
    avg_pct = round(df_today["productive_pct"].mean(), 1) \
        if not df_today.empty and df_today["productive_pct"].notna().any() else 0.0
    total_active = round(df_today["active_hours"].sum(), 2) if not df_today.empty else 0.0
    phone_mins = round(df_today["phone_mins"].sum(), 1) if not df_today.empty else 0.0

    states = live.get("states", {})
    unknown = len([e for e in states if str(e).lower() == "unknown"])
    employee_states = {e: s for e, s in states.items()
                       if str(e).lower() != "unknown"}
    present_now = len([s for s in employee_states.values()
                       if s not in ("AWAY", "", "NOT_OBSERVED", "NOT OBSERVED")])

    health = live.get("camera_health", {})
    cams_online = sum(1 for h in health.values() if h.get("health") == "ONLINE")
    cams_total = len(health)
    return {
        "recognised": recognised,
        "avg_pct": avg_pct,
        "total_active": total_active,
        "phone_mins": phone_mins,
        "present_now": present_now,
        "unknown": unknown,
        "cams_online": cams_online,
        "cams_total": cams_total,
    }


# ======================================================================
# Phase 42 -- premium SOC design system (UI only; no backend changes)
# ======================================================================

_NAV = [
    ("Dashboard", "dashboard", ":material/dashboard:"),
    ("Live Monitoring", "live", ":material/videocam:"),
    ("Employees", "employees", ":material/people:"),
    ("Security", "security", ":material/shield:"),
    ("Reports", "reports", ":material/description:"),
    ("Analytics", "analytics", ":material/query_stats:"),
    ("Cameras", "cameras", ":material/photo_camera:"),
    ("Admin Control Center", "admin", ":material/settings_suggest:"),
    ("Settings", "settings", ":material/settings:"),
    ("Deploy / System Check", "deploy", ":material/rocket_launch:"),
]
_NAV_LABELS = [f"{icon} {label}" for label, _key, icon in _NAV]
_NAV_KEYS = {f"{icon} {label}": key for label, key, icon in _NAV}

_STATE_VIEW = {
    "ACTIVE":       ("ACTIVE",       "green",  "timelapse",      "Present and active."),
    "ON_PHONE":     ("ON PHONE",     "orange", "phone_iphone",   "On a personal phone call."),
    "ON BREAK":     ("ON BREAK",     "gray",   "coffee",         "On a scheduled break."),
    "AWAY":         ("AWAY",         "gray",   "person_off",     "Away from the station."),
    "NOT_OBSERVED": ("NOT OBSERVED", "gray",   "visibility_off", "On roster; not observed this session."),
    "UNKNOWN":      ("UNKNOWN",      "violet", "question_mark",  "Unrecognised person -- never attributed to an employee."),
}

_CAMERA_VIEW = {
    "ONLINE":          ("ONLINE",          "green",  "videocam"),
    "LOW_FPS":         ("LOW FPS",         "orange", "speed"),
    "RECONNECTING":    ("RECONNECTING",    "orange", "sync"),
    "NO_FRAME":        ("NO FRAME",        "red",    "videocam_off"),
    "OFFLINE":         ("OFFLINE",         "red",    "videocam_off"),
    "FROZEN_FRAME":    ("FROZEN FRAME",    "violet", "pause_circle"),
    "NOT_CONFIGURED":  ("NOT CONFIGURED",  "gray",   "settings_input_antenna"),
}

_CHIP_TONES = {
    "green":  {"bg": "#0E271D", "fg": "#9BE8C4", "bd": "#1F6B45"},
    "orange": {"bg": "#291E11", "fg": "#F6C794", "bd": "#7A5324"},
    "red":    {"bg": "#2A1412", "fg": "#F0A39B", "bd": "#7E2C25"},
    "gray":   {"bg": "#151D2B", "fg": "#B7C0CE", "bd": "#38445A"},
    "violet": {"bg": "#201628", "fg": "#CDBCF7", "bd": "#5A468C"},
    "blue":   {"bg": "#14233F", "fg": "#BAD3FF", "bd": "#2F5ED6"},
}

_COMPONENT_CSS = """
<style>
.p42 { font-family:"Inter","Segoe UI",system-ui,sans-serif; color:#E6EDF7; }
.p42-ic { font-family:"Material Symbols Outlined"; font-weight:400; font-size:15px;
          line-height:1; vertical-align:-2px; }
.p42-chip { display:inline-flex; align-items:center; gap:5px; padding:3px 10px;
            border-radius:999px; font-size:11px; font-weight:600; letter-spacing:.5px;
            border:1px solid; white-space:nowrap; }
.p42-chip-lg { font-size:12px; padding:4px 12px; }
.p42-chip-sm { font-size:10px; padding:2px 7px; }
.p42-strip { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:2px 0 12px; }
.p42-strip .p42-mut { color:#8B96A8; font-size:12px; }
.p42-topbar { display:flex; align-items:center; justify-content:space-between;
              gap:16px; padding:14px 2px 12px; border-bottom:1px solid #1E2A40; }
.p42-brand { display:flex; align-items:center; gap:12px; }
.p42-brand-mark { width:40px; height:40px; border-radius:10px; flex:0 0 auto;
                  background:#14213D; border:1px solid #1E2A40; display:grid; place-items:center;
                  color:#4C7DF0; font-size:20px; }
.p42-brand-name { font-size:19px; font-weight:700; letter-spacing:.3px; line-height:1.2; }
.p42-brand-sub { font-size:10.5px; color:#8B96A8; letter-spacing:1.8px; text-transform:uppercase; }
.p42-topbar-right { display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
.p42-role-badge { display:inline-flex; align-items:center; gap:6px; font-size:11px;
                  font-weight:600; letter-spacing:.8px; text-transform:uppercase;
                  color:#BAD3FF; border:1px solid #2F5ED6; background:#14233F;
                  padding:3px 10px; border-radius:999px; }
.p42-clock { text-align:right; font-size:12px; }
.p42-clock b { font-size:15px; font-weight:700; letter-spacing:.4px; }
.p42-clock span { color:#8B96A8; }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr));
         gap:12px; margin:6px 0 14px; }
.p42-card { background:#111A2B; border:1px solid #1E2A40; border-radius:12px; padding:14px 16px; }
.p42-cam-head { display:flex; align-items:center; justify-content:space-between; gap:8px; }
.p42-cam-name { font-weight:700; font-size:14px; }
.p42-cam-id { font-size:10px; color:#8B96A8; letter-spacing:.8px; text-transform:uppercase; margin-top:1px; }
.p42-cam-meta { display:grid; grid-template-columns:1fr 1fr; gap:9px; margin-top:11px; font-size:12px; }
.p42-cam-meta b { display:block; font-size:9.5px; color:#8B96A8; letter-spacing:.9px;
                  text-transform:uppercase; font-weight:600; }
.p42-cam-meta span { color:#C9D4E3; }
.p42-cam-note { margin-top:11px; font-size:11.5px; color:#F0A39B; }
.p42-person { display:flex; gap:12px; align-items:flex-start; }
.p42-avatar { width:38px; height:38px; border-radius:50%; flex:0 0 auto; display:grid;
              place-items:center; font-weight:700; font-size:13px; color:#E6EDF7;
              background:#1A2B4D; border:1px solid #1E2A40; }
.p42-avatar.unk { background:#241D33; color:#CDBCF7; }
.p42-person .p42-body { min-width:0; flex:1; }
.p42-person-name { font-weight:700; font-size:14px; line-height:1.25; }
.p42-person-id { font-size:10.5px; color:#8B96A8; letter-spacing:.6px; }
.p42-person-meta { margin-top:8px; display:flex; flex-direction:column; gap:3px; font-size:12px; }
.p42-person-meta .p42-row { display:flex; justify-content:space-between; gap:10px; }
.p42-person-meta .p42-row span:first-child { color:#8B96A8; }
.p42-person-meta .p42-row span:last-child { color:#C9D4E3; }
.p42-person-meta .p42-row.p42-phone { color:#F6C794; }
.p42-empty { border:1px dashed #1E2A40; border-radius:12px; padding:20px; color:#8B96A8; font-size:13px; }
.p42-footer { margin-top:34px; padding-top:12px; border-top:1px solid #1E2A40;
              color:#8B96A8; font-size:11px; display:flex; gap:18px; flex-wrap:wrap; }
.p42-mut { color:#8B96A8; font-size:11px; }

/* ---- Phase 65 premium SOC upgrade ---- */
.p42-page-head { padding:2px 0 14px; border-bottom:1px solid #1E2A40; margin:0 0 14px; }
.p42-page-title { font-size:21px; font-weight:800; letter-spacing:.2px; color:#EFF3F9; }
.p42-page-sub { font-size:12.5px; color:#8B96A8; margin-top:4px; max-width:900px; line-height:1.5; }
.p42-panel { border-radius:12px; padding:20px 22px; margin:8px 0 16px; display:flex;
             align-items:flex-start; gap:14px; border:1px solid; }
.p42-panel .p42-panel-badge { font-size:24px; line-height:1; flex:0 0 auto; }
.p42-panel .p42-panel-title { font-weight:800; font-size:15px; letter-spacing:.3px; }
.p42-panel .p42-panel-msg { font-size:13px; margin-top:4px; line-height:1.55; }
.p42-panel-off { background:#0E1624; border-color:#263450; }
.p42-panel-off .p42-panel-title { color:#C9D4E3; }
.p42-panel-off .p42-panel-msg { color:#93A0B5; }
.p42-panel-warn { background:#201A10; border-color:#5A441F; }
.p42-panel-warn .p42-panel-title { color:#F6C794; }
.p42-panel-warn .p42-panel-msg { color:#D6C2A0; }
.p42-error-card { background:#2A1412; border:1px solid #7E2C25; border-radius:12px;
                  padding:18px 20px; margin:8px 0 16px; }
.p42-error-card .p42-error-title { color:#F0A39B; font-weight:800; font-size:14px;
                                   letter-spacing:.6px; text-transform:uppercase; }
.p42-error-card .p42-error-msg { color:#C9D4E3; font-size:13px; margin-top:6px; line-height:1.5; }
.p42-diag-id { font-size:10.5px; color:#8B96A8; margin-top:10px;
               font-family:"Cascadia Code","Consolas",monospace; }
.p42-nav-legend { font-size:10px; letter-spacing:1.1px; color:#5E6B82;
                  text-transform:uppercase; margin:2px 0 6px; line-height:1.7; }
.p42-nav-legend b { color:#8B96A8; font-weight:700; }
.p42-session-card { border:1px solid #1E2A40; border-radius:12px; background:#0E1624;
                    padding:12px 14px; margin-top:6px; }
.p42-session-card .p42-s-row { display:flex; justify-content:space-between; gap:10px;
                               font-size:12.5px; padding:2px 0; }
.p42-session-card .p42-s-row span:first-child { color:#8B96A8; }
.p42-session-card .p42-s-row span:last-child { color:#C9D4E3; font-weight:600; }
.p42-card:hover { border-color:#2E4676; box-shadow:0 4px 18px rgba(0,0,0,.35); }
.p42-kpi-accent { border-top:2px solid #4C7DF0; }
</style>
"""


def _view_state(state) -> dict:
    """Normalize a live state to (label, tone, icon, hint).  'Unknown' stays
    neutral violet -- never red, never an employee.  Pure, never throws."""
    key = str(state or "").strip().upper()
    row = _STATE_VIEW.get(key)
    if row is not None:
        label, tone, icon, hint = row
        return {"key": key, "label": label, "tone": tone, "icon": icon, "hint": hint}
    label = str(state).replace("_", " ") or "—"
    return {"key": key, "label": label, "tone": "gray", "icon": "circle", "hint": ""}


def _view_camera(code) -> dict:
    key = str(code or "OFFLINE").strip().upper()
    row = _CAMERA_VIEW.get(key)
    if row is None:
        return {"key": key, "label": key.replace("_", " "), "tone": "gray", "icon": "videocam"}
    label, tone, icon = row
    return {"key": key, "label": label, "tone": tone, "icon": icon}


def _chip_html(view: dict, size: str = "") -> str:
    tone = _CHIP_TONES.get(view["tone"], _CHIP_TONES["gray"])
    cls = "p42-chip" if not size else f"p42-chip {size}"
    return (f"<span class='{cls}' style='background:{tone['bg']};color:{tone['fg']};"
            f"border-color:{tone['bd']};'>"
            f"<span class='p42-ic'>{view['icon']}</span>{html.escape(view['label'])}</span>")


def _safe_cam_source(src) -> str:
    src = str(src or "")
    if not src:
        return "—"
    if "://" in src:
        return "RTSP (hidden)"
    return src


def _live_kpis(live: dict, df_today: pd.DataFrame) -> dict:
    """Real-data-only KPI summary used by the dashboard KPI cards.

    ``active``/``away``/``on_phone`` come from the live snapshot states;
    ``open_incidents`` from the security snapshot.  Nothing is ever invented:
    with no snapshot the live-derived fields are zero, never guessed.
    """
    m = _live_overview_metrics(df_today, live)
    states = live.get("states", {}) if live else {}
    emp_states = {k: v for k, v in states.items()
                  if not str(k).lower().startswith("unknown")}
    cam_health = live.get("camera_health", {}) if live else {}
    security = live.get("security", {}) if live else {}
    return {
        "recognised": m["recognised"],
        "avg_pct": m["avg_pct"],
        "total_active": m["total_active"],
        "phone_mins": m["phone_mins"],
        "active": sum(1 for s in emp_states.values() if str(s).upper() == "ACTIVE"),
        "on_phone": sum(1 for s in emp_states.values() if str(s).upper() == "ON_PHONE"),
        "away": sum(1 for s in emp_states.values() if str(s).upper() == "AWAY"),
        "unknown": sum(1 for k in states if str(k).lower().startswith("unknown")),
        "present": m["present_now"],
        "cameras_online": sum(1 for h in cam_health.values()
                              if str(h.get("health", "")).upper() == "ONLINE"),
        "cameras_total": len(cam_health),
        "employees_total": len(live.get("employees", {})) if live else 0,
        "open_incidents": security.get("open_incidents"),
    }


def _snapshot_status(live: dict) -> dict:
    """Classify the daemon snapshot as LIVE / STALE / OFFLINE (never pretends)."""
    if not live:
        age = live_state_age()
        if age is None:
            return {"kind": "offline", "label": "NO SNAPSHOT", "tone": "gray",
                    "icon": "link_off", "iso": "—", "age": None}
        return {"kind": "stale", "label": "STALE", "tone": "orange",
                "icon": "schedule", "iso": "—", "age": age}
    return {"kind": "live", "label": "LIVE", "tone": "green",
            "icon": "monitor_heart", "iso": live.get("iso", "—"),
            "age": time.time() - live.get("updated", time.time())}


def _inject_shell_css() -> None:
    """Global (app-level) scoped styles: metrics, sidebar, tables, expanders.

    Rendered once per session via ``st.markdown`` so it styles the host app
    document (``st.html`` output lives in a sandboxed iframe and cannot reach
    native widgets).  Selectors are scoped to Streamlit test ids / element keys
    so no external page is affected.
    """
    if st.session_state.get("_p42_css_injected"):
        return
    st.session_state["_p42_css_injected"] = True
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

html, body, [class*="st-emotion-cache"], [data-testid="stAppViewContainer"] {
    font-family: "Inter","Segoe UI",system-ui,sans-serif;
}
[data-testid="stAppViewContainer"] { background:#0A0F1C; }
h1,h2,h3,h4 { letter-spacing:.2px; color:#E6EDF7; }
p, label, span, div { color:#E6EDF7; }

[data-testid="stSidebar"] { background:#0C1322; border-right:1px solid #1E2A40; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color:#C9D4E3; }
section[data-testid="stSidebar"] .p42-nav-label {
    font-size:10.5px; letter-spacing:1.4px; text-transform:uppercase; color:#8B96A8; margin-bottom:4px;
}

div[data-testid="stMetric"] {
    background:#111A2B; border:1px solid #1E2A40; border-radius:12px;
    padding:12px 16px 14px; box-shadow:0 1px 0 rgba(255,255,255,.02);
}
div[data-testid="stMetricLabel"] {
    font-size:10.5px; letter-spacing:1.1px; text-transform:uppercase; color:#8B96A8;
}
div[data-testid="stMetricValue"] { font-weight:700; font-size:1.65rem; }

[data-testid="stDataFrame"] {
    border:1px solid #1E2A40; border-radius:10px; overflow:hidden; background:#0E1624;
}
[data-testid="stDataFrame"] thead th {
    background:#0F1626; color:#8B96A8; font-size:11px; letter-spacing:.6px; text-transform:uppercase;
}
[data-testid="stDataFrame"] tbody tr:hover { background:rgba(76,125,240,.07); }

[data-testid="stExpander"] {
    border:1px solid #1E2A40; border-radius:12px; background:#0E1624;
}

section[data-testid="stSidebar"] .st-key-p42_nav [data-testid="stRadio"] label p {
    font-size:13px; padding:6px 10px; border-radius:8px; margin:1px 0;
}
section[data-testid="stSidebar"] .st-key-p42_nav [data-testid="stRadio"] label:hover p {
    background:#14203A;
}
section[data-testid="stSidebar"] .st-key-p42_nav p { color:#C9D4E3; }

div[data-testid="stVerticalBlockBorderWrapper"] { background:#0E1624; }

.stButton > button, .stButton > button[kind="primary"] {
    border-radius:9px; font-weight:600;
}
.stButton > button[kind="primary"] { background:#2A47A8; border:1px solid #4C7DF0; }
.stSidebar button:hover { border-color:#4C7DF0; }

div[data-testid="stHorizontalBlock"] { gap:0.75rem; }
hr { border-color:#1E2A40 !important; }

/* ---- Phase 65: premium SOC shell refinements ---- */
section[data-testid="stSidebar"] .st-key-p42_nav [role="radiogroup"] label[aria-checked="true"] p {
    background:#1B2C52; border:1px solid #3B67C9; color:#DCE7FF;
    box-shadow:inset 2px 0 0 #4C7DF0;
}
section[data-testid="stSidebar"] .st-key-p42_nav [role="radiogroup"] label[aria-checked="true"] {
    background:transparent;
}
div[data-testid="stMetric"] { border-top:2px solid #4C7DF0; box-shadow:0 2px 10px rgba(0,0,0,.25); }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] > div > div:first-child {
    padding-top:10px;
}
[data-testid="stAppViewContainer"]::before {
    content:""; position:fixed; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,#4C7DF0 0%,#2A47A8 45%,#0E1624 100%);
    z-index:9999;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_topbar(role: str) -> None:
    snap = _snapshot_status(load_live_state())
    clock = time.strftime("%H:%M:%S")
    day = time.strftime("%a, %d %b %Y")
    role_badge = (f"<span class='p42-role-badge'><span class='p42-ic'>admin_panel_settings</span>"
                  f"Admin</span>") if role == "admin" else (
                  f"<span class='p42-role-badge'><span class='p42-ic'>visibility</span>Viewer</span>")
    sys_label = {"live": "SYSTEM ONLINE", "stale": "SYSTEM DEGRADED",
                 "offline": "DAEMON OFFLINE"}[snap["kind"]]
    sys_tone = {"live": "green", "stale": "orange", "offline": "gray"}[snap["kind"]]
    updated = (f"Last updated {html.escape(str(snap['iso']))}"
               if snap["iso"] != "—" else "Last updated —")
    st.html(
        _COMPONENT_CSS + f"""
<div class='p42 p42-topbar'>
  <div class='p42-brand'>
    <div class='p42-brand-mark'><span class='p42-ic'>security</span></div>
    <div>
      <div class='p42-brand-name'>AI CCTV Intelligence</div>
      <div class='p42-brand-sub'>Security Operations Center</div>
    </div>
  </div>
  <div class='p42-topbar-right'>
    {_chip_html({"label": sys_label, "tone": sys_tone, "icon": snap["icon"]}, "p42-chip-lg")}
    {_chip_html({"label": f"Snapshot: {snap['label']}", "tone": snap["tone"], "icon": snap["icon"]})}
    <span class='p42-mut'>{updated}</span>
    {role_badge}
    <div class='p42-clock'><b>{clock}</b><br/><span>{day}</span></div>
  </div>
</div>
"""
    )


def _render_snapshot_strip(live: dict) -> None:
    snap = _snapshot_status(live)
    parts = [
        _chip_html({"label": snap["label"], "tone": snap["tone"], "icon": snap["icon"]}),
    ]
    if snap["kind"] == "live":
        parts.append(f"<span class='p42-mut'>Snapshot {html.escape(str(snap['iso']))}"
                     f" &middot; {snap['age']:.0f}s ago</span>")
    elif snap["kind"] == "stale" and isinstance(snap["age"], (int, float)):
        parts.append(f"<span class='p42-mut'>Last snapshot {snap['age']:.0f}s ago"
                     " &middot; daemon may be stopped</span>")
    else:
        parts.append("<span class='p42-mut'>Start the daemon: "
                     "`python main.py --source auto --headless --no-email`</span>")
    if live:
        fps = live.get("fps", 0.0)
        det = live.get("last_detection_sec")
        det_note = "no detection yet"
        if isinstance(det, (int, float)):
            det_note = f"last detection {int(det)}s ago" if det < 3600 \
                else f"last detection {det / 3600:.1f}h ago"
        parts.append(f"<span class='p42-mut'>Daemon FPS {fps:.1f} &middot; {det_note}</span>")
    st.html(_COMPONENT_CSS + f"<div class='p42 p42-strip'>{''.join(parts)}</div>")


def _panel_html(kind: str, icon: str, title: str, msg: str) -> str:
    """Premium status panel (off/warn). ``msg`` may contain inline HTML."""
    cls = ("p42-panel p42-panel-warn" if kind == "warn"
           else "p42-panel p42-panel-off")
    return (
        f"<div class='{cls}'>"
        f"<span class='p42-ic p42-panel-badge'>{icon}</span>"
        f"<div><div class='p42-panel-title'>{html.escape(title)}</div>"
        f"<div class='p42-panel-msg'>{msg}</div></div></div>"
    )


def _render_daemon_offline_panel() -> None:
    """Honest premium empty-state: no live inference snapshot available.

    Never presented as a security incident and never fabricated.  Explains
    that the dashboard is online while the inference daemon is not running.
    """
    st.html(_COMPONENT_CSS + _panel_html(
        "off", "sensors_off",
        "DAEMON OFFLINE — No live inference snapshot available.",
        "The Streamlit dashboard is online, but the CCTV inference daemon is "
        "not running in this environment. Start it on the local/office machine "
        "with <code>python main.py --source auto --headless --no-email</code>. "
        "Configuration, settings, reports, analytics and historical data remain "
        "fully available."
    ))


def _render_error_card(title: str, msg: str) -> None:
    """Clean user-facing error card -- raw tracebacks are never shown.

    The technical detail is logged internally; the UI gets a safe diagnostic
    id so support can cross-reference logs without exposing internals.
    """
    logging.getLogger("cctv.dashboard").error("UI error [%s]: %s", title, msg)
    diag = f"{int(time.time()) % 100000:05d}"
    st.html(
        _COMPONENT_CSS +
        "<div class='p42 p42-error-card'>"
        f"<div class='p42-error-title'>{html.escape(title)}</div>"
        f"<div class='p42-error-msg'>{html.escape(msg)}</div>"
        f"<div class='p42-diag-id'>Diagnostic id <b>{diag}</b> — this incident "
        "was logged internally.</div></div>"
    )


def _render_dashboard_metrics(live: dict, m: dict, df_today: pd.DataFrame = None) -> None:
    """Premium 9-card SOC KPI grid (real data only, never invented).

    Layout preserves the existing metric labels and values (regression tests
    assert them); the 9 KPIs match the SOC overview spec.
    """
    k = _live_kpis(live, pd.DataFrame())
    cams = f"{k['cameras_online']}/{k['cameras_total']}" if k["cameras_total"] else "—"
    open_inc = "—" if k["open_incidents"] is None else str(k["open_incidents"])
    has_data = df_today is not None and not df_today.empty
    r1a, r1b, r1c = st.columns(3)
    r1a.metric("Cameras Online", cams,
               help="Operational camera feeds (health == ONLINE). A camera that is "
                    "offline / not delivering usable frames is never counted as AWAY.")
    r1b.metric("People Active", f"{k['active']}",
               help="Employees currently ACTIVE in the live snapshot.")
    r1c.metric("Away", f"{k['away']}",
               help="Employees currently AWAY (from the daemon, not a frontend timer).")
    r2a, r2b, r2c = st.columns(3)
    r2a.metric("On Phone", f"{k['on_phone']}",
               help="Employees currently in ON_PHONE state (one alert per episode).")
    r2b.metric("Open Incidents", open_inc,
               help="Open incidents from the security snapshot (real security events).")
    r2c.metric("Recognised Today", m["recognised"] if has_data else "n/a",
               help="Employees with at least one real observation today (roster "
                    "rows that were never observed are not counted).")
    r3a, r3b, r3c = st.columns(3)
    r3a.metric("Average Productivity", f"{m['avg_pct']:.0f}%" if m["avg_pct"] else "n/a",
               help="Mean productive percentage across observed employees today.")
    r3b.metric("Active Hours Today", f"{m['total_active']:.1f}h" if has_data else "n/a",
               help="Total active hours logged so far today (real analytics).")
    r3c.metric("Phone Today", f"{m['phone_mins']:.0f}m" if has_data else "n/a",
               help="Total phone minutes observed today (real analytics).")


def _render_camera_cards(live: dict) -> None:
    if not live:
        st.html(_COMPONENT_CSS + "<div class='p42 p42-empty'>No live daemon snapshot — "
                "camera status is unavailable while `main.py` is not running.</div>")
        return
    health = live.get("camera_health", {})
    if not health:
        st.html(_COMPONENT_CSS + "<div class='p42 p42-empty'>No cameras configured.</div>")
        return
    try:
        import config as _cfg
        configured = set(_cfg.CAMERAS.keys()) | {c.get("source") or c.get("id")
                                                 for c in health.values()}
    except Exception:  # pragma: no cover
        configured = {c.get("id") for c in health.values()}
    cards = []
    for cid in sorted(health):
        h = health[cid]
        code = h.get("health", "OFFLINE")
        view = _view_camera(code)
        chip = _chip_html(view)
        last = h.get("last_frame", 0)
        last_s = time.strftime("%H:%M:%S", time.localtime(last)) if last else "—"
        fps = ("—" if str(code).upper() == "OFFLINE"
               else f"{round(float(h.get('fps', 0) or 0), 2)}")
        src = h.get("source") or h.get("id") or cid
        source = _safe_cam_source(src)
        configured_s = ("Configured" if (src in configured or cid in configured)
                        else "Not configured")
        usable = h.get("usable")
        usable_s = "—" if usable is None else ("Usable frames" if usable else "Not usable")
        note = ""
        if str(code).upper() in ("OFFLINE", "NO_FRAME", "RECONNECTING") or \
                h.get("unusable_reason") == CAM_DARK_BLANK_FRAME:
            reason = h.get("unusable_reason", "") or ""
            note = ("Camera not delivering usable frames — this is <b>not</b> employee "
                    "absence and never counts as AWAY." if not reason else
                    f"Feed not usable ({reason}) — preserved states are frozen; "
                    "this is never counted as AWAY.")
        kind = h.get("kind") or "rtsp"
        res = h.get("resolution") or "—"
        rconn = int(h.get("reconnects", 0) or 0)
        cards.append(
            f"<div class='p42-card'><div class='p42-cam-head'>"
            f"<div><div class='p42-cam-name'>{html.escape(cid)}</div>"
            f"<div class='p42-cam-id'>{html.escape(configured_s)}</div></div>{chip}</div>"
            f"<div class='p42-cam-meta'>"
            f"<div><b>Last frame</b><span>{last_s}</span></div>"
            f"<div><b>FPS</b><span>{fps}</span></div>"
            f"<div><b>Source</b><span>{html.escape(source)}</span></div>"
            f"<div><b>Type</b><span>{html.escape(kind)}</span></div>"
            f"<div><b>Resolution</b><span>{html.escape(res)}</span></div>"
            f"<div><b>Reconnects</b><span>{rconn}</span></div>"
            f"<div><b>Frames</b><span>{h.get('frames_read', 0)}</span></div>"
            f"</div>{f'<div class=\'p42-cam-note\'>{note}</div>' if note else ''}</div>"
        )
    st.html(_COMPONENT_CSS + f"<div class='p42 cards'>{''.join(cards)}</div>")


def _initials(name: str) -> str:
    parts = [p for p in re.split(r"\s+", str(name or "").strip()) if p]
    if not parts:
        return "?"
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()


def _render_people_cards(live: dict) -> None:
    if not live:
        return
    states = live.get("states", {})
    employees = live.get("employees", {})
    last_seen_top = live.get("last_seen", {})
    durations = live.get("durations", {})
    telemetry = live.get("telemetry", {})
    now = time.time()

    def _fmt_secs(secs) -> str:
        secs = float(secs or 0)
        h, rem = divmod(int(secs), 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    cards = []
    for emp_id in sorted(employees):
        emp = employees[emp_id]
        state = emp.get("state", "")
        view = _view_state(state)
        name = emp.get("name", "") or emp_id
        ls = emp.get("last_seen")
        ls_s = time.strftime("%H:%M:%S", time.localtime(ls)) if ls else "—"
        ago = f"{(now - ls):.0f}s ago" if ls else "—"
        session = _fmt_secs(emp.get("session_sec", 0))
        source = _safe_cam_source(emp.get("source", ""))
        phone_sec = float(emp.get("phone_sec", 0) or 0)
        phone_state = _view_state(state)["key"] == "ON_PHONE"
        phone_row = ""
        if phone_state or phone_sec > 0:
            val = "NOW" if phone_state else f"{phone_sec:.0f}s"
            phone_row = (f"<div class='p42-row p42-phone'><span>Phone</span>"
                         f"<span class='p42-ic'>phone_iphone</span> {val}</div>")
        cards.append(
            f"<div class='p42-card'><div class='p42-person'>"
            f"<div class='p42-avatar'>{html.escape(_initials(name))}</div>"
            f"<div class='p42-body'>"
            f"<div class='p42-person-name'>{html.escape(name)}</div>"
            f"<div class='p42-person-id'>{html.escape(emp_id)}</div>"
            f"<div style='margin-top:7px'>{_chip_html(view)}</div>"
            f"<div class='p42-person-meta'>"
            f"<div class='p42-row'><span>Camera</span><span>{html.escape(source)}</span></div>"
            f"<div class='p42-row'><span>Last seen</span><span>{ls_s} · {ago}</span></div>"
            f"<div class='p42-row'><span>Session</span><span>{session}</span></div>"
            f"{phone_row}</div></div></div></div>"
        )

    unknown = [k for k in states if str(k).lower().startswith("unknown")]
    for k in sorted(unknown, key=lambda s: str(s).lower()):
        view = _view_state("UNKNOWN")
        ls = last_seen_top.get(k)
        ls_s = time.strftime("%H:%M:%S", time.localtime(ls)) if ls else "—"
        sess = _fmt_secs(durations.get(k, 0.0) or 0.0)
        cards.append(
            f"<div class='p42-card'><div class='p42-person'>"
            f"<div class='p42-avatar unk'><span class='p42-ic'>question_mark</span></div>"
            f"<div class='p42-body'>"
            f"<div class='p42-person-name'>{html.escape(str(k))}</div>"
            f"<div class='p42-person-id'>Not an employee · tracked separately</div>"
            f"<div style='margin-top:7px'>{_chip_html(view)}</div>"
            f"<div class='p42-person-meta'>"
            f"<div class='p42-row'><span>Last seen</span><span>{ls_s}</span></div>"
            f"<div class='p42-row'><span>Session</span><span>{sess}</span></div>"
            f"</div></div></div></div>"
        )

    if not cards:
        st.html(_COMPONENT_CSS + "<div class='p42 p42-empty'>No tracked people right "
                "now. Presence cards appear as soon as the daemon detects someone.</div>")
        return
    st.html(_COMPONENT_CSS + f"<div class='p42 cards'>{''.join(cards)}</div>")


def _render_footer(role: str) -> None:
    now = time.strftime("%H:%M:%S")
    st.html(
        _COMPONENT_CSS + f"""
<div class='p42 p42-footer'>
  <span>AI CCTV Intelligence · Security Operations Center</span>
  <span>Role: {html.escape(role)}</span>
  <span>Auto-refresh: {DASH_REFRESH_SEC}s</span>
  <span>Snapshot rendered at {now}</span>
  <span>Daemon: python main.py --source auto --headless --no-email</span>
</div>
"""
    )


def _show_kv_list(items: list) -> None:
    """Render label/value pairs as compact read-only rows (settings page)."""
    for label, value in items:
        st.markdown(f"- **{label}:** {value}")


def page_live_monitoring() -> None:
    live = load_live_state()
    st.html(
        _COMPONENT_CSS +
        "<div class='p42 p42-page-head'>"
        "<div class='p42-page-title'>Live Monitoring</div>"
        "<div class='p42-page-sub'>Live camera status, people presence and "
        "inference capabilities from the daemon snapshot — real backend data "
        "only, refreshed with the rest of this page.</div></div>",
    )
    _render_snapshot_strip(live)
    if not live:
        _render_daemon_offline_panel()
    _render_camera_cards(live)
    if live:
        _render_people_cards(live)
    else:
        st.html(_COMPONENT_CSS + _panel_html(
            "off", "group",
            "PEOPLE PRESENCE UNAVAILABLE",
            "Employee presence states come from the daemon snapshot. With the "
            "daemon offline there is no per-person presence to show — nothing "
            "here is simulated."
        ))
    tab_live_employees()
    tab_ai_capabilities()


# ======================================================================
# TABS
# ======================================================================

def _productivity_bar_figure(df_today: pd.DataFrame):
    """Bar figure of per-employee productivity; None if nothing plottable.

    Returns None when there are no rows with a numerable percentage (e.g. the
    roster exists but nobody has been observed yet) so callers show a friendly
    empty state instead of feeding an empty frame to plotly (which raises on
    some versions).
    """
    plotted = df_today.dropna(subset=["productive_pct"])
    if plotted.empty:
        return None
    text = plotted["productive_pct"].apply(
        lambda v: f"{v:.0f}%" if pd.notna(v) else "n/a")
    fig = px.bar(
        plotted,
        x="employee_id", y="productive_pct",
        color="productive_pct", color_continuous_scale="Greens",
        range_color=[0, 100],
        labels={"productive_pct": "Productivity (%)", "employee_id": "Employee"},
        text=text,
    )
    fig.update_layout(height=400, yaxis_range=[0, 110])
    fig.update_traces(textposition="outside")
    return fig


def tab_live_overview():
    st.header("Live Overview")
    st.html(
        _COMPONENT_CSS +
        "<div class='p42 p42-page-sub'>Real-time operational status from the "
        "daemon snapshot and today's analytics — auto-refresh every "
        f"{DASH_REFRESH_SEC}s (today: {date.today().isoformat()}). "
        "Every figure is real observed data: cameras offline or an absent "
        "daemon are never counted as AWAY, and nothing is fabricated.</div>",
    )
    today = date.today()
    df_today = cached_day_metrics(today.isoformat())
    live = load_live_state()
    age = live_state_age()
    if not live and age is not None and age > 60:
        st.html(_COMPONENT_CSS + _panel_html(
            "warn", "schedule",
            "LIVE DATA STALE — last daemon snapshot is "
            f"{age:.0f}s old.",
            "The snapshot is old; live KPIs below reflect the last written "
            "snapshot only. Start the daemon on the office machine with "
            "<code>python main.py --source auto --headless --no-email</code> "
            "to resume live tracking."
        ))

    _inject_shell_css()
    _render_snapshot_strip(live)
    m = _live_overview_metrics(df_today, live)

    if not live:
        _render_daemon_offline_panel()

    _render_dashboard_metrics(live, m, df_today)

    st.subheader("Today's Productivity per Employee")
    if df_today.empty:
        st.info("No data available yet today.\n\n"
                "Start the daemon to open the webcam and begin logging:\n\n"
                "    python main.py --source webcam --headless\n\n"
                "Add employees first under the **Employees** tab so their face "
                "can be recognised, then restart the daemon. KPIs and the bar "
                "chart populate as soon as the first ACTIVE / ON_PHONE / AWAY "
                "state is recorded.")
        return

    fig = _productivity_bar_figure(df_today)
    if fig is None:
        st.info("Employees are registered, but no observed activity has been "
                "logged yet today. Productivity bars appear once the daemon "
                "records ACTIVE / ON_PHONE / AWAY states.")
    else:
        st.plotly_chart(fig, width="stretch")

    if st.checkbox("Show detailed table", value=False):
        cols = ["employee_id", "name", "active_hours", "phone_mins",
                "away_mins", "unobserved_seconds", "productive_pct", "status"]
        cols = [c for c in cols if c in df_today.columns]
        st.dataframe(df_today[cols], width="stretch")


def tab_live_employees():
    st.header("Live Employee Table")
    live = load_live_state()
    if not live:
        age = live_state_age()
        if age is not None and age > 60:
            st.warning(
                f"**LIVE DATA STALE** -- the last daemon snapshot is {age:.0f}s old. "
                "Start `main.py --source auto --headless --no-email` to resume."
            )
        else:
            st.warning("No live daemon snapshot found. Is `main.py` running?")
        return

    states = live.get("states", {})
    cam_health = live.get("camera_health", {})
    fps = live.get("fps", 0.0)
    employees = live.get("employees", {})
    last_seen_top = live.get("last_seen", {})
    durations = live.get("durations", {})
    sources = live.get("sources", {})

    # -- Camera health warning ----------------------------------------------
    # Distinguish a dead feed (no frames / blank / dark) -- states are frozen,
    # never counted as AWAY -- from a FROZEN_FRAME feed (frames still flowing,
    # scene static: presence still monitored).
    _hard_offline = ("OFFLINE", "NO_FRAME", "RECONNECTING")
    dead = [c for c, h in cam_health.items()
            if h.get("health") in _hard_offline
            or h.get("unusable_reason") == CAM_DARK_BLANK_FRAME]
    frozen = [c for c, h in cam_health.items()
              if h.get("health") == "FROZEN_FRAME" and h.get("usable")]
    low_fps = [c for c, h in cam_health.items()
               if h.get("health") == "LOW_FPS"]
    if dead:
        cam_names = ", ".join(sorted(dead))
        st.warning(
            f"Camera {'is' if len(dead) == 1 else 'are'} not delivering usable frames: "
            f"**{cam_names}**. This is **not** employee absence and never counts as "
            "AWAY time. Employee states below are **frozen** until a usable frame "
            "arrives. If the feed is blank/dark, try `--source auto` or "
            "`--source <index>`."
        )
    if frozen:
        cam_names = ", ".join(sorted(frozen))
        st.info(
            f"Camera **{cam_names}** feed is flowing but the scene appears static "
            "(FROZEN_FRAME). Monitoring continues -- verify the lens isn't covered "
            "or the feed isn't stuck. Presence is still tracked normally (a frozen "
            "scene never produces false AWAY)."
        )
    if low_fps:
        low_names = ", ".join(
            f"{c} ({cam_health[c].get('fps', 0.0):.1f} fps)" for c in sorted(low_fps)
        )
        st.info(f"Camera {low_names} delivering frames but below the healthy rate.")

    # -- Unknown person row (tracked separately, never an employee) ---------
    unknown = 1 if ("Unknown" in states or "UNKNOWN" in states) else 0
    unknown_last_seen = last_seen_top.get("Unknown") or last_seen_top.get("UNKNOWN")

    rows = []
    for emp_id in sorted(employees.keys()):
        emp = employees[emp_id]
        name = emp.get("name", "") or emp_id
        display = f"{name} ({emp_id})" if name and name != emp_id else emp_id
        last_seen_ts = emp.get("last_seen")
        if last_seen_ts:
            from datetime import datetime as _dt
            last_seen_str = _dt.fromtimestamp(last_seen_ts).strftime("%H:%M:%S")
        else:
            last_seen_str = "—"
        sess_sec = emp.get("session_sec", 0)
        if last_seen_ts:
            session_str = (f"{int(sess_sec // 3600)}:"
                           f"{int(sess_sec % 3600) // 60:02d}")
        else:
            session_str = "—"
        rows.append({
            "Employee": display,
            "Status": _status_badge(emp.get("state", "")),
            "Session (h:mm)": session_str,
            "Source": emp.get("source", "—") or "—",
            "Active (h)": round(emp.get("active_sec", 0) / 3600, 2),
            "Phone (min)": round(emp.get("phone_sec", 0) / 60, 1),
            "Away (min)": round(emp.get("away_sec", 0) / 60, 1),
            "Last Seen": last_seen_str,
        })

    if unknown:
        from datetime import datetime as _dt
        unk_last_str = (_dt.fromtimestamp(unknown_last_seen).strftime("%H:%M:%S")
                        if unknown_last_seen else "—")
        unk_sess_sec = durations.get("Unknown", 0.0) or \
            durations.get("UNKNOWN", 0.0) or 0.0
        unk_session_str = (f"{int(unk_sess_sec // 3600)}:"
                           f"{int(unk_sess_sec % 3600) // 60:02d}"
                           if unknown_last_seen else "—")
        u_telemetry = live.get("telemetry", {}).get("Unknown", {}) or \
            live.get("telemetry", {}).get("UNKNOWN", {})
        rows.append({
            "Employee": "Unknown person",
            "Status": _status_badge(states.get("Unknown", "Unknown") or
                                    states.get("UNKNOWN", "")),
            "Session (h:mm)": unk_session_str,
            "Source": sources.get("Unknown", "—") or sources.get("UNKNOWN", "—"),
            "Active (h)": round(u_telemetry.get("ACTIVE", 0) / 3600, 2),
            "Phone (min)": round(u_telemetry.get("ON_PHONE", 0) / 60, 1),
            "Away (min)": round(u_telemetry.get("AWAY", 0) / 60, 1),
            "Last Seen": unk_last_str,
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch",
                     column_config={"Status": st.column_config.TextColumn(width="medium")})
    else:
        st.info("No tracked employee currently in view.")

    if cam_health:
        with st.expander("Camera health details"):
            cam_rows = []
            for cid in sorted(cam_health):
                h = cam_health[cid]
                cam_rows.append({
                    "Camera": cid,
                    "Source": _safe_cam_source(h.get("source", "—")),
                    "Health": h.get("health", "—"),
                    "Usable": "yes" if h.get("usable") else
                              ("n/a" if h.get("usable") is None else "no"),
                    "Reason": h.get("unusable_reason", "—"),
                    "Frames": h.get("frames_read", 0),
                    "FPS": h.get("fps", 0.0),
                    "Frame mean": h.get("frame_mean", "—"),
                    "Reconnects": h.get("reconnects", 0),
                })
            st.dataframe(pd.DataFrame(cam_rows), width="stretch")

    if unknown:
        from datetime import datetime as _dt
        unk_labels = sorted({
            (tr.get("identity_label") or tr.get("identity") or "")
            for tr in ((live.get("ai", {}) or {}).get("spatial_tracks", {}) or {}).get("tracks", [])
            if ((tr.get("identity_label") or tr.get("identity") or "") != "Unknown"
                and "Unknown" in (tr.get("identity_label") or tr.get("identity") or ""))
        })
        st.caption(f":gray[{', '.join(unk_labels) if unk_labels else f'+ {unknown} unknown person(s)'} "
                   f"currently seen (not attributed to any employee). "
                   "Unknown faces never affect productivity.]")
    if not rows and not unknown:
        st.caption("Waiting for the first detection...")

    online = sum(1 for h in cam_health.values() if h.get("health") == "ONLINE")
    last_det = live.get("last_detection_sec")
    det_note = "No detection yet"
    if last_det is not None:
        det_note = f"Last detection {int(last_det)}s ago" if last_det < 3600 \
            else f"Last detection {last_det / 3600:.1f}h ago"
    st.caption(f"Snapshot {live.get('iso', '')}  |  Daemon FPS: {fps:.1f}  |  "
               f"Cameras online: {online}/{len(cam_health)}  |  {det_note}")


def tab_historical():
    st.header("Historical Analytics")

    conn = get_readonly_connection()
    min_d = max_d = date.today()
    if conn is not None:
        try:
            row = conn.execute(
                "SELECT MIN(date) AS mn, MAX(date) AS mx FROM activity_logs"
            ).fetchone()
            if row and row["mn"]:
                min_d = date.fromisoformat(row["mn"][:10])
            if row and row["mx"]:
                max_d = date.fromisoformat(row["mx"][:10])
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    preset = st.radio("Range", ["Today", "Yesterday", "7 days", "30 days", "Custom"],
                      horizontal=True)
    if preset == "Today":
        start = end = date.today()
    elif preset == "Yesterday":
        start = end = date.today() - timedelta(days=1)
    elif preset == "7 days":
        end = date.today()
        start = end - timedelta(days=6)
    elif preset == "30 days":
        end = date.today()
        start = end - timedelta(days=29)
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            start = st.date_input("From", min_d, min_value=min_d, max_value=max_d)
        with col_b:
            end = st.date_input("To", max_d, min_value=min_d, max_value=max_d)
        if start > end:
            st.error("From must be ≤ To.")
            return

    df = cached_range_metrics(start.isoformat(), end.isoformat())
    if df.empty:
        st.info("No data in the selected range.")
        return

    st.subheader("Productivity Trend")
    by_day = df.groupby("date")["productive_pct"].mean().reset_index()
    fig = px.line(by_day, x="date", y="productive_pct", markers=True,
                  labels={"date": "Date", "productive_pct": "Productivity (%)"})
    fig.add_hrect(y0=80, y1=110, fillcolor="green", opacity=0.12, line_width=0)
    fig.add_hrect(y0=50, y1=80, fillcolor="yellow", opacity=0.12, line_width=0)
    fig.update_layout(height=360, yaxis_range=[0, 110])
    st.plotly_chart(fig, width="stretch")

    st.subheader("Active Hours per Employee")
    agg = df.groupby("employee_id").agg(
        active_hours=("active_hours", "sum"),
        phone_mins=("phone_mins", "sum"),
        away_mins=("away_mins", "sum"),
        avg_pct=("productive_pct", "mean"),
    ).reset_index().sort_values("active_hours", ascending=False)
    fig2 = px.bar(agg, x="employee_id", y="active_hours", color="avg_pct",
                  color_continuous_scale="Greens",
                  labels={"active_hours": "Active Hours", "employee_id": "Employee"})
    st.plotly_chart(fig2, width="stretch")

    with st.expander("Daily breakdown table"):
        st.dataframe(df.sort_values(["date", "employee_id"]), width="stretch")


def tab_camera_health():
    st.header("Camera Health")
    live = load_live_state()
    if not live:
        st.warning("No live daemon snapshot found. Camera health is only visible "
                   "while `main.py` is running.")
        return

    health = live.get("camera_health", {})
    if not health:
        st.info("No cameras configured.")
        return

    # Configured camera IDs (from config.CAMERAS + local sources).  We only
    # report "Configured" vs "Not configured" - never the RTSP URL/password.
    try:
        import config as _cfg
        _configured = set(_cfg.CAMERAS.keys()) | {c.get("source") or c.get("id")
                                                  for c in health.values()}
    except Exception:  # pragma: no cover - config may be unavailable
        _configured = {c.get("id") for c in health.values()}

    state_color = {"ONLINE": "#2ecc71", "NO_FRAME": "#e67e22",
                   "RECONNECTING": "#f39c12", "OFFLINE": "#e74c3c",
                   "FROZEN_FRAME": "#d35400", "LOW_FPS": "#d35400"}
    note = {
        "ONLINE": "—",
        "NO_FRAME": "No fresh frame in the stale window (camera may be frozen).",
        "RECONNECTING": "Stream dropped; automatically reconnecting.",
        "OFFLINE": "Stream is not open.",
        "FROZEN_FRAME": "Stream flows but the scene has not changed for a while "
                        "(A12 degraded state).",
        "LOW_FPS": "Delivering frames below the health FPS threshold (A12).",
    }
    rows = []
    for cid, h in sorted(health.items()):
        hs = h.get("health", "OFFLINE")
        last = h.get("last_frame", 0)
        last_s = f"{time.strftime('%H:%M:%S', time.localtime(last))}" if last else "-"
        frame_age_s = "-"
        if last:
            frame_age_s = f"{max(0.0, time.time() - last):.1f}s"
        src = h.get("source") or h.get("id") or cid
        configured = "Configured" if (src in _configured or cid in _configured) \
            else "Not configured"
        usable = h.get("usable")
        usable_s = "—" if usable is None else ("Yes" if usable else "No")
        reason = h.get("unusable_reason", "—") or "—"
        rows.append({
            "Camera": cid,
            "Status": f":{state_color.get(hs, '#888')}[●] {hs}",
            "Configured": configured,
            "Last Frame": last_s,
            "Frame Age": frame_age_s,
            "FPS": (round(float(h.get("fps", 0) or 0), 2)
                    if hs != "OFFLINE" else "—"),
            "Usable": usable_s,
            "Reason": reason,
            "Reconnects": h.get("reconnects", 0),
            "Frames": h.get("frames_read", 0),
            "Note": note.get(hs, hs),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    online = sum(1 for h in health.values() if h.get("health") == "ONLINE")
    st.caption(f"{online}/{len(health)} cameras online. "
               "A camera marked OFFLINE / NO_FRAME / RECONNECTING, or a blank/black "
               "feed (Usable = No), is *not* counted as employee AWAY time -- "
               "employee states freeze until a usable frame arrives. A FROZEN_FRAME "
               "feed still flows and presence is still monitored. RTSP credentials "
               "are never displayed - only Configured / Not configured.")


def _can_export(role: str) -> bool:
    from src.domain import PERM_EXPORT, role_has_permission
    return role_has_permission(role, PERM_EXPORT)


def _audit_report_export(report_name: str, role: str) -> None:
    try:
        from src import database as db
        wconn = _open_write_conn()
        try:
            db.audit(wconn, "report.export", actor=role,
                     resource=report_name, detail="downloaded")
        finally:
            wconn.close()
    except Exception as exc:  # noqa: BLE001 - audit failure must not break download
        logging.getLogger("cctv.dashboard").warning(
            "report export audit failed for %s: %s", report_name, exc)

def tab_ai_capabilities():
    """Phase A/B dashboard surface: spatial tracks, motion, camera selection,
    and the honest detector-capability panel (B15 vocabulary)."""
    st.header("AI Capabilities (Phase A/B)")
    live = load_live_state()
    if not live:
        st.warning("No live daemon snapshot found. Capability status is only "
                   "available while `main.py` is running.")
        return
    ai = live.get("ai", {})

    sel = ai.get("camera_selection", {})
    if sel:
        st.subheader("Camera Selection")
        mode = sel.get("mode", "AUTO")
        st.caption(f"Mode: **{mode}** | Cameras: "
                   f"{', '.join(str(c) for c in sel.get('camera_ids', []))}"
                   + (f" | device index: {sel.get('device_index')}"
                      if sel.get("device_index") is not None else ""))

    caps = (ai.get("capabilities", {}) or {}).get("detectors", [])
    if caps:
        st.subheader("Detector Capability Panel")
        status_color = {"AVAILABLE": "#2ecc71", "DISABLED": "#95a5a6",
                        "DEGRADED": "#e67e22", "NOT_CONFIGURED": "#e67e22",
                        "FUTURE_MODEL_REQUIRED": "#d35400",
                        "UNAVAILABLE": "#95a5a6"}
        rows = [{
            "Detector": (d.get("name") or d.get("detector_id")),
            "Status": f":{status_color.get(d.get('status'), '#888')}[●] "
                      f"{d.get('status', 'UNAVAILABLE')}",
            "Detail": d.get("status_detail", "") or "—",
            "Events": (d.get("event_types") or "—"),
        } for d in caps]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Statuses are the registry's B15 vocabulary and are never "
                   "invented: AVAILABLE / DISABLED / DEGRADED / "
                   "NOT_CONFIGURED / FUTURE_MODEL_REQUIRED.")

    ppass = (ai.get("phone_pass", {}) or {})
    if ppass:
        st.subheader("Phone Detection Pass (Phase 44/63)")
        pp_rows = [{
            "Source": ppass.get("source"),
            "Class": ppass.get("phone_class"),
            "Imgsz": ppass.get("imgsz"),
            "Cadence": ppass.get("cadence"),
            "Calls": ppass.get("calls"),
            "Phones seen": ppass.get("boxes_seen"),
            "Conf mean": ppass.get("conf_mean"),
            "Conf min": ppass.get("conf_min"),
        }]
        st.dataframe(pd.DataFrame(pp_rows), width="stretch", hide_index=True)
        st.caption("Live metrics of the dedicated phone pass. Model/source is "
                   "the admin or env selection; cadence is the per-camera "
                   "cycle gating. Empty = pass never ran yet.")

    phdiag = (ai.get("phone_diag", {}) or {})
    if phdiag.get("enabled"):
        st.subheader("Phone/Person Association (CCTV_PHONE_DIAG=1)")
        cam_rows = [{
            "Camera": c,
            "Cycles": d.get("cycles"),
            "Persons": d.get("person_dets"),
            "Phones": d.get("phone_dets"),
            "Conf mean": d.get("phone_conf_mean"),
            "Matched": d.get("matched_to_person"),
            "Unmatched": d.get("unmatched_to_person"),
        } for c, d in (phdiag.get("cameras") or {}).items()]
        if cam_rows:
            st.dataframe(pd.DataFrame(cam_rows), width="stretch",
                         hide_index=True)
            st.caption("Diagnostic measurement only — never changes detection "
                       "behaviour.")
        else:
            st.caption("Diagnostics enabled; no camera samples yet.")

    tracks = (ai.get("spatial_tracks", {}) or {})
    tr_rows = []
    for t in tracks.get("tracks", []):
        tr_rows.append({
            "Track": t.get("track_id"),
            "Cam": t.get("cam"),
            "Identity": t.get("identity_label") or t.get("identity"),
            "Score": t.get("identity_score"),
            "Phone": ("yes" if t.get("phone") else "no"),
            "Phone Sec": t.get("phone_sec"),
            "Box": str(t.get("bbox")),
        })
    if tr_rows:
        st.subheader(f"Spatial Person Tracks ({tracks.get('count', 0)})")
        st.dataframe(pd.DataFrame(tr_rows), width="stretch", hide_index=True)
    elif caps:
        st.subheader("Spatial Person Tracks")
        st.info("No live spatial tracks right now (A5/A6 tracker is idle).")

    seat_zone = (ai.get("seat_zones", {}) or {})
    if seat_zone:
        st.subheader(f"Desk / Seat Zones ({seat_zone.get('zone_count', 0)})")
        st.caption(
            "Which desk a person currently occupies (Phase 59). Seat is "
            "**context only** — it never overrides face identity: an unknown "
            "face stays Unknown, and an EMP002 sitting in EMP001's seat is "
            "reported as EMP002 with a neutral mismatch observation.")
        sz_rows = [{
            "Zone": z.get("name") or z.get("zone_id"),
            "Camera": z.get("camera_id"),
            "Status": z.get("status"),
            "Assigned": z.get("assigned_employee_id") or "—",
            "Current": z.get("current_identity") or "—",
            "Identity": z.get("identity_state") or "—",
            "Persons": z.get("persons"),
        } for z in seat_zone.get("zones", [])]
        if sz_rows:
            st.dataframe(pd.DataFrame(sz_rows), width="stretch", hide_index=True)
        else:
            st.info("No desk/seat zones configured yet (data/seat_zones.json "
                    "or the admin editor below).")

    mstate = (ai.get("motion", {}) or {})
    if mstate:
        st.subheader("Motion Status (A8)")
        m_rows = [{
            "Camera": c,
            "Motion": ("yes" if s.get("motion") else "no"),
            "Score": s.get("score"),
            "Active Since (s ago)": s.get("active_since"),
            "Last Event (s ago)": s.get("last_fire_sec"),
        } for c, s in sorted(mstate.items())]
        st.dataframe(pd.DataFrame(m_rows), width="stretch", hide_index=True)
        st.caption("Motion is advisory only: it records into the motion_events "
                   "table and is never treated as a security alarm on its own.")

    role = st.session_state.get("cctv_role", "viewer")
    if role == "admin":
        st.subheader("Desk / Seat Zone Editor (Phase 59)")
        st.caption("Add or update a desk/seat zone. The polygon uses normalised "
                   "coordinates 0..1 for the referenced camera's frame. "
                   "Assigning an employee is a *seat-map hint* — it never "
                   "overrides face identity.")
        with st.form("seatzone_form"):
            sz_id = st.text_input("Zone ID (e.g. A01)")
            sz_cam = st.text_input("Camera")
            sz_name = st.text_input("Name (optional)")
            sz_poly = st.text_input(
                "Polygon JSON (e.g. [[0.2,0.1],[0.8,0.1],[0.8,0.9],[0.2,0.9]])")
            sz_assigned = st.text_input("Assigned employee (optional, e.g. EMP001)")
            sz_enabled = st.checkbox("Enabled", value=True)
            sz_submit = st.form_submit_button("Save Zone")
        if sz_submit:
            if not (sz_id.strip() and sz_cam.strip() and sz_poly.strip()):
                st.error("Zone ID, camera, and polygon are required.")
            else:
                try:
                    conn = _open_write_conn()
                    try:
                        store = SeatZoneStore(conn)
                        store.add(
                            zone_id=sz_id.strip(),
                            camera_id=sz_cam.strip(),
                            name=sz_name.strip(),
                            polygon=sz_poly.strip(),
                            enabled=sz_enabled,
                            assigned_employee_id=sz_assigned.strip() or None,
                        )
                    finally:
                        conn.close()
                    st.success(f"Desk zone {sz_id} saved. Restart the daemon "
                               "(or wait for its next config refresh) to apply.")
                except ValueError as _e:
                    st.error(str(_e))


def tab_reports(role: str = "viewer"):
    st.header("Report Center")
    st.caption("Download any daily Excel report. Reports are generated "
               "automatically at the configured end-of-day hour, on graceful "
               "daemon shutdown, or on demand with the **r** hotkey in the live "
               "HUD. Values match the dashboard and database exactly.")

    reports_dir = Path(REPORT_OUTPUT_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)
    xlsx_files = sorted(reports_dir.glob("*.xlsx"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if xlsx_files:
        st.subheader(f"{len(xlsx_files)} report(s)")
        can_export = _can_export(role)
        for path in xlsx_files:
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([4, 2, 1, 1])
                c1.write(f"**{path.name}**")
                c1.caption(f"{path.stat().st_size / 1024:.1f} KB")
                c2.caption(f"Generated {date.fromtimestamp(path.stat().st_mtime)}")
                c3.caption("Ready")
                with open(path, "rb") as f:
                    data = f.read()
                if c4.download_button(
                        "Download", data=data, file_name=path.name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_{path.name}", disabled=not can_export):
                    _audit_report_export(path.name, role)
        if not can_export:
            st.caption("Your role cannot export reports.")
    else:
        st.info("No reports generated yet. Start the daemon and let a work day "
                "complete (or press **r** in the live HUD) and the first report "
                "will appear here.")

    email_status = last_email_status()
    st.subheader("Email Delivery")
    ok = email_status.get("ok")
    if ok is True:
        st.success(f"Last report email sent ({email_status.get('at')}).")
    elif ok is False:
        st.error(f"Last report email FAILED ({email_status.get('attempts')} "
                 f"attempts): {email_status.get('message')}")
    else:
        st.info("No email attempts recorded yet.")


def tab_employees(role: str):
    st.html(
        _COMPONENT_CSS +
        "<div class='p42 p42-page-head'>"
        "<div class='p42-page-title'>Employee Management</div>"
        "<div class='p42-page-sub'>Roster rows the daemon recognises against. "
        "Employees need a row here before a face photo can be attached for "
        "identification.</div></div>",
    )
    conn = get_readonly_connection()
    if conn is None:
        detail = f" ({_LAST_DB_ERROR})" if _LAST_DB_ERROR else ""
        st.error(f"Database unavailable.{detail}")
        return
    try:
        store = EmployeeStore(conn, WORK_SCHEDULE)
        emps = store.list()
    finally:
        conn.close()

    if not emps:
        st.html(_COMPONENT_CSS + _panel_html(
            "off", "group",
            "NO EMPLOYEES CONFIGURED",
            "No roster rows exist yet. Create the first employee below — until a "
            "person is registered the daemon cannot match anyone to the roster."
        ))
    else:
        faces = {p.stem for p in _faces_dir().glob("*")
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}
        df = pd.DataFrame([{
            "Employee ID": e.employee_id,
            "Name": e.name or "—",
            "Department": e.department or "—",
            "Designation": e.designation or "—",
            "Active": "✓" if e.active else "—",
            "Photo on disk": "✓" if e.employee_id in faces else "—",
            "Enrolled in model": "✓" if e.enrolled else "—",
        } for e in emps])
        st.dataframe(df, width="stretch")
        st.caption("**Photo on disk** = a face image is waiting in `data/faces/`. "
                   "**Enrolled in model** = the face registry has validated it "
                   "(validated at daemon startup or via the Validate button below).")

    if role != "admin":
        st.caption("Viewer role: read-only. Ask an admin to modify employees.")
        return

    st.subheader("Add / Edit Employee")
    st.caption("Start here: an employee row is needed before a face photo can "
               "be attached.")
    with st.form("emp_form"):
        eid = st.text_input("Employee ID (e.g. EMP009)")
        name = st.text_input("Name")
        dept = st.text_input("Department")
        desig = st.text_input("Designation")
        active = st.checkbox("Active", value=True)
        submit = st.form_submit_button("Save")
    if submit:
        safe_eid = _safe_employee_id(eid)
        if not eid.strip():
            st.error("Employee ID is required.")
        elif not safe_eid:
            st.error("Invalid Employee ID (letters, digits, _ and - only).")
        elif safe_eid.lower() in ("unknown", "__person__"):
            st.error("Reserved ID; choose a different one.")
        else:
            inserted = False
            conn = _open_write_conn()
            try:
                from src import database as db
                from src.rbac import AccessGuard, AccessDenied
                guard = AccessGuard(conn, actor=role, role=role)
                # Backend-authoritative: a viewer reaching this path is denied
                # and the denial is audited, even if the UI hid the form.
                guard.require_manage_users()
                inserted = EmployeeStore(conn, WORK_SCHEDULE).upsert(
                    safe_eid, name=name, department=dept,
                    designation=desig, active=active)
                db.audit(conn, "employee.upsert", actor=role,
                         resource=safe_eid,
                         detail="insert" if inserted else "update")
                conn.commit()
            except AccessDenied:
                st.error("Your role cannot modify employees.")
            finally:
                conn.close()
            if inserted:
                st.cache_data.clear()
                st.success(f"{'Inserted' if inserted else 'Updated'} {safe_eid}.")
                st.rerun()

    st.subheader("Face Enrollment")
    st.caption("Three steps:  **1)** add the employee above, **2)** upload one "
               "clear front-facing photo (good light, close up), **3)** validate. "
               "Validation loads the face model once -- the first run on CPU can "
               "take 30-60 s; later runs are faster.")
    enroll_col, status_col = st.columns([1, 1])
    with enroll_col:
        upload = st.file_uploader("Enrollment photo (JPG / PNG / BMP)",
                                  type=["jpg", "jpeg", "png", "bmp"],
                                  key="enroll_upload")
        sel_emp = st.selectbox(
            "Assign photo to employee",
            [e.employee_id for e in emps] if emps else ["(create an employee first)"],
            key="enroll_emp",
        )
        if st.button("Save photo to `data/faces/`", key="save_photo"):
            if not emps:
                st.warning("Create the employee first (form above).")
            elif upload is None:
                st.warning("Choose a photo to upload first.")
            else:
                conn = _open_write_conn()
                outcome = None  # "denied" | True | False
                msg = ""
                try:
                    from src import database as db
                    from src.rbac import AccessGuard, AccessDenied
                    AccessGuard(conn, actor=role, role=role).require_manage_users()
                    outcome, msg = save_enrollment_image(
                        sel_emp, upload.name, upload.getvalue())
                    if outcome:
                        db.audit(conn, "employee.enroll_photo", actor=role,
                                 resource=sel_emp, detail=f"photo {upload.name}")
                        conn.commit()
                except AccessDenied:
                    outcome = "denied"
                    msg = "Your role cannot modify employees."
                finally:
                    conn.close()
                if outcome is True:
                    st.success(msg)
                    st.caption("Now click **Validate enrollment** to confirm the face.")
                else:
                    st.error(msg)
        if st.button("Validate enrollment", key="validate_enroll"):
            st.info("Loading the face model and scanning `data/faces/` -- one moment...")
            try:
                from src.face_registry import FaceRegistry  # heavy import; on demand
                registry = FaceRegistry()
                registry.rebuild()
                status = registry.employee_status()
            except Exception as exc:  # noqa: BLE001 - user-friendly surface
                st.error(f"Face validation failed: {exc}")
                status = None
            if status:
                st.success(f"Validation finished -- {registry.num_enrolled_employees} "
                           f"employee(s) enrolled, {registry.num_registered} embedding(s). "
                           "Restart the daemon so it picks up the new faces.")
                st.dataframe(enrollment_status_frame(status), width="stretch")
            st.cache_data.clear()
    with status_col:
        st.caption("Current status without a full re-scan (reads the cached registry):")
        if st.button("Show enrollment status (fast)", key="status_fast"):
            status_map = cached_enrollment_status()
            if not status_map:
                st.info("No cached status yet. Click **Validate enrollment** "
                        "once to build it.")
            else:
                st.dataframe(enrollment_status_frame(status_map), width="stretch")
        st.caption("Status commands:")
        st.caption("- `ENROLLED` -- face ready for recognition")
        st.caption("- `NO_FACE` -- no face found in the photo")
        st.caption("- `MULTIPLE_FACES` -- more than one face; largest is used")
        st.caption("- `LOW_QUALITY` -- face too small/unusable")
        st.caption("- `INVALID_IMAGE` -- could not be read")


def tab_security(role):
    """Security / Incidents tab (Phase 31)."""
    import config
    from src.rbac import AccessGuard, AccessDenied
    conn = get_readonly_connection()
    if conn is None:
        st.info("No database found. Start the daemon to create it.")
        return

    st.header(":material/shield: Security & Incidents")

    live = load_live_state()
    security = live.get("security", {})

    # --- Overview cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cameras Offline", security.get("cameras_offline", "n/a"))
    c2.metric("Open Incidents", security.get("open_incidents", "n/a"))
    c3.metric("High-Severity Open", security.get("high_severity_open", "n/a"))
    c4.metric("Evidence Mode", security.get("evidence_mode", config.EVIDENCE_MODE))

    st.caption(f"Evidence storage: {security.get('evidence_storage_mb', 0)} MB "
               f"(recorded evidence, not continuous video)")

    # --- Privacy indicator ---
    st.caption("📌 **Privacy indicator:** LIVE MONITORING + ANALYTICS "
               "(event-driven evidence)" 
               if config.EVIDENCE_MODE != "OFF" else
               "📌 **Privacy indicator:** LIVE MONITORING + ANALYTICS only — "
               "no evidence recording (mode=OFF).")

    st.divider()

    # --- Incident timeline ---
    st.subheader("Incident Timeline")
    try:
        inc = IncidentEngine(conn)
        incidents = inc.list(limit=100)
        if incidents:
            import pandas as pd
            df = pd.DataFrame(incidents)
            show = df[["incident_id", "first_seen", "event_type", "severity",
                       "camera", "zone", "employee_id", "status"]]
            st.dataframe(show, width="stretch")
        else:
            st.info("No incidents recorded yet.")
    except Exception as exc:
        st.warning(f"Could not load incidents: {exc}")

    st.divider()

    # --- Filters ---
    st.subheader("Incident Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        day_filter = st.text_input("Date (YYYY-MM-DD)", "", key="sec_day")
    with f2:
        type_filter = st.text_input("Event type", "", key="sec_type")
    with f3:
        sev_filter = st.text_input("Severity", "", key="sec_sev")
    with f4:
        status_filter = st.text_input("Status", "", key="sec_status")

    try:
        events = EventStore(conn)
        ev_list = events.events(day=day_filter or None,
                                event_type=type_filter or None,
                                severity=sev_filter or None,
                                status=status_filter or None,
                                limit=200)
        if ev_list:
            import pandas as pd
            df = pd.DataFrame(ev_list)
            show_cols = [c for c in ["timestamp", "event_type", "severity",
                                     "camera", "zone", "employee_id",
                                     "confidence", "incident_id", "status"]
                         if c in df.columns]
            st.dataframe(df[show_cols], width="stretch")
        else:
            st.info("No security events match the filters.")
    except Exception as exc:
        st.warning(f"Could not load events: {exc}")

    st.divider()

    # --- Security analytics ---
    st.subheader("Security Analytics (today)")
    try:
        evs = EventStore(conn).events(day=date.today().isoformat(), limit=500)
        if evs:
            import pandas as pd
            df = pd.DataFrame(evs)
            by_type = df["event_type"].value_counts()
            st.write("**Events by type (today):**")
            st.dataframe(by_type.rename("count"), width="stretch")
        else:
            st.info("No security events today.")
    except Exception as exc:
        st.warning(f"Could not load analytics: {exc}")

    # --- Investigation workbench (Phase 32) ---
    st.divider()
    st.subheader(":material/search: Investigation Workbench")
    try:
        from src import database as db
        from src.search import SecuritySearch
        # system health + camera intelligence (read-only)
        from src.camera_intelligence import CameraIntelligence, SystemHealth
        ci = CameraIntelligence(conn)

        inc_ids = [i["incident_id"] for i in IncidentEngine(conn).list(limit=200)]
        selected = st.selectbox("Incident to investigate",
                                [""] + inc_ids, key="inv_incident")
        if selected:
            search = SecuritySearch(conn)
            tl = search.incident_timeline(selected)
            inc = tl.get("incident") or {}
            st.markdown(
                f"**{inc.get('event_type')}** — {inc.get('severity')} — "
                f"status={inc.get('status')} — review={inc.get('review_state', 'DETECTED')} — "
                f"count={inc.get('count')}")
            if inc.get("correlation_reason"):
                st.info(f"Correlation: {inc.get('correlation_reason')}")
            if tl.get("notes"):
                st.write("**Investigation notes:**")
                for n in tl["notes"]:
                    st.caption(f"[{n.get('created_at')}] {n.get('author')}: {n.get('note')}")

            t1, t2, t3 = st.tabs(["Related Events", "Alerts", "Evidence"])
            with t1:
                evs = tl.get("related_events") or []
                if evs:
                    import pandas as pd
                    df = pd.DataFrame(evs)
                    cols = [c for c in ["timestamp", "event_type", "severity",
                                        "camera", "zone", "incident_id"]
                            if c in df.columns]
                    st.dataframe(df[cols], width="stretch")
                else:
                    st.caption("No events.")
            with t2:
                al = tl.get("alerts") or []
                if al:
                    import pandas as pd
                    df = pd.DataFrame(al)
                    cols = [c for c in ["created_at", "event_type", "severity",
                                        "channel", "status", "attempts"]
                            if c in df.columns]
                    st.dataframe(df[cols], width="stretch")
                else:
                    st.caption("No alerts.")
            with t3:
                ev = tl.get("evidence") or []
                if ev:
                    import pandas as pd
                    df = pd.DataFrame(ev)
                    cols = [c for c in ["kind", "camera", "capture_type",
                                        "size_bytes", "sha256", "available",
                                        "retention_deadline"] if c in df.columns]
                    st.dataframe(df[cols], width="stretch")
                else:
                    st.caption("No evidence recorded (evidence is event-driven).")

            # Authorized write actions (RBAC)
            from src.rbac import AccessGuard, AccessDenied
            wconn = _open_write_conn()
            guard = AccessGuard(wconn, actor=role, role=role)
            try:
                r1, r2, r3 = st.columns(3)
                with r1:
                    note = st.text_input("Add investigation note", key="inv_note")
                    if st.button("Add note", key="btn_note") and note:
                        try:
                            guard.require("investigate")
                            db.add_incident_note(wconn, selected, role, note)
                            st.success("Note added.")
                            st.rerun()
                        except AccessDenied:
                            st.error("Your role cannot investigate incidents.")
                with r2:
                    state = st.selectbox(
                        "Review state",
                        ["DETECTED", "SUSPECTED", "REQUIRES_REVIEW",
                         "CONFIRMED_BY_OPERATOR", "DISMISSED"],
                        key="inv_review")
                    if st.button("Set review state", key="btn_review"):
                        try:
                            guard.set_review_state(selected, state)
                            st.success(f"Review state → {state}")
                            st.rerun()
                        except AccessDenied:
                            st.error("Your role cannot set review state.")
                with r3:
                    if st.button("Acknowledge", key="btn_ack"):
                        try:
                            guard.acknowledge_incident(selected)
                            st.success("Acknowledged.")
                            st.rerun()
                        except AccessDenied:
                            st.error("Your role cannot acknowledge incidents.")
            finally:
                wconn.close()
    except Exception as exc:
        st.warning(f"Investigation workbench unavailable: {exc}")

    # --- Admin actions (security operator / admin) ---
    # Authorization is enforced through AccessGuard (backend-authoritative,
    # audited denials), not just the UI role check.
    if role in ("admin", "security_operator"):
        st.divider()
        st.subheader("Incident Actions")
        a1, a2, a3 = st.columns(3)
        with a1:
            res_id = st.text_input("Incident ID to resolve", key="res_id")
            if st.button("Resolve", key="btn_resolve"):
                try:
                    wconn = _open_write_conn()
                    guard = AccessGuard(wconn, actor=role, role=role)
                    if not res_id:
                        st.error("Enter an incident ID.")
                    else:
                        guard.require("resolve", resource=res_id)
                        if IncidentEngine(wconn).resolve(res_id, actor=role):
                            st.success(f"Resolved {res_id}")
                        else:
                            st.error("Incident not found or already closed.")
                    wconn.close()
                    st.rerun()
                except AccessDenied:
                    st.error("Your role cannot resolve incidents.")
                except Exception as exc:
                    st.error(f"Resolve failed: {exc}")
        with a2:
            dis_id = st.text_input("Incident ID to dismiss", key="dis_id")
            if st.button("Dismiss", key="btn_dismiss"):
                try:
                    wconn = _open_write_conn()
                    guard = AccessGuard(wconn, actor=role, role=role)
                    if not dis_id:
                        st.error("Enter an incident ID.")
                    else:
                        guard.require("dismiss", resource=dis_id)
                        if IncidentEngine(wconn).dismiss(dis_id, actor=role):
                            st.success(f"Dismissed {dis_id}")
                        else:
                            st.error("Incident not found or already closed.")
                    wconn.close()
                    st.rerun()
                except AccessDenied:
                    st.error("Your role cannot dismiss incidents.")
                except Exception as exc:
                    st.error(f"Dismiss failed: {exc}")
    else:
        st.caption("Admin / Security Operator actions (resolve/dismiss) are "
                   "restricted to authorized roles.")

    # --- Phase 33: advisory intelligence (read-only, evidence-based) ---
    st.divider()
    with st.expander(":material/psychology: Phase 33 — Advisory Intelligence (read-only)", expanded=False):
        try:
            from src import database as db
            from src.camera_intelligence import SystemHealth
            from src.detector_registry import DetectorRegistry
            from src.risk_scoring import IncidentRiskScorer
            from src.evidence_access import EvidenceAccess

            sh = SystemHealth(conn).summary_v2()
            st.markdown(
                f"**System Health 2.0:** {sh.get('status')} — "
                f"db={sh.get('db_size_mb', 0)} MB — "
                f"alert delivery {sh.get('alert_delivery_pct', 0)}%")

            cov = CameraIntelligence(conn).coverage_analysis()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Covered zones", cov.get("covered_zones"))
            c2.metric("Uncovered zones", cov.get("uncovered_zones"))
            c3.metric("Total cameras", len(cov.get("cameras", [])))
            c4.metric("Sample window (h)", cov.get("sample_window_days"))

            st.write("**Detector registry health**")
            reg = DetectorRegistry(conn)
            rows = [r for r in reg.health_summary() or []]
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), width="stretch")
            else:
                st.caption("No detectors registered yet.")

            st.write("**Risk-scored open incidents (advisory — affects "
                     "incidents, never people)**")
            risks = IncidentRiskScorer(conn).list(limit=200)
            open_risks = [r for r in risks if r.get("status") != "RESOLVED"
                          and r.get("review_state") != "DISMISSED"]
            if open_risks:
                import pandas as pd
                df = pd.DataFrame(open_risks)
                cols = [c for c in ["incident_id", "risk_score", "severity",
                                    "factors", "recommended"] if c in df.columns]
                st.dataframe(df[cols], width="stretch")
            else:
                st.caption("No risk-scored open incidents.")

            st.caption("These panels are advisory and read-only. Risk scores "
                       "apply to incidents only — never to employees. All "
                       "detected patterns require human review.")
        except Exception as exc:
            st.warning(f"Phase 33 intelligence unavailable: {exc}")

    # --- Phase 34: SOC operator view (attention queue, reconstruction, continuity) ---
    st.divider()
    with st.expander(":material/track_changes: Phase 34 — SOC Operator View (read-only)", expanded=False):
        try:
            from src import database as db
            from src.investigation import (IncidentReconstruction,
                                           PersonContinuity, EventContextRollup)
            from src.incidents import IncidentEngine
            from src.risk_scoring import IncidentRiskScorer

            inc_eng = IncidentEngine(conn)

            # --- Attention queue: open/unresolved incidents, prioritized ---
            st.subheader("Attention queue (what needs attention now)")
            open_incs = inc_eng.list(status="OPEN")
            open_incs = open_incs + [i for i in inc_eng.list(status="ACKNOWLEDGED")
                                     if i["incident_id"] not in {x["incident_id"]
                                                                 for x in open_incs}]
            # Unresolved = open/acknowledged and not dismissed.
            unresolved = [i for i in open_incs
                          if i.get("review_state") != "DISMISSED"]
            critical_high = [i for i in unresolved
                             if i.get("severity") in ("CRITICAL", "HIGH")]
            st.caption(f"{len(unresolved)} unresolved · "
                       f"{len(critical_high)} critical/high")
            risk_map = {}
            try:
                for r in db.list_incident_risk(conn, limit=500):
                    risk_map[r["incident_id"]] = r.get("risk_score") or 0.0
            except Exception:
                risk_map = {}
            queue = sorted(
                unresolved,
                key=lambda i: (
                    risk_map.get(i["incident_id"], 0.0),
                    i.get("severity", "") in ("CRITICAL", "HIGH"),
                ),
                reverse=True)[:50]
            if queue:
                import pandas as pd
                df = pd.DataFrame(queue)
                show = [c for c in ["incident_id", "event_type", "severity",
                                    "status", "temporal_state", "review_state",
                                    "camera", "zone", "first_seen"]
                        if c in df.columns]
                df["risk_score"] = df["incident_id"].map(risk_map)
                st.dataframe(df[show + (["risk_score"] if "risk_score" not in show else [])],
                             width="stretch")
            else:
                st.success("No unresolved incidents needing attention.")

            # --- Incident reconstruction + event context ---
            st.subheader("Incident reconstruction & event context")
            inc_ids = [i["incident_id"] for i in inc_eng.list(limit=200)]
            sel = st.selectbox("Incident to reconstruct", [""] + inc_ids,
                               key="p34_recon")
            if sel:
                rec = IncidentReconstruction(conn).reconstruct(sel, pre_sec=30,
                                                               post_sec=30)
                ctx = EventContextRollup(conn).rollup(sel)
                st.markdown(f"**{rec.get('event_type')}** — {rec.get('severity')} — "
                            f"status={rec.get('status')} — "
                            f"temporal={rec.get('temporal_state')} — "
                            f"review={rec.get('review_state')}")
                st.markdown(f"- Event context: **{ctx.get('label')}** "
                            f"(observations={ctx.get('observation_count')})")
                st.caption(f"Pre-event evidence: **{rec.get('pre_event_evidence')}**")
                rows = rec.get("window_rows") or []
                if rows:
                    import pandas as pd
                    wdf = pd.DataFrame(rows)
                    cols = [c for c in ["ts", "kind", "label", "detail"]
                            if c in wdf.columns]
                    st.caption(f"Reconstruction window (T-{rec.get('pre_sec')}s .. "
                               f"T+{rec.get('post_sec')}s):")
                    st.dataframe(wdf[cols], width="stretch")
                else:
                    st.caption("No event-driven rows in the reconstruction window.")

            # --- Person continuity ---
            st.subheader("Person continuity (first/last seen, dwell, zones)")
            ident = st.text_input("Identity (employee id or 'Unknown')", "",
                                  key="p34_identity")
            if ident:
                pc = PersonContinuity(conn).resolve(ident)
                st.markdown(f"- identity={ident} **status={pc.get('status')}**")
                st.markdown(f"- first_seen={pc.get('first_seen')} · "
                            f"last_seen={pc.get('last_seen')} · "
                            f"dwell={pc.get('dwell_seconds')}s")
                st.markdown(f"- zone transitions: {pc.get('zone_transitions') or 'none'}")
                st.markdown(f"- camera path: {pc.get('camera_path') or 'none'}")
                if pc.get("incidents"):
                    st.caption("Related incidents: " + ", ".join(pc["incidents"]))

            st.caption("All Phase 34 panels are advisory, evidence-backed, and "
                       "read-only. Unknown identities stay Unknown; pre-event "
                       "evidence is reported as unavailable, never fabricated.")
        except Exception as exc:
            st.warning(f"Phase 34 SOC view unavailable: {exc}")


def tab_admin_cameras(role: str) -> None:
    """Phase 61 -- Admin → Cameras: persistent camera management.

    Add / edit / enable-disable / test-connect / apply the runtime camera
    configuration.  All writes go to ``admin_cameras`` (SQLite); the daemon
    reconciles within seconds, so no code edit or restart is required.  Never
    displays a saved password or a credential-bearing URL.

    Cloud-safe (Phase 65): the OpenCV capture runtime is loaded lazily and
    only for "Test Connection"; configuration and management never need it.
    """
    if role != "admin":
        st.info("Viewer role: read-only. Ask an admin to configure cameras.")
        return

    from src.camera_store import CameraStore, test_camera_connection
    from src import database as db
    from src.domain import CameraSourceKind

    conn = _open_write_conn()
    try:
        db.init_db(conn)  # idempotent; guarantees the admin tables exist
    except Exception:  # pragma: no cover - surface gracefully
        st.error("Could not open the application database for camera settings.")
        try:
            conn.close()
        except Exception:
            pass
        return
    try:
        store = CameraStore(conn)
        records = store.list()
        live = (load_live_state() or {}).get("camera_health", {}) or {}
        _kind_labels = {k.value: k.value for k in CameraSourceKind}
        _kind_choices = list(_kind_labels)

        st.markdown(
            _COMPONENT_CSS +
            "<div class='p42 p42-page-head'>"
            "<div class='p42-page-title'>Camera Management</div>"
            "<div class='p42-page-sub'>Persistent IP CCTV / RTSP / local camera configuration. "
            "Configuration, validation and apply/reconcile need no capture runtime; "
            "only “Test Connection” touches the camera runtime.</div></div>",
            unsafe_allow_html=True,
        )

        enabled_total = sum(1 for r in records if r.enabled)
        online = sum(1 for r in records
                     if r.enabled and live.get(r.camera_id, {}).get("health") == "ONLINE")
        disabled = len(records) - enabled_total
        offline = (enabled_total - online) if live else None
        ov1, ov2, ov3, ov4 = st.columns(4)
        ov1.metric("Total Cameras", f"{len(records)}",
                   help="Admin-configured cameras; ids absent from the store fall "
                        "back to the `.env` configuration.")
        ov2.metric("Online", f"{online}",
                   help="Enabled cameras with health ONLINE in the daemon snapshot.")
        ov3.metric("Offline", "—" if offline is None else f"{offline}",
                   help="Enabled cameras not ONLINE (no live snapshot = unknown).")
        ov4.metric("Disabled", f"{disabled}",
                   help="Configured but disabled cameras (kept, never started).")

        st.subheader(":material/photo_camera: Camera list")
        if not records:
            st.html(_COMPONENT_CSS +
                    "<div class='p42 p42-empty'>No admin-configured cameras yet. "
                    "Use the form below to add one (or keep using `.env` cameras "
                    "unchanged).</div>")
        else:
            rows = []
            for r in records:
                s = r.safe_dict()
                h = live.get(r.camera_id, {})
                if h:
                    status = h.get("health") or "UNKNOWN"
                    res = h.get("resolution") or "-"
                    fps = h.get("fps") if h.get("health") != "OFFLINE" else "-"
                else:
                    status = "N/A (no live daemon)"
                    res = "-"
                    fps = "-"
                rows.append({
                    "Name": r.name,
                    "ID": r.camera_id,
                    "Type": s["kind"],
                    "Location": r.location or "-",
                    "Status": status,
                    "Resolution": res,
                    "FPS": fps,
                    "Reconnect Policy": (f"{r.reconnect_base:g}s → {r.reconnect_max:g}s "
                                         f"(×{r.reconnect_factor:g})"),
                    "Credentials": s["credentials"],
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                         column_config={"ID": st.column_config.TextColumn(width="small")})
            st.caption("Connection status comes from the running daemon's "
                       "health snapshot (OFFLINE / ONLINE / NO_FRAME / ...). "
                       "Passwords and credential-bearing RTSP URLs are never "
                       "displayed — only CONFIGURED / NONE.")

        # -- Enable / disable ------------
        st.markdown("**Enable / Disable**")
        with st.expander("Toggle cameras", expanded=False):
            for r in records:
                c1, c2 = st.columns([3, 1])
                c1.write(f"{r.camera_id} · {r.name}")
                new_val = c2.toggle(
                    "Enabled", value=bool(r.enabled),
                    key=f"cam61_enable_{r.camera_id}")
                if new_val != bool(r.enabled):
                    store.set_enabled(r.camera_id, new_val)
                    st.caption(f"{r.camera_id} {'enabled' if new_val else 'disabled'} "
                               "— saved and queued for runtime apply.")
                    st.rerun()

        # -- Apply handshake status ------
        st.markdown("**Runtime apply status**")
        st_c = store.apply_state()
        if not st_c:
            st.caption("No apply handshake yet. The daemon records one once it "
                       "runs a reconcile.")
        else:
            st.caption(
                f"Requested: {st_c.get('requested_revision', '')[:12]} "
                f"at {st_c.get('requested_at') or '-'} · "
                f"Applied: {st_c.get('applied_revision', '')[:12]} "
                f"at {st_c.get('applied_at') or '-'}."
            )
            if (st_c.get("requested_revision") != st_c.get("applied_revision")):
                st.info("A configuration apply is pending — the daemon will "
                        "apply it within a few seconds.")

        st.divider()

        # -- Add camera --------------------
        st.subheader(":material/add_call: Add camera")
        with st.form("cam61_add", clear_on_submit=False):
            c_id = st.text_input("Camera ID",
                                 help="Stable identifier, e.g. CAM01 / cam_01")
            c_name = st.text_input("Camera name")
            c_kind = st.selectbox("Camera type", _kind_choices, index=1)
            c_url = st.text_input("URL / source",
                                  help="For RTSP paste `rtsp://host[:port]/path`. "
                                       "Credentials can stay out of the URL — use "
                                       "the separate username/password fields.")
            c_user = st.text_input("Username (optional)")
            c_pass = st.text_input("Password (optional)", type="password")
            c_loc = st.text_input("Location (optional)")
            c_en = st.checkbox("Enabled", value=True)
            c_fps = st.number_input("Target FPS (0 = system default)",
                                    min_value=0, max_value=240, value=0)
            with st.expander("Reconnect behaviour"):
                c_rb = st.number_input("Reconnect base (s)", min_value=0.0,
                                       value=2.0, step=0.5)
                c_rm = st.number_input("Reconnect max (s)", min_value=0.1,
                                       value=30.0, step=1.0)
                c_rf = st.number_input("Reconnect factor (>=1.0)", min_value=1.0,
                                       value=2.5, step=0.1)
            col_a, col_b = st.columns(2)
            t_res = col_a.form_submit_button("Test Connection")
            save_res = col_b.form_submit_button("Add Camera")

            if t_res:
                probe = test_camera_connection(
                    c_url, kind=c_kind, username=c_user, password=c_pass,
                    reconnect=(c_rb, c_rm, c_rf), timeout=6.0)
                if probe.get("health") == "RUNTIME_UNAVAILABLE":
                    st.error(
                        "TEST CONNECTION UNAVAILABLE IN CLOUD — the camera "
                        "capture runtime (OpenCV) is not available in this "
                        "deployment environment. Camera configuration and "
                        "management remain available.")
                elif probe["ok"]:
                    st.success(
                        f"Connection OK — {probe.get('health')} "
                        f"{probe.get('resolution') or ''} "
                        f"@ {probe.get('fps')} fps.")
                else:
                    st.error(
                        f"Connection unavailable — {probe.get('error')} "
                        f"(probe took {probe.get('elapsed')}s).")
            if save_res:
                try:
                    store.add(
                        camera_id=c_id, name=c_name, kind=c_kind, url=c_url,
                        username=c_user, password=c_pass, enabled=c_en,
                        location=c_loc, fps_target=c_fps,
                        reconnect_base=c_rb, reconnect_max=c_rm,
                        reconnect_factor=c_rf,
                    )
                    st.success(f"Camera {c_id} saved.")
                    store.request_apply()
                    st.rerun()
                except ValueError as exc:
                    st.error(f"Could not save camera: {exc}")

        st.divider()

        # -- Edit camera --------------------
        st.subheader(":material/edit: Edit camera")
        if not records:
            st.caption("Nothing to edit yet.")
        else:
            edit_id = st.selectbox(
                "Select camera to edit", [r.camera_id for r in records],
                key="cam61_edit_sel")
            src = store.get(edit_id)
            if src is not None:
                st.caption(
                    f"Editing {src.camera_id} ({src.name}). The identity "
                    "`camera_id` never changes on edit. Saved password stays "
                    "hidden (change it via the replacement field only).")
                with st.form("cam61_edit"):
                    e_name = st.text_input("Camera name",
                                           value=src.name, key="c61e_name")
                    e_kind = st.selectbox("Camera type", _kind_choices,
                                          index=_kind_choices.index(src.kind),
                                          key="c61e_kind")
                    e_url = st.text_input("URL / source", value=src.url,
                                          key="c61e_url")
                    e_user = st.text_input("Username (optional)",
                                           value=src.username, key="c61e_user")
                    st.caption("Password: saved (********). Leave the "
                               "replacement empty to keep the saved one.")
                    e_pass = st.text_input("Replace password (optional)",
                                           type="password", key="c61e_pass")
                    e_loc = st.text_input("Location (optional)",
                                          value=src.location, key="c61e_loc")
                    e_en = st.checkbox("Enabled", value=bool(src.enabled),
                                       key="c61e_en")
                    e_fps = st.number_input("Target FPS (0 = system default)",
                                            min_value=0, max_value=240,
                                            value=int(src.fps_target or 0),
                                            key="c61e_fps")
                    with st.expander("Reconnect behaviour"):
                        e_rb = st.number_input("Reconnect base (s)",
                                               min_value=0.0,
                                               value=float(src.reconnect_base or 0.0),
                                               step=0.5, key="c61e_rb")
                        e_rm = st.number_input("Reconnect max (s)",
                                               min_value=0.1,
                                               value=float(src.reconnect_max),
                                               step=1.0, key="c61e_rm")
                        e_rf = st.number_input("Reconnect factor (>=1.0)",
                                               min_value=1.0,
                                               value=float(src.reconnect_factor),
                                               step=0.1, key="c61e_rf")
                    col_a, col_b = st.columns(2)
                    e_test = col_a.form_submit_button("Test Connection")
                    e_save = col_b.form_submit_button("Save Changes")
                    if e_test:
                        probe = test_camera_connection(
                            e_url, kind=e_kind, username=e_user,
                            password=(e_pass or src.password),
                            reconnect=(e_rb, e_rm, e_rf), timeout=6.0)
                        if probe.get("health") == "RUNTIME_UNAVAILABLE":
                            st.error(
                                "TEST CONNECTION UNAVAILABLE IN CLOUD — the camera "
                                "capture runtime (OpenCV) is not available in this "
                                "deployment environment. Camera configuration and "
                                "management remain available.")
                        elif probe["ok"]:
                            st.success(
                                f"Connection OK — {probe.get('health')} "
                                f"{probe.get('resolution') or ''} "
                                f"@ {probe.get('fps')} fps.")
                        else:
                            st.error(
                                f"Connection unavailable — {probe.get('error')} "
                                f"(probe took {probe.get('elapsed')}s).")
                    if e_save:
                        updates = {
                            "name": e_name, "kind": e_kind, "url": e_url,
                            "username": e_user, "location": e_loc,
                            "enabled": bool(e_en), "fps_target": e_fps,
                            "reconnect_base": e_rb, "reconnect_max": e_rm,
                            "reconnect_factor": e_rf,
                        }
                        if e_pass:
                            updates["password"] = e_pass
                        try:
                            store.update(src.camera_id, updates)
                            st.success(f"Camera {src.camera_id} saved.")
                            store.request_apply()
                            st.rerun()
                        except ValueError as exc:
                            st.error(f"Could not save camera: {exc}")

        st.divider()

        # -- Delete (safe) + Apply ---------
        st.subheader(":material/delete: Remove / Apply")
        if records:
            with st.expander("Remove a camera (safe)", expanded=False):
                del_id = st.selectbox(
                    "Camera to remove", [r.camera_id for r in records],
                    key="cam61_del_sel")
                warn = st.warning(
                    "Removing deletes the admin record only when no historical "
                    "event/incident/health/zone data still references this "
                    "camera_id. If such history exists, disable the camera "
                    "instead.")
                if st.button("Remove camera", key="cam61_del_btn"):
                    try:
                        if store.delete(del_id):
                            st.success(f"Camera {del_id} removed.")
                            store.request_apply()
                            st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        if st.button(":material/sync: Apply Camera Configuration", key="cam61_apply"):
            store.request_apply()
            st.success("Apply requested — the daemon will reconcile the runtime "
                       "pool within a few seconds.")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def tab_settings():
    import config
    st.header("Settings (read-only)")
    st.caption("Configuration is environment-driven. All values below are "
               "read-only — change them in the `.env` file / environment and "
               "restart. Nothing here silently changes settings, and sensitive "
               "values (SMTP password, RTSP credentials, dashboard passwords) "
               "are never shown.")

    snapshot = {}
    try:
        snapshot = config.runtime_config_snapshot()
    except Exception as exc:  # pragma: no cover - surface gracefully
        st.warning(f"Config snapshot unavailable: {exc}")

    def _kv(label, value):
        return (label, str(value))

    with st.expander(":material/tune: General", expanded=True):
        _show_kv_list([
            _kv("Input source", snapshot.get("input_source", "auto")),
            _kv("Evidence mode", snapshot.get("evidence_mode", config.EVIDENCE_MODE)),
            _kv("Auto-refresh interval (s)", DASH_REFRESH_SEC),
            _kv("Day metrics freshness (s)", DASH_DAY_METRICS_TTL_SEC),
            _kv("Range metrics freshness (s)", DASH_RANGE_METRICS_TTL_SEC),
            _kv("Login gate enabled",
                "yes" if _auth_enabled() else "no (open access, local dev)"),
        ])

    with st.expander(":material/videocam: Camera & Capture", expanded=False):
        _show_kv_list([
            _kv("Target FPS per camera", config.TARGET_FPS_PER_CAMERA),
            _kv("Frame stale cutoff (s)", config.FRAME_STALE_SEC),
            _kv("Startup webcam wait (s)", config.STARTUP_FRAME_WAIT_SEC),
            _kv("Reliability FPS target", snapshot.get("reliability_fps_target")),
        ])

    with st.expander(":material/manage_search: Recognition & Detection", expanded=False):
        _show_kv_list([
            _kv("Face-detect size (px)", config.FACE_DETECT_SIZE),
            _kv("Confidence threshold", config.CONF_THRESHOLD),
            _kv("Face similarity threshold", config.FACE_SIMILARITY_THRESHOLD),
        ])

    with st.expander(":material/schedule: Scheduling & Productivity", expanded=False):
        _show_kv_list([
            _kv("Working window",
                f"{WORK_SCHEDULE.get('start')} – {WORK_SCHEDULE.get('end')}"),
            _kv("Unpaid lunch break", WORK_SCHEDULE.get("lunch")),
            _kv("End-of-day report hour",
                f"{config.EOD_REPORT_HOUR:02d}:00" if config.EOD_REPORT_HOUR else "disabled"),
        ])

    with st.expander(":material/security: Security & Privacy", expanded=False):
        _show_kv_list([
            _kv("Dashboard auth", "enabled" if _auth_enabled() else "open (dev)"),
            _kv("Fail-closed mode", "on" if _auth_fail_closed() else "off"),
            _kv("Session timeout (min)",
                DASH_SESSION_MINUTES if DASH_SESSION_MINUTES > 0 else "never"),
            _kv("Evidence retention (days)", snapshot.get("evidence_retention_days")),
            _kv("Correlation window (s)", snapshot.get("correlation_window_sec")),
            _kv("Risk scoring enabled", "yes" if snapshot.get("risk_enabled") else "no"),
        ])

    with st.expander(":material/notifications: Alerts & Notifications", expanded=False):
        alert_rows = [
            _kv("Incident alert dedup (s)", snapshot.get("incident_alert_dedup_sec")),
            _kv("Temporal repeat threshold", snapshot.get("temporal_repeat_threshold")),
            _kv("Temporal escalate count", snapshot.get("temporal_escalate_count")),
            _kv("Temporal silence (days)", snapshot.get("temporal_silence_days")),
            _kv("Anomaly min samples", snapshot.get("anomaly_min_samples")),
            _kv("Anomaly z-score", snapshot.get("anomaly_zscore")),
        ]
        try:
            es = last_email_status()
            alert_rows.append(_kv("Email last delivery",
                                  "sent" if es.get("ok") is True
                                  else ("failed" if es.get("ok") is False else "no attempts")))
        except Exception:  # pragma: no cover - email status may be unavailable
            pass
        _show_kv_list(alert_rows)

    with st.expander(":material/dns: System & Storage", expanded=False):
        _show_kv_list([
            _kv("Embeddings registry present",
                "yes" if Path(config.EMBEDDINGS_FILE).exists() else "no (not built yet)"),
            _kv("Report output", str(REPORT_OUTPUT_DIR)),
            _kv("Faces directory", str(FACES_DIR)),
        ])

    st.divider()
    st.caption("System and secret configuration (SMTP password, RTSP credentials, "
               "dashboard passwords) is never shown here. Change these in the "
               "`.env` file at the project root and restart.")


def tab_deployment(role: str):
    """Phase 38 -- deployment checklist + honest system health.

    Surfaces the real pre-flight posture and an executable deployment checklist
    in the dashboard.  Always reflects live backend state; never fakes a green
    status when a capability (RTSP/GPU/SMTP/Docker) is unavailable in this
    environment.
    """
    st.header("Deployment & System Check")
    st.caption("Pre-flight posture and deployment checklist. Run "
               "`python -m src.preflight` on the target host for the full "
               "command-line report with `--json` / `--verbose`.")

    # --- Honest pre-flight status (read-only) ---
    with st.expander("Pre-flight status (updated on render)", expanded=True):
        try:
            from src.preflight import run_all_checks
            results = run_all_checks()
            if results:
                import pandas as pd
                df = pd.DataFrame([r.to_dict() for r in results])
                df = df.rename(columns={
                    "category": "Area", "name": "Check", "status": "Status",
                    "detail": "Detail", "remedy": "Remedy"})
                _icon = {"PASS": ":green[PASS]", "WARN": ":orange[WARN]",
                         "FAIL": ":red[FAIL]", "SKIP": ":gray[SKIP]"}
                df["Status"] = df["Status"].map(lambda s: _icon.get(s, s))
                show_cols = [c for c in ["Status", "Area", "Check", "Detail"]
                             if c in df.columns]
                st.dataframe(df[show_cols], width="stretch", hide_index=True)
                n_fail = len([r for r in results if r.status == r.FAIL])
                n_warn = len([r for r in results if r.status == r.WARN])
                n_pass = len([r for r in results if r.status == r.PASS])
                n_skip = len([r for r in results if r.status == r.SKIP])
                st.markdown(f":green[**PASS {n_pass}**] · :orange[WARN {n_warn}] · "
                            f":red[FAIL {n_fail}] · :gray[SKIP {n_skip}]")
                if n_fail:
                    st.error(f"{n_fail} FAIL item(s) -- resolve before pilot.")
                elif n_warn:
                    st.warning(f"{n_warn} WARN item(s) -- review before pilot.")
                else:
                    st.success("READY -- all checks passed.")
            else:
                st.info("No pre-flight results.")
        except Exception as exc:  # pragma: no cover - surface gracefully
            st.warning(f"Pre-flight check unavailable: {exc}")

    # --- Executable deployment checklist (Phase 38 section 16) ---
    st.subheader("Deployment checklist")
    checklist = [
        ("Hardware", "Server/desktop with >= 4 CPU cores, >= 8 GB RAM, SSD storage; "
                     "NVIDIA GPU (with CUDA drivers) for accelerated inference"),
        ("OS", "Windows 10/11 or Linux; Python 3.10-3.12 (3.12 recommended)"),
        ("Runtime", "Set up the `.venv` with `setup_env.ps1`; verify `python -m pytest` "
                    "passes (full suite green)"),
        ("GPU / CUDA", "Only if hardware present: validate via `python -m src.preflight` "
                       "(GPU section); otherwise CPU fallback is used explicitly"),
        ("Cameras", "Enroll employee faces into `data/faces/EMPxxxx.jpg`; verify recognition "
                    "on the webcam path first"),
        ("Network / RTSP", "Provide RTSP URLs in `.env` (`CCTV_CAM_<NN>_URL`); validate each "
                           "with `python -m src.rtsp_harness --url <endpoint>`"),
        ("Storage", "Confirm `data/` is on SSD with capacity for the retention window; "
                    "evidence/backups grow over time"),
        ("Database", "WAL-mode SQLite auto-created; run `PRAGMA integrity_check` (see "
                     "backup/restore)"),
        ("Models", "`yolo11n.pt` present; insightface (`buffalo_l`) installed; embeddings "
                   "rebuilt on enrollment"),
        ("SMTP", "Set real SMTP creds in `.env`; test with `main.py --test-email`"),
        ("Docker", "If containerised: `docker compose up -d --build`; verify healthchecks; "
                   "persist `./data`"),
        ("Security / RBAC", "Set `CCTV_DASH_AUTH=1` + `CCTV_DASH_PASS`/`CCTV_DASH_VIEWER_PASS`; "
                            "deploy behind VPN/reverse-proxy + TLS"),
        ("Backup", "Enable `CCTV_AUTO_BACKUP_DAILY=1`; perform a restore drill"),
        ("Retention", "Set retention windows in `.env` (opt-in); incidents are never auto-purged"),
        ("Monitoring", "Review the Security (System Health 2.0) + Camera Health tabs and the "
                       "application log"),
    ]
    for section, task in checklist:
        st.markdown(f"- **{section}:** {task}")

    st.divider()
    st.caption("This panel is advisory and read-only. Real RTSP/GPU/SMTP/Docker "
               "capabilities are reported as PASS/WARN/FAIL/SKIP by the pre-flight "
               "check and are never fabricated.")


# ======================================================================
# Main
# ======================================================================

# ======================================================================
# Phase 62 -- Admin Control Center
# ======================================================================
# One dashboard page that gives an administrator the single, persistent source
# of truth for office intelligence configuration: employees (display name /
# enable / desk+chair assignment), desks (== seat zones), chairs, the office &
# lunch schedule and the away/phone/talking threshold knobs.  Everything is
# persisted in the existing application SQLite DB (``settings`` table), so the
# daemon and the dashboard always agree.  All mutations are RBAC-gated and
# audited; nothing secret (passwords / embeddings) is ever logged.
#
# Architecture contract: this page REUSES the existing stores (EmployeeStore,
# SeatZoneStore, ChairStore, CameraStore, SettingsStore).  It does not create
# duplicate registries/stores/trackers.  Desk == seat zone in this model: the
# seat_zones table holds the desk polygons; there is no second "desks" table.

_ADMIN_SUB = [
    ("Overview", "overview"),
    ("Cameras", "cameras"),
    ("Employees", "employees"),
    ("Desks & Zones", "zones"),
    ("Chairs", "chairs"),
    ("Schedule", "schedule"),
    ("Thresholds", "thresholds"),
    ("System", "system"),
]
_ADMIN_SUB_LABELS = [label for label, _k in _ADMIN_SUB]
_ADMIN_SUB_KEYS = {label: key for label, key in _ADMIN_SUB}


def _admin_people_label(emp_id: str, name: str, display_name: str) -> str:
    shown = display_name or name
    return f"{emp_id} — {shown}" if shown else emp_id


def _admin_safe_id(raw: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_\-]+", str(raw or "").strip()))


def _admin_polygon_text(text: str) -> tuple[str | None, str]:
    """Validate polygon input (JSON list of [x, y] 0..1 points)."""
    from src.zones import parse_polygon
    poly = parse_polygon(text)
    if not poly:
        return None, "Polygon must be a JSON list of >= 3 [x, y] points in 0..1"
    return json.dumps(poly), ""


def tab_admin(role: str) -> None:
    """Phase 62 -- Admin Control Center (top-level navigation page)."""
    st.header("Admin Control Center")
    st.caption(
        "Single persistent place to manage employees, desks (zones), chairs, "
        "the office/lunch schedule and detection thresholds. All admin writes "
        "are permission-checked and audited."
    )
    choice = st.radio("p62_admin_sub", _ADMIN_SUB_LABELS,
                      key="p62_admin_sub", horizontal=True,
                      label_visibility="collapsed")
    page = _ADMIN_SUB_KEYS[choice]
    if page == "overview":
        tab_admin_overview(role)
    elif page == "cameras":
        tab_admin_cameras(role)
    elif page == "employees":
        tab_admin_employees(role)
    elif page == "zones":
        tab_admin_zones(role)
    elif page == "chairs":
        tab_admin_chairs(role)
    elif page == "schedule":
        tab_admin_schedule(role)
    elif page == "thresholds":
        tab_admin_thresholds(role)
    else:
        tab_admin_system(role)


def tab_admin_overview(role: str) -> None:
    """Read-only dashboard of the current admin-managed configuration state."""
    st.subheader("Configuration Overview")
    conn = get_readonly_connection()
    if conn is None:
        st.error("Database unavailable.")
        return
    try:
        from src.admin_store import SettingsStore
        emps = EmployeeStore(conn, WORK_SCHEDULE).list()
        zones = SeatZoneStore(conn).list()
        chairs = ChairStore(conn).list()
        cams = []
        try:
            from src.camera_store import CameraStore
            cams = CameraStore(conn).list()
        except Exception:  # pragma: no cover - cameras optional
            cams = []
        settings = SettingsStore(conn).get()
    except Exception as exc:  # pragma: no cover - surface gracefully
        st.error(f"Could not read configuration overview: {exc}")
        return
    finally:
        conn.close()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Employees", f"{len(emps)}",
              help=f"{sum(1 for e in emps if e.active)} active")
    m2.metric("Desks (zones)", f"{len(zones)}",
              help=f"{sum(1 for z in zones if z.enabled)} enabled")
    m3.metric("Chairs", f"{len(chairs)}",
              help=f"{sum(1 for c in chairs if c.enabled)} enabled")
    m4.metric("Cameras (admin)", f"{len(cams)}",
              help="configured via Admin → Cameras")

    s1, s2, s3 = st.columns(3)
    s1.info(f"Timezone **{settings.tz}**\n\nOffice **{settings.office_start} → "
            f"{settings.office_end}** · Lunch **{settings.lunch_start} → "
            f"{settings.lunch_end}**")
    s2.info(f"Away **{settings.away_seconds:g}s** · Phone "
            f"**{settings.phone_seconds:g}s** · Talking "
            f"**{settings.talking_seconds:g}s**")
    s3.info(f"Phone pass · class **{settings.phone_class}** · imgsz "
            f"**{settings.phone_imgsz}** · cadence "
            f"**{settings.phone_cadence}**\n\nModel: "
            f"{settings.phone_model_path or 'base YOLO'} · evidence "
            f"**{settings.phone_evidence_min}/{settings.phone_evidence_window}**")
    st.caption("Identity note: desk/chair assignments are context only — an "
               "unknown face is never relabelled, an empty desk is VACANT.")
    st.caption(f"Session role: **{role}** (admin ops are RBAC-gated).")


def tab_admin_employees(role: str) -> None:
    """Admin → Employees: define / rename (display label), enable-disable and
    assign employees to a desk (zone) and chair.  ``employee_id`` is the
    canonical identity and is NEVER changed."""
    st.subheader("Employees")
    st.caption("Identity-safe: the stable `employee_id` is never rewritable; "
               "only the presentation display label changes.")
    conn = get_readonly_connection()
    try:
        emps = EmployeeStore(conn, WORK_SCHEDULE).list()
        zones = SeatZoneStore(conn).list()
        chairs = ChairStore(conn).list()
    finally:
        conn.close()

    if not emps:
        st.info("No employees yet. Create the first one below.")
    else:
        df = pd.DataFrame([{
            "Employee ID": e.employee_id,
            "Display label": e.display_name or e.name or "—",
            "Department": e.department or "—",
            "Active": "✓" if e.active else "—",
            "Desk (zone)": next((z.zone_id for z in zones
                                 if z.assigned_employee_id == e.employee_id),
                                "—"),
            "Chair": next((c.chair_id for c in chairs
                           if c.assigned_employee_id == e.employee_id), "—"),
        } for e in emps])
        st.dataframe(df, width="stretch")

    if role != "admin":
        st.caption("Viewer role: read-only. Ask an admin to modify employees.")
        return

    from src.admin_store import validate_assignment

    with st.form("p62_emp_create"):
        st.markdown("**Add employee**")
        eid = st.text_input("Employee ID (e.g. EMP012)")
        c1, c2 = st.columns(2)
        name = c1.text_input("Name")
        dname = c2.text_input("Display label (optional, shown in UIs)")
        c3, c4 = st.columns(2)
        dept = c3.text_input("Department")
        desig = c4.text_input("Designation")
        active = st.checkbox("Active", value=True)
        create = st.form_submit_button("Save employee")
    if create:
        safe_eid = _safe_employee_id(eid)
        if not eid.strip():
            st.error("Employee ID is required.")
        elif not safe_eid:
            st.error("Invalid Employee ID (letters, digits, _ and - only).")
        elif safe_eid.lower() in ("unknown", "__person__"):
            st.error("Reserved ID; choose a different one.")
        elif any(e.employee_id == safe_eid for e in emps):
            st.error(f"{safe_eid} already exists.")
        else:
            conn = _open_write_conn()
            try:
                from src import database as db
                from src.rbac import AccessGuard, AccessDenied
                AccessGuard(conn, actor=role, role=role).require_manage_users()
                EmployeeStore(conn, WORK_SCHEDULE).upsert(
                    safe_eid, name=name, department=dept,
                    designation=desig, active=active)
                EmployeeStore(conn, WORK_SCHEDULE).set_display_name(
                    safe_eid, dname)
                db.audit(conn, "employee.create", actor=role,
                         resource=safe_eid, detail="p62 admin create")
                conn.commit()
                st.success(f"Saved {safe_eid}.")
                st.rerun()
            except AccessDenied:
                st.error("Your role cannot modify employees.")
            finally:
                conn.close()
        return

    if emps:
        st.divider()
        with st.form("p62_emp_rename"):
            st.markdown("**Rename display label** (identity/`employee_id` unchanged)")
            chosen = st.selectbox(
                "Employee",
                [e.employee_id for e in emps],
                format_func=lambda eid: _admin_people_label(
                    eid, next(e.name for e in emps if e.employee_id == eid),
                    next(e.display_name for e in emps if e.employee_id == eid)),
            )
            new_label = st.text_input("Display label (empty clears)")
            rename = st.form_submit_button("Rename")
        if rename:
            conn = _open_write_conn()
            try:
                from src import database as db
                from src.rbac import AccessGuard, AccessDenied
                AccessGuard(conn, actor=role, role=role).require_manage_users()
                EmployeeStore(conn, WORK_SCHEDULE).set_display_name(
                    chosen, new_label)
                db.audit(conn, "employee.rename", actor=role,
                         resource=chosen,
                         detail=f"display_name={'<cleared>' if not new_label.strip() else new_label.strip()}")
                conn.commit()
                st.success(f"Renamed {chosen}.")
                st.rerun()
            except AccessDenied:
                st.error("Your role cannot rename employees.")
            finally:
                conn.close()

        st.divider()
        with st.form("p62_emp_active"):
            st.markdown("**Enable / disable** (never deletes history)")
            chosen = st.selectbox(
                "Employee", [e.employee_id for e in emps],
                key="p62_emp_active_sel")
            emp = next((e for e in emps if e.employee_id == chosen), None)
            new_active = st.checkbox("Active", value=bool(emp and emp.active))
            toggle = st.form_submit_button("Update")
        if toggle:
            conn = _open_write_conn()
            try:
                from src import database as db
                from src.rbac import AccessGuard, AccessDenied
                AccessGuard(conn, actor=role, role=role).require_manage_users()
                EmployeeStore(conn, WORK_SCHEDULE).set_active(chosen, new_active)
                db.audit(conn, "employee.active", actor=role, resource=chosen,
                         detail="enabled" if new_active else "disabled")
                conn.commit()
                st.success(f"{chosen} {'enabled' if new_active else 'disabled'}.")
                st.rerun()
            except AccessDenied:
                st.error("Your role cannot modify employees.")
            finally:
                conn.close()

        st.divider()
        with st.form("p62_emp_assign"):
            st.markdown("**Assign desk (zone) / chair** — validated against "
                        "missing or disabled targets")
            chosen = st.selectbox(
                "Employee", [e.employee_id for e in emps],
                key="p62_emp_assign_sel")
            zchoice = st.selectbox("Desk (zone) — or blank to unassign",
                                   [""] + [z.zone_id for z in zones])
            cchoice = st.selectbox("Chair — or blank to unassign",
                                   [""] + [ch.chair_id for ch in chairs])
            assign = st.form_submit_button("Save assignment")
        if assign:
            z_id = zchoice or None
            c_id = cchoice or None
            conn = _open_write_conn()
            try:
                from src import database as db
                from src.rbac import AccessGuard, AccessDenied
                problems = validate_assignment(
                    conn, employee_id=chosen, zone_id=z_id, chair_id=c_id)
                if problems:
                    st.error("; ".join(problems))
                else:
                    AccessGuard(conn, actor=role,
                                role=role).require_manage_users()
                    if z_id:
                        SeatZoneStore(conn).set_assignment(z_id, chosen)
                    for z in zones:
                        if (z.assigned_employee_id == chosen
                                and z.zone_id != (z_id or "")):
                            SeatZoneStore(conn).set_assignment(z.zone_id, None)
                    if c_id:
                        ChairStore(conn).set_assignment(c_id, chosen)
                    for c in chairs:
                        if (c.assigned_employee_id == chosen
                                and c.chair_id != (c_id or "")):
                            ChairStore(conn).set_assignment(c.chair_id, None)
                    if z_id:
                        db.audit(conn, "employee.assign_desk", actor=role,
                                 resource=chosen,
                                 detail=f"zone={z_id} (context hint only)")
                    if c_id:
                        db.audit(conn, "employee.assign_chair", actor=role,
                                 resource=chosen,
                                 detail=f"chair={c_id} (context hint only)")
                    conn.commit()
                    st.success(f"Assignment saved for {chosen}.")
                    st.rerun()
            except AccessDenied:
                st.error("Your role cannot assign desks/chairs.")
            finally:
                conn.close()


def tab_admin_zones(role: str) -> None:
    """Admin → Desks & Zones (== seat zones): the desk polygons."""
    st.subheader("Desks & Zones")
    st.caption("A desk is a seat zone: a polygon attached to one camera with "
               "an optional assigned employee (context hint).")
    conn = get_readonly_connection()
    try:
        zones = SeatZoneStore(conn).list()
        cams = []
        try:
            from src.camera_store import CameraStore
            cams = CameraStore(conn).list()
        except Exception:  # pragma: no cover - cameras optional
            cams = []
    finally:
        conn.close()

    if zones:
        st.dataframe(pd.DataFrame([{
            "Desk/Zone": z.zone_id,
            "Camera": z.camera_id,
            "Name": z.name or "—",
            "Enabled": "✓" if z.enabled else "—",
            "Assigned to": z.assigned_employee_id or "—",
            "Points": str(len(z.polygon)) if z.has_polygon() else "—",
        } for z in zones]), width="stretch")

    if role != "admin":
        st.caption("Viewer role: read-only.")
        return

    cam_ids = [c.camera_id for c in cams]
    from src.admin_store import validate_assignment

    with st.form("p62_zone_create"):
        st.markdown("**Add desk (zone)**")
        zid = st.text_input("Desk ID (e.g. DESK-A1 / ZONE-A1)")
        cam = st.selectbox("Camera", cam_ids or ["(add cameras first)"])
        zname = st.text_input("Name (optional)")
        poly = st.text_area(
            "Polygon (JSON): e.g. [[0.4,0.2],[0.7,0.2],[0.7,0.6],[0.4,0.6]]",
            height=70)
        make = st.form_submit_button("Add desk")
    if make:
        if not _admin_safe_id(zid):
            st.error("Desk ID must be letters/digits/_/- only.")
            return
        if not cam_ids:
            st.error("Add a camera first (Admin → Cameras).")
            return
        poly_ok, err = _admin_polygon_text(poly)
        if not poly_ok:
            st.error(err)
            return
        conn = _open_write_conn()
        try:
            from src import database as db
            from src.rbac import AccessGuard, AccessDenied
            AccessGuard(conn, actor=role, role=role).require_configure_zones()
            store = SeatZoneStore(conn)
            if store.get(zid):
                st.error(f"Desk {zid} already exists.")
            else:
                store.add(zone_id=zid, camera_id=cam, polygon=poly_ok,
                          name=zname, enabled=True)
                db.audit(conn, "desk.create", actor=role, resource=zid,
                         detail="seat zone created")
                conn.commit()
                st.success(f"Desk {zid} added.")
                st.rerun()
        except AccessDenied:
            st.error("Your role cannot modify desks/zones.")
        finally:
            conn.close()

    if zones:
        st.divider()
        with st.form("p62_zone_edit"):
            st.markdown("**Edit desk** — rename, move camera, re-draw polygon")
            zsel = st.selectbox("Desk", [z.zone_id for z in zones])
            z = next((z for z in zones if z.zone_id == zsel), None)
            zcam = st.selectbox("Camera", cam_ids or [z.camera_id],
                                index=(cam_ids.index(z.camera_id)
                                       if cam_ids and z and z.camera_id in cam_ids
                                       else 0))
            zname = st.text_input("Name", value=(z.name if z else ""))
            zpoly = st.text_area(
                "Polygon (JSON)", height=70,
                value=json.dumps(z.polygon) if z and z.polygon else "")
            zen = st.checkbox("Enabled", value=bool(z and z.enabled))
            edit = st.form_submit_button("Save desk")
        if edit:
            if not cam_ids:
                st.error("Add a camera first (Admin → Cameras).")
                return
            poly_ok, err = _admin_polygon_text(zpoly)
            if not poly_ok:
                st.error(err)
                return
            conn = _open_write_conn()
            try:
                from src import database as db
                from src.rbac import AccessGuard, AccessDenied
                AccessGuard(conn, actor=role, role=role).require_configure_zones()
                SeatZoneStore(conn).update(
                    zsel, camera_id=zcam, name=zname,
                    polygon=poly_ok, enabled=zen)
                db.audit(conn, "desk.update", actor=role, resource=zsel,
                         detail="seat zone edited")
                conn.commit()
                st.success(f"Desk {zsel} saved.")
                st.rerun()
            except AccessDenied:
                st.error("Your role cannot modify desks/zones.")
            finally:
                conn.close()

        st.divider()
        with st.form("p62_zone_assign"):
            st.markdown("**Assign employee to desk** (context hint only)")
            zsel = st.selectbox("Desk", [z.zone_id for z in zones],
                                key="p62_zone_assign_sel")
            emps = _employees_readonly() or []
            emp_choice = st.selectbox(
                "Employee — or blank to unassign",
                [""] + [e.employee_id for e in emps],
                index=0,
                key="p62_zone_assign_emp")
            do_assign = st.form_submit_button("Save")
        if do_assign:
            conn = _open_write_conn()
            try:
                from src import database as db
                from src.rbac import AccessGuard, AccessDenied
                problems = validate_assignment(
                    conn, zone_id=zsel, employee_id=emp_choice or None)
                if problems:
                    st.error("; ".join(problems))
                else:
                    AccessGuard(conn, actor=role,
                                role=role).require_configure_zones()
                    SeatZoneStore(conn).set_assignment(zsel, emp_choice or None)
                    db.audit(conn, "desk.assignment", actor=role,
                             resource=zsel,
                             detail=f"employee={emp_choice or '<cleared>'}")
                    conn.commit()
                    st.success(f"Desk {zsel} assignment saved.")
                    st.rerun()
            except AccessDenied:
                st.error("Your role cannot modify desks/zones.")
            finally:
                conn.close()

        st.divider()
        with st.form("p62_zone_delete"):
            st.markdown("**Delete desk** — removes the desk config and its "
                        "chairs; events/history are NOT touched")
            zsel = st.selectbox("Desk", [z.zone_id for z in zones],
                                key="p62_zone_del_sel")
            z = next((z for z in zones if z.zone_id == zsel), None)
            confirm = st.checkbox("Confirm: delete this desk config")
            do_del = st.form_submit_button("Delete desk")
        if do_del:
            if not confirm:
                st.error("Tick the confirmation box to delete.")
            else:
                conn = _open_write_conn()
                try:
                    from src import database as db
                    from src.rbac import AccessGuard, AccessDenied
                    AccessGuard(conn, actor=role,
                                role=role).require_configure_zones()
                    for c in ChairStore(conn).for_zone(zsel):
                        ChairStore(conn).remove(c.chair_id)
                    SeatZoneStore(conn).remove(zsel)
                    db.audit(conn, "desk.delete", actor=role, resource=zsel,
                             detail="seat zone + chairs removed (history kept)")
                    conn.commit()
                    st.success(f"Desk {zsel} deleted.")
                    st.rerun()
                except AccessDenied:
                    st.error("Your role cannot modify desks/zones.")
                finally:
                    conn.close()


def _employees_readonly():
    conn = get_readonly_connection()
    try:
        return EmployeeStore(conn, WORK_SCHEDULE).list()
    finally:
        conn.close()


def tab_admin_chairs(role: str) -> None:
    """Admin → Chairs: define, enable, assign seat-level locations."""
    st.subheader("Chairs")
    conn = get_readonly_connection()
    try:
        zones = SeatZoneStore(conn).list()
        chairs = ChairStore(conn).list()
    finally:
        conn.close()

    if chairs:
        zone_name = {z.zone_id: (z.name or z.zone_id) for z in zones}
        st.dataframe(pd.DataFrame([{
            "Chair": c.chair_id,
            "Desk": c.zone_id,
            "Desk name": zone_name.get(c.zone_id, "—"),
            "Camera": c.camera_id,
            "Label": c.label or "—",
            "Enabled": "✓" if c.enabled else "—",
            "Assigned to": c.assigned_employee_id or "—",
        } for c in chairs]), width="stretch")

    if role != "admin":
        st.caption("Viewer role: read-only.")
        return

    from src.admin_store import validate_assignment
    emps = _employees_readonly() or []

    with st.form("p62_chair_create"):
        st.markdown("**Add chair**")
        cid = st.text_input("Chair ID (e.g. A1-C1)")
        zsel = st.selectbox("Desk (zone)",
                            [z.zone_id for z in zones] or ["(add a desk first)"])
        cname = st.text_input("Name (optional)")
        clabel = st.text_input("Label (optional, e.g. A01-C1)")
        make = st.form_submit_button("Add chair")
    if make:
        if not _admin_safe_id(cid):
            st.error("Chair ID must be letters/digits/_/- only.")
            return
        if not zones:
            st.error("Add a desk (Desks & Zones) first.")
            return
        conn = _open_write_conn()
        try:
            from src import database as db
            from src.rbac import AccessGuard, AccessDenied
            AccessGuard(conn, actor=role, role=role).require_configure_zones()
            store = ChairStore(conn)
            if store.get(cid):
                st.error(f"Chair {cid} already exists.")
            else:
                z = next(z for z in zones if z.zone_id == zsel)
                store.add(chair_id=cid, camera_id=z.camera_id, zone_id=zsel,
                          name=cname, label=clabel, enabled=True)
                db.audit(conn, "chair.create", actor=role, resource=cid,
                         detail=f"zone={zsel}")
                conn.commit()
                st.success(f"Chair {cid} added.")
                st.rerun()
        except AccessDenied:
            st.error("Your role cannot modify chairs.")
        finally:
            conn.close()

    if chairs:
        st.divider()
        with st.form("p62_chair_edit"):
            st.markdown("**Edit chair** — name, label, desk, enabled")
            csel = st.selectbox("Chair", [c.chair_id for c in chairs])
            c = next((c for c in chairs if c.chair_id == csel), None)
            cname = st.text_input("Name", value=(c.name if c else ""),
                                  key="p62_chair_edit_name")
            clabel = st.text_input("Label", value=(c.label if c else ""))
            zsel = st.selectbox("Desk (zone)",
                                [z.zone_id for z in zones] or [c.zone_id],
                                index=([z.zone_id for z in zones].index(c.zone_id)
                                       if c and c.zone_id in [z.zone_id for z in zones]
                                       else 0))
            cen = st.checkbox("Enabled", value=bool(c and c.enabled),
                              key="p62_chair_edit_en")
            edit = st.form_submit_button("Save chair")
        if edit:
            if not zones:
                st.error("Add a desk (Desks & Zones) first.")
                return
            conn = _open_write_conn()
            try:
                from src import database as db
                from src.rbac import AccessGuard, AccessDenied
                AccessGuard(conn, actor=role, role=role).require_configure_zones()
                z = next(z for z in zones if z.zone_id == zsel)
                ChairStore(conn).update(csel, name=cname, label=clabel,
                                        camera_id=z.camera_id, zone_id=zsel,
                                        enabled=cen)
                db.audit(conn, "chair.update", actor=role, resource=csel,
                         detail=f"zone={zsel}")
                conn.commit()
                st.success(f"Chair {csel} saved.")
                st.rerun()
            except AccessDenied:
                st.error("Your role cannot modify chairs.")
            finally:
                conn.close()

        st.divider()
        with st.form("p62_chair_assign"):
            st.markdown("**Assign employee to chair** (context hint only)")
            csel = st.selectbox("Chair", [c.chair_id for c in chairs],
                                key="p62_chair_assign_sel")
            emp_choice = st.selectbox(
                "Employee — or blank to unassign",
                [""] + [e.employee_id for e in emps],
                index=0, key="p62_chair_assign_emp")
            do_assign = st.form_submit_button("Save")
        if do_assign:
            conn = _open_write_conn()
            try:
                from src import database as db
                from src.rbac import AccessGuard, AccessDenied
                problems = validate_assignment(
                    conn, chair_id=csel, employee_id=emp_choice or None)
                if problems:
                    st.error("; ".join(problems))
                else:
                    AccessGuard(conn, actor=role,
                                role=role).require_configure_zones()
                    ChairStore(conn).set_assignment(csel, emp_choice or None)
                    db.audit(conn, "chair.assignment", actor=role,
                             resource=csel,
                             detail=f"employee={emp_choice or '<cleared>'}")
                    conn.commit()
                    st.success(f"Chair {csel} assignment saved.")
                    st.rerun()
            except AccessDenied:
                st.error("Your role cannot modify chairs.")
            finally:
                conn.close()

        st.divider()
        with st.form("p62_chair_delete"):
            st.markdown("**Delete chair** — config only; history untouched")
            csel = st.selectbox("Chair", [c.chair_id for c in chairs],
                                key="p62_chair_del_sel")
            confirm = st.checkbox("Confirm: delete this chair config")
            do_del = st.form_submit_button("Delete chair")
        if do_del:
            if not confirm:
                st.error("Tick the confirmation box to delete.")
            else:
                conn = _open_write_conn()
                try:
                    from src import database as db
                    from src.rbac import AccessGuard, AccessDenied
                    AccessGuard(conn, actor=role,
                                role=role).require_configure_zones()
                    ChairStore(conn).remove(csel)
                    db.audit(conn, "chair.delete", actor=role, resource=csel,
                             detail="chair config removed (history kept)")
                    conn.commit()
                    st.success(f"Chair {csel} deleted.")
                    st.rerun()
                except AccessDenied:
                    st.error("Your role cannot modify chairs.")
                finally:
                    conn.close()


def tab_admin_schedule(role: str) -> None:
    """Admin → Schedule: the single authoritative office/lunch schedule."""
    st.subheader("Office & Lunch Schedule")
    conn = get_readonly_connection()
    settings = None
    if conn is not None:
        try:
            from src.admin_store import SettingsStore
            settings = SettingsStore(conn).get()
        finally:
            conn.close()
    if settings is None:
        st.error("Database unavailable.")
        return
    st.caption(
        "Persisted in the application database and applied to the runtime on "
        "save + at daemon startup. Office hours and lunch must be HH:MM (24h), "
        "lunch inside office hours, timezone a valid IANA zone (default "
        "Asia/Kolkata)."
    )
    if role != "admin":
        st.caption("Viewer role: read-only.")
        return

    with st.form("p62_sched"):
        tz = st.text_input("Timezone (IANA)", settings.tz)
        c1, c2 = st.columns(2)
        os_ = c1.text_input("Office start (HH:MM)", settings.office_start)
        oe = c2.text_input("Office end (HH:MM)", settings.office_end)
        c3, c4 = st.columns(2)
        ls = c3.text_input("Lunch start (HH:MM)", settings.lunch_start)
        le = c4.text_input("Lunch end (HH:MM)", settings.lunch_end)
        save = st.form_submit_button("Save schedule")
    if save:
        conn = _open_write_conn()
        try:
            from src.admin_store import SettingsStore
            from src.rbac import AccessGuard, AccessDenied
            AccessGuard(conn, actor=role, role=role).require_configure_settings()
            SettingsStore(conn).update(actor=role, timezone=tz,
                                       office_start=os_, office_end=oe,
                                       lunch_start=ls, lunch_end=le)
            SettingsStore(conn).apply_to_config()
            conn.commit()
            st.success("Schedule saved and applied to the runtime.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
        except AccessDenied:
            st.error("Your role cannot change the schedule.")
        finally:
            conn.close()


def tab_admin_thresholds(role: str) -> None:
    """Admin → Thresholds: away/phone/talking knobs + the Phase 63 phone pass."""
    st.subheader("Detection Thresholds")
    conn = get_readonly_connection()
    settings = None
    if conn is not None:
        try:
            from src.admin_store import SettingsStore
            settings = SettingsStore(conn).get()
        finally:
            conn.close()
    if settings is None:
        st.error("Database unavailable.")
        return
    st.caption(
        "Centralized, persisted thresholds (seconds). Defaults: away 10 s, "
        "phone 5 s, talking 15 s. Away/phone are wired into the live "
        "employee FSM; phone-pass knobs below steer the dedicated YOLO phone "
        "detection pass. Unpersisted knobs keep their env defaults."
    )
    if role != "admin":
        st.caption("Viewer role: read-only.")
        return

    with st.form("p62_thresh"):
        c1, c2, c3 = st.columns(3)
        away = c1.number_input(
            "Away threshold (seconds)", min_value=0.0, step=1.0,
            value=float(settings.away_seconds))
        phone = c2.number_input(
            "Phone threshold (seconds)", min_value=0.0, step=1.0,
            value=float(settings.phone_seconds))
        talking = c3.number_input(
            "Talking threshold (seconds)", min_value=0.0, step=1.0,
            value=float(settings.talking_seconds))
        st.divider()
        st.markdown("**Phone detection pass (Phase 63)**")
        st.caption(
            "Steers the cadence-gated, higher-resolution phone pass. Empty "
            "model path = reuse the base YOLO model (COCO class 67).")
        p_class = st.number_input(
            "Phone class id", min_value=0, max_value=999, step=1,
            value=int(settings.phone_class), help="67 = COCO cell phone; a "
            "fine-tuned single-class model commonly uses 0.")
        p_model = st.text_input(
            "Phone model path (absolute .pt path, empty = base model)",
            value=str(settings.phone_model_path))
        pc1, pc2, pc3 = st.columns(3)
        p_imgsz = pc1.number_input(
            "Phone pass resolution (imgsz)", min_value=32, max_value=4096,
            step=32, value=int(settings.phone_imgsz))
        p_cadence = pc2.number_input(
            "Phone pass cadence (cycles)", min_value=1, max_value=60, step=1,
            value=int(settings.phone_cadence))
        p_ev = pc3.number_input(
            "Evidence window (frames)", min_value=1, max_value=60, step=1,
            value=int(settings.phone_evidence_window))
        p_ev_min = st.number_input(
            "Evidence minimum (frames within window)", min_value=1,
            max_value=60, step=1, value=int(settings.phone_evidence_min),
            help="Raw phone observations required inside the trailing window "
            "before a track reports phone (temporal hysteresis).")
        save = st.form_submit_button("Save thresholds")
    if save:
        conn = _open_write_conn()
        try:
            from src.admin_store import SettingsStore
            from src.rbac import AccessGuard, AccessDenied
            AccessGuard(conn, actor=role, role=role).require_configure_settings()
            SettingsStore(conn).update(
                actor=role, away_seconds=away, phone_seconds=phone,
                talking_seconds=talking, phone_class=p_class,
                phone_model_path=p_model, phone_imgsz=p_imgsz,
                phone_cadence=p_cadence, phone_evidence_window=p_ev,
                phone_evidence_min=p_ev_min)
            SettingsStore(conn).apply_to_config()
            conn.commit()
            st.success("Thresholds saved and applied.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
        except AccessDenied:
            st.error("Your role cannot change thresholds.")
        finally:
            conn.close()


def tab_admin_system(role: str) -> None:
    """Admin → System: read-only inspection of persisted admin state."""
    st.subheader("System")
    conn = get_readonly_connection()
    if conn is None:
        st.error("Database unavailable.")
        return
    try:
        from src import database as db
        from src.admin_store import SettingsStore
        store = SettingsStore(conn)
        snapshot = store.snapshot()
        apply_state = db.get_camera_apply_state(conn)
        tries = conn.execute(
            "SELECT count(*) FROM admin_cameras").fetchone()
        cam_count = int(tries[0]) if tries else 0
    except Exception as exc:  # pragma: no cover - surface gracefully
        st.error(f"Could not read system state: {exc}")
        return
    finally:
        conn.close()

    st.dataframe(pd.DataFrame([
        {"Setting": k, "Value": str(v)} for k, v in snapshot.items()
    ]), width="stretch")
    if apply_state:
        st.info(f"**Camera apply state** — requested rev "
                f"`{apply_state['requested_revision'] or '—'}` · applied rev "
                f"`{apply_state['applied_revision'] or '—'}` · "
                f"{apply_state.get('detail') or ''}")
    st.caption(f"Admin cameras persisted: {cam_count}. History is never "
               f"deleted by enable/disable or delete actions.")


def main():
    role = do_auth()
    if not role:
        return

    _inject_shell_css()
    _render_topbar(role)

    with st.sidebar:
        snap = _snapshot_status(load_live_state())
        conn_tag = {"live": "Live data", "stale": "Stale data",
                    "offline": "Daemon offline"}[snap["kind"]]
        st.html(
            _COMPONENT_CSS +
            "<div class='p42 p42-session-card'>"
            f"<div class='p42-s-row'><span>Role</span>"
            f"<span>{'Admin' if role == 'admin' else 'Viewer'}</span></div>"
            f"<div class='p42-s-row'><span>System</span><span>{html.escape(conn_tag)}</span></div>"
            f"<div class='p42-s-row'><span>Snapshot</span><span>{html.escape(snap['label'])}</span></div>"
            "</div>"
        )
        st.divider()
        st.markdown(
            "<span class='p42-nav-legend'><b>MONITORING</b> Dashboard · Live &nbsp;·&nbsp; "
            "<b>PEOPLE</b> Employees &nbsp;·&nbsp; <b>SECURITY</b> Security · Reports · "
            "Analytics &nbsp;·&nbsp; <b>INFRASTRUCTURE</b> Cameras · Admin · Settings · "
            "Deploy</span>",
            unsafe_allow_html=True,
        )
        choice = st.radio("cctv_nav", _NAV_LABELS, key="p42_nav",
                          label_visibility="collapsed")
        st.divider()
        if st.button(":material/refresh: Refresh now", key="p42_refresh"):
            st.rerun()
        if st.button(":material/logout: Log out", key="p42_logout"):
            st.session_state.pop("cctv_role", None)
            st.session_state.pop("cctv_login_ts", None)
            st.rerun()
        st.caption(f"Auto-refresh: {DASH_REFRESH_SEC}s")

    try:
        page = _NAV_KEYS[choice]
        if page == "dashboard":
            if DASH_REFRESH_SEC:
                _auto_fragment(tab_live_overview, DASH_REFRESH_SEC)
            else:
                tab_live_overview()
        elif page == "live":
            if DASH_REFRESH_SEC:
                _auto_fragment(page_live_monitoring, DASH_REFRESH_SEC)
            else:
                page_live_monitoring()
        elif page == "employees":
            tab_employees(role)
        elif page == "security":
            if DASH_REFRESH_SEC:
                _auto_fragment(tab_security, DASH_REFRESH_SEC, role)
            else:
                tab_security(role)
        elif page == "reports":
            tab_reports(role)
        elif page == "analytics":
            tab_historical()
        elif page == "cameras":
            tab_admin_cameras(role)
        elif page == "admin":
            tab_admin(role)
        elif page == "settings":
            tab_settings()
        else:
            tab_deployment(role)
    except Exception as exc:  # Phase 65: never expose a raw traceback in the UI
        _render_error_card(
            "Something went wrong while rendering this page",
            f"An unexpected error occurred ({type(exc).__name__}). The incident "
            "was logged internally; use the diagnostic id to correlate it. "
            "Refresh now to retry."
        )

    _render_footer(role)


def _auto_fragment(func, every: float, *args):
    """Render a live tab inside an auto-refreshing Streamlit fragment."""
    @st.fragment(run_every=every)
    def _frag():
        func(*args)
    _frag()


if __name__ == "__main__":
    import config
    main()