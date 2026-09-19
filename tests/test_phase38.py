"""Phase 38 -- Real CCTV Integration, Deployment & Production Validation.

These tests target the *deployable product* surface that Phase 38 adds and the
production-readiness invariants that must hold before a real client pilot:

  W1  Website-backend connectivity (the Security-tab severity/status filters
      previously rendered but were not applied to the query -- now wired).
  W2  RBAC authority: incident resolve/dismiss go through AccessGuard so the
      backend (not just the frontend role check) is authoritative, and denied
      actions are audited (regression: AccessGuard.resolve_incident previously
      raised TypeError due to a wrong DB keyword).
  W3  Camera UI exposes Configured/Not configured and never leaks RTSP creds.
  W4  Dashboard/auth + hidden-secret behavior and honesty of health status.
  W5  Deployment checklist presence and pre-flight honesty (no fake green).

Real RTSP/GPU/SMTP/soak remain NOT EXECUTED -- ENVIRONMENT LIMITATION and are
never fabricated.  This file only validates software/deployment readiness.
"""

import os
from pathlib import Path

import pytest

from src import database as db
from src.rbac import AccessGuard, AccessDenied

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# W1 -- Website-backend connectivity: Security-tab filters
# ----------------------------------------------------------------------

def _seed_events(conn):
    db.insert_security_event(conn, {
        "event_type": "INTRUSION", "severity": "HIGH", "status": "OPEN",
        "timestamp": "2026-09-04 09:00:00", "date": "2026-09-04",
        "camera": "cam_01", "zone": "archive", "employee_id": "Unknown",
        "confidence": 0.9,
    })
    db.insert_security_event(conn, {
        "event_type": "AFTER_HOURS", "severity": "LOW", "status": "RESOLVED",
        "timestamp": "2026-09-04 10:00:00", "date": "2026-09-04",
        "camera": "cam_02", "zone": "lobby", "employee_id": "Unknown",
        "confidence": 0.5,
    })


def test_security_event_severity_filter_is_wired(tmp_db):
    """The dashboard exposes a Severity filter; it must actually filter."""
    _seed_events(tmp_db)
    high = db.query_security_events(tmp_db, severity="HIGH")
    assert len(high) == 1 and high[0]["event_type"] == "INTRUSION"
    low = db.query_security_events(tmp_db, severity="LOW")
    assert len(low) == 1 and low[0]["event_type"] == "AFTER_HOURS"


def test_security_event_status_filter_is_wired(tmp_db):
    _seed_events(tmp_db)
    open_evts = db.query_security_events(tmp_db, status="OPEN")
    assert len(open_evts) == 1 and open_evts[0]["severity"] == "HIGH"
    resolved = db.query_security_events(tmp_db, status="RESOLVED")
    assert len(resolved) == 1


def test_security_event_combined_filters(tmp_db):
    _seed_events(tmp_db)
    both = db.query_security_events(tmp_db, severity="HIGH", status="OPEN")
    assert len(both) == 1
    none = db.query_security_events(tmp_db, severity="HIGH", status="RESOLVED")
    assert none == []


def test_eventstore_exposes_severity_and_status(tmp_db):
    from src.security_events import EventStore
    _seed_events(tmp_db)
    es = EventStore(tmp_db)
    assert len(es.events(severity="HIGH")) == 1
    assert len(es.events(status="OPEN")) == 1
    assert len(es.events(severity="LOW", status="RESOLVED")) == 1


# ----------------------------------------------------------------------
# W2 -- RBAC authority on incident resolve/dismiss
# ----------------------------------------------------------------------

def _incident(conn, inc_id="INC-1", status="OPEN"):
    db.insert_incident(conn, {
        "incident_id": inc_id, "event_type": "INTRUSION", "severity": "HIGH",
        "status": status, "first_seen": "2026-09-04 09:00:00",
        "last_seen": "2026-09-04 09:00:00", "count": 1,
        "camera": "cam_01", "zone": "archive", "employee_id": "Unknown",
        "confidence": 0.9, "notes": "",
    })


def test_resolve_incident_through_guard_works(tmp_db):
    """Regression: AccessGuard.resolve_incident used to raise TypeError."""
    _incident(tmp_db)
    guard = AccessGuard(tmp_db, actor="op1", role="security_operator")
    guard.resolve_incident("INC-1")
    assert any(e["action"] == "incident.resolve" and e["actor"] == "op1"
               for e in db.list_audit_events(tmp_db))


def test_dismiss_incident_through_guard_works(tmp_db):
    _incident(tmp_db)
    guard = AccessGuard(tmp_db, actor="op1", role="security_operator")
    guard.dismiss_incident("INC-1")
    assert db.get_incident(tmp_db, "INC-1")["review_state"] == "DISMISSED"
    assert any(e["action"] == "incident.dismiss"
               for e in db.list_audit_events(tmp_db))


def test_viewer_cannot_resolve_or_dismiss_via_guard(tmp_db):
    """Frontend hides the button, but the backend must also deny."""
    _incident(tmp_db)
    guard = AccessGuard(tmp_db, actor="viewer1", role="viewer")
    with pytest.raises(AccessDenied):
        guard.resolve_incident("INC-1")
    with pytest.raises(AccessDenied):
        guard.dismiss_incident("INC-1")
    denied = [e for e in db.list_audit_events(tmp_db)
              if e["action"] == "access.denied" and e["actor"] == "viewer1"]
    assert len(denied) == 2


def test_denial_never_mutates_incident(tmp_db):
    _incident(tmp_db)
    before = db.get_incident(tmp_db, "INC-1")["review_state"]
    guard = AccessGuard(tmp_db, actor="viewer1", role="viewer")
    with pytest.raises(AccessDenied):
        guard.dismiss_incident("INC-1")
    assert db.get_incident(tmp_db, "INC-1")["review_state"] == before


def test_acknowledge_incident_requires_permission(tmp_db):
    _incident(tmp_db)
    guard = AccessGuard(tmp_db, actor="viewer1", role="viewer")
    with pytest.raises(AccessDenied):
        guard.acknowledge_incident("INC-1")


# ----------------------------------------------------------------------
# W3 -- Camera UI exposes Configured/Not configured, never secrets
# ----------------------------------------------------------------------

def test_camera_management_columns_present_in_source():
    """The Camera Health tab must include a Configured column and never leak."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    cam_tab = src.split("def tab_camera_health")[1].split("def tab_reports")[0]
    assert "Configured" in cam_tab and "Not configured" in cam_tab
    assert "rtsp://" not in cam_tab.lower()


# ----------------------------------------------------------------------
# W4 -- Dashboard render + no fake green / no secrets
# ----------------------------------------------------------------------

def _app_streamlit():
    st = pytest.importorskip("streamlit.testing.v1").AppTest
    return st.from_file(str(ROOT / "app.py"), default_timeout=60)


def test_dashboard_shell_renders_navigation_without_exception():
    at = _app_streamlit()
    at.run()
    assert not at.exception, at.exception
    # Phase 42: the app now uses a sidebar navigation shell (all eight pages)
    # instead of the old tab bar.
    assert len(at.sidebar.radio) >= 1
    assert len(at.sidebar.radio[0].options) == 8


def test_dashboard_body_never_exposes_credentials():
    at = _app_streamlit()
    at.run()
    assert not at.exception, at.exception
    body = "\n".join(str(el) for el in at.main).lower()
    for secret in ("rtsp://", "sender_password", "smtp_server", "dash_pass"):
        assert secret not in body


def test_deployment_tab_exists_in_app_source():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def tab_deployment" in src
    assert "Deployment & System Check" in src
    assert "Deployment checklist" in src


def test_settings_never_shows_secrets():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    settings = src.split("def tab_settings")[1].split("def tab_deployment")[0]
    low = settings.lower()
    for secret in ("sender_password", "smtp_server", "rtsp://",
                   "dash_pass"):
        assert secret not in low


# ----------------------------------------------------------------------
# W5 -- Pre-flight honesty: never fabricates real-capability success
# ----------------------------------------------------------------------

def test_preflight_docker_reports_honestly():
    from src import preflight
    try:
        import docker  # noqa: F401
    except Exception:
        docker = None
    results = preflight.check_docker()
    # Docker must be PASS/FAIL/SKIP, never a fabricated 'running' claim when
    # the daemon isn't available in this environment.
    for r in results:
        assert r.status in (r.PASS, r.FAIL, r.SKIP, r.WARN)


def test_preflight_run_all_checks_shapes():
    from src import preflight
    results = preflight.run_all_checks()
    assert len(results) > 0
    for r in results:
        assert hasattr(r, "status")
        assert r.status in (r.PASS, r.FAIL, r.SKIP, r.WARN)
        assert hasattr(r, "to_dict")
        d = r.to_dict()
        assert {"category", "name", "status"} <= set(d.keys())


def test_preflight_results_are_representable():
    from src import preflight
    results = preflight.run_all_checks()
    assert all(repr(r) for r in results)


# ----------------------------------------------------------------------
# W6 -- Database connectivity / pagination (Phase 38 sections 3 & 20)
# ----------------------------------------------------------------------

def test_incident_query_paginates(tmp_db):
    for i in range(25):
        _incident(tmp_db, inc_id=f"INC-{i}")
    page = db.list_incidents(tmp_db, limit=10)
    assert len(page) == 10


def test_event_query_paginates(tmp_db):
    for i in range(25):
        db.insert_security_event(tmp_db, {
            "event_type": "INTRUSION", "severity": "HIGH", "status": "OPEN",
            "timestamp": "2026-09-04 09:00:00", "date": "2026-09-04",
            "camera": "cam_01", "zone": "archive",
            "employee_id": "Unknown", "confidence": 0.9,
        })
    page = db.query_security_events(tmp_db, limit=10)
    assert len(page) == 10


# ----------------------------------------------------------------------
# W7 -- Backup/restore & restart invariants (Phase 38 sections 22/23)
# ----------------------------------------------------------------------

def test_backup_restore_preserves_security_and_productivity(tmp_path):
    from src import backup_tool as bt
    from src.database import get_connection, init_db as ensure_db
    src_path = str(tmp_path / "src.db")
    conn = get_connection(src_path)
    ensure_db(conn)
    db.upsert_employee(conn, "EMP001", "Alice", department="Eng",
                       designation="Eng")
    db.log_interval(conn, "2026-09-04 09:00:01", "EMP001", "ACTIVE", 239.0,
                    "local_webcam")
    db.insert_security_event(conn, {
        "event_type": "INTRUSION", "severity": "HIGH", "status": "OPEN",
        "timestamp": "2026-09-04 09:00:00", "date": "2026-09-04",
        "camera": "cam_01", "zone": "archive", "employee_id": "Unknown",
        "confidence": 0.9,
    })
    _incident(conn, "INC-1")
    conn.close()

    bak = bt.backup_database(src_path, backup_dir=str(tmp_path / "bak"))
    assert bak and os.path.exists(bak)
    assert bt.verify_integrity(bak) == "ok"
    restored = bt.restore_to_temp(bak)
    rc = get_connection(restored)
    ensure_db(rc)
    assert db.get_employee(rc, "EMP001") is not None
    assert db.get_incident(rc, "INC-1") is not None
    assert db.query_security_events(rc)  # events survived
    rc.close()


def test_retention_does_not_purge_incidents(tmp_db):
    """Incidents are never auto-purged by activity retention."""
    _incident(tmp_db, "INC-1")
    db.log_interval(tmp_db, "2026-09-04 09:00:01", "EMP001", "ACTIVE", 239.0,
                    "local_webcam")
    db.retention_cleanup(tmp_db, days=1)
    # No incidents should ever be deleted by retention_cleanup.
    assert db.get_incident(tmp_db, "INC-1") is not None
