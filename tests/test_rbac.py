"""Tests for Phase 32 RBAC enforcement layer."""

import pytest

from src import database as db
from src.rbac import AccessGuard, AccessDenied


def _incident(conn, inc_id="INC-1"):
    db.insert_incident(conn, {
        "incident_id": inc_id, "event_type": "INTRUSION", "severity": "HIGH",
        "status": "OPEN", "first_seen": "2026-09-03 09:00:00",
        "last_seen": "2026-09-03 09:00:00", "count": 1,
        "camera": "cam_01", "zone": "archive", "employee_id": "Unknown",
        "confidence": 0.8, "notes": "",
    })


def test_viewer_can_view_not_resolve(tmp_db):
    _incident(tmp_db)
    guard = AccessGuard(tmp_db, actor="viewer1", role="viewer")
    guard.require_view()  # ok
    with pytest.raises(AccessDenied):
        guard.resolve_incident("INC-1")


def test_security_operator_can_acknowledge(tmp_db):
    _incident(tmp_db)
    guard = AccessGuard(tmp_db, actor="op1", role="security_operator")
    guard.acknowledge_incident("INC-1")
    assert db.get_incident(tmp_db, "INC-1")["acknowledged_at"]


def test_denial_is_audited(tmp_db):
    _incident(tmp_db)
    guard = AccessGuard(tmp_db, actor="viewer1", role="viewer")
    with pytest.raises(AccessDenied):
        guard.resolve_incident("INC-1")
    events = db.list_audit_events(tmp_db)
    assert any(e["action"] == "access.denied" and e["actor"] == "viewer1"
               for e in events)


def test_manager_can_export_not_evidence(tmp_db):
    guard = AccessGuard(tmp_db, actor="mgr1", role="manager")
    guard.require_export()  # ok
    guard.require_investigate()  # ok
    with pytest.raises(AccessDenied):
        guard.require_evidence_view()


def test_admin_manage_users(tmp_db):
    guard = AccessGuard(tmp_db, actor="admin1", role="admin")
    guard.require_manage_users()
    guard.require_configure_cameras()
    guard.require_configure_zones()
    guard.require_configure_alerts()
    guard.require_evidence_view()
    guard.require_export()
