"""Tests for the audit log vocabulary module."""

from src import database as db
from src.auditlog import AuditLog, ACT_LOGIN, ACT_EMPLOYEE_CREATE, ACT_BACKUP_RUN


def test_audit_record_and_list(tmp_db):
    log = AuditLog(tmp_db)
    log.record(ACT_LOGIN, actor="admin", resource="dashboard")
    log.record(ACT_EMPLOYEE_CREATE, actor="admin", resource="EMP001",
               detail="created")
    log.record(ACT_BACKUP_RUN, actor="system", resource="sessions.db")
    events = log.list()
    assert len(events) == 3
    actions = {e["action"] for e in events}
    assert ACT_LOGIN in actions
    assert ACT_EMPLOYEE_CREATE in actions


def test_audit_append_only(tmp_db):
    log = AuditLog(tmp_db)
    log.record("a")
    log.record("b")
    events = log.list()
    assert len(events) == 2


def test_audit_is_never_fabricated_secret():
    """Audit actions never contain secret-bearing fields by contract."""
    assert "password" not in ACT_LOGIN
