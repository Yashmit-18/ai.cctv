"""Append-only audit log writer (Phase 31).

Records security-relevant actions for investigation and compliance.
Never logs secrets, passwords, RTSP credentials, or biometric embeddings.
"""

from __future__ import annotations

import logging
from datetime import datetime

from src import database as db

logger = logging.getLogger("cctv.audit")

# Vocabulary of tracked actions.
ACT_LOGIN = "login"
ACT_LOGOUT = "logout"
ACT_EMPLOYEE_CREATE = "employee.create"
ACT_EMPLOYEE_UPDATE = "employee.update"
ACT_EMPLOYEE_DELETE = "employee.delete"
ACT_ENROLLMENT_CHANGE = "enrollment.change"
ACT_CONFIG_CHANGE = "config.change"
ACT_INCIDENT_ACK = "incident.acknowledged"
ACT_INCIDENT_RESOLVE = "incident.resolved"
ACT_INCIDENT_DISMISS = "incident.dismissed"
ACT_EVIDENCE_DELETE = "evidence.delete"
ACT_REPORT_GENERATE = "report.generate"
ACT_RETENTION_RUN = "retention.run"
ACT_BACKUP_RUN = "backup.run"
ACT_ZONE_CREATE = "zone.create"
ACT_ZONE_UPDATE = "zone.update"
ACT_ZONE_DELETE = "zone.delete"
ACT_ALERT_RULE_UPDATE = "alert_rule.update"


class AuditLog:
    """Thin append-only writer over the ``audit_log`` table."""

    def __init__(self, conn):
        self._conn = conn

    def record(self, action: str, *, actor: str = "system",
               resource: str | None = None, detail: str | None = None) -> None:
        """Write one audit row.  Never raises."""
        try:
            db.audit(self._conn, action, actor=actor,
                     resource=resource, detail=detail)
        except Exception as exc:
            logger.warning("Audit log write failed: %s", exc)

    def list(self, limit: int = 200) -> list[dict]:
        return db.list_audit_events(self._conn, limit=limit)
