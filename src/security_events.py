"""Canonical security-event model and persistence wrapper (Phase 31).

A :class:`SecurityEvent` carries the canonical fields required by the Phase 31
brief: ``incident_id, event_type, severity, timestamp, camera, zone,
employee_id`` (``"Unknown"`` allowed), ``confidence``, ``status,
evidence_ref, created_at, resolved_at, notes``.  Workplace privacy and
accuracy rules are enforced here:

* ``employee_id`` is an identifier only -- never a claim of intent.
* Event types use the reserved, non-fabricated vocabulary from
  :mod:`src.domain` (``*_SUSPECTED`` for machine-only determinations).
* Severity defaults come from ``DEFAULT_EVENT_SEVERITY`` and persist as-is.

Persistence goes through :mod:`src.database` so the daemon and dashboard share
the same tables (``security_events`` + ``incidents``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from src import database as db
from src.domain import (
    DEFAULT_EVENT_SEVERITY,
    INCIDENT_OPEN,
    SEV_INFO,
    SEVERITY_ORDER,
    UNKNOWN_ID,
)

logger = logging.getLogger("cctv.security_events")


def severity_to_index(severity: str) -> int:
    """Rank a severity for comparisons (``INFO`` = 0 .. ``CRITICAL`` = 4)."""
    try:
        return SEVERITY_ORDER.index(str(severity).upper())
    except ValueError:
        return 0


def severity_ge(severity: str, min_severity: str) -> bool:
    """True when ``severity`` is at least ``min_severity`` on the ladder."""
    return severity_to_index(severity) >= severity_to_index(min_severity)


@dataclass
class SecurityEvent:
    """One immutable security event.  Fields follow the Phase 31 model."""

    event_type: str
    severity: str | None = field(default=None)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    camera: str | None = None
    zone: str | None = None
    employee_id: str | None = None
    confidence: float = 0.0
    duration_seconds: float | None = None
    person_count: int | None = None
    status: str = INCIDENT_OPEN
    incident_id: str | None = None
    evidence_ref: str | None = None
    details: str | None = None

    def __post_init__(self):
        self.severity = DEFAULT_EVENT_SEVERITY.get(
            self.event_type, SEV_INFO
        ) if not self.severity else str(self.severity).upper()

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "camera": self.camera,
            "zone": self.zone,
            "employee_id": self.employee_id if self.employee_id else UNKNOWN_ID,
            "confidence": round(float(self.confidence or 0.0), 3),
            "duration_seconds": self.duration_seconds,
            "person_count": self.person_count,
            "status": self.status,
            "evidence_ref": self.evidence_ref,
            "details": self.details,
        }


class EventStore:
    """Thin persistence API over ``security_events`` + ``incidents``."""

    def __init__(self, conn):
        self._conn = conn

    # ------------------------------------------------------------------
    # Inserts
    # ------------------------------------------------------------------

    def record(self, event: SecurityEvent, incident_id: str | None = None) -> int:
        """Persist one event (optionally bound to an incident)."""
        if incident_id:
            event.incident_id = incident_id
        return db.insert_security_event(self._conn, event.to_dict())

    def record_dict(self, ev: dict, incident_id: str | None = None) -> int:
        if incident_id:
            ev["incident_id"] = incident_id
        return db.insert_security_event(self._conn, ev)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def events(self, *, day: str | None = None, event_type: str | None = None,
               camera: str | None = None, zone: str | None = None,
               employee_id: str | None = None, incident_id: str | None = None,
               severity: str | None = None, status: str | None = None,
               limit: int = 500) -> list[dict]:
        return db.query_security_events(
            self._conn, day=day, event_type=event_type, camera=camera,
            zone=zone, employee_id=employee_id, incident_id=incident_id,
            severity=severity, status=status,
            limit=limit,
        )