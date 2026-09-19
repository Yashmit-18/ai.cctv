"""Central incident engine (Phase 31).

Every :class:`SecurityEvent` is either attached to an existing **open**
incident (grouping/deduplication by event type + camera + zone) or raises a
new incident with a stable human-readable id (``INC-YYYYMMDD-NNNN``).

Incident lifecycle is explicit and audit-trackable: ``OPEN -> ACKNOWLEDGED ->
RESOLVED`` or ``DISMISSED`` (false positive).  Workers treat an incident as
``REQUIRES REVIEW`` until a human acknowledges/resolves/dismisses it.

Employees and ``activity_logs`` remain the canonical productivity source; an
incident never alters employee state or productivity.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime

from src import database as db
from src.domain import DEFAULT_EVENT_SEVERITY, INCIDENT_OPEN
from src.security_events import severity_to_index

logger = logging.getLogger("cctv.security_events")


def _bump_correlation_ids(correlation_ids: str | None, event_id: int) -> str:
    """Append ``event_id`` to the incident's correlation_ids CSV string."""
    current = [x for x in (correlation_ids or "").split(",") if x]
    if str(event_id) not in current:
        current.append(str(event_id))
    return ",".join(current)


class IncidentEngine:
    """Creates/updates incidents from security events."""

    def __init__(self, conn):
        self._conn = conn
        self._lock = threading.Lock()
        self._seq = 0
        self._day = None

    # ------------------------------------------------------------------
    # Event -> incident binding
    # ------------------------------------------------------------------

    def bind(self, event_dict: dict) -> str:
        """Attach ``event_dict`` to an incident.  Returns the incident id.

        If an OPEN/ACKNOWLEDGED incident matches the (event_type, camera,
        zone) dedup key it is reused (count += 1, last_seen updated);
        otherwise a new incident is created.  The event dict's
        ``incident_id`` is set in place.
        """
        with self._lock:
            ev_type = event_dict.get("event_type")
            camera = event_dict.get("camera")
            zone = event_dict.get("zone")
            ev_day = (event_dict.get("timestamp") or "")[:10]
            existing = db.find_open_incident(self._conn, ev_type, camera, zone)
            if existing and (ev_day and existing.get("first_seen", "")[:10] == ev_day):
                inc_id = existing["incident_id"]
                db.update_incident(
                    self._conn, inc_id,
                    last_seen=event_dict.get("timestamp"),
                    count=int(existing.get("count") or 1) + 1,
                    occurrences=int(existing.get("occurrences") or 1) + 1,
                    confidence=event_dict.get("confidence"),
                )
                if not existing.get("evidence_ref"):
                    if event_dict.get("evidence_ref"):
                        db.update_incident(self._conn, inc_id,
                                           evidence_ref=event_dict["evidence_ref"])
                # Accelerate an open HIGH toward CRITICAL never happens: the
                # incident severity is the first-seen severity (stable).
            else:
                inc_id = self._new_id(event_dict.get("timestamp"))
                severity = event_dict.get("severity") or DEFAULT_EVENT_SEVERITY.get(
                    ev_type, "INFO")
                db.insert_incident(self._conn, {
                    "incident_id": inc_id,
                    "event_type": ev_type,
                    "severity": severity,
                    "status": INCIDENT_OPEN,
                    "first_seen": event_dict.get("timestamp"),
                    "last_seen": event_dict.get("timestamp"),
                    "count": 1,
                    "camera": camera,
                    "zone": zone,
                    "employee_id": event_dict.get("employee_id"),
                    "confidence": event_dict.get("confidence"),
                    "evidence_ref": event_dict.get("evidence_ref"),
                })
            event_dict["incident_id"] = inc_id
            return inc_id

    def attach(self, event_dict: dict, event_id: int,
               correlation_reason: str | None = None) -> str:
        """Attach a *correlated* event to a specific existing incident.

        Unlike :meth:`bind`, ``attach`` never creates a new incident.  It links
        the event to ``event_dict["incident_id"]`` (which the caller resolved
        via the correlator), records the correlation reason, and updates the
        incident's aggregated camera/zone/employee scope.  The underlying event
        is always preserved in ``security_events`` and linked via
        ``incident_events`` -- correlation never deletes source events.
        """
        with self._lock:
            inc_id = event_dict["incident_id"]
            inc = db.get_incident(self._conn, inc_id)
            if inc is None:
                # Fall back to a fresh bind if the target vanished.
                return self.bind(event_dict)
            db.update_incident(
                self._conn, inc_id,
                last_seen=event_dict.get("timestamp"),
                count=int(inc.get("count") or 1) + 1,
                occurrences=int(inc.get("occurrences") or 1) + 1,
                confidence=event_dict.get("confidence"),
                correlation_ids=_bump_correlation_ids(inc.get("correlation_ids"),
                                                      event_id),
            )
            if correlation_reason:
                db.set_incident_correlation(self._conn, inc_id, correlation_reason)
            logger.info(
                "[CORRELATION] event=%s attached to incident=%s reason=%s",
                event_id, inc_id, correlation_reason,
            )
            return inc_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    #: Phase 54 M05 -- advisory lifecycle events that *self-heal* their own
    #: condition.  When such an event fires for an incident that is itself only
    #: INFO severity (never HIGH/CRITICAL), the incident is auto-resolved with
    #: a full audit trail instead of accumulating forever as OPEN.
    ADVISORY_SELF_CLOSING_EVENTS = {"CAMERA_RECOVERED"}

    def auto_close_advisory(self, event_dict: dict,
                            incident_id: str | None = None,
                            actor: str = "system") -> bool:
        """Phase 54 M05 -- auto-close a matched INFO advisory incident.

        Called right after an event is bound/attached to an incident.  Only a
        :data:`ADVISORY_SELF_CLOSING_EVENTS` event type can trigger it, and it
        only ever resolves an incident whose own severity is ``INFO`` -- a
        HIGH/CRITICAL incident (e.g. a CAMERA_OFFLINE that a CAMERA_RECOVERED
        correlated onto) is left OPEN for human review.  Every auto-close
        writes the same ``incident.resolved`` audit record as a manual resolve,
        so the lifecycle stays explicit and trackable.
        """
        ev_type = event_dict.get("event_type")
        if ev_type not in self.ADVISORY_SELF_CLOSING_EVENTS:
            return False
        inc_id = incident_id or event_dict.get("incident_id")
        if not inc_id:
            return False
        inc = db.get_incident(self._conn, inc_id)
        if not inc or inc["status"] in ("RESOLVED", "DISMISSED"):
            return False
        if (inc.get("severity") or "INFO") != "INFO":
            return False
        ts = event_dict.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        notes = f"auto-closed by {ev_type} at {ts}"
        ok = self.resolve(inc_id, actor=actor, notes=notes)
        if ok:
            logger.info(
                "[AUTO-CLOSE] incident=%s closed by %s (INFO advisory "
                "self-healing lifecycle); no human review required.",
                inc_id, ev_type)
        return ok

    def acknowledge(self, incident_id: str, actor: str = "system",
                    *, _no_audit: bool = False) -> bool:
        inc = db.get_incident(self._conn, incident_id)
        if not inc or inc["status"] in ("RESOLVED", "DISMISSED"):
            return False
        db.update_incident(
            self._conn, incident_id,
            status="ACKNOWLEDGED",
            acknowledged_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        if not _no_audit:
            db.audit(self._conn, "incident.acknowledged", actor=actor,
                     resource=incident_id, detail=inc["event_type"])
        return True

    def resolve(self, incident_id: str, actor: str = "system",
                notes: str | None = None, *, _no_audit: bool = False) -> bool:
        inc = db.get_incident(self._conn, incident_id)
        if not inc or inc["status"] in ("RESOLVED", "DISMISSED"):
            return False
        db.update_incident(
            self._conn, incident_id,
            status="RESOLVED", notes=notes or inc.get("notes") or "",
            resolved_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        if not _no_audit:
            db.audit(self._conn, "incident.resolved", actor=actor,
                     resource=incident_id, detail=inc["event_type"])
        return True

    def dismiss(self, incident_id: str, actor: str = "system",
                notes: str | None = None, *, _no_audit: bool = False) -> bool:
        inc = db.get_incident(self._conn, incident_id)
        if not inc or inc["status"] in ("RESOLVED", "DISMISSED"):
            return False
        db.update_incident(
            self._conn, incident_id,
            status="DISMISSED", notes=notes or inc.get("notes") or "",
            resolved_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            review_state="DISMISSED",
        )
        if not _no_audit:
            db.audit(self._conn, "incident.dismissed", actor=actor,
                     resource=incident_id, detail=inc["event_type"])
        return True

    def set_notes(self, incident_id: str, notes: str, actor: str = "system") -> None:
        inc = db.get_incident(self._conn, incident_id)
        if inc:
            db.update_incident(self._conn, incident_id, notes=notes)
            db.audit(self._conn, "incident.notes", actor=actor,
                     resource=incident_id, detail=notes[:200])

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, incident_id: str) -> dict | None:
        return db.get_incident(self._conn, incident_id)

    def list(self, *, status: str | None = None, event_type: str | None = None,
             camera: str | None = None, zone: str | None = None,
             employee_id: str | None = None, day: str | None = None,
             limit: int = 500) -> list[dict]:
        return db.list_incidents(
            self._conn, status=status, event_type=event_type, camera=camera,
            zone=zone, employee_id=employee_id, day=day, limit=limit,
        )

    def open_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE status IN ('OPEN','ACKNOWLEDGED')"
        ).fetchone()
        return int(row[0]) if row else 0

    def high_severity_open_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM incidents "
            "WHERE status IN ('OPEN','ACKNOWLEDGED') AND severity IN ('HIGH','CRITICAL')"
        ).fetchone()
        return int(row[0]) if row else 0

    def last_incident_time(self) -> str | None:
        row = self._conn.execute("SELECT MAX(first_seen) FROM incidents").fetchone()
        return str(row[0]) if row and row[0] else None

    # ------------------------------------------------------------------
    # Id generation
    # ------------------------------------------------------------------

    def _new_id(self, timestamp: str | None = None) -> str:
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        day = ts[:10].replace("-", "")
        prefix = f"INC-{day}"
        if day != getattr(self, "_day", None):
            self._seq = 0
            self._day = day
        while True:
            self._seq += 1
            cand = f"{prefix}-{self._seq:04d}"
            if db.get_incident(self._conn, cand) is None:
                return cand