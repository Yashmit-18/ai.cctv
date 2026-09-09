"""Phase 31 -- incident engine lifecycle + grouping tests."""

import re

import pytest

from src.domain import (
    INCIDENT_ACKNOWLEDGED,
    INCIDENT_DISMISSED,
    INCIDENT_OPEN,
    INCIDENT_RESOLVED,
    INTRUSION,
    SEV_HIGH,
    SEV_INFO,
    UNKNOWN_PRESENCE,
)
from src.incidents import IncidentEngine
from src.security_events import (
    SecurityEvent,
    severity_ge,
    severity_to_index,
)

INC_ID_RE = re.compile(r"^INC-\d{8}-\d{4}$")


def _ev(event_type, camera=None, zone=None, severity=None, ts="2026-09-02 08:00:00",
        employee_id="Unknown"):
    return {
        "event_type": event_type, "timestamp": ts, "camera": camera,
        "zone": zone, "employee_id": employee_id,
        "severity": severity,
    }


class TestSeverity:
    def test_order(self):
        assert severity_to_index("INFO") == 0
        assert severity_to_index("CRITICAL") == 4
        assert severity_to_index("bogus") == 0

    def test_ge(self):
        assert severity_ge("HIGH", "MEDIUM")
        assert not severity_ge("LOW", "MEDIUM")


class TestBind:
    def test_bind_creates_incident(self, tmp_db):
        engine = IncidentEngine(tmp_db)
        ev = _ev(UNKNOWN_PRESENCE, camera="CAM1")
        inc_id = engine.bind(ev)
        assert INC_ID_RE.match(inc_id)
        assert ev["incident_id"] == inc_id
        inc = engine.get(inc_id)
        assert inc["status"] == INCIDENT_OPEN
        assert inc["severity"] == "MEDIUM"  # DEFAULT_EVENT_SEVERITY

    def test_bind_dedups_same_key(self, tmp_db):
        engine = IncidentEngine(tmp_db)
        a = engine.bind(_ev(INTRUSION, camera="CAM1", zone="SERVER_ROOM",
                            severity=SEV_HIGH, ts="2026-09-02 08:00:00"))
        b = engine.bind(_ev(INTRUSION, camera="CAM1", zone="SERVER_ROOM",
                            severity=SEV_HIGH, ts="2026-09-02 08:05:00"))
        assert a == b
        inc = engine.get(a)
        assert inc["count"] == 2
        assert inc["last_seen"] == "2026-09-02 08:05:00"
        # Only one row + two events linked.
        assert len(engine.list()) == 1

    def test_bind_distinct_camera_new_incident(self, tmp_db):
        engine = IncidentEngine(tmp_db)
        a = engine.bind(_ev(INTRUSION, camera="CAM1", zone="Z"))
        b = engine.bind(_ev(INTRUSION, camera="CAM2", zone="Z"))
        assert a != b

    def test_bind_after_resolve_new_incident(self, tmp_db):
        engine = IncidentEngine(tmp_db)
        a = engine.bind(_ev(INTRUSION, camera="CAM1", zone="Z"))
        engine.resolve(a)
        b = engine.bind(_ev(INTRUSION, camera="CAM1", zone="Z"))
        assert a != b
        assert engine.get(a)["status"] == INCIDENT_RESOLVED
        assert engine.get(b)["status"] == INCIDENT_OPEN


class TestLifecycle:
    def test_ack_resolve_dismiss(self, tmp_db):
        engine = IncidentEngine(tmp_db)
        inc_id = engine.bind(_ev(UNKNOWN_PRESENCE, camera="CAM1"))
        assert engine.acknowledge(inc_id, actor="alice") is True
        assert engine.get(inc_id)["status"] == INCIDENT_ACKNOWLEDGED
        assert engine.resolve(inc_id, actor="alice", notes="clear-fp") is True
        assert engine.get(inc_id)["status"] == INCIDENT_RESOLVED
        assert engine.get(inc_id)["notes"] == "clear-fp"
        # Dismissing a resolved incident is refused (lifecycle safety).
        assert engine.dismiss(inc_id) is False

    def test_dismiss_records_notes(self, tmp_db):
        engine = IncidentEngine(tmp_db)
        inc_id = engine.bind(_ev(INTRUSION, camera="CAM1"))
        assert engine.dismiss(inc_id, actor="bob", notes="false positive") is True
        inc = engine.get(inc_id)
        assert inc["status"] == INCIDENT_DISMISSED
        assert inc["notes"] == "false positive"
        assert inc["resolved_at"] is not None

    def test_audit_events_written(self, tmp_db):
        engine = IncidentEngine(tmp_db)
        inc_id = engine.bind(_ev(UNKNOWN_PRESENCE, camera="CAM1"))
        engine.acknowledge(inc_id, actor="alice")
        from src.database import list_audit_events
        actions = [e["action"] for e in list_audit_events(tmp_db)]
        assert "incident.acknowledged" in actions

    def test_id_uniqueness_across_days(self, tmp_db):
        engine = IncidentEngine(tmp_db)
        a = engine.bind(_ev(INTRUSION, ts="2026-09-01 08:00:00"))
        b = engine.bind(_ev(INTRUSION, ts="2026-09-02 08:00:00"))
        assert a.startswith("INC-20260901-")
        assert b.startswith("INC-20260902-")
        assert a != b


class TestQueries:
    def test_open_counts(self, tmp_db):
        engine = IncidentEngine(tmp_db)
        engine.bind(_ev(INTRUSION, camera="CAM1", severity=SEV_HIGH))
        engine.bind(_ev(UNKNOWN_PRESENCE, camera="CAM2"))
        assert engine.open_count() == 2
        assert engine.high_severity_open_count() == 1
        assert engine.last_incident_time() is not None

    def test_list_filters(self, tmp_db):
        engine = IncidentEngine(tmp_db)
        engine.bind(_ev(INTRUSION, camera="CAM1", zone="Z"))
        engine.bind(_ev(UNKNOWN_PRESENCE, camera="CAM2"))
        assert len(engine.list(status=INCIDENT_OPEN)) == 2
        assert len(engine.list(event_type=INTRUSION)) == 1
        assert len(engine.list(camera="CAM2")) == 1
        assert len(engine.list(day="2026-09-02")) == 2


class TestSecurityEvent:
    def test_dataclass_defaults(self):
        ev = SecurityEvent(event_type=UNKNOWN_PRESENCE)
        assert ev.severity == "MEDIUM"  # default map
        assert ev.status == INCIDENT_OPEN

    def test_to_dict_unknown_fallback(self):
        ev = SecurityEvent(event_type=UNKNOWN_PRESENCE, employee_id=None)
        d = ev.to_dict()
        assert d["employee_id"] == "Unknown"

    def test_to_dict_cleans_confidence(self):
        ev = SecurityEvent(event_type=INTRUSION, confidence=0.87654)
        assert ev.to_dict()["confidence"] == 0.877

    def test_info_event_default(self):
        ev = SecurityEvent(event_type="CAMERA_RECOVERED")
        assert ev.severity == SEV_INFO