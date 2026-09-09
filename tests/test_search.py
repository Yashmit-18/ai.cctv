"""Tests for the Phase 32 structured security search layer."""

import pytest

from src import database as db
from src.search import SecuritySearch


@pytest.fixture
def seeded(tmp_db):
    db.insert_incident(tmp_db, {
        "incident_id": "INC-1", "event_type": "INTRUSION", "severity": "HIGH",
        "status": "OPEN", "first_seen": "2026-09-03 09:00:00",
        "last_seen": "2026-09-03 09:05:00", "count": 3, "camera": "cam_01",
        "zone": "archive", "employee_id": "Unknown", "confidence": 0.8,
        "notes": "recurring after-hours access",
    })
    db.insert_incident(tmp_db, {
        "incident_id": "INC-2", "event_type": "CAMERA_OFFLINE", "severity": "MEDIUM",
        "status": "RESOLVED", "first_seen": "2026-09-02 10:00:00",
        "last_seen": "2026-09-02 10:01:00", "count": 1, "camera": "cam_02",
        "zone": "lobby", "employee_id": "", "confidence": 1.0, "notes": "",
    })
    return tmp_db


def test_search_incidents_by_status(seeded):
    s = SecuritySearch(seeded)
    out = s.search_incidents(filters={"status": "OPEN"})
    assert [i["incident_id"] for i in out] == ["INC-1"]


def test_search_incidents_by_camera(seeded):
    s = SecuritySearch(seeded)
    out = s.search_incidents(filters={"camera": "cam_02"})
    assert [i["incident_id"] for i in out] == ["INC-2"]


def test_unknown_filter_column_ignored(seeded):
    # An attacker-supplied non-allow-listed key must never reach SQL.
    s = SecuritySearch(seeded)
    out = s.search_incidents(filters={"status": "OPEN",
                                      "evil); DROP TABLE incidents;--": "x"})
    assert len(out) >= 1
    assert out[0]["status"] == "OPEN"


def test_search_incidents_text(seeded):
    s = SecuritySearch(seeded)
    out = s.search_incidents_text("after-hours")
    assert [i["incident_id"] for i in out] == ["INC-1"]


def test_search_events_by_type(seeded):
    db.insert_security_event(seeded, {
        "event_type": "INTRUSION", "severity": "HIGH", "camera": "cam_01",
        "timestamp": "2026-09-03 09:00:00",
    })
    s = SecuritySearch(seeded)
    out = s.search_events(filters={"event_type": "INTRUSION"})
    assert out and out[0]["event_type"] == "INTRUSION"


def test_search_alerts_by_severity(seeded):
    db.insert_alert(seeded, {
        "created_at": "2026-09-03 09:00:00", "severity": "CRITICAL",
        "channel": "log", "message": "boom",
    })
    s = SecuritySearch(seeded)
    out = s.search_alerts(filters={"severity": "CRITICAL"})
    assert out and out[0]["severity"] == "CRITICAL"


def test_incident_timeline(seeded):
    db.link_incident_event(seeded, "INC-1", 1, "INTRUSION", "cam_01", "archive")
    db.add_incident_note(seeded, "INC-1", "analyst", "reviewed, no threat")
    s = SecuritySearch(seeded)
    tl = s.incident_timeline("INC-1")
    assert tl["incident"]["incident_id"] == "INC-1"
    assert "notes" in tl
    assert tl["notes"][0]["note"].startswith("reviewed")


def test_incident_timeline_missing(seeded):
    s = SecuritySearch(seeded)
    assert s.incident_timeline("NOPE") == {}
