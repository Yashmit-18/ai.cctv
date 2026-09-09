"""Tests for Phase 32 security analytics."""

from datetime import date, datetime, timedelta

from src import database as db
from src.security_analytics import SecurityAnalytics


def _seed_incident(conn, inc_id, etype, sev, status, first_seen, zone):
    db.insert_incident(conn, {
        "incident_id": inc_id, "event_type": etype, "severity": sev,
        "status": status, "first_seen": first_seen,
        "last_seen": first_seen, "count": 1, "camera": "cam",
        "zone": zone, "employee_id": "Unknown", "confidence": 0.5,
        "notes": "",
    })


def test_incident_by_status(tmp_db):
    _seed_incident(tmp_db, "I1", "INTRUSION", "HIGH", "OPEN",
                   "2026-09-03 09:00:00", "archive")
    _seed_incident(tmp_db, "I2", "INTRUSION", "HIGH", "RESOLVED",
                   "2026-09-03 10:00:00", "archive")
    a = SecurityAnalytics(tmp_db)
    by = {r["key"]: r["count"] for r in a.incident_by_status()}
    assert by["OPEN"] == 1
    assert by["RESOLVED"] == 1


def test_hotspots_by_zone(tmp_db):
    _seed_incident(tmp_db, "I1", "INTRUSION", "HIGH", "OPEN",
                   "2026-09-03 09:00:00", "archive")
    _seed_incident(tmp_db, "I2", "LOITERING", "MEDIUM", "OPEN",
                   "2026-09-03 10:00:00", "lobby")
    _seed_incident(tmp_db, "I3", "INTRUSION", "HIGH", "OPEN",
                   "2026-09-03 11:00:00", "archive")
    a = SecurityAnalytics(tmp_db)
    hot = {r["key"]: r["count"] for r in a.hotspots_by_zone()}
    assert hot["archive"] == 2
    assert hot["lobby"] == 1


def test_incidents_by_day(tmp_db):
    # Seed relative to "today" so the incident stays inside any n_days window.
    today = date.today()
    _seed_incident(tmp_db, "I1", "INTRUSION", "HIGH", "OPEN",
                   today.strftime("%Y-%m-%d") + " 09:00:00", "archive")
    a = SecurityAnalytics(tmp_db)
    days = a.incidents_by_day(n_days=3)
    assert any(d["day"] == today.strftime("%Y-%m-%d") and d["count"] == 1 for d in days)


def test_alert_funnel(tmp_db):
    db.insert_alert(tmp_db, {"created_at": "2026-09-03 09:00:00",
                             "severity": "HIGH", "status": "OPEN_ABSENT",
                             "channel": "log", "message": "x"})
    db.insert_alert(tmp_db, {"created_at": "2026-09-03 09:01:00",
                             "severity": "HIGH", "status": "RESOLVED",
                             "channel": "log", "message": "y"})
    a = SecurityAnalytics(tmp_db)
    funnel = {r["key"]: r["count"] for r in a.alert_funnel()}
    assert funnel.get("OPEN_ABSENT") == 1


def test_snapshot_shape(tmp_db):
    _seed_incident(tmp_db, "I1", "INTRUSION", "HIGH", "OPEN",
                   "2026-09-03 09:00:00", "archive")
    a = SecurityAnalytics(tmp_db)
    s = a.snapshot()
    assert s["incidents"]["open"] == 1
    assert "by_status" in s["incidents"]
    assert "hotspots_by_zone" in s
    assert "evidence" in s
