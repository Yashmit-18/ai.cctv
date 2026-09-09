"""Phase 31 -- security schema + persistence helper tests.

Validates the additive schema (created in ``init_db``), event persistence,
incident CRUD + dedup, audit log, alert rules, zones and retention.
"""

import sqlite3

import pytest

from src import database as db
from src.domain import (
    CAMERA_OFFLINE,
    INCIDENT_ACKNOWLEDGED,
    INCIDENT_DISMISSED,
    INCIDENT_OPEN,
    INCIDENT_RESOLVED,
    INTRUSION,
    SEV_HIGH,
    SEV_INFO,
    SEV_MEDIUM,
    UNKNOWN_PRESENCE,
)

TABLES = (
    "security_events", "incidents", "alert_rules", "alerts",
    "audit_log", "security_zones", "evidence_files",
)


def _table_names(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def _columns(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


class TestSchema:
    def test_security_tables_exist(self, tmp_db):
        for t in TABLES:
            assert t in _table_names(tmp_db)

    def test_security_events_columns(self, tmp_db):
        cols = _columns(tmp_db, "security_events")
        for c in ("incident_id", "event_type", "severity", "timestamp",
                  "camera", "zone", "employee_id", "person_count",
                  "confidence", "status", "evidence_ref", "details"):
            assert c in cols

    def test_incidents_columns(self, tmp_db):
        cols = _columns(tmp_db, "incidents")
        for c in ("incident_id", "event_type", "severity", "status",
                  "first_seen", "last_seen", "count", "notes", "resolved_at"):
            assert c in cols

    def test_security_indexes_created(self, tmp_db):
        idxs = {r[0] for r in tmp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()}
        for idx in ("idx_sec_type", "idx_sec_date", "idx_inc_status",
                    "idx_zones_camera", "idx_evidence_incident"):
            assert idx in idxs

    def test_activity_schema_untouched(self, tmp_db):
        # Canonical productivity tables must still exist with date/source.
        assert "employees" in _table_names(tmp_db)
        assert "activity_logs" in _table_names(tmp_db)
        cols = _columns(tmp_db, "activity_logs")
        assert {"date", "source"} <= cols

    def test_reinit_is_idempotent(self, tmp_db):
        db.init_db(tmp_db)  # second run must not raise
        assert True


class TestSecurityEvents:
    def test_insert_and_query_roundtrip(self, tmp_db):
        ev = {
            "event_type": UNKNOWN_PRESENCE, "severity": SEV_MEDIUM,
            "timestamp": "2026-09-02 08:15:00", "camera": "CAM1",
            "zone": None, "employee_id": "Unknown", "confidence": 0.81,
            "person_count": 1, "status": INCIDENT_OPEN,
            "incident_id": "INC-20260902-0001", "details": "sample",
        }
        row_id = db.insert_security_event(tmp_db, ev)
        assert row_id > 0
        hits = db.query_security_events(tmp_db, day="2026-09-02")
        assert len(hits) == 1
        assert hits[0]["event_type"] == UNKNOWN_PRESENCE
        assert hits[0]["date"] == "2026-09-02"
        assert hits[0]["employee_id"] == "Unknown"

    def test_query_filters(self, tmp_db):
        for i, cam in enumerate(("CAM1", "CAM2")):
            db.insert_security_event(tmp_db, {
                "event_type": UNKNOWN_PRESENCE, "severity": SEV_MEDIUM,
                "timestamp": f"2026-09-02 09:{i:02d}:00", "camera": cam,
                "employee_id": "Unknown",
            })
        assert len(db.query_security_events(tmp_db, camera="CAM1")) == 1
        assert len(db.query_security_events(tmp_db, event_type=INTRUSION)) == 0
        assert len(db.query_security_events(tmp_db, limit=1)) == 1

    def test_unknown_id_default(self, tmp_db):
        # employee_id=None stores UNKNOWN-like string only when caller maps it;
        # raw helper keeps whatever it is given (None -> None).
        db.insert_security_event(tmp_db, {
            "event_type": UNKNOWN_PRESENCE, "severity": SEV_INFO,
            "timestamp": "2026-09-02 09:00:00", "employee_id": None,
        })
        hits = db.query_security_events(tmp_db)
        assert hits[0]["employee_id"] is None


class TestIncidents:
    def test_insert_and_get(self, tmp_db):
        db.insert_incident(tmp_db, {
            "incident_id": "INC-20260902-0001", "event_type": INTRUSION,
            "severity": SEV_HIGH, "status": INCIDENT_OPEN,
            "first_seen": "2026-09-02 08:00:00", "last_seen": "2026-09-02 08:05:00",
            "count": 3, "camera": "CAM1",
        })
        inc = db.get_incident(tmp_db, "INC-20260902-0001")
        assert inc is not None
        assert inc["severity"] == SEV_HIGH
        assert inc["count"] == 3

    def test_update_incident(self, tmp_db):
        inc_id = "INC-20260902-0002"
        db.insert_incident(tmp_db, {
            "incident_id": inc_id, "event_type": INTRUSION,
            "severity": SEV_HIGH, "status": INCIDENT_OPEN,
            "first_seen": "2026-09-02 08:00:00",
        })
        db.update_incident(tmp_db, inc_id, status=INCIDENT_RESOLVED,
                          notes="reviewed", resolved_at="2026-09-02 09:00:00")
        inc = db.get_incident(tmp_db, inc_id)
        assert inc["status"] == INCIDENT_RESOLVED
        assert inc["notes"] == "reviewed"

    def test_list_incidents_filters(self, tmp_db):
        for i in range(2):
            db.insert_incident(tmp_db, {
                "incident_id": f"INC-20260902-{i+1:04d}",
                "event_type": INTRUSION if i == 0 else CAMERA_OFFLINE,
                "severity": SEV_HIGH, "status": INCIDENT_OPEN,
                "first_seen": f"2026-09-02 0{i}:00:00",
            })
        assert len(db.list_incidents(tmp_db, event_type=INTRUSION)) == 1
        assert len(db.list_incidents(tmp_db, day="2026-09-02")) == 2
        assert len(db.list_incidents(tmp_db, status="OPEN")) == 2

    def test_find_open_incident_dedup(self, tmp_db):
        a = "INC-20260902-0001"
        db.insert_incident(tmp_db, {
            "incident_id": a, "event_type": INTRUSION, "severity": SEV_HIGH,
            "status": INCIDENT_OPEN, "first_seen": "2026-09-02 08:00:00",
            "camera": "CAM1", "zone": "SERVER_ROOM",
        })
        # Exact key match (camera + zone).
        assert db.find_open_incident(tmp_db, INTRUSION, "CAM1", "SERVER_ROOM")["incident_id"] == a
        # Wildcard camera/zone also matches.
        assert db.find_open_incident(tmp_db, INTRUSION, None, None)["incident_id"] == a
        # Different camera => new incident (no match).
        assert db.find_open_incident(tmp_db, INTRUSION, "CAM2", None) is None
        # Resolved incidents no longer match.
        db.update_incident(tmp_db, a, status=INCIDENT_RESOLVED)
        assert db.find_open_incident(tmp_db, INTRUSION, "CAM1", "SERVER_ROOM") is None


class TestAudit:
    def test_audit_append_only(self, tmp_db):
        db.audit(tmp_db, "incident.acknowledged", actor="alice",
                 resource="INC-20260902-0001")
        db.audit(tmp_db, "config.update", actor="system")
        events = db.list_audit_events(tmp_db)
        assert len(events) == 2
        assert events[0]["action"] != events[1]["action"]
        actions = {e["action"] for e in events}
        assert "incident.acknowledged" in actions
        assert "config.update" in actions


class TestAlertRules:
    def test_upsert_and_list(self, tmp_db):
        rule = {
            "rule_name": "intrusion_high", "event_type": INTRUSION,
            "min_severity": "HIGH", "channels": "dashboard,log",
            "cooldown_sec": 600, "enabled": True,
        }
        db.upsert_alert_rule(tmp_db, rule)
        db.upsert_alert_rule(tmp_db, {**rule, "cooldown_sec": 300})
        rules = db.list_alert_rules(tmp_db)
        assert len(rules) == 1  # upsert, not duplicate
        assert rules[0]["cooldown_sec"] == 300

    def test_insert_and_query_alerts(self, tmp_db):
        db.insert_alert(tmp_db, {
            "created_at": "2026-09-02 08:00:00", "rule_name": "intrusion",
            "event_type": INTRUSION, "severity": SEV_HIGH, "camera": "CAM1",
            "incident_id": "INC-20260902-0001", "channel": "log",
            "message": "alert!",
        })
        alerts = db.query_alerts(tmp_db)
        assert len(alerts) == 1
        assert alerts[0]["channel"] == "log"
        assert db.last_alert_time(tmp_db) == "2026-09-02 08:00:00"


class TestZones:
    def test_upsert_list_delete(self, tmp_db):
        db.upsert_zone(tmp_db, {
            "zone_name": "SERVER_ROOM", "camera": "CAM1",
            "polygon": '[[0.2,0.2],[0.8,0.2],[0.8,0.8],[0.2,0.8]]',
            "enabled": 1, "allowed_employees": "E1,E2",
            "alert_policy": "INTRUSION",
        })
        zones = db.list_zones(tmp_db)
        assert len(zones) == 1
        assert "E1" in zones[0]["allowed_employees"]
        db.delete_zone(tmp_db, "SERVER_ROOM")
        assert db.list_zones(tmp_db) == []


class TestEvidence:
    def test_file_tracking(self, tmp_db):
        db.insert_evidence_file(tmp_db, "INC-20260902-0001",
                               r"data\evidence\2026-09-02\INC-20260902-0001\x.jpg",
                               "snapshot", "2026-09-02 08:00:00", 10_000)
        rows = db.list_evidence_files(tmp_db, incident_id="INC-20260902-0001")
        assert len(rows) == 1
        assert rows[0]["size_bytes"] == 10_000
        db.delete_evidence_file(tmp_db, rows[0]["path"])
        assert db.list_evidence_files(tmp_db) == []


class TestRetention:
    def test_security_retention_only_touches_requested_domains(self, tmp_db):
        db.insert_security_event(tmp_db, {
            "event_type": UNKNOWN_PRESENCE, "severity": SEV_INFO,
            "timestamp": "2026-01-01 00:00:00", "employee_id": "Unknown",
        })
        db.insert_security_event(tmp_db, {
            "event_type": UNKNOWN_PRESENCE, "severity": SEV_INFO,
            "timestamp": "2026-09-02 00:00:00", "employee_id": "Unknown",
        })
        # Incidents are NEVER auto-purged.
        db.insert_incident(tmp_db, {
            "incident_id": "INC-20260101-0001", "event_type": INTRUSION,
            "severity": SEV_HIGH, "status": INCIDENT_OPEN,
            "first_seen": "2026-01-01 00:00:00",
        })
        removed = db.security_retention_cleanup(tmp_db, events_days=30)
        assert removed["events"] == 1  # only the January event
        assert db.query_security_events(tmp_db)  # Sept event survives
        assert db.get_incident(tmp_db, "INC-20260101-0001") is not None

    def test_negative_days_means_untouched(self, tmp_db):
        db.insert_security_event(tmp_db, {
            "event_type": UNKNOWN_PRESENCE, "severity": SEV_INFO,
            "timestamp": "2026-01-01 00:00:00", "employee_id": "Unknown",
        })
        removed = db.security_retention_cleanup(tmp_db)
        assert removed == {"events": 0, "alerts": 0, "audit": 0}
        assert len(db.query_security_events(tmp_db)) == 1