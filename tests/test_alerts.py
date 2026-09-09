"""Tests for the alert engine."""

import pytest

from src import database as db
from src.alerts import AlertEngine
from src.domain import SEV_HIGH, SEV_MEDIUM


@pytest.fixture
def alert_conn(tmp_db):
    # seed a couple of alert rules
    db.upsert_alert_rule(tmp_db, {
        "rule_name": "unknown_high",
        "event_type": "UNKNOWN_PRESENCE",
        "min_severity": "MEDIUM",
        "channels": "log,dashboard",
        "cooldown_sec": 0,
        "enabled": True,
    })
    db.upsert_alert_rule(tmp_db, {
        "rule_name": "offline_high",
        "event_type": "CAMERA_OFFLINE",
        "min_severity": "HIGH",
        "channels": "log",
        "cooldown_sec": 0,
        "enabled": True,
    })
    db.upsert_alert_rule(tmp_db, {
        "rule_name": "disabled_rule",
        "event_type": "UNKNOWN_PRESENCE",
        "min_severity": "LOW",
        "channels": "log",
        "cooldown_sec": 0,
        "enabled": False,
    })
    return tmp_db


def test_match_by_severity(alert_conn):
    engine = AlertEngine(alert_conn)
    ev = {
        "event_type": "UNKNOWN_PRESENCE",
        "severity": SEV_MEDIUM,
        "camera": "cam_01",
        "employee_id": "Unknown",
    }
    dispatched = engine.evaluate(ev)
    types = {d["channel"] for d in dispatched}
    assert "log" in types
    assert "dashboard" in types


def test_below_min_severity_not_dispatched(alert_conn):
    engine = AlertEngine(alert_conn)
    ev = {
        "event_type": "CAMERA_OFFLINE",
        "severity": SEV_MEDIUM,  # below rule's HIGH
        "camera": "cam_01",
    }
    dispatched = engine.evaluate(ev)
    assert dispatched == []


def test_disabled_rule_not_fired(alert_conn):
    engine = AlertEngine(alert_conn)
    ev = {
        "event_type": "UNKNOWN_PRESENCE",
        "severity": SEV_MEDIUM,
        "camera": "cam_02",
    }
    dispatched = engine.evaluate(ev)
    assert all(d["rule_name"] != "disabled_rule" for d in dispatched)


def test_cooldown_suppresses_repeats(tmp_db):
    db.upsert_alert_rule(tmp_db, {
        "rule_name": "camera_offline",
        "event_type": "CAMERA_OFFLINE",
        "min_severity": "HIGH",
        "channels": "log",
        "cooldown_sec": 60,
        "enabled": True,
    })
    engine = AlertEngine(tmp_db)
    ev = {
        "event_type": "CAMERA_OFFLINE",
        "severity": SEV_HIGH,
        "camera": "cam_01",
    }
    first = engine.evaluate(ev)
    assert len(first) == 1  # fired
    # immediate second call within cooldown -> suppressed
    second = engine.evaluate(ev)
    assert second == []


def test_alert_persisted(tmp_db):
    db.upsert_alert_rule(tmp_db, {
        "rule_name": "unknown_high",
        "event_type": "UNKNOWN_PRESENCE",
        "min_severity": "MEDIUM",
        "channels": "log",
        "cooldown_sec": 0,
        "enabled": True,
    })
    engine = AlertEngine(tmp_db)
    engine.evaluate({"event_type": "UNKNOWN_PRESENCE", "severity": SEV_MEDIUM,
                     "camera": "cam_01"})
    alerts = db.query_alerts(tmp_db)
    assert len(alerts) == 1
    assert alerts[0]["rule_name"] == "unknown_high"
    assert alerts[0]["channel"] == "log"


def test_unknown_event_type_not_matched(tmp_db):
    db.upsert_alert_rule(tmp_db, {
        "rule_name": "only_unknown",
        "event_type": "UNKNOWN_PRESENCE",
        "min_severity": "LOW",
        "channels": "log",
        "cooldown_sec": 0,
        "enabled": True,
    })
    engine = AlertEngine(tmp_db)
    dispatched = engine.evaluate({"event_type": "INTRUSION", "severity": SEV_HIGH,
                                  "camera": "cam_01"})
    assert dispatched == []


# ----------------------------------------------------------------------
# Phase 32: lifecycle, retry, escalation
# ----------------------------------------------------------------------

def test_delivered_status_recorded(alert_conn):
    engine = AlertEngine(alert_conn)
    dispatched = engine.evaluate({"event_type": "UNKNOWN_PRESENCE",
                                  "severity": SEV_MEDIUM, "camera": "cam_01"})
    assert any(d["status"] == "DELIVERED" for d in dispatched)
    # persisted too
    rows = db.query_alerts(alert_conn)
    assert all(r["status"] == "DELIVERED" for r in rows)


def test_email_failure_sets_failed_and_retries(tmp_db):
    db.upsert_alert_rule(tmp_db, {
        "rule_name": "email_only",
        "event_type": "UNKNOWN_PRESENCE",
        "min_severity": "LOW",
        "channels": "email",  # not a real connected channel -> honest failure
        "cooldown_sec": 0,
        "enabled": True,
    })
    engine = AlertEngine(tmp_db, max_retries=2, retry_delay_sec=0)
    engine.evaluate({"event_type": "UNKNOWN_PRESENCE", "severity": SEV_MEDIUM,
                     "camera": "cam_01"})
    row = db.query_alerts(tmp_db)[0]
    assert row["status"] == "FAILED"
    assert row["attempts"] == 1
    # retry escalates attempts until max_retries then stops
    engine.retry_pending()
    assert db.query_alerts(tmp_db)[0]["attempts"] == 2
    engine.retry_pending()
    assert db.query_alerts(tmp_db)[0]["attempts"] == 3
    engine.retry_pending()  # exhausted -> no further attempts
    assert db.query_alerts(tmp_db)[0]["attempts"] == 3


def test_acknowledge_and_resolve(tmp_db):
    db.upsert_alert_rule(tmp_db, {
        "rule_name": "u",
        "event_type": "UNKNOWN_PRESENCE",
        "min_severity": "LOW",
        "channels": "log",
        "cooldown_sec": 0,
        "enabled": True,
    })
    engine = AlertEngine(tmp_db)
    engine.evaluate({"event_type": "UNKNOWN_PRESENCE", "severity": SEV_MEDIUM,
                     "camera": "cam_01"})
    row = db.query_alerts(tmp_db)[0]
    engine.acknowledge(row["id"])
    assert db.query_alerts(tmp_db)[0]["status"] == "ACKNOWLEDGED"
    engine.resolve(row["id"])
    assert db.query_alerts(tmp_db)[0]["status"] == "RESOLVED"


def test_escalate_stale_skips_acknowledged(tmp_db):
    db.upsert_alert_rule(tmp_db, {
        "rule_name": "u",
        "event_type": "UNKNOWN_PRESENCE",
        "min_severity": "LOW",
        "channels": "log",
        "cooldown_sec": 0,
        "enabled": True,
    })
    engine = AlertEngine(tmp_db, escalation_sec=0)
    engine.evaluate({"event_type": "UNKNOWN_PRESENCE", "severity": SEV_HIGH,
                     "camera": "cam_01"})
    row = db.query_alerts(tmp_db)[0]
    engine.acknowledge(row["id"])
    escalated = engine.escalate_stale()
    assert escalated == []  # acknowledged alerts are never escalated
    assert db.query_alerts(tmp_db)[0]["status"] == "ACKNOWLEDGED"


def test_escalate_stale_high_severity(tmp_db):
    db.upsert_alert_rule(tmp_db, {
        "rule_name": "u",
        "event_type": "UNKNOWN_PRESENCE",
        "min_severity": "LOW",
        "channels": "log",
        "cooldown_sec": 0,
        "enabled": True,
    })
    engine = AlertEngine(tmp_db, escalation_sec=0)
    engine.evaluate({"event_type": "UNKNOWN_PRESENCE", "severity": SEV_HIGH,
                     "camera": "cam_01"})
    escalated = engine.escalate_stale()
    assert len(escalated) == 1
    assert db.query_alerts(tmp_db)[0]["status"] == "ESCALATED"
    assert db.query_alerts(tmp_db)[0]["escalated"] == 1


def test_escalate_skips_low_severity(tmp_db):
    db.upsert_alert_rule(tmp_db, {
        "rule_name": "u",
        "event_type": "UNKNOWN_PRESENCE",
        "min_severity": "LOW",
        "channels": "log",
        "cooldown_sec": 0,
        "enabled": True,
    })
    engine = AlertEngine(tmp_db, escalation_sec=0)
    engine.evaluate({"event_type": "UNKNOWN_PRESENCE", "severity": SEV_MEDIUM,
                     "camera": "cam_01"})
    escalated = engine.escalate_stale()
    assert escalated == []
