"""Phase 35 tests -- Production Deployment, Real-World Validation & 24/7 Reliability.

These cover only genuine hardening behaviour added in Phase 35 (no feature-padding):

* WAL-safe backup **restore** (new `restore_database` / `restore_to_temp`) and a
  full backup -> restore -> verify -> table/record presence drill.
* Idempotency hardening: `upsert_incident_risk` keeps a single latest row per
  incident; `link_incident_event` never creates duplicate link rows on re-link.
* Security retention wiring: `security_retention_cleanup` prunes events/alerts/
  audit but never incidents, and never touches productivity data.
* Configuration hardening: out-of-range thresholds, negative evidence retention,
  empty camera URL, and invalid SMTP are all rejected by `validate_config`.
* Production-safety boundary: security retention never removes employees or
  activity logs.
"""

import os
import sqlite3

import pytest

import config as _cfg
from src import database as db
from src.backup_tool import (backup_database, restore_database, restore_to_temp,
                             verify_integrity)


# ----------------------------------------------------------------------
# Backup -> restore drill (WAL-safe, never touches production DB)
# ----------------------------------------------------------------------

def test_restore_database_roundtrip(tmp_db, tmp_path):
    db.upsert_employee(tmp_db, "EMP001", name="Alice", department="Eng")
    db.insert_incident(tmp_db, {"incident_id": "I1", "event_type": "INTRUSION",
                                "severity": "HIGH", "status": "OPEN"})
    db_path = tmp_db.execute("PRAGMA database_list").fetchone()[2]
    backup_dir = str(tmp_path / "backups")
    bk = backup_database(db_path, backup_dir)

    target = str(tmp_path / "restored.db")
    restore_database(bk, target)
    assert verify_integrity(target) == "ok"

    conn = sqlite3.connect(target)
    emp = conn.execute("SELECT COUNT(*) FROM employees WHERE employee_id='EMP001'").fetchone()
    inc = conn.execute("SELECT COUNT(*) FROM incidents WHERE incident_id='I1'").fetchone()
    conn.close()
    assert emp[0] == 1 and inc[0] == 1


def test_restore_to_temp_is_valid(tmp_db, tmp_path):
    db.upsert_employee(tmp_db, "EMP001")
    db_path = tmp_db.execute("PRAGMA database_list").fetchone()[2]
    bk = backup_database(db_path, str(tmp_path / "bks"))
    tmp = restore_to_temp(bk)
    try:
        assert verify_integrity(tmp) == "ok"
        conn = sqlite3.connect(tmp)
        emp = conn.execute("SELECT COUNT(*) FROM employees WHERE employee_id='EMP001'").fetchone()
        conn.close()
        assert emp[0] == 1
    finally:
        os.remove(tmp)


def test_restore_refuses_corrupt_backup(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"garbage")
    with pytest.raises(ValueError):
        restore_database(str(bad), str(tmp_path / "out.db"))


def test_secure_retention_never_touches_employees_or_productivity(tmp_db):
    db.upsert_employee(tmp_db, "EMP001", name="Alice")
    db.log_interval(tmp_db, "2020-01-01 09:00:00", "EMP001", "ACTIVE", 60.0)
    db.insert_security_event(tmp_db, {"event_type": "OLD", "severity": "LOW",
                                      "camera": "c", "zone": "z",
                                      "timestamp": "2020-01-01 09:00:00",
                                      "incident_id": None})
    r = db.security_retention_cleanup(tmp_db, events_days=1000, alerts_days=1000,
                                      audit_days=1000)
    assert r["events"] >= 1
    emp = db.get_employee(tmp_db, "EMP001")
    assert emp["employee_id"] == "EMP001"
    intervals = db.query_day_intervals(tmp_db, "2020-01-01")
    assert intervals and intervals[0][2] == "ACTIVE"


# ----------------------------------------------------------------------
# Idempotency hardening
# ----------------------------------------------------------------------

def test_upsert_incident_risk_idempotent_single_latest_row(tmp_db):
    db.insert_incident(tmp_db, {"incident_id": "I1", "event_type": "INTRUSION",
                                "severity": "HIGH", "status": "OPEN"})
    db.upsert_incident_risk(tmp_db, "I1", 0.5, components="{}", rationale="r1")
    db.upsert_incident_risk(tmp_db, "I1", 0.9, components="{}", rationale="r2")
    rows = tmp_db.execute(
        "SELECT score, rationale FROM incident_risk WHERE incident_id='I1'").fetchall()
    assert len(rows) == 1              # single latest row, not growing
    assert rows[0][0] == 0.9 and rows[0][1] == "r2"


def test_link_incident_event_idempotent(tmp_db):
    db.insert_incident(tmp_db, {"incident_id": "I1", "event_type": "INTRUSION",
                                "severity": "HIGH", "status": "OPEN"})
    ev_id = db.insert_security_event(tmp_db, {"event_type": "INTRUSION",
                                              "severity": "HIGH", "camera": "c",
                                              "zone": "z",
                                              "timestamp": "2026-09-03 10:00:00",
                                              "incident_id": "I1"})
    db.link_incident_event(tmp_db, "I1", ev_id, "INTRUSION", "c", "z")
    db.link_incident_event(tmp_db, "I1", ev_id, "INTRUSION", "c", "z")  # re-link
    rows = tmp_db.execute(
        "SELECT COUNT(*) FROM incident_events WHERE incident_id='I1' AND event_id=?",
        (ev_id,)).fetchone()
    assert rows[0] == 1


# ----------------------------------------------------------------------
# Configuration hardening
# ----------------------------------------------------------------------

def test_config_flags_out_of_range_thresholds(monkeypatch):
    monkeypatch.setattr(_cfg, "CONF_THRESHOLD", 1.5)
    assert any("CONF_THRESHOLD" in p for p in _cfg.validate_config())


def test_config_flags_negative_evidence_retention(monkeypatch):
    monkeypatch.setattr(_cfg, "EVIDENCE_RETENTION_DAYS", -1)
    assert any("EVIDENCE_RETENTION_DAYS" in p for p in _cfg.validate_config())


def test_config_flags_empty_camera_url(monkeypatch):
    monkeypatch.setattr(_cfg, "CAMERAS", {"cam_01": "  "})
    assert any("camera cam_01" in p for p in _cfg.validate_config())


def test_config_flags_invalid_smtp(monkeypatch):
    monkeypatch.setattr(_cfg, "SENDER_EMAIL", "ops@example.com")
    monkeypatch.setattr(_cfg, "SMTP_SERVER", "   ")
    assert any("SMTP_SERVER" in p for p in _cfg.validate_config())


def test_config_clean_for_valid_phase35(monkeypatch):
    # A valid config must not trip the new Phase 35 checks.
    monkeypatch.setattr(_cfg, "CONF_THRESHOLD", 0.4)
    monkeypatch.setattr(_cfg, "FACE_SIMILARITY_THRESHOLD", 0.6)
    monkeypatch.setattr(_cfg, "EVIDENCE_RETENTION_DAYS", 30)
    problems = _cfg.validate_config()
    assert not any("CAMERAS" in p or "RETENTION" in p or "THRESHOLD" in p
                   for p in problems)


# ----------------------------------------------------------------------
# Runtime snapshot still excludes secrets (production-safety)
# ----------------------------------------------------------------------

def test_config_snapshot_never_includes_credentials():
    snap = _cfg.runtime_config_snapshot()
    blob = " ".join(f"{k}={v}" for k, v in snap.items()).lower()
    assert "password" not in blob
    assert "smtp" not in blob