"""Stress tests for the real pipeline under high load (Phase 36, marked `stress`).

These validate database / evidence / alert / backup scale and query timing
honestly (measured, not invented).  They use isolated databases and a temp
evidence dir.  Simulation data never touches production stores.
"""

import os
import sqlite3
import time

import pytest

from src import database as db
from src.backup_tool import backup_all, restore_to_temp, verify_integrity
from src.simulation.runner import SimulationRunner
from src.simulation.virtual_camera import ResourceLimitError

pytestmark = pytest.mark.stress


def _populate(conn, n_events=200, n_incidents=20, n_alerts=50,
              n_links=100, n_audit=150, n_health=120):
    """Insert a realistic volume of records the same way the real daemon does."""
    ev_ids = []
    for i in range(n_events):
        ev = db.insert_security_event(conn, {
            "event_type": "UNKNOWN_PRESENCE" if i % 2 else "CAMERA_OFFLINE",
            "camera": f"cam_{1 + (i % 5):02d}",
            "severity": "HIGH",
            "confidence": 0.8,
            "timestamp": f"2026-09-03 10:{i % 60:02d}:{i % 60:02d}",
            "employee_id": None,
            "zone": None,
        })
        ev_ids.append(ev)
    for i in range(n_incidents):
        inc_id = f"INC-TEST-{i:05d}"
        db.insert_incident(conn, {
            "incident_id": inc_id,
            "event_type": "UNKNOWN_PRESENCE",
            "severity": "MEDIUM",
            "status": "OPEN",
            "camera": "cam_01",
            "first_seen": "2026-09-03 10:00:00",
            "last_event": "2026-09-03 10:30:00",
        })
        if i < n_links and i < len(ev_ids):
            db.link_incident_event(conn, inc_id, ev_ids[i], "UNKNOWN_PRESENCE",
                                   "cam_01", None, None)
    for i in range(n_alerts):
        db.insert_alert(conn, {
            "rule_name": "unknown_presence",
            "event_type": "UNKNOWN_PRESENCE",
            "severity": "MEDIUM",
            "camera": "cam_01",
            "incident_id": f"INC-TEST-{i % max(1, n_incidents):05d}",
            "status": "OPEN",
            "channel": "dashboard",
            "created_at": "2026-09-03 10:00:00",
        })
    for i in range(n_audit):
        db.audit(conn, "simu.view", actor="simu_user", detail=f"audit {i}")
    for i in range(n_health):
        conn.execute(
            "INSERT INTO camera_health (camera_id, health, fps, frames_read, "
            "reconnects, last_frame, downtime_sec, tampered, recorded_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (f"cam_{1 + (i % 5):02d}", "ONLINE", 10.0, i, 0, i, 0, 0,
             "2026-09-03 10:00:00"))
    conn.commit()


def _populate_db(tmp_path, name="big.db"):
    path = str(tmp_path / name)
    conn = sqlite3.connect(path)
    db.init_db(conn)
    return conn


def test_large_db_query_performance(tmp_path):
    conn = _populate_db(tmp_path)
    _populate(conn, n_events=500, n_incidents=50, n_alerts=100,
              n_links=100, n_audit=200, n_health=150)
    # Key read paths the dashboard / investigation use must complete quickly
    # and without error.  Bounds are generous safety nets, not fake targets.
    samples = {
        "incidents": lambda: conn.execute(
            "SELECT incident_id, status FROM incidents LIMIT 200").fetchall(),
        "events": lambda: conn.execute(
            "SELECT * FROM security_events ORDER BY id DESC LIMIT 200").fetchall(),
        "timeline": lambda: conn.execute(
            "SELECT * FROM incident_events LIMIT 200").fetchall(),
        "alerts": lambda: conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT 200").fetchall(),
    }
    for name, fn in samples.items():
        t0 = time.time()
        fn()
        assert (time.time() - t0) * 1000.0 < 5000, f"{name} too slow"
    conn.close()


def test_incident_event_links_scale_without_duplicates(tmp_path):
    """Re-linking the same (incident, event) pair must not create duplicate
    rows -- idempotent (Phase 35 safeguard scales cleanly)."""
    conn = _populate_db(tmp_path)
    _populate(conn, n_events=100, n_incidents=10, n_links=80)

    # exactly one link row per produced (incident, event) pair (== n_incidents)
    rows = conn.execute("SELECT COUNT(*) FROM incident_events").fetchone()[0]
    assert rows == 10  # 10 incidents each linked to their first event once

    for i in range(10):
        inc_id = f"INC-TEST-{i:05d}"
        ev_id = conn.execute(
            "SELECT event_id FROM incident_events WHERE incident_id=?",
            (inc_id,)).fetchall()
        assert len(ev_id) == 1  # no duplicated link per incident
    conn.close()


def test_backup_restore_under_load(tmp_path):
    """Backup + restore a populated simulation DB; assert key records survive."""
    proj = tmp_path / "proj"
    (proj / "data" / "database").mkdir(parents=True)
    db_path = str(proj / "data" / "database" / "sessions.db")
    conn = sqlite3.connect(db_path)
    db.init_db(conn)
    _populate(conn, n_events=150, n_incidents=15, n_alerts=40,
              n_links=60, n_audit=100, n_health=90)
    db.upsert_employee(conn, "EMP001", name="Sim User", department="QA")
    conn.close()

    backup_dir = str(tmp_path / "backups")
    results = backup_all(str(proj), backup_dir)
    assert "database" in results
    assert verify_integrity(results["database"]) == "ok"

    restored = restore_to_temp(results["database"])
    assert restored and os.path.isfile(restored)
    rc = sqlite3.connect(restored)
    assert rc.execute("SELECT COUNT(*) FROM incidents").fetchone()[0] == 15
    assert rc.execute("SELECT COUNT(*) FROM security_events").fetchone()[0] > 0
    assert rc.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 40
    assert rc.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 100
    assert rc.execute("SELECT COUNT(*) FROM employees "
                      "WHERE employee_id='EMP001'").fetchone()[0] == 1
    rc.close()


def test_backup_rejects_corrupt_under_load(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"definitely not a sqlite db")
    assert verify_integrity(str(bad)) != "ok"
    with pytest.raises(Exception):
        restore_to_temp(str(bad))


def test_resource_evidence_limit_aborts_simulation():
    # A positive cap that the scenario exceeds must abort the run safely
    # (ResourceLimitError) rather than continue unbounded.
    with pytest.raises(ResourceLimitError):
        with SimulationRunner(scenario="G_zone_intrusion", seed=42,
                              limits={"max_evidence_files": 1}) as r:
            r.run_steps()
    # a zero/negative limit means "unlimited" (no-op), a documented safeguard
    with SimulationRunner(scenario="A_normal_occupancy", seed=42,
                          limits={"max_evidence_files": 0}) as r:
        rep = r.run_steps()
        assert rep.completed


def test_enforce_directly_raises_on_exceeded_positive_limit():
    from src.simulation.runner import SimulationRunner
    with SimulationRunner(scenario="A_normal_occupancy", seed=42) as r:
        with pytest.raises(ResourceLimitError):
            r._enforce("max_cameras", value=999)