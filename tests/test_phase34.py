"""Phase 34 tests -- Next-Generation AI Office Intelligence.

Covers the high-value, hardware-independent additions:

* Incident reconstruction (temporal pre/post window, never-fabricated
  ``PRE_EVENT_EVIDENCE_UNAVAILABLE``).
* Person/identity continuity (first/last seen, dwell, zone transitions,
  cross-camera path; Unknown stays UNRESOLVED).
* Advanced event context rollup (``CORRELATED_SECURITY_INCIDENT`` /
  ``REQUIRES_HUMAN_REVIEW`` -- never an accusation).
* Adaptive time-of-day baselines (hour/day-of-week, INSUFFICIENT_DATA guard).
* Extended explainable incident risk context factors (zone/after-hours/
  evidence/confidence) -- incident-only, bounded.
* Safety: investigation layer is read-only and never touches productivity.
"""

import os

import pytest

import config
from src import database as db
from src.investigation import (
    IncidentReconstruction, PRE_EVENT_EVIDENCE_UNAVAILABLE,
    PersonContinuity, EventContextRollup, CORRELATED_SECURITY_INCIDENT,
)
from src.anomaly_engine import BaselineTracker, AnomalyEngine, STATUS_INSUFFICIENT
from src.risk_scoring import IncidentRiskScorer


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _mk_incident(conn, incident_id="INC-1", event_type="UNKNOWN_PRESENCE",
                 severity="MEDIUM", status="OPEN", camera="cam_01", zone="Z1",
                 employee_id=None, occurrences=1, confidence=0.7,
                 first_seen="2026-09-03 10:00:00"):
    db.insert_incident(conn, {
        "incident_id": incident_id, "event_type": event_type,
        "severity": severity, "status": status,
        "first_seen": first_seen, "last_seen": first_seen, "count": occurrences,
        "camera": camera, "zone": zone, "employee_id": employee_id,
        "confidence": confidence,
    })
    db.update_incident(conn, incident_id, occurrences=occurrences)
    return incident_id


def _mk_event(conn, incident_id, event_type, ts, camera="cam_01", zone="Z1",
              employee_id=None, severity="MEDIUM"):
    ev_id = db.insert_security_event(conn, {
        "incident_id": incident_id, "event_type": event_type,
        "severity": severity, "timestamp": ts, "camera": camera,
        "zone": zone, "employee_id": employee_id,
    })
    db.link_incident_event(conn, incident_id, ev_id, event_type, camera, zone)


def _mk_evidence(conn, incident_id, captured_at, camera="cam_01", path=None,
                 kind="snapshot"):
    db.insert_evidence_file(
        conn, incident_id, path or f"{incident_id}.jpg",
        kind=kind, captured_at=captured_at, size_bytes=2048, camera=camera)


# ----------------------------------------------------------------------
# P0 -- Incident reconstruction
# ----------------------------------------------------------------------

def test_reconstruction_builds_window(tmp_db):
    _mk_incident(tmp_db, first_seen="2026-09-03 10:00:00")
    _mk_event(tmp_db, "INC-1", "UNKNOWN_PRESENCE", "2026-09-03 10:00:00")
    _mk_event(tmp_db, "INC-1", "LOITERING_SUSPECTED", "2026-09-03 10:00:20")
    _mk_evidence(tmp_db, "INC-1", "2026-09-03 09:59:40")
    rec = IncidentReconstruction(tmp_db).reconstruct("INC-1", pre_sec=30,
                                                     post_sec=30)
    assert rec["found"] is True
    assert rec["window_rows"]
    assert any(r["kind"] == "EVIDENCE" for r in rec["window_rows"])


def test_reconstruction_pre_event_unavailable(tmp_db):
    _mk_incident(tmp_db, first_seen="2026-09-03 10:00:00")
    _mk_event(tmp_db, "INC-1", "UNKNOWN_PRESENCE", "2026-09-03 10:00:00")
    rec = IncidentReconstruction(tmp_db).reconstruct("INC-1", pre_sec=30)
    assert rec["pre_event_evidence"] == PRE_EVENT_EVIDENCE_UNAVAILABLE


def test_reconstruction_pre_event_present(tmp_db):
    _mk_incident(tmp_db, first_seen="2026-09-03 10:00:00")
    _mk_event(tmp_db, "INC-1", "UNKNOWN_PRESENCE", "2026-09-03 10:00:00")
    _mk_evidence(tmp_db, "INC-1", "2026-09-03 09:59:45")  # within pre-30s
    rec = IncidentReconstruction(tmp_db).reconstruct("INC-1", pre_sec=30)
    assert rec["pre_event_evidence"] == "EVIDENCE_PRESENT"


# ----------------------------------------------------------------------
# P0 -- Person continuity (never fabricates identity)
# ----------------------------------------------------------------------

def test_person_continuity_resolved_and_dwell(tmp_db):
    _mk_incident(tmp_db, event_type="INTRUSION", employee_id="EMP001",
                 first_seen="2026-09-03 09:00:00")
    _mk_event(tmp_db, "INC-1", "INTRUSION", "2026-09-03 09:00:00",
              camera="cam_1", zone="Z1", employee_id="EMP001")
    _mk_event(tmp_db, "INC-1", "LOITERING_SUSPECTED", "2026-09-03 09:05:00",
              camera="cam_2", zone="Z2", employee_id="EMP001")
    pc = PersonContinuity(tmp_db).resolve("EMP001")
    assert pc["status"] == "RESOLVED_TO_EMPLOYEE"
    assert pc["first_seen"] == "2026-09-03 09:00:00"
    assert pc["last_seen"] == "2026-09-03 09:05:00"
    assert pc["dwell_seconds"] == 300.0
    assert pc["zone_transitions"] == ["Z1", "Z2"]
    assert pc["camera_path"] == ["cam_1", "cam_2"]


def test_person_continuity_unknown_stays_unresolved(tmp_db):
    _mk_incident(tmp_db, event_type="UNKNOWN_PRESENCE", employee_id=None,
                 first_seen="2026-09-03 09:00:00")
    from src.domain import UNKNOWN_ID
    _mk_event(tmp_db, "INC-1", "UNKNOWN_PRESENCE", "2026-09-03 09:00:00",
              camera="cam_1", zone="Z1", employee_id=UNKNOWN_ID)
    pc = PersonContinuity(tmp_db).resolve(UNKNOWN_ID)
    assert pc["unresolved"] is True
    assert pc["status"] == "UNRESOLVED"


# ----------------------------------------------------------------------
# P0 -- Advanced event context rollup
# ----------------------------------------------------------------------

def test_event_context_correlated_per_ngu(tmp_db):
    _mk_incident(tmp_db, event_type="UNKNOWN_PRESENCE", zone="Z1")
    _mk_event(tmp_db, "INC-1", "UNKNOWN_PRESENCE", "2026-09-03 10:00:00", zone="Z1")
    _mk_event(tmp_db, "INC-1", "LOITERING_SUSPECTED", "2026-09-03 10:01:00", zone="Z1")
    _mk_event(tmp_db, "INC-1", "INTRUSION", "2026-09-03 10:02:00", zone="Z1")
    _mk_event(tmp_db, "INC-1", "AFTER_HOURS_ACTIVITY", "2026-09-03 10:03:00", zone="Z1")
    ctx = EventContextRollup(tmp_db).rollup("INC-1")
    assert ctx["is_correlated_security_incident"] is True
    assert ctx["label"].startswith(CORRELATED_SECURITY_INCIDENT)
    assert "HUMAN_REVIEW" in ctx["label"]
    assert "criminal" not in ctx["label"].lower()
    assert ctx["requires_human_review"] is True


def test_event_context_single_observation(tmp_db):
    _mk_incident(tmp_db, event_type="UNKNOWN_PRESENCE", zone="Z1")
    _mk_event(tmp_db, "INC-1", "UNKNOWN_PRESENCE", "2026-09-03 10:00:00", zone="Z1")
    ctx = EventContextRollup(tmp_db).rollup("INC-1")
    assert ctx["is_correlated_security_incident"] is False


# ----------------------------------------------------------------------
# P1 -- Adaptive time-of-day baselines
# ----------------------------------------------------------------------

def test_time_baseline_insufficient_guard(tmp_db):
    ae = AnomalyEngine(tmp_db)
    # only a couple of samples -> INSUFFICIENT_DATA marker, no spike claim
    for _ in range(2):
        bt = BaselineTracker(tmp_db, min_samples=5)
        bt.record_occupancy_time("Z1", "cam_01", 3, "2026-09-03 10:00:00")
    res = ae.evaluate_occupancy_time("Z1", "cam_01", 40, "2026-09-03 10:00:05")
    assert res is not None
    assert res["status"] == STATUS_INSUFFICIENT
    assert res["anomaly_type"] == "TIME_OCCUPANCY_INSUFFICIENT"


def test_time_baseline_anomaly_after_ready(tmp_db):
    ae = AnomalyEngine(tmp_db)
    bt = BaselineTracker(tmp_db, min_samples=5)
    ae._baselines = bt
    # Build a stable baseline around ~3 occupancy at the same hour.
    for val in (3, 2, 3, 4, 3, 2):
        bt.record_occupancy_time("Z1", "cam_01", val, "2026-09-03 10:00:00")
    # A big spike in the same hour now raises a TIME_OCCUPANCY_ANOMALY.
    res = ae.evaluate_occupancy_time("Z1", "cam_01", 40, "2026-09-03 10:30:00")
    assert res is not None
    assert res["anomaly_type"] == "TIME_OCCUPANCY_ANOMALY"


# ----------------------------------------------------------------------
# P1 -- Extended explainable risk context factors (incident-only)
# ----------------------------------------------------------------------

# Ensure the extension is gated OFF in the safety tests via monkeypatch to keep
# the core score deterministic, and ON to verify the bonus factors.

def test_risk_context_factors_present_and_bounded(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "RISK_CONTEXT_FACTORS", True)
    _mk_incident(tmp_db, severity="HIGH", zone="Z1")
    # Zone Z1 is NOT configured as restricted in this db (no security_zones row)
    # so zone_sensitive stays 0; after_hours False; no evidence -> all 0.
    rs = IncidentRiskScorer(tmp_db)
    res = rs.score("INC-1")
    assert 0.0 <= res["score"] <= 1.0
    assert "corroboration_types" in res["components"]
    assert "zone_sensitive" in res["components"]
    assert "rationale" in res


def test_risk_context_factors_zone_afterhours_evidence(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "RISK_CONTEXT_FACTORS", True)
    db.upsert_zone(tmp_db, {"zone_name": "Z1", "camera": "cam_01",
                            "polygon": "[]", "alert_policy": "INTRUSION"})
    _mk_incident(tmp_db, severity="HIGH", zone="Z1")
    _mk_event(tmp_db, "INC-1", "AFTER_HOURS_ACTIVITY", "2026-09-03 22:00:00",
              zone="Z1")
    _mk_evidence(tmp_db, "INC-1", "2026-09-03 22:00:00")
    rs = IncidentRiskScorer(tmp_db)
    res = rs.score("INC-1")
    assert res["components"]["zone_sensitive"] > 0.0
    assert res["components"]["after_hours"] > 0.0
    assert res["components"]["evidence"] > 0.0
    assert 0.0 <= res["score"] <= 1.0
    assert "zone_sensitive" in res["rationale"]


def test_risk_gated_off_no_context_bonus(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "RISK_CONTEXT_FACTORS", False)
    db.upsert_zone(tmp_db, {"zone_name": "Z1", "camera": "cam_01",
                            "polygon": "[]", "alert_policy": "INTRUSION"})
    _mk_incident(tmp_db, severity="HIGH", zone="Z1")
    _mk_evidence(tmp_db, "INC-1", "2026-09-03 10:00:00")
    rs = IncidentRiskScorer(tmp_db)
    res = rs.score("INC-1")
    assert res["components"]["zone_sensitive"] == 0.0
    assert res["components"]["evidence"] == 0.0


# ----------------------------------------------------------------------
# C -- Config snapshot includes Phase 34 tunables
# ----------------------------------------------------------------------

def test_config_snapshot_has_phase34(tmp_db):
    snap = config.runtime_config_snapshot()
    assert "risk_context_factors" in snap


# ----------------------------------------------------------------------
# Safety -- investigation is read-only & never touches productivity
# ----------------------------------------------------------------------

def test_investigation_never_mutates_or_touches_productivity(tmp_db):
    db.upsert_employee(tmp_db, "EMP001", name="A", department="D")
    db.log_interval(tmp_db, "2026-09-03 09:00:00", "EMP001", "ACTIVE", 60.0)
    _mk_incident(tmp_db, event_type="INTRUSION", employee_id="EMP001")
    before_inc = db.get_incident(tmp_db, "INC-1")
    IncidentReconstruction(tmp_db).reconstruct("INC-1")
    PersonContinuity(tmp_db).resolve("EMP001")
    EventContextRollup(tmp_db).rollup("INC-1")
    after_inc = db.get_incident(tmp_db, "INC-1")
    assert after_inc == before_inc  # read-only: incident untouched
    emp = db.get_employee(tmp_db, "EMP001")
    assert emp["employee_id"] == "EMP001"
    intervals = db.query_day_intervals(tmp_db, "2026-09-03")
    assert intervals and intervals[0][2] == "ACTIVE"
