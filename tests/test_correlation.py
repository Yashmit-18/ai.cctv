"""Tests for the Phase 32 event correlation engine."""

from datetime import datetime, timedelta

import pytest

from src import database as db
from src.correlation import EventCorrelator
from src.incidents import IncidentEngine
from src.security_engine import SecurityEngine


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _make_event(**kw):
    ev = {
        "event_type": "UNKNOWN_PRESENCE",
        "severity": "MEDIUM",
        "timestamp": _ts(datetime.now()),
        "camera": "cam_01",
        "zone": None,
        "employee_id": "Unknown",
        "confidence": 0.8,
    }
    ev.update(kw)
    return ev


def _through_engine(engine, event_dict) -> dict:
    """Persist + bind an event through the security engine's _process_event,
    returning the processed event with incident_id set."""
    return engine._process_event(dict(event_dict))


def test_correlator_is_integrated_into_engine(tmp_db):
    engine = SecurityEngine(tmp_db)
    assert engine._correlator is not None


def test_unknown_then_intrusion_correlate_into_one_incident(tmp_db):
    engine = SecurityEngine(tmp_db)
    zone = engine._zones
    zone.add(zone_name="server_room", camera="cam_01",
             polygon="[[0,0],[1,0],[1,1],[0,1]]",
             allowed=[], policy="INTRUSION")

    # 1) Unknown presence event (person-threat family)
    ev1 = _through_engine(engine, _make_event(
        event_type="UNKNOWN_PRESENCE", camera="cam_01", employee_id="Unknown"))
    inc1 = engine._incidents.get(ev1["incident_id"])
    assert inc1["event_type"] == "UNKNOWN_PRESENCE"

    # 2) Intrusion on the same camera shortly after -> should correlate
    ev2 = _through_engine(engine, _make_event(
        event_type="INTRUSION", camera="cam_01", zone="server_room",
        employee_id="Unknown"))
    assert ev2["incident_id"] == ev1["incident_id"], (
        "Intrusion should join the Unknown incident (same camera + identity)")

    # The incident now aggregates both event types via incident_events links.
    links = db.list_incident_events(tmp_db, ev2["incident_id"])
    linked_types = {l["event_type"] for l in links}
    assert "UNKNOWN_PRESENCE" in linked_types
    assert "INTRUSION" in linked_types

    # Underlying events are preserved (never deleted).
    evs = db.query_security_events(tmp_db, incident_id=ev2["incident_id"], limit=50)
    ev_types = {e["event_type"] for e in evs}
    assert "UNKNOWN_PRESENCE" in ev_types and "INTRUSION" in ev_types


def test_does_not_correlate_across_identities(tmp_db):
    engine = SecurityEngine(tmp_db)
    _through_engine(engine, _make_event(
        event_type="UNKNOWN_PRESENCE", camera="cam_01", employee_id="Unknown"))
    ev = _through_engine(engine, _make_event(
        event_type="INTRUSION", camera="cam_01", zone="server_room",
        employee_id="EMP001"))
    # Different identity (Unknown vs EMP001) -> must NOT correlate to Unknown's incident.
    linked = [e["event_type"] for e in
              db.query_security_events(tmp_db, incident_id=ev["incident_id"], limit=50)]
    assert "UNKNOWN_PRESENCE" not in linked


def test_camera_offline_recovered_lifecycle_correlates(tmp_db):
    engine = SecurityEngine(tmp_db)
    now = datetime.now()
    ev1 = _through_engine(engine, _make_event(
        event_type="CAMERA_OFFLINE", camera="cam_01",
        timestamp=_ts(now - timedelta(seconds=10))))
    ev2 = _through_engine(engine, _make_event(
        event_type="CAMERA_RECOVERED", camera="cam_01",
        timestamp=_ts(now - timedelta(seconds=5))))
    # Recovered joins the offline lifecycle incident on the same camera.
    assert ev2["incident_id"] == ev1["incident_id"]


def test_correlation_reason_is_explainable(tmp_db):
    engine = SecurityEngine(tmp_db)
    ev1 = _through_engine(engine, _make_event(
        event_type="UNKNOWN_PRESENCE", camera="cam_01", employee_id="Unknown"))
    ev2 = _through_engine(engine, _make_event(
        event_type="INTRUSION", camera="cam_01", zone="server_room",
        employee_id="Unknown"))
    links = db.list_incident_events(tmp_db, ev2["incident_id"])
    intrusion_link = [l for l in links if l["event_type"] == "INTRUSION"]
    assert intrusion_link and intrusion_link[0]["correlation_reason"]
    assert "UNKNOWN_PRESENCE" in intrusion_link[0]["correlation_reason"]


def test_correlator_never_creates_event_without_source_preserved(tmp_db):
    engine = SecurityEngine(tmp_db)
    ev = _through_engine(engine, _make_event(event_type="UNKNOWN_PRESENCE"))
    # Exactly one matching source event row must exist (not duplicated/deleted).
    evs = db.query_security_events(
        tmp_db, event_type="UNKNOWN_PRESENCE", camera="cam_01", limit=50)
    assert len(evs) == 1
    assert evs[0]["incident_id"] == ev["incident_id"]
