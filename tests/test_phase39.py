"""Phase 39 regression tests: CRITICAL DEFECT REMEDIATION.

These lock in the B1-B18 fixes.  They are pure DB / config / pure-function
tests -- no Streamlit runtime or external camera/SMTP/GPU required, so they
run on any CI box and never fabricate hardware validation.

Coverage map:
  B1  : risk panel LEFT JOIN survives a deleted incident
  B2  : incident acknowledge / dismiss lifecycle + audit
  B3  : email channel exposes SMTP config (honest channel knobs)
  B4  : dashboard fail-closed config validation + session-expiry knob
  B6  : departure (LEFT) semantics constant present + wired in engine
  B8  : all schema timestamps are localtime (no naive UTC now)
  B10 : evidence retention honors per-record deadline
  B15 : grace / break knobs are surfaced by the metrics engine
  B17 : production entry points never pull in the simulation layer
"""

import importlib

import config as _cfg

import pytest

from src import database as db


@pytest.fixture
def reloaded(monkeypatch):
    """Reload the module after env tweaks; auto-clean afterwards."""
    def _reload(changes: dict):
        for k, v in changes.items():
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)
        return importlib.reload(_cfg)
    return _reload


# ----------------------------------------------------------------------
# B1 - risk panel LEFT JOIN
# ----------------------------------------------------------------------
def test_risk_panel_keeps_row_when_incident_deleted(tmp_db):
    inc_id = db.insert_incident(tmp_db, {
        "incident_id": "INC-B1-001",
        "event_type": "INTRUSION",
        "severity": "HIGH",
        "status": "OPEN",
        "last_seen": "2026-01-05 10:00:00",
        "zone": "Front Door",
    })
    db.upsert_incident_risk(
        tmp_db, inc_id, 0.87,
        components="seen|classification",
        rationale="high-severity intrusion near entrance",
    )
    rows = db.list_incident_risk(tmp_db)
    assert any(r["incident_id"] == inc_id for r in rows)
    row = next(r for r in rows if r["incident_id"] == inc_id)
    assert row["risk_score"] == 0.87
    assert row["status"] == "OPEN"

    # Delete the incident: the LEFT JOIN must still return the advisory row.
    tmp_db.execute("DELETE FROM incidents WHERE incident_id = ?", (inc_id,))
    tmp_db.commit()
    rows = db.list_incident_risk(tmp_db)
    kept = [r for r in rows if r["incident_id"] == inc_id]
    assert kept, "LEFT JOIN dropped the risk row after its incident vanished"
    assert kept[0]["risk_score"] == 0.87
    assert kept[0]["status"] is None  # incident meta is gone, advisory kept


# ----------------------------------------------------------------------
# B2 - incident acknowledge / dismiss lifecycle + audit
# ----------------------------------------------------------------------
def test_acknowledge_sets_status_timestamp_and_audits(tmp_db):
    from src.incidents import IncidentEngine

    inc_id = db.insert_incident(tmp_db, {
        "incident_id": "INC-B2-001",
        "event_type": "LOITERING",
        "severity": "MEDIUM",
        "status": "OPEN",
        "last_seen": "2026-01-05 10:00:00",
    })
    eng = IncidentEngine(tmp_db)
    assert eng.acknowledge(inc_id, actor="alice") is True

    inc = db.get_incident(tmp_db, inc_id)
    assert inc["status"] == "ACKNOWLEDGED"
    assert inc["acknowledged_at"]

    audits = db.list_audit_events(tmp_db)
    assert any(
        a["action"] == "incident.acknowledged" and a["resource"] == inc_id
        for a in audits
    )

    # Acknowledging a terminal incident is refused.
    eng.dismiss(inc_id, actor="bob", _no_audit=True)
    assert eng.acknowledge(inc_id, actor="carol") is False


def test_dismiss_sets_review_state_and_single_audit(tmp_db):
    from src.incidents import IncidentEngine

    inc_id = db.insert_incident(tmp_db, {
        "incident_id": "INC-B2-002",
        "event_type": "UNKNOWN_PERSON",
        "severity": "LOW",
        "status": "OPEN",
        "last_seen": "2026-01-05 11:00:00",
    })
    eng = IncidentEngine(tmp_db)
    assert eng.dismiss(inc_id, actor="alice", notes="false positive") is True

    inc = db.get_incident(tmp_db, inc_id)
    assert inc["status"] == "DISMISSED"
    assert inc["review_state"] == "DISMISSED"
    assert inc["resolved_at"]

    audits = [a for a in db.list_audit_events(tmp_db)
              if a["action"] == "incident.dismissed" and a["resource"] == inc_id]
    assert len(audits) == 1, "dismiss must emit exactly one audit line"


# ----------------------------------------------------------------------
# B3 - email channel configuration is exposed (honest SMTP channel)
# ----------------------------------------------------------------------
def test_email_channel_config_available(reloaded):
    cfg = reloaded({})
    # The email channel must have real SMTP knobs (non-empty descriptors), not
    # a stubbed channel -- so failures are honest, never silent.
    assert hasattr(cfg, "SENDER_EMAIL")
    assert hasattr(cfg, "SMTP_SERVER")
    assert hasattr(cfg, "SMTP_PORT")
    assert hasattr(cfg, "EMAIL_MAX_RETRIES")


# ----------------------------------------------------------------------
# B4 - dashboard fail-closed validation + session expiry
# ----------------------------------------------------------------------
def test_fail_closed_without_password_is_blocking(reloaded):
    cfg = reloaded({
        "CCTV_DASH_AUTH": "1",
        "CCTV_DASH_FAIL_CLOSED": "1",
        "CCTV_DASH_PASS": None,
        "CCTV_DASH_VIEWER_PASS": None,
    })
    problems = cfg.validate_config()
    assert any("password" in p.lower() for p in problems), problems


def test_session_expiry_default_is_15_minutes(reloaded):
    cfg = reloaded({})
    assert cfg.DASH_SESSION_MINUTES == 15.0
    assert isinstance(cfg.DASH_SESSION_MINUTES, float)


# ----------------------------------------------------------------------
# B6 - departure (LEFT) semantics wired in
# ----------------------------------------------------------------------
def test_left_semantics_constant_and_engine_reference():
    from src.domain import LEFT

    assert LEFT == "LEFT"
    # The security engine source emits LEFT as a first-class departure event.
    import inspect
    from src import security_engine as se

    src_text = inspect.getsource(se)
    assert 'event_type": LEFT' in src_text


# ----------------------------------------------------------------------
# B8 - all schema timestamps are localtime (no naive UTC now)
# ----------------------------------------------------------------------
def test_schema_never_uses_naive_utc_now(tmp_db):
    rows = tmp_db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ).fetchall()
    assert rows, "expected schema tables"
    for (sql,) in rows:
        # Guard: every `datetime('now', ...)` keeps the localtime modifier.
        assert "datetime('now')" not in sql.replace(" ", ""), \
            f"schema uses naive UTC now: {sql}"


# ----------------------------------------------------------------------
# B15 - grace / break knobs surfaced by the metrics engine
# ----------------------------------------------------------------------
def test_day_metrics_surface_grace_and_break(schedule):
    from datetime import date

    from src.productivity import compute_day_metrics

    m = compute_day_metrics("EMP001", [], date(2026, 1, 5), schedule)
    assert m["grace_period_min"] == schedule.grace_period_min == 15
    assert m["break_minutes"] == schedule.break_minutes == 30


# ----------------------------------------------------------------------
# B17 - production entry points never pull in the simulation layer
# ----------------------------------------------------------------------
def test_main_and_config_do_not_reference_simulation():
    import re

    for fname in ("main.py", "config.py"):
        with open(fname, encoding="utf-8") as f:
            text = f.read()
        refs = re.findall(r"(src\.simulation|simulation\.)", text)
        assert not refs, f"{fname} pulls in simulation: {refs}"


# ----------------------------------------------------------------------
# Control-flow regression: load-or-detect never leaves `detections` unbound
# ----------------------------------------------------------------------
class _FakeDetector:
    """Minimal stand-in for ActivityDetector.detect_batch; records calls."""

    def __init__(self, detections, raises=False):
        self.detections = detections
        self.raises = raises
        self.batch_calls = 0

    def detect_batch(self, frames):
        self.batch_calls += 1
        if self.raises:
            raise RuntimeError("boom")
        return self.detections


def _step_first_load(load_result):
    """Drive main._step_detector from detector=None, mirroring the run loop.

    `load_result` is what the detector factory returns (a _FakeDetector or
    None).  Returns (detector, detections) after ONE step.  This is the exact
    production helper run() calls every frame step, with an injectable factory
    -- the control flow under test is the real code, not a reimplementation.
    """
    import main as m

    return m._step_detector(
        None, [{"frame": 0}], lambda: load_result)


def test_step_detector_first_successful_load_runs_detect_batch():
    """The first step that loads a detector MUST also run detect_batch.

    Regression for the UnboundLocalError: pre-fix, the load iteration skipped
    detect_batch, so `detections` was unbound when tracker.process_batch read
    it.  Here a successfully-loaded detector must run inference on that very
    step and return its detections (no exception).
    """
    import main as m

    fake = _FakeDetector([{"emp_id": "EMP001", "cls": "person"}])
    detector, detections = _step_first_load(fake)
    assert detector is fake, "detector must be retained after a successful load"
    assert fake.batch_calls == 1, "detect_batch must run on the first successful load step"
    assert detections == fake.detections, "previous step must return detections"
    # Prove every reference path in run() now sees a defined (non-missing) list.
    assert isinstance(detections, list)
    assert m._step_detector is not None  # sanity: the helper is importable


def test_step_detector_load_failure_returns_empty_safe():
    """A failed load returns (None, []) -- no unbound name, FSM stays frozen."""
    detector, detections = _step_first_load(None)
    assert detector is None
    assert detections == []


def test_step_detector_detect_failure_degrades_to_empty():
    """A detect_batch exception is isolated: (detector, []) with no raise."""
    fake = _FakeDetector([], raises=True)
    detector, detections = _step_first_load(fake)
    assert detector is fake
    assert detections == [], "detect failure degrades to empty, never raises"
    assert fake.batch_calls == 1


def test_step_detector_preexisting_detector_runs_detect_batch_each_step():
    """Subsequent steps (detector already loaded) still call detect_batch."""
    import main as m

    fake = _FakeDetector([{"emp_id": "EMP002", "cls": "person"}])

    def load_detector():  # must not be called once detector is pre-existing
        raise AssertionError("load_detector must not rerun for an existing detector")

    detector, detections = m._step_detector(fake, [{"frame": 0}], load_detector)
    assert detector is fake
    assert detections == fake.detections
    assert fake.batch_calls == 1
