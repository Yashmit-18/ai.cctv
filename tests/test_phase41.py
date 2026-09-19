"""Phase 41 regression tests -- wall-clock state machine refinement.

Wall-clock rules implemented in ``src/tracker.py``:

* AWAY commits when the employee's *own* last presence is more than
  ``AWAY_AFTER_SEC`` (default 3s) in the past; 0..3s grace keeps the
  previous state visible.
* ON_PHONE commits after `>` ``PHONE_AFTER_SEC`` (default 5s) of continuous
  phone detection; missed-frame gaps shorter than ``PHONE_GAP_GRACE_SEC`` do
  not end an episode; exactly ONE PHONE_USE security event per episode.
* Camera offline freezes the FSM and never advances AWAY time.
* Per-employee independence: one employee leaving never drags another AWAY,
  and Unknown/unmatched detections never create employee productivity
  records.

No model loading, no network, deterministic fake clock.
"""

import time as _realtime

import pytest

import src.tracker as T
from src.database import query_security_events
from src.tracker import MultiTracker


class FakeTime:
    """Deterministic wall clock shared by the whole tracker module."""
    now = 10_000.0

    @classmethod
    def time(cls) -> float:
        return cls.now

    @classmethod
    def strftime(cls, fmt, tt=None):
        return _realtime.strftime(fmt, tt or _realtime.localtime(cls.now))

    @classmethod
    def localtime(cls, tt=None):
        return _realtime.localtime(tt or cls.now)


@pytest.fixture
def fake_clock(monkeypatch):
    FakeTime.now = 10_000.0
    monkeypatch.setattr(T, "time", FakeTime)
    return FakeTime


@pytest.fixture
def recorder(monkeypatch):
    """Capture interval insertions instead of hitting a real DB."""
    seen = []

    def fake_log(conn, ts, emp, state, dur, source=""):
        seen.append((ts, emp, state, dur, source))

    monkeypatch.setattr(T, "log_interval", fake_log)
    return seen


@pytest.fixture
def instant_commit(monkeypatch):
    """Deterministic thresholds; smoothing minimal so wall-clock drives it."""
    monkeypatch.setattr(T, "SMOOTHING_BUFFER_SEC", 0.0)
    monkeypatch.setattr(T, "IDENTITY_STABILITY_FRAMES", 1)
    monkeypatch.setattr(T, "AWAY_AFTER_SEC", 3.0)
    monkeypatch.setattr(T, "PHONE_AFTER_SEC", 5.0)
    monkeypatch.setattr(T, "PHONE_GAP_GRACE_SEC", 2.0)


def _present() -> dict:
    return {"present": True, "phone_detected": False}


def _phone() -> dict:
    return {"present": True, "phone_detected": True}


def _absent() -> dict:
    return {"present": False, "phone_detected": False}


# ---------------------------------------------------------------- AWAY threshold

def test_absent_under_grace_is_not_away(fake_clock, recorder, instant_commit):
    """2.9s absence (<3s) must keep ACTIVE -- no AWAY flicker."""
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())                 # last_seen = t0
    FakeTime.now += 2.9
    t.process(_absent())
    assert t.committed_state == "ACTIVE"
    assert t.current_state == "ACTIVE"
    assert recorder == []


def test_absent_over_threshold_commits_away(fake_clock, recorder, instant_commit):
    """3.1s absence (>3s) commits AWAY with the ACTIVE interval logged."""
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 3.1
    t.process(_absent())
    assert t.committed_state == "AWAY"
    assert t.current_state == "AWAY"
    assert recorder and recorder[-1][1] == "EMP001"
    assert recorder[-1][2] == "ACTIVE"
    assert 2.9 <= recorder[-1][3] <= 3.3


def test_return_after_away_recovers_active(fake_clock, recorder, instant_commit):
    """A returning employee goes straight back to ACTIVE."""
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 3.1
    t.process(_absent())                  # first AWAY frame
    assert t.committed_state == "AWAY"
    FakeTime.now += 1
    t.process(_present())                 # return: candidate -> ACTIVE
    FakeTime.now += 0.01
    t.process(_present())                 # confirming frame -> commit ACTIVE
    assert t.committed_state == "ACTIVE"
    assert t.current_state == "ACTIVE"
    assert recorder[-1][2] == "AWAY"      # previous interval was AWAY


def test_camera_offline_never_advances_away(fake_clock, recorder, instant_commit):
    """Camera offline >3s freezes the FSM; absence restarts after recovery."""
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())                 # ACTIVE @ t0
    FakeTime.now += 60
    t.process(_absent(), camera_online=False)
    t.process(_absent(), camera_online=False)
    assert t.committed_state == "ACTIVE"
    assert t.current_state == "ACTIVE"
    assert recorder == []                 # no offline AWAY time fabricated
    # Recovery: offline time must NOT count toward AWAY (last_seen reset).
    FakeTime.now += 1
    t.process(_absent(), camera_online=True)
    assert t.committed_state == "ACTIVE"
    assert recorder == []
    # Real absence resumes only after valid frames come back.
    FakeTime.now += 3.1
    t.process(_absent(), camera_online=True)
    assert t.committed_state == "AWAY"


# ---------------------------------------------------------------- ON_PHONE threshold

def test_phone_under_threshold_is_not_on_phone(fake_clock, recorder, instant_commit):
    """4.9s of continuous phone use (<5s) must stay ACTIVE-safe."""
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())                 # ACTIVE
    FakeTime.now += 1
    t.process(_phone())                   # phone episode starts
    FakeTime.now += 3.9
    t.process(_phone())                   # 3.9s < 5s
    assert t.committed_state == "ACTIVE"
    assert t.current_state == "ACTIVE"
    assert recorder == []


def test_phone_over_threshold_commits_on_phone(fake_clock, recorder, instant_commit):
    """Continuous phone >5s commits ON_PHONE."""
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    FakeTime.now += 5.0
    t.process(_phone())
    assert t.committed_state == "ON_PHONE"
    assert t.current_state == "ON_PHONE"
    assert recorder[-1][2] == "ACTIVE"    # before -> after


def test_phone_gap_under_grace_keeps_episode(fake_clock, recorder, instant_commit):
    """A <2s detection gap inside an episode does not restart the timer."""
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    FakeTime.now += 1.9
    t.process(_phone())                   # 1.9s gap, cumulative 1.9
    FakeTime.now += 0.1
    t.process(_phone())                   # cumulative 2.0
    FakeTime.now += 3.1
    t.process(_phone())                   # cumulative 5.1 -> ON_PHONE
    assert t.committed_state == "ON_PHONE"


def test_phone_gap_over_grace_restarts_episode(fake_clock, recorder, instant_commit):
    """A >2s detection gap ends the episode; the ruler starts over."""
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())                   # episode starts
    FakeTime.now += 4
    t.process(_phone())                   # gap between frames >2s vs last phone
    assert t.committed_state == "ACTIVE"  # run closed before threshold hit
    FakeTime.now += 5.0
    t.process(_phone())                   # fresh episode >=5s -> ON_PHONE
    assert t.committed_state == "ON_PHONE"


# ---------------------------------------------------------------- phone alerts

def test_phone_alert_emitted_exactly_once(tmp_db, fake_clock, instant_commit):
    """Crossing the threshold persists exactly ONE PHONE_USE event."""
    t = T.EmployeeTracker(tmp_db, "EMP001", source="cam1")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    FakeTime.now += 5.0
    t.process(_phone())                   # threshold crossed -> ON_PHONE
    events = query_security_events(tmp_db, event_type="PHONE_USE")
    assert len(events) == 1
    ev = events[0]
    assert ev["employee_id"] == "EMP001"
    assert ev["camera"] == "cam1"
    assert ev["severity"] == "INFO"
    assert ev["status"] == "NEW"
    assert ev["duration_seconds"] >= 5.0


def test_phone_alert_never_repeats_for_same_episode(tmp_db, fake_clock, instant_commit):
    """Continued phone use after the alert does not emit more events."""
    t = T.EmployeeTracker(tmp_db, "EMP001", source="cam1")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    FakeTime.now += 5.0
    t.process(_phone())                   # ON_PHONE + 1 event
    FakeTime.now += 60
    for _ in range(20):
        t.process(_phone())               # phone still in use
        FakeTime.now += 1
    assert query_security_events(tmp_db, event_type="PHONE_USE").__len__() == 1


def test_phone_stops_then_new_episode_re_arms_alert(tmp_db, fake_clock, instant_commit):
    """Recovery to ACTIVE closes the episode; a later one alerts again."""
    t = T.EmployeeTracker(tmp_db, "EMP001", source="cam1")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())
    FakeTime.now += 5.0
    t.process(_phone())                   # episode 1 -> ON_PHONE + 1 event
    assert t.committed_state == "ON_PHONE"
    FakeTime.now += 0.5
    t.process(_present())                 # phone put down -> candidate ACTIVE
    FakeTime.now += 0.01
    t.process(_present())                 # confirming frame -> commit ACTIVE
    assert t.committed_state == "ACTIVE"
    FakeTime.now += 1.0
    t.process(_phone())                   # episode 2 starts
    FakeTime.now += 5.0
    t.process(_phone())                   # -> ON_PHONE + 2nd event
    assert t.committed_state == "ON_PHONE"
    events = query_security_events(tmp_db, event_type="PHONE_USE")
    assert len(events) == 2


# ---------------------------------------------------------------- per-employee independence

def test_two_employees_two_independent_states(tmp_db, fake_clock, instant_commit):
    """EMP001 ACTIVE while EMP002 ON_PHONE, fully independent."""
    t = MultiTracker(tmp_db)
    dets = [
        {"cam": "c1", "emp_id": "EMP001", "phone": False},
        {"cam": "c1", "emp_id": "EMP002", "phone": True},
    ]
    for _ in range(6):
        t.process_batch(dets, True)
        FakeTime.now += 1
    states = t.live_states()
    assert states.get("EMP001") == "ACTIVE"
    assert states.get("EMP002") == "ON_PHONE"


def test_one_leaving_drags_no_one_else(tmp_db, fake_clock, instant_commit):
    """EMP002 leaves (stops appearing) -> EMP002 AWAY, EMP001 stays ACTIVE.

    Also proves a phone episode dies into AWAY when the person leaves --
    an employee is never ON_PHONE forever.
    """
    t = MultiTracker(tmp_db)
    dets_both = [
        {"cam": "c1", "emp_id": "EMP001", "phone": False},
        {"cam": "c1", "emp_id": "EMP002", "phone": True},
    ]
    for _ in range(6):
        t.process_batch(dets_both, True)
        FakeTime.now += 1
    assert t.live_states().get("EMP002") == "ON_PHONE"

    dets_only_emp1 = [{"cam": "c1", "emp_id": "EMP001", "phone": False}]
    for _ in range(8):
        t.process_batch(dets_only_emp1, True)
        FakeTime.now += 1
    states = t.live_states()
    assert states.get("EMP001") == "ACTIVE"
    assert states.get("EMP002") == "AWAY"


def test_unknown_detections_never_create_employee_records(tmp_db, fake_clock):
    """Unknown boxes feed presence but never spawn an employee tracker."""
    t = MultiTracker(tmp_db)
    dets_unknown = [{"cam": "c1", "emp_id": "Unknown", "phone": False,
                     "person_present": True}]
    for _ in range(8):
        t.process_batch(dets_unknown, True)
        FakeTime.now += 1
    assert t.live_states() == {}
    assert tmp_db.execute(
        "SELECT COUNT(*) FROM activity_logs").fetchone()[0] == 0
    assert tmp_db.execute(
        "SELECT COUNT(*) FROM employees WHERE id IN ('Unknown','UNKNOWN')"
    ).fetchone()[0] == 0
    # A real employee still tracks perfectly afterwards.
    t.process_batch([{"cam": "c1", "emp_id": "EMP001", "phone": False}], True)
    assert t.live_states().get("EMP001") == "ACTIVE"


# ---------------------------------------------------------------- irregular frames

def test_phone_threshold_uses_wall_clock_not_frame_count(fake_clock, recorder, instant_commit):
    """Five tense frames whose gaps sum to >5s must still commit."""
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1
    t.process(_phone())                   # episode start, 0.0s
    FakeTime.now += 0.7                    # frame 2
    t.process(_phone())
    FakeTime.now += 0.7                    # frame 3, cum 1.4
    t.process(_phone())
    FakeTime.now += 0.7                    # frame 4, cum 2.1
    t.process(_phone())
    FakeTime.now += 0.7                    # frame 5, cum 2.8
    t.process(_phone())
    FakeTime.now += 0.7                    # frame 6, cum 3.5
    t.process(_phone())
    FakeTime.now += 0.7                    # frame 7, cum 4.2
    t.process(_phone())
    FakeTime.now += 0.7                    # frame 8, cum 4.9 < 5
    t.process(_phone())
    assert t.committed_state == "ACTIVE"  # 4.9s over 9 frames: still not ON_PHONE
    FakeTime.now += 0.7                    # cum 5.6 >= 5
    t.process(_phone())
    assert t.committed_state == "ON_PHONE"


def test_absence_threshold_uses_wall_clock_not_frame_count(fake_clock, recorder, instant_commit):
    """Absence gaps of 0.5/1.0/1.6s accumulate to >3s across three frames."""
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())                 # last_seen = t0
    FakeTime.now += 0.5
    t.process(_absent())                  # 0.5s absent
    FakeTime.now += 1.0
    t.process(_absent())                  # 1.5s absent
    assert t.committed_state == "ACTIVE"
    FakeTime.now += 1.6
    t.process(_absent())                  # 3.1s absent -> AWAY
    assert t.committed_state == "AWAY"
    assert recorder[-1][1] == "EMP001"
    assert 2.9 <= recorder[-1][3] <= 3.3