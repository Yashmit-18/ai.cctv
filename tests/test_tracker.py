"""Tests for the employee-state FSM (offline freeze, presence patience, dedup)."""

import time as _realtime

import pytest

import src.tracker as T


class FakeTime:
    """Monotonic fake clock shared across the tracker module."""
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
def fast_buffer(monkeypatch):
    monkeypatch.setattr(T, "SMOOTHING_BUFFER_SEC", 5.0)
    monkeypatch.setattr(T, "IDENTITY_STABILITY_FRAMES", 3)
    monkeypatch.setattr(T, "PRESENCE_PATIENCE_SEC", 8.0)
    return T.SMOOTHING_BUFFER_SEC


def _present() -> dict:
    return {"present": True, "phone_detected": False}


def _phone() -> dict:
    return {"present": True, "phone_detected": True}


def _absent() -> dict:
    return {"present": False, "phone_detected": False}


# ---------------------------------------------------------------- first observation

def test_first_observation_adopted_immediately(fake_clock, recorder):
    t = T.EmployeeTracker(None, "EMP001", source="cam1")
    t.process(_present())
    assert t.committed_state == "ACTIVE"
    assert t.current_state == "ACTIVE"
    assert recorder == []  # first state adopted, nothing logged yet


def test_first_observation_never_begins_as_away(fake_clock, recorder):
    t = T.EmployeeTracker(None, "EMP002")
    t.process(_phone())
    assert t.committed_state == "ON_PHONE"
    assert t.current_state == "ON_PHONE"


# ---------------------------------------------------------------- offline freeze

def test_offline_freeze_holds_state_no_away(fake_clock, recorder):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())          # becomes ACTIVE
    FakeTime.now += 60             # camera offline for a minute
    t.process(_absent(), camera_online=False)
    assert t.committed_state == "ACTIVE"
    assert recorder == []          # no AWAY time fabricated


def test_offline_then_recover_no_instant_commit(fake_clock, recorder, fast_buffer):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 1000
    for _ in range(5):             # repeated offline cycles
        t.process(_absent(), camera_online=False)
    assert t.committed_state == "ACTIVE"
    # back online: raw=AWAY. Buffer must still apply before any commit.
    t.process(_absent(), camera_online=True)
    assert t.committed_state == "ACTIVE"
    assert recorder == []


# ---------------------------------------------------------------- presence / wall-clock away

def test_brief_face_gap_holds_active_within_grace(fake_clock, recorder, fast_buffer):
    """Absence under the 3s wall-clock grace keeps ACTIVE (no AWAY flicker)."""
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())                       # ACTIVE
    FakeTime.now += 2                           # < 3s
    t.process(_absent(), person_present=True)   # face unmatched but person there
    assert t.committed_state == "ACTIVE"
    assert t.current_state == "ACTIVE"
    assert recorder == []


def test_absence_over_grace_promotes_to_away_even_with_person(fake_clock, recorder, fast_buffer):
    """Absence beyond 3s -> AWAY even while a person box remains on screen."""
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())                       # ACTIVE
    FakeTime.now += 4
    t.process(_absent(), person_present=True)   # > 3s absent
    assert t.committed_state == "AWAY"
    assert recorder and recorder[-1][1] == "EMP001"
    assert recorder[-1][2] == "ACTIVE"


def test_person_gone_promotes_to_away_after_patience(fake_clock, recorder, fast_buffer):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())
    FakeTime.now += 10
    # person leaves entirely
    t.process(_absent(), person_present=False)
    FakeTime.now += 10
    t.process(_absent(), person_present=False)
    FakeTime.now += 10
    t.process(_absent(), person_present=False)  # buffer elapsed
    assert t.current_state == "AWAY" or t.committed_state == "AWAY"


def test_observer_commit_logs_previous_interval(fake_clock, recorder, fast_buffer):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())                    # t=0 ACTIVE
    FakeTime.now += 10
    for _ in range(3):                       # 3 stable ON_PHONE frames
        t.process(_phone())
    FakeTime.now += 10
    t.process(_phone())                      # buffer elapsed -> commit
    assert t.committed_state == "ON_PHONE"
    assert recorder and recorder[-1][1] == "EMP001"
    assert recorder[-1][2] == "ACTIVE"
    assert recorder[-1][3] >= 20


# ---------------------------------------------------------------- dedup / multi-cam

def test_dedup_two_cameras_one_employee(fake_clock, recorder, fast_buffer):
    m = T.MultiTracker(None)
    dets = [
        {"emp_id": "EMP001", "present": True, "phone": False},
        {"emp_id": "EMP001", "present": True, "phone": False},
    ]
    merged = m._deduplicate(dets)
    assert merged["EMP001"]["present"] is True
    # process_batch should create exactly ONE tracker
    m.process_batch(dets, camera_online=True)
    assert m.employee_ids == ["EMP001"]


def test_person_sentinel_does_not_create_tracker(fake_clock, recorder, fast_buffer):
    m = T.MultiTracker(None)
    dets = [{"emp_id": "__person__", "present": True, "person_present": True}]
    m.process_batch(dets, camera_online=True)
    assert m.employee_ids == []
    assert m._person_detected(dets) is True


def test_process_batch_offline_freezes(fake_clock, recorder, fast_buffer):
    m = T.MultiTracker(None)
    m.process_batch([{"emp_id": "EMP001", "present": True}], camera_online=True)
    FakeTime.now += 50
    m.process_batch([{"emp_id": "EMP001", "present": False}], camera_online=False)
    assert m.live_state("EMP001") == "ACTIVE"


def test_flush_all_logs_final_interval(fake_clock, recorder, fast_buffer):
    m = T.MultiTracker(None)
    m.process_batch([{"emp_id": "EMP001", "present": True}], camera_online=True)
    FakeTime.now += 5
    m.flush_all()
    assert any(rec[1] == "EMP001" for rec in recorder)


# ---------------------------------------------------------------- backward alias

def test_state_tracker_legacy_smoke(fake_clock, recorder, fast_buffer):
    st = T.StateTracker(None)
    st.process("EMP001", _present())
    assert st.current_state("EMP001") == "ACTIVE"
    st.process("EMP001", _present())
    assert st.current_state("EMP001") == "ACTIVE"
    FakeTime.now += 100
    FakeTime.now += 5
    st.process("EMP001", _present())  # stays ACTIVE, no-op
    assert st.current_state("EMP001") == "ACTIVE"