"""Regression tests: offline time exclusion + live session metadata (P2/7/12).

Guards the fix where a camera-offline gap must never bleed into the next
committed interval (or into the final flush), and exposes the new
``duration_seconds`` / ``live_durations`` surface the dashboard relies on.
"""

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
    seen = []
    monkeypatch.setattr(T, "log_interval",
                        lambda conn, ts, emp, state, dur, source="": seen.append(
                            (ts, emp, state, dur, source)))
    return seen


@pytest.fixture
def fast_buffer(monkeypatch):
    monkeypatch.setattr(T, "SMOOTHING_BUFFER_SEC", 5.0)
    monkeypatch.setattr(T, "IDENTITY_STABILITY_FRAMES", 3)
    monkeypatch.setattr(T, "PRESENCE_PATIENCE_SEC", 8.0)


def _present(): return {"present": True, "phone_detected": False}
def _phone(): return {"present": True, "phone_detected": True}
def _absent(): return {"present": False, "phone_detected": False}


# ---------------------------------------------------------------- offline gap

def test_offline_gap_excluded_from_committed_interval(fake_clock, recorder, fast_buffer):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())                          # ACTIVE at t=10000
    FakeTime.now += 30
    FakeTime.now += 100                            # camera offline 100s
    t.process(_absent(), camera_online=False)      # freeze advances _since
    # back online; AWAY commits after the normal patience/buffer
    for _ in range(30):
        t.process(_absent(), camera_online=True, person_present=False)
        FakeTime.now += 1
    assert t.committed_state == "AWAY"
    assert recorder
    ts, emp, state, dur, src = recorder[-1]
    assert state == "ACTIVE"
    # offline gap (100s) must NOT be in the interval duration
    assert dur < 10, f"offline gap leaked into interval: {dur}s"


def test_offline_gap_excluded_from_flush(fake_clock, recorder):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())                          # ACTIVE
    FakeTime.now += 20
    FakeTime.now += 200                            # 200s offline
    t.process(_absent(), camera_online=False)
    FakeTime.now += 5                              # some online time afterwards
    t.flush()
    assert recorder
    assert recorder[-1][2] == "ACTIVE"
    assert recorder[-1][3] < 20                    # gap excluded, not 220s


def test_offline_before_first_observation_stays_blank(fake_clock, recorder):
    t = T.EmployeeTracker(None, "EMP001")
    FakeTime.now += 5000
    t.process(_absent(), camera_online=False)
    assert t.committed_state == "AWAY"
    assert t._state is None
    assert recorder == []


# ---------------------------------------------------------------- metadata

def test_duration_seconds_tracks_online_only(fake_clock):
    t = T.EmployeeTracker(None, "EMP001")
    assert t.duration_seconds == 0.0
    t.process(_present())
    FakeTime.now += 5
    assert t.duration_seconds == 5.0
    FakeTime.now += 30                            # offline gap
    t.process(_absent(), camera_online=False)
    FakeTime.now += 3
    assert t.duration_seconds == 3.0


def test_live_sources_propagated_from_camera(fake_clock):
    m = T.MultiTracker(None)
    m.process_batch([{"cam": "local_webcam", "emp_id": "EMP001", "present": True}],
                    camera_online=True)
    assert m.live_sources() == {"EMP001": "local_webcam"}


def test_live_durations_multitracker(fake_clock):
    m = T.MultiTracker(None)
    m.process_batch([{"emp_id": "EMP001", "present": True}], camera_online=True)
    FakeTime.now += 4
    d = m.live_durations()
    assert "EMP001" in d
    assert 0.0 < d["EMP001"] <= 4.0


# ---------------------------------------------------------------- identity floor

def test_identity_flicker_does_not_flip_state(fake_clock, recorder, fast_buffer):
    t = T.EmployeeTracker(None, "EMP001")
    t.process(_present())                          # ACTIVE
    for _ in range(3):                             # rapid flicker ON_PHONE->ACTIVE
        FakeTime.now += 1
        t.process(_phone())
        FakeTime.now += 1
        t.process(_present())
    assert t.committed_state == "ACTIVE"
    assert recorder == []                          # nothing committed