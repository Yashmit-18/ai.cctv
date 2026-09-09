"""Tests for the camera offline/recovered security FSM."""

import pytest

from src.camera_state import CameraStateManager
from src.domain import CAMERA_OFFLINE, CAMERA_RECOVERED


def test_no_event_when_online():
    mgr = CameraStateManager(offline_trigger_sec=15)
    events = mgr.update("cam_01", True, now=100.0)
    assert events == []


def test_offline_after_threshold():
    mgr = CameraStateManager(offline_trigger_sec=15)
    mgr.update("cam_01", True, now=0.0)
    mgr.update("cam_01", False, now=10.0)   # offline since=10, elapsed=10 < 15
    assert mgr.update("cam_01", False, now=20.0) == []  # elapsed=10 still < 15
    events = mgr.update("cam_01", False, now=40.0)  # elapsed=30 >= 15
    assert len(events) == 1
    assert events[0]["event_type"] == CAMERA_OFFLINE
    assert events[0]["camera"] == "cam_01"
    assert events[0]["confidence"] == 1.0


def test_offline_not_fired_repeatedly():
    mgr = CameraStateManager(offline_trigger_sec=15)
    mgr.update("cam_01", True, now=0.0)
    mgr.update("cam_01", False, now=20.0)  # offline since=20; no fire yet (0 elapsed)
    mgr.update("cam_01", False, now=40.0)  # elapsed=20 >= 15 -> fires
    assert mgr.update("cam_01", False, now=60.0) == []  # not re-fired


def test_recovery_after_offline():
    mgr = CameraStateManager(offline_trigger_sec=15)
    mgr.update("cam_01", True, now=0.0)
    mgr.update("cam_01", False, now=20.0)  # offline fires
    events = mgr.update("cam_01", True, now=30.0)
    assert len(events) == 1
    assert events[0]["event_type"] == CAMERA_RECOVERED
    assert events[0]["duration_seconds"] >= 9.0  # downtime tracked


def test_downtime_tracking():
    mgr = CameraStateManager(offline_trigger_sec=15)
    mgr.update("cam_01", True, now=0.0)
    mgr.update("cam_01", False, now=20.0)
    mgr.update("cam_01", False, now=40.0)  # fires offline, records downtime
    assert mgr.downtime("cam_01") > 0


def test_is_online_defaults():
    mgr = CameraStateManager()
    assert mgr.is_online("cam_unknown") is True


def test_never_employee_away():
    """Offline state must never touch employee states (guaranteed by design)."""
    mgr = CameraStateManager(offline_trigger_sec=15)
    mgr.update("cam_01", False, now=30.0)
    # The FSM only emits CAMERA events, never employee states.
    events = mgr.update("cam_01", False, now=100.0)
    assert all(e["event_type"].startswith("CAMERA") for e in events)
