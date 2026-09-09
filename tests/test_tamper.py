"""Tests for the camera tamper monitor."""

import numpy as np
import pytest

from src.domain import CAMERA_TAMPER_CLEARED, CAMERA_TAMPER_SUSPECTED
from src.tamper_monitor import TamperMonitor


def _frame(value=128, size=(64, 64, 3)):
    """Create a solid-color BGR frame."""
    return np.full(size, value, dtype=np.uint8)


def test_frozen_frame_fires_tamper():
    mon = TamperMonitor(persist_sec=0)
    frame = _frame(100)
    events = []
    # Feed two identical frames -> frozen
    events += mon.analyze("cam_01", frame, now=0.0)
    events += mon.analyze("cam_01", frame, now=1.0)
    types = [e["event_type"] for e in events]
    assert CAMERA_TAMPER_SUSPECTED in types


def test_persist_threshold_prevents_false_positive():
    mon = TamperMonitor(persist_sec=30)
    frame = _frame(100)
    events = []
    events += mon.analyze("cam_01", frame, now=0.0)
    events += mon.analyze("cam_01", frame, now=1.0)  # frozen but < 30s
    assert not any(e["event_type"] == CAMERA_TAMPER_SUSPECTED for e in events)


def test_extreme_darkness():
    mon = TamperMonitor(persist_sec=0, enable_dark=True)
    dark = _frame(5)  # mean ~5 < dark_mean 30
    events = []
    events += mon.analyze("cam_01", dark, now=0.0)
    events += mon.analyze("cam_01", dark, now=1.0)
    assert any(e["event_type"] == CAMERA_TAMPER_SUSPECTED for e in events)


def test_normal_frame_no_tamper():
    mon = TamperMonitor(persist_sec=0)
    f1 = _frame(100)
    f2 = _frame(120)  # different enough -> not frozen
    events = []
    events += mon.analyze("cam_01", f1, now=0.0)
    events += mon.analyze("cam_01", f2, now=1.0)
    assert not any(e["event_type"] == CAMERA_TAMPER_SUSPECTED for e in events)


def test_recovery_emits_cleared():
    mon = TamperMonitor(persist_sec=0)
    dark = _frame(5)
    events = []
    events += mon.analyze("cam_01", dark, now=0.0)
    events += mon.analyze("cam_01", dark, now=1.0)
    assert any(e["event_type"] == CAMERA_TAMPER_SUSPECTED for e in events)
    clear_events = mon.analyze("cam_01", _frame(100), now=2.0)
    assert any(e["event_type"] == CAMERA_TAMPER_CLEARED for e in clear_events)


def test_none_frame_ignored():
    mon = TamperMonitor(persist_sec=0)
    assert mon.analyze("cam_01", None, now=0.0) == []
