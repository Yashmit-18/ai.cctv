"""Tests for domain constants/helpers (no env/config dependency)."""

from src.domain import (
    CAM_NO_FRAME,
    CAM_OFFLINE,
    CAM_ONLINE,
    UNKNOWN_ID,
    Employee,
    WorkSchedule,
    redact_url,
)


# ---------------------------------------------------------------- redact_url

def test_redact_url_basic_credentials():
    assert redact_url("rtsp://admin:secret@host/stream") == "rtsp://admin:***@host/stream"


def test_redact_url_no_credentials_unchanged():
    assert redact_url("rtsp://host/stream") == "rtsp://host/stream"


def test_redact_url_none_safe():
    assert redact_url(None) == "None"
    assert redact_url("") == ""


def test_redact_url_no_userinfo_but_at_host():
    # username-only authority: keep the username, hide whatever follows
    assert redact_url("rtsp://user@host/stream") == "rtsp://user:***@host/stream"


# ---------------------------------------------------------------- WorkSchedule

def test_schedule_from_config_hhmm_strings():
    s = WorkSchedule.from_config({
        "start": "09:30",
        "end": "18:30",
        "lunch": "12:00-13:30",
        "breaks": 45,
    })
    assert s.start_min == 9 * 60 + 30
    assert s.end_min == 18 * 60 + 30
    assert s.lunch_min == (720, 810)
    assert s.break_minutes == 45


def test_schedule_from_config_int_minutes():
    s = WorkSchedule.from_config({"start": 540, "end": 1080, "lunch": (720, 780)})
    assert (s.start_min, s.end_min) == (540, 1080)
    assert s.lunch_min == (720, 780)


def test_schedule_from_config_none_gives_default_none():
    # from_config(None) returns None; callers fall back to their own default
    assert WorkSchedule.from_config(None) is None
    assert WorkSchedule.from_config({}) == WorkSchedule()


def test_schedule_lunch_sorted_even_if_reversed():
    s = WorkSchedule.from_config({"lunch": (780, 720)})
    assert s.lunch_min == (720, 780)


# ---------------------------------------------------------------- constants / model

def test_state_and_health_constants():
    assert UNKNOWN_ID == "Unknown"
    assert CAM_ONLINE == "ONLINE" and CAM_OFFLINE == "OFFLINE"
    assert CAM_NO_FRAME == "NO_FRAME"


def test_employee_dataclass_defaults():
    e = Employee("EMP001")
    assert e.name == "" and e.active is True and e.enrolled is False
    assert e.metadata == {}