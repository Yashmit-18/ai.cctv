"""Tests for the canonical productivity engine (tests/test_productivity)."""

from datetime import date, datetime, timedelta

from src.domain import UNKNOWN_ID, WorkSchedule
from src.productivity import (
    Interval,
    _overlap_len,
    _window_seconds,
    compute_day_metrics,
    from_rows,
    split_interval_by_day,
)

DAY = date(2026, 9, 1)
T0 = datetime(2026, 9, 1, 0, 0, 0)
WS = WorkSchedule(start_min=540, end_min=1080, lunch_min=(720, 780), break_minutes=30)


def _iv(start_hm: str, end_hm: str, state: str, emp="EMP001") -> Interval:
    s = datetime.strptime(f"2026-09-01 {start_hm}", "%Y-%m-%d %H:%M")
    e = datetime.strptime(f"2026-09-01 {end_hm}", "%Y-%m-%d %H:%M")
    return Interval(s, e, emp, state, source="cam1")


# ---------------------------------------------------------------- windows

def test_window_seconds_excludes_lunch():
    # 09:00-18:00 is 9h = 32400s; lunch 12:00-13:00 subtracts 3600s -> 28800
    assert _window_seconds(WS) == 28800.0


def test_window_seconds_lunch_partially_outside():
    sched = WorkSchedule(start_min=540, end_min=600, lunch_min=(720, 780))
    # 09:00-10:00 = 3600s; lunch outside window entirely -> no subtraction
    assert _window_seconds(sched) == 3600.0
    assert _overlap_len(sched) == 0


def test_window_seconds_no_lunch_defined():
    sched = WorkSchedule(start_min=540, end_min=1080, lunch_min=(0, 0))
    assert _window_seconds(sched) == 32400.0


# ---------------------------------------------------------------- midnight

def test_split_interval_by_day_midnight():
    start = datetime(2026, 9, 1, 23, 59, 30)
    end = datetime(2026, 9, 2, 0, 1, 30)
    parts = split_interval_by_day(start, end, "EMP001", "ACTIVE", "cam1")
    assert len(parts) == 2
    assert parts[0].duration == 30.0
    assert parts[1].duration == 90.0
    assert parts[0].start_ts.day == 1 and parts[1].start_ts.day == 2


def test_split_interval_no_boundary_single_part():
    parts = split_interval_by_day(T0 + timedelta(hours=10), T0 + timedelta(hours=11),
                                  "EMP001", "AWAY")
    assert len(parts) == 1
    assert parts[0].duration == 3600.0


# ---------------------------------------------------------------- metrics

def test_full_day_active_is_100pct_and_expected_seconds():
    # A continuous 09:00-18:00 ACTIVE interval counts the whole span; the
    # expected window (net of lunch) is 8h = 28800s.
    ivs = [Interval(T0 + timedelta(minutes=540), T0 + timedelta(minutes=1080),
                    "EMP001", "ACTIVE")]
    m = compute_day_metrics("EMP001", ivs, DAY, WS)
    assert m["productive_seconds"] == 32400
    assert m["expected_working_seconds"] == 28800
    assert m["productive_pct"] == 100.0
    assert m["phone_seconds"] == 0 and m["away_seconds"] == 0
    assert m["unobserved_seconds"] == 0
    assert m["active_hours"] == 9.0
    assert m["status"] == "OBSERVED"


def test_away_during_lunch_not_counted():
    # away over lunch only, inside window 09-18: nothing counted
    ivs = [_iv("12:00", "13:00", "AWAY")]
    m = compute_day_metrics("EMP001", ivs, DAY, WS)
    assert m["away_seconds"] == 0.0
    assert m["productive_pct"] is None
    assert m["unobserved_seconds"] == 28800.0


def test_away_outside_window_not_counted():
    ivs = [_iv("18:30", "19:00", "AWAY")]
    m = compute_day_metrics("EMP001", ivs, DAY, WS)
    assert m["away_seconds"] == 0.0


def test_phone_reduces_pct():
    ivs = [
        _iv("09:00", "11:00", "ACTIVE"),
        _iv("11:00", "12:00", "ON_PHONE"),
        _iv("13:00", "16:00", "ACTIVE"),
        _iv("16:00", "17:00", "ON_PHONE"),
        _iv("17:00", "18:00", "AWAY"),
    ]
    m = compute_day_metrics("EMP001", ivs, DAY, WS)
    # productive = 2h + 3h = 5h; phone = 1h + 1h = 2h; away = 1h
    assert m["productive_seconds"] == 18000
    assert m["phone_seconds"] == 7200
    assert m["away_seconds"] == 3600
    assert m["productive_pct"] == round(5 / 8 * 100, 1)
    assert m["accounted_seconds"] == 28800
    assert m["unobserved_seconds"] == 0


def test_unobserved_never_punishes():
    # only 1h observed ACTIVE out of a 9h window
    ivs = [_iv("09:00", "10:00", "ACTIVE")]
    m = compute_day_metrics("EMP001", ivs, DAY, WS)
    assert m["productive_seconds"] == 3600
    assert m["unobserved_seconds"] == 28800 - 3600
    assert m["productive_pct"] == 100.0  # unobserved excluded from denominator


def test_in_out_time_measured():
    ivs = [_iv("09:05", "10:00", "ACTIVE"), _iv("17:55", "18:00", "AWAY")]
    m = compute_day_metrics("EMP001", ivs, DAY, WS)
    assert m["in_time"].startswith("2026-09-01 09:05")
    assert m["out_time"].startswith("2026-09-01 18:00")


def test_no_intervals_means_not_observed():
    m = compute_day_metrics("EMP001", [], DAY, WS)
    assert m["status"] == "NOT OBSERVED"
    assert m["productive_pct"] is None
    assert m["productive_seconds"] == 0


def test_unknown_id_still_computes_row_but_upstream_filters_it():
    # This module is agnostic; the *filter* is in src.analytics.is_unknown().
    assert UNKNOWN_ID == "Unknown"


def test_from_rows_skips_malformed_timestamp():
    rows = [
        ("not-a-date", "EMP001", "ACTIVE", 60.0, "cam1"),
        ("2026-09-01 10:00:00", "EMP001", "ACTIVE", 60.0, "cam1"),
    ]
    ivs = from_rows(rows)
    assert len(ivs) == 1


def test_from_rows_filters_by_employee():
    rows = [
        ("2026-09-01 10:00:00", "EMP999", "ACTIVE", 60.0, "cam1"),
        ("2026-09-01 11:00:00", "EMP001", "ACTIVE", 60.0, "cam1"),
    ]
    ivs = from_rows(rows, employee_id="EMP001")
    assert [i.employee_id for i in ivs] == ["EMP001"]


def test_compute_midnight_span_intervals_two_days():
    ivs = split_interval_by_day(datetime(2026, 9, 1, 23, 30), datetime(2026, 9, 2, 0, 30),
                                "EMP001", "ACTIVE", "cam1")
    assert len(ivs) == 2  # halves fall in different scoring days