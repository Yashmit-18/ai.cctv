"""Tests for shared analytics and the Unknown-filtering rule."""

from datetime import date, datetime

import pytest

from src.analytics import employee_day_metrics, employee_range_metrics, is_unknown
from src.database import log_interval, upsert_employee
from src.domain import UNKNOWN_ID, WorkSchedule

DAY = date(2026, 9, 1)


def _log(conn, day_str, hm, emp, state, dur_min, cam="cam1"):
    log_interval(conn, f"{day_str} {hm}:00", emp, state, dur_min * 60, source=cam)


@pytest.fixture
def schedule():
    return WorkSchedule(start_min=540, end_min=1080, lunch_min=(720, 780),
                        break_minutes=30)


# ---------------------------------------------------------------- unknown filter

def test_is_unknown_variants():
    assert is_unknown("Unknown")
    assert is_unknown("UNKNOWN")
    assert is_unknown("unknown")
    assert not is_unknown("EMP001")


def test_unknown_rows_never_produce_metric_rows(tmp_db, schedule):
    _log(tmp_db, "2026-09-01", "09:00", UNKNOWN_ID, "ACTIVE", 60)
    _log(tmp_db, "2026-09-01", "11:00", UNKNOWN_ID, "ON_PHONE", 30)
    rows = employee_day_metrics(tmp_db, DAY, schedule)
    # Unknown is the ONLY record -> no recognised employee rows at all
    assert rows == []


def test_unknown_and_recognised_in_same_day(tmp_db, schedule):
    _log(tmp_db, "2026-09-01", "09:00", UNKNOWN_ID, "ACTIVE", 60)
    _log(tmp_db, "2026-09-01", "10:00", "EMP001", "ACTIVE", 60)
    rows = employee_day_metrics(tmp_db, DAY, schedule)
    ids = [r["employee_id"] for r in rows]
    assert "EMP001" in ids
    assert UNKNOWN_ID not in ids
    emp = [r for r in rows if r["employee_id"] == "EMP001"][0]
    assert emp["productive_seconds"] == 3600


def test_enrolled_but_unobserved_still_rostered(tmp_db, schedule):
    upsert_employee(tmp_db, "EMP007", name="Ghost", department="IT",
                    designation="Dev", active=True)
    rows = employee_day_metrics(tmp_db, DAY, schedule)
    assert rows  # roster row exists even with zero observations
    g = [r for r in rows if r["employee_id"] == "EMP007"][0]
    assert g["status"] == "NOT OBSERVED"
    assert g["name"] == "Ghost"


# ---------------------------------------------------------------- day metrics

def test_day_metrics_with_phone_and_away(tmp_db, schedule):
    _log(tmp_db, "2026-09-01", "09:00", "EMP001", "ACTIVE", 120)
    _log(tmp_db, "2026-09-01", "11:00", "EMP001", "ON_PHONE", 30)
    _log(tmp_db, "2026-09-01", "12:00", "EMP001", "AWAY", 60)
    rows = employee_day_metrics(tmp_db, DAY, schedule)
    emp = [r for r in rows if r["employee_id"] == "EMP001"][0]
    assert emp["productive_seconds"] == 7200
    assert emp["phone_seconds"] == 1800
    assert emp["away_seconds"] == 0  # full lunch window excluded
    assert emp["productive_pct"] == 80.0  # 2h / (2h + 0.5h)


def test_day_metrics_late_early_outtimes(tmp_db, schedule):
    _log(tmp_db, "2026-09-01", "09:06", "EMP001", "ACTIVE", 5)
    _log(tmp_db, "2026-09-01", "17:55", "EMP001", "ACTIVE", 5)
    emp = [r for r in employee_day_metrics(tmp_db, DAY, schedule)
           if r["employee_id"] == "EMP001"][0]
    assert emp["in_time"].startswith("2026-09-01 09:06")
    assert emp["out_time"].startswith("2026-09-01 18:00")  # end of last obs


# ---------------------------------------------------------------- range metrics

def test_range_metrics_splits_by_day(tmp_db, schedule):
    log_interval(tmp_db, "2026-08-31 10:00:00", "EMP001", "ACTIVE", 3600, source="cam1")
    log_interval(tmp_db, "2026-09-01 10:00:00", "EMP001", "ACTIVE", 3600, source="cam1")
    by_day = employee_range_metrics(tmp_db, date(2026, 8, 31), date(2026, 9, 1), schedule)
    assert set(by_day.keys()) == {date(2026, 8, 31), date(2026, 9, 1)}
    assert len(by_day[date(2026, 8, 31)]) == 1
    assert len(by_day[date(2026, 9, 1)]) == 1
    assert by_day[date(2026, 9, 1)][0]["productive_seconds"] == 3600
    assert by_day[date(2026, 8, 31)][0]["productive_seconds"] == 3600


def test_range_metrics_midnight_interval_lands_on_both_days(tmp_db):
    # 24h working window so both sides of midnight are scored.
    aweek = WorkSchedule(start_min=0, end_min=60 * 24, lunch_min=(0, 0),
                         break_minutes=0)
    # single observation spanning midnight: 30s on 08-31, 90s on 09-01
    log_interval(tmp_db, "2026-08-31 23:59:30", "EMP001", "ACTIVE", 120, source="cam1")
    by_day = employee_range_metrics(tmp_db, date(2026, 8, 31), date(2026, 9, 1), aweek)
    assert by_day[date(2026, 8, 31)][0]["productive_seconds"] == 30
    assert by_day[date(2026, 9, 1)][0]["productive_seconds"] == 90


def test_range_metrics_empty_range(tmp_db, schedule):
    by_day = employee_range_metrics(tmp_db, date(2026, 9, 1), date(2026, 9, 5), schedule)
    for dt, rows in by_day.items():
        assert rows == []