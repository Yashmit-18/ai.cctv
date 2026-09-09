"""Invariant checks on shared analytics output (Phase 7/12).

Guards that every metric feed (day + range) stays within sane bounds and
that the roster/status contract the dashboard relies on is stable.
"""

from datetime import date

from src.analytics import employee_day_metrics, employee_range_metrics
from src.database import log_interval, upsert_employee

DAY = date(2026, 9, 1)


def _seed(conn, schedule):
    upsert_employee(conn, "EMP001", name="Alice", department="IT",
                    designation="Dev", active=True)
    upsert_employee(conn, "EMP002", name="Bob", department="IT",
                    designation="QA", active=True)
    upsert_employee(conn, "EMP003", name="Zed", department="HR",
                    designation="HR", active=True)   # enrolled roster, no obs
    # EMP001 fully productive
    log_interval(conn, "2026-09-01 09:10:00", "EMP001", "ACTIVE", 7200, source="cam1")
    # EMP002 phone + away heavy
    log_interval(conn, "2026-09-01 13:30:00", "EMP002", "ON_PHONE", 2700, source="cam1")
    log_interval(conn, "2026-09-01 14:15:00", "EMP002", "AWAY", 1800, source="cam1")


def test_day_metrics_pct_in_bounds(tmp_db, schedule):
    _seed(tmp_db, schedule)
    for row in employee_day_metrics(tmp_db, DAY, schedule):
        assert row["employee_id"] != "UNKNOWN"
        assert row["unobserved_seconds"] >= 0
        assert row["status"] in ("OBSERVED", "NOT OBSERVED")
        assert row["active_hours"] >= 0
        assert row["phone_mins"] >= 0 and row["away_mins"] >= 0
        if row["status"] == "OBSERVED":
            assert row["productive_pct"] is not None
            assert 0.0 <= row["productive_pct"] <= 100.0
        else:
            # roster-only row: no observed time -> pct is undefined (None),
            # and the dashboard coerces that to NaN, never a phony number.
            assert row["productive_pct"] is None


def test_pure_away_day_is_zero_not_negative(tmp_db, schedule):
    upsert_employee(tmp_db, "EMP004", name="Night", active=True)
    log_interval(tmp_db, "2026-09-01 13:30:00", "EMP004", "AWAY", 3600, source="cam1")
    row = [r for r in employee_day_metrics(tmp_db, DAY, schedule)
           if r["employee_id"] == "EMP004"][0]
    assert row["productive_pct"] == 0.0
    assert row["status"] == "OBSERVED"


def test_metrics_time_bucket_invariants(tmp_db, schedule):
    _seed(tmp_db, schedule)
    display = {
        "active_hours", "phone_mins", "away_mins", "unobserved_seconds",
        "productive_pct", "status", "in_time", "out_time",
        "expected_working_seconds",
    }
    for row in employee_day_metrics(tmp_db, DAY, schedule):
        assert display.issubset(row.keys())
    # The three seeded + any roster-only rows must not double-count time
    by_id = {r["employee_id"]: r for r in employee_day_metrics(tmp_db, DAY, schedule)}
    emp1 = by_id["EMP001"]
    assert emp1["active_hours"] == 2.0
    assert emp1["productive_pct"] == 100.0


def test_range_metrics_pct_in_bounds(tmp_db, schedule):
    _seed(tmp_db, schedule)
    by_day = employee_range_metrics(tmp_db, DAY, DAY, schedule)
    rows = by_day.get(DAY, [])
    assert rows
    for row in rows:
        if row["status"] == "OBSERVED":
            assert row["productive_pct"] is not None
            assert 0.0 <= row["productive_pct"] <= 100.0


def test_roster_rows_only_for_known_employees(tmp_db, schedule):
    _seed(tmp_db, schedule)
    ids = {r["employee_id"] for r in employee_day_metrics(tmp_db, DAY, schedule)}
    assert {"EMP001", "EMP002", "EMP003"} == ids