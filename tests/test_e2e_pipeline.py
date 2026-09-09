"""End-to-end pipeline integration test.

Seeds a temp DB with a realistic synthetic workday and pushes it through the
full data path:

    DB (log_interval) -> productivity.from_rows -> analytics
                        -> reporter.generate_daily_report -> .xlsx

This is the same path the daemon, dashboard and report share.
"""

from datetime import date, datetime, timedelta
from pathlib import Path

from src.analytics import employee_day_metrics, employee_range_metrics
from src.database import upsert_employee
from src.domain import UNKNOWN_ID, WorkSchedule
from src.reporter import generate_daily_report
from src.database import log_interval

DAY = date(2026, 9, 1)
WS = WorkSchedule(start_min=540, end_min=1080, lunch_min=(720, 780),
                  break_minutes=30)


def _seed_workday(tmp_db):
    """One employee, a clean 8.7h day: active, lunch away, phone, away."""
    upsert_employee(tmp_db, "EMP001", name="Alice", department="IT",
                    designation="Engineer")
    spec = [
        ("09:00", "11:00", "ACTIVE"),    # 2h
        ("11:00", "11:30", "ON_PHONE"),  # 0.5h phone
        ("11:30", "12:00", "ACTIVE"),    # 0.5h
        ("12:00", "13:00", "AWAY"),      # lunch (excluded)
        ("13:00", "16:00", "ACTIVE"),    # 3h
        ("16:00", "16:20", "ON_PHONE"),  # 20 min phone
        ("16:20", "18:00", "AWAY"),      # genuine away before end
    ]
    for hm, hm2, state in spec:
        start = datetime.strptime(f"2026-09-01 {hm}", "%Y-%m-%d %H:%M")
        end = datetime.strptime(f"2026-09-01 {hm2}", "%Y-%m-%d %H:%M")
        dur = (end - start).total_seconds()
        log_interval(tmp_db, start.strftime("%Y-%m-%d %H:%M:%S"),
                     "EMP001", state, dur, source="cam_01")
    # An unknown person seen for the whole morning (must not count anywhere).
    log_interval(tmp_db, "2026-09-01 09:00:00", UNKNOWN_ID, "ACTIVE",
                 180 * 60, source="cam_01")


def test_e2e_reporter_and_metrics_agree(tmp_db):
    _seed_workday(tmp_db)

    # --- analytics engine --------------------------------------------
    metrics = employee_day_metrics(tmp_db, DAY, WS)
    emp = [r for r in metrics if r["employee_id"] == "EMP001"][0]
    assert UNKNOWN_ID not in [r["employee_id"] for r in metrics]
    # productive = 2h + 0.5h + 3h = 5.5h; phone = 0.5h + 0.33h = 0.833h;
    # away = 1h40m (16:20-18:00) minus nothing (not lunch).
    assert emp["productive_seconds"] == pytest_approx(5.5 * 3600)
    assert emp["phone_seconds"] == pytest_approx((30 + 20) * 60, abs=1)
    assert emp["away_seconds"] == pytest_approx(100 * 60, abs=1)
    expected_pct = 5.5 * 3600 / (5.5 * 3600 + 50 * 60 + 100 * 60) * 100
    assert emp["productive_pct"] == pytest_approx(expected_pct, abs=0.1)

    # --- reporter ------------------------------------------------------
    report = generate_daily_report(tmp_db, DAY)
    path = Path(report)
    assert path.exists() and path.suffix == ".xlsx"

    import openpyxl
    wb = openpyxl.load_workbook(path)
    assert "Summary" in wb.sheetnames
    assert "Activity Timeline" in wb.sheetnames
    ws = wb["Summary"]
    # header row + one data row (EMP001) + totals row
    assert ws.max_row >= 3

    # The timeline sheet keeps the Unknown row for monitoring.
    tl = wb["Activity Timeline"]
    unknown_cells = [c for row in tl.iter_rows(values_only=True) for c in row
                     if c == UNKNOWN_ID]
    assert unknown_cells
    path.unlink(missing_ok=True)


def pytest_approx(value, abs=0.001):
    import pytest
    return pytest.approx(value, abs=abs)


def test_e2e_empty_day_produces_roster_only(tmp_db):
    upsert_employee(tmp_db, "EMP001", name="Alice")
    metrics = employee_day_metrics(tmp_db, DAY, WS)
    assert len(metrics) == 1
    assert metrics[0]["status"] == "NOT OBSERVED"
    report = generate_daily_report(tmp_db, DAY)
    Path(report).unlink(missing_ok=True)


def test_e2e_two_employees_two_days_range(tmp_db):
    upsert_employee(tmp_db, "EMP001", name="Alice")
    upsert_employee(tmp_db, "EMP002", name="Bob")
    log_interval(tmp_db, "2026-08-31 10:00:00", "EMP001", "ACTIVE", 3600, "cam_01")
    log_interval(tmp_db, "2026-09-01 10:00:00", "EMP002", "ACTIVE", 3600, "cam_01")
    by_day = employee_range_metrics(tmp_db, date(2026, 8, 31), DAY, WS)
    row_01 = [r for r in by_day[date(2026, 8, 31)] if r["employee_id"] == "EMP001"][0]
    row_02 = [r for r in by_day[DAY] if r["employee_id"] == "EMP002"][0]
    assert row_01["productive_seconds"] == 3600
    assert row_02["productive_seconds"] == 3600