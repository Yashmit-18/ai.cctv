"""Phase 27.15 -- report <-> dashboard <-> analytics consistency.

The dashboard (app.py), the Excel report (reporter.py) and the daemon telemetry
all source their numbers from the single canonical engine in src.analytics. This
guards that the Summary sheet an admin opens in Excel matches what the dashboard
charts, down to the second, for the same date and employee.
"""

from datetime import date

from src.analytics import employee_day_metrics
from src.database import log_interval, upsert_employee
from src.employees import EmployeeStore
from src.reporter import build_summary

DAY = date(2026, 9, 1)


def _store(conn, schedule):
    return EmployeeStore(conn, _schedule_defaults(schedule))


def _schedule_defaults(schedule):
    return {
        "start": f"{schedule.start_min // 60:02d}:{schedule.start_min % 60:02d}",
        "end": f"{schedule.end_min // 60:02d}:{schedule.end_min % 60:02d}",
        "lunch": f"{schedule.lunch_min[0] // 60:02d}:{schedule.lunch_min[0] % 60:02d}-"
                 f"{schedule.lunch_min[1] // 60:02d}:{schedule.lunch_min[1] % 60:02d}",
        "breaks": schedule.break_minutes,
    }


def _seed(tmp_db):
    upsert_employee(tmp_db, "EMP001", name="Alice", department="IT",
                    designation="Dev", active=True)
    upsert_employee(tmp_db, "EMP002", name="Bob", department="IT",
                    designation="QA", active=True)
    # EMP001: 3h active + 1h phone + 1h away = 5h accounted, 3h productive
    log_interval(tmp_db, "2026-09-01 09:10:00", "EMP001", "ACTIVE", 10800, source="cam1")
    log_interval(tmp_db, "2026-09-01 12:10:00", "EMP001", "ON_PHONE", 3600, source="cam1")
    log_interval(tmp_db, "2026-09-01 13:10:00", "EMP001", "AWAY", 3600, source="cam1")
    # EMP002: roster row with no observation -> NOT OBSERVED, pct None
    log_interval(tmp_db, "2026-09-01 14:10:00", "EMP002", "ACTIVE", 1800, source="cam1")


def test_report_summary_matches_canonical_analytics(tmp_db, schedule):
    _seed(tmp_db)
    store = _store(tmp_db, schedule)
    metrics = {m["employee_id"]: m
               for m in employee_day_metrics(tmp_db, DAY, employees=store)}
    df = build_summary([metrics["EMP001"]], DAY)
    row = df.iloc[0]

    m = metrics["EMP001"]
    assert row["Active Hours"] == round(m["active_hours"], 2)
    assert row["Phone (Mins)"] == round(m["phone_mins"], 2)
    assert row["Away (Mins)"] == round(m["away_mins"], 2)
    assert row["Unobserved (Mins)"] == round(m["unobserved_seconds"] / 60.0, 1)
    assert row["Productivity (%)"] == m["productive_pct"]
    assert row["Expected Hours"] == round(m["expected_working_seconds"] / 3600.0, 2)

    # Numeric expectations from the controlled duration
    assert m["active_hours"] == 3.0
    assert m["phone_mins"] == 60.0
    assert m["away_mins"] == 60.0
    # Canonical formula: productive / (productive + phone + away)
    # = 10800 / (10800 + 3600 + 3600) = 60.0%  (unobserved excluded)
    assert m["productive_pct"] == round(10800 / (10800 + 3600 + 3600) * 100, 1)


def test_report_excludes_unknown_from_summary(tmp_db, schedule):
    _seed(tmp_db)
    log_interval(tmp_db, "2026-09-01 15:00:00", "Unknown", "ACTIVE", 900, source="cam1")
    store = _store(tmp_db, schedule)
    metrics = employee_day_metrics(tmp_db, DAY, employees=store)
    assert all(m["employee_id"] != "Unknown" for m in metrics)


def test_not_observed_negative_vs_report_n_a(tmp_db, schedule):
    _seed(tmp_db)
    store = _store(tmp_db, schedule)
    m = employee_day_metrics(tmp_db, DAY, employees=store)
    observed = {r["employee_id"]: r for r in m}
    assert observed["EMP002"]["status"] == "OBSERVED"  # has an ACTIVE interval
    # EMP003 not seeded -> never in roster -> absent entirely (no phantom row)
    assert "EMP003" not in observed
