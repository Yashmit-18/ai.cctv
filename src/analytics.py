"""Shared analytics: turn raw DB activity rows into per-employee metrics.

Both :mod:`src.reporter` and the Streamlit dashboard (``app.py``) use this as
their single canonical source for productivity numbers.  The heavy lifting
(decode intervals, filter ``Unknown``, compute working-window productivity,
split midnight intervals) lives in :mod:`src.productivity`.
"""

from __future__ import annotations

import logging
from datetime import date

from src import database as db
from src.domain import UNKNOWN_ID, WorkSchedule
from src.productivity import compute_day_metrics, from_rows
from src.employees import EmployeeStore

logger = logging.getLogger("cctv.analytics")

_UNKNOWN_IDS = {"Unknown", "UNKNOWN", "unknown"}


def is_unknown(employee_id: str) -> bool:
    return employee_id in _UNKNOWN_IDS


def employee_day_metrics(conn, day: date, schedule: WorkSchedule | None = None,
                         employees: EmployeeStore | None = None) -> list[dict]:
    """Return sorted per-employee metrics rows for ``day``.

    Only recognised employees produce a row -- ``Unknown`` is always
    excluded from productivity analytics.
    """
    if employees is None:
        employees = EmployeeStore(conn, _cfg_schedule(schedule))
    rows = db.query_day_intervals(conn, day.isoformat())
    return _compute_from_rows(rows, day, employees, schedule)


def employee_range_metrics(conn, start: date, end: date,
                           schedule: WorkSchedule | None = None,
                           employees: EmployeeStore | None = None) -> dict[date, list[dict]]:
    """Return ``{date: [metrics_rows]}`` for the inclusive range."""
    if employees is None:
        employees = EmployeeStore(conn, _cfg_schedule(schedule))
    rows = db.query_range_intervals(conn, start.isoformat(), end.isoformat())
    return _compute_range_from_rows(rows, start, end, employees, schedule)


def _cfg_schedule(schedule: WorkSchedule | None) -> dict | None:
    if schedule is not None:
        return {
            "start": _fmt(schedule.start_min),
            "end": _fmt(schedule.end_min),
            "lunch": f"{_fmt(schedule.lunch_min[0])}-{_fmt(schedule.lunch_min[1])}",
            "grace": schedule.grace_period_min,
            "breaks": schedule.break_minutes,
        }
    return None


def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _compute_from_rows(rows, day: date, employees: EmployeeStore,
                       schedule: WorkSchedule | None) -> list[dict]:
    """Filter Unknown, decode intervals, compute metrics per employee."""
    non_unknown = [r for r in rows if not is_unknown(r[1])]

    # Observed employees this day, plus the enrolled roster so every employee
    # appears even when they have zero observations that day.
    received_ids = {r[1] for r in non_unknown}
    enrolled_ids = {e.employee_id for e in employees.list()
                    if e.employee_id not in _UNKNOWN_IDS}
    all_ids = sorted(received_ids | enrolled_ids)
    if not all_ids:
        return []

    out: list[dict] = []
    for emp_id in all_ids:
        emp_rows = [r for r in non_unknown if r[1] == emp_id]
        intervals = from_rows(emp_rows)
        sched = schedule or employees.schedule_for(emp_id)
        m = compute_day_metrics(emp_id, intervals, day, sched)
        emp = employees.get(emp_id)
        m["name"] = emp.name if emp else ""
        m["department"] = emp.department if emp else ""
        m["designation"] = emp.designation if emp else ""
        m["enrolled"] = bool(emp.enrolled) if emp else True
        out.append(m)
    out.sort(key=lambda x: x["employee_id"])
    return out


def _compute_range_from_rows(rows, start: date, end: date, employees: EmployeeStore,
                             schedule: WorkSchedule | None) -> dict[date, list[dict]]:
    """Return ``{day: metrics}`` for each day in [start, end].

    Intervals that cross midnight are attributed to *every* day they touch:
    :func:`~src.productivity.from_rows` splits them per-day, and
    :func:`_compute_from_rows` clips each side to the correct calendar day.
    """
    non_unknown = [r for r in rows if not is_unknown(r[1])]

    buckets: dict[date, list] = {}
    for r in non_unknown:
        for iv in from_rows([r]):
            d = iv.start_ts.date()
            buckets.setdefault(d, []).append(r)

    result: dict[date, list[dict]] = {}
    d = start
    while d <= end:
        result[d] = _compute_from_rows(buckets.get(d, []), d, employees, schedule)
        d = date.fromordinal(d.toordinal() + 1)
    return result


# Minimal public convenience -------------------------------------------------

def today_metrics(conn, schedule: WorkSchedule | None = None,
                  employees: EmployeeStore | None = None) -> list[dict]:
    return employee_day_metrics(conn, date.today(), schedule, employees)
