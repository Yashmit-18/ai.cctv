"""Centralised productivity calculation engine (single source of truth).

All productivity numbers -- in the daemon telemetry, the Excel report and the
Streamlit dashboard -- MUST be derived through this module so the formula is
defined exactly once.

Model
-----
For a given day and employee we hold a list of observed state intervals
(``ACTIVE`` / ``ON_PHONE`` / ``AWAY``).  Productivity is scored against the
employee's **expected working window**:

    expected_working_seconds = (scheduled_end - scheduled_start) - lunch_window

Inside the window we bucket observation into:

    * productive   : ACTIVE seconds
    * phone        : ON_PHONE seconds
    * away         : AWAY seconds  (i.e. present-but-absent within window)
    * unobserved   : remaining window seconds with no observation
                     (camera downtime / blocked / outside any activity row)

The productivity score is the fraction of *accounted* (observed, in-window)
time that was productive:

    score = productive / max(1, productive + phone + away)

Unobserved time is deliberately EXCLUDED from the denominator so camera
downtime and non-working gaps never penalise an employee.  Time outside the
working window and inside the (unpaid) lunch window is also excluded.

Unknown faces never produce an employee row (handled upstream) -- this
module only ever receives recognised employee intervals.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from src.domain import WorkSchedule

# State / input interval keys
_INTERVAL_KEYS = ("timestamp", "employee_id", "state", "duration_seconds", "source")


class Interval:
    """A single observed state interval for one employee over ``duration``."""

    __slots__ = ("start_ts", "end_ts", "employee_id", "state", "source")

    def __init__(self, start_ts: datetime, end_ts: datetime,
                 employee_id: str, state: str, source: str = ""):
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.employee_id = employee_id
        self.state = state
        self.source = source

    @property
    def duration(self) -> float:
        return max(0.0, (self.end_ts - self.start_ts).total_seconds())


def _window_seconds(schedule: WorkSchedule) -> float:
    """Length in seconds of the expected working window excluding lunch."""
    work_end = schedule.end_min - schedule.start_min
    return float(max(0, work_end - _overlap_len(schedule))) * 60.0


def _overlap_len(schedule: WorkSchedule) -> int:
    s0, e0 = schedule.lunch_min
    # fraction of lunch that falls inside [start, end]
    lo = max(schedule.start_min, s0)
    hi = min(schedule.end_min, e0)
    return max(0, hi - lo)


def _day_bounds(day: date) -> tuple[datetime, datetime]:
    return datetime(day.year, day.month, day.day), \
           datetime(day.year, day.month, day.day) + timedelta(days=1)


def split_interval_by_day(start_ts: datetime, end_ts: datetime,
                          employee_id: str, state: str, source: str = "") -> list[Interval]:
    """Split an [start_ts, end_ts] span across calendar-day boundaries.

    e.g. 23:59:30 -> 00:01:30 becomes two intervals: 30 sec on day 1 and
    90 sec on day 2.  Returns one :class:`Interval` per touched day.
    """
    result: list[Interval] = []
    cur = start_ts
    while cur < end_ts:
        day_end = datetime(cur.year, cur.month, cur.day) + timedelta(days=1)
        seg_end = min(end_ts, day_end)
        if seg_end > cur:
            result.append(Interval(cur, seg_end, employee_id, state, source))
        cur = seg_end
    return result


def compute_day_metrics(employee_id: str, intervals: list[Interval],
                        day: date, schedule: WorkSchedule) -> dict:
    """Aggregate one employee's intervals into a metrics dict for ``day``.

    Parameters
    ----------
    employee_id : str
        The recognised employee id.
    intervals : list[Interval]
        Intervals overlapping ``day`` (already filtered to this employee).
    day : date
        The calendar day being scored.
    schedule : WorkSchedule
        Expected working window.

    Returns
    -------
    dict with keys:
        employee_id, day, expected_working_seconds, in_time, out_time,
        productive_seconds, phone_seconds, away_seconds, unobserved_seconds,
        accounted_seconds, active_hours, phone_mins, away_mins, total_hours,
        productive_pct, status
    """
    productive = 0.0
    phone = 0.0
    away = 0.0
    in_time: datetime | None = None
    out_time: datetime | None = None

    for it in intervals:
        if it.state == "ACTIVE":
            productive += _seconds_in_window_impl(it, day, schedule, "ACTIVE")
        elif it.state == "ON_PHONE":
            phone += _seconds_in_window_impl(it, day, schedule, "ON_PHONE")
        elif it.state == "AWAY":
            away += _seconds_in_window_impl(it, day, schedule, "AWAY")

    # in/out time = first/last observation within the calendar day
    w_start, w_end = _day_bounds(day)
    for it in intervals:
        obs_s = max(it.start_ts, w_start)
        obs_e = min(it.end_ts, w_end)
        if obs_e > obs_s:
            if in_time is None or obs_s < in_time:
                in_time = obs_s
            if out_time is None or obs_e > out_time:
                out_time = obs_e

    observed_accounted = productive + phone + away
    expected = _window_seconds(schedule)
    unobserved = max(0.0, expected - observed_accounted)

    pct = round(productive / max(1.0, observed_accounted) * 100.0, 1) \
        if observed_accounted > 1e-6 else None

    return {
        "employee_id": employee_id,
        "day": day,
        "expected_working_seconds": round(expected, 2),
        "grace_period_min": schedule.grace_period_min,
        "break_minutes": schedule.break_minutes,
        "in_time": in_time.strftime("%Y-%m-%d %H:%M:%S") if in_time else "",
        "out_time": out_time.strftime("%Y-%m-%d %H:%M:%S") if out_time else "",
        "productive_seconds": round(productive, 2),
        "phone_seconds": round(phone, 2),
        "away_seconds": round(away, 2),
        "unobserved_seconds": round(unobserved, 2),
        "accounted_seconds": round(observed_accounted, 2),
        "active_hours": round(productive / 3600.0, 2),
        "phone_mins": round(phone / 60.0, 1),
        "away_mins": round(away / 60.0, 1),
        "total_hours": round((productive + phone) / 3600.0, 2),
        "productive_pct": pct,
        "status": "NOT OBSERVED" if observed_accounted <= 1e-6 else "OBSERVED",
    }


def _seconds_in_window_impl(it: Interval, day: date, schedule: WorkSchedule,
                            state: str) -> float:
    """Portion of ``it`` counted for ``state`` within the working window,
    excluding the (unpaid) lunch window for AWAY."""
    w_start, _ = _day_bounds(day)
    win_s = w_start + timedelta(minutes=schedule.start_min)
    win_e = w_start + timedelta(minutes=schedule.end_min)
    s = max(it.start_ts, win_s)
    e = min(it.end_ts, win_e)
    if e <= s:
        return 0.0
    dur = (e - s).total_seconds()
    if state == "AWAY" and schedule.lunch_min:
        l_start = win_s + timedelta(minutes=schedule.lunch_min[0] - schedule.start_min)
        l_end = win_s + timedelta(minutes=schedule.lunch_min[1] - schedule.start_min)
        ls = max(s, l_start)
        le = min(e, l_end)
        if le > ls:
            dur -= (le - ls).total_seconds()
    return max(0.0, dur)


# Backward/utility API ----------------------------------------------------

def from_rows(rows: list[tuple], employee_id: str | None = None) -> list[Interval]:
    """Build :class:`Interval` objects from raw DB rows.

    ``rows`` are ``(timestamp, employee_id, state, duration_seconds, source)``.
    Each row is a *closed* interval: duration counted from ``timestamp``.
    Intervals across midnight are split into per-day pieces.
    """
    out: list[Interval] = []
    for row in rows:
        ts_str, emp, state, dur, source = row
        try:
            start_ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            continue
        if employee_id is not None and emp != employee_id:
            continue
        end_ts = start_ts + timedelta(seconds=float(dur or 0.0))
        out.extend(split_interval_by_day(start_ts, end_ts, emp, state, source or ""))
    return out
