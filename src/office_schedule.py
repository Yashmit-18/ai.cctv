"""Office / monitoring-window scheduling layer (Phase 59b).

Purpose
-------
Decide *WHEN* the office is in a monitoring-relevant window.  This engine is
purely administrative context around the CCTV_OFFICE_* config knobs -- it
answers questions like "is the office open?" / "is it lunch?" / "is this a
monitoring period?" from a wall clock and a timezone.

Scope & safety
--------------
* The schedule NEVER relabels anyone Dutch and NEVER force-vacates a desk.
  Knowing lunch is 14:00–14:35 tells the *context* layer what lens to use; it
  changes no identity, no occupancy, and no employee state.
* The engine is tz-safe (Asia/Kolkata by default) and validates its own
  knobs at construction (24-hour "HH:MM" strings, sane ordering).
* Legacy ``WORK_SCHEDULE`` (Phase 2B) is untouched: this engine reads the
  additive ``CCTV_OFFICE_*`` knobs and never mutates the legacy constants.

Windows provided
----------------
* ``is_office_hours(ts)``       -- inside [CCTV_OFFICE_START, CCTV_OFFICE_END]
* ``is_lunch_break(ts)``        -- inside [CCTV_LUNCH_START, CCTV_LUNCH_END]
* ``is_monitoring_period(ts)``  -- office hours MINUS the lunch window
* ``is_scheduled_break(ts)``    -- inside the (optional) break window(s)
"""

from __future__ import annotations

import datetime as _dt
import logging
import zoneinfo

import config  # noqa: E402  (config lives at PROJECT ROOT)

logger = logging.getLogger(__name__)


def parse_hhmm(value: str, *, knob: str) -> tuple[int, int]:
    """Parse a ``HH:MM`` (24h) string into ``(hour, minute)``.

    Raises ``ValueError`` for non-24h / non-numeric input so a single bad
    knob fails loudly at construction instead of silently at runtime."""
    text = str(value or "").strip()
    try:
        h, m = text.split(":")
        hour = int(h)
        minute = int(m)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"{knob} must be a 24h 'HH:MM' string, got {value!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(
            f"{knob} out of range (0..23:0..59), got {value!r}")
    return hour, minute


class OfficeSchedule:
    """Tz-aware office / monitoring window engine.

    Parameters
    ----------
    start, end, lunch_start, lunch_end
        Optional 24h ``HH:MM`` override strings.  ``None`` falls back to the
        additive ``CCTV_OFFICE_*`` config knobs (Phase 59b).
    tz
        IANA timezone name (default ``Asia/Kolkata``).  Falls back to UTC
        only when the platform lacks the zone.
    break_windows
        Optional list of ``(start, end)`` HH:MM windows considered scheduled
        breaks (in addition to lunch).
    """

    def __init__(self, *, start: str | None = None, end: str | None = None,
                 lunch_start: str | None = None, lunch_end: str | None = None,
                 tz: str | None = None,
                 break_windows: list[tuple[str, str]] | None = None):
        self.start = (start if start is not None
                      else getattr(config, "CCTV_OFFICE_START", "10:00"))
        self.end = (end if end is not None
                    else getattr(config, "CCTV_OFFICE_END", "18:30"))
        self.lunch_start = (lunch_start if lunch_start is not None
                            else getattr(config, "CCTV_LUNCH_START", "14:00"))
        self.lunch_end = (lunch_end if lunch_end is not None
                          else getattr(config, "CCTV_LUNCH_END", "14:35"))
        name = tz or getattr(config, "CCTV_OFFICE_TZ", "Asia/Kolkata")
        try:
            self.tz = zoneinfo.ZoneInfo(name)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError):
            logger.warning("Office tz %r unavailable; falling back to UTC.", name)
            self.tz = _dt.timezone.utc

        self._hh = {
            "start": parse_hhmm(self.start, knob="CCTV_OFFICE_START"),
            "end": parse_hhmm(self.end, knob="CCTV_OFFICE_END"),
            "lunch_start": parse_hhmm(self.lunch_start, knob="CCTV_LUNCH_START"),
            "lunch_end": parse_hhmm(self.lunch_end, knob="CCTV_LUNCH_END"),
        }
        _assert_ordered("office hours", self._hh["start"], self._hh["end"])
        _assert_ordered("lunch", self._hh["lunch_start"], self._hh["lunch_end"])
        self.break_windows = [
            (parse_hhmm(s, knob="break start"), parse_hhmm(e, knob="break end"))
            for s, e in (break_windows or [])
        ]
        for s, e in self.break_windows:
            _assert_ordered("break", s, e)

    # ------------------------------------------------------------------
    # Time math
    # ------------------------------------------------------------------
    def _minute_of_day(self, ts: float | _dt.datetime | None = None) -> int:
        dt = self._as_datetime(ts)
        return dt.hour * 60 + dt.minute

    def _as_datetime(self, ts):
        if ts is None:
            local = _dt.datetime.now(self.tz)
            return local
        if isinstance(ts, _dt.datetime):
            if ts.tzinfo is None:
                ts = ts.astimezone(self.tz) if self.tz else ts
            return ts.astimezone(self.tz)
        return _dt.datetime.fromtimestamp(ts, self.tz)

    # ------------------------------------------------------------------
    # Public windows
    # ------------------------------------------------------------------
    def is_office_hours(self, ts=None) -> bool:
        m = self._minute_of_day(ts)
        return self._hh["start"][0] * 60 + self._hh["start"][1] <= m < \
            self._hh["end"][0] * 60 + self._hh["end"][1]

    def is_lunch_break(self, ts=None) -> bool:
        m = self._minute_of_day(ts)
        return self._hh["lunch_start"][0] * 60 + self._hh["lunch_start"][1] <= \
            m < self._hh["lunch_end"][0] * 60 + self._hh["lunch_end"][1]

    def is_monitoring_period(self, ts=None) -> bool:
        """Office hours outside the lunch window (the active monitoring
        window for desk/chair/churn context)."""
        return self.is_office_hours(ts) and not self.is_lunch_break(ts)

    def is_scheduled_break(self, ts=None) -> bool:
        """Inside any configured break window (lunch is treated separately)."""
        m = self._minute_of_day(ts)
        for s, e in self.break_windows:
            if s[0] * 60 + s[1] <= m < e[0] * 60 + e[1]:
                return True
        return False

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Sanitized, serializable snapshot of the schedule knobs."""
        return {
            "tz": str(self.tz),
            "start": self.start,
            "end": self.end,
            "lunch_start": self.lunch_start,
            "lunch_end": self.lunch_end,
            "break_windows": [f"{_fmt(s)}-{_fmt(e)}"
                              for s, e in self.break_windows],
        }


def _assert_ordered(label: str, a: tuple[int, int], b: tuple[int, int]) -> None:
    av, bv = a[0] * 60 + a[1], b[0] * 60 + b[1]
    if bv <= av:
        raise ValueError(
            f"{label} window ends ({b}) before it starts ({a})")


def _fmt(hhmm: tuple[int, int]) -> str:
    return "%02d:%02d" % hhmm
