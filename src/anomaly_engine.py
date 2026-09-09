"""Explainable anomaly engine + behavioral baselines (Phase 33, B1/B2/B3).

A **purely advisory** intelligence layer.  It never accuses, never judges a
person, and never feeds productivity.  It detects two explainable operational
patterns:

* ``UNUSUALLY_HIGH_OCCUPANCY``  -- a zone/camera's person count is well above
  its historical baseline (B1).
* ``REPEATED_PATTERN_DETECTED`` -- the same event repeatedly recurs at the same
  place/time (B3) -- reported as a pattern, never as "malicious behaviour".

Baselines (B2) are computed per ``(scope, scope_value, metric)`` from observed
samples.  Until a zone has enough samples the status is ``INSUFFICIENT_DATA``
and no anomaly is reported -- we refuse to fabricate anomalies from thin data.

Every anomaly carries a human-readable ``reason`` (components + thresholds) so
operators can verify it.  Nothing here is an accusation.
"""

from __future__ import annotations

import json
import logging
import statistics
import time

from src import database as db

logger = logging.getLogger("cctv.anomaly")

STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"
STATUS_READY = "READY"

MIN_SAMPLES_FOR_BASELINE = 5
ZSCORE_THRESHOLD = 3.0          # std-deviations above mean -> anomalous
PATTERN_MIN_OCCURRENCES = 4     # same zone+event repeats -> repeated pattern
PATTERN_WINDOW_DAYS = 3

# Module-level event types we consider "behavioural" for baselines (person count
# related).  Lifecycle/tamper events are excluded (they are not behaviour).
_BEHAVIOUR_EVENT_TYPES = {
    "UNKNOWN_PRESENCE", "INTRUSION", "AFTER_HOURS_ACTIVITY",
    "LOITERING_SUSPECTED", "UNUSUAL_OCCUPANCY",
}


class BaselineTracker:
    """Builds and reads behavioural baselines for occupancy metrics."""

    def __init__(self, conn, *, min_samples: int = MIN_SAMPLES_FOR_BASELINE):
        self._conn = conn
        self._min_samples = min_samples

    def observed(self, scope: str, scope_value: str, metric: str, value: float) -> float:
        """Accumulate one sample into a running baseline (Welford-ish via
        stored mean/std/sample_count).  Returns current status.
        """
        base = db.get_baseline(self._conn, scope=scope, scope_value=scope_value,
                               metric=metric)
        if base:
            n = int(base.get("sample_count") or 0)
            mean = float(base.get("mean") or 0.0)
            old_m2 = (float(base.get("std") or 0.0) ** 2) * n
            n1 = n + 1
            delta = value - mean
            new_mean = mean + delta / n1
            new_m2 = old_m2 + delta * (value - new_mean)
            std = (new_m2 / n1) ** 0.5 if n1 > 1 else 0.0
            status = STATUS_READY if n1 >= self._min_samples else STATUS_INSUFFICIENT
        else:
            n1, new_mean, std, status = 1, value, 0.0, STATUS_INSUFFICIENT
        db.upsert_baseline(self._conn, scope=scope, scope_value=scope_value,
                           metric=metric, mean=new_mean, std=std,
                           sample_count=n1, status=status)
        return status

    def record_occupancy(self, zone: str, camera: str, count: int) -> None:
        self.observed("zone", zone, "occupancy", float(count))
        self.observed("camera", camera, "occupancy", float(count))

    def baseline(self, scope: str, scope_value: str, metric: str) -> dict | None:
        return db.get_baseline(self._conn, scope=scope, scope_value=scope_value,
                               metric=metric)

    def is_anomalous(self, scope: str, scope_value: str, metric: str,
                     value: float) -> bool:
        base = self.baseline(scope, scope_value, metric)
        if not base or base.get("status") != STATUS_READY:
            return False
        mean = float(base.get("mean") or 0.0)
        std = float(base.get("std") or 0.0)
        if std <= 0:
            return False
        return (value - mean) / std >= ZSCORE_THRESHOLD and value > mean

    # ------------------------------------------------------------------
    # Phase 34 -- adaptive time-of-day & day-of-week baselines (P1)
    # ------------------------------------------------------------------
    @staticmethod
    def _hour_metric(hour: int) -> str:
        return f"occupancy_h{hour:02d}"

    @staticmethod
    def _dow_metric(weekday: int) -> str:
        return f"occupancy_dow{weekday}"

    def record_occupancy_time(self, zone: str, camera: str, count: int,
                              timestamp=None) -> None:
        """Record occupancy into hour-of-day and day-of-week baselines.

        Uses the existing ``(scope, scope_value, metric)`` baseline store with
        the time dimension encoded into ``metric`` (no schema change).  When a
        timestamps is supplied it is parsed; otherwise the wall clock is used.
        """
        dt = self._as_datetime(timestamp)
        self.observed("zone", zone, self._hour_metric(dt.hour), float(count))
        self.observed("zone", zone, self._dow_metric(dt.weekday()), float(count))

    def time_baseline(self, zone: str, count: int,
                      timestamp=None) -> dict:
        """Return the READY hour-of-day baseline for ``zone`` or an
        ``INSUFFICIENT_DATA`` marker (never an unreliable baseline)."""
        dt = self._as_datetime(timestamp)
        base = self.baseline("zone", zone, self._hour_metric(dt.hour))
        if not base or base.get("status") != STATUS_READY:
            return {"status": STATUS_INSUFFICIENT}
        return base

    @staticmethod
    def _as_datetime(timestamp):
        from datetime import datetime
        if timestamp is None:
            return datetime.now()
        try:
            return datetime.strptime(str(timestamp)[:19], "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return datetime.now()


class AnomalyEngine:
    """Detects, stores and explains advisory anomalies."""

    def __init__(self, conn, *, baselines: BaselineTracker | None = None):
        self._conn = conn
        self._baselines = baselines or BaselineTracker(conn)

    # -- occupancy anomaly (B1) ----------------------------------------
    def evaluate_occupancy(self, zone: str, camera: str, count: int) -> dict | None:
        """Record an occupancy sample; return an anomaly dict when unusual.

        The anomaly is evaluated against the **pre-sample** baseline so the
        current observation cannot inflate the very baseline used to judge it.
        Returns a dict with ``anomaly_type``, ``reason``, ``data`` or None.
        Status is advisory; never accusatory.
        """
        base = self._baselines.baseline("zone", zone, "occupancy")
        anomaly = None
        if count > 0 and base and base.get("status") == STATUS_READY:
            mean = float(base.get("mean") or 0.0)
            std = float(base.get("std") or 0.0)
            if std > 0 and count > mean:
                z = (count - mean) / std
                if z >= ZSCORE_THRESHOLD:
                    reason = (
                        f"zone={zone} camera={camera} occupancy {count} is {z:.1f} "
                        f"std-dev above baseline mean {mean:.1f} "
                        f"(baseline n={base['sample_count']})"
                    )
                    self._store("UNUSUALLY_HIGH_OCCUPANCY", "zone", zone, "MEDIUM",
                                reason, data={"camera": camera, "count": count,
                                              "mean": round(mean, 2), "z": round(z, 2)})
                    anomaly = {"anomaly_type": "UNUSUALLY_HIGH_OCCUPANCY",
                               "scope": "zone", "scope_value": zone,
                               "severity": "MEDIUM", "reason": reason}
        # Record the sample AFTER evaluation (never pollutes the judging baseline).
        self._baselines.record_occupancy(zone, camera, count)
        return anomaly

    # -- time-of-day occupancy anomaly (Phase 34, P1) ------------------
    def evaluate_occupancy_time(self, zone: str, camera: str, count: int,
                                timestamp=None) -> dict | None:
        """Advisory occupancy anomaly against the hour-of-day baseline.

        Returns ``None`` (or an ``INSUFFICIENT_DATA``-typed dict marker) while
        the time-bucketed baseline has too few samples -- we refuse to raise an
        anomaly from an unreliable baseline.  When a genuine spike is found it
        is returned as an explainable anomaly dict with the ``reason`` included.
        """
        base = self._baselines.time_baseline(zone, count, timestamp)
        if base.get("status") != STATUS_READY:
            dt = self._baselines._as_datetime(timestamp)
            reason = (
                f"insufficient {self._baselines._hour_metric(dt.hour)} "
                f"baseline samples for zone={zone} "
                f"(need {self._baselines._min_samples})"
            )
            self._store("TIME_OCCUPANCY_INSUFFICIENT", "zone", zone, "INFO",
                        reason, data={"hour": dt.hour, "count": count})
            return {"anomaly_type": "TIME_OCCUPANCY_INSUFFICIENT",
                    "scope": "zone", "scope_value": zone,
                    "status": STATUS_INSUFFICIENT, "reason": reason}
        mean = float(base.get("mean") or 0.0)
        std = float(base.get("std") or 0.0)
        dt = self._baselines._as_datetime(timestamp)
        if std <= 0 or not (count > mean):
            # Record only; no anomaly.
            self._baselines.record_occupancy_time(zone, camera, count, timestamp)
            return None
        z = (count - mean) / std
        if z < ZSCORE_THRESHOLD:
            self._baselines.record_occupancy_time(zone, camera, count, timestamp)
            return None
        reason = (
            f"zone={zone} camera={camera} occupancy {count} at hour {dt.hour} "
            f"is {z:.1f} std-dev above that hour's baseline mean {mean:.1f} "
            f"(n={base['sample_count']})"
        )
        self._store("TIME_OCCUPANCY_ANOMALY", "zone", zone, "MEDIUM", reason,
                    data={"hour": dt.hour, "count": count, "mean": round(mean, 2),
                          "z": round(z, 2)})
        self._baselines.record_occupancy_time(zone, camera, count, timestamp)
        return {"anomaly_type": "TIME_OCCUPANCY_ANOMALY",
                "scope": "zone", "scope_value": zone, "severity": "MEDIUM",
                "reason": reason}

    # -- recurring/pattern detection (B3) ------------------------------
    def detect_recurring_patterns(self, *, window_days: int = PATTERN_WINDOW_DAYS,
                                  min_occurrences: int = PATTERN_MIN_OCCURRENCES,
                                  actor: str = "system") -> list[dict]:
        """Detect events that repeatedly recur at the same (zone,event).

        Reported as a *pattern* (e.g. repeated after-hours activity on a zone),
        explicitly advisory -- never 'malicious' by default.  Returns anomalies.
        """
        from datetime import date, timedelta, datetime
        since = (date.today() - timedelta(days=window_days)).strftime("%Y-%m-%d")
        found: list[dict] = []
        # Group behavioural events by (zone, event_type) over the window.
        grouped: dict[tuple, list] = {}
        for ev in self._all_events_since(since):
            et = ev.get("event_type")
            zone = ev.get("zone") or "UNKNOWN_ZONE"
            if et not in _BEHAVIOUR_EVENT_TYPES:
                continue
            key = (zone, et)
            grouped.setdefault(key, []).append(ev)
        for (zone, et), events in grouped.items():
            if len(events) < min_occurrences:
                continue
            cameras = sorted({e.get("camera") for e in events if e.get("camera")})
            reason = (
                f"{et} recurred {len(events)}x on zone={zone} cameras={cameras} "
                f"within {window_days}d (pattern, advisory)"
            )
            self._store("REPEATED_PATTERN_DETECTED", "zone", zone, "LOW",
                        reason, data={"event_type": et, "count": len(events),
                                      "cameras": cameras})
            found.append({"anomaly_type": "REPEATED_PATTERN_DETECTED",
                          "scope": "zone", "scope_value": zone,
                          "severity": "LOW", "reason": reason})
        if found:
            db.audit(self._conn, "anomaly.patterns", actor=actor,
                     resource="campus", detail=f"{len(found)} patterns detected")
        return found

    def _all_events_since(self, since_day: str):
        rows = self._conn.execute(
            "SELECT event_type, zone, camera FROM security_events "
            "WHERE date >= ?", (since_day,)).fetchall()
        cols = ["event_type", "zone", "camera"]
        return [dict(zip(cols, r)) for r in rows]

    # -- shared store ---------------------------------------------------
    def _store(self, anomaly_type: str, scope: str, scope_value: str,
               severity: str, reason: str, data: dict | None) -> None:
        db.insert_anomaly(
            self._conn, anomaly_type=anomaly_type, scope=scope,
            scope_value=scope_value, severity=severity, reason=reason,
            data=json.dumps(data) if data else None,
        )

    def list(self, limit: int = 100) -> list[dict]:
        return db.list_anomalies(self._conn, limit=limit)

    def snapshot(self) -> dict:
        anomalies = db.list_anomalies(self._conn, limit=500)
        highest = max((a.get("severity") or "INFO" for a in anomalies),
                      default="INFO")
        return {
            "count": len(anomalies),
            "by_type": _count_by(anomalies, "anomaly_type"),
            "by_severity": _count_by(anomalies, "severity"),
            "highest_severity": highest,
        }


def _count_by(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r.get(key) or "UNKNOWN"] = out.get(r.get(key) or "UNKNOWN", 0) + 1
    return out
