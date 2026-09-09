"""Temporal intelligence (Phase 33, A1).

Assigns an explicit, deterministic temporal lifecycle state to each incident so
the investigation view and dashboards never have to guess *what stage* an
incident is in:

* ``FIRST_SEEN``  -- incident just created (1 occurrence) and still open.
* ``CONTINUING``  -- still active but not yet flagged as repeating.
* ``REPEATED``    -- the same incident has recurred several times (high
                     occurrence count / many events over time).
* ``ESCALATED``   -- sustained HIGH/CRITICAL severity and/or repeated escalation.
* ``ENDED``       -- resolved/dismissed by an operator, or gone silent beyond
                     the silence window.

The temporal state is **advisory metadata about an incident**, never about a
person.  It is a product of the incident's own event history in
``security_events``/``incident_events`` plus its lifecycle status.

Invariants
----------
* ``ENDED`` is terminal: once set it is never un-set by the recommender.
* ``ESCALATED`` requires sustained high severity, never a single event.
* The module never turns an incident into an accusation and never touches
  employee/productivity data.
"""

from __future__ import annotations

import logging

from src import database as db

logger = logging.getLogger("cctv.temporal")

T_FIRST_SEEN = "FIRST_SEEN"
T_CONTINUING = "CONTINUING"
T_REPEATED = "REPEATED"
T_ESCALATED = "ESCALATED"
T_ENDED = "ENDED"

TEMPORAL_STATES = (T_FIRST_SEEN, T_CONTINUING, T_REPEATED, T_ESCALATED, T_ENDED)

# Thresholds (tunable, transparent).
REPEAT_THRESHOLD = 5          # occurrences -> REPEATED
ESCALATE_SEVERITIES = ("HIGH", "CRITICAL")
ESCALATE_COUNT = 3            # high-sev events -> ESCALATED (sustained)
SILENCE_DAYS = 1              # no activity -> recommend ENDED


class TemporalIntelligence:
    """Recommends and persists the temporal state for incidents."""

    def __init__(self, conn, *, repeat_threshold: int = REPEAT_THRESHOLD,
                 escalate_count: int = ESCALATE_COUNT,
                 silence_days: int = SILENCE_DAYS):
        self._conn = conn
        self._repeat_threshold = repeat_threshold
        self._escalate_count = escalate_count
        self._silence_days = silence_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh_all(self, actor: str = "system") -> list[dict]:
        """Recompute temporal state for every incident (open + recent).

        Returns list of ``{incident_id, from, to}`` transitions applied.
        Used by the main tick and the dashboard.  Safe to call frequently.
        """
        applied: list[dict] = []
        for inc in db.list_incidents(self._conn, limit=1000):
            new_state = self.recommend(inc)
            cur = inc.get("temporal_state") or T_FIRST_SEEN
            if new_state != cur:
                # REFRESH_ALL must not un-persist an operator's ENDED if the
                # incident is genuinely terminal.
                if cur == T_ENDED and new_state != T_ENDED:
                    continue
                db.update_incident(self._conn, inc["incident_id"],
                                   temporal_state=new_state)
                db.audit(self._conn, "incident.temporal", actor=actor,
                         resource=inc["incident_id"], detail=f"{cur}->{new_state}")
                applied.append({
                    "incident_id": inc["incident_id"],
                    "from": cur, "to": new_state,
                })
        return applied

    def recommend(self, inc: dict) -> str:
        """Compute the temporal state for a single incident row."""
        if not inc:
            return T_FIRST_SEEN
        status = inc.get("status")
        # Terminal states.
        if status in ("RESOLVED", "DISMISSED"):
            return T_ENDED

        occurrences = int(inc.get("occurrences") or 0)
        # Count high-severity events for this incident.
        high_count = self._high_severity_event_count(inc["incident_id"])

        if occurrences >= self._repeat_threshold:
            # Repeated AND sustained-high -> escalated if enough high events.
            if high_count >= self._escalate_count:
                return T_ESCALATED
            return T_REPEATED

        if high_count >= self._escalate_count:
            return T_ESCALATED

        if self._is_silent(inc):
            return T_ENDED

        if occurrences <= 1:
            return T_FIRST_SEEN
        return T_CONTINUING

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _high_severity_event_count(self, incident_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM security_events WHERE incident_id = ? "
            "AND severity IN ('HIGH','CRITICAL')", (incident_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def _is_silent(self, inc: dict) -> bool:
        if self._silence_days <= 0:
            return False
        last_seen = inc.get("last_seen") or inc.get("first_seen") or ""
        if not last_seen:
            return False
        from datetime import datetime, timedelta
        try:
            last = datetime.strptime(str(last_seen)[:19], "%Y-%m-%d %H:%M:%S")
            return (datetime.now() - last) >= timedelta(days=self._silence_days)
        except (ValueError, TypeError):
            return False
