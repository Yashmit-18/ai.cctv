"""Investigation intelligence (Phase 34).

A read-only, evidence-backed investigation layer that lets an operator
reconstruct an incident and a person's movement **without** manually hunting
through unrelated tables.  Three self-contained capabilities:

* `IncidentReconstruction` -- a temporal pre/post window around an incident
  (default T-30s .. T+30s) merging the incident's events, alerts and evidence.
  When an incident has no pre-event evidence the result explicitly reports
  ``PRE_EVENT_EVIDENCE_UNAVAILABLE`` (never fabricates a pre-event).
* `PersonContinuity` -- resolves, from the security event history, a single
  identity's movement: first_seen / last_seen / dwell duration / zone
  transitions / cross-camera path.  It never upgrades ``Unknown`` to an
  employee and never invents identity where confidence is insufficient.
* `EventContextRollup` -- combines a person-threat incident's observations
  (``UNKNOWN_PRESENCE + LOITERING + INTRUSION + AFTER_HOURS ...``) into one
  higher-context summary labelled **CORRELATED SECURITY INCIDENT /
  REQUIRES HUMAN REVIEW** -- never "criminal activity" and never a judgement of
  intent.

Non-negotiable invariants
-------------------------
* Read-only: this module **never mutates** incidents, events, alerts, evidence,
  or any productivity/employee data.
* Evidence-backed: every reconstruction/context statement is derived only from
  actually-recorded rows, with the recorded basis included.
* No fabrication: identity that is genuinely unknown stays ``UNKNOWN``;
  pre-event evidence that does not exist is reported as unavailable.
* Advisory: every output is decision-support for a human reviewer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from src import database as db
from src.domain import INTRUSION, AFTER_HOURS_ACTIVITY, LOITERING_SUSPECTED, \
    UNKNOWN_PRESENCE, UNKNOWN_ID

logger = logging.getLogger("cctv.investigation")

# Sentinel for "there is no pre-event evidence" (never fabricated).
PRE_EVENT_EVIDENCE_UNAVAILABLE = "PRE_EVENT_EVIDENCE_UNAVAILABLE"

# Person-threat observations that can combine into a higher-context incident.
_PERSON_THREAT_TYPES = (UNKNOWN_PRESENCE, INTRUSION, AFTER_HOURS_ACTIVITY,
                        LOITERING_SUSPECTED)

# The higher-context label used when >= 2 distinct person-threat observations
# co-occur for the same incident (explicitly non-accusatory).
CORRELATED_SECURITY_INCIDENT = "CORRELATED_SECURITY_INCIDENT"
LABEL_REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"


def _parse(ts: str | None):
    """Parse 'YYYY-MM-DD HH:MM:SS' to datetime, else None."""
    if not ts:
        return None
    try:
        return datetime.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def _sql_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class IncidentReconstruction:
    """Reconstruct a temporal pre/post window around an incident."""

    def __init__(self, conn):
        self._conn = conn

    def reconstruct(self, incident_id: str, *, pre_sec: int = 30,
                    post_sec: int = 30) -> dict:
        """Return the reconstructed incident window (read-only).

        Windowing is done on the incident's own event-driven rows only; it
        never triggers recording.  ``pre_sec``/``post_sec`` bound how far we
        look before the incident's first_seen and after its last_seen.
        """
        inc = db.get_incident(self._conn, incident_id)
        if not inc:
            return {"incident_id": incident_id, "found": False}

        first = _parse(inc.get("first_seen"))
        last = _parse(inc.get("last_seen") or inc.get("first_seen"))
        rows: list[dict] = []

        # Merge this incident's events, alerts and evidence into a single
        # chronological stream so the operator can re-read the sequence.
        seen: set[tuple] = set()
        for ev in db.query_security_events(self._conn, incident_id=incident_id,
                                           limit=10000):
            ts = ev.get("timestamp")
            if ts not in seen:
                seen.add((ts, "EVENT", ev.get("event_type")))
                rows.append({
                    "ts": ts, "kind": "EVENT", "label": ev.get("event_type"),
                    "detail": f"camera={ev.get('camera','-')} "
                              f"zone={ev.get('zone','-')} sev={ev.get('severity','-')}",
                })
        for al in db.query_alerts_filtered(self._conn, incident_id=incident_id,
                                           limit=1000):
            rows.append({
                "ts": al.get("created_at"), "kind": "ALERT",
                "label": al.get("rule_name") or "alert",
                "detail": f"{al.get('status','-')} {al.get('channel','-')}",
            })
        for ev in db.list_evidence_files(self._conn, incident_id=incident_id):
            rows.append({
                "ts": ev.get("captured_at"), "kind": "EVIDENCE",
                "label": ev.get("capture_type") or ev.get("kind"),
                "detail": f"camera={ev.get('camera','-')}",
            })

        # Restrict to the T-pre .. T+post window around the incident window.
        window_rows: list[dict] = []
        if first and last:
            lo = first - timedelta(seconds=pre_sec)
            hi = last + timedelta(seconds=post_sec)
            for r in rows:
                rt = _parse(r["ts"])
                if rt is not None and lo <= rt <= hi:
                    window_rows.append(r)
        else:
            window_rows = rows

        window_rows.sort(key=lambda r: (r.get("ts") or "", r["kind"], r["label"]))

        pre_ev = [r for r in window_rows
                  if first and _parse(r["ts"]) is not None
                  and r["kind"] == "EVENT" and _parse(r["ts"]) < first]
        pre_evidence = self._evidence_present(incident_id, first, pre_sec)

        return {
            "incident_id": incident_id,
            "found": True,
            "event_type": inc.get("event_type"),
            "severity": inc.get("severity"),
            "status": inc.get("status"),
            "temporal_state": inc.get("temporal_state"),
            "review_state": inc.get("review_state"),
            "camera": inc.get("camera"),
            "zone": inc.get("zone"),
            "first_seen": inc.get("first_seen"),
            "last_seen": inc.get("last_seen"),
            "pre_sec": pre_sec,
            "post_sec": post_sec,
            "window_rows": window_rows,
            "pre_event_events": pre_ev,
            "pre_event_evidence": pre_evidence,
            "post_event_events": [r for r in window_rows
                                  if last and _parse(r["ts"]) is not None
                                  and r["kind"] == "EVENT" and _parse(r["ts"]) > last],
            "post_event_evidence": self._evidence_present(incident_id, last,
                                                          post_sec, after=True),
        }

    def _evidence_present(self, incident_id: str, anchor, span: int,
                          *, after: bool = False) -> str:
        """Report whether event-driven evidence exists around ``anchor``."""
        if not anchor:
            return PRE_EVENT_EVIDENCE_UNAVAILABLE
        bound = anchor + timedelta(seconds=span) if after \
            else anchor - timedelta(seconds=span)
        # Evidence is event-driven; check if any captured evidence timestamp
        # falls on the requested side of the anchor within the span.
        for ev in db.list_evidence_files(self._conn, incident_id=incident_id):
            et = _parse(ev.get("captured_at"))
            if et is None:
                continue
            if after and anchor <= et <= bound:
                return "EVIDENCE_PRESENT"
            if not after and bound <= et <= anchor:
                return "EVIDENCE_PRESENT"
        return PRE_EVENT_EVIDENCE_UNAVAILABLE


class PersonContinuity:
    """Resolve a single identity's movement from the event history."""

    def __init__(self, conn):
        self._conn = conn

    def resolve(self, identity: str, *, day: str | None = None,
                limit: int = 10000) -> dict:
        """Build a continuity record for ``identity`` (employee or Unknown).

        Never fabricates identity: if ``identity`` is the Unknown sentinel, the
        continuity is reported as Unknown and never linked to an employee.
        """
        query = ("SELECT timestamp, camera, zone, event_type, incident_id "
                 "FROM security_events WHERE employee_id = ?")
        params: list = [identity]
        if day:
            query += " AND date = ?"
            params.append(day)
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        events = [dict(zip(("timestamp", "camera", "zone", "event_type",
                             "incident_id"), r)) for r in rows]

        was_unknown = identity == UNKNOWN_ID
        first_seen = events[0]["timestamp"] if events else None
        last_seen = events[-1]["timestamp"] if events else None
        dwell = None
        if first_seen and last_seen:
            f = _parse(first_seen)
            l = _parse(last_seen)
            if f and l:
                dwell = round((l - f).total_seconds(), 1)

        # Zone transitions: distinct consecutive zones in event order.
        zone_seq: list[str] = []
        for ev in events:
            z = ev.get("zone")
            if z and (not zone_seq or zone_seq[-1] != z):
                zone_seq.append(z)

        # Cross-camera path: distinct consecutive cameras.
        cam_seq: list[str] = []
        for ev in events:
            c = ev.get("camera")
            if c and (not cam_seq or cam_seq[-1] != c):
                cam_seq.append(c)

        cameras = sorted({e.get("camera") for e in events if e.get("camera")})
        zones = sorted({e.get("zone") for e in events if e.get("zone")})

        return {
            "identity": identity,
            "unresolved": was_unknown,
            "status": "UNRESOLVED" if was_unknown else "RESOLVED_TO_EMPLOYEE",
            "first_seen": first_seen,
            "last_seen": last_seen,
            "dwell_seconds": dwell,
            "cameras": cameras,
            "zones": zones,
            "zone_transitions": zone_seq,
            "camera_path": cam_seq,
            "event_count": len(events),
            "incidents": sorted({e.get("incident_id") for e in events
                                 if e.get("incident_id")}),
        }


class EventContextRollup:
    """Combine an incident's observations into a higher-context summary."""

    def __init__(self, conn):
        self._conn = conn

    def rollup(self, incident_id: str) -> dict:
        """Return a higher-context summary for a person-threat incident.

        When >= 2 distinct person-threat observations are present for this
        incident the result is labelled ``CORRELATED_SECURITY_INCIDENT`` /
        ``REQUIRES_HUMAN_REVIEW``.  This is an explicit, explainable flag --
        NOT an accusation and NOT "criminal activity".
        """
        inc = db.get_incident(self._conn, incident_id)
        if not inc:
            return {"incident_id": incident_id, "found": False}

        obs_types = {inc.get("event_type")}
        for ev in db.query_security_events(self._conn, incident_id=incident_id,
                                           limit=10000):
            et = ev.get("event_type")
            if et in _PERSON_THREAT_TYPES:
                obs_types.add(et)

        present = sorted(t for t in _PERSON_THREAT_TYPES if t in obs_types)
        distinct = [t for t in present]
        is_correlated = len(distinct) >= 2

        factors = [
            f"observed::{t}" for t in distinct
        ]
        # After-hours context (evidence-backed from an AFTER_HOURS observation).
        if AFTER_HOURS_ACTIVITY in obs_types:
            factors.append("context::after_hours")
        # Unknown-identity context (never upgraded to employee).
        if UNKNOWN_PRESENCE in obs_types:
            factors.append("identity::UNKNOWN")

        label = f"{CORRELATED_SECURITY_INCIDENT} ({LABEL_REQUIRES_HUMAN_REVIEW})" \
            if is_correlated else "SINGLE_OBSERVATION"
        return {
            "incident_id": incident_id,
            "found": True,
            "label": label,
            "is_correlated_security_incident": is_correlated,
            "requires_human_review": True,
            "observations": distinct,
            "observation_count": len(distinct),
            "factors": factors,
            "evidence_basis": f"incident={inc.get('event_type')} + "
                              f"{len(distinct) - 1} related person-threat event(s)",
        }
