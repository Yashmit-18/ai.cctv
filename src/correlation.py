"""Event correlation engine (Phase 32).

Turns the flat event stream into **correlated incidents**.  Where the Phase 31
incident engine only groups identical ``(event_type, camera, zone)`` rows, the
correlator links *different but related* event types into one investigation
when they co-occur within a configurable temporal/spatial window.

Design invariants (Phase 32, non-negotiable):

* **Preservation** -- correlation never deletes or overwrites the underlying
  ``security_events``; every correlated incident keeps references to its events
  (``incident_events`` rows).
* **Explainability** -- every correlation carries a human-readable reason
  stored on the incident (``correlation_reason``).
* **Honesty** -- correlated events are *related observations*, never a claim
  about intent or guilt.  Terminology stays ``SUSPECTED``/``REQUIRES REVIEW``.
* **Camera health is a lifecycle, not a threat** -- ``CAMERA_OFFLINE`` +
  ``CAMERA_RECOVERED`` (and tamper suspected/cleared) are correlated as a
  single camera-health lifecycle, never folded into a person-threat incident.

Implemented rules
-----------------
* **Person-threat family** -- ``UNKNOWN_PRESENCE``, ``INTRUSION``,
  ``AFTER_HOURS_ACTIVITY``, ``LOITERING_SUSPECTED``: co-occur in time on the
  same camera (or same zone across cameras) and match on person identity
  (Unknown-to-Unknown, or same employee) -> one incident.
* **Camera offline lifecycle** -- ``CAMERA_OFFLINE`` -> ``CAMERA_RECOVERED``.
* **Camera tamper lifecycle** -- ``CAMERA_TAMPER_SUSPECTED`` ->
  ``CAMERA_TAMPER_CLEARED``.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from datetime import datetime

from src import database as db
from src.domain import (
    AFTER_HOURS_ACTIVITY,
    CAMERA_OFFLINE,
    CAMERA_RECOVERED,
    CAMERA_TAMPER_CLEARED,
    CAMERA_TAMPER_SUSPECTED,
    INTRUSION,
    LOITERING_SUSPECTED,
    UNKNOWN_ID,
    UNKNOWN_PRESENCE,
)
from src.incidents import IncidentEngine

logger = logging.getLogger("cctv.correlation")

# Cross-camera correlation marker: when a genuine cross-camera match cannot be
# established (no shared zone AND no admin-configured adjacency), correlation
# is honestly reported as unavailable rather than fabricated.
CROSS_CAMERA_CORRELATION_UNAVAILABLE = "CROSS_CAMERA_CORRELATION_UNAVAILABLE"

# ----------------------------------------------------------------------
# Correlation families (event types that belong to one investigation)
# ----------------------------------------------------------------------
PERSON_THREAT_FAMILY = (UNKNOWN_PRESENCE, INTRUSION, AFTER_HOURS_ACTIVITY,
                        LOITERING_SUSPECTED)
CAMERA_LIFECYCLE_PAIR = (CAMERA_OFFLINE, CAMERA_RECOVERED)
TAMPER_LIFECYCLE_PAIR = (CAMERA_TAMPER_SUSPECTED, CAMERA_TAMPER_CLEARED)

# LIFECYCLE event types are NEVER grouped into a person-threat incident.
_LIFECYCLE_TYPES = set(CAMERA_LIFECYCLE_PAIR) | set(TAMPER_LIFECYCLE_PAIR)


def _identity_key(employee_id: str | None) -> str:
    """Normalize an employee id for identity matching (Unknown preserved)."""
    if not employee_id or employee_id == UNKNOWN_ID:
        return UNKNOWN_ID
    return employee_id


def _timestamp_epoch(ts: str | None) -> float | None:
    """Best-effort parse of ``'YYYY-MM-DD HH:MM:SS'`` to epoch seconds."""
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()
    except (ValueError, TypeError):
        return None


class EventCorrelator:
    """Links new events to a recent compatible open incident.

    The correlator is **stateless in the DB sense** -- it inspects recent
    incidents and the in-memory recent-event window to decide, for a freshly
    persisted event, whether it belongs to an existing open incident.
    """

    def __init__(self, conn, *,
                 incident_engine: IncidentEngine | None = None,
                 window_sec: float = 300.0,
                 recent_limit: int = 200,
                 topology=None):
        self._conn = conn
        self._incidents = incident_engine or IncidentEngine(conn)
        self._window_sec = window_sec
        # optional admin-configured CameraTopology (A3); None => no adjacency info
        self._topology = topology
        # ring buffer of recent normalized events for cross-event lookups
        self._recent: collections.deque = collections.deque(maxlen=recent_limit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correlate(self, event: dict, event_id: int) -> dict | None:
        """Given a persisted event, decide if it belongs to an existing open
        incident.

        Returns a dict with the matching incident id + reason when a genuine
        correlation is found, else ``None``.  The caller decides whether to
        re-bind the event to that incident.
        """
        ev_type = event.get("event_type")
        camera = event.get("camera")
        zone = event.get("zone")
        emp = event.get("employee_id")
        ev_ts = event.get("timestamp")

        # Record in window for future correlations.
        self._add_to_window(event, event_id)

        # Lifecycle events (camera/tamper) have their own pairing rule.
        if ev_type in CAMERA_LIFECYCLE_PAIR or ev_type in TAMPER_LIFECYCLE_PAIR:
            return self._correlate_lifecycle(event, ev_type, camera, ev_ts)

        # Person-threat family: find an open compatible incident in window.
        if ev_type in PERSON_THREAT_FAMILY:
            return self._correlate_person_threat(
                ev_type, camera, zone, emp, ev_ts)

        return None

    # ------------------------------------------------------------------
    # Person-threat family
    # ------------------------------------------------------------------

    def _correlate_person_threat(self, ev_type: str, camera: str | None,
                                 zone: str | None, emp: str | None,
                                 ev_ts: str | None) -> dict | None:
        identity = _identity_key(emp)
        for inc in self._open_recent(PERSON_THREAT_FAMILY, ev_ts):
            inc_type = inc.get("event_type")
            if inc_type == ev_type:
                continue  # identical-type dedup is IncidentEngine's job
            if not self._same_space(inc, camera, zone):
                continue
            if not self._same_identity(inc, identity):
                continue
            return {
                "incident_id": inc["incident_id"],
                "reason": (
                    f"related to {inc_type} on camera={camera} zone={zone or '-'} "
                    f"within {int(self._window_sec)}s window "
                    f"(event={ev_type}, identity={identity})"
                ),
            }
        return None

    # ------------------------------------------------------------------
    # Cross-camera correlation (Phase 33, A2)
    # ------------------------------------------------------------------

    def cross_camera_correlate(self, event: dict, event_id: int) -> dict | None:
        """Correlate an event across cameras sharing the same identity.

        A genuine cross-camera match requires **either** a shared zone (the
        event's zone matches an open incident's zone on another camera) **or**
        an admin-configured adjacency between the two cameras via ``topology``
        (A3).  Returns a dict with ``confidence`` describing the match, or a
        dict with ``status=CROSS_CAMERA_CORRELATION_UNAVAILABLE`` when neither
        a shared zone nor adjacency can be established -- never a fabricated
        match.

        The match describes *the same identity observed moving through the
        space*; it makes no claim about intent or guilt.
        """
        ev_type = event.get("event_type")
        camera = event.get("camera")
        zone = event.get("zone")
        emp = event.get("employee_id")
        ev_ts = event.get("timestamp")
        if ev_type not in PERSON_THREAT_FAMILY:
            return {"status": CROSS_CAMERA_CORRELATION_UNAVAILABLE,
                    "reason": "event type is not a person-threat observation"}
        identity = _identity_key(emp)

        candidates = self._open_recent(PERSON_THREAT_FAMILY, ev_ts)
        best = None
        for inc in candidates:
            if inc.get("event_type") == ev_type:
                continue
            if not self._same_identity(inc, identity):
                continue
            inc_cam = inc.get("camera")
            inc_zone = inc.get("zone")
            # Same camera => not cross-camera.
            if camera and inc_cam and camera == inc_cam:
                continue
            # Shared zone => genuine cross-camera (high confidence).
            if zone and inc_zone and zone == inc_zone:
                confidence = 0.9
                basis = f"shared zone={zone} across cameras {inc_cam} -> {camera}"
            # Admin-configured adjacency => genuine (medium-high confidence).
            elif camera and inc_cam and self._adjacent(camera, inc_cam):
                confidence = 0.7
                basis = (f"admin-configured camera adjacency "
                         f"{inc_cam} -> {camera}")
            else:
                continue  # no honest basis for a cross-camera match
            cand = {
                "incident_id": inc["incident_id"],
                "confidence": confidence,
                "reason": (
                    f"same identity {identity} observed cross-camera "
                    f"({basis}) within {int(self._window_sec)}s"
                ),
            }
            if best is None or confidence > best["confidence"]:
                best = cand

        if best is None:
            return {"status": CROSS_CAMERA_CORRELATION_UNAVAILABLE,
                    "reason": (
                        f"no shared zone or configured adjacency for "
                        f"camera={camera or '-'} zone={zone or '-'}")}
        return best

    def _adjacent(self, camera_a: str, camera_b: str) -> bool:
        if self._topology is None:
            return False
        try:
            return bool(self._topology.is_adjacent(camera_a, camera_b))
        except Exception:
            return False

    def _same_space(self, inc: dict, camera: str | None,
                    zone: str | None) -> bool:
        """Same camera OR same zone (cross-camera via zone is allowed)."""
        inc_cam = inc.get("camera")
        inc_zone = inc.get("zone")
        if camera and inc_cam and camera == inc_cam:
            return True
        if zone and inc_zone and zone == inc_zone:
            return True
        # empty-zone person threat: require same camera to be safe
        return bool(camera and inc_cam and camera == inc_cam)

    def _same_identity(self, inc: dict, identity: str) -> bool:
        inc_id = _identity_key(inc.get("employee_id"))
        # Unknown correlates with Unknown; employee correlates with same employee.
        return inc_id == identity

    # ------------------------------------------------------------------
    # Lifecycle pairing (offline->recovered, tamper suspected->cleared)
    # ------------------------------------------------------------------

    def _correlate_lifecycle(self, event: dict, ev_type: str,
                             camera: str | None, ev_ts: str | None) -> dict | None:
        pair = (ev_type in CAMERA_LIFECYCLE_PAIR and CAMERA_LIFECYCLE_PAIR) or \
               (ev_type in TAMPER_LIFECYCLE_PAIR and TAMPER_LIFECYCLE_PAIR)
        if not pair:
            return None
        partner = pair[0] if ev_type == pair[1] else pair[1]
        for inc in self._open_recent((partner,), ev_ts):
            if inc.get("camera") and camera and inc["camera"] != camera:
                continue
            return {
                "incident_id": inc["incident_id"],
                "reason": f"part of a {partner} -> {ev_type} camera lifecycle "
                          f"(camera={camera or '-s'})",
            }
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _open_recent(self, event_types: tuple[str, ...], ev_ts: str | None) -> list[dict]:
        """Return OPEN/ACKNOWLEDGED incidents of ``event_types`` seen recently."""
        ts = _timestamp_epoch(ev_ts)
        out = []
        window_end = time.time() if ts is None else ts
        window_start = window_end - self._window_sec
        for inc in self._incidents.list(limit=500):
            if inc.get("status") not in ("OPEN", "ACKNOWLEDGED"):
                continue
            if inc.get("event_type") not in event_types:
                continue
            inc_ts = _timestamp_epoch(inc.get("last_seen") or inc.get("first_seen"))
            if inc_ts is None:
                continue
            if window_start <= inc_ts <= window_end + self._window_sec:
                out.append(inc)
        return out

    def _add_to_window(self, event: dict, event_id: int) -> None:
        self._recent.append({
            "event_id": event_id,
            "event_type": event.get("event_type"),
            "camera": event.get("camera"),
            "zone": event.get("zone"),
            "employee_id": event.get("employee_id"),
            "timestamp": event.get("timestamp"),
            "severity": event.get("severity"),
            "confidence": event.get("confidence"),
        })
