"""Desk / seat zone tracking layer (Phase 59).

Purpose
-------
Deal the *where* of a person: which desk/seat zone a spatial track currently
occupies.  A desk zone is a polygon (normalised 0..1) attached to one camera,
optionally assigned to one employee.

CRITICAL IDENTITY RULE
----------------------
A seat is CONTEXT ONLY -- it never becomes an identity authority.

    Person Detection
          +
    Spatial Track
          +
    Desk/Seat Zone      <-- this module
          +
    Face Identity       <-- canonical identity (src.tracker / face_registry)
          +
    Temporal Continuity
          ↓
    Employee State

Consequences enforced here:

* EMP001 assigned to A01, but a strongly-confirmed EMP002 sits in A01 -> the
  zone reports "A01 occupied by EMP002" and may emit an *observation*
  :data:`SEAT_ASSIGNMENT_MISMATCH` (neutral wording, never accusatory).  The
  person is NEVER relabelled EMP001.
* A person in A01 whose face is Unknown stays UNKNOWN.
* An empty A01 is reported VACANT -- it never becomes "EMP001 AWAY" by itself.
* ``employee_id`` inside seat events is only ever the *face-confirmed*
  occupant, never an inference from the seat assignment.

Temporal stability
------------------
Occupancy/ vacancy require consecutive evidence frames (hysteresis):

* zone enters OCCUPIED only after ``confirm_frames`` consecutive frames with a
  footpoint inside;
* zone leaves OCCUPIED only after ``clear_frames`` consecutive frames with no
  footpoint inside.

A single noisy frame cannot flip a stable zone.  Timing/frame knobs are
configurable (``config.SEAT_ZONE_CONFIRM_FRAMES`` / ``_CLEAR_FRAMES``).

Camera semantics
----------------
* Zones whose camera is not configured (``active_cameras``) are ignored.
* Zones whose camera is not currently producing usable frames are FROZEN: no
  evidence accrues, no transition runs, and the zone is never force-vacated --
  so a camera outage can never turn into fake vacancy / fake employee AWAY.

Events
------
Transition-based, sanitized, persisted to ``seat_events`` (never raw
biometrics/credentials): ``ZONE_OCCUPIED``, ``ZONE_VACATED``,
``ZONE_IDENTITY_CONFIRMED``, ``ZONE_IDENTITY_UNKNOWN``,
``SEAT_ASSIGNMENT_MISMATCH``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from config import (SEAT_EVENT_COOLDOWN_SEC, SEAT_ZONE_CLEAR_FRAMES,
                    SEAT_ZONE_CONFIRM_FRAMES)
from src import database as db
from src.domain import UNKNOWN_ID
from src.zones import parse_polygon, point_in_polygon

logger = logging.getLogger("cctv.seat_zones")

# ----------------------------------------------------------------------
# Neutral, transition-driven event vocabulary (Phase 59)
# ----------------------------------------------------------------------
ZONE_OCCUPIED = "ZONE_OCCUPIED"
ZONE_VACATED = "ZONE_VACATED"
ZONE_IDENTITY_CONFIRMED = "ZONE_IDENTITY_CONFIRMED"
ZONE_IDENTITY_UNKNOWN = "ZONE_IDENTITY_UNKNOWN"
SEAT_ASSIGNMENT_MISMATCH = "SEAT_ASSIGNMENT_MISMATCH"

SEAT_EVENT_TYPES: tuple[str, ...] = (
    ZONE_OCCUPIED, ZONE_VACATED, ZONE_IDENTITY_CONFIRMED,
    ZONE_IDENTITY_UNKNOWN, SEAT_ASSIGNMENT_MISMATCH,
)

ZONE_STATUS_VACANT = "VACANT"
ZONE_STATUS_OCCUPIED = "OCCUPIED"
ZONE_STATUS_DISABLED = "DISABLED"

_IDENTITY_STATE_CONFIRMED = "CONFIRMED"
_IDENTITY_STATE_UNKNOWN = "UNKNOWN"


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------

@dataclass
class SeatZone:
    """One configurable desk/seat zone.

    ``polygon`` holds normalised 0..1 points.  ``assigned_employee_id`` is the
    *expected* sitter (optional) -- a contextual hint, NEVER an identity
    conclusion.
    """

    zone_id: str
    camera_id: str
    name: str = ""
    polygon: list[tuple[float, float]] | None = None
    enabled: bool = True
    assigned_employee_id: str | None = None

    @property
    def qualified_id(self) -> str:
        """Uniquely addressable id: ``camera_id:zone_id`` (multi-camera safe)."""
        return f"{self.camera_id}:{self.zone_id}"

    @property
    def has_polygon(self) -> bool:
        return bool(self.polygon and len(self.polygon) >= 3)

    def contains(self, x: float, y: float) -> bool:
        if not self.has_polygon:
            return False
        return point_in_polygon(float(x), float(y), self.polygon)

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "camera_id": self.camera_id,
            "name": self.name,
            "polygon": json.dumps(self.polygon) if self.polygon else None,
            "enabled": self.enabled,
            "assigned_employee_id": self.assigned_employee_id,
        }


def _zone_from_row(row: dict) -> SeatZone:
    poly = None
    try:
        poly = parse_polygon(row.get("polygon"))
    except ValueError:
        poly = None
    return SeatZone(
        zone_id=row["zone_id"],
        camera_id=row["camera_id"],
        name=row.get("name", ""),
        polygon=poly,
        enabled=bool(row.get("enabled", 1)),
        assigned_employee_id=row.get("assigned_employee_id") or None,
    )


# ----------------------------------------------------------------------
# Persistent configuration (SQLite ``seat_zones``; optional JSON seeding)
# ----------------------------------------------------------------------

class SeatZoneStore:
    """CRUD wrapper over the ``seat_zones`` table (single source of truth)."""

    def __init__(self, conn):
        self._conn = conn

    # -- Writes --------------------------------------------------------
    def add(self, *, zone_id: str, camera_id: str, name: str = "",
            polygon: list | str | None = None,
            enabled: bool = True,
            assigned_employee_id: str | None = None) -> SeatZone:
        if not zone_id or not str(zone_id).strip():
            raise ValueError("zone_id is required")
        if not camera_id or not str(camera_id).strip():
            raise ValueError("camera_id is required")
        poly = parse_polygon(polygon)
        if not poly:
            raise ValueError(f"seat zone {zone_id} needs a polygon with >= 3 points")
        zone = SeatZone(
            zone_id=str(zone_id).strip(),
            camera_id=camera_id.strip(),
            name=name or "",
            polygon=poly,
            enabled=bool(enabled),
            assigned_employee_id=assigned_employee_id or None,
        )
        db.upsert_seat_zone(self._conn, zone.to_dict())
        return zone

    def set_enabled(self, zone_id: str, enabled: bool) -> None:
        zone = self.get(zone_id)
        if zone:
            zone.enabled = bool(enabled)
            db.upsert_seat_zone(self._conn, zone.to_dict())

    def update(self, zone_id: str, **fields) -> SeatZone | None:
        """Partial edit of a desk/seat zone (Phase 62 Admin Control Center).

        ``zone_id`` (the identity) is never rewritten.  Fields: ``name``,
        ``camera_id``, ``polygon``, ``enabled``, ``assigned_employee_id``.
        Polygons are re-validated (>= 3 normalised points).  Raises
        ``ValueError`` for invalid input.  Returns the updated zone or None.
        """
        zone = self.get(zone_id)
        if zone is None:
            return None
        if "name" in fields:
            zone.name = str(fields["name"] or "")
        if "camera_id" in fields:
            cam = str(fields["camera_id"] or "").strip()
            if not cam:
                raise ValueError("zone camera_id is required")
            zone.camera_id = cam
        if "polygon" in fields:
            poly = parse_polygon(fields["polygon"])
            if not poly:
                raise ValueError(
                    f"seat zone {zone_id} needs a polygon with >= 3 points")
            zone.polygon = poly
        if "enabled" in fields:
            zone.enabled = bool(fields["enabled"])
        if "assigned_employee_id" in fields:
            zone.assigned_employee_id = (
                fields["assigned_employee_id"] or None)
        db.upsert_seat_zone(self._conn, zone.to_dict())
        return zone

    def set_assignment(self, zone_id: str,
                       assigned_employee_id: str | None) -> None:
        zone = self.get(zone_id)
        if zone:
            zone.assigned_employee_id = assigned_employee_id or None
            db.upsert_seat_zone(self._conn, zone.to_dict())

    def remove(self, zone_id: str) -> None:
        db.delete_seat_zone(self._conn, zone_id)

    # -- Reads ---------------------------------------------------------
    def get(self, zone_id: str) -> SeatZone | None:
        for z in self.list(enabled_only=False):
            if z.zone_id == zone_id:
                return z
        return None

    def list(self, *, enabled_only: bool = False) -> list[SeatZone]:
        out = []
        for row in db.list_seat_zones(self._conn, enabled_only=enabled_only):
            out.append(_zone_from_row(row))
        return out

    def for_camera(self, camera_id: str) -> list[SeatZone]:
        return [z for z in self.list() if z.camera_id == camera_id]


def load_seat_zone_defaults(path: str) -> list[dict]:
    """Read desk/seat zone definitions from a JSON file for first-run seeding.

    Shape: ``{"zones": [ {...} ]}`` or a bare list.  Suppresses the file
    entirely (returns []) when absent or empty -- it is optional.
    """
    import os
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read seat-zone defaults %s: %s", path, exc)
        return []
    zones = data.get("zones") if isinstance(data, dict) else data
    return zones if isinstance(zones, list) else []


def seed_seat_zones(conn, store: SeatZoneStore, path: str) -> int:
    """Seed desk/seat zones from ``path`` when the table is empty."""
    if db.list_seat_zones(conn):
        return 0
    added = 0
    for cfg in load_seat_zone_defaults(path):
        try:
            store.add(
                zone_id=cfg.get("zone_id"),
                camera_id=cfg.get("camera_id"),
                name=cfg.get("name", ""),
                polygon=cfg.get("polygon"),
                enabled=bool(cfg.get("enabled", True)),
                assigned_employee_id=cfg.get("assigned_employee_id"),
            )
            added += 1
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Skipping bad seat zone %r: %s", cfg.get("zone_id"), exc)
    if added:
        logger.info("Seeded %d desk/seat zone(s).", added)
    return added


# ----------------------------------------------------------------------
# Per-zone runtime state
# ----------------------------------------------------------------------

class _ZoneState:
    __slots__ = (
        "status", "occ_evidence", "vac_evidence", "occupant_track",
        "occupant_identity", "occupied_since", "vacated_since", "episode",
        "reported_identity", "reported_mismatch", "last_identity_event",
        "last_persons",
    )

    def __init__(self):
        self.status = ZONE_STATUS_VACANT
        self.occ_evidence = 0
        self.vac_evidence = 0
        self.occupant_track: str | None = None
        self.occupant_identity: str | None = None
        self.occupied_since: float | None = None
        self.vacated_since: float = time.time()
        self.episode = 0
        # identity/conflict already *reported* for the current episode
        self.reported_identity: str | None = None
        self.reported_mismatch = False
        self.last_identity_event: float = 0.0
        self.last_persons = 0


# ----------------------------------------------------------------------
# The engine
# ----------------------------------------------------------------------

class SeatZoneTracker:
    """Tracks desk/seat zone occupancy from spatial tracks (context only).

    Parameters
    ----------
    conn : sqlite3.Connection | None
        Write connection used to persist seat events.  ``None`` keeps the
        tracker fully in-memory (tests / read-only environments).
    store : SeatZoneStore | None
        Optional persistent config.  When given, zones are (re)loaded from it
        on :meth:`reload` and at construction.
    confirm_frames : int
        Consecutive frames of footpoint presence required to mark a zone
        OCCUPIED (hysteresis).
    clear_frames : int
        Consecutive frames of absence required to mark a zone VACANT.
    event_cooldown_sec : float
        Minimum gap between repeated identity/mismatch events for the same
        zone (identity *reports*; transitions are never gated by it).
    active_cameras : set[str] | None
        Configured, enabled camera ids.  Zones on any other camera are
        ignored (disabled camera).  ``None`` = every zone's camera accepted.
    """

    def __init__(self, conn=None, store: SeatZoneStore | None = None,
                 *, confirm_frames: int | None = None,
                 clear_frames: int | None = None,
                 event_cooldown_sec: float | None = None,
                 active_cameras: set[str] | None = None):
        self._conn = conn
        self._store = store
        self.confirm_frames = max(1, int(
            confirm_frames if confirm_frames is not None
            else SEAT_ZONE_CONFIRM_FRAMES))
        self.clear_frames = max(1, int(
            clear_frames if clear_frames is not None else SEAT_ZONE_CLEAR_FRAMES))
        self.cooldown = max(0.0, float(
            event_cooldown_sec if event_cooldown_sec is not None
            else SEAT_EVENT_COOLDOWN_SEC))
        self.active_cameras = (
            set(active_cameras) if active_cameras is not None else None)
        self._all_zones: dict[str, SeatZone] = {}
        self._zones: dict[str, SeatZone] = {}
        self._zone_state: dict[str, _ZoneState] = {}
        self._track_zone: dict[str, str] = {}
        self._last_tick = 0.0
        self.recent_events: list[dict] = []
        self._metric = {"membership_checks": 0, "last_latency_ms": 0.0}
        self.reload()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def reload(self):
        """(Re)load enabled zones from the store (if any) into memory."""
        zones: list[SeatZone] = []
        if self._store is not None:
            zones = self._store.list()
        self._install_zones(zones)

    def set_zones(self, zones: list[SeatZone]):
        """Inject zones directly (test seam / non-DB configuration)."""
        self._install_zones(zones)

    def _install_zones(self, zones: list[SeatZone]) -> None:
        """Split configured zones into ``_all_zones`` (truth) and ``_zones``
        (the enabled + camera-active subset that evaluation runs over)."""
        self._all_zones = {z.zone_id: z for z in zones}
        keep: dict[str, SeatZone] = {}
        for z in zones:
            if not z.enabled:
                continue
            if self.active_cameras is not None \
                    and z.camera_id not in self.active_cameras:
                continue
            keep[z.zone_id] = z
        self._zones = keep
        # Drop runtime state for zones that no longer exist.
        for zid in list(self._zone_state):
            if zid not in self._all_zones:
                self._zone_state.pop(zid, None)
        # Fresh config: keep prior runtime state for surviving zones so a
        # config refresh never resets live occupancy.

    @property
    def zones(self) -> list[SeatZone]:
        return sorted(self._zones.values(), key=lambda z: (z.camera_id, z.zone_id))

    def zone(self, zone_id: str) -> SeatZone | None:
        return self._zones.get(zone_id)

    # ------------------------------------------------------------------
    # Track -> zone membership
    # ------------------------------------------------------------------

    def _zone_for_track(self, track) -> SeatZone | None:
        """Return the first enabled polygon zone that contains the track's
        footpoint on the track's camera.  ``None`` = outside all zones."""
        cam = getattr(track, "cam", None)
        if not cam:
            return None
        try:
            fx, fy = track.footpoint_normalized
        except Exception:
            return None
        candidates = sorted(
            (z for z in self._zones.values() if z.camera_id == cam),
            key=lambda z: z.zone_id,
        )
        for z in candidates:
            self._metric["membership_checks"] += 1
            if z.contains(fx, fy):
                return z
        return None

    def track_zone(self, track_id: str) -> str | None:
        """Last-known zone membership (``camera_id:zone_id``) for a track."""
        return self._track_zone.get(track_id)

    # ------------------------------------------------------------------
    # Per-cycle processing
    # ------------------------------------------------------------------

    def tick(self, tracks, usable_camera_ids: set[str] | None = None,
             now: float | None = None):
        """Advance zone state from this cycle's active spatial tracks.

        ``tracks``: iterable of ``Track`` objects (see ``src.tracker``).
        ``usable_camera_ids``: cameras that produced a usable live frame this
        cycle.  Zones on a camera NOT in this set are frozen -- a camera
        outage never force-vacates a desk and never invents employee absence.
        Returns this cycle's emitted events (sanitized dicts).
        """
        now = time.time() if now is None else now
        self._last_tick = now
        usable = set(usable_camera_ids) if usable_camera_ids is not None else None
        events: list[dict] = []

        memberships: dict[str, SeatZone] = {}
        for track in tracks:
            z = self._zone_for_track(track)
            if z is not None:
                memberships[track.track_id] = z
                # CONTEXT ONLY: attach the stable desk id to the track.  This
                # never influences identity/employee decisions (documented in
                # ``Track.zone``); it just exposes *where* the person is.
                try:
                    track.zone = z.qualified_id
                except Exception:
                    pass

        grouped: dict[str, list[str]] = {}
        for track in tracks:
            z = memberships.get(track.track_id)
            if z is None:
                continue
            if usable is not None and z.camera_id not in usable:
                continue
            grouped.setdefault(z.zone_id, []).append(track.track_id)

        for zone in self.zones:
            zid = zone.zone_id
            if usable is not None and zone.camera_id not in usable:
                continue  # frozen -- camera not usable this cycle
            occupants = grouped.get(zid) or ()
            self._tick_zone(zone, tuple(occupants), tracks, now, events)

        # Track -> zone map for observability (raw, this tick).
        self._track_zone = {}
        for track in tracks:
            z = memberships.get(track.track_id)
            if z is not None:
                self._track_zone[track.track_id] = z.qualified_id
        return events

    def _tick_zone(self, zone: SeatZone, occupant_ids: tuple[str, ...],
                   tracks, now: float, events: list[dict]) -> None:
        st = self._zone_state.setdefault(zone.zone_id, _ZoneState())
        # Locate the occupant track (first of any occupants, deterministic).
        occupant = None
        for t in tracks:
            if getattr(t, "track_id", None) in occupant_ids:
                occupant = t
                break

        if occupant is not None:
            st.occ_evidence += 1
            st.vac_evidence = 0
            st.last_persons = len(occupant_ids)
        else:
            st.vac_evidence += 1
            st.occ_evidence = 0

        if st.status == ZONE_STATUS_VACANT:
            if st.occ_evidence >= self.confirm_frames and occupant is not None:
                self._mark_occupied(zone, st, occupant, now, events)
        elif st.status == ZONE_STATUS_OCCUPIED:
            if st.vac_evidence >= self.clear_frames:
                self._mark_vacant(zone, st, now, events)
            elif occupant is not None:
                self._refresh_occupant_identity(zone, st, occupant, now, events)

    def _mark_occupied(self, zone: SeatZone, st: _ZoneState, track,
                       now: float, events: list[dict]) -> None:
        st.status = ZONE_STATUS_OCCUPIED
        st.occupant_track = track.track_id
        st.occupant_identity = track.identity or UNKNOWN_ID
        st.occupied_since = now
        st.episode += 1
        st.occ_evidence = 0
        st.reported_identity = None
        st.reported_mismatch = False
        self._emit_occupied_event(zone, st, track, now, events)
        self._emit_identity_events(zone, st, track, now, events, force=True)

    def _mark_vacant(self, zone: SeatZone, st: _ZoneState, now: float,
                     events: list[dict]) -> None:
        st.status = ZONE_STATUS_VACANT
        st.vac_evidence = 0
        st.occupant_track = None
        st.occupant_identity = None
        st.vacated_since = now
        st.last_persons = 0
        events.append(self._event(zone, ZONE_VACATED, now, st))

    def _refresh_occupant_identity(self, zone: SeatZone, st: _ZoneState,
                                   track, now: float, events: list[dict]) -> None:
        ident = track.identity or UNKNOWN_ID
        st.occupant_identity = ident
        st.occupant_track = track.track_id
        if ident == st.reported_identity:
            return
        self._emit_identity_events(zone, st, track, now, events, force=False)

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit_occupied_event(self, zone: SeatZone, st: _ZoneState,
                             track, now: float, events: list[dict]) -> None:
        ev = self._event(zone, ZONE_OCCUPIED, now, st, track=track)
        events.append(ev)

    def _emit_identity_events(self, zone: SeatZone, st: _ZoneState, track,
                              now: float, events: list[dict],
                              force: bool) -> None:
        ident = track.identity or UNKNOWN_ID
        if not force and now - st.last_identity_event < self.cooldown:
            return
        known = ident not in (None, UNKNOWN_ID)
        mismatch = bool(zone.assigned_employee_id) and known \
            and zone.assigned_employee_id != ident
        if known:
            events.append(self._event(zone, ZONE_IDENTITY_CONFIRMED, now, st,
                                      track=track, identity=ident))
            if mismatch:
                events.append(self._event(
                    zone, SEAT_ASSIGNMENT_MISMATCH, now, st,
                    track=track, identity=ident,
                    details=f"assigned={zone.assigned_employee_id} "
                            f"confirmed={ident}"))
        else:
            events.append(self._event(zone, ZONE_IDENTITY_UNKNOWN, now, st,
                                      track=track))
        st.reported_identity = ident
        st.reported_mismatch = mismatch
        st.last_identity_event = now

    def _event(self, zone: SeatZone, event_type: str, now: float,
               st: _ZoneState, track=None, identity: str | None = None,
               details: str | None = None) -> dict:
        ev = {
            "timestamp": _iso(now),
            "event_type": event_type,
            "camera_id": zone.camera_id,
            "zone_id": zone.qualified_id,
            "track_id": getattr(track, "track_id", None) if track is not None
            else st.occupant_track,
            "employee_id": identity if identity is not None else st.occupant_identity,
            "details": details,
        }
        self.recent_events.append(ev)
        if len(self.recent_events) > 200:
            self.recent_events = self.recent_events[-200:]
        if self._conn is not None:
            try:
                db.insert_seat_event(self._conn, ev)
            except Exception as exc:  # isolation: events never break the loop
                logger.debug("Seat event persist skipped (isolated): %s", exc)
        return ev

    # ------------------------------------------------------------------
    # Serialisation (dashboard / tests)
    # ------------------------------------------------------------------

    def claims(self) -> list[dict]:
        """Structured per-occupant claims for the seat context.

        Each claim::

          {"employee_id", "camera_id", "zone_id", "track_id",
           "identity_state", "zone_state", "timestamp"}

        ``employee_id`` is the face-confirmed occupant only (never inferred
        from the seat assignment); ``identity_state`` is CONFIRMED / UNKNOWN.
        """
        out: list[dict] = []
        for zone in self.zones:
            st = self._zone_state.get(zone.zone_id)
            if st is None or st.status != ZONE_STATUS_OCCUPIED:
                continue
            ident = st.occupant_identity
            known = ident not in (None, UNKNOWN_ID)
            out.append({
                "employee_id": ident if known else UNKNOWN_ID,
                "camera_id": zone.camera_id,
                "zone_id": zone.zone_id,
                "qualified_id": zone.qualified_id,
                "track_id": st.occupant_track,
                "identity_state": (_IDENTITY_STATE_CONFIRMED if known
                                   else _IDENTITY_STATE_UNKNOWN),
                "zone_state": st.status,
                "timestamp": _iso(self._last_tick),
            })
        return out

    def snapshot(self) -> dict:
        """Live per-zone status for the dashboard (never biometrics/creds)."""
        rows = []
        all_zones: list[SeatZone] = []
        if self._store is not None:
            all_zones = self._store.list(enabled_only=False)
        all_zones = sorted(all_zones or list(self._all_zones.values()),
                           key=lambda z: (z.camera_id, z.zone_id))
        for zone in all_zones:
            st = self._zone_state.get(zone.zone_id)
            if not zone.enabled:
                status = ZONE_STATUS_DISABLED
            elif self.active_cameras is not None \
                    and zone.camera_id not in self.active_cameras:
                status = "CAMERA_DISABLED"
            elif st is None:
                status = ZONE_STATUS_VACANT
            else:
                status = st.status
            ident_state = None
            occupant = None
            track_id = None
            if st is not None and st.status == ZONE_STATUS_OCCUPIED:
                occupant = st.occupant_identity
                track_id = st.occupant_track
                known = occupant not in (None, UNKNOWN_ID)
                ident_state = (_IDENTITY_STATE_CONFIRMED if known
                               else _IDENTITY_STATE_UNKNOWN)
            rows.append({
                "zone_id": zone.zone_id,
                "qualified_id": zone.qualified_id,
                "camera_id": zone.camera_id,
                "name": zone.name or zone.zone_id,
                "status": status,
                "assigned_employee_id": zone.assigned_employee_id,
                "current_identity": occupant,
                "identity_state": ident_state,
                "track_id": track_id,
                "persons": st.last_persons if st is not None else 0,
                "occupied_since": (round(st.occupied_since, 2)
                                   if st is not None and st.occupied_since
                                   else None),
                "vacated_since": (round(st.vacated_since, 2)
                                  if st is not None and st.vacated_since
                                  else None),
                "episode": st.episode if st is not None else 0,
            })
        return {
            "enabled": bool(self._zones),
            "zone_count": len(rows),
            "occupied_count": sum(1 for r in rows
                                  if r["status"] == ZONE_STATUS_OCCUPIED),
            "zones": rows,
        }

    def metrics(self) -> dict:
        return {
            "tracks": len(self._track_zone),
            "zones": len(self._zones),
            "membership_checks": self._metric["membership_checks"],
            "last_latency_ms": self._metric["last_latency_ms"],
        }