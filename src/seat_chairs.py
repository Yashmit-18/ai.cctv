"""Chair / desk-context tracking layer (Phase 59b).

Purpose
-------
Deal the *which chair* of a person: the (optional) desk/chair a spatial track
currently occupies, as pure **context** bound to a seat zone.  A chair is a
child of a seat zone (``zone_id`` is mandatory) and may be assigned to one
employee as a home-desk *hint*.

CRITICAL IDENTITY RULE
----------------------
A chair is CONTEXT ONLY -- it never becomes an identity authority.

    Person Detection
          +
    Spatial Track
          +
    Desk/Seat Zone      <-- src.seat_zones
          +
    Chair               <-- this module (children of a seat zone)
          +
    Face Identity       <-- canonical identity (src.tracker / face_registry)
          +
    Temporal Continuity
          |
          v
    Employee State

Consequences enforced here:

* ``assigned_employee_id`` on a chair is a home-desk HINT for context only.
  It is NEVER used to relabel the occupant: a strongly-confirmed EMP002 in
  EMP001's chair stays EMP002 (optionally emitting a neutral
  ``CHAIR_ASSIGNMENT_MISMATCH`` observation).
* A person whose face is Unknown stays UNKNOWN, even in an assigned chair.
* An empty chair reports VACANT -- it never becomes "EMP001 AWAY" by itself.
* ``employee_id`` inside chair events is only ever the *face-confirmed*
  occupant, never an inference from the chair assignment.
* Chairs never fuse identity across cameras or tracks.

Chair == child of a zone
------------------------
There is intentional 1 desk -> 0..N chairs (mirrors database.py intent note):
a desk zone may have zero, one, or more chairs, configured independently.
``current_zone_id`` (the seat-zone engine's answer) is not relabeled by chair
membership -- a chair simply narrows *where* within the zone.

Events
------
Transition-based, sanitized, persisted to ``chair_events`` (never raw
biometrics/credentials): ``CHAIR_OCCUPIED``, ``CHAIR_VACATED``,
``CHAIR_IDENTITY_CONFIRMED``, ``CHAIR_IDENTITY_UNKNOWN``,
``CHAIR_ASSIGNMENT_MISMATCH``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from . import database as db

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

CHAIR_OCCUPIED = "CHAIR_OCCUPIED"
CHAIR_VACATED = "CHAIR_VACATED"
CHAIR_IDENTITY_CONFIRMED = "CHAIR_IDENTITY_CONFIRMED"
CHAIR_IDENTITY_UNKNOWN = "CHAIR_IDENTITY_UNKNOWN"
CHAIR_ASSIGNMENT_MISMATCH = "CHAIR_ASSIGNMENT_MISMATCH"

CHAIR_EVENT_TYPES = frozenset({
    CHAIR_OCCUPIED, CHAIR_VACATED,
    CHAIR_IDENTITY_CONFIRMED, CHAIR_IDENTITY_UNKNOWN,
    CHAIR_ASSIGNMENT_MISMATCH,
})

DEFAULT_CHAIR_STORE_PATH = "chair_store.json"   # legacy unused placeholder


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------

@dataclass
class Chair:
    """One configurable chair inside a seat zone.

    ``chair_id`` is unique per camera (children of a zone are scoped to the
    camera the zone lives on).  ``assigned_employee_id`` is the *expected*
    home-desk sitter (optional) -- a contextual hint, NEVER an identity
    conclusion.
    """

    chair_id: str
    camera_id: str
    zone_id: str
    name: str = ""
    label: str = ""
    enabled: bool = True
    assigned_employee_id: str | None = None

    @property
    def qualified_id(self) -> str:
        """Uniquely addressable id: ``camera_id:zone_id:chair_id``."""
        return f"{self.camera_id}:{self.zone_id}:{self.chair_id}"

    def to_dict(self) -> dict:
        return {
            "chair_id": self.chair_id,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "name": self.name,
            "label": self.label,
            "enabled": self.enabled,
            "assigned_employee_id": self.assigned_employee_id,
        }


def _chair_from_row(row: dict) -> Chair:
    return Chair(
        chair_id=row["chair_id"],
        camera_id=row["camera_id"],
        zone_id=row["zone_id"],
        name=row.get("name", ""),
        label=row.get("label", ""),
        enabled=bool(row.get("enabled", 1)),
        assigned_employee_id=row.get("assigned_employee_id") or None,
    )


# ----------------------------------------------------------------------
# Persistent store (chairs table; optional JSON seeding)
# ----------------------------------------------------------------------

class ChairStore:
    """CRUD wrapper over the ``chairs`` table (context-only seats)."""

    def __init__(self, conn):
        self._conn = conn

    # -- Writes --------------------------------------------------------
    def add(self, *, chair_id: str, camera_id: str, zone_id: str,
            name: str = "", label: str = "",
            enabled: bool = True,
            assigned_employee_id: str | None = None) -> Chair:
        if not chair_id or not str(chair_id).strip():
            raise ValueError("chair_id is required")
        if not camera_id or not str(camera_id).strip():
            raise ValueError("camera_id is required")
        if not zone_id or not str(zone_id).strip():
            raise ValueError("zone_id is required (a chair belongs to a zone)")
        chair = Chair(
            chair_id=str(chair_id).strip(),
            camera_id=camera_id.strip(),
            zone_id=zone_id.strip(),
            name=name or "",
            label=label or "",
            enabled=bool(enabled),
            assigned_employee_id=assigned_employee_id or None,
        )
        db.upsert_chair(self._conn, chair.to_dict())
        return chair

    def set_enabled(self, chair_id: str, enabled: bool) -> None:
        db.set_chair_enabled(self._conn, chair_id, bool(enabled))

    def update(self, chair_id: str, **fields) -> Chair | None:
        """Partial edit of a chair (Phase 62 Admin Control Center).

        ``chair_id`` (the identity) is never rewritten.  Fields: ``name``,
        ``label``, ``camera_id``, ``zone_id``, ``enabled``,
        ``assigned_employee_id``.  Raises ``ValueError`` for invalid input.
        Returns the updated chair or None when it does not exist.
        """
        chair = self.get(chair_id)
        if chair is None:
            return None
        for k, setter in (
            ("name", "_str"),
            ("label", "_str"),
            ("camera_id", "_req"),
            ("zone_id", "_req"),
        ):
            if k not in fields:
                continue
            val = str(fields[k] or "").strip()
            if setter == "_req" and not val:
                raise ValueError(f"chair {k} is required")
            setattr(chair, k, val)
        if "enabled" in fields:
            chair.enabled = bool(fields["enabled"])
        if "assigned_employee_id" in fields:
            chair.assigned_employee_id = fields["assigned_employee_id"] or None
        db.upsert_chair(self._conn, chair.to_dict())
        return chair

    def set_assignment(self, chair_id: str,
                       assigned_employee_id: str | None) -> None:
        chair = self.get(chair_id)
        if chair:
            chair.assigned_employee_id = assigned_employee_id or None
            db.upsert_chair(self._conn, chair.to_dict())

    def remove(self, chair_id: str) -> None:
        db.delete_chair(self._conn, chair_id)

    # -- Reads ---------------------------------------------------------
    def get(self, chair_id: str) -> Chair | None:
        row = db.get_chair(self._conn, chair_id)
        return _chair_from_row(row) if row else None

    def list(self, *, enabled_only: bool = False) -> list[Chair]:
        out = []
        for row in db.list_chairs(self._conn, enabled_only=enabled_only):
            out.append(_chair_from_row(row))
        return out

    def for_zone(self, zone_id: str, *, enabled_only: bool = False) -> list[Chair]:
        return [c for c in self.list(enabled_only=enabled_only)
                if c.zone_id == zone_id]

    def for_camera(self, camera_id: str, *, enabled_only: bool = False):
        return [c for c in self.list(enabled_only=enabled_only)
                if c.camera_id == camera_id]


# ----------------------------------------------------------------------
# Defaults / seeding
# ----------------------------------------------------------------------

def load_chair_defaults(path: str) -> list[dict]:
    """Read chair definitions from a JSON file for first-run seeding.

    Shape: ``{"chairs": [ {...} ]}`` or a bare list.  Suppresses the file
    entirely (returns []) when absent or empty -- it is optional.
    """
    import os
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:  # noqa: BLE001 - seeding must never crash boot
        logger.warning("Ignoring bad chairs seed file %s: %s", path, exc)
        return []
    items = payload if isinstance(payload, list) else payload.get("chairs", [])
    return [d for d in items if isinstance(d, dict)]


def seed_chairs(conn, store: ChairStore, path: str) -> int:
    """Seed chairs from ``path`` when the chairs table is empty.  Returns the
    number of chairs added.  Seeding is contextual only -- it never assigns
    employee identity."""
    if db.list_chairs(conn):
        return 0
    added = 0
    for cfg in load_chair_defaults(path):
        try:
            store.add(
                chair_id=cfg.get("chair_id"),
                camera_id=cfg.get("camera_id"),
                zone_id=cfg.get("zone_id"),
                name=cfg.get("name", ""),
                label=cfg.get("label", ""),
                enabled=cfg.get("enabled", True),
                assigned_employee_id=cfg.get("assigned_employee_id"),
            )
            added += 1
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Skipping bad chair %r: %s", cfg.get("chair_id"), exc)
    if added:
        logger.info("Seeded %d chair(s).", added)
    return added
