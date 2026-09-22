"""Restricted / security zones (Phase 31).

Zones are fully configurable (never hard-coded): name, camera, polygon
(normalised 0..1) or ``"full"`` for the whole frame, allowed employees,
optional active schedule, and an alert policy.

Policies
--------
* ``BYPASS``             -- zone is monitored for analytics only, never alerts.
* ``INTRUSION``          -- any person in the zone when not allowed -> INTRUSION.
* ``UNKNOWN_ONLY``       -- only unidentified people trigger INTRUSION.
* ``AFTER_HOURS_INTRUSION`` -- only outside configured office hours.

A person is "inside" a zone when the centre of their person bounding box lies
within the polygon.  An offline camera NEVER evaluates intrusion (the caller
guards on live frames before calling this module).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

from src import database as db
from src.domain import UNKNOWN_ID

logger = logging.getLogger("cctv.zones")

POLICY_BYPASS = "BYPASS"
POLICY_INTRUSION = "INTRUSION"
POLICY_UNKNOWN_ONLY = "UNKNOWN_ONLY"
POLICY_AFTER_HOURS_INTRUSION = "AFTER_HOURS_INTRUSION"

POLICIES = (POLICY_BYPASS, POLICY_INTRUSION, POLICY_UNKNOWN_ONLY,
            POLICY_AFTER_HOURS_INTRUSION)


@dataclass
class Zone:
    zone_name: str
    camera: str = ""
    enabled: bool = True
    polygon: list[tuple[float, float]] | None = None  # normalized pts or None=full
    allowed: set[str] = None  # None => all employees allowed
    schedule: dict | None = None      # {"start": "HH:MM", "end": "HH:MM"} or None
    policy: str = POLICY_INTRUSION
    capacity: int | None = None  # per-zone occupancy cap (None falls back to global)

    def __post_init__(self):
        if self.allowed is None:
            self.allowed = set()

    @property
    def full_frame(self) -> bool:
        return self.polygon is None

    def contains(self, x: float, y: float) -> bool:
        """True when the point (normalised 0..1) is inside the zone."""
        if self.full_frame:
            return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        return _point_in_polygon(x, y, self.polygon)

    def time_is_active(self, clock_minutes: int) -> bool:
        """True when the zone's optional schedule window is active (24x7 if unset)."""
        if not self.schedule:
            return True
        return _window_active(clock_minutes, self.schedule)

    def allows(self, employee_id: str) -> bool:
        """True when ``employee_id`` may be present (empty allow-list = all)."""
        return not self.allowed or employee_id in self.allowed or employee_id == UNKNOWN_ID and False

    def to_dict(self) -> dict:
        return {
            "zone_name": self.zone_name,
            "camera": self.camera,
            "enabled": self.enabled,
            "polygon": json.dumps(self.polygon) if self.polygon else "full",
            "allowed_employees": ",".join(sorted(self.allowed or set())),
            "schedule": json.dumps(self.schedule) if self.schedule else None,
            "alert_policy": self.policy,
            "capacity": self.capacity,
        }


def _window_active(clock_minutes: int, schedule: dict) -> bool:
    from src.domain import is_time_in_window
    return is_time_in_window(clock_minutes, schedule.get("start"),
                             schedule.get("end"))


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Public ray-casting point-in-polygon test (normalised coords 0..1).

    Shared geometry primitive used by the security-zone layer (Phase 31) and
    the desk/seat zone layer (Phase 59) so the polygon logic is defined once.
    """
    return _point_in_polygon(x, y, poly)


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (normalised coords)."""
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def parse_polygon(raw: str | list) -> list[tuple[float, float]] | None:
    """Parse a polygon definition.

    Accepted forms:
      ``"full"``                                   -> None (whole frame)
      ``"[[0,0],[1,0],[1,1],[0,1]]"`` (JSON)        -> normalized points
      ``[[0.2,0.2],[0.8,0.2],[0.8,0.8],[0.2,0.8]]`` (already a list)
    Bounding the points to 0..1 is enforced; a malformed polygon raises
    :class:`ValueError`.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if s.lower() == "full":
            return None
        try:
            raw = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid polygon JSON: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("polygon must be a non-empty list of [x, y] points")
    pts = []
    for p in raw:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"invalid polygon point: {p!r}")
        px = float(p[0])
        py = float(p[1])
        if not (0.0 <= px <= 1.0 and 0.0 <= py <= 1.0):
            raise ValueError(f"polygon point out of range: {p!r}")
        pts.append((px, py))
    if len(pts) < 3:
        raise ValueError("polygon needs at least 3 points")
    return pts


class ZoneStore:
    """CRUD + evaluation wrapper over the ``security_zones`` table."""

    def __init__(self, conn):
        self._conn = conn

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def add(self, *, zone_name: str, camera: str,
            polygon: str | list | None = None,
            allowed: list[str] | None = None,
            schedule: dict | None = None,
            policy: str = POLICY_INTRUSION,
            enabled: bool = True,
            capacity: int | None = None) -> Zone:
        poly_normalized = parse_polygon(polygon)
        zone = Zone(
            zone_name=zone_name, camera=camera, enabled=enabled,
            polygon=poly_normalized,
            allowed=set(allowed or []), schedule=schedule, policy=policy,
            capacity=capacity,
        )
        if policy not in POLICIES:
            raise ValueError(f"unknown zone policy: {policy}")
        db.upsert_zone(self._conn, zone.to_dict())
        return zone

    def update_enabled(self, zone_name: str, enabled: bool) -> None:
        zone = self.get(zone_name)
        if zone:
            zone.enabled = enabled
            db.upsert_zone(self._conn, zone.to_dict())

    def remove(self, zone_name: str) -> None:
        db.delete_zone(self._conn, zone_name)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get(self, zone_name: str) -> Zone | None:
        for z in self.list(enabled_only=False):
            if z.zone_name == zone_name:
                return z
        return None

    def list(self, *, enabled_only: bool = True) -> list[Zone]:
        zones = []
        for row in db.list_zones(self._conn, enabled_only=enabled_only):
            try:
                poly = parse_polygon(row.get("polygon"))
            except ValueError:
                poly = None
            schedule = None
            if row.get("schedule"):
                try:
                    schedule = json.loads(row["schedule"])
                except json.JSONDecodeError:
                    schedule = None
            allowed = {
                e for e in (row.get("allowed_employees") or "").split(",")
                if e and e != "*"
            }
            if row.get("allowed_employees", "").strip() == "*":
                allowed = set()
            zones.append(Zone(
                zone_name=row["zone_name"], camera=row.get("camera", ""),
                enabled=bool(row.get("enabled", 1)), polygon=poly,
                allowed=allowed, schedule=schedule,
                policy=row.get("alert_policy", POLICY_INTRUSION),
                capacity=row.get("capacity"),
            ))
        return zones

    def for_camera(self, camera: str) -> list[Zone]:
        return [z for z in self.list() if z.camera == camera]

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_presence(self, camera: str, x: float, y: float,
                          employee_id: str, clock_minutes: int,
                          now_iso: str | None = None) -> list[dict]:
        """Return a list of non-bypass zone hits for a person at (x, y).

        Each hit: ``{"zone": Zone, "employee_id": str, "intrusion": bool}``
        where ``intrusion`` is True only when the zone policy/allow-list says
        the person should not be there.  Camera-offline is entirely the
        caller's concern -- this function never sees dead cameras.
        """
        out = []
        for zone in self.list(enabled_only=True):
            if zone.camera and zone.camera != camera:
                continue
            if zone.policy == POLICY_BYPASS:
                continue  # analytics only -- never produces hits
            if not zone.contains(x, y):
                continue
            if zone.policy == POLICY_AFTER_HOURS_INTRUSION:
                # Evaluated both when active (allowed) and inactive (intrusion).
                out.append({"zone": zone, "employee_id": employee_id,
                            "intrusion": not zone.time_is_active(clock_minutes)})
                continue
            if not zone.time_is_active(clock_minutes):
                continue
            if zone.policy == POLICY_INTRUSION and not zone.allows(employee_id):
                out.append({"zone": zone, "employee_id": employee_id,
                            "intrusion": True})
            elif zone.policy == POLICY_UNKNOWN_ONLY and employee_id == UNKNOWN_ID:
                out.append({"zone": zone, "employee_id": employee_id,
                            "intrusion": True})
            else:
                out.append({"zone": zone, "employee_id": employee_id,
                            "intrusion": False})
        return out


def load_zone_defaults(path: str) -> list[dict]:
    """Read zone definitions from a JSON file for first-run seeding.

    File shape: ``{"zones": [ {...} ]}`` or a bare list.  Suppresses the file
    entirely (returns []) when absent or empty -- it is optional.
    """
    import os
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read zone defaults %s: %s", path, exc)
        return []
    zones = data.get("zones") if isinstance(data, dict) else data
    return zones if isinstance(zones, list) else []


def seed_zones(conn, store: ZoneStore, path: str) -> int:
    """Seed zones from ``path`` when the table is empty.  Returns rows added."""
    if db.list_zones(conn):
        return 0
    added = 0
    for cfg in load_zone_defaults(path):
        try:
            store.add(
                zone_name=cfg.get("zone_name"),
                camera=cfg.get("camera", ""),
                polygon=cfg.get("polygon"),
                allowed=cfg.get("allowed_employees"),
                schedule=cfg.get("schedule"),
                policy=cfg.get("alert_policy", POLICY_INTRUSION),
            )
            added += 1
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping bad zone %r: %s", cfg.get("zone_name"), exc)
    if added:
        logger.info("Seeded %d security zone(s).", added)
    return added