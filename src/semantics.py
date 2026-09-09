"""Employee activity semantics (Phase 33, D2).

A shared, read-only vocabulary for what a row/record actually means.  It exists
to keep the many subsystems honest about the *semantic* of each record:

* ``OBSERVED``          -- a direct, sensor-derived observation (e.g. YOLO
                          presence).  This is the only class that implies
                          physical attendance on camera.
* ``INFERRED``          -- derived from logic over observations (e.g. AWAY time
                          from absence of observation).  Never implies
                          attendance.
* ``SECURITY_EVENT``    -- a security-detector event (intrusion, unknown
                          presence, tamper, offline).  **Never** used to judge
                          employee behaviour or productivity.
* ``PRODUCTIVITY_STATE`` -- an employee productivity classification
                          (ACTIVE/ON_PHONE/AWAY).  Computed by the tracker only.
* ``IDENTITY``          -- a face-recognition identity decision (employee id /
                          Unknown).  Confidence-dependent; weak match never
                          auto-promotes to employee.

This module is intentionally dependency-light (no DB/config) so it can be
imported anywhere and unit-tested in isolation.  It never mutates data.
"""

from __future__ import annotations

OBSERVED = "OBSERVED"
INFERRED = "INFERRED"
SECURITY_EVENT = "SECURITY_EVENT"
PRODUCTIVITY_STATE = "PRODUCTIVITY_STATE"
IDENTITY = "IDENTITY"

# Security event types that are pure SECURITY_EVENT semantics and must never be
# treated as a productivity/attendance signal.
SECURITY_EVENT_TYPES = frozenset({
    "UNKNOWN_PRESENCE", "INTRUSION", "AFTER_HOURS_ACTIVITY",
    "CAMERA_OFFLINE", "CAMERA_RECOVERED",
    "CAMERA_TAMPER_SUSPECTED", "CAMERA_TAMPER_CLEARED",
    "LOITERING_SUSPECTED", "UNUSUAL_OCCUPANCY",
    "OBJECT_LEFT_BEHIND", "ASSET_REMOVAL_SUSPECTED",
    "FALL_SUSPECTED", "FIGHT_SUSPECTED", "FIRE_SMOKE_SUSPECTED",
})

# Employee states that are PRODUCTIVITY_STATE semantics.
PRODUCTIVITY_STATES = frozenset({"ACTIVE", "ON_PHONE", "AWAY"})


def classify(record: dict) -> str:
    """Classify a generic record dict into one semantic class.

    Order of precedence: IDENTITY > SECURITY_EVENT > PRODUCTIVITY_STATE >
    OBSERVED/INFERRED.  ``record`` is a dict such as an event/interval row.
    """
    if not record:
        return INFERRED

    event_type = record.get("event_type")
    state = record.get("state")

    if event_type in SECURITY_EVENT_TYPES:
        return SECURITY_EVENT

    if state in PRODUCTIVITY_STATES:
        return PRODUCTIVITY_STATE

    if event_type in ("ARRIVED", "LEFT"):
        return OBSERVED

    # Default: rows without an explicit security/productivity marker are
    # inferred unless flagged observed.
    return OBSERVED if record.get("observed") else INFERRED


def is_security(record: dict) -> bool:
    return classify(record) == SECURITY_EVENT


def is_productivity(record: dict) -> bool:
    return classify(record) == PRODUCTIVITY_STATE


def semantics_summary(records: list[dict]) -> dict[str, int]:
    """Count records by semantic class (for dashboards / health checks)."""
    out = {OBSERVED: 0, INFERRED: 0, SECURITY_EVENT: 0,
           PRODUCTIVITY_STATE: 0, IDENTITY: 0}
    for r in records or []:
        out[classify(r)] = out.get(classify(r), 0) + 1
    return out
