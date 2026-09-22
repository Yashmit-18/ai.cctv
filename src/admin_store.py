"""Centralized Admin Control Center configuration layer (Phase 62).

Purpose
-------
Give the administrator ONE persistent, dashboard-driven source of truth for
office intelligence configuration that is NOT code: schedule (timezone, office
hours, lunch), threshold knobs (away / phone / talking), the Phase 63
phone-detection pass (class / model path / resolution / cadence / evidence
window) and safe assignment validation for employees <-> desks (seat zones) <-> 
chairs.

Architecture contract
---------------------
* The Admin Control Center is CONFIGURATION ONLY.  It never becomes a second
  identity engine, tracker, productivity engine, camera manager, schedule
  engine or state machine.
* Configuration flows DOWN into the existing runtime services:

      ADMIN PANEL -> settings table (SQLite) -> config module attrs
                  -> existing runtime (OfficeSchedule / FUTURE_* knobs)

  The existing ``OfficeSchedule`` consumes the persisted schedule through the
  project's own ``config`` module attributes (the same knobs it already reads),
  so there is exactly ONE authoritative schedule source, and the Phase 59c
  reserved ``FUTURE_AWAY/PHONE/TALKING_THRESHOLD_SECONDS`` knobs become the
  persisted thresholds the admin manages.
* No new KV store / database: ``settings`` lives in the existing application
  SQLite file (see ``src.database``).
* Identity safety is preserved: assignment validations below never turn
  UNKNOWN into an employee and never let desk/chair context override a
  face-confirmed identity.
"""

from __future__ import annotations

import zoneinfo
from dataclasses import dataclass

import config  # noqa: E402  (config lives at PROJECT ROOT)
from src import database as db
from src.office_schedule import parse_hhmm

# ----------------------------------------------------------------------
# Defaults (must mirror the Phase 59b/59c env defaults exactly)
# ----------------------------------------------------------------------

OFFICE_TZ_DEFAULT = "Asia/Kolkata"
OFFICE_START_DEFAULT = "10:00"
OFFICE_END_DEFAULT = "18:30"
LUNCH_START_DEFAULT = "14:00"
LUNCH_END_DEFAULT = "14:35"

AWAY_SECONDS_DEFAULT = 10.0
PHONE_SECONDS_DEFAULT = 5.0
TALKING_SECONDS_DEFAULT = 15.0

THRESHOLD_DEFAULTS = {
    "away_seconds": AWAY_SECONDS_DEFAULT,
    "phone_seconds": PHONE_SECONDS_DEFAULT,
    "talking_seconds": TALKING_SECONDS_DEFAULT,
}

# Defaults for the Phase 63 phone-detection pass (must mirror the config.py
# env defaults exactly -- PHONE_CLASS / PHONE_MODEL_PATH / PHONE_IMGSZ /
# PHONE_DETECT_CADENCE / PHONE_EVIDENCE_WINDOW / PHONE_EVIDENCE_MIN).
PHONE_CLASS_DEFAULT = 67
PHONE_MODEL_PATH_DEFAULT = ""
PHONE_IMGSZ_DEFAULT = 1280
PHONE_CADENCE_DEFAULT = 3
PHONE_EVIDENCE_WINDOW_DEFAULT = 5
PHONE_EVIDENCE_MIN_DEFAULT = 2

# settings keys persisted in the ``settings`` table
KEY_TZ = "office.timezone"
KEY_OFFICE_START = "office.start"
KEY_OFFICE_END = "office.end"
KEY_LUNCH_START = "lunch.start"
KEY_LUNCH_END = "lunch.end"
KEY_AWAY = "thresholds.away_seconds"
KEY_PHONE = "thresholds.phone_seconds"
KEY_TALKING = "thresholds.talking_seconds"
KEY_PHONE_CLASS = "detection.phone_class"
KEY_PHONE_MODEL_PATH = "detection.phone_model_path"
KEY_PHONE_IMGSZ = "detection.phone_imgsz"
KEY_PHONE_CADENCE = "detection.phone_cadence"
KEY_PHONE_EVIDENCE_WINDOW = "detection.phone_evidence_window"
KEY_PHONE_EVIDENCE_MIN = "detection.phone_evidence_min"

SETTING_KEYS: tuple[str, ...] = (
    KEY_TZ, KEY_OFFICE_START, KEY_OFFICE_END, KEY_LUNCH_START, KEY_LUNCH_END,
    KEY_AWAY, KEY_PHONE, KEY_TALKING,
    KEY_PHONE_CLASS, KEY_PHONE_MODEL_PATH, KEY_PHONE_IMGSZ, KEY_PHONE_CADENCE,
    KEY_PHONE_EVIDENCE_WINDOW, KEY_PHONE_EVIDENCE_MIN,
)


@dataclass
class OfficeSettings:
    """The centralized, persisted office configuration snapshot.

    All values are validated by :func:`validate_office_settings` before any
    write.  ``away/phone/talking_seconds`` mirror the Phase 59c reserved
    ``FUTURE_*`` threshold knobs (10 / 5 / 15 by default).
    """

    tz: str = OFFICE_TZ_DEFAULT
    office_start: str = OFFICE_START_DEFAULT
    office_end: str = OFFICE_END_DEFAULT
    lunch_start: str = LUNCH_START_DEFAULT
    lunch_end: str = LUNCH_END_DEFAULT
    away_seconds: float = AWAY_SECONDS_DEFAULT
    phone_seconds: float = PHONE_SECONDS_DEFAULT
    talking_seconds: float = TALKING_SECONDS_DEFAULT
    phone_class: int = PHONE_CLASS_DEFAULT
    phone_model_path: str = PHONE_MODEL_PATH_DEFAULT
    phone_imgsz: int = PHONE_IMGSZ_DEFAULT
    phone_cadence: int = PHONE_CADENCE_DEFAULT
    phone_evidence_window: int = PHONE_EVIDENCE_WINDOW_DEFAULT
    phone_evidence_min: int = PHONE_EVIDENCE_MIN_DEFAULT

    def to_dict(self) -> dict:
        return {
            "timezone": self.tz,
            "office_start": self.office_start,
            "office_end": self.office_end,
            "lunch_start": self.lunch_start,
            "lunch_end": self.lunch_end,
            "away_seconds": self.away_seconds,
            "phone_seconds": self.phone_seconds,
            "talking_seconds": self.talking_seconds,
            "phone_class": self.phone_class,
            "phone_model_path": self.phone_model_path,
            "phone_imgsz": self.phone_imgsz,
            "phone_cadence": self.phone_cadence,
            "phone_evidence_window": self.phone_evidence_window,
            "phone_evidence_min": self.phone_evidence_min,
        }

    def schedule_kwargs(self) -> dict:
        """Kwargs accepted by the existing ``OfficeSchedule`` engine."""
        return {
            "tz": self.tz,
            "start": self.office_start,
            "end": self.office_end,
            "lunch_start": self.lunch_start,
            "lunch_end": self.lunch_end,
        }


# ----------------------------------------------------------------------
# Validation (clear errors; never silently mutates user entries)
# ----------------------------------------------------------------------

def validate_timezone(name: str) -> str:
    """Return the zone name if it resolves, else raise ``ValueError``."""
    text = str(name or "").strip()
    try:
        zoneinfo.ZoneInfo(text)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        raise ValueError(
            f"{text!r} is not a valid IANA timezone (e.g. Asia/Kolkata)") from None
    return text


def validate_threshold_seconds(name: str, value) -> float:
    """Non-negative numeric threshold.  Raises ``ValueError``.

    Phase 59c reserved knobs require ``>= 0`` (a value of ``0`` here means
    "instantly" -- allowed by the existing runtime).
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if num < 0.0:
        raise ValueError(f"{name} must be >= 0 (got {num:g})")
    if num != num or num == float("inf"):  # NaN / inf
        raise ValueError(f"{name} must be a finite number")
    return num


def validate_positive_int(name: str, value, *, minimum: int = 0) -> int:
    """Whole number >= ``minimum``.  Raises ``ValueError``."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if n < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got {n})")
    return n


def validate_phone_model_path(name: str, value) -> str:
    """Trimmed model path; empty string reuses the base YOLO model."""
    return str(value or "").strip()


def validate_office_settings(**kwargs) -> dict:
    """Validate a (partial) settings payload.  Returns the sanitized full set.

    Raises ``ValueError`` with a human-readable message on the FIRST problem so
    the admin UI can surface it directly.
    """
    base = OfficeSettings()
    payload = {
        "timezone": kwargs.get("timezone", base.tz),
        "office_start": kwargs.get("office_start", base.office_start),
        "office_end": kwargs.get("office_end", base.office_end),
        "lunch_start": kwargs.get("lunch_start", base.lunch_start),
        "lunch_end": kwargs.get("lunch_end", base.lunch_end),
        "away_seconds": kwargs.get("away_seconds", base.away_seconds),
        "phone_seconds": kwargs.get("phone_seconds", base.phone_seconds),
        "talking_seconds": kwargs.get("talking_seconds", base.talking_seconds),
        "phone_class": kwargs.get("phone_class", base.phone_class),
        "phone_model_path": kwargs.get("phone_model_path",
                                       base.phone_model_path),
        "phone_imgsz": kwargs.get("phone_imgsz", base.phone_imgsz),
        "phone_cadence": kwargs.get("phone_cadence", base.phone_cadence),
        "phone_evidence_window": kwargs.get(
            "phone_evidence_window", base.phone_evidence_window),
        "phone_evidence_min": kwargs.get(
            "phone_evidence_min", base.phone_evidence_min),
    }
    tz = validate_timezone(payload["timezone"])
    os_, oe = (payload["office_start"], payload["office_end"])
    ls, le = (payload["lunch_start"], payload["lunch_end"])
    start = parse_hhmm(os_, knob="office start")
    end = parse_hhmm(oe, knob="office end")
    if end <= start:
        raise ValueError(
            f"office hours must end after they start ({os_} -> {oe})")
    lunch_start = parse_hhmm(ls, knob="lunch start")
    lunch_end = parse_hhmm(le, knob="lunch end")
    if lunch_end <= lunch_start:
        raise ValueError(
            f"lunch must end after it starts ({ls} -> {le})")
    # lunch is normally inside office hours -- reject clearly-invalid config
    if lunch_start < start or lunch_end > end:
        raise ValueError(
            f"lunch window ({ls}-{le}) must lie inside office hours "
            f"({os_}-{oe})")
    phone_class = validate_positive_int(
        "phone_class", payload["phone_class"])
    phone_imgsz = validate_positive_int(
        "phone_imgsz", payload["phone_imgsz"], minimum=32)
    phone_cadence = validate_positive_int(
        "phone_cadence", payload["phone_cadence"], minimum=1)
    phone_evidence_window = validate_positive_int(
        "phone_evidence_window", payload["phone_evidence_window"], minimum=1)
    phone_evidence_min = validate_positive_int(
        "phone_evidence_min", payload["phone_evidence_min"], minimum=1)
    if phone_evidence_min > phone_evidence_window:
        raise ValueError(
            f"phone evidence min ({phone_evidence_min}) must be <= "
            f"phone evidence window ({phone_evidence_window})")
    return {
        "timezone": tz,
        "office_start": os_,
        "office_end": oe,
        "lunch_start": ls,
        "lunch_end": le,
        "away_seconds": validate_threshold_seconds(
            "away_seconds", payload["away_seconds"]),
        "phone_seconds": validate_threshold_seconds(
            "phone_seconds", payload["phone_seconds"]),
        "talking_seconds": validate_threshold_seconds(
            "talking_seconds", payload["talking_seconds"]),
        "phone_class": phone_class,
        "phone_model_path": validate_phone_model_path(
            "phone_model_path", payload["phone_model_path"]),
        "phone_imgsz": phone_imgsz,
        "phone_cadence": phone_cadence,
        "phone_evidence_window": phone_evidence_window,
        "phone_evidence_min": phone_evidence_min,
    }


# ----------------------------------------------------------------------
# Settings store (SQLite ``settings``)
# ----------------------------------------------------------------------

class SettingsStore:
    """Read/write the centralized persisted office configuration.

    ``get()`` merges persisted values over the env defaults so a deployment that
    never touched the admin panel behaves exactly as before (backward
    compatible).  Writes are validated before persisting.
    """

    def __init__(self, conn):
        self.conn = conn

    # -- read -----------------------------------------------------------
    def get(self) -> OfficeSettings:
        return OfficeSettings(
            tz=db.get_setting(self.conn, KEY_TZ) or OFFICE_TZ_DEFAULT,
            office_start=db.get_setting(self.conn, KEY_OFFICE_START)
            or OFFICE_START_DEFAULT,
            office_end=db.get_setting(self.conn, KEY_OFFICE_END)
            or OFFICE_END_DEFAULT,
            lunch_start=db.get_setting(self.conn, KEY_LUNCH_START)
            or LUNCH_START_DEFAULT,
            lunch_end=db.get_setting(self.conn, KEY_LUNCH_END)
            or LUNCH_END_DEFAULT,
            away_seconds=float(db.get_setting(self.conn, KEY_AWAY)
                               or AWAY_SECONDS_DEFAULT),
            phone_seconds=float(db.get_setting(self.conn, KEY_PHONE)
                                or PHONE_SECONDS_DEFAULT),
            talking_seconds=float(db.get_setting(self.conn, KEY_TALKING)
                                  or TALKING_SECONDS_DEFAULT),
            phone_class=int(db.get_setting(self.conn, KEY_PHONE_CLASS)
                            or PHONE_CLASS_DEFAULT),
            phone_model_path=db.get_setting(self.conn, KEY_PHONE_MODEL_PATH)
            or PHONE_MODEL_PATH_DEFAULT,
            phone_imgsz=int(db.get_setting(self.conn, KEY_PHONE_IMGSZ)
                            or PHONE_IMGSZ_DEFAULT),
            phone_cadence=int(db.get_setting(self.conn, KEY_PHONE_CADENCE)
                              or PHONE_CADENCE_DEFAULT),
            phone_evidence_window=int(
                db.get_setting(self.conn, KEY_PHONE_EVIDENCE_WINDOW)
                or PHONE_EVIDENCE_WINDOW_DEFAULT),
            phone_evidence_min=int(
                db.get_setting(self.conn, KEY_PHONE_EVIDENCE_MIN)
                or PHONE_EVIDENCE_MIN_DEFAULT),
        )

    def keys_set(self) -> tuple[str, ...]:
        return tuple(k for k in SETTING_KEYS if db.get_setting(self.conn, k))

    # -- write ----------------------------------------------------------
    def update(self, *, actor: str = "system", **fields) -> OfficeSettings:
        """Merge ``fields`` into the current settings, validate, persist.

        Only keys actually changed are persisted/committed.  Returns the new
        authoritative :class:`OfficeSettings`.
        """
        base = self.get()
        merged = {
            "timezone": fields.get("timezone", base.tz),
            "office_start": fields.get("office_start", base.office_start),
            "office_end": fields.get("office_end", base.office_end),
            "lunch_start": fields.get("lunch_start", base.lunch_start),
            "lunch_end": fields.get("lunch_end", base.lunch_end),
            "away_seconds": fields.get("away_seconds", base.away_seconds),
            "phone_seconds": fields.get("phone_seconds", base.phone_seconds),
            "talking_seconds": fields.get("talking_seconds", base.talking_seconds),
            "phone_class": fields.get("phone_class", base.phone_class),
            "phone_model_path": fields.get("phone_model_path",
                                           base.phone_model_path),
            "phone_imgsz": fields.get("phone_imgsz", base.phone_imgsz),
            "phone_cadence": fields.get("phone_cadence", base.phone_cadence),
            "phone_evidence_window": fields.get(
                "phone_evidence_window", base.phone_evidence_window),
            "phone_evidence_min": fields.get(
                "phone_evidence_min", base.phone_evidence_min),
        }
        validated = validate_office_settings(**merged)
        mapping = {
            "timezone": KEY_TZ,
            "office_start": KEY_OFFICE_START,
            "office_end": KEY_OFFICE_END,
            "lunch_start": KEY_LUNCH_START,
            "lunch_end": KEY_LUNCH_END,
            "away_seconds": KEY_AWAY,
            "phone_seconds": KEY_PHONE,
            "talking_seconds": KEY_TALKING,
            "phone_class": KEY_PHONE_CLASS,
            "phone_model_path": KEY_PHONE_MODEL_PATH,
            "phone_imgsz": KEY_PHONE_IMGSZ,
            "phone_cadence": KEY_PHONE_CADENCE,
            "phone_evidence_window": KEY_PHONE_EVIDENCE_WINDOW,
            "phone_evidence_min": KEY_PHONE_EVIDENCE_MIN,
        }
        changed: list[str] = []
        for field, key in mapping.items():
            value = validated[field]
            new_text = str(value)
            old_text = db.get_setting(self.conn, key)
            if old_text is None or old_text != new_text:
                db.set_setting(self.conn, key, new_text)
                changed.append(field)
        if changed:
            db.audit(self.conn, "settings.update", actor=actor,
                     resource="office_settings",
                     detail=",".join(changed))
        return self.get()

    # -- apply to live runtime -----------------------------------------
    def apply_to_config(self) -> OfficeSettings:
        """Make the persisted settings authoritative for the running process.

        Writes the existing ``config`` module attributes (the same knobs the
        runtime already reads), so a persisted schedule/threshold/phone-pass
        change takes effect for subsequently constructed ``OfficeSchedule``,
        ``ActivityDetector`` and FSM objects ('the ONE authoritative config
        source') without restarting.

        FSM thresholds (away/phone seconds): the reserved Phase 59c
        ``FUTURE_AWAY/PHONE_THRESHOLD_SECONDS`` knobs are always written and
        ``main.py`` feeds them into the live FSM on the next startup.

        Phase 63 phone-pass knobs (``PHONE_CLASS`` / ``PHONE_MODEL_PATH`` /
        ``PHONE_IMGSZ`` / ``PHONE_DETECT_CADENCE`` / ``PHONE_EVIDENCE_WINDOW``
        / ``PHONE_EVIDENCE_MIN``) are rebound **only when the admin actually
        persisted them** -- an env-tuned deployment that never touched the
        panel keeps its exact env values (backward compatible).
        """
        s = self.get()
        keys = set(self.keys_set())
        config.CCTV_OFFICE_TZ = s.tz
        config.CCTV_OFFICE_START = s.office_start
        config.CCTV_OFFICE_END = s.office_end
        config.CCTV_LUNCH_START = s.lunch_start
        config.CCTV_LUNCH_END = s.lunch_end
        config.FUTURE_AWAY_THRESHOLD_SECONDS = s.away_seconds
        config.FUTURE_PHONE_THRESHOLD_SECONDS = s.phone_seconds
        config.FUTURE_TALKING_THRESHOLD_SECONDS = s.talking_seconds
        if KEY_PHONE_CLASS in keys:
            config.PHONE_CLASS = int(s.phone_class)
        if KEY_PHONE_MODEL_PATH in keys:
            config.PHONE_MODEL_PATH = s.phone_model_path
        if KEY_PHONE_IMGSZ in keys:
            config.PHONE_IMGSZ = int(s.phone_imgsz)
        if KEY_PHONE_CADENCE in keys:
            config.PHONE_DETECT_CADENCE = int(s.phone_cadence)
        if KEY_PHONE_EVIDENCE_WINDOW in keys:
            config.PHONE_EVIDENCE_WINDOW = int(s.phone_evidence_window)
        if KEY_PHONE_EVIDENCE_MIN in keys:
            config.PHONE_EVIDENCE_MIN = int(s.phone_evidence_min)
        return s

    def snapshot(self) -> dict:
        s = self.get()
        return {
            **s.to_dict(),
            "persisted_keys": list(self.keys_set()),
        }


# ----------------------------------------------------------------------
# Assignment validation (employee <-> desk/seat zone <-> chair)
# ----------------------------------------------------------------------

def validate_assignment(conn, *, employee_id: str | None = None,
                        zone_id: str | None = None,
                        chair_id: str | None = None) -> list[str]:
    """Sanity-check an (optional) assignment target before persisting.

    Returns a list of error strings (empty == OK).  Enforces:
      * assignment to a missing / inactive employee is invalid;
      * assignment to a disabled desk (seat zone) is invalid;
      * assignment to a disabled chair is invalid;
      * a chair belongs to an existing seat zone (invalid otherwise).

    It never converts UNKNOWN into an employee and never lets desk/chair
    context override a face-confirmed identity -- these are CONFIG hints only.
    """
    problems: list[str] = []
    if employee_id:
        emp = db.get_employee(conn, employee_id)
        if emp is None:
            problems.append(f"employee {employee_id!r} does not exist")
        elif not bool(emp.get("active", 1)):
            problems.append(
                f"employee {employee_id!r} is disabled; enable it first")
    if zone_id:
        zone = db.get_seat_zone(conn, zone_id)
        if zone is None:
            problems.append(f"desk/seat zone {zone_id!r} does not exist")
        elif not bool(zone.get("enabled", 1)):
            problems.append(
                f"desk/seat zone {zone_id!r} is disabled; enable it first")
    if chair_id:
        chair = db.get_chair(conn, chair_id)
        if chair is None:
            problems.append(f"chair {chair_id!r} does not exist")
        else:
            if not bool(chair.get("enabled", 1)):
                problems.append(
                    f"chair {chair_id!r} is disabled; enable it first")
            zone_of = db.get_seat_zone(conn, chair.get("zone_id") or "")
            if zone_of is None:
                problems.append(
                    f"chair {chair_id!r} references unknown zone "
                    f"{chair.get('zone_id')!r}")
    return problems