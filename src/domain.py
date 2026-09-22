"""Domain models and constants for the CCTV productivity tracker.

Central, dependency-light definitions shared across the daemon, reporter,
dashboard and tests.  Kept free of business logic so it can be imported
everywhere without side effects (no config / env loading here).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Employee states (dynamic face-recognition era)
# ----------------------------------------------------------------------
ACTIVE: str = "ACTIVE"
ON_PHONE: str = "ON_PHONE"
AWAY: str = "AWAY"

# Sentinel employee id for faces seen but not recognized.
UNKNOWN_ID: str = "Unknown"

# Ordered state list for display / aggregation.
STATE_ORDER: tuple[str, ...] = (ACTIVE, ON_PHONE, AWAY)


def redact_url(url: str) -> str:
    """Strip credentials (user:pass@) from a URL string for safe display/logs.

    e.g. ``rtsp://admin:secret@host/stream`` -> ``rtsp://admin:***@host/stream``
    """
    s = str(url)
    try:
        scheme, _, rest = s.partition("://")
        if not rest:
            return s
        authority, sep, path = rest.partition("/")
        if "@" in authority:
            userinfo, _, hostport = authority.rpartition("@")
            user = userinfo.split(":", 1)[0] if userinfo else ""
            authority = f"{user}:***@{hostport}" if user else f"***@{hostport}"
        return f"{scheme}://{authority}{sep}{path}"
    except Exception:
        return s

# ----------------------------------------------------------------------
# Camera health states
# ----------------------------------------------------------------------
CAM_ONLINE: str = "ONLINE"
CAM_OFFLINE: str = "OFFLINE"
CAM_RECONNECTING: str = "RECONNECTING"
CAM_NO_FRAME: str = "NO_FRAME"
# Phase A (A12): degraded-but-flowing states a live camera can land in.
CAM_FROZEN: str = "FROZEN_FRAME"
CAM_LOW_FPS: str = "LOW_FPS"
CAM_DEGRADED: str = "DEGRADED"  # umbrella: FROZEN_FRAME / LOW_FPS / NO_FRAME
# Phase 58: frame-content gate used when a live camera is producing a dark or
# blank feed (mean brightness below BLACK_FRAME_MEAN).  Never grounds an
# employee AWAY -- the feed simply cannot be trusted for productivity states.
CAM_DARK_BLANK_FRAME: str = "DARK_BLANK_FRAME"

CAMERA_HEALTH_ORDER: tuple[str, ...] = (
    CAM_ONLINE, CAM_LOW_FPS, CAM_FROZEN, CAM_NO_FRAME, CAM_RECONNECTING,
    CAM_OFFLINE,
)

DEGRADED_STATES: tuple[str, ...] = (CAM_FROZEN, CAM_LOW_FPS, CAM_NO_FRAME)


# ----------------------------------------------------------------------
# Camera source kinds  (Phase 58)
# ----------------------------------------------------------------------
SourceKind = str


class CameraSourceKind(str, enum.Enum):
    """The kind of video source a camera is fed by.

    All kinds ultimately deliver the same type of frame to the AI pipeline
    (the common camera-source abstraction is ``VideoCapture`` in
    ``src/camera.py``); this only tags *how* the source is opened.
    """

    LOCAL = "local"
    RTSP = "rtsp"
    VIDEO_FILE = "video_file"
    TEST = "test"

    @classmethod
    def infer(cls, source) -> "CameraSourceKind":
        """Best-effort kind guess for  a raw ``source`` value.

        * int / numeric string -> LOCAL (device index)
        * string starting ``rtsp:`` -> RTSP
        * ``test`` reserved marker -> TEST
        * anything else (e.g. ``.mp4`` path) -> VIDEO_FILE
        """
        s = str(source).strip().lower()
        if isinstance(source, int) or s.isdigit():
            return cls.LOCAL
        if s in ("webcam", "local", "laptop", "usb"):
            return cls.LOCAL
        if s.startswith("rtsp:"):
            return cls.RTSP
        if s == "test":
            return cls.TEST
        return cls.VIDEO_FILE


@dataclass(frozen=True)
class CameraConfig:
    """Production-safe per-camera configuration (Phase 58).

    Never expose ``password`` beyond construction: ``display()`` and
    ``safe_dict()`` are the ONLY approved outward shapes and they never
    include credentials or fully-authenticated URLs.
    """

    camera_id: str
    name: str
    kind: CameraSourceKind = CameraSourceKind.RTSP
    url: str = ""                       # raw RTSP/file source (may embed creds)
    username: str = ""
    password: str = ""
    enabled: bool = True
    location: str = ""
    fps_target: float = 0.0             # 0 = use system default
    reconnect_base: float = 2.0
    reconnect_max: float = 30.0
    reconnect_factor: float = 2.5

    def display(self) -> str:
        """Single-line, credential-free summary for logs/reports."""
        if self.kind is CameraSourceKind.RTSP:
            return (f"{self.camera_id} ({self.name or self.camera_id}) "
                    f"RTSP host={self._rtsp_host()} credentials=REDACTED")
        return f"{self.camera_id} ({self.name or self.camera_id}) {self.kind.value}"

    def safe_dict(self) -> dict:
        """Serializable config snapshot with credentials redacted."""
        return {
            "id": self.camera_id,
            "name": self.name,
            "kind": self.kind.value,
            "enabled": self.enabled,
            "location": self.location,
            "fps_target": self.fps_target,
            "reconnect_base": self.reconnect_base,
            "reconnect_max": self.reconnect_max,
            "reconnect_factor": self.reconnect_factor,
            "host": self._rtsp_host() if self.kind is CameraSourceKind.RTSP else "",
            "credentials": "REDACTED",
        }

    def _rtsp_host(self) -> str:
        """Host only (no credentials) from an RTSP URL, or '' if unparseable."""
        s = self.url or ""
        if s.startswith("rtsp://"):
            rest = s[len("rtsp://"):]
            authority = rest.split("/", 1)[0]
            if "@" in authority:
                authority = authority.rsplit("@", 1)[-1]
            return f"rtsp://{authority}"
        return ""

# ----------------------------------------------------------------------
# Security event types  (Phase 31)
# ----------------------------------------------------------------------
UNKNOWN_PRESENCE = "UNKNOWN_PRESENCE"
INTRUSION = "INTRUSION"
AFTER_HOURS_ACTIVITY = "AFTER_HOURS_ACTIVITY"
CAMERA_OFFLINE = "CAMERA_OFFLINE"
CAMERA_RECOVERED = "CAMERA_RECOVERED"
CAMERA_TAMPER_SUSPECTED = "CAMERA_TAMPER_SUSPECTED"
CAMERA_TAMPER_CLEARED = "CAMERA_TAMPER_CLEARED"
LOITERING_SUSPECTED = "LOITERING_SUSPECTED"
UNUSUAL_OCCUPANCY = "UNUSUAL_OCCUPANCY"
ARRIVED = "ARRIVED"
LEFT = "LEFT"
DETECTOR_UNAVAILABLE = "DETECTOR_UNAVAILABLE"

# Phase-A core-vision events (never fabricated; produced only by real
# detectors -- see src/motion.py and src/entry_exit.py).
# MOTION_DETECTED -- frame-level motion signal; advisory only, never treated
#   as suspicious on its own by the correlation engine.
MOTION_DETECTED = "MOTION_DETECTED"
# ENTRY/EXIT_CROSSING -- a spatial track crossed a configured virtual line.
ENTRY_CROSSING = "ENTRY_CROSSING"
EXIT_CROSSING = "EXIT_CROSSING"
# UNEXPECTED_STAY -- opt-in heuristic: an employee entered via a line
#   crossing but no exit crossing was seen before the configured window
#   elapsed.  LOW severity, never accusatory.
UNEXPECTED_STAY = "UNEXPECTED_STAY"

# Reserved for future model / detector-plugin events (never fabricated).
OBJECT_LEFT_BEHIND = "OBJECT_LEFT_BEHIND"
ASSET_REMOVAL_SUSPECTED = "ASSET_REMOVAL_SUSPECTED"
FALL_SUSPECTED = "FALL_SUSPECTED"
FIGHT_SUSPECTED = "FIGHT_SUSPECTED"
FIRE_SMOKE_SUSPECTED = "FIRE_SMOKE_SUSPECTED"

SECURITY_EVENT_TYPES: tuple[str, ...] = (
    UNKNOWN_PRESENCE, INTRUSION, AFTER_HOURS_ACTIVITY,
    CAMERA_OFFLINE, CAMERA_RECOVERED,
    CAMERA_TAMPER_SUSPECTED, CAMERA_TAMPER_CLEARED,
    LOITERING_SUSPECTED, UNUSUAL_OCCUPANCY, ARRIVED, LEFT,
    DETECTOR_UNAVAILABLE,
    MOTION_DETECTED, ENTRY_CROSSING, EXIT_CROSSING, UNEXPECTED_STAY,
    OBJECT_LEFT_BEHIND, ASSET_REMOVAL_SUSPECTED,
    FALL_SUSPECTED, FIGHT_SUSPECTED, FIRE_SMOKE_SUSPECTED,
)

# ----------------------------------------------------------------------
# Incident severity / status (canonical)
# ----------------------------------------------------------------------
SEV_INFO = "INFO"
SEV_LOW = "LOW"
SEV_MEDIUM = "MEDIUM"
SEV_HIGH = "HIGH"
SEV_CRITICAL = "CRITICAL"

SEVERITY_ORDER: tuple[str, ...] = (SEV_INFO, SEV_LOW, SEV_MEDIUM, SEV_HIGH, SEV_CRITICAL)

INCIDENT_OPEN = "OPEN"
INCIDENT_ACKNOWLEDGED = "ACKNOWLEDGED"
INCIDENT_RESOLVED = "RESOLVED"
INCIDENT_DISMISSED = "DISMISSED"

INCIDENT_STATUS_ORDER: tuple[str, ...] = (
    INCIDENT_OPEN, INCIDENT_ACKNOWLEDGED, INCIDENT_RESOLVED, INCIDENT_DISMISSED,
)

# ----------------------------------------------------------------------
# Investigation / review state  (Phase 32)
# ----------------------------------------------------------------------
# Operators move an incident through human review phases.  A machine only
# ever DETECTS / SUSPECTS; it never labels a person as an offender.  Only an
# operator may reach CONFIRMED_BY_OPERATOR.  DISMISSED = operator false positive.
REVIEW_DETECTED = "DETECTED"
REVIEW_SUSPECTED = "SUSPECTED"
REVIEW_REQUIRES_REVIEW = "REQUIRES_REVIEW"
REVIEW_CONFIRMED = "CONFIRMED_BY_OPERATOR"
REVIEW_DISMISSED = "DISMISSED"

REVIEW_STATES: tuple[str, ...] = (
    REVIEW_DETECTED, REVIEW_SUSPECTED, REVIEW_REQUIRES_REVIEW,
    REVIEW_CONFIRMED, REVIEW_DISMISSED,
)

# Default severity per event type (overridable by alert rules / zone policy).
DEFAULT_EVENT_SEVERITY: dict[str, str] = {
    UNKNOWN_PRESENCE: SEV_MEDIUM,
    INTRUSION: SEV_HIGH,
    AFTER_HOURS_ACTIVITY: SEV_MEDIUM,
    CAMERA_OFFLINE: SEV_HIGH,
    CAMERA_RECOVERED: SEV_INFO,
    CAMERA_TAMPER_SUSPECTED: SEV_HIGH,
    CAMERA_TAMPER_CLEARED: SEV_INFO,
    LOITERING_SUSPECTED: SEV_MEDIUM,
    UNUSUAL_OCCUPANCY: SEV_MEDIUM,
    ARRIVED: SEV_INFO,
    LEFT: SEV_INFO,
    DETECTOR_UNAVAILABLE: SEV_HIGH,
    MOTION_DETECTED: SEV_INFO,
    ENTRY_CROSSING: SEV_INFO,
    EXIT_CROSSING: SEV_INFO,
    UNEXPECTED_STAY: SEV_LOW,
}

# ----------------------------------------------------------------------
# Evidence recording modes  (Phase 31)
# ----------------------------------------------------------------------
RECORD_OFF = "OFF"
RECORD_EVENT_ONLY = "EVENT_ONLY"
RECORD_HIGH_SEVERITY_ONLY = "HIGH_SEVERITY_ONLY"
RECORD_CONTINUOUS = "CONTINUOUS"  # reserved; never silently enabled

RECORDING_MODES: tuple[str, ...] = (
    RECORD_OFF, RECORD_EVENT_ONLY, RECORD_HIGH_SEVERITY_ONLY, RECORD_CONTINUOUS,
)

# ----------------------------------------------------------------------
# Dashboard security roles
# ----------------------------------------------------------------------
ROLE_ADMIN = "admin"
ROLE_SECURITY_OPERATOR = "security_operator"
ROLE_MANAGER = "manager"
ROLE_VIEWER = "viewer"

SECURITY_ROLES: tuple[str, ...] = (ROLE_ADMIN, ROLE_SECURITY_OPERATOR, ROLE_MANAGER, ROLE_VIEWER)

# Permission vocabulary (Phase 32).  A role grants a subset of these.
PERM_VIEW = "view"
PERM_INVESTIGATE = "investigate"
PERM_ACKNOWLEDGE = "acknowledge"
PERM_RESOLVE = "resolve"
PERM_DISMISS = "dismiss"
PERM_EXPORT = "export"
PERM_VIEW_EVIDENCE = "view_evidence"
PERM_DELETE_EVIDENCE = "delete_evidence"
PERM_CONFIGURE_CAMERAS = "configure_cameras"
PERM_CONFIGURE_ZONES = "configure_zones"
PERM_CONFIGURE_ALERTS = "configure_alerts"
PERM_MANAGE_USERS = "manage_users"
PERM_CONFIGURE_SETTINGS = "configure_settings"

# Default permission set per role.  UI checks are a convenience, NOT a
# security boundary -- privileged actions are audited and the dashboard
# should sit behind a reverse proxy / VPN for real enforcement.
ROLE_PERMISSIONS: dict[str, tuple[str, ...]] = {
    ROLE_ADMIN: (
        PERM_VIEW, PERM_INVESTIGATE, PERM_ACKNOWLEDGE, PERM_RESOLVE,
        PERM_DISMISS, PERM_EXPORT, PERM_VIEW_EVIDENCE, PERM_DELETE_EVIDENCE,
        PERM_CONFIGURE_CAMERAS, PERM_CONFIGURE_ZONES, PERM_CONFIGURE_ALERTS,
        PERM_MANAGE_USERS, PERM_CONFIGURE_SETTINGS,
    ),
    ROLE_SECURITY_OPERATOR: (
        PERM_VIEW, PERM_INVESTIGATE, PERM_ACKNOWLEDGE, PERM_RESOLVE,
        PERM_DISMISS, PERM_EXPORT, PERM_VIEW_EVIDENCE,
    ),
    ROLE_MANAGER: (
        PERM_VIEW, PERM_INVESTIGATE, PERM_EXPORT,
    ),
    ROLE_VIEWER: (
        PERM_VIEW,
    ),
}


def role_has_permission(role: str | None, permission: str) -> bool:
    """True when ``role`` grants ``permission``.

    Unknown / missing roles are treated as the most-restrictive VIEWER, never
    escalated to admin (defense-in-depth: a lost session role must not grant
    privileged permissions).
    """
    role = (role or ROLE_VIEWER).lower()
    return permission in ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS[ROLE_VIEWER])


# ----------------------------------------------------------------------
# Detector health status  (Phase 32)
# ----------------------------------------------------------------------
DET_HEALTHY = "HEALTHY"
DET_DEGRADED = "DEGRADED"
DET_UNAVAILABLE = "UNAVAILABLE"      # FUTURE_MODEL_REQUIRED etc.
DET_FAILED = "FAILED"

SYS_HEALTHY = "HEALTHY"
SYS_DEGRADED = "DEGRADED"
SYS_FAILED = "FAILED"
SYS_NOT_CONFIGURED = "NOT_CONFIGURED"

CAM_STATUS_HEALTHY = "HEALTHY"
CAM_STATUS_DEGRADED = "DEGRADED"
CAM_STATUS_TAMPER_SUSPECTED = "TAMPER_SUSPECTED"


@dataclass
class Employee:
    """An enrolled person in the monitoring system.

    ``enrolled`` reflects whether usable face embeddings exist for the id.
    Sensitive biometric data is never stored here -- only reference ids and
    metadata.
    """

    employee_id: str
    name: str = ""
    department: str = ""
    designation: str = ""
    active: bool = True
    enrolled: bool = False
    display_name: str = ""
    schedule: "WorkSchedule | None" = None
    metadata: dict = field(default_factory=dict)


@dataclass
class WorkSchedule:
    """Expected working window used for productivity scoring.

    Times are minutes-from-midnight (local).  Dictionary form is also
    accepted for config parsing.
    """

    start_min: int = 9 * 60          # 09:00
    end_min: int = 18 * 60           # 18:00
    grace_period_min: int = 15       # tolerated lateness (not penalized)
    lunch_min: tuple[int, int] = (12 * 60, 13 * 60)   # unpaid window
    break_minutes: int = 30          # additional paid breaks allowed

    @classmethod
    def from_config(cls, cfg: dict | None) -> "WorkSchedule | None":
        if cfg is None:
            return None
        def _to_min(v, default):
            if v is None:
                return default
            v = str(v)
            return _hhmm(v) if ":" in v else int(v)
        lunch = cfg.get("lunch", (12 * 60, 13 * 60))
        if isinstance(lunch, (list, tuple)) and len(lunch) == 2:
            lunch = (_to_min(lunch[0], 12 * 60), _to_min(lunch[1], 13 * 60))
        elif isinstance(lunch, str):
            a, b = lunch.split("-", 1)
            lunch = (_to_min(a, 12 * 60), _to_min(b, 13 * 60))
        return cls(
            start_min=_to_min(cfg.get("start"), 9 * 60),
            end_min=_to_min(cfg.get("end"), 18 * 60),
            grace_period_min=_to_min(cfg.get("grace"), 15),
            lunch_min=tuple(sorted(int(x) for x in lunch)),
            break_minutes=_to_min(cfg.get("breaks"), 30),
        )


def _hhmm(value: str) -> int:
    h, m = value.split(":")
    return int(h) * 60 + int(m)


def _hhmm_to_minutes(value: str) -> int:
    """Parse ``'HH:MM'`` (or int minutes) into minutes-from-midnight."""
    if isinstance(value, int):
        return value
    v = str(value).strip()
    if not v:
        return 0
    try:
        return _hhmm(v)
    except ValueError:
        return int(v)


def is_time_in_window(clock_minutes: int, start: str | None = None,
                      end: str | None = None) -> bool:
    """Return True when ``clock_minutes`` (minutes since midnight) falls inside
    ``[start, end)`` (``'HH:MM'``).  ``None`` start/end => always True (24x7).

    Handles overnight windows (``22:00-06:00``): True when after start OR
    before end.
    """
    if start is None or end is None:
        return True
    s = _hhmm_to_minutes(start)
    e = _hhmm_to_minutes(end)
    if s == e:
        return True  # 24x7 degenerate window
    if s < e:
        return s <= clock_minutes < e
    return clock_minutes >= s or clock_minutes < e
