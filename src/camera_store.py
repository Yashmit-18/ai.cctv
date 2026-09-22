"""Persistent Admin Camera Management (Phase 61).

Purpose
-------
Give the administrator a persistent, dashboard-driven way to configure the
runtime camera pool (IP CCTV / RTSP / local / video-file), so real cameras can
be added, edited, enabled/disabled and connection-tested without editing
``.env`` and without a code restart.

Design contract
---------------
* The runtime camera manager stays THE single manager: ``MultiCameraManager``.
  Its ``CameraConfig`` model and ``VideoCapture`` transport are reused
  exactly---nothing here re-implements capture, health or RTSP.
* ONE authoritative runtime configuration.  Precedence:

      persistent admin cameras (SQLite `admin_cameras`)
                  > down to
      environment CameraConfigs (Phase 58) -- bootstrap/default/fallback

  A camera id managed in the store wins; ids absent from the store fall back
  to the environment so existing Phase 58 deployments keep working untouched.
  When the store is empty the environment configuration is used verbatim.
* Credentials live in the store ONLY so the transport boundary can
  recompose the RTSP URL.  Every outward shape (``safe_dict``, ``display``,
  ``repr``, revert logs, test result, dashboard table) redacts them.

Apply / reload protocol
-----------------------
The dashboard writes an "apply request" (revision) into the shared SQLite
``camera_apply_state`` row.  The daemon's loop calls :func:`reconcile_runtime`
periodically, which applies changes whenever the store revision changes or an
explicit request is pending, recording the applied revision/timestamp.  The
manager-reconcile path (``MultiCameraManager.apply_configs``) stops removed/
disabled cameras, re-opens changed sources, starts new ones and leaves
unaffected cameras untouched.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from urllib.parse import quote, urlparse

from src import database as db
from src.camera import VideoCapture
from src.domain import CameraConfig, CameraSourceKind, redact_url

logger = logging.getLogger("cctv.camera_store")

# Sentinel used in every outward shape where a secret WOULD have appeared.
REDACTED = "********"

ALLOWED_KINDS: tuple[str, ...] = tuple(k.value for k in CameraSourceKind)

_CAM_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

# Columns that may be updated on an existing camera (camera_id is the stable
# identity and can never be rewritten by an edit).
_UPDATE_COLS = (
    "name", "kind", "enabled", "location", "url", "username", "password",
    "fps_target", "reconnect_base", "reconnect_max", "reconnect_factor",
)
_COL_TO_DB = {
    "name": "name", "kind": "kind", "enabled": "enabled", "location": "location",
    "url": "url", "username": "username", "password": "password",
    "fps_target": "fps_target", "reconnect_base": "reconnect_base",
    "reconnect_max": "reconnect_max", "reconnect_factor": "reconnect_factor",
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _host_of(url: str) -> str:
    """Host (no credentials) from a URL string, or ''."""

    def _clean(s: str) -> str:
        authority = s.split("/", 1)[0]
        if "@" in authority:
            authority = authority.rsplit("@", 1)[-1]
        return authority

    s = str(url or "")
    if "://" in s:
        scheme, _, rest = s.partition("://")
        return f"{scheme}://{_clean(rest)}"
    return _clean(s) if s else ""


def compose_rtsp_url(url: str, username: str = "", password: str = "") -> str:
    """Fold split credentials into an RTSP URL at the transport boundary.

    Never mutates or rewrites the stored base URL: it returns a fresh string
    used only to open the source.  Credentials are percent-encoded so special
    characters (``@ : / # % ?``) cannot break the URL.  A URL that already
    embeds ``user:pass@`` is left untouched (existing credentials win).
    """
    s = str(url or "")
    if not s or not username:
        return s
    try:
        scheme, _, rest = s.partition("://")
        if not rest:
            return s
        authority, sep, path = rest.partition("/")
        if "@" in authority:
            return s
        user = quote(username, safe="")
        pw = quote(password, safe="") if password else ""
        userinfo = f"{user}:{pw}" if pw else user
        return f"{scheme}://{userinfo}@{authority}{sep}{path}"
    except Exception:  # pragma: no cover - defensive; never leak secrets
        return s


# ----------------------------------------------------------------------
# Validation (rule 18 -- syntax only, never reachability)
# ----------------------------------------------------------------------

def validate_camera_fields(*, camera_id, name, kind, enabled=True,
                           location="", url="", username="", password="",
                           fps_target=0.0, reconnect_base=2.0,
                           reconnect_max=30.0, reconnect_factor=2.5) -> dict:
    """Validate the persistent camera fields.  Raises ``ValueError``.

    Only *syntax* is validated here: an ``rtsp://...`` URL that parses is
    accepted even when no device answers it (reachability is checked live by
    :func:`test_camera_connection`, never guessed here).
    """
    camera_id = str(camera_id or "").strip()
    if not camera_id:
        raise ValueError("camera_id is required")
    if not _CAM_ID_RE.fullmatch(camera_id):
        raise ValueError("camera_id may only contain letters, digits, dots, "
                         "underscores and hyphens")
    name = str(name or "").strip()
    if not name:
        raise ValueError("camera name is required")
    try:
        kind_enum = CameraSourceKind(str(kind).strip().lower())
    except (ValueError, TypeError):
        raise ValueError(
            f"invalid camera type {kind!r}; supported: {', '.join(ALLOWED_KINDS)}"
        ) from None
    url = str(url or "").strip()
    if kind_enum is CameraSourceKind.RTSP:
        if not url:
            raise ValueError("RTSP cameras require an RTSP URL")
        parsed = urlparse(url)
        if parsed.scheme.lower() != "rtsp" or not parsed.netloc:
            raise ValueError("RTSP URL must look like rtsp://host[:port]/path "
                             "(reachability is checked separately)")
    elif kind_enum is CameraSourceKind.VIDEO_FILE:
        if not url:
            raise ValueError("video_file cameras require a source path")
    try:
        fps = float(fps_target or 0.0)
    except (TypeError, ValueError):
        raise ValueError("fps_target must be a number") from None
    if fps < 0.0 or fps > 240.0:
        raise ValueError("fps_target must be between 0 (default) and 240")
    try:
        rb, rm, rf = (float(reconnect_base), float(reconnect_max),
                      float(reconnect_factor))
    except (TypeError, ValueError):
        raise ValueError("reconnect settings must be numbers") from None
    if rb < 0.0:
        raise ValueError("reconnect_base must be >= 0")
    if rm <= 0.0:
        raise ValueError("reconnect_max must be > 0")
    if rf < 1.0:
        raise ValueError("reconnect_factor must be >= 1.0")
    return {
        "camera_id": camera_id,
        "name": name,
        "kind": kind_enum.value,
        "enabled": bool(enabled),
        "location": str(location or "").strip(),
        "url": url,
        "username": str(username or "").strip(),
        "password": str(password or ""),
        "fps_target": fps,
        "reconnect_base": rb,
        "reconnect_max": rm,
        "reconnect_factor": rf,
    }


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------

@dataclass
class CameraRecord:
    """One persistent admin camera record (1:1 with runtime CameraConfig).

    ``password`` is stored only so the transport boundary can recompose the
    RTSP URL.  It never appears in ``repr`` / ``safe_dict`` / ``display``.
    """

    camera_id: str
    name: str
    kind: str
    enabled: bool
    location: str
    url: str
    username: str = ""
    password: str = field(default="", repr=False)
    fps_target: float = 0.0
    reconnect_base: float = 2.0
    reconnect_max: float = 30.0
    reconnect_factor: float = 2.5
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        self.enabled = bool(self.enabled)
        self.fps_target = float(self.fps_target or 0.0)
        self.reconnect_base = float(self.reconnect_base or 2.0)
        self.reconnect_max = float(self.reconnect_max or 30.0)
        self.reconnect_factor = float(self.reconnect_factor or 2.5)

    def kind_enum(self) -> CameraSourceKind:
        return CameraSourceKind(self.kind)

    def composed_url(self) -> str:
        """Credential-folded URL for the transport boundary (fresh string)."""
        if self.kind_enum() is CameraSourceKind.RTSP:
            return compose_rtsp_url(self.url, self.username, self.password)
        return self.url

    def transport_url(self):
        """Value handed to ``VideoCapture`` (local cameras use device 0)."""
        if self.kind_enum() is CameraSourceKind.LOCAL and not self.url:
            return 0
        return self.composed_url()

    def to_config(self) -> CameraConfig:
        return CameraConfig(
            camera_id=self.camera_id,
            name=self.name,
            kind=self.kind_enum(),
            url=self.transport_url(),
            username=self.username,
            password=self.password,
            enabled=self.enabled,
            location=self.location,
            fps_target=self.fps_target,
            reconnect_base=self.reconnect_base,
            reconnect_max=self.reconnect_max,
            reconnect_factor=self.reconnect_factor,
        )

    def has_credentials(self) -> bool:
        if self.username or self.password:
            return True
        s = str(self.url or "")
        authority = s.split("://", 1)[-1].split("/", 1)[0] if "://" in s else ""
        return "@" in authority

    def safe_dict(self) -> dict:
        """Serializable outward shape -- never contains the password."""
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "kind": self.kind,
            "enabled": bool(self.enabled),
            "location": self.location,
            "url": redact_url(self.url),          # user:***@host form if embedded
            "host": _host_of(self.url),
            "username": self.username,
            "password": REDACTED,
            "credentials": "CONFIGURED" if self.has_credentials() else "NONE",
            "fps_target": self.fps_target,
            "reconnect_base": self.reconnect_base,
            "reconnect_max": self.reconnect_max,
            "reconnect_factor": self.reconnect_factor,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def display(self) -> str:
        return (f"{self.camera_id} ({self.name}) {self.kind} "
                f"host={_host_of(self.url) or '-'} credentials="
                f"{'REDACTED' if self.has_credentials() else 'NONE'}")

    def __repr__(self) -> str:  # never copy the password into repr/text
        return (f"CameraRecord(camera_id={self.camera_id!r}, name={self.name!r}, "
                f"kind={self.kind!r}, enabled={self.enabled}, "
                f"host={_host_of(self.url)!r}, credentials={REDACTED})")


# ----------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------

class CameraStore:
    """SQLite-backed persistence for admin camera configuration."""

    def __init__(self, conn):
        self.conn = conn

    # -- read -----------------------------------------------------------

    def list(self) -> list[CameraRecord]:
        rows = self.conn.execute(
            "SELECT * FROM admin_cameras ORDER BY camera_id").fetchall()
        cols = [d[1] for d in self.conn.execute(
            "PRAGMA table_info(admin_cameras)").fetchall()]
        return [_record_from_row(dict(zip(cols, r))) for r in rows]

    def get(self, camera_id: str) -> CameraRecord | None:
        rows = self.conn.execute(
            "SELECT * FROM admin_cameras WHERE camera_id = ?",
            (camera_id,)).fetchall()
        if not rows:
            return None
        cols = [d[1] for d in self.conn.execute(
            "PRAGMA table_info(admin_cameras)").fetchall()]
        return _record_from_row(dict(zip(cols, rows[0])))

    # -- write ----------------------------------------------------------

    def add(self, *, camera_id, name, kind, url, username="", password="",
            enabled=True, location="", fps_target=0.0, reconnect_base=2.0,
            reconnect_max=30.0, reconnect_factor=2.5, now=None) -> CameraRecord:
        if self.get(camera_id):
            raise ValueError(f"camera_id {camera_id!r} already exists")
        fields = validate_camera_fields(
            camera_id=camera_id, name=name, kind=kind, url=url,
            username=username, password=password, enabled=enabled,
            location=location, fps_target=fps_target,
            reconnect_base=reconnect_base, reconnect_max=reconnect_max,
            reconnect_factor=reconnect_factor,
        )
        stamp = now or _now()
        self.conn.execute(
            """
            INSERT INTO admin_cameras
                (camera_id, name, kind, enabled, location, url, username,
                 password, fps_target, reconnect_base, reconnect_max,
                 reconnect_factor, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fields["camera_id"], fields["name"], fields["kind"],
             1 if fields["enabled"] else 0, fields["location"], fields["url"],
             fields["username"], fields["password"], fields["fps_target"],
             fields["reconnect_base"], fields["reconnect_max"],
             fields["reconnect_factor"], stamp, stamp),
        )
        self.conn.commit()
        logger.info("Camera %s added (%s).", fields["camera_id"], fields["kind"])
        return self.get(fields["camera_id"])  # type: ignore[return-value]

    def update(self, camera_id: str, fields: dict | None = None, **kw) -> CameraRecord:
        """Apply a partial update.  ``camera_id`` (the identity) never changes.

        Unlisted columns keep their stored values, so an edit that leaves the
        password untouched never needs to resend it.
        """
        existing = self.get(camera_id)
        if existing is None:
            raise ValueError(f"camera_id {camera_id!r} does not exist")
        fields = dict(fields or {})
        fields.update(kw)
        merged = existing.safe_dict()  # safe baseline (password masked only)
        for key, value in fields.items():
            if key in _UPDATE_COLS:
                merged[key] = value
        # None/omitted password means "leave the saved one untouched".
        password_value = existing.password if (
            "password" not in fields or fields.get("password") is None
        ) else str(fields.get("password") or "")
        # Re-validate the merged full record (keeps invariants intact).
        validated = validate_camera_fields(
            camera_id=camera_id,
            name=merged.get("name", existing.name),
            kind=merged.get("kind", existing.kind),
            enabled=merged.get("enabled", existing.enabled),
            location=merged.get("location", existing.location),
            url=merged.get("url", existing.url),
            username=merged.get("username", existing.username),
            password=password_value,
            fps_target=merged.get("fps_target", existing.fps_target),
            reconnect_base=merged.get("reconnect_base", existing.reconnect_base),
            reconnect_max=merged.get("reconnect_max", existing.reconnect_max),
            reconnect_factor=merged.get("reconnect_factor", existing.reconnect_factor),
        )
        set_to = []
        vals: list = []
        for key in _UPDATE_COLS:
            if key not in fields:
                continue
            db_col = _COL_TO_DB[key]
            set_to.append(f"{db_col} = ?")
            if key == "enabled":
                vals.append(1 if validated["enabled"] else 0)
            else:
                vals.append(validated[key])
        stamp = _now()
        set_to.append("updated_at = ?")
        vals.append(stamp)
        vals.append(camera_id)
        self.conn.execute(
            f"UPDATE admin_cameras SET {', '.join(set_to)} "
            f"WHERE camera_id = ?",
            vals,
        )
        self.conn.commit()
        logger.debug("Camera %s updated. (%s)", camera_id,
                     ", ".join(fields.keys()))
        return self.get(camera_id)  # type: ignore[return-value]

    def set_enabled(self, camera_id: str, enabled: bool) -> CameraRecord | None:
        cur = self.get(camera_id)
        if cur is None:
            return None
        return self.update(camera_id, {"enabled": bool(enabled)})

    def delete(self, camera_id: str) -> bool:
        """Hard delete only when no historical record still references it.

        Otherwise raise ``ValueError`` recommending safe disable -- preserves
        the integrity of security events / incidents / health / seat zones /
        topology that carry this camera_id.
        """
        if db.has_camera_usage_history(self.conn, camera_id):
            raise ValueError(
                f"camera {camera_id!r} has historical references (events, "
                "health, incidents, zones or topology); disable it instead "
                "of deleting so past records stay intact")
        cur = self.conn.execute(
            "DELETE FROM admin_cameras WHERE camera_id = ?", (camera_id,))
        self.conn.commit()
        logger.info("Camera %s deleted from admin configuration.", camera_id)
        return cur.rowcount > 0

    # -- apply handshake -------------------------------------------------

    def revision(self) -> str:
        """Content hash over every record (incl. credentials + enabled flag).

        Empty store -> '' so an env-only deployment never triggers an apply.
        """
        records = self.list()
        if not records:
            return ""
        lines = []
        for r in records:
            lines.append("|".join([
                r.camera_id, r.name, r.kind, "1" if r.enabled else "0",
                r.location, r.url, r.username, r.password,
                repr(r.fps_target), repr(r.reconnect_base),
                repr(r.reconnect_max), repr(r.reconnect_factor),
            ]))
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

    def request_apply(self) -> str:
        """Dashboard action: record the current revision as a pending request."""
        rev = self.revision()
        db.set_camera_apply_request(self.conn, rev)
        return rev

    def apply_state(self) -> dict | None:
        return db.get_camera_apply_state(self.conn)


def _record_from_row(row: dict) -> CameraRecord:
    vals = dict(row or {})
    return CameraRecord(
        camera_id=vals.get("camera_id") or "",
        name=vals.get("name") or "",
        kind=vals.get("kind") or CameraSourceKind.RTSP.value,
        enabled=bool(vals.get("enabled", 1)),
        location=vals.get("location") or "",
        url=vals.get("url") or "",
        username=vals.get("username") or "",
        password=vals.get("password") or "",
        fps_target=float(vals.get("fps_target") or 0.0),
        reconnect_base=float(vals.get("reconnect_base") or 2.0),
        reconnect_max=float(vals.get("reconnect_max") or 30.0),
        reconnect_factor=float(vals.get("reconnect_factor") or 2.5),
        created_at=vals.get("created_at") or "",
        updated_at=vals.get("updated_at") or "",
    )


# ----------------------------------------------------------------------
# Resolution: ONE authoritative runtime configuration
# ----------------------------------------------------------------------

def runtime_configs(conn, *, env_configs: dict | None = None) -> dict[str, CameraConfig]:
    """Resolve the authoritative runtime ``CameraConfig`` set.

    * no persistent records -> environment configuration verbatim (Phase 58
      deployments keep working untouched);
    * otherwise persistent records win; camera ids absent from the store fall
      back to the environment.
    """
    store = CameraStore(conn)
    records = store.list()
    if env_configs is None:
        import config
        env_configs = config.camera_configs()
    if not records:
        return dict(env_configs)
    present = {r.camera_id for r in records}
    resolved: dict[str, CameraConfig] = {}
    for cid, cfg in env_configs.items():
        if cid not in present:
            resolved[cid] = cfg
    for r in records:
        try:
            resolved[r.camera_id] = r.to_config()
        except (TypeError, ValueError):
            logger.warning("Skipping invalid camera record %r.", r.camera_id)
    return resolved


# ----------------------------------------------------------------------
# Test Connection (diagnostic only -- never persists anything)
# ----------------------------------------------------------------------

def test_camera_connection(url="", *, kind=None, username="", password="",
                           reconnect=(2.0, 30.0, 2.5), timeout: float = 8.0) -> dict:
    """Bounded, clean diagnostic probe of a *currently entered* source.

    Opens the source exactly once through the existing ``VideoCapture`` path,
    waits up to ``timeout`` seconds for a live connection, reports sanitized
    telemetry and always releases the capture in ``finally``.  The camera is
    NOT registered anywhere just because the test succeeded.  Without physical
    hardware this honestly reports ``ok=False`` with a sanitized reason.

    The result dict never contains the password or a credential-bearing URL.
    """
    reconnect = tuple(reconnect or (2.0, 30.0, 2.5))
    try:
        kind_enum = (kind if isinstance(kind, CameraSourceKind)
                     else CameraSourceKind(str(kind or "").strip().lower()))
    except (ValueError, TypeError):
        return {
            "ok": False, "connected": False, "health": "INVALID_CONFIG",
            "resolution": "", "width": 0, "height": 0, "fps": 0.0,
            "frames_read": 0, "reconnects": 0, "elapsed": 0.0,
            "error": "invalid camera type",
            "url": redact_url(str(url or "")),
        }

    if kind_enum is CameraSourceKind.RTSP:
        source = compose_rtsp_url(str(url or ""), username, password)
    elif kind_enum is CameraSourceKind.LOCAL and not str(url or "").strip():
        source = 0
    else:
        source = url

    cap = VideoCapture(source=source, kind=kind_enum,
                       camera_id="_probe", reconnect=reconnect)
    start = time.monotonic()
    connected = False
    info: dict = {}
    try:
        deadline = start + max(0.1, float(timeout))
        cap.start()
        while time.monotonic() < deadline:
            info = cap.health_info()
            if info.get("connected"):
                connected = True
                break
            time.sleep(0.05)
    finally:
        try:
            if not info:
                info = cap.health_info()
            cap.stop()
        except Exception:  # pragma: no cover - release must always proceed
            logger.debug("Test probe teardown failed.", exc_info=True)

    if not connected:
        return {
            "ok": False, "connected": False,
            "health": info.get("health") or "OFFLINE",
            "resolution": info.get("resolution") or "",
            "width": info.get("width") or 0, "height": info.get("height") or 0,
            "fps": round(info.get("fps") or 0.0, 2),
            "frames_read": info.get("frames_read") or 0,
            "reconnects": info.get("reconnects") or 0,
            "elapsed": round(time.monotonic() - start, 2),
            "error": ("connection unavailable: source not reachable (and no "
                      "physical camera is attached in this environment)"),
            "url": redact_url(str(source)),
        }
    return {
        "ok": True, "connected": True, "health": info.get("health") or "ONLINE",
        "resolution": info.get("resolution") or "",
        "width": info.get("width") or 0, "height": info.get("height") or 0,
        "fps": round(info.get("fps") or 0.0, 2),
        "frames_read": info.get("frames_read") or 0,
        "reconnects": info.get("reconnects") or 0,
        "elapsed": round(time.monotonic() - start, 2),
        "error": "",
        "url": redact_url(str(source)),
    }


# ----------------------------------------------------------------------
# Daemon-side apply handshake
# ----------------------------------------------------------------------

def reconcile_runtime(conn, manager, *, env_configs: dict | None = None,
                      force: bool = False) -> dict:
    """Idempotent, periodic reconcile ran by the daemon loop.

    Applies the authoritative configuration when either
      * the store revision changed since the last applied revision (auto), or
      * an admin explicitly requested an apply (dashboard button).

    Returns a small status dict; never raises on no-change.
    """
    store = CameraStore(conn)
    revision = store.revision()
    state = store.apply_state() or {}
    applied = state.get("applied_revision") or ""
    requested = state.get("requested_revision") or ""

    reason = ""
    if force:
        reason = "forced apply"
    elif revision and revision != applied:
        reason = f"configuration changed ({applied[:8]} -> {revision[:8]})"
    elif requested and requested != applied:
        reason = "admin requested apply"
    if not reason:
        return {"action": "up_to_date", "revision": revision, "applied": applied}

    desired = runtime_configs(conn, env_configs=env_configs)
    summary = manager.apply_configs(desired)
    db.set_camera_applied(conn, revision, f"{reason}; {summary}")
    logger.info("Camera configuration applied: %s | %s", reason, summary)
    return {"action": "applied", "revision": revision, "applied": revision,
            "summary": summary, "reason": reason}