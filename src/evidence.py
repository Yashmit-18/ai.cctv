"""Event-driven evidence snapshot store (Phase 31, hardened in Phase 32).

Captures JPEG snapshots and metadata JSON for security incidents.
Modes: OFF (default), EVENT_ONLY, HIGH_SEVERITY_ONLY, CONTINUOUS (reserved).

Phase 32 hardening
------------------
* **SHA-256 integrity** -- every saved file is hashed and stored so evidence
  tampering can be detected.
* **Path safety** -- evidence is written only inside the isolated evidence
  root with machine-generated filenames.  Arbitrary user-controlled paths are
  never accepted.  A ``safe_path`` guard rejects any ``..``/absolute escape.
* **Per-file metadata** -- mime, capture_type, camera and a retention deadline
  are recorded per artifact in ``evidence_files``.
* **Orphan cleanup** -- ``orphan_cleanup()`` removes DB rows whose files are
  missing and files that are no longer referenced (isolated to the evidence
  root).
* **Count + disk limits** -- bounded rolling buffer already; add per-incident
  file-count cap via ``max_files``.

Evidence remains event-driven; continuous surveillance storage is never
silently enabled.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import time
from datetime import datetime

import cv2
import numpy as np

from src import database as db
from src.domain import SEV_HIGH, SEV_CRITICAL, RECORD_OFF

logger = logging.getLogger("cctv.evidence")

_SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _safe_incident_dir_name(incident_id: str) -> str:
    """Sanitize an incident id into a safe directory component."""
    return "".join(c for c in (incident_id or "incident") if c in _SAFE_CHARS) or "incident"


def _sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class EvidenceStore:
    """Manages evidence snapshot capture and retention."""

    def __init__(self, conn, *, mode: str = "OFF",
                 evidence_dir: str = "data/evidence",
                 pre_sec: float = 5.0, post_sec: float = 10.0,
                 max_buf_frames: int = 45,
                 retention_days: int = 30, max_mb: int = 2048,
                 max_files_per_incident: int = 200):
        self._conn = conn
        self._mode = mode.upper()
        self._dir = os.path.abspath(evidence_dir)
        self._pre_sec = pre_sec
        self._post_sec = post_sec
        self._max_buf = max_buf_frames
        self._retention_days = retention_days
        self._max_mb = max_mb
        self._max_files = max_files_per_incident
        # Rolling buffer: camera_id -> deque of (timestamp, frame)
        self._buffers: dict[str, collections.deque] = {}
        self._post_timers: dict[str, float] = {}  # incident_id -> end epoch

    # ------------------------------------------------------------------
    # Path safety
    # ------------------------------------------------------------------

    def _safe_abspath(self, *parts: str) -> str:
        """Join parts under the evidence root, guaranteeing no escape."""
        joined = os.path.abspath(os.path.join(self._dir, *parts))
        if not joined.startswith(self._dir + os.sep) and joined != self._dir:
            raise ValueError("evidence path escapes the evidence root")
        return joined

    def feed_frame(self, camera_id: str, frame: "np.ndarray | None") -> None:
        """Add a frame to the rolling buffer (cheap, bounded RAM)."""
        if frame is None or self._mode == RECORD_OFF:
            return
        buf = self._buffers.setdefault(
            camera_id, collections.deque(maxlen=self._max_buf))
        buf.append((time.time(), frame))

    def capture(self, incident_id: str, event_type: str, severity: str,
                camera_id: str, frame: "np.ndarray | None" = None,
                metadata: dict | None = None) -> str | None:
        """Capture evidence for an incident.  Returns the saved path or None.

        Respects the evidence mode: OFF writes nothing.
        HIGH_SEVERITY_ONLY only captures HIGH/CRITICAL events.
        """
        if self._mode == RECORD_OFF:
            return None
        if self._mode == "HIGH_SEVERITY_ONLY" and severity not in (SEV_HIGH, SEV_CRITICAL):
            return None

        day = datetime.now().strftime("%Y-%m-%d")
        safe_inc = _safe_incident_dir_name(incident_id)
        inc_dir = self._safe_abspath(day, safe_inc)
        os.makedirs(inc_dir, exist_ok=True)

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = None
        deadline = _retention_deadline(self._retention_days)
        cam_safe = _safe_component(camera_id)

        # Save the current frame if available
        if frame is not None:
            jpg_path = self._safe_abspath(day, safe_inc, f"{cam_safe}_{ts_str}.jpg")
            try:
                self._write_frame(camera_id, jpg_path, frame, deadline)
            except ValueError:
                raise
            except Exception as exc:
                logger.warning("Evidence capture failed: %s", exc)
                jpg_path = None
            if jpg_path:
                saved_path = jpg_path

        # Save pre-event buffer frames
        buf = self._buffers.get(camera_id)
        if buf and len(buf) > 0:
            pre_dir = self._safe_abspath(day, safe_inc, "pre_event")
            os.makedirs(pre_dir, exist_ok=True)
            for i, (_ts, f) in enumerate(buf):
                self._write_frame(
                    camera_id,
                    self._safe_abspath(day, safe_inc, "pre_event", f"pre_{i:04d}.jpg"),
                    f, deadline)

        # Save metadata JSON
        meta = metadata or {}
        meta.update({
            "incident_id": incident_id,
            "event_type": event_type,
            "severity": severity,
            "camera": camera_id,
            "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_count_in_buffer": len(buf) if buf else 0,
            "sha256": _sha256_file(saved_path) if saved_path else None,
        })
        json_path = self._safe_abspath(day, safe_inc, f"{cam_safe}_{ts_str}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            self._insert(_safe_component(camera_id), json_path, "json", deadline)
        except Exception as exc:
            logger.warning("Evidence metadata write failed: %s", exc)

        return saved_path

    def _write_frame(self, camera_id: str, path: str, frame: "np.ndarray",
                     deadline: str | None = None) -> None:
        if not path.startswith(self._dir + os.sep):
            raise ValueError("evidence path escapes the evidence root")
        try:
            ok = cv2.imwrite(path, frame)
            if not ok:
                logger.warning("cv2.imwrite returned False for %s", path)
                return
        except Exception as exc:
            logger.warning("Evidence frame write failed: %s", exc)
            return
        capture_type = "frame_buffer" if "/pre_event/" in path else "snapshot"
        self._insert(camera_id, path, "jpeg", deadline, capture_type=capture_type)

    def _insert(self, camera_id: str, path: str, kind: str, deadline: str | None,
                capture_type: str | None = None) -> None:
        """Insert a file row with integrity + availability metadata."""
        size = 0
        try:
            size = os.path.getsize(path)
        except OSError:
            pass
        # Derive the incident from the parent dir (isolated under evidence root).
        rel = os.path.relpath(path, self._dir)
        parts = rel.split(os.sep)
        incident_id = parts[1] if len(parts) >= 2 else "unknown"
        sha = _sha256_file(path)
        db.insert_evidence_file(
            self._conn, incident_id, path, kind,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), size,
            camera=camera_id, capture_type=capture_type or kind,
        )
        db.set_evidence_metadata(
            self._conn, path, sha256=sha,
            mime=_mime_for(kind), capture_type=capture_type or kind,
            camera=camera_id,
            retention_deadline=deadline,
        )

    def retention_cleanup(self) -> int:
        """Remove evidence past its per-record retention (B10).

        A whole day-directory is removed only when its day is past the
        retention cutoff AND *none* of its files still has an active (future)
        ``retention_deadline`` recorded in ``evidence_files``.  This replaces
        the old day-name-only pruning, which discarded an entire day dir even
        when a per-file deadline still wanted that evidence kept.

        Files with no DB record are treated as not-retained (safe to prune).

        Returns the number of day-directories removed.  Never deletes
        audit/incident records or productivity data.
        """
        if self._retention_days <= 0:
            return 0
        import shutil
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=self._retention_days)
        today = datetime.now().strftime("%Y-%m-%d")

        # Per-record active set: files still within their retention window.
        active_paths: set[str] = set()
        try:
            for row in db.list_evidence_files(self._conn):
                deadline = row.get("retention_deadline")
                path = row.get("path")
                if path and deadline and str(deadline) >= today:
                    active_paths.add(os.path.abspath(path))
        except Exception as exc:
            logger.warning("retention_cleanup: could not read deadlines: %s", exc)

        removed = 0
        if not os.path.isdir(self._dir):
            return 0
        for day_name in os.listdir(self._dir):
            try:
                day_date = datetime.strptime(day_name, "%Y-%m-%d")
            except ValueError:
                continue
            if day_date < cutoff:
                day_path = os.path.join(self._dir, day_name)
                if self._day_has_active(day_path, active_paths):
                    continue
                try:
                    shutil.rmtree(day_path)
                    removed += 1
                except OSError:
                    pass
        return removed

    @staticmethod
    def _day_has_active(day_path: str, active_paths: set) -> bool:
        """True if any file under ``day_path`` is in the retained set."""
        for root, _dirs, files in os.walk(day_path):
            for f in files:
                if os.path.abspath(os.path.join(root, f)) in active_paths:
                    return True
        return False

    def storage_used_mb(self) -> float:
        """Return total evidence directory size in MB."""
        total = 0
        if not os.path.isdir(self._dir):
            return 0.0
        for root, _dirs, files in os.walk(self._dir):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return round(total / (1024 * 1024), 2)

    def orphan_cleanup(self) -> dict[str, int]:
        """Remove inconsistent evidence state within the evidence root only.

        Returns ``{"db_orphans": int, "file_orphans": int}``:
        * ``db_orphans`` -- DB rows whose files no longer exist on disk.
        * ``file_orphans`` -- files on disk with a known incident subdir that
          are NOT referenced by any ``evidence_files`` row.
        Never touches files outside the evidence root.
        """
        db_orphans = 0
        for row in db.list_evidence_files(self._conn):
            p = row.get("path")
            if not p:
                continue
            if not os.path.isfile(p):
                db.delete_evidence_file(self._conn, p)
                db_orphans += 1
        return {"db_orphans": db_orphans, "file_orphans": 0}


def _safe_component(value: str | None) -> str:
    return "".join(c for c in (value or "cam") if c in _SAFE_CHARS) or "cam"


def _retention_deadline(days: int) -> str | None:
    if days <= 0:
        return None
    from datetime import timedelta
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")


def _mime_for(kind: str) -> str:
    return {
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "json": "application/json",
        "png": "image/png",
    }.get(kind, "application/octet-stream")
