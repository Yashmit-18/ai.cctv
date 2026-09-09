"""Pooled multi-camera manager (Phase 7).

The ``MultiCameraManager`` owns one ``VideoCapture`` reader thread per
camera and exposes a batch-oriented API.  Unlike the single-stream
``VideoCapture.read()``, ``get_latest_batch()`` returns a dictionary keyed
by camera ID holding the most recent frame from *each* camera.

Resilience
----------
* Each camera reader runs independently, so a dropped camera does not stall
  the rest of the batch.
* Each ``VideoCapture`` already implements exponential-backoff reconnects
  internally (see ``src/camera.py``), so the pool simply keeps polling for
  whatever is available.
* The batch keeps the same shape on every call: cameras with no fresh frame
  simply map to ``None``, allowing the caller to detect and skip missing
  streams without breaking upstream GPU batching.
"""

import logging
import threading
import time

from config import CAMERAS, FRAME_STALE_SEC, TARGET_FPS_PER_CAMERA
from src.camera import VideoCapture

logger = logging.getLogger("cctv.camera_manager")


class MultiCameraManager:
    """Manages a pool of per-camera ``VideoCapture`` threads.

    Parameters
    ----------
    cameras : dict[str, str] | None
        Mapping of camera_id -> source (RTSP URL, file, or index).
        Defaults to ``config.CAMERAS``.
    max_batch : int | None
        Largest number of live frames to emit in a single batch.  ``None``
        emits one entry per configured camera.
    """

    def __init__(self, cameras: dict[str, str] | None = None,
                 max_batch: int | None = None):
        self._cameras = dict(cameras) if cameras is not None else dict(CAMERAS)
        self._max_batch = max_batch
        self._captures: dict[str, VideoCapture] = {}
        self._lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "MultiCameraManager":
        with self._lock:
            if self._started:
                return self
            self._captures = {
                cam_id: VideoCapture(source=url).start()
                for cam_id, url in self._cameras.items()
            }
            self._started = True
            logger.info("Started %d camera streams.", len(self._captures))
        return self

    def stop(self):
        with self._lock:
            captures = list(self._captures.values())
            self._captures.clear()
            self._started = False
        for cap in captures:
            cap.stop()
        logger.info("Stopped all camera streams.")

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    def get_latest_batch(self) -> dict[str, object]:
        """Return ``{cam_id: latest_frame_or_None, ...}`` for all cameras."""
        with self._lock:
            captures = dict(self._captures)
        batch: dict[str, object] = {}
        for cam_id, cap in captures.items():
            batch[cam_id] = cap.read()
        if self._max_batch is not None:
            # Emit the most "fresh" subset if we exceed the limit.  Python
            # dict preserves insertion order, so this simply caps the size.
            ordered = [cid for cid, f in batch.items() if f is not None]
            batch = {cid: batch[cid] for cid in ordered[:self._max_batch]}
        return batch

    def latest_frames(self, limit: int | None = None, max_age: float | None = None) -> dict[str, object]:
        """Return only cameras with a *fresh* frame.

        A frame older than ``max_age`` seconds (default ``FRAME_STALE_SEC``)
        is treated as stale and excluded: a dead camera that still holds its
        last frame must not keep the pipeline "online" (and thereby risk
        fabricating employee AWAY/ACTIVE time).

        ``limit`` caps the number of live frames (used for GPU batch size
        control downstream).
        """
        max_age = FRAME_STALE_SEC if max_age is None else max_age
        now = time.time()
        batch = self.get_latest_batch()
        live = {}
        for cid, f in batch.items():
            if f is None:
                continue
            cap = self._captures.get(cid)
            ts = cap.last_frame_timestamp if cap else 0.0
            if ts and (now - ts) > max_age:
                continue  # stale frame -- camera is effectively dead
            live[cid] = f
        if limit is not None:
            live = {cid: live[cid] for cid in list(live)[:limit]}
        return live

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def camera_ids(self) -> list[str]:
        return list(self._cameras.keys())

    @property
    def active_camera_ids(self) -> list[str]:
        with self._lock:
            caps = dict(self._captures)
        return [cid for cid, cap in caps.items() if cap.is_connected]

    @property
    def online_count(self) -> int:
        """Number of cameras currently producing fresh frames (health ONLINE)."""
        info = self.camera_health()
        return sum(1 for h in info.values() if h.get("health") == "ONLINE")

    def is_connected(self, cam_id: str) -> bool:
        cap = self._captures.get(cam_id)
        return bool(cap and cap.is_connected)

    def get_capture(self, cam_id: str):
        """Return the underlying ``VideoCapture`` for a camera (or None)."""
        return self._captures.get(cam_id)

    def camera_health(self) -> dict[str, dict]:
        """Per-camera health snapshot for the dashboard/HUD.

        Returns ``{cam_id: {source(redacted), health, frames_read, reconnects,
        last_frame, connected}}``.  RTSP credentials are redacted.
        """
        from src.domain import redact_url
        with self._lock:
            caps = dict(self._captures)
        out: dict[str, dict] = {}
        for cid, cap in caps.items():
            h = cap.health_info()
            h["id"] = cid
            h["source"] = redact_url(h.get("source", cid))
            out[cid] = h
        return out

    # Rate limiter helper used by the orchestrator to cap per-camera fps.
    @staticmethod
    def frame_interval(rate: float = TARGET_FPS_PER_CAMERA) -> float:
        """Return the sleep interval that yields ``rate`` fps (0 disables)."""
        return 1.0 / rate if rate and rate > 0 else 0.0
