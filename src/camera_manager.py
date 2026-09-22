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
    configs : dict[str, CameraConfig] | None
        Optional (Phase 58) richer per-camera configuration.  When supplied,
        a camera is only started if its config ``enabled`` is True, and the
        config's per-camera reconnect policy/kind are applied.  A camera that
        appears in ``configs`` but not in ``cameras`` is still started (from
        its config URL); a camera in ``cameras`` with no config entry keeps
        the legacy plain-source behaviour.
    max_batch : int | None
        Largest number of live frames to emit in a single batch.  ``None``
        emits one entry per configured camera.
    """

    def __init__(self, cameras: dict[str, str] | None = None,
                 configs: dict | None = None,
                 max_batch: int | None = None):
        self._cameras = dict(cameras) if cameras is not None else dict(CAMERAS)
        self._configs: dict[str, dict] = dict(configs) if configs else {}
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
            captures: dict[str, VideoCapture] = {}
            # Phase 58: full config model (name/enabled/reconnect/kind).  A
            # disabled camera is excluded from the pool entirely.
            for cam_id, cfg in self._configs.items():
                params = MultiCameraManager._cfg(cfg, self._cameras.get(cam_id, ""))
                if params is None:
                    continue  # disabled / no usable source
                src, kind, reconnect = params
                cap = VideoCapture(
                    source=src,
                    kind=kind or None,
                    camera_id=cam_id,
                    reconnect=reconnect,
                ).start()
                captures[cam_id] = cap
            # Legacy sources not covered by the config model.
            for cam_id, url in self._cameras.items():
                if cam_id in captures:
                    continue
                captures[cam_id] = VideoCapture(source=url).start()
            self._captures = captures
            self._started = True
            logger.info("Started %d camera streams.", len(self._captures))
        return self

    @staticmethod
    def _cfg(cfg, legacy_src: str) -> tuple[str, object, tuple[float, float, float]] | None:
        """Normalize a config entry (dict or CameraConfig dataclass).

        Returns ``(source, kind, reconnect_policy)`` or ``None`` when the
        camera is disabled or has no usable source.
        """
        def _get(key, default=None):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        if not _get("enabled", True):
            return None
        src = _get("url")
        if src is None or src == "":
            src = legacy_src
        if src is None or (isinstance(src, str) and not src):
            return None
        kind = _get("kind")
        reconn = (
            _get("reconnect_base", 2.0),
            _get("reconnect_max", 30.0),
            _get("reconnect_factor", 2.5),
        )
        return src, kind, reconn

    @staticmethod
    def _kind_str(kind) -> str:
        """Normalize a ``CameraSourceKind`` (or its string value) to text."""
        if kind is None:
            return ""
        return getattr(kind, "value", str(kind))

    @staticmethod
    def _same_source(cap, src, kind, reconnect) -> bool:
        """Whether a running capture already serves the desired source - so a
        reconcile can keep it untouched instead of rebuilding it."""
        if str(getattr(cap, "source", None)) != str(src):
            return False
        if MultiCameraManager._kind_str(getattr(cap, "kind", None)) != \
                MultiCameraManager._kind_str(kind):
            return False
        return tuple(getattr(cap, "_reconnect_cfg", (2.0, 30.0, 2.5))) == \
            tuple(reconnect)

    def apply_configs(self, configs: dict) -> dict:
        """Reconcile the running pool with a new Phase 58 configuration set
        (Phase 61 runtime reload).

        * cameras disabled/removed from the set are stopped and closed;
        * cameras whose source/kind/reconnect policy changed are closed and
          reopened with the new source;
        * cameras still configured are started when new to this pool;
        * cameras with an unchanged source are left running untouched.

        Returns a summary dict ``{"started", "updated", "kept", "stopped"}``.
        This is incremental: it never rebuilds the whole pool and never touches
        unaffected cameras.
        """
        desired: dict[str, tuple] = {}
        for cam_id, cfg in dict(configs).items():
            params = MultiCameraManager._cfg(cfg, self._cameras.get(cam_id, ""))
            if params is None:
                continue  # disabled or no usable source -> must not run
            desired[cam_id] = params

        with self._lock:
            before = dict(self._captures)
            self._configs = dict(configs)

        if not self._started:
            # Pool not started yet: just remember the configs for start().
            return {"started": 0, "updated": 0, "kept": 0, "stopped": 0,
                    "stored": True}

        stop_list: list = []
        new_caps: dict[str, object] = {}
        started = updated = kept = stopped = 0
        for cam_id, cap in before.items():
            params = desired.get(cam_id)
            if params is None:
                stop_list.append(cap)
                stopped += 1
                continue
            src, kind, reconnect = params
            if MultiCameraManager._same_source(cap, src, kind, reconnect):
                new_caps[cam_id] = cap
                kept += 1
            else:
                stop_list.append(cap)
                updated += 1  # closed below, reopened from new source

        for cam_id, params in desired.items():
            if cam_id in new_caps:
                continue
            src, kind, reconnect = params
            cap = VideoCapture(
                source=src,
                kind=kind or None,
                camera_id=cam_id,
                reconnect=reconnect,
            ).start()
            new_caps[cam_id] = cap
            if cam_id in before:
                updated += 0  # replacement already counted above
            else:
                started += 1

        with self._lock:
            self._captures = new_caps
        for cap in stop_list:
            cap.stop()
        logger.info(
            "Camera config applied: %d started, %d updated, %d kept, %d stopped.",
            started, updated, kept, stopped)
        return {"started": started, "updated": updated, "kept": kept,
                "stopped": stopped}

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
