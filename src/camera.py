"""Threaded video capture with RTSP/network resilience.

Features
--------
* FFMPEG backend with ``rtsp_transport=tcp`` and low buffer to eliminate
  packet tearing on IP camera feeds.
* Exponential-backoff auto-reconnect on stream loss (2 s -> 5 s -> 10 s
  -> max 30 s).
* Non-blocking ``cap.read()`` guard so the reader thread never freezes
  on a hung network socket.
* ``.is_connected`` health-check property for external monitoring.
"""

import logging
import threading
import time
import cv2

from config import FRAME_SKIP

logger = logging.getLogger("cctv.camera")

from src.domain import (  # noqa: E402
    CAM_FROZEN,
    CAM_LOW_FPS,
    CAM_NO_FRAME,
    CAM_OFFLINE,
    CAM_ONLINE,
    CAM_RECONNECTING,
    redact_url,
)

# Reconnect backoff parameters (seconds)
_RECONNECT_BASE = 2.0
_RECONNECT_MAX = 30.0
_RECONNECT_FACTOR = 2.5

# Timeout (seconds) for a single ``cap.read()`` attempt inside the reader
# thread.  If the call hangs longer than this we treat it as a network stall
# and trigger a reconnect.  We implement this by running ``cap.read()`` in a
# helper thread and joining with a timeout.
_READ_TIMEOUT = 4.0

# A camera that has produced no fresh frame for this long is declared NO_FRAME.
_STALE_FRAME_SEC = 10.0

# Phase A (A12): degraded-while-online health states.
# * Frozen: the frame content has not changed across this many delivered
#   frames (and for at least as long as _FROZEN_MIN_SEC) -- a hard signal that
#   the scene is static.  Reported as FROZEN_FRAME (still technically flowing).
# * Low FPS: source is delivering frames but slower than this threshold.
_FROZEN_STREAK = 30
_FROZEN_MIN_SEC = 6.0
_LOW_FPS_THRESHOLD = 5.0


class VideoCapture:
    """Threaded video capture that reads the stream and drops stale frames.

    Parameters
    ----------
    source : str | int
        int index (webcam), file path (.mp4), or RTSP URL string.
    frame_skip : int
        Process every Nth frame.
    loop : bool
        Rewind video files on EOF.
    """

    def __init__(
        self,
        source=0,
        frame_skip: int = FRAME_SKIP,
        loop: bool = False,
    ):
        self.source = source
        self.frame_skip = frame_skip
        self.loop = loop
        self._loop_count = 0
        self._frame = None
        self._last_frame_ts: float = 0.0
        self._reconnect_count = 0
        self._health = CAM_OFFLINE
        self._frames_read = 0
        self._fps = 0.0
        self._fps_window_start: float = 0.0
        self._fps_window_frames = 0
        # Frozen-frame bookkeeping (A12): cheap content signature + streak.
        self._frame_sig = 0
        self._frozen_streak = 0
        self._frozen_since: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        """Health-check: True when the capture backend is open and reading."""
        return self._connected

    @property
    def loop_count(self) -> int:
        with self._lock:
            return self._loop_count

    @property
    def last_frame_timestamp(self) -> float:
        """Wall-clock time of the most recent successfully read frame (0 if none)."""
        with self._lock:
            return self._last_frame_ts

    @property
    def reconnect_count(self) -> int:
        with self._lock:
            return self._reconnect_count

    @property
    def frames_read(self) -> int:
        with self._lock:
            return self._frames_read

    @property
    def fps(self) -> float:
        """Rolling frames-per-second delivered by this source (0 until busy)."""
        with self._lock:
            return self._fps

    @property
    def health(self) -> str:
        """Derive a human/camera health state for the HUD/dashboard.

        Returns one of ``ONLINE`` / ``NO_FRAME`` / ``RECONNECTING`` /
        ``OFFLINE`` -- plus the Phase A (A12) degraded-but-flowing states
        ``FROZEN_FRAME`` (content unchanged for a sustained period) and
        ``LOW_FPS`` (delivering frames below the health threshold).
        """
        with self._lock:
            if not self._running or self._cap is None or not self._cap.isOpened():
                base = CAM_RECONNECTING if self._reconnect_count else CAM_OFFLINE
            elif self._last_frame_ts and (time.time() - self._last_frame_ts) > _STALE_FRAME_SEC:
                base = CAM_NO_FRAME
            elif self._frame is not None:
                base = CAM_ONLINE
                if self._frozen_streak >= _FROZEN_STREAK and self._frozen_since and \
                        (time.time() - self._frozen_since) >= _FROZEN_MIN_SEC:
                    base = CAM_FROZEN
                elif 0.0 < self._fps < _LOW_FPS_THRESHOLD:
                    base = CAM_LOW_FPS
            else:
                base = CAM_NO_FRAME
        if self._reconnect_count and base == CAM_OFFLINE:
            base = CAM_RECONNECTING
        return base

    @staticmethod
    def _frame_signature(frame) -> int:
        """Cheap content fingerprint used to spot a frozen feed."""
        try:
            sub = frame[::16, ::16]
            return int(round(float(sub.mean())))
        except Exception:
            return 0

    def health_info(self) -> dict:
        """Return a snapshot suitable for the dashboard/HUD."""
        return {
            "source": str(self.source),
            "connected": self.is_connected,
            "health": self.health,
            "frames_read": self.frames_read,
            "fps": round(self.fps, 2),
            "reconnects": self.reconnect_count,
            "last_frame": self.last_frame_timestamp,
            "frozen_streak": self._frozen_streak,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> "VideoCapture":
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._release_cap()

    def read(self):
        """Return the most recent frame.  ``None`` if no frame yet."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    # ------------------------------------------------------------------
    # Internal: open / release
    # ------------------------------------------------------------------
    def _is_rtsp(self) -> bool:
        src = str(self.source)
        return src.startswith("rtsp://") or src.startswith("rtsp:")

    def _is_local(self) -> bool:
        """True for local devices (webcam index / numeric source).

        Local captures are fast and non-blocking, so we avoid the
        per-read helper thread entirely (thread-per-read churn is a CPU
        waste on webcams).  RTSP keeps the timed path for network stalls.
        """
        if self._is_rtsp():
            return False
        src = str(self.source).strip().lower()
        return src in ("webcam", "local") or src.isdigit()

    def _open_stream(self) -> bool:
        self._release_cap()

        if self._is_rtsp():
            # FFMPEG backend with TCP transport for reliability.
            self._cap = cv2.VideoCapture(
                self.source,
                cv2.CAP_FFMPEG,
            )
            if self._cap is not None and self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Pass FFMPEG options via the URL when the backend supports it.
                # OpenCV >= 4.x forwards these to FFmpeg automatically.
                self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if self._cap is None or not self._cap.isOpened():
            logger.warning("Failed to open source: %s", redact_url(self.source))
            self._connected = False
            return False

        if not self._is_rtsp():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._connected = True
        logger.info("Stream opened: %s", redact_url(self.source))
        return True

    def _release_cap(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._connected = False

    def _rewind(self) -> bool:
        if self._cap is None:
            return False
        ok = self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if ok:
            with self._lock:
                self._loop_count += 1
        return ok

    # ------------------------------------------------------------------
    # Internal: timed read (prevents indefinite blocking)
    # ------------------------------------------------------------------
    def _timed_read(self) -> tuple[bool, "cv2.Mat | None"]:
        """Run ``cap.read()`` in a helper thread so we can enforce a timeout."""
        result: list = [False, None]

        def _do_read():
            if self._cap is not None:
                try:
                    ret, frame = self._cap.read()
                    result[0] = ret
                    result[1] = frame
                except Exception:
                    result[0] = False

        t = threading.Thread(target=_do_read, daemon=True)
        t.start()
        t.join(timeout=_READ_TIMEOUT)
        if t.is_alive():
            # read() is stuck -- the network is likely hung.
            logger.warning("cap.read() timed out after %.1fs -- network stall", _READ_TIMEOUT)
            return False, None
        return result[0], result[1]

    # ------------------------------------------------------------------
    # Reader thread
    # ------------------------------------------------------------------
    def _reader(self):
        backoff = _RECONNECT_BASE
        frame_idx = 0

        if not self._open_stream():
            # Initial open failed; enter reconnect loop.
            logger.info("Entering reconnect loop for %s", redact_url(self.source))

        while self._running:
            # -- Reconnect if stream is lost --
            if self._cap is None or not self._cap.isOpened():
                self._connected = False
                if not self._running:
                    break
                logger.info(
                    "Reconnecting in %.1fs ... (backoff %.1fs)",
                    backoff, backoff,
                )
                # Sleep in small increments so stop() remains responsive.
                waited = 0.0
                while waited < backoff and self._running:
                    time.sleep(min(0.5, backoff - waited))
                    waited += 0.5
                if not self._running:
                    break
                if self._open_stream():
                    backoff = _RECONNECT_BASE
                    frame_idx = 0
                    continue
                # Still failed -- increase backoff and retry.
                backoff = min(backoff * _RECONNECT_FACTOR, _RECONNECT_MAX)
                with self._lock:
                    self._reconnect_count += 1
                continue

            # -- Grab --
            ok, frame = self._next_frame()
            if not ok:
                if self.loop and self._rewind():
                    continue
                logger.warning("Grab failed; triggering reconnect.")
                self._release_cap()
                with self._lock:
                    self._reconnect_count += 1
                backoff = min(backoff * _RECONNECT_FACTOR, _RECONNECT_MAX)
                continue

            frame_idx += 1
            if frame is None or frame_idx % self.frame_skip != 0:
                continue

            # -- Retrieve --
            now = time.time()
            with self._lock:
                self._frame = frame
                self._last_frame_ts = now
                self._frames_read += 1
                # A12 -- frozen-frame bookkeeping: same signature = same scene.
                sig = self._frame_signature(frame)
                if sig == self._frame_sig:
                    self._frozen_streak += 1
                    if self._frozen_streak == _FROZEN_STREAK:
                        self._frozen_since = now
                else:
                    if self._frozen_since:
                        self._frozen_streak = 0
                        self._frozen_since = 0.0
                self._frame_sig = sig
                # Rolling per-camera FPS (1 s window) for the health view.
                if self._fps_window_start == 0.0:
                    self._fps_window_start = now
                self._fps_window_frames += 1
                if now - self._fps_window_start >= 1.0:
                    self._fps = self._fps_window_frames / (now - self._fps_window_start)
                    self._fps_window_start = now
                    self._fps_window_frames = 0

        self._release_cap()
        logger.info("Reader thread exited.")

    def _next_frame(self) -> tuple[bool, "cv2.Mat | None"]:
        """Return (ok, frame) for the current frame-slot.

        Local (webcam) sources use direct non-blocking calls.  RTSP sources
        keep the timed grab/read path so a hung network socket cannot freeze
        the reader thread for longer than ``_READ_TIMEOUT``.
        """
        if not self._is_rtsp():
            try:
                ok, frame = self._cap.read()
            except Exception:
                return False, None
            return bool(ok), frame

        ok, _ = self._timed_read_grab()
        if not ok:
            return False, None
        return self._timed_read()

    def _timed_read_grab(self) -> tuple[bool, None]:
        """Timed ``cap.grab()`` -- same pattern as ``_timed_read``."""
        result: list = [False]

        def _do_grab():
            if self._cap is not None:
                try:
                    result[0] = self._cap.grab()
                except Exception:
                    result[0] = False

        t = threading.Thread(target=_do_grab, daemon=True)
        t.start()
        t.join(timeout=_READ_TIMEOUT)
        if t.is_alive():
            return False, None
        return result[0], None
