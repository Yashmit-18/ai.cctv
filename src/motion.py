"""Frame-level motion detection (Phase A -- A8).

Produces ``MOTION_DETECTED`` events when the fraction of changed pixels stays
above a threshold for a minimum duration.  Motion is deliberately advisory:

* it is stored independently (``motion_events`` table), never turned into a
  security alarm on its own;
* the correlation engine may later pair it with a *threatened* context
  (e.g. motion followed by tamper / silhouette) -- the engine owns that
  decision, not this module.

Implementation
--------------
* Frames are downscaled to a configurable working width for latency.
* Per camera: abs-diff of downscaled grayscale, thresholded, fraction of
  changed pixels = score.
* An event fires once a run of ``score >= min_score`` lasts at least
  ``min_duration_sec``; successive events per camera respect a cooldown.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from config import (MOTION_COOLDOWN_SEC, MOTION_DEBUG, MOTION_DOWNSCALE,
                    MOTION_MIN_DURATION_SEC, MOTION_MIN_SCORE)
from src.domain import MOTION_DETECTED

logger = logging.getLogger("cctv.motion")

_PIXEL_DIFF_THRESHOLD = 25  # grayscale delta considered "changed"


def _downscaled_resolution(w: int, h: int, max_width: int) -> tuple[int, int]:
    """Width-preserving downscale target, honouring ``max_width``."""
    if max_width <= 0 or w <= max_width:
        return w, h
    scale = max_width / float(w)
    return int(round(w * scale)), int(round(h * scale))


class MotionDetector:
    """Per-camera frame differencing with event gating."""

    def __init__(self, *, min_score: float = MOTION_MIN_SCORE,
                 min_duration_sec: float = MOTION_MIN_DURATION_SEC,
                 cooldown_sec: float = MOTION_COOLDOWN_SEC,
                 downscale: int = MOTION_DOWNSCALE,
                 pixel_diff: int = _PIXEL_DIFF_THRESHOLD,
                 enabled: bool = True,
                 debug: bool | None = None):
        self._min_score = float(min_score)
        self._min_duration = float(min_duration_sec)
        self._cooldown = float(cooldown_sec)
        self._downscale = int(downscale)
        self._pixel_diff = int(pixel_diff)
        self._enabled = bool(enabled)
        self._debug = MOTION_DEBUG if debug is None else bool(debug)
        # cam -> previous downscaled gray frame
        self._prev: dict[str, np.ndarray] = {}
        # cam -> active motion run {"since": float, "score": float, "peak": float}
        self._active: dict[str, dict] = {}
        # cam -> last emitted event epoch (cooldown gate)
        self._fire_at: dict[str, float] = {}
        # cam -> live status (serialised for the dashboard)
        self._status: dict[str, dict] = {}

    # ------------------------------------------------------------------

    def process(self, frames: dict[str, object],
                ts: float | None = None) -> tuple[dict, list[dict]]:
        """Run one motion cycle over ``frames``.

        Returns ``(status_by_camera, events)``.  ``status_by_camera`` is a
        live, JSON-friendly view; ``events`` is a list of newly-emitted
        MOTION_DETECTED event dicts (ready for ``db.insert_motion_event``).
        """
        now = time.time() if ts is None else ts

        if not self._enabled:
            self._prev = {}
            self._active = {}
            self._status = {}
            return {}, []

        events: list[dict] = []

        # Drop cameras that stopped delivering frames (never diff stale scenes).
        for cam in list(self._prev):
            if cam not in frames:
                self._prev.pop(cam, None)
                self._active.pop(cam, None)

        for cam, frame in frames.items():
            if frame is None:
                continue
            try:
                gray = self._to_gray(frame)
            except Exception:  # pragma: no cover -- degraded frame guard
                continue
            if gray is None:
                continue

            prev = self._prev.get(cam)
            self._prev[cam] = gray
            if prev is None:
                # First frame for this camera: prime the reference only.
                self._status[cam] = {
                    "motion": False, "score": 0.0, "active_since": None,
                    "primed": True,
                }
                continue

            score, nbox = self._score(prev, gray)
            self._advance(cam, score, nbox, now, events)
            run = self._active.get(cam)
            status = {
                "motion": run is not None,
                "score": round(run["score"], 4) if run else round(score, 4),
                "active_since": round(run["since"] - now, 3) if run else None,
                "last_fire_sec": round(now - self._fire_at[cam], 1)
                if cam in self._fire_at else None,
            }
            self._status[cam] = status

        return dict(self._status), events

    # ------------------------------------------------------------------

    def _to_gray(self, frame) -> np.ndarray | None:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return None
        fw, fh = _downscaled_resolution(w, h, self._downscale)
        if (fw, fh) != (w, h):
            small = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (7, 7), 0)

    @staticmethod
    def _score(prev: np.ndarray, gray: np.ndarray) -> tuple[float, tuple]:
        diff = cv2.absdiff(prev, gray)
        mask = (diff > _PIXEL_DIFF_THRESHOLD).astype(np.uint8) * 255
        changed = cv2.countNonZero(mask)
        total = mask.shape[0] * mask.shape[1]
        score = changed / float(total) if total else 0.0

        nbox = None
        ys, xs = np.nonzero(mask)
        if ys.size:
            nbox = (
                round(float(xs.min()) / mask.shape[1], 4),
                round(float(ys.min()) / mask.shape[0], 4),
                round(float(xs.max()) / mask.shape[1], 4),
                round(float(ys.max()) / mask.shape[0], 4),
            )
        return score, nbox

    def _advance(self, cam: str, score: float, nbox: tuple,
                 now: float, events: list[dict]) -> None:
        run = self._active.get(cam)
        if score >= self._min_score:
            if run is None:
                run = {"since": now, "score": score, "peak": score}
                self._active[cam] = run
            else:
                run["score"] = max(run["score"], score)
                run["peak"] = max(run["peak"], score)
            # A long-enough run fires immediately; _fire enforces the
            # cooldown, so continuous motion re-fires once per window.
            if now - run["since"] >= self._min_duration:
                self._fire(cam, run, now, events)
                self._active.pop(cam, None)
            return

        # Motion ended within a run that never matured -> discard quietly.
        if run is not None:
            self._active.pop(cam, None)

    def _fire(self, cam: str, run: dict, now: float, events: list[dict]) -> None:
        if cam in self._fire_at and now - self._fire_at[cam] < self._cooldown:
            return
        self._fire_at[cam] = now
        duration = now - run["since"]
        ev = {
            "event_type": MOTION_DETECTED,
            "camera": cam,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "confidence": round(min(1.0, run["peak"]), 4),
            "motion_score": round(run["peak"], 4),
            "duration_seconds": round(duration, 2),
            "bbox": None,
        }
        events.append(ev)
        if self._debug:
            logger.info(
                "MOTION_DEBUG camera=%s score=%.4f duration=%.1fs",
                cam, run["peak"], duration,
            )

    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Live per-camera status for ``live_state.json`` (JSON-safe)."""
        return dict(self._status)