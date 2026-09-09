"""Per-camera tamper detection (Phase 31).

Analyses cheap grayscale statistics on frames already in the batch pipeline.
Detects: frozen frames, extreme darkness, extreme brightness.

Uses persistence thresholds to avoid false positives on normal darkness
or single-frame glitches.  Fires CAMERA_TAMPER_SUSPECTED with confidence
and CAMERA_TAMPER_CLEARED on recovery.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from src.domain import CAMERA_TAMPER_CLEARED, CAMERA_TAMPER_SUSPECTED

logger = logging.getLogger("cctv.tamper")

# Minimum frames before a tamper condition is confirmed.
_MIN_TAMPER_FRAMES = 3


class TamperMonitor:
    """Per-camera tamper state machine.

    Call ``analyze(camera_id, frame)`` each frame.  Returns events when
    transitions occur.
    """

    def __init__(self, persist_sec: float = 30.0,
                 dark_mean: float = 30.0, bright_mean: float = 235.0,
                 frozen_diff: float = 0.5, low_std: float = 5.0,
                 enable_dark: bool = True):
        self._persist = persist_sec
        self._dark_mean = dark_mean
        self._bright_mean = bright_mean
        self._frozen_diff = frozen_diff
        self._low_std = low_std
        self._enable_dark = enable_dark
        self._state: dict[str, dict] = {}  # camera_id -> internal state

    def analyze(self, camera_id: str, frame: "np.ndarray",
                now: float | None = None) -> list[dict]:
        """Analyze one frame.  Returns 0 or 1 event dicts."""
        now = now or time.time()
        if frame is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))

        # Compare with previous frame for frozen detection
        prev = self._state.get(camera_id)
        frozen = False
        if prev is not None and prev.get("prev_gray") is not None:
            diff = float(np.mean(cv2.absdiff(gray, prev["prev_gray"])))
            frozen = diff < self._frozen_diff

        # Determine if this frame exhibits a tamper signal
        tamper_reason = None
        if frozen:
            tamper_reason = "frozen_frame"
        elif self._enable_dark and mean_val < self._dark_mean and std_val < self._low_std:
            tamper_reason = "extreme_darkness"
        elif mean_val > self._bright_mean and std_val < self._low_std:
            tamper_reason = "extreme_brightness"

        # Update state
        s = self._state.setdefault(camera_id, {})
        s["prev_gray"] = gray
        s["last_mean"] = mean_val
        s["last_std"] = std_val

        events: list[dict] = []
        tamper_active = s.get("tamper_active", False)

        if tamper_reason:
            if not tamper_active:
                s["tamper_start"] = now
                s["tamper_reason"] = tamper_reason
                s["tamper_active"] = False  # not yet confirmed
            elapsed = now - s.get("tamper_start", now)
            if elapsed >= self._persist and not tamper_active:
                s["tamper_active"] = True
                denom = max(self._persist * 2, 1.0)
                confidence = min(1.0, elapsed / denom)
                logger.info("[SECURITY] CAMERA_TAMPER_SUSPECTED camera=%s reason=%s persist=%.1fs",
                            camera_id, tamper_reason, elapsed)
                events.append({
                    "event_type": CAMERA_TAMPER_SUSPECTED,
                    "camera": camera_id,
                    "confidence": round(confidence, 3),
                    "details": tamper_reason,
                    "duration_seconds": round(elapsed, 1),
                })
        else:
            if tamper_active:
                # Recovery
                s["tamper_active"] = False
                s.pop("tamper_start", None)
                s.pop("tamper_reason", None)
                logger.info("[SECURITY] CAMERA_TAMPER_CLEARED camera=%s", camera_id)
                events.append({
                    "event_type": CAMERA_TAMPER_CLEARED,
                    "camera": camera_id,
                    "confidence": 1.0,
                })
            else:
                # Reset start if condition was transient
                s.pop("tamper_start", None)
                s.pop("tamper_reason", None)

        return events

    def is_tampered(self, camera_id: str) -> bool:
        return self._state.get(camera_id, {}).get("tamper_active", False)
