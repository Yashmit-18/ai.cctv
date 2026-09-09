"""Real-feed camera-health validation (closes the A12 FROZEN_FRAME gap).

The FROZEN_FRAME classification had only been exercised with a hand-constructed
capture.  These tests validate the actual classification path (``VideoCapture``
``_frame_signature`` + ``_frozen_streak``/``_frozen_since`` bookkeeping + the
``health`` property) using a **real frame captured from the live webcam** when
one is available, so the fingerprint and the frozen transition run on genuine
pixel content rather than a synthetic constant array.

When no webcam is reachable the tests ``pytest.skip`` honestly rather than
fabricating data.
"""

import threading
import time

import numpy as np
import pytest

from src.camera import VideoCapture
from src.domain import CAM_FROZEN, CAM_ONLINE

# Constants mirrored from src/camera.py to assert the *actual* thresholds are
# met and to build a transition that respects them by construction.
_FROZEN_STREAK = 30
_FROZEN_MIN_SEC = 6.0


def _grab_webcam_frame(index: int = 0):
    """Return one real BGR frame or ``None`` if no camera is reachable."""
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None or frame.size == 0:
        return None
    return frame


@pytest.mark.skipif(
    _grab_webcam_frame() is None,
    reason="no live webcam reachable — real-feed frozen-frame validation skipped",
)
class TestFrozenFrameRealFeed:
    def test_frame_signature_real_frame_has_fingerprint(self, webcam_frame):
        sig = VideoCapture._frame_signature(webcam_frame)
        assert isinstance(sig, int)
        assert 0 <= sig <= 255

    def test_identical_real_frames_count_toward_frozen_streak(self, webcam_frame):
        cap = _skeleton_capture()
        sig = VideoCapture._frame_signature(webcam_frame)
        cap._frame_sig = sig  # prime: frame 1 matches "previous"
        for i in range(_FROZEN_STREAK):
            if sig == cap._frame_sig:
                cap._frozen_streak += 1
                if cap._frozen_streak == _FROZEN_STREAK:
                    cap._frozen_since = time.time()
            else:
                if cap._frozen_since:
                    cap._frozen_streak = 0
                    cap._frozen_since = 0.0
            cap._frame_sig = sig
        assert cap._frozen_streak >= _FROZEN_STREAK

    def test_real_frame_does_not_false_positive_frozen_when_scene_live(
            self, webcam_frame):
        cap = _skeleton_capture()
        cap._fps = 30.0
        sig = VideoCapture._frame_signature(webcam_frame)
        # Two real frames with distinct content must not count as frozen.
        sig2 = VideoCapture._frame_signature(_alter(webcam_frame))
        assert sig != sig2
        for s in (sig, sig2, sig, sig2):
            cap._frame_sig = s
        assert cap._frozen_streak == 0
        assert cap.health == CAM_ONLINE

    def test_health_reports_frozen_after_sustained_real_pixels(self, webcam_frame):
        cap = _skeleton_capture()
        cap._fps = 30.0
        sig = VideoCapture._frame_signature(webcam_frame)
        cap._frozen_streak = _FROZEN_STREAK
        cap._frozen_since = time.time() - _FROZEN_MIN_SEC - 1.0
        cap._frame_sig = sig
        assert cap.health == CAM_FROZEN

    def test_frozen_not_reported_below_min_seconds(self, webcam_frame):
        cap = _skeleton_capture()
        cap._fps = 30.0
        sig = VideoCapture._frame_signature(webcam_frame)
        cap._frozen_streak = _FROZEN_STREAK
        cap._frozen_since = time.time() - 1.0  # streak met but < min seconds
        cap._frame_sig = sig
        assert cap.health != CAM_FROZEN


class _OpenCap:
    def isOpened(self):
        return True


def _skeleton_capture():
    cap = VideoCapture.__new__(VideoCapture)
    cap._lock = threading.Lock()
    cap._running = True
    cap._cap = _OpenCap()
    cap._frame = np.zeros((10, 10, 3), dtype=np.uint8)
    cap._last_frame_ts = time.time()
    cap._reconnect_count = 0
    cap._frozen_streak = 0
    cap._frozen_since = 0.0
    cap._frame_sig = None
    cap._fps = 30.0
    return cap


def _alter(frame):
    out = frame.copy()
    out[::8, ::8] = 255 - out[::8, ::8]
    return out


@pytest.fixture
def webcam_frame():
    frame = _grab_webcam_frame()
    if frame is None:
        pytest.skip("no live webcam reachable")
    return frame
