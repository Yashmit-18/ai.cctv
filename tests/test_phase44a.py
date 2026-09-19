"""Phase 44A regression tests -- real-DroidCam pipeline performance baseline.

Phase 44A measured the real pipeline (``main.py --source 0 --headless`` with a
DroidCam virtual camera) and attributed the ~0.9 FPS bottleneck:

* capture:          ~1.4 ms  (threaded reader, latest-frame policy -- fine)
* detect:           ~850 ms  (**the bottleneck**, steady-state)
* state / live:     ~10 ms  (fine)
* per-frame split (this CPU, 640x480, 2 persons):
    insightface ``app.get``  ~516 ms   <- ran EVERY cycle (face_cadence 1)
    YOLO person+phone @640   ~87 ms     (runs every cycle, required)
    YOLO phone-only @1280    ~262 ms    every PHONE_DETECT_CADENCE=3 (~87 ms avg)

The fix is the Phase 43 cadence knob made the *default*: recognition now runs
once every 5 processed cycles per camera (``CCTV_FACE_DETECT_CADENCE=5``), so
the ~516 ms InsightFace call amortizes to ~103 ms while YOLO person/phone
presence still runs every cycle.  These tests lock:

1. the tuned production defaults (cadence + rate caps) so a future change to
   the hot-path defaults is a deliberate decision;
2. the cadence gate itself at the new default (1 rare InsightFace call per 5
   cycles) and that YOLO-derived person presence is emitted on skip frames --
   the guarantee that thinned recognition never fakes an AWAY/absent employee.

No models, no network, no webcams.  Fake wall-clock only where needed.
"""

import numpy as np

import config as _cfg
from src.detector import ActivityDetector


class _BoxLike:
    def __init__(self, cls, xyxy, conf):
        self.cls = cls
        self.conf = conf
        self.xyxy = np.array([xyxy])  # helper reads box.xyxy[0].tolist()


class _Boxes:
    def __init__(self):
        self._items = [_BoxLike(0.0, (100.0, 40.0, 300.0, 460.0), 0.9)]

    def __iter__(self):
        return iter(self._items)


class _Result:
    boxes = _Boxes()


class _Model:
    def predict(self, frames, **kwargs):
        return [_Result() for _ in frames]


class _CountingApp:
    """InsightFace stand-in that counts ``app.get`` calls and finds no faces."""

    def __init__(self):
        self.calls = 0

    def get(self, frame):
        self.calls += 1
        return []


class _CountingReg:
    def __init__(self):
        self.app = _CountingApp()


def _det_stub(face_cadence):
    det = ActivityDetector.__new__(ActivityDetector)
    det.conf = 0.3
    det._device = "cpu"
    det._half = False
    det.model = _Model()
    det._face_registry = _CountingReg()
    det._face_cadence = face_cadence
    det._face_ticks = {}
    det._phone_diag_on = False
    det._diag_cycles = 0
    det._phone_stats = {}
    return det


# ======================================================================
# 1) Production hot-path defaults (locked after the real measurement)
# ======================================================================

def test_face_detect_cadence_default_is_5():
    """Phase 44A tuning: recognition once every 5 cycles, not every frame."""
    assert _cfg.FACE_DETECT_CADENCE == 5


def test_phone_and_reid_cadences_keep_baseline_defaults():
    """The phone (3) and appearance (3) passes keep their Phase 44 cadences."""
    assert _cfg.PHONE_DETECT_CADENCE == 3
    assert _cfg.REID_CADENCE == 3


def test_target_fps_default_caps_loop_at_2fps():
    """Loop throttle unchanged -- the sleep cap, not the AI, is the 2 FPS wall."""
    assert _cfg.TARGET_FPS_PER_CAMERA == 2


# ======================================================================
# 2) Cadence gate at the new default + presence-always guarantee
# ======================================================================

def test_face_cadence_5_runs_insightface_once_per_five_cycles():
    det = _det_stub(face_cadence=5)
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    for i in range(10):
        det.detect_batch({"cam1": frame})
        expected = sum(1 for j in range(i + 1) if j % 5 == 0)
        assert det._face_registry.app.calls == expected, f"cycle {i + 1}"


def test_face_cadence_5_emits_person_presence_on_skip_frames():
    """Recognition is thinned; YOLO body presence is NOT (no fake AWAY)."""
    det = _det_stub(face_cadence=5)
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    det.detect_batch({"cam1": frame})   # tick: face pass runs
    for _ in range(4):                  # cycles 2..5: cadence skips
        det.detect_batch({"cam1": frame})
    out = det.detect_batch({"cam1": frame})   # skip frame (cycle 5)
    person_dets = [d for d in out if d.get("person_box") is not None]
    sentinel = [d for d in out if "person_present" in d]
    assert person_dets, "YOLO person entries must appear on every cycle"
    assert sentinel and all(d.get("person_present") is True for d in sentinel)
    det.detect_batch({"cam1": frame})   # cycle 6: tick
    # The expensive recognition only ran on the two tick cycles (1, 6).
    assert det._face_registry.app.calls == 2