"""Phase 45 regression tests -- performance attribution, phone, Re-ID safety.

Phase 45 is the "real-lag / phone / side-pose person Re-ID" release.  These
tests lock (in order of the Phase 45 plan):

  1. Performance attribution (Part F extension): opt-in per-stage timings so
     ``CCTV_PIPELINE_TIMING=1`` publishes an itemized split of ``detect_ms``
     (person-YOLO / phone pass / face / appearance Re-ID).  Zero cost and zero
     behavioural change when the flag is off (the default).
  2. Latest-frame / no-stale-queue guarantees on the live-loop path.
  3. Phone diagnostics + temporal logic (5 s ACTIVE, one alert per episode,
     no false 5 s episodes, intermittent-miss smoothing).
  4. Conservative person Re-ID (identity priority, no raw-vector leaks).

This file grows across the phase; sections are appended as their code lands.
No models, no network, no webcams.  Fake wall-clock only where needed.
"""

import time

import numpy as np

import config as _cfg
from src.detector import ActivityDetector


# ======================================================================
# 1) Performance attribution (Phase 45 Part 2/18) -- opt-in stage timings
# ======================================================================

class _BoxLike:
    def __init__(self, cls, xyxy, conf):
        self.cls = cls
        self.conf = conf
        self.xyxy = np.array([xyxy])


class _Boxes:
    def __init__(self, boxes):
        self._items = boxes

    def __iter__(self):
        return iter(self._items)


class _Result:
    def __init__(self, boxes=None):
        self.boxes = _Boxes(boxes if boxes is not None
                            else [_BoxLike(0.0, (100.0, 40.0, 300.0, 460.0),
                                           0.9)])


class _Model:
    def __init__(self, boxes=None, pause_s=0.0):
        self._boxes = boxes
        self._pause = pause_s

    def predict(self, frames, **kwargs):
        if self._pause:
            time.sleep(self._pause)
        return [_Result(self._boxes) for _ in frames]


class _CountingApp:
    def __init__(self, pause_s=0.0):
        self.calls = 0
        self._pause = pause_s

    def get(self, frame):
        self.calls += 1
        if self._pause:
            time.sleep(self._pause)
        return []


class _CountingReg:
    def __init__(self, app):
        self.app = app


def _det_stub(*, face_cadence=1, perf_on=False, phone_detector=None,
              reid_enabled=False, reid_registry=None, model=None,
              config=None):
    cfg = config if config is not None else {
        "CONF": 0.3, "PHONE_CLASS": 67, "PERSON_CLASS": 0,
    }
    det = ActivityDetector.__new__(ActivityDetector)
    det.conf = cfg["CONF"]
    det._device = "cpu"
    det._half = False
    det.model = model if model is not None else _Model()
    det._face_registry = _CountingReg(_CountingApp())
    det._face_cadence = face_cadence
    det._face_ticks = {}
    det._phone_diag_on = False
    det._diag_cycles = 0
    det._phone_stats = {}
    det._phone_cadence = getattr(_cfg, "PHONE_DETECT_CADENCE", 3)
    det._phone_ticks = {}
    det._reid_cadence = getattr(_cfg, "REID_CADENCE", 3)
    det._reid_ticks = {}
    det._perf_on = perf_on
    det._perf_raw = {"person_yolo_ms": 0.0, "phone_ms": 0.0,
                     "face_ms": 0.0, "reid_ms": 0.0}
    det._phone_detector = phone_detector
    det._reid_enabled = reid_enabled
    det._reid_registry = reid_registry
    return det


_STAGE_KEYS = ("person_yolo_ms", "phone_ms", "face_ms", "reid_ms")


def test_stage_timing_is_safe_for_legacy_stubs():
    """A ``__new__`` stub without perf plumbing must report {} (never raise)."""
    det = ActivityDetector.__new__(ActivityDetector)
    st = det.stage_timing()
    assert isinstance(st, dict)


def test_stage_timing_defaults_off():
    """Production default: CCTV_PIPELINE_TIMING=0 -> zero timing overhead."""
    assert _cfg.PIPELINE_TIMING is False or _cfg.PIPELINE_TIMING == 0
    det = _det_stub(perf_on=False)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    det.detect_batch({"cam1": frame})
    st = det.stage_timing()
    for key in _STAGE_KEYS:
        assert key in st and st[key] == 0.0


def test_stage_timing_records_person_yolo_when_on():
    det = _det_stub(perf_on=True, model=_Model(pause_s=0.005))
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    det.detect_batch({"cam1": frame})
    st = det.stage_timing()
    assert st["person_yolo_ms"] > 0.0
    for key in _STAGE_KEYS:
        assert key in st and st[key] >= 0.0


def test_stage_timing_resets_every_cycle():
    """A slow stage must not accumulate across calls (each cycle re-measures)."""
    det = _det_stub(perf_on=True, model=_Model(pause_s=0.005))
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    det.detect_batch({"cam1": frame})
    first = det.stage_timing()["person_yolo_ms"]
    det.detect_batch({"cam1": frame})
    second = det.stage_timing()["person_yolo_ms"]
    det.detect_batch({"cam1": frame})
    third = det.stage_timing()["person_yolo_ms"]
    assert first > 0.0
    assert second < first * 3 + 5.0   # no unbounded accumulation
    assert third < first * 3 + 5.0


def test_face_stage_timing_follows_cadence():
    """face_ms measured only on cadence tick cycles (Part 18 budget)."""
    app = _CountingApp(pause_s=0.004)
    det = _det_stub(face_cadence=5, perf_on=True)
    det._face_registry = _CountingReg(app)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    for i in range(11):
        det.detect_batch({"cam1": frame})
        on_tick = (i % 5 == 0)
        face_ms = det.stage_timing()["face_ms"]
        if on_tick:
            assert face_ms > 0.0, f"cycle {i} should run the face stage"
        else:
            assert face_ms == 0.0, f"cycle {i} must skip the face stage"


def test_phone_stage_timing_follows_cadence():
    class _Phone:
        ready = True

        def detect(self, frame):
            time.sleep(0.004)
            return []

    det = _det_stub(perf_on=True, phone_detector=_Phone())
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    cadence = _cfg.PHONE_DETECT_CADENCE
    for i in range(cadence * 2 + 1):
        det.detect_batch({"cam1": frame})
        phone_ms = det.stage_timing()["phone_ms"]
        if i % cadence == 0:
            assert phone_ms > 0.0, f"cycle {i} should run the phone pass"
        else:
            assert phone_ms == 0.0, f"cycle {i} must skip the phone pass"


def test_reid_stage_timing_when_enabled():
    class _Embed:
        model_source = "built-in"

        def embed(self, crop):
            time.sleep(0.004)
            return np.zeros(8, dtype=np.float32)

    class _Reg:
        extractor = _Embed()

        def identify(self, emb):
            return ("Unknown", 0.0, 0.0, False, None)

    det = _det_stub(perf_on=True, reid_enabled=True, reid_registry=_Reg())
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    cadence = _cfg.REID_CADENCE
    for i in range(cadence * 2 + 1):
        det.detect_batch({"cam1": frame})
        reid_ms = det.stage_timing()["reid_ms"]
        if i % cadence == 0:
            assert reid_ms > 0.0, f"cycle {i} should run the appearance pass"
        else:
            assert reid_ms == 0.0, f"cycle {i} must skip the appearance pass"


def test_main_published_keys_contract():
    """The exact key set main.py folds into the EMA (asserted here)."""
    keys = {"capture_ms", "person_yolo_ms", "phone_ms", "face_ms", "reid_ms",
            "tracking_ms", "db_ms", "events_ms", "detect_ms", "state_ms",
            "live_state_ms", "other_ms", "frame_age_ms", "total_ms"}
    # ensure the source of truth (this test) stays in sync with main.py
    import inspect
    import main as _main_mod
    src = inspect.getsource(_main_mod)
    for key in keys:
        quoted = '"%s"' % key
        assert quoted in src, f"main.py no longer publishes {quoted}"


# ======================================================================
# 2) Loop FPS cap for single local sources (Phase 45 Part 2/19/23)
# ======================================================================

def test_local_source_faster_cap_default():
    """--source <index> should no longer idle half the loop on a sleep."""
    assert _cfg.LOCAL_SOURCE_DEFAULT_TARGET_FPS >= 4
    assert _cfg.effective_target_fps(local_source=True, env={}) == \
        _cfg.LOCAL_SOURCE_DEFAULT_TARGET_FPS


def test_multi_camera_keeps_original_cap():
    """The multi-camera round-robin budget is untouched (Phase 44A default)."""
    assert _cfg.TARGET_FPS_PER_CAMERA == 2
    assert _cfg.effective_target_fps(local_source=False, env={}) == 2


def test_explicit_env_override_wins_for_local_source():
    """An operator-set CCTV_TARGET_FPS must always beat the local default."""
    assert _cfg.effective_target_fps(
        local_source=True, env={"CCTV_TARGET_FPS": "6"}) == 6
    assert _cfg.effective_target_fps(
        local_source=True, env={"CCTV_TARGET_FPS": "2"}) == 2


def test_local_source_cap_is_validated():
    problems = _cfg.validate_config()
    assert all("CCTV_LOCAL_TARGET_FPS" not in p for p in problems)


def test_frame_interval_reflects_cap():
    from src.camera_manager import MultiCameraManager
    assert MultiCameraManager.frame_interval(4) < \
        MultiCameraManager.frame_interval(2)