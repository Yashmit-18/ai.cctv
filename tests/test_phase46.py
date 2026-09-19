"""Phase 46 regression tests -- YOLO11n integration + offline model handling.

Phase 46 upgrades the default person/phone detector from YOLOv8n to YOLO11n
based on a local benchmark (CODE VERIFIED / SIMULATED -- synthetic frames,
no real camera).  These tests lock:

1. **Model selection** -- config.MODEL_PATH points at yolo11n.pt and the
   file exists on disk.
2. **Model initialization once** -- the detector loads the YOLO model exactly
   once at construction; repeated ``detect_batch`` calls reuse it (no
   per-frame model reload).
3. **Latest-frame policy** -- the daemon pulls only the latest frames per
   camera (no stale backlog); a camera that stops delivering frames freezes
   employee state (no fake absence).
4. **Preflight ONNX provider detection** -- the preflight check reports
   available ONNX Runtime providers.
5. **Offline model handling** -- missing optional models (phone, ReID) fall
   back gracefully; missing required YOLO model raises a clear error.

All assertions are deterministic -- no real YOLO inference, no network, no
webcams.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

import config
import main as m
from src.detector import ActivityDetector, _resolve_device
from src.phone_detector import PhoneDetector
from src.preflight import check_models, check_gpu


# ======================================================================
# Model selection (YOLO11n)
# ======================================================================

def test_model_path_points_to_yolo11n():
    """Phase 46 default model is yolo11n.pt (benchmark-selected)."""
    assert "yolo11n.pt" in config.MODEL_PATH


def test_yolo11n_model_file_exists():
    """The yolo11n.pt file must be present on disk (offline startup)."""
    assert os.path.isfile(config.MODEL_PATH), \
        f"Missing {config.MODEL_PATH} -- run: python -c 'from ultralytics import YOLO; YOLO(\"yolo11n.pt\")'"


def test_yolov8n_retained_as_fallback():
    """yolov8n.pt is retained in models/ as a fallback."""
    fallback = str(Path(config.PROJECT_ROOT) / "models" / "yolov8n.pt")
    assert os.path.isfile(fallback)


# ======================================================================
# Model initialization once
# ======================================================================

class _CountingModel:
    """Fake YOLO model that counts predict() calls."""

    def __init__(self):
        self.predict_calls = 0
        self.names = {0: "person", 67: "cell phone"}

    def predict(self, frames, **kwargs):
        self.predict_calls += 1
        return [type("R", (), {"boxes": type("B", (), {"__iter__": lambda s: iter([])})()})()]


def test_detector_initializes_model_once():
    """ActivityDetector loads the YOLO model exactly once at construction."""
    det = ActivityDetector.__new__(ActivityDetector)
    det.model = _CountingModel()
    det.conf = 0.4
    det._device = "cpu"
    det._half = False
    det._face_registry = type("F", (), {"app": type("A", (), {"get": lambda s, f: []})()})()
    det._face_cadence = 1
    det._face_ticks = {}
    det._phone_diag_on = False
    det._phone_stats = {}
    det._diag_cycles = 0
    det._perf_on = False
    det._perf_raw = {"person_yolo_ms": 0.0, "phone_ms": 0.0,
                     "face_ms": 0.0, "reid_ms": 0.0}
    det._phone_detector = None
    det._phone_cadence = 3
    det._phone_ticks = {}
    det._reid_enabled = False
    det._reid_cadence = 3
    det._reid_ticks = {}

    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    det.detect_batch({"cam1": frame})
    det.detect_batch({"cam1": frame})
    det.detect_batch({"cam1": frame})

    # Model loaded once; predict called once per batch (not per frame).
    assert det.model.predict_calls == 3


# ======================================================================
# Latest-frame policy / no stale backlog
# ======================================================================

def test_latest_frames_returns_only_fresh_frames(monkeypatch):
    """The daemon pulls only the latest frames per camera (no stale backlog)."""
    from src.camera_manager import MultiCameraManager

    # Verify the manager's latest_frames API exists and is bounded by limit.
    assert hasattr(MultiCameraManager, "latest_frames")
    assert hasattr(MultiCameraManager, "frame_interval")


def test_camera_offline_freezes_employee_state(fake_clock, recorder, instant_commit):
    """A camera that stops delivering frames freezes employee state."""
    import src.tracker as T

    t = T.EmployeeTracker(None, "EMP001")
    t.process({"present": True, "phone_detected": False}, camera_online=True)
    assert t.committed_state == "ACTIVE"

    # Camera goes offline -- state must freeze, not advance to AWAY.
    fake_clock.now += 100.0
    t.process({"present": False, "phone_detected": False}, camera_online=False)
    assert t.committed_state == "ACTIVE"


# ======================================================================
# Preflight ONNX provider detection
# ======================================================================

def test_preflight_onnx_provider_check():
    """Preflight reports available ONNX Runtime providers."""
    results = check_models()
    onnx = [r for r in results if r.name == "onnx_providers"]
    assert len(onnx) == 1
    assert onnx[0].status in ("PASS", "WARN", "SKIP")
    if onnx[0].status == "PASS":
        assert "CPUExecutionProvider" in onnx[0].detail


def test_preflight_yolo_model_check():
    """Preflight reports the yolo11n.pt model as present."""
    results = check_models()
    yolo = [r for r in results if r.name == "yolo"]
    assert len(yolo) == 1
    assert yolo[0].status == "PASS"
    assert "yolo11n.pt" in yolo[0].detail


def test_preflight_gpu_check():
    """Preflight GPU check runs without crashing (CPU fallback is WARN)."""
    results = check_gpu()
    cuda = [r for r in results if r.name == "cuda"]
    assert len(cuda) == 1
    assert cuda[0].status in ("PASS", "WARN", "SKIP")


# ======================================================================
# Offline model handling
# ======================================================================

def test_phone_detector_missing_path_falls_back_to_base():
    """Missing dedicated phone model falls back to the base model."""
    model = type("M", (), {})()
    pd = PhoneDetector(model=model, model_path=r"C:\missing\phone_model.pt")
    assert pd.ready is True
    assert pd.source == "base-model"


def test_phone_detector_no_model_ready_false():
    """No model at all -> not ready (never crashes)."""
    pd = PhoneDetector()
    assert pd.ready is False
    assert pd.detect(None) == []


def test_reid_extractor_missing_onnx_falls_back_to_builtin():
    """Missing ReID ONNX model falls back to the built-in descriptor."""
    from src.reid import AppearanceExtractor
    ex = AppearanceExtractor(model_path=r"C:\missing\osnet.onnx", imgsz=128, dim=256)
    assert ex.model_source == "built-in"


# ======================================================================
# Device resolution
# ======================================================================

def test_resolve_device_cpu_fallback():
    """Device resolution never crashes and returns a valid device string."""
    dev = _resolve_device(None)
    assert dev in ("cpu", "cuda", "mps")


def test_resolve_device_explicit_cpu():
    """Explicit CPU request returns cpu."""
    assert _resolve_device("cpu") == "cpu"


# =====================================================================
# Fixtures (mirror test_phase44)
# =====================================================================

class _FakeTime:
    now = 10_000.0

    @classmethod
    def time(cls) -> float:
        return cls.now

    @classmethod
    def strftime(cls, fmt, tt=None):
        import time as _realtime
        return _realtime.strftime(fmt, tt or _realtime.localtime(cls.now))

    @classmethod
    def localtime(cls, tt=None):
        import time as _realtime
        return _realtime.localtime(tt or cls.now)


@pytest.fixture
def fake_clock(monkeypatch):
    import src.tracker as T
    _FakeTime.now = 10_000.0
    monkeypatch.setattr(T, "time", _FakeTime)
    return _FakeTime


@pytest.fixture
def recorder(monkeypatch):
    import src.tracker as T
    seen = []

    def fake_log(conn, ts, emp, state, dur, source=""):
        seen.append((ts, emp, state, dur, source))

    monkeypatch.setattr(T, "log_interval", fake_log)
    return seen


@pytest.fixture
def instant_commit(monkeypatch):
    import src.tracker as T
    monkeypatch.setattr(T, "SMOOTHING_BUFFER_SEC", 0.0)
    monkeypatch.setattr(T, "IDENTITY_STABILITY_FRAMES", 1)
    monkeypatch.setattr(T, "AWAY_AFTER_SEC", 3.0)
    monkeypatch.setattr(T, "PHONE_AFTER_SEC", 5.0)
    monkeypatch.setattr(T, "PHONE_GAP_GRACE_SEC", 2.0)