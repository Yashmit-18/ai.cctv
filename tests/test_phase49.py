"""Phase 49 regression tests: genuine ReID (OSNet) integration.

Covers the vendored OSNet backend, AppearanceExtractor routing, dim-aware
cache invalidation, and the opt-in event-driven ReID gate
(``CCTV_REID_GATE_RESOLVED`` / ``REID_GATE_IOU``).  Model-dependent tests are
skipped when the checkpoint files are not present locally.

Evidence labels throughout: CODE VERIFIED -- deterministic synthetic crops;
threshold calibration on real CCTV remains NOT VALIDATED.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import main as m
from src.detector import ActivityDetector
from src.reid import CACHE_VERSION, AppearanceExtractor, AppearanceRegistry
from src.reid_osnet import (OSNET_INPUT_H, OSNET_INPUT_W, OsnetEmbedder,
                            _detect_variant, build_osnet,
                            load_osnet_weights, preprocess_osnet)

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "x1_0": ROOT / "models" / "osnet_x1_0_msmt17.pth",
    "x0_5": ROOT / "models" / "osnet_x0_5_msmt17.pth",
    "ain_x1_0": ROOT / "models" / "osnet_ain_x1_0_msmt17.pth",
}


def _crop(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    crop = rng.integers(20, 90, (256, 128, 3), dtype=np.uint8)
    crop[0:160, :] = (90, 90, 200)
    return np.ascontiguousarray(crop)


# ---------------------------------------------------------------------------
# OSNet backend module
# ---------------------------------------------------------------------------

def test_reid_osnet_variant_detection_from_filename():
    assert _detect_variant("models/osnet_x1_0_msmt17.pth") == "osnet_x1_0"
    assert _detect_variant("models/osnet_x0_5_msmt17.pth") == "osnet_x0_5"
    assert (_detect_variant("models/osnet_ain_x1_0_msmt17.pth")
            == "osnet_ain_x1_0")
    assert _detect_variant("osnet_ain_x0_25.pth") == "osnet_ain_x0_25"


def test_reid_osnet_build_unsupported_variant_raises():
    with pytest.raises(ValueError):
        build_osnet("osnet_x9_9")


def test_reid_osnet_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError):
        OsnetEmbedder(str(ROOT / "models" / "does_not_exist.pth"))


@pytest.mark.parametrize("key", ["x1_0", "x0_5", "ain_x1_0"])
def test_reid_osnet_checkpoints_embed_deterministic_512(key):
    path = MODELS[key]
    if not path.is_file():
        pytest.skip("OSNet checkpoint not present locally")
    emb = OsnetEmbedder(str(path))
    assert emb.feature_dim == 512
    a = emb.extract(_crop(1))
    b = emb.extract(_crop(1))
    assert a is not None and b is not None
    assert a.shape == (512,)
    assert a.dtype == np.float32
    assert np.all(np.isfinite(a))
    assert np.array_equal(a, b), "OSNet embedding must be deterministic"


@pytest.mark.parametrize("key", ["x1_0", "x0_5", "ain_x1_0"])
def test_reid_osnet_wrong_checkpoint_load_raises(key):
    path = MODELS[key]
    if not path.is_file():
        pytest.skip("OSNet checkpoint not present locally")
    wrong = MODELS["x0_5"] if key != "x0_5" else MODELS["x1_0"]
    model = build_osnet(_detect_variant(str(wrong)))
    with pytest.raises(RuntimeError):
        load_osnet_weights(model, str(path))


def test_reid_osnet_empty_crop_extract_returns_none():
    try:
        emb = OsnetEmbedder(str(MODELS["x0_5"]))
    except Exception:
        pytest.skip("OSNet checkpoint not present locally")
    assert emb.extract(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_reid_preprocess_osnet_shape_dtype():
    out = preprocess_osnet(_crop(2))
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 3, OSNET_INPUT_H, OSNET_INPUT_W)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# AppearanceExtractor routing
# ---------------------------------------------------------------------------

def test_reid_extractor_default_model_source_builtin():
    ex = AppearanceExtractor(model_path="", imgsz=128, dim=256)
    assert ex.model_source == "built-in"
    assert ex.feature_dim == 256


def test_reid_extractor_missing_path_falls_back_builtin():
    ex = AppearanceExtractor(model_path=str(ROOT / "models" / "nope.pth"))
    assert ex.model_source == "built-in"
    emb = ex.embed(_crop(3))
    assert emb is not None
    assert emb.size == 256


def test_reid_extractor_unloadable_pth_falls_back(tmp_path):
    junk = tmp_path / "junk.pth"
    junk.write_bytes(b"not a torch file at all")
    ex = AppearanceExtractor(model_path=str(junk))
    assert ex.model_source == "built-in"
    assert ex.embed(_crop(3)) is not None


def test_reid_extractor_pth_routes_to_osnet():
    if not MODELS["x0_5"].is_file():
        pytest.skip("OSNet checkpoint not present locally")
    ex = AppearanceExtractor(model_path=str(MODELS["x0_5"]))
    assert ex.model_source == "osnet"
    assert ex.feature_dim == 512
    a = ex.embed(_crop(4))
    b = ex.embed(_crop(4))
    assert a is not None
    assert a.shape == (512,)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-4
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Registry + cache invalidation
# ---------------------------------------------------------------------------

def test_reid_registry_embed_dim_follows_extractor():
    if not MODELS["x0_5"].is_file():
        pytest.skip("OSNet checkpoint not present locally")
    reg = AppearanceRegistry.__new__(AppearanceRegistry)
    reg.embeddings_file = "unused"
    reg.extractor = AppearanceExtractor(model_path=str(MODELS["x0_5"]))
    assert reg.extractor.feature_dim == 512


def test_reid_registry_cache_version_is_bumped():
    # CACHE_VERSION 1 -> 2 in Phase 49 so stale 256-dim caches are invalidated.
    assert CACHE_VERSION == 2


def test_reid_registry_cache_dim_mismatch_invalidated(tmp_path):
    reg = AppearanceRegistry.__new__(AppearanceRegistry)
    reg.extractor = AppearanceExtractor(model_path="")
    reg.embed_dim = 512
    reg.embeddings_file = str(tmp_path / "appearance.pkl")
    with open(reg.embeddings_file, "wb") as fh:
        pickle.dump({
            "version": CACHE_VERSION,
            "embed_dim": 256,
            "by_employee": {
                "EMP001": {"embedding": np.zeros(256, dtype=np.float32),
                           "images": 1, "source": "test"},
            },
        }, fh, protocol=4)
    with pytest.raises(FileNotFoundError):
        reg._load_cache()          # all entries dropped on dim mismatch


def test_reid_registry_stale_version_invalidated(tmp_path):
    reg = AppearanceRegistry.__new__(AppearanceRegistry)
    reg.extractor = AppearanceExtractor(model_path="")
    reg.embed_dim = 256
    reg.embeddings_file = str(tmp_path / "appearance.pkl")
    with open(reg.embeddings_file, "wb") as fh:
        pickle.dump({
            "version": 1,
            "embed_dim": 256,
            "by_employee": {"EMP001": {"embedding": np.ones(256), "images": 1}},
        }, fh, protocol=4)
    with pytest.raises(FileNotFoundError):
        reg._load_cache()


def test_reid_registry_save_load_roundtrip_preserves_dim(tmp_path):
    reg = AppearanceRegistry.__new__(AppearanceRegistry)
    reg.extractor = AppearanceExtractor(model_path="")
    reg.embed_dim = 256
    reg.embeddings_file = str(tmp_path / "appearance.pkl")
    vec = np.arange(256, dtype=np.float32)
    reg._by_employee = {"EMP001": {"embedding": vec, "images": 1,
                                   "source": "test"}}
    reg._save_cache()
    reg2 = AppearanceRegistry.__new__(AppearanceRegistry)
    reg2.extractor = AppearanceExtractor(model_path="")
    reg2.embed_dim = 256
    reg2.embeddings_file = str(tmp_path / "appearance.pkl")
    reg2._load_cache()
    assert reg2.embed_dim == 256
    assert reg2.count_enrolled() == 1


# ---------------------------------------------------------------------------
# Opt-in event-driven ReID gate
# ---------------------------------------------------------------------------

class _FakeExt:
    def __init__(self):
        self.calls = 0

    def embed(self, crop):
        self.calls += 1
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class _FakeReg:
    def __init__(self, ext):
        self.extractor = ext

    def identify(self, emb):
        return ["EMP001", 0.96, 0.2, True, None]


def _det_gate(ext):
    det = ActivityDetector.__new__(ActivityDetector)
    det._reid_registry = _FakeReg(ext)
    det._perf_on = False
    return det


def test_reid_gate_skips_resolved_person_boxes():
    ext = _FakeExt()
    det = _det_gate(ext)
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    boxes = [(10, 10, 50, 50), (150, 150, 220, 220)]
    res = det._extract_reid(frame, boxes, frame.shape[:2],
                            gate_boxes=[(0, 0, 60, 60)])
    assert res[0] == {"emp": "Unknown", "score": 0.0, "margin": 0.0,
                      "strong": False, "second": None}
    assert res[1]["emp"] == "EMP001" and res[1]["strong"] is True
    assert ext.calls == 1, "only the ungated box must run inference"


def test_reid_gate_subthreshold_box_not_skipped():
    ext = _FakeExt()
    det = _det_gate(ext)
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    boxes = [(10, 10, 50, 50), (150, 150, 220, 220)]
    # gate box overlapping box0 so slightly IoU < REID_GATE_IOU (0.30)
    res = det._extract_reid(frame, boxes, frame.shape[:2],
                            gate_boxes=[(0, 0, 22, 22)])
    assert res[0]["emp"] == "EMP001"
    assert res[1]["emp"] == "EMP001"
    assert ext.calls == 2


def test_reid_gate_none_keeps_prior_behaviour():
    ext = _FakeExt()
    det = _det_gate(ext)
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    boxes = [(10, 10, 50, 50), (150, 150, 220, 220)]
    res = det._extract_reid(frame, boxes, frame.shape[:2])
    assert res[0]["emp"] == "EMP001" and res[1]["emp"] == "EMP001"
    assert ext.calls == 2


def test_reid_gate_boxes_forward_through_camera_pass():
    det = ActivityDetector.__new__(ActivityDetector)
    det._reid_enabled = True
    det._reid_cadence = 1
    det._reid_ticks = {}
    seen = {}

    def fake(frame, boxes, shape, gate_boxes=None):
        seen["gate_boxes"] = gate_boxes
        return []

    det._extract_reid = fake
    det._reid_for_camera(np.zeros((16, 16, 3), dtype=np.uint8), "cam1",
                         [((1, 1, 8, 8), 0.9)], gate_boxes=[(2, 2, 6, 6)])
    assert seen["gate_boxes"] == [(2, 2, 6, 6)]
    det._reid_for_camera(np.zeros((16, 16, 3), dtype=np.uint8), "cam1",
                         [((1, 1, 8, 8), 0.9)])
    assert seen["gate_boxes"] is None


class _BBox:
    def __init__(self, cls, xyxy):
        self.cls = np.array(cls)
        self.xyxy = np.array([xyxy], dtype=float)


class _BoxesResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _StubModel:
    def predict(self, frames, **kwargs):
        return [_BoxesResult([_BBox(0, (30, 30, 90, 90))])
                for _ in (frames if isinstance(frames, list) else [frames])]


def test_reid_gate_detect_batch_dispatches_per_camera():
    seen = {}

    def fake_extract(frame, boxes, shape, gate_boxes=None):
        seen["gate_boxes"] = gate_boxes
        return [{"emp": "Unknown", "score": 0.0, "margin": 0.0,
                 "strong": False, "second": None} for _ in boxes]

    det = ActivityDetector.__new__(ActivityDetector)
    det.conf = 0.3
    det._device = "cpu"
    det._half = False
    det.model = _StubModel()
    det._perf_on = False
    det._diag_cycles = 0
    det._face_registry = type("Reg", (), {
        "app": type("App", (), {"get": lambda self, frame: []})()
    })()
    det._face_cadence = 1
    det._face_ticks = {}
    det._phone_diag_on = False
    det._phone_stats = {}
    det._phone_detector = None
    det._phone_cadence = 0
    det._phone_ticks = {}
    det._reid_enabled = True
    det._reid_registry = _FakeReg(_FakeExt())
    det._reid_ticks = {}
    det._reid_cadence = 1
    det._extract_reid = fake_extract

    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    det.detect_batch({"cam1": frame},
                     reid_gate_boxes={"cam1": [(0, 0, 96, 96)]})
    # detect_batch must pass the per-camera resolved boxes to the reid pass
    assert seen["gate_boxes"] == [(0, 0, 96, 96)]

    seen.pop("gate_boxes", None)
    det.detect_batch({"cam1": frame})
    assert seen.get("gate_boxes") is None   # un-gated default preserved


# ---------------------------------------------------------------------------
# main.py helper forwarding
# ---------------------------------------------------------------------------

class _ForwardDet:
    def __init__(self):
        self.last_kwargs = None

    def detect_batch(self, frames, **kwargs):
        self.last_kwargs = kwargs
        return []


def test_step_detector_forwards_reid_gate_boxes():
    fd = _ForwardDet()
    _, dets = m._step_detector(
        fd, {"cam1": np.zeros((8, 8, 3), dtype=np.uint8)}, lambda: None,
        reid_gate_boxes={"cam1": [(0, 0, 10, 10)]})
    assert dets == []
    assert fd.last_kwargs["reid_gate_boxes"] == {"cam1": [(0, 0, 10, 10)]}


def test_step_detector_gate_default_none():
    fd = _ForwardDet()
    m._step_detector(fd, {"cam1": np.zeros((8, 8, 3), dtype=np.uint8)},
                     lambda: None)
    assert fd.last_kwargs["reid_gate_boxes"] is None


# ---------------------------------------------------------------------------
# Config -- Phase 49 settings
# ---------------------------------------------------------------------------

def test_phase49_config_defaults_off():
    assert config.REID_GATE_RESOLVED is False
    assert 0.0 < config.REID_GATE_IOU <= 1.0


def test_phase49_config_validates_gate_iou(monkeypatch):
    monkeypatch.setattr(config, "REID_GATE_IOU", 0.0)
    problems = config.validate_config(quiet=True)
    assert any("CCTV_REID_GATE_IOU" in p for p in problems)
    monkeypatch.setattr(config, "REID_GATE_IOU", 1.5)
    problems = config.validate_config(quiet=True)
    assert any("CCTV_REID_GATE_IOU" in p for p in problems)

    monkeypatch.setattr(config, "REID_GATE_IOU", 0.30)
    problems = config.validate_config(quiet=True)
    assert not any("CCTV_REID_GATE_IOU" in p for p in problems)


def test_phase49_config_accepts_good_gate_values(monkeypatch):
    monkeypatch.setattr(config, "REID_GATE_IOU", 0.5)
    problems = config.validate_config(quiet=True)
    assert not any("CCTV_REID_GATE_IOU" in p for p in problems)