"""Phase 44 regression tests -- dedicated phone pass + appearance ReID.

Phase 44 addresses the two root causes left over from Phase 43 as MODEL
LIMITATION / NOT VALIDATED:

1. **Dedicated phone pass** (``src/phone_detector.py``).  YOLOv8n's COCO
   ``cell phone`` class is weak on small phones in CCTV frames.  Instead of
   dropping the global confidence threshold, a separate cadence-gated,
   higher-resolution phone pass runs on top of the stock 640 pass with an IoU
   dedupe, and the spatial tracker gates each track's phone report through a
   trailing evidence window so a single random false positive can never start
   a phone episode.

2. **Appearance ReID** (``src/reid.py``).  A fully profile/back-facing person
   has no face, so InsightFace cannot identify them and Phase 43 only knew
   them as "Unknown N".  Appearance identity is a conservative SECONDARY
   signal: only STRONG matches (similarity threshold + ambiguity margin) ever
   carry identity weight, adoption repeats over ``appear_adopt_frames``
   consecutive votes, a face identity always outranks appearance and can never
   be demoted by it, and unresolved tracks stay Unknown.

All assertions are deterministic -- fake models, synthetic images, a fake wall
clock.  No real YOLO / ONNX / InsightFace inference, no network, no webcams.
"""

import os
import sys
import time as _realtime  # noqa: F401  (module seam parity with tracker tests)
import types  # noqa: F401

import numpy as np
import pytest

import config
import main as m
import src.tracker as T
from src.detector import ActivityDetector, _iou, _merge_phone_dets
from src.domain import UNKNOWN_ID
from src.phone_detector import PhoneDetector, _normalize_box
from src.reid import (CACHE_VERSION, AppearanceExtractor, AppearanceRegistry,
                      _employee_id_from_name)
from src.tracker import MultiTracker, SpatialTracker, Track


# ======================================================================
# Fixtures (same seams as test_phase43)
# ======================================================================

class _FakeTime:
    now = 10_000.0

    @classmethod
    def time(cls) -> float:
        return cls.now

    @classmethod
    def strftime(cls, fmt, tt=None):
        return _realtime.strftime(fmt, tt or _realtime.localtime(cls.now))

    @classmethod
    def localtime(cls, tt=None):
        return _realtime.localtime(tt or cls.now)


@pytest.fixture
def fake_clock(monkeypatch):
    _FakeTime.now = 10_000.0
    monkeypatch.setattr(T, "time", _FakeTime)
    return _FakeTime


@pytest.fixture
def recorder(monkeypatch):
    seen = []

    def fake_log(conn, ts, emp, state, dur, source=""):
        seen.append((ts, emp, state, dur, source))

    monkeypatch.setattr(T, "log_interval", fake_log)
    return seen


@pytest.fixture
def instant_commit(monkeypatch):
    monkeypatch.setattr(T, "SMOOTHING_BUFFER_SEC", 0.0)
    monkeypatch.setattr(T, "IDENTITY_STABILITY_FRAMES", 1)
    monkeypatch.setattr(T, "AWAY_AFTER_SEC", 3.0)
    monkeypatch.setattr(T, "PHONE_AFTER_SEC", 5.0)
    monkeypatch.setattr(T, "PHONE_GAP_GRACE_SEC", 2.0)


# ======================================================================
# Detection-dict builder with Phase 44 appearance fields
# ======================================================================

def _person_box(box, cam="cam1", phone=False,
                reid_emp=None, reid_score=0.0, reid_margin=0.0,
                reid_strong=False):
    det = {"cam": cam, "det_type": "person", "emp_id": "__person__",
           "person_box": tuple(box), "person_conf": 0.9,
           "frame_w": 640, "frame_h": 480, "phone": phone}
    if reid_emp is not None:
        det["reid_emp"] = reid_emp
        det["reid_score"] = reid_score
        det["reid_margin"] = reid_margin
        det["reid_strong"] = bool(reid_strong)
    return det


def _cycle(spatial, tracker, dets, *, camera_online=True):
    """One daemon-loop iteration: spatial -> claims -> merge -> FSM."""
    dets = list(dets)
    spatial.process(dets)
    claims = spatial.claims()
    if claims:
        dets = m._merge_person_claims(dets, claims)
    tracker.process_batch(dets, camera_online=camera_online)


# ======================================================================
# PhoneDetector -- model selection & normalised output
# ======================================================================

class _PConf:
    def __init__(self, v):
        self._v = float(v)

    def item(self):
        return self._v


class _PBox:
    def __init__(self, xyxy, conf):
        self.xyxy = np.array([xyxy], dtype=np.float64)
        self.conf = _PConf(conf)


class _PResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _PModel:
    def __init__(self, boxes, half=False):
        self.boxes = boxes
        self.last_kwargs = None

    def predict(self, frame, **kwargs):
        self.last_kwargs = kwargs
        return [_PResult([_PBox(b, c) for (b, c) in self.boxes])]


def test_normalize_box_coerces_to_int_tuple():
    box = _PBox((12.7, 33.2, 99.5, 210.9), 0.9)
    assert _normalize_box(box) == (13, 33, 100, 211)
    assert all(isinstance(v, int) for v in _normalize_box(box))


def test_phone_detector_no_model_ready_false():
    pd = PhoneDetector()
    assert pd.ready is False
    assert pd.source == "none"
    assert pd.detect(np.zeros((16, 16, 3), dtype=np.uint8)) == []
    st = pd.stats()
    assert st["calls"] == 0 and st["boxes_seen"] == 0
    assert st["conf_mean"] is None and st["conf_min"] is None


def test_phone_detector_reuses_base_model_when_no_dedicated_path():
    model = _PModel([])
    pd = PhoneDetector(model=model, model_path="")
    assert pd.ready is True
    assert pd.source == "base-model"


def test_phone_detector_missing_path_falls_back(monkeypatch):
    model = _PModel([])
    pd = PhoneDetector(model=model, model_path=r"C:\missing\phone_model.pt")
    assert pd.ready is True                      # never silently disabled
    assert pd.source == "base-model"
    pd_none = PhoneDetector(model=None, model_path=r"C:\missing\phone_model.pt")
    assert pd_none.ready is False
    assert pd_none.source == "none"

    fake_mod = types.ModuleType("ultralytics")
    class _RaisingYolo:
        def __init__(self, *_a, **_k):
            raise RuntimeError("cannot load on this host")

    fake_mod.YOLO = _RaisingYolo
    monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)
    pd_fail = PhoneDetector(model=model, model_path=r"C:\missing\phone_model.pt")
    assert pd_fail.ready is True                  # failed load -> base model
    assert pd_fail.source == "base-model"


def test_phone_detector_dedicated_model_path(monkeypatch, tmp_path):
    fake_mod = types.ModuleType("ultralytics")

    class _FakeYolo:
        def __init__(self, path):
            self.path = path
            self.predict = _PModel([]).predict

    fake_mod.YOLO = _FakeYolo
    monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)
    p = tmp_path / "phone_finetuned.pt"
    p.write_bytes(b"x")
    pd = PhoneDetector(model_path=str(p))
    assert pd.ready is True
    assert pd.source == str(p)


def test_phone_detector_output_and_stats():
    boxes = [((10.4, 20.7, 45.2, 80.1), 0.93), ((100.0, 100.0, 130.0, 150.0), 0.61)]
    model = _PModel(boxes)
    pd = PhoneDetector(model=model, phone_class=67, conf=0.4, imgsz=1280,
                       device="cpu")
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    out = pd.detect(frame)
    assert len(out) == 2
    assert out[0] == ((10, 21, 45, 80), 0.93)
    assert out[1] == ((100, 100, 130, 150), 0.61)
    pd.detect(frame)
    st = pd.stats()
    assert st["calls"] == 2
    assert st["boxes_seen"] == 4
    assert st["conf_mean"] == pytest.approx(0.77, abs=0.001)
    assert st["conf_min"] == pytest.approx(0.61, abs=0.001)
    assert st["phone_class"] == 67
    assert st["imgsz"] == 1280


def test_phone_detector_predict_kwargs():
    model = _PModel([])
    pd = PhoneDetector(model=model, phone_class=0, conf=0.35, imgsz=1280,
                       device="cuda", half=True)
    pd.detect(np.zeros((16, 16, 3), dtype=np.uint8))
    assert model.last_kwargs["conf"] == 0.35
    assert model.last_kwargs["classes"] == [0]
    assert model.last_kwargs["imgsz"] == 1280
    assert model.last_kwargs["device"] == "cuda"
    assert model.last_kwargs["verbose"] is False
    assert model.last_kwargs["half"] is True
    # half=False (CPU default) does NOT request half -> no deprecation noise.
    model2 = _PModel([])
    pd2 = PhoneDetector(model=model2)
    pd2.detect(np.zeros((16, 16, 3), dtype=np.uint8))
    assert "half" not in model2.last_kwargs


# ======================================================================
# AppearanceExtractor -- built-in descriptor
# ======================================================================

def _crop(color, shape=(600, 400, 3)):
    img = np.zeros(shape, dtype=np.uint8)
    img[..., 0] = color[0]
    img[..., 1] = color[1]
    img[..., 2] = color[2]
    return img


def test_builtin_extractor_embed_is_normalized_deterministic():
    ex = AppearanceExtractor(imgsz=128, dim=256)
    emb = ex.embed(_crop((40, 120, 200)))
    assert emb is not None
    assert emb.shape == (256,)
    assert np.all(np.isfinite(emb))
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)
    emb2 = ex.embed(_crop((40, 120, 200)))
    assert np.array_equal(emb, emb2)


def test_builtin_extractor_empty_crop_returns_none():
    ex = AppearanceExtractor(imgsz=128, dim=256)
    assert ex.embed(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    assert ex.embed(None) is None


def test_builtin_extractor_descriptor_dim_padding():
    ex = AppearanceExtractor(imgsz=64, dim=256)
    emb = ex.embed(_crop((255, 0, 0)))
    assert emb is not None and emb.size == 256


def test_builtin_extractor_cross_color_dissimilarity():
    ex = AppearanceExtractor(imgsz=128, dim=256)
    red = ex.embed(_crop((255, 0, 0)))
    blue = ex.embed(_crop((0, 0, 255)))
    same = ex.embed(_crop((255, 0, 0)))
    assert float(np.dot(red, same)) > 0.99
    assert float(np.dot(red, blue)) < 0.95


def test_extractor_invalid_onnx_falls_back(monkeypatch, tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_text("this is not an onnx model")
    ex = AppearanceExtractor(model_path=str(bad), imgsz=64, dim=256)
    assert ex.model_source == "built-in"
    assert ex.embed(_crop((10, 20, 30))) is not None


# ======================================================================
# AppearanceRegistry -- enrolment, cache, identity queries
# ======================================================================

def _write_images(directory, specs):
    """``specs`` = {name: color}; images written as PNG (deterministic)."""
    import cv2 as _cv2
    d = directory
    d.mkdir(parents=True, exist_ok=True)
    for name, color in specs.items():
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        img[..., 0] = color[0]
        img[..., 1] = color[1]
        img[..., 2] = color[2]
        _cv2.imwrite(str(d / f"{name}.png"), img)


def _registry(tmp_path, *, reid_specs, faces_specs=None, threshold=0.5,
              margin=0.01, dim=256):
    """Build a fresh AppearanceRegistry driven by explicit attributes and
    temp directories (never the production data/appearance.pkl)."""
    reid_dir = tmp_path / "reid"
    faces_dir = tmp_path / "faces"
    _write_images(reid_dir, reid_specs)
    if faces_specs:
        _write_images(faces_dir, faces_specs)
    reg = object.__new__(AppearanceRegistry)
    reg.reid_dir = str(reid_dir)
    reg.fallback_dir = str(faces_dir)
    reg.embeddings_file = str(tmp_path / "appearance.pkl")
    reg.embed_dim = dim
    reg.strong_threshold = threshold
    reg.margin_min = margin
    reg.extractor = AppearanceExtractor(imgsz=64, dim=dim)
    reg._by_employee = {}
    reg._loaded_from = None
    return reg


def test_employee_id_from_name_mapping():
    assert _employee_id_from_name("EMP001") == "EMP001"
    assert _employee_id_from_name("EMP001_2") == "EMP001"
    assert _employee_id_from_name("EMP002_back") == "EMP002"
    assert _employee_id_from_name("screenshot") is None
    assert _employee_id_from_name("") is None


def test_registry_rebuild_counts_and_gallery(tmp_path):
    reg = _registry(tmp_path,
                    reid_specs={"EMP001": (255, 0, 0), "EMP001_2": (150, 0, 0),
                                "EMP002": (0, 0, 255)})
    result = reg.rebuild()
    assert result["status"] == "rebuilt"
    assert result["loaded_from"] == str(tmp_path / "reid")
    assert reg.count_enrolled() == 2
    assert reg.employee_status() == {"EMP001": "ENROLLED",
                                     "EMP002": "ENROLLED"}
    gallery = reg.gallery()
    assert set(gallery) == {"EMP001", "EMP002"}
    assert gallery["EMP001"].shape == (256,)
    assert gallery["EMP001"].shape == gallery["EMP002"].shape


def test_registry_falls_back_to_faces_dir(tmp_path):
    reg = _registry(tmp_path, reid_specs={},
                    faces_specs={"EMP007": (200, 200, 0)})
    reg.rebuild()
    assert reg.count_enrolled() == 1
    assert reg.employee_status() == {"EMP007": "ENROLLED"}


def test_registry_cache_roundtrip(tmp_path):
    reg1 = _registry(tmp_path, reid_specs={"EMP001": (255, 0, 0)})
    reg1.rebuild()
    assert (tmp_path / "appearance.pkl").exists()

    reg2 = _registry(tmp_path, reid_specs={"EMP001": (255, 0, 0)})
    assert reg2.ensure_built()["loaded_from"] == "cache"
    assert reg2.count_enrolled() == 1
    assert reg2.employee_status() == {"EMP001": "ENROLLED"}


def test_registry_stale_cache_rebuilds_from_images(tmp_path):
    import pickle
    reg1 = _registry(tmp_path, reid_specs={"EMP001": (255, 0, 0)})
    reg1.rebuild()
    # Corrupt/stale the on-disk cache.
    with open(tmp_path / "appearance.pkl", "wb") as fh:
        pickle.dump({"version": CACHE_VERSION + 99, "by_employee": {}},
                    fh, protocol=4)
    reg2 = _registry(tmp_path, reid_specs={"EMP001": (255, 0, 0)})
    assert reg2.ensure_built()["status"] == "rebuilt"
    assert reg2.count_enrolled() == 1


def test_registry_identify_strong_self_match(tmp_path):
    reg = _registry(tmp_path, reid_specs={"EMP001": (255, 0, 0),
                                          "EMP002": (0, 0, 255)})
    reg.rebuild()
    emb = reg.extractor.embed(_solid_crop((255, 0, 0)))
    emp, score, margin, strong, second = reg.identify(emb)
    assert emp == "EMP001"
    assert strong is True
    assert score > 0.9
    assert margin >= 0.01
    assert second in ("EMP002", None)


def _solid_crop(color):
    return np.clip(np.stack([np.full((96, 96), color[c], dtype=np.uint8)
                             for c in range(3)], axis=-1), 0, 255)


def test_registry_identify_empty_gallery(tmp_path):
    reg = _registry(tmp_path, reid_specs={})
    assert reg.identify(None) == ["Unknown", 0.0, 0.0, False, None]
    assert reg.identify(np.zeros(256, dtype=np.float32)) == \
        ["Unknown", 0.0, 0.0, False, None]


def test_registry_identify_weak_below_threshold_reports_unknown(tmp_path):
    reg = _registry(tmp_path, reid_specs={"EMP001": (255, 0, 0),
                                          "EMP002": (0, 0, 255)})
    reg.rebuild()
    reg.strong_threshold = 1.5                     # unreachable bar (score<=1)
    emb = reg.extractor.embed(_solid_crop((255, 0, 0)))
    emp, score, margin, strong, second = reg.identify(emb)
    assert strong is False
    assert emp == "EMP001"                           # best candidate named
    assert score < 1.5


def test_registry_identify_margin_guard_rejects_identical_enrollees(tmp_path):
    # The SAME appearance under two IDs -> zero margin -> not strong.
    reg = _registry(tmp_path, reid_specs={"EMP001": (255, 0, 0),
                                          "EMP002": (255, 0, 0)})
    reg.rebuild()
    emb = reg.extractor.embed(_solid_crop((255, 0, 0)))
    emp, score, margin, strong, second = reg.identify(emb)
    assert strong is False
    assert margin == pytest.approx(0.0, abs=1e-6)
    assert second == "EMP002"


def test_registry_cache_never_stores_raw_images(tmp_path):
    import pickle
    reg = _registry(tmp_path, reid_specs={"EMP001": (255, 0, 0)})
    reg.rebuild()
    with open(reg.embeddings_file, "rb") as fh:
        data = pickle.load(fh)
    assert data["version"] == CACHE_VERSION
    assert set(data["by_employee"]) == {"EMP001"}
    assert "images" in data["by_employee"]["EMP001"]       # count yes
    assert data["by_employee"]["EMP001"]["embedding"].shape == (256,)


# ======================================================================
# Tracker -- phone evidence gate (per-track hysteresis)
# ======================================================================

def _make_track(now=0.0, *, adopt=2, switch=2, app_adopt=2, window=0, min_=0):
    return Track("t1", "cam1", (0, 0, 100, 200), 0.9, 640, 480, now,
                 adopt_frames=adopt, switch_frames=switch,
                 appear_adopt_frames=app_adopt,
                 phone_window=window, phone_min=min_)


def test_phone_evidence_gate_min2_rejects_single_false_positive():
    t = _make_track(window=5, min_=2)
    t.observe(True, 1.0)
    assert t.phone is False and t.phone_since is None
    t.observe(False, 2.0)                          # a clean frame in between
    assert t.phone is False
    t.observe(True, 3.0)                           # second detection in window
    assert t.phone is True
    assert t.phone_since == 3.0


def test_phone_evidence_window_opt_out_is_immediate():
    t = _make_track(window=0, min_=0)              # legacy semantics
    t.observe(True, 5.0)
    assert t.phone is True
    assert t.phone_since == 5.0


def test_phone_evidence_old_detections_expire_out_of_window():
    t = _make_track(window=3, min_=2)
    t.observe(True, 1.0)
    t.observe(True, 2.0)                           # reported -> phone_since=2
    assert t.phone is True
    for n in range(3, 7):                          # 3 clean frames push old out
        t.observe(False, float(n))
    assert t.phone is False


# ======================================================================
# Tracker -- appearance identity policy (secondary, conservative)
# ======================================================================

def test_appearance_adopts_after_repeat_strong_votes():
    t = _make_track(app_adopt=2)
    t.update((0, 0, 100, 200), 0.9, False, 1.0,
             reid_emp="EMP001", reid_score=0.95, reid_margin=0.2,
             reid_strong=True)
    assert t.identity in (None, UNKNOWN_ID)        # not yet adopted
    t.update((0, 0, 100, 200), 0.9, False, 2.0,
             reid_emp="EMP001", reid_score=0.95, reid_margin=0.2,
             reid_strong=True)
    assert t.identity == "EMP001"
    assert t.identity_source == "appearance"


def test_appearance_needs_consecutive_strong_votes():
    t = _make_track(app_adopt=3)
    for n in range(2):
        t.update((0, 0, 100, 200), 0.9, False, float(n) + 1.0,
                 reid_emp="EMP001", reid_score=0.95, reid_margin=0.2,
                 reid_strong=True)
    assert t.identity in (None, UNKNOWN_ID)
    t.update((0, 0, 100, 200), 0.9, False, 3.0,
             reid_emp="EMP001", reid_score=0.95, reid_margin=0.2,
             reid_strong=True)
    assert t.identity == "EMP001"
    assert t.identity_source == "appearance"


def test_weak_appearance_vote_carries_no_identity_weight():
    t = _make_track(app_adopt=2)
    for n in range(5):
        t.update((0, 0, 100, 200), 0.9, False, float(n) + 1.0,
                 reid_emp="EMP001", reid_score=0.5, reid_margin=0.0,
                 reid_strong=False)
    assert t.identity is None                      # no vote ever registered
    assert t.identity_source is None


def test_face_identity_outranks_appearance_and_cannot_be_demoted():
    t = _make_track(adopt=2, switch=2, app_adopt=2)
    for n in range(2):
        t.register_vote("EMP001", 0.9, float(n) + 1.0)
    assert t.identity == "EMP001"
    assert t.identity_source == "face"
    # A strong appearance vote for a DIFFERENT employee must not override the
    # face-sourced identity, even with many appearances.
    for n in range(6):
        t.register_appearance_vote("EMP002", 0.98, 100.0 + n)
    assert t.identity == "EMP001"
    assert t.identity_source == "face"


def test_appearance_switches_between_two_appearance_identities():
    t = _make_track(switch=2, app_adopt=2)
    t.register_appearance_vote("EMP001", 0.95, 1.0)
    t.register_appearance_vote("EMP001", 0.95, 2.0)
    assert t.identity == "EMP001"
    assert t.identity_source == "appearance"
    t.register_appearance_vote("EMP002", 0.97, 3.0)      # EMP001 run resets
    assert t.identity == "EMP001"                         # not yet enough votes
    t.register_appearance_vote("EMP002", 0.97, 4.0)
    assert t.identity == "EMP002"
    assert t.identity_source == "appearance"


def test_appearance_refresh_keeps_same_identity():
    t = _make_track(app_adopt=2)
    for n in range(2):
        t.register_appearance_vote("EMP001", 0.95, float(n) + 1.0)
    assert t.identity == "EMP001"
    t.register_appearance_vote("EMP001", 0.93, 10.0)
    assert t.identity == "EMP001"
    assert t.identity_updated == 10.0


# ======================================================================
# SpatialTracker + bridge -- appearance identifies a back-pose employee
# ======================================================================

def test_appearance_identifies_side_pose_and_keeps_employee_active(
        fake_clock, recorder, instant_commit):
    """A back/side pose with strong appearance votes is claimed and held
    ACTIVE by the Phase 43 person-bridge -- no face needed."""
    st = SpatialTracker(adopt_frames=2, switch_frames=2,
                        appear_adopt_frames=2, phone_window=0, phone_min=0)
    mt = MultiTracker(None)
    pbox = (100, 40, 300, 460)
    for _ in range(4):
        _FakeTime.now += 1.0
        _cycle(st, mt, [_person_box(pbox, reid_emp="EMP001", reid_score=0.96,
                                    reid_margin=0.2, reid_strong=True)])
    assert st.claims() == {"EMP001": {"cam": "cam1", "phone": False}}
    assert mt.live_state("EMP001") == "ACTIVE"
    snap = st.snapshot(_FakeTime.now)
    assert any(t["identity"] == "EMP001" and t["identity_source"] == "appearance"
               for t in snap["tracks"])


def test_weak_appearance_never_adopts_or_claims(fake_clock, recorder,
                                                instant_commit):
    st = SpatialTracker(adopt_frames=2, switch_frames=2,
                        appear_adopt_frames=2, phone_window=0, phone_min=0)
    mt = MultiTracker(None)
    for _ in range(5):
        _FakeTime.now += 1.0
        _cycle(st, mt, [_person_box((100, 40, 300, 460), reid_emp="EMP001",
                                    reid_score=0.5, reid_margin=0.0,
                                    reid_strong=False)])
    assert st.claims() == {}
    assert mt.live_states() == {}


def test_appearance_identity_not_adopted_into_employee_until_strong(
        fake_clock, recorder, instant_commit):
    st = SpatialTracker(adopt_frames=1, switch_frames=1,
                        appear_adopt_frames=3, phone_window=0, phone_min=0)
    mt = MultiTracker(None)
    pbox = (100, 40, 300, 460)
    for _ in range(2):                              # below adopt bar
        _FakeTime.now += 1.0
        _cycle(st, mt, [_person_box(pbox, reid_emp="EMP001", reid_score=0.96,
                                    reid_margin=0.2, reid_strong=True)])
    assert st.claims() == {}
    _FakeTime.now += 1.0                            # third strong vote adopts
    _cycle(st, mt, [_person_box(pbox, reid_emp="EMP001", reid_score=0.96,
                                reid_margin=0.2, reid_strong=True)])
    assert st.claims() == {"EMP001": {"cam": "cam1", "phone": False}}


# ======================================================================
# Detector -- IoU merge + cadence-gated phone/ReID passes
# ======================================================================

class _BBox:
    def __init__(self, cls, xyxy):
        self.cls = np.array(cls)
        self.xyxy = np.array([xyxy], dtype=float)


class _BoxesResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _PersonModel:
    """Minimal ultralytics-style model: a person box (+ optional phone box)."""

    def __init__(self, phones=False):
        self._phones = phones

    def predict(self, frames, **kwargs):
        out = []
        for _ in (frames if isinstance(frames, list) else [frames]):
            boxes = [_BBox(0, (30, 30, 90, 90))]
            if self._phones:
                boxes.append(_BBox(67, (180, 180, 210, 210)))
            out.append(_BoxesResult(boxes))
        return out


class _CountingReg:
    def __init__(self):
        self.app = _CountingApp()
        self.identify = lambda emb: ("EMP001", 0.8)
        self.threshold = 0.6
        self.employee_name = lambda emp_id: emp_id


class _CountingApp:
    def __init__(self):
        self.calls = 0

    def get(self, frame):
        self.calls += 1
        return [_FakeFace()]


class _FakeFace:
    bbox = np.array([120, 60, 220, 160], dtype=float)
    normed_embedding = np.zeros(512, dtype=np.float32)


def test_iou_basic():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    ov = _iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.0 < ov < 1.0


def test_merge_phone_dets_dedupes_overlap_and_adds_new():
    base = [((10, 10, 40, 40), 0.6)]
    merged = _merge_phone_dets(base, [((15, 15, 35, 35), 0.95)])   # overlap
    assert len(merged) == 1
    assert merged[0][0] == (10, 10, 40, 40)                        # base kept
    merged = _merge_phone_dets(base, [((300, 300, 340, 340), 0.8)])
    assert len(merged) == 2                                        # brand-new added


class _FakePhonePass:
    """PhoneDetector stand-in: ready + deterministic detect() results."""

    def __init__(self, boxes, ready=True):
        self._boxes = boxes
        self._ready = ready
        self.calls = 0

    @property
    def ready(self):
        return self._ready

    def detect(self, frame):
        self.calls += 1
        return list(self._boxes)

    def stats(self):
        return {"calls": self.calls, "boxes_seen": len(self._boxes),
                "conf_mean": 0.9, "conf_min": 0.9, "source": "test",
                "phone_class": 67, "imgsz": 1280}


def _det44_stub(*, phone_pass=None, phone_cadence=3, reid_enabled=True,
                reid_cadence=1, extract_reid=None, model_phones=False):
    det = ActivityDetector.__new__(ActivityDetector)
    det.conf = 0.3
    det._device = "cpu"
    det._half = False
    det.model = _PersonModel(phones=model_phones)
    det._face_registry = _CountingReg()
    det._face_cadence = 1
    det._face_ticks = {}
    det._phone_diag_on = False
    det._diag_cycles = 0
    det._phone_stats = {}
    det._phone_detector = phone_pass
    det._phone_cadence = phone_cadence
    det._phone_ticks = {}
    det._reid_enabled = reid_enabled
    det._reid_registry = None
    det._reid_cadence = reid_cadence
    det._reid_ticks = {}
    det._extract_reid = extract_reid or (
        lambda frame, boxes, shape, gate_boxes=None: [
            {"emp": "Unknown", "score": 0.0, "margin": 0.0,
             "strong": False, "second": None} for _ in boxes])
    return det


def test_enhance_phone_dets_cadence_and_merge():
    det = _det44_stub(
        phone_pass=_FakePhonePass([((300, 300, 340, 340), 0.95)]),
        phone_cadence=2)
    base = [((10, 10, 40, 40), 0.6)]
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    for i, expected_len in enumerate([2, 1, 2, 1]):     # tick/skip/tick/skip
        out = det._enhance_phone_dets(frame, "cam1", base)
        assert len(out) == expected_len
    assert det._phone_detector.calls == 2                # 4 cycles / cadence 2


def test_enhance_phone_dets_skips_overlapping_duplicate():
    det = _det44_stub(
        phone_pass=_FakePhonePass([((15, 15, 35, 35), 0.95)]),
        phone_cadence=1)
    base = [((10, 10, 40, 40), 0.6)]
    out = det._enhance_phone_dets(np.zeros((16, 16, 3), dtype=np.uint8),
                                  "cam1", base)
    assert len(out) == 1                              # duplicate dropped
    assert out[0][0] == (10, 10, 40, 40)


def test_enhance_phone_dets_unavailable_returns_base():
    det = _det44_stub(phone_pass=None)
    base = [((10, 10, 40, 40), 0.6)]
    assert det._enhance_phone_dets(np.zeros((16, 16, 3), dtype=np.uint8),
                                   "cam1", base) == base


def test_reid_cadence_gating():
    det = _det44_stub(reid_cadence=2,
                      extract_reid=lambda frame, boxes, shape, gate_boxes=None: [
                          {"emp": "EMP001", "score": 0.96, "margin": 0.2,
                           "strong": True, "second": None} for _ in boxes])
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    person_dets = [((10, 10, 90, 90), 0.8)]
    assert det._reid_for_camera(frame, "cam1", person_dets) is not None  # tick
    assert det._reid_for_camera(frame, "cam1", person_dets) is None      # skip
    assert det._reid_for_camera(frame, "cam1", person_dets) is not None  # tick


def test_reid_disabled_returns_none():
    det = _det44_stub(reid_enabled=False)
    person_dets = [((10, 10, 90, 90), 0.8)]
    assert det._reid_for_camera(np.zeros((16, 16, 3), dtype=np.uint8),
                                "cam1", person_dets) is None


def test_detect_batch_wires_enhanced_phone_and_reid_into_output():
    """The full detect_batch path: the phone pass + appearance pass feed the
    per-person entries the spatial tracker consumes."""
    det = _det44_stub(
        phone_pass=_FakePhonePass([((70, 60, 110, 90), 0.87)]),   # near person
        phone_cadence=1,
        extract_reid=lambda frame, boxes, shape, gate_boxes=None: [
            {"emp": "EMP002", "score": 0.97, "margin": 0.3,
             "strong": True, "second": None} for _ in boxes])
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    out = det.detect_batch({"cam1": frame})
    assert det._phone_detector.calls == 1
    people = [d for d in out if d.get("det_type") == "person"]
    assert people, "person entries must be emitted"
    assert people[0]["reid_emp"] == "EMP002"
    assert people[0]["reid_strong"] is True
    assert people[0]["phone"] is True          # enhances person-level phone


def test_phase44_diagnostics_payload():
    det = _det44_stub(
        phone_pass=_FakePhonePass([((70, 60, 110, 90), 0.87)]),
        phone_cadence=3, reid_cadence=2)
    det._reid_registry = type("R", (), {
        "count_enrolled": lambda self: 3,
        "extractor": type("E", (), {"model_source": "built-in"})(),
    })()
    diag = det.phase44_diagnostics()
    assert diag["phone_pass"]["cadence"] == 3
    assert diag["phone_pass"]["calls"] == 0
    assert diag["phone_pass"]["source"] == "test"
    assert diag["reid"]["enabled"] is True
    assert diag["reid"]["cadence"] == 2
    assert diag["reid"]["enrolled"] == 3
    assert diag["reid"]["extractor"] == "built-in"


def test_phase44_diagnostics_defensive_without_phase44_attrs():
    det = ActivityDetector.__new__(ActivityDetector)
    det._phone_cadence = 3
    det._reid_cadence = 2
    diag = det.phase44_diagnostics()          # must not raise
    assert diag["phone_pass"] is None
    assert diag["reid"]["enabled"] is False
    assert diag["reid"]["cadence"] == 2


# ======================================================================
# Config validation -- Phase 44 settings
# ======================================================================

def _problems(monkeypatch, **overrides):
    for key, value in overrides.items():
        monkeypatch.setattr(config, key, value)
    return config.validate_config(quiet=True)


def test_config_rejects_bad_phase44_values(monkeypatch):
    assert any("CCTV_PHONE_IMGSZ" in p
               for p in _problems(monkeypatch, PHONE_IMGSZ=16))
    assert any("CCTV_PHONE_DETECT_CADENCE" in p
               for p in _problems(monkeypatch, PHONE_DETECT_CADENCE=0))
    assert any("CCTV_PHONE_EVIDENCE_MIN" in p
               for p in _problems(monkeypatch, PHONE_EVIDENCE_MIN=5,
                                  PHONE_EVIDENCE_WINDOW=3))
    assert any("CCTV_REID_CADENCE" in p
               for p in _problems(monkeypatch, REID_CADENCE=0))
    assert any("CCTV_REID_IMGSZ" in p
               for p in _problems(monkeypatch, REID_IMGSZ=8))
    assert any("CCTV_REID_STRONG_THRESHOLD" in p
               for p in _problems(monkeypatch, REID_STRONG_THRESHOLD=1.5))
    assert any("CCTV_REID_MARGIN_MIN" in p
               for p in _problems(monkeypatch, REID_MARGIN_MIN=-0.1))
    assert any("CCTV_REID_ADOPT" in p
               for p in _problems(monkeypatch, REID_ADOPT_FRAMES=0))


def test_config_accepts_good_phase44_values(monkeypatch):
    problems = _problems(
        monkeypatch, PHONE_IMGSZ=1280, PHONE_DETECT_CADENCE=3,
        PHONE_EVIDENCE_WINDOW=5, PHONE_EVIDENCE_MIN=2,
        REID_CADENCE=3, REID_IMGSZ=128, REID_STRONG_THRESHOLD=0.9,
        REID_MARGIN_MIN=0.06, REID_ADOPT_FRAMES=5)
    assert not any("PHONE" in p or "REID" in p for p in problems)