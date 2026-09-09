"""FaceRegistry per-file enrollment status (Phase 5 / dashboard).

Uses a fake insightface app and a temp faces dir so no model / network is
needed.  Exercises the CACHE_VERSION-3 status plumbing end-to-end.
"""

from pathlib import Path

import cv2
import numpy as np

import config
import src.face_registry as fr
from src.face_registry import FaceRegistry, CACHE_VERSION


class _Face:
    def __init__(self, bbox):
        self.bbox = bbox
        emb = np.random.rand(512).astype(np.float32)
        self.normed_embedding = emb / max(1e-9, np.linalg.norm(emb))


class _FakeApp:
    def __init__(self, factory):
        self.factory = factory  # callable(img) -> list[_Face]

    def get(self, img):
        return self.factory(img)


def _new_registry(monkeypatch, tmp_path, app):
    faces_dir = tmp_path / "faces"
    emb_file = tmp_path / "embeddings.pkl"
    monkeypatch.setattr(fr, "FACES_DIR", faces_dir)
    monkeypatch.setattr(fr, "EMBEDDINGS_FILE", emb_file)
    reg = FaceRegistry.__new__(FaceRegistry)
    reg.threshold = 0.6
    reg._app = app
    reg._employee_ids = []
    reg._embeddings = None
    reg._by_employee = {}
    reg._file_status = {}
    reg._load_or_build()
    return reg, emb_file


def _jpg(path, size=(200, 200)):
    img = np.full((size[0], size[1], 3), 255, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_status_end_to_end(monkeypatch, tmp_path):
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    # Distinct image sizes encode per-file behaviour for the fake app:
    # 100 -> one face, 110 -> two faces, 120 -> no face, 130 -> tiny face.
    _jpg(faces_dir / "EMP001.jpg", size=(100, 100))
    _jpg(faces_dir / "EMP002.jpg", size=(110, 110))
    _jpg(faces_dir / "EMP003.jpg", size=(120, 120))
    # EMP004: corrupt -> INVALID_IMAGE (cv2.imread returns None)
    (faces_dir / "EMP004.jpg").write_bytes(b"not-an-image")
    _jpg(faces_dir / "EMP005.jpg", size=(130, 130))

    def factory(img):
        if img.shape[0] == 100:
            return [_Face((10, 10, 150, 150))]
        if img.shape[0] == 110:
            return [_Face((10, 10, 150, 150)), _Face((30, 30, 200, 200))]
        if img.shape[0] == 120:
            return []
        if img.shape[0] == 130:
            # area = 50*50 = 2500 < MIN_FACE_AREA (120*120)
            return [_Face((20, 20, 70, 70))]
        return []  # unreachable for valid images

    reg, emb_file = _new_registry(monkeypatch, tmp_path, _FakeApp(factory))

    status = reg.scan_status()
    assert status["EMP001"] == "ENROLLED"
    assert status["EMP002"] == "MULTIPLE_FACES"
    assert status["EMP003"] == "NO_FACE"
    assert status["EMP004"] == "INVALID_IMAGE"
    assert status["EMP005"] == "LOW_QUALITY"

    assert set(reg.enrolled_employee_ids) == {"EMP001", "EMP002"}

    agg = reg.employee_status()
    assert agg["EMP001"] == "ENROLLED"
    assert agg["EMP002"] == "MULTIPLE_FACES"
    assert agg["EMP003"] == "NO_FACE"
    assert agg["EMP004"] == "INVALID_IMAGE"
    assert agg["EMP005"] == "LOW_QUALITY"

    # cache persisted with the status map
    assert emb_file.exists()
    import pickle
    cached = pickle.loads(emb_file.read_bytes())
    assert cached["version"] == CACHE_VERSION
    assert cached["file_status"]["EMP003"] == "NO_FACE"


def test_cache_round_trip_preserves_status(monkeypatch, tmp_path):
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    _jpg(faces_dir / "EMP001.jpg")

    app = _FakeApp(lambda img: [_Face((10, 10, 150, 150))])
    reg, emb_file = _new_registry(monkeypatch, tmp_path, app)

    # Real cache path: __init__ skips insightface init via monkeypatch
    monkeypatch.setattr(FaceRegistry, "_init_insightface", lambda self, detect_size=640: None)
    reg2 = FaceRegistry()
    assert reg2.scan_status() == {"EMP001": "ENROLLED"}
    assert reg2.employee_status() == {"EMP001": "ENROLLED"}


def test_empty_registry_status_is_empty(monkeypatch, tmp_path):
    faces_dir = tmp_path / "faces"
    reg, _ = _new_registry(monkeypatch, tmp_path, _FakeApp(lambda img: []))
    assert reg.scan_status() == {}
    assert reg.employee_status() == {}
    assert reg.enrolled_employee_ids == []


def test_confusability_report_flags_near_duplicate_identities(monkeypatch, tmp_path):
    """Two DIFFERENT employees whose embeddings are extremely close must be
    surfaced by confusability_report so the operator is never handed two
    face-indistinguishable identities."""
    np.random.seed(7)
    anchor = np.random.rand(512).astype(np.float32)
    anchor = anchor / np.linalg.norm(anchor)
    # EMP002 is a distinct embedding at cosine ~0.95 to EMP001: below the
    # near-duplicate dedup (DUPLICATE_COS=0.985) floor so it enrolls as its
    # own row, but still confusable with EMP001.
    orth = np.random.rand(512).astype(np.float32)
    orth -= (anchor @ orth) * anchor
    orth = orth / np.linalg.norm(orth)
    emp2 = 0.95 * anchor + np.sqrt(1 - 0.95 ** 2) * orth
    emp2 = emp2 / np.linalg.norm(emp2)

    monkeypatch.setattr(config, "FACE_CONFUSABILITY_MAX", 0.9)
    reg, _ = _rebuild_two_employees(monkeypatch, tmp_path, anchor, emp2)

    report = reg.confusability_report()
    assert report, "confusable pair must be flagged"
    pair = report[0]
    assert set([pair["employee_a"], pair["employee_b"]]) == {"EMP001", "EMP002"}
    assert pair["similarity"] >= 0.9


def _rebuild_two_employees(monkeypatch, tmp_path, emp1, emp2):
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    _jpg(faces_dir / "EMP001.jpg", size=(101, 101))
    _jpg(faces_dir / "EMP002.jpg", size=(102, 102))

    class _FakeFace:
        def __init__(self, emb):
            self.bbox = (10, 10, 150, 150)
            self.normed_embedding = emb

    def factory(img):
        return [_FakeFace(emp1 if img.shape[0] == 101 else emp2)]

    reg, _ = _new_registry(monkeypatch, tmp_path, _FakeApp(factory))
    return reg, _


def test_confusability_report_empty_for_single_employee(monkeypatch, tmp_path):
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    _jpg(faces_dir / "EMP001.jpg", size=(100, 100))

    class _FakeFace:
        def __init__(self, emb):
            self.bbox = (10, 10, 150, 150)
            self.normed_embedding = emb

    reg, _ = _new_registry(monkeypatch, tmp_path,
                           _FakeApp(lambda img: [_FakeFace(_Face((10, 10, 150, 150)).normed_embedding)]))
    assert reg.confusability_report() == []