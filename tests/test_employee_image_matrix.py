"""Employee Add-Image / enrollment test matrix (20 SIMULATED scenarios).

Regressions for the current task's Part B: the R68A fix keeps the upload path
model-free and the scan in the background, but the reported "Add Image still
failing" symptom had to be reproduced and classified BEFORE trusting the fix.
The real-model reproduction (temp storage) passed locally; these 20 scenarios
pin the matrix that gates it:

  scenarios  1-13  upload -> Save fast path (bytes only, never the model):
                   canonical filenames, JPEG/PNG/BMP, size/pixel caps, magic
                   bytes, corrupt decode, reserved ids, path traversal;
  scenario   14    one employee, two uploads -> ONE embedding (near-duplicate
                   face dedup) -- the "single registry rule";
  scenario   15-16 background single-flight job state machine + user-safe
                   result surface;
  scenario   17-19 the model-facing scan decisions (ENROLLED / NO_FACE /
                   MULTIPLE_FACES) driven through the REAL registry build
                   path with a stubbed face detector (no heavyweight model in
                   pytest);
  scenario   20    the fast cached-status reader reflects the new enrollment.

All scenarios are SIMULATED -- images are generated in-process, and no real
InsightFace model runs inside pytest (the real-model run is the external
`repro_enrollment_full.py` harness).  Token thresholds / FSM semantics are
untouched.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import app
import src.face_registry as fr


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _make_img(path: Path, kind: str, size=(320, 320)) -> bytes:
    from PIL import Image
    if kind == "jpeg":
        Image.new("RGB", size, (120, 130, 140)).save(path, "JPEG")
    elif kind == "png":
        Image.new("RGB", size, (1, 2, 3)).save(path, "PNG")
    elif kind == "bmp":
        Image.new("RGB", size, (5, 5, 5)).save(path, "BMP")
    else:
        raise ValueError(kind)
    return path.read_bytes()


def _face(seed: int = 1):
    rng = np.random.default_rng(seed)
    f = type("F", (), {})()
    f.bbox = [0, 0, 160, 160]
    emb = rng.standard_normal(512, dtype=np.float32)
    f.normed_embedding = emb / np.linalg.norm(emb)
    return f


class _StubApp:
    """Stubbed InsightFace model: controllable return of 0/1/2 faces."""

    def __init__(self, result):
        self._result = result

    def get(self, img):
        return list(self._result)


class _PillowCV2:
    """Real image decode through PIL (local stand-in for lazy cv2 imread)."""

    @staticmethod
    def imread(path):
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))


class _StubReg:
    """Minimal shared-registry stand-in for background-job tests."""

    def __init__(self, status=None, enrolled=1, registered=1):
        self._status = dict(status or {"EMP001": fr.ENROLLED})
        self._enrolled = enrolled
        self._registered = registered
        self.rebuild_calls = 0

    def rebuild(self):
        self.rebuild_calls += 1

    def employee_status(self) -> dict:
        return dict(self._status)

    @property
    def num_enrolled_employees(self) -> int:
        return self._enrolled

    @property
    def num_registered(self) -> int:
        return self._registered


@pytest.fixture
def save_dir(tmp_path, monkeypatch):
    """Patch app.save_enrollment_image storage into tmp (no real data/)."""
    faces_dir = tmp_path / "faces"
    monkeypatch.setitem(app._faces_dir.__globals__, "FACES_DIR", str(faces_dir))
    return faces_dir


@pytest.fixture(autouse=True)
def clean_registry(tmp_path, monkeypatch):
    """Isolate process-global registry + job state between scenarios."""
    monkeypatch.setattr(fr, "_REGISTRY_GLOBAL", None)
    monkeypatch.setattr(fr, "FACES_DIR", tmp_path / "faces")
    monkeypatch.setattr(fr, "EMBEDDINGS_FILE", tmp_path / "embeddings.pkl")
    original = dict(fr.ENROLL_JOB)
    fr.ENROLL_JOB.update({
        "state": "idle", "start": 0.0, "end": 0.0, "elapsed": 0.0,
        "status": {}, "enrolled": 0, "embeddings": 0,
        "message": "", "error_type": "",
    })
    yield
    fr.ENROLL_JOB.clear()
    fr.ENROLL_JOB.update(original)


def _scanning_registry(monkeypatch, tmp_path_faces):
    """Real ``FaceRegistry`` build path with a stubbed detector + PIL decode."""
    reg = fr.FaceRegistry.__new__(fr.FaceRegistry)
    reg._app = _StubApp([_face(1)])                       # noqa: SLF001
    reg._init_insightface = lambda detect_size=640: None  # noqa: SLF001
    monkeypatch.setattr(fr, "_import_cv2", lambda: _PillowCV2)
    return reg


# ======================================================================
# 1-13. Upload -> Save fast path (no model)
# ======================================================================

def test_matrix_01_valid_jpeg_saved_under_canonical_name(save_dir, tmp_path):
    jpg = _make_img(tmp_path / "x.jpg", "jpeg")
    ok, msg = app.save_enrollment_image("EMP001", "randomname.jpg", jpg)
    assert ok and "Saved" in msg
    assert sorted(p.name for p in save_dir.iterdir()) == ["EMP001.jpg"]
    assert (save_dir / "EMP001.jpg").read_bytes() == jpg


def test_matrix_02_jpeg_extension_canonicalizes_to_dot_jpg(save_dir, tmp_path):
    jpg = _make_img(tmp_path / "x.jpeg", "jpeg")
    ok, _ = app.save_enrollment_image("EMP001", "photo.jpeg", jpg)
    assert ok
    assert sorted(p.suffix for p in save_dir.iterdir()) == [".jpg"]
    assert (save_dir / "EMP001.jpg").exists()


def test_matrix_03_png_saved_under_canonical_name(save_dir, tmp_path):
    png = _make_img(tmp_path / "x.png", "png")
    ok, _ = app.save_enrollment_image("EMP001", "face.png", png)
    assert ok
    assert (save_dir / "EMP001.png").read_bytes() == png


def test_matrix_04_bmp_accepted(save_dir, tmp_path):
    bmp = _make_img(tmp_path / "x.bmp", "bmp")
    ok, _ = app.save_enrollment_image("EMP001", "old.bmp", bmp)
    assert ok
    assert (save_dir / "EMP001.bmp").exists()


def test_matrix_05_empty_upload_rejected_fast(save_dir):
    ok, msg = app.save_enrollment_image("EMP001", "empty.jpg", b"")
    assert not ok
    assert "empty" in msg.lower()
    assert not save_dir.exists() or not list(save_dir.iterdir())


def test_matrix_06_oversize_image_rejected(save_dir):
    big = b"\xff\xd8\xff" + b"Z" * (12 * 1024 * 1024)   # > 10 MB cap
    ok, msg = app.save_enrollment_image("EMP001", "big.jpg", big)
    assert not ok
    assert "too large" in msg
    assert not save_dir.exists() or not list(save_dir.iterdir())


def test_matrix_07_oversize_pixels_rejected(save_dir, tmp_path):
    huge = _make_img(tmp_path / "huge.jpg", "jpeg", size=(4000, 6500))  # 26 MP
    ok, msg = app.save_enrollment_image("EMP001", "huge.jpg", huge)
    assert not ok
    assert "pixels" in msg
    assert not save_dir.exists() or not list(save_dir.iterdir())


def test_matrix_08_unsupported_type_rejected(save_dir, tmp_path):
    data = _make_img(tmp_path / "x.gif", "jpeg")
    ok, msg = app.save_enrollment_image("EMP001", "anim.gif", data)
    assert not ok
    assert "Unsupported image type" in msg
    assert not save_dir.exists() or not list(save_dir.iterdir())


def test_matrix_09_magic_bytes_mismatch_rejected(save_dir, tmp_path):
    png_bytes = _make_img(tmp_path / "x.png", "png")
    ok, msg = app.save_enrollment_image("EMP001", "lying.jpg", png_bytes)
    assert not ok
    assert "not a valid JPEG" in msg
    assert not save_dir.exists() or not list(save_dir.iterdir())


def test_matrix_10_corrupt_image_rejected_fast(save_dir):
    junk = b"\xff\xd8\xff" + b"\x00\x01\x02" * 40       # jpeg magic, junk payload
    ok, msg = app.save_enrollment_image("EMP001", "junk.jpg", junk)
    assert not ok
    assert "could not be decoded" in msg
    assert not save_dir.exists() or not list(save_dir.iterdir())


def test_matrix_11_reserved_unknown_id_rejected(save_dir, tmp_path):
    jpg = _make_img(tmp_path / "x.jpg", "jpeg")
    ok, msg = app.save_enrollment_image("unknown", "x.jpg", jpg)
    assert not ok
    assert "not a valid employee id" in msg
    assert not save_dir.exists() or not list(save_dir.iterdir())


def test_matrix_12_reserved_person_id_rejected(save_dir, tmp_path):
    jpg = _make_img(tmp_path / "x.jpg", "jpeg")
    ok, msg = app.save_enrollment_image("__person__", "x.jpg", jpg)
    assert not ok
    assert "not a valid employee id" in msg


def test_matrix_13_path_traversal_id_rejected(save_dir, tmp_path):
    jpg = _make_img(tmp_path / "x.jpg", "jpeg")
    ok, msg = app.save_enrollment_image("../../etc/passwd", "x.jpg", jpg)
    assert not ok
    assert not save_dir.exists() or not list(save_dir.iterdir())


# ======================================================================
# 14. One employee, two photos -> ONE embedding (single-registry dedup)
# ======================================================================

def test_matrix_14_two_uploads_same_employee_yield_single_embedding(monkeypatch, tmp_path):
    """Two identical photos of EMP001 de-duplicate to a single embedding row;
    the employee is enrolled once (never double-counted at registry level)."""
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    jpg = _make_img(tmp_path / "a.jpg", "jpeg")
    png = _make_img(tmp_path / "b.png", "png")
    (faces_dir / "EMP001.jpg").write_bytes(jpg)
    (faces_dir / "EMP001.png").write_bytes(png)
    reg = _scanning_registry(monkeypatch, faces_dir)
    reg._load_or_build()
    assert reg.employee_status().get("EMP001") == fr.ENROLLED
    assert reg.num_enrolled_employees == 1
    assert reg.num_registered == 1  # de-duplicated: one embedding row


# ======================================================================
# 15-16. Background single-flight enrollment job
# ======================================================================

def test_matrix_15_validate_is_single_flight(monkeypatch):
    monkeypatch.setattr(fr, "get_shared_registry", lambda: _StubReg())
    fr.ENROLL_JOB.update({"state": "running"})
    assert app._start_enroll_job() is False          # already running
    fr.ENROLL_JOB.update({"state": "idle"})
    assert app._start_enroll_job() is True           # a fresh job starts
    time.sleep(0.3)  # allow worker to flip state back to idle/done


def test_matrix_16_job_transitions_and_surfaces_result(monkeypatch):
    stub = _StubReg({"EMP001": fr.ENROLLED}, enrolled=1, registered=1)
    monkeypatch.setattr(fr, "get_shared_registry", lambda: stub)
    assert app._start_enroll_job() is True
    deadline = time.time() + 15
    while fr.ENROLL_JOB.get("state") not in ("done", "error") \
            and time.time() < deadline:
        time.sleep(0.05)
    assert fr.ENROLL_JOB["state"] == "done"
    assert fr.ENROLL_JOB["enrolled"] == 1
    assert fr.ENROLL_JOB["status"].get("EMP001") == fr.ENROLLED
    assert stub.rebuild_calls == 1


# ======================================================================
# 17-19. Model-facing scan decisions (real build path, stubbed detector)
# ======================================================================

def test_matrix_17_scan_one_face_becomes_enrolled(monkeypatch, tmp_path):
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    (faces_dir / "EMP001.jpg").write_bytes(_make_img(tmp_path / "a.jpg", "jpeg"))
    reg = _scanning_registry(monkeypatch, faces_dir)
    reg._load_or_build()
    assert reg.employee_status().get("EMP001") == fr.ENROLLED
    assert reg._file_status.get("EMP001") == fr.ENROLLED  # noqa: SLF001


def test_matrix_18_scan_blank_image_reports_no_face(monkeypatch, tmp_path):
    from PIL import Image
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), (240, 240, 240)).save(
        faces_dir / "EMP001.jpg", "JPEG")
    reg = fr.FaceRegistry.__new__(fr.FaceRegistry)
    reg._app = _StubApp([])                                  # noqa: SLF001
    reg._init_insightface = lambda detect_size=640: None     # noqa: SLF001
    monkeypatch.setattr(fr, "_import_cv2", lambda: _PillowCV2)
    reg._load_or_build()
    assert reg.employee_status().get("EMP001") == fr.NO_FACE
    assert reg.num_enrolled_employees == 0


def test_matrix_19_scan_multi_face_reports_multiple(monkeypatch, tmp_path):
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    (faces_dir / "EMP001.jpg").write_bytes(_make_img(tmp_path / "a.jpg", "jpeg"))
    reg = fr.FaceRegistry.__new__(fr.FaceRegistry)
    reg._app = _StubApp([_face(1), _face(2)])                # noqa: SLF001
    reg._init_insightface = lambda detect_size=640: None     # noqa: SLF001
    monkeypatch.setattr(fr, "_import_cv2", lambda: _PillowCV2)
    reg._load_or_build()
    assert reg.employee_status().get("EMP001") == fr.MULTIPLE_FACES


# ======================================================================
# 20. Fast cached status reflects a new enrollment
# ======================================================================

def test_matrix_20_cached_status_reflects_new_enrollment(tmp_path, monkeypatch):
    """Fast reader: reads the embeddings-cache pickle (no model) keyed by file
    stem, surfacing ENROLLED / NO_FACE for employees saved on disk."""
    import pickle

    import config
    cache = tmp_path / "embeddings.pkl"
    with open(cache, "wb") as f:
        pickle.dump({"file_status": {"EMP001": fr.ENROLLED,
                                     "EMP002": fr.NO_FACE}}, f)
    monkeypatch.setattr(config, "EMBEDDINGS_FILE", str(cache))
    status = app.cached_enrollment_status()
    assert status.get("EMP001") == fr.ENROLLED
    assert status.get("EMP002") == fr.NO_FACE