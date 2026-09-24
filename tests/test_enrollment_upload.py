"""Employee photo upload + enrollment regression tests.

Root-cause regression for the production issue "on Streamlit Cloud, uploading
an employee image shows 'Connection lost'".  The upload interaction itself is
lightweight (bytes only); the previous failure vector was the synchronous
InsightFace model load + full rescan inside the "Validate enrollment" button
run, which on a cold/first-use Cloud instance blocked the script thread for
tens of seconds (model download + ~300 MB onnx load) -- the long run /
"Connection lost".  These tests lock in:

  * upload validation WITHOUT the face model (JPG/PNG/BMP, magic bytes,
    decode, size + pixel caps, corrupt/invalid/mismatched types),
  * persistence + canonical employee-id filenames (never the upload name),
  * the background single-flight enrollment job and its user-safe results
    (ENROLLED / NO_FACE / MULTIPLE_FACES / INVALID_IMAGE / LOW_QUALITY,
    controlled errors with no traceback / secrets),
  * shared process-wide registry (init at most once),
  * the real dashboard flow (AppTest): admin can upload/save/validate,
    viewer cannot.
"""

import builtins
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import app
from src import database as db
from src.employees import EmployeeStore


_ROOT = Path(__file__).resolve().parent.parent
_APP = _ROOT / "app.py"
_AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def _make_img(path: Path, kind: str) -> bytes:
    from PIL import Image
    if kind == "jpeg":
        Image.new("RGB", (64, 48), (12, 34, 56)).save(path, "JPEG")
    elif kind == "png":
        Image.new("RGB", (30, 30), (1, 2, 3)).save(path, "PNG")
    elif kind == "bmp":
        Image.new("RGB", (20, 20), (5, 5, 5)).save(path, "BMP")
    else:
        raise ValueError(kind)
    return path.read_bytes()


@pytest.fixture
def faces(tmp_path, monkeypatch):
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(app._faces_dir.__globals__, "FACES_DIR", str(faces_dir))
    return faces_dir


@pytest.fixture(autouse=True)
def clean_job_state(monkeypatch):
    """Isolate the process-global enrollment job state between tests."""
    import src.face_registry as fr
    monkeypatch.setattr(fr, "_REGISTRY_GLOBAL", None)
    original = dict(fr.ENROLL_JOB)
    fr.ENROLL_JOB.update({
        "state": "idle", "start": 0.0, "end": 0.0, "elapsed": 0.0,
        "status": {}, "enrolled": 0, "embeddings": 0,
        "message": "", "error_type": "",
    })
    yield
    fr.ENROLL_JOB.clear()
    fr.ENROLL_JOB.update(original)


def _enable_auth(monkeypatch, admin="ph66-admin", viewer="ph66-viewer"):
    """Turn on dashboard auth in config AND the imported app module."""
    import config
    monkeypatch.setattr(config, "DASH_AUTH_ENABLED", True)
    monkeypatch.setattr(config, "DASH_ADMIN_PASS", admin)
    monkeypatch.setattr(config, "DASH_VIEWER_PASS", viewer)
    monkeypatch.setattr(config, "DASH_USERNAME", "admin")
    monkeypatch.setattr(config, "DASH_FAIL_CLOSED", True)
    monkeypatch.setattr(app, "DASH_AUTH_ENABLED", True)
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", admin)
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", viewer)
    monkeypatch.setattr(app, "DASH_USERNAME", "admin")
    monkeypatch.setattr(app, "DASH_FAIL_CLOSED", True)


def _apptest_env(monkeypatch, tmp_path):
    """Point config at tmp storage (crosses into the AppTest script run)."""
    import config as _cfg
    monkeypatch.setattr(_cfg, "FACES_DIR", str(tmp_path / "faces"))
    monkeypatch.setattr(_cfg, "EMBEDDINGS_FILE", str(tmp_path / "embeddings.pkl"))
    monkeypatch.setattr(_cfg, "DB_PATH", str(tmp_path / "probe.db"))
    monkeypatch.setenv("CCTV_LIVE_STATE_FILE", str(tmp_path / "missing.json"))
    conn = db.get_connection(str(tmp_path / "probe.db"))
    try:
        db.init_db(conn)
        EmployeeStore(conn, app.WORK_SCHEDULE).upsert(
            "EMP900", name="Probe User", department="QA",
            designation="Tester", active=True)
        conn.commit()
    finally:
        conn.close()


def _login(at, username: str, password: str):
    fields = {str(w.label): w for w in at.text_input}
    if "Username" in fields:
        fields["Username"].set_value(username)
    fields["Password"].set_value(password)
    [b for b in at.button if "Sign In" in str(b.label)][0].click()
    at.run()
    at.run()
    return at


def _nav_employees(at):
    nav = at.sidebar.radio[0]
    nav.set_value(next(o for o in nav.options if "Employees" in o))
    at.run()
    return at


# ----------------------------------------------------------------------
# Unit: upload validation + persistence (no face model)
# ----------------------------------------------------------------------
def test_jpeg_upload_saved_canonically(tmp_path, faces):
    jpg = _make_img(tmp_path / "src.jpg", "jpeg")
    ok, msg = app.save_enrollment_image("EMP001", "face.JPEG", jpg)
    assert ok
    target = faces / "EMP001.jpg"                     # canonical ext
    assert target.exists()
    assert "EMP001.jpg" in msg
    assert list(faces.iterdir()) == [target]


def test_png_upload_saved_as_png(tmp_path, faces):
    png = _make_img(tmp_path / "src.png", "png")
    ok, msg = app.save_enrollment_image("EMP002", "photo.png", png)
    assert ok
    assert (faces / "EMP002.png").exists()
    assert "EMP002.png" in msg


def test_bmp_upload_saved_as_bmp(tmp_path, faces):
    bmp = _make_img(tmp_path / "src.bmp", "bmp")
    ok, _ = app.save_enrollment_image("EMP003", "photo.bmp", bmp)
    assert ok
    assert (faces / "EMP003.bmp").exists()


def test_invalid_file_type_rejected(tmp_path, faces):
    jpg = _make_img(tmp_path / "src.jpg", "jpeg")
    ok, msg = app.save_enrollment_image("EMP001", "photo.exe", jpg)
    assert not ok and "Unsupported image type" in msg
    ok, msg = app.save_enrollment_image("EMP001", "script.exe", jpg)
    assert not ok
    assert list(faces.iterdir()) == []


def test_corrupt_image_rejected(tmp_path, faces):
    jpg = b"\xff\xd8\xff" + os.urandom(64)
    ok, msg = app.save_enrollment_image("EMP001", "broken.jpg", jpg)
    assert not ok and "could not be decoded" in msg
    png = b"\x89PNG\r\n\x1a\n" + os.urandom(64)
    ok, msg = app.save_enrollment_image("EMP001", "broken.png", png)
    assert not ok and "could not be decoded" in msg
    assert list(faces.iterdir()) == []


def test_empty_upload_rejected(faces):
    ok, msg = app.save_enrollment_image("EMP001", "empty.jpg", b"")
    assert not ok and "empty" in msg
    assert list(faces.iterdir()) == []


def test_magic_bytes_must_match_named_type(tmp_path, faces):
    jpg = _make_img(tmp_path / "src.jpg", "jpeg")
    ok, msg = app.save_enrollment_image("EMP001", "sneaky.png", jpg)
    assert not ok and "not a valid PNG" in msg
    assert list(faces.iterdir()) == []


def test_excessive_file_size_rejected(tmp_path, faces, monkeypatch):
    monkeypatch.setattr(app, "_ENROLL_IMAGE_MAX_BYTES", 100)
    jpg = _make_img(tmp_path / "src.jpg", "jpeg")
    ok, msg = app.save_enrollment_image("EMP001", "big.jpg", jpg)
    assert not ok and "too large" in msg
    assert list(faces.iterdir()) == []


def test_excessive_pixel_count_rejected(tmp_path, faces, monkeypatch):
    monkeypatch.setattr(app, "_ENROLL_IMAGE_MAX_PIXELS", 100)
    jpg = _make_img(tmp_path / "src.jpg", "jpeg")     # 64x48 = 3072 px
    ok, msg = app.save_enrollment_image("EMP001", "wide.jpg", jpg)
    assert not ok and "too many pixels" in msg
    assert list(faces.iterdir()) == []


def test_saved_bytes_persist_verbatim(tmp_path, faces):
    jpg = _make_img(tmp_path / "src.jpg", "jpeg")
    ok, _ = app.save_enrollment_image("EMP010", "photo.jpg", jpg)
    assert ok
    stored = (faces / "EMP010.jpg").read_bytes()
    assert stored == jpg
    import cv2
    import numpy as np
    arr = cv2.imdecode(np.frombuffer(stored, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None and arr.shape[:2] == (48, 64)


def test_stored_filename_never_uses_uploaded_name(tmp_path, faces):
    jpg = _make_img(tmp_path / "src.jpg", "jpeg")
    ok, _ = app.save_enrollment_image(
        "EMP042", "../../evil top secret.jpeg", jpg)
    assert ok
    names = sorted(p.name for p in faces.iterdir())
    assert names == ["EMP042.jpg"]
    assert not any("evil" in n or ".." in n for n in names)


def test_invalid_employee_id_still_rejected(tmp_path, faces):
    jpg = _make_img(tmp_path / "src.jpg", "jpeg")
    ok, msg = app.save_enrollment_image("unknown", "photo.jpg", jpg)
    assert not ok and "not a valid employee id" in msg
    assert list(faces.iterdir()) == []


def test_repeat_save_is_idempotent_single_canonical_file(tmp_path, faces):
    jpg = _make_img(tmp_path / "src.jpg", "jpeg")
    assert app.save_enrollment_image("EMP001", "a.jpeg", jpg)[0]
    assert app.save_enrollment_image("EMP001", "b.jpeg", jpg)[0]
    assert sorted(p.name for p in faces.iterdir()) == ["EMP001.jpg"]


# ----------------------------------------------------------------------
# Unit: background enrollment job statuses + single-flight + errors
# ----------------------------------------------------------------------
class _StubReg:
    def __init__(self, status=None, enrolled=0, emb=0):
        self._status = status or {}
        self._enrolled = enrolled
        self._emb = emb
        self.rebuilt = 0

    def rebuild(self):
        self.rebuilt += 1

    def employee_status(self):
        return self._status

    @property
    def num_enrolled_employees(self):
        return self._enrolled

    @property
    def num_registered(self):
        return self._emb


@pytest.mark.parametrize("status,enrolled,emb", [
    ("ENROLLED", 1, 1),
    ("NO_FACE", 0, 0),
    ("MULTIPLE_FACES", 1, 1),
    ("INVALID_IMAGE", 0, 0),
    ("LOW_QUALITY", 0, 0),
])
def test_enrollment_statuses_propagate(status, enrolled, emb, monkeypatch):
    import src.face_registry as fr
    monkeypatch.setattr(fr, "get_shared_registry",
                        lambda: _StubReg({"EMP001": status}, enrolled, emb))
    app._enroll_worker()
    assert fr.ENROLL_JOB["state"] == "done"
    assert fr.ENROLL_JOB["status"] == {"EMP001": status}
    assert fr.ENROLL_JOB["enrolled"] == enrolled
    assert fr.ENROLL_JOB["embeddings"] == emb
    assert fr.ENROLL_JOB["error_type"] == ""


def test_enrollment_failure_is_controlled(monkeypatch):
    import src.face_registry as fr

    def boom():
        raise RuntimeError("secret path /env/DASH_PASS=topsecret Traceback!")

    monkeypatch.setattr(fr, "get_shared_registry", boom)
    app._enroll_worker()
    assert fr.ENROLL_JOB["state"] == "error"
    assert fr.ENROLL_JOB["error_type"] == "RuntimeError"
    msg = fr.ENROLL_JOB["message"]
    for bad in ("Traceback", "topsecret", "DASH_PASS", "secret", "rtsp://",
                "CCTV_", "Traceback!"):
        assert bad not in msg
    assert "RuntimeError" in msg


def test_start_enroll_job_is_single_flight(monkeypatch):
    import src.face_registry as fr
    fr.ENROLL_JOB.update({"state": "running"})
    assert app._start_enroll_job() is False          # no second job/thread
    fr.ENROLL_JOB.update({"state": "idle"})
    monkeypatch.setattr(fr, "get_shared_registry",
                        lambda: _StubReg({"EMP001": "ENROLLED"}, 1, 1))
    assert app._start_enroll_job() is True           # single worker started
    for _ in range(200):
        if fr.ENROLL_JOB["state"] != "running":
            break
        time.sleep(0.01)
    assert fr.ENROLL_JOB["state"] == "done"


def test_shared_registry_initialised_at_most_once(monkeypatch):
    """Model init must not repeat per validate click / rerun."""
    import src.face_registry as fr
    calls = []

    class Counting:
        def __init__(self, detect_size=640):
            calls.append(detect_size)

    monkeypatch.setattr(fr, "_REGISTRY_GLOBAL", None)
    monkeypatch.setattr(fr, "FaceRegistry", Counting)
    fr.get_shared_registry()
    fr.get_shared_registry()
    assert len(calls) == 1


def test_reset_enroll_job_clears_stale_results(monkeypatch):
    import src.face_registry as fr
    fr.ENROLL_JOB.update({
        "state": "done", "status": {"EMP001": "ENROLLED"},
        "enrolled": 1, "embeddings": 1, "elapsed": 4.2,
    })
    app._reset_enroll_job()
    assert fr.ENROLL_JOB["state"] == "idle"
    assert fr.ENROLL_JOB["status"] == {}
    assert fr.ENROLL_JOB["elapsed"] == 0.0


# ----------------------------------------------------------------------
# AppTest: real dashboard flow (admin uploads / viewer cannot)
# ----------------------------------------------------------------------
def test_admin_upload_saves_canonical_file_end_to_end(monkeypatch, tmp_path):
    _enable_auth(monkeypatch)
    _apptest_env(monkeypatch, tmp_path)
    jpg = _make_img(tmp_path / "src.jpg", "jpeg")

    at = _AppTest.from_file(str(_APP), default_timeout=90).run()
    assert not at.exception, at.exception
    _login(at, "admin", "ph66-admin")
    _nav_employees(at)
    assert not at.exception, at.exception

    up = [u for u in at.file_uploader if "Enrollment photo" in str(u.label)]
    assert len(up) == 1
    up[0].set_value(("probe photo.JPEG", jpg, "image/jpeg"))
    at.run()

    [b for b in at.button if "Save photo" in str(b.label)][0].click()
    at.run()
    assert not at.exception, at.exception
    success = " ".join(str(s.value) for s in at.success)
    assert "Saved EMP900.jpg" in success
    target = tmp_path / "faces" / "EMP900.jpg"
    assert target.exists()
    assert target.read_bytes() == jpg


def test_viewer_cannot_upload_or_validate(monkeypatch, tmp_path):
    _enable_auth(monkeypatch)
    _apptest_env(monkeypatch, tmp_path)
    at = _AppTest.from_file(str(_APP), default_timeout=90).run()
    _login(at, "", "ph66-viewer")
    _nav_employees(at)
    assert not at.exception, at.exception
    assert not [b for b in at.button if "Save photo" in str(b.label)]
    assert not [b for b in at.button if "Validate enrollment" in str(b.label)]
    assert not [u for u in at.file_uploader if "Enrollment photo" in str(u.label)]


def test_admin_validate_runs_single_flight_and_surfaces_result(monkeypatch,
                                                              tmp_path):
    _enable_auth(monkeypatch)
    _apptest_env(monkeypatch, tmp_path)
    import src.face_registry as fr
    monkeypatch.setattr(fr, "get_shared_registry",
                        lambda: _StubReg({"EMP900": "ENROLLED"}, 1, 1))
    jpg = _make_img(tmp_path / "src.jpg", "jpeg")

    at = _AppTest.from_file(str(_APP), default_timeout=90).run()
    _login(at, "admin", "ph66-admin")
    _nav_employees(at)
    assert not at.exception, at.exception

    up = [u for u in at.file_uploader if "Enrollment photo" in str(u.label)][0]
    up.set_value(("emp900.jpg", jpg, "image/jpeg"))
    at.run()
    [b for b in at.button if "Save photo" in str(b.label)][0].click()
    at.run()
    assert (tmp_path / "faces" / "EMP900.jpg").exists()

    [b for b in at.button if "Validate enrollment" in str(b.label)][0].click()
    at.run()
    at.run()
    ok = False
    for _ in range(50):
        if at.exception:
            break
        if "Validation finished" in " ".join(str(s.value) for s in at.success):
            ok = True
            break
        time.sleep(0.05)
        at.run()
    assert ok, "background validate never surfaced its result"
    assert "1 employee(s) enrolled" in \
        " ".join(str(s.value) for s in at.success)


def test_upload_survives_rerun_and_replay_creates_no_duplicate(monkeypatch,
                                                              tmp_path):
    """Rerun/replay safety: the uploaded bytes survive reruns and re-saving
    the same employee keeps ONE canonical file (never a duplicate row/file).
    """
    _enable_auth(monkeypatch)
    _apptest_env(monkeypatch, tmp_path)
    jpg = _make_img(tmp_path / "src.jpg", "jpeg")

    at = _AppTest.from_file(str(_APP), default_timeout=90).run()
    _login(at, "admin", "ph66-admin")
    _nav_employees(at)
    assert not at.exception, at.exception

    up = [u for u in at.file_uploader if "Enrollment photo" in str(u.label)][0]
    up.set_value(("emp900.jpg", jpg, "image/jpeg"))
    at.run()
    [b for b in at.button if "Save photo" in str(b.label)][0].click()
    at.run()
    assert not at.exception, at.exception
    assert (tmp_path / "faces" / "EMP900.jpg").read_bytes() == jpg

    # A rerun (e.g. navigation / auto-refresh) must keep the file and content.
    at.run()
    assert not at.exception, at.exception
    up2 = [u for u in at.file_uploader if "Enrollment photo" in str(u.label)][0]
    assert up2.value is not None, "uploaded file must survive a rerun"

    # Re-saving the same employee must overwrite, never duplicate.
    [b for b in at.button if "Save photo" in str(b.label)][0].click()
    at.run()
    assert not at.exception, at.exception
    files = sorted(p.name for p in (tmp_path / "faces").iterdir())
    assert files == ["EMP900.jpg"]
    assert (tmp_path / "faces" / "EMP900.jpg").read_bytes() == jpg


def test_resave_blank_state_is_controlled_warning(monkeypatch, tmp_path):
    """Clicking Save with no photo is a controlled warning, not an error."""
    _enable_auth(monkeypatch)
    _apptest_env(monkeypatch, tmp_path)
    at = _AppTest.from_file(str(_APP), default_timeout=90).run()
    _login(at, "admin", "ph66-admin")
    _nav_employees(at)
    assert not at.exception, at.exception
    [b for b in at.button if "Save photo" in str(b.label)][0].click()
    at.run()
    assert not at.exception, at.exception
    assert any("Choose a photo to upload first" in str(w.value)
               for w in at.warning)
    faces_dir = tmp_path / "faces"
    assert not faces_dir.exists() or list(faces_dir.iterdir()) == []


def test_uploads_for_different_employees_do_not_collide(monkeypatch, tmp_path,
                                                        faces):
    """Each employee's photo maps to its OWN canonical file; a second upload
    to another employee must never overwrite the first employee's image.
    """
    _enable_auth(monkeypatch)
    _apptest_env(monkeypatch, tmp_path)

    conn = db.get_connection(str(tmp_path / "probe.db"))
    try:
        EmployeeStore(conn, app.WORK_SCHEDULE).upsert(
            "EMP901", name="Second", department="QA",
            designation="Tester", active=True)
    finally:
        conn.close()

    jpg_a = _make_img(tmp_path / "a.jpg", "jpeg")
    png_b = _make_img(tmp_path / "b.png", "png")
    assert app.save_enrollment_image("EMP900", "x.jpg", jpg_a)[0]
    assert app.save_enrollment_image("EMP901", "y.png", png_b)[0]
    assert sorted(p.name for p in faces.iterdir()) == ["EMP900.jpg", "EMP901.png"]
    assert (faces / "EMP900.jpg").read_bytes() == jpg_a
    assert (faces / "EMP901.png").read_bytes() == png_b


def test_default_smoke_faces_directory_is_untouched():
    """Sanity: the module still exposes the canonical faces dir helper."""
    assert app._faces_dir().name == "faces"


# ----------------------------------------------------------------------
# Cloud import boundary: face_registry must NOT import cv2 at module scope
# ----------------------------------------------------------------------
# On Streamlit Cloud `import cv2` raises (GUI build needs a missing libGL --
# Phase 63/65).  The enrollment-status fragment (app._enroll_job_snapshot)
# lazy-imports `src.face_registry` on EVERY 2s fragment rerun, so a
# module-level `import cv2` inside face_registry crashed that render path with
# an ImportError wrapped as FragmentHandledException (DIAG-27803).  These
# tests pin the boundary: module import + job snapshot need no cv2; only the
# image-decode functions obtain it, lazily, at execution time.

def test_face_registry_import_and_job_snapshot_without_cv2():
    """Fresh interpreter with cv2 UNAVAILABLE: importing src.face_registry and
    reading the enrollment job snapshot (the exact dashboard fragment path)
    succeeds with zero OpenCV involvement."""
    code = textwrap.dedent("""
        import sys
        sys.modules["cv2"] = None  # any `import cv2` now raises ImportError
        import src.face_registry as fr

        with fr.ENROLL_LOCK:
            snap = dict(fr.ENROLL_JOB)          # == app._enroll_job_snapshot()
        assert snap["state"] == "idle"
        assert snap["status"] == {}
        assert snap["message"] == ""
        # The sentinel we set (None) must still be there: the module import
        # never replaced it with a real cv2 module (which would also have
        # raised, since `import cv2` fails in this interpreter).
        assert sys.modules["cv2"] is None, "cv2 was imported by the module"

        # status/decision helpers are pure numpy -- no cv2 either
        assert fr.ENROLLED == "ENROLLED"
        assert fr.RECOG_CONFIRMED == "CONFIRMED"
        assert fr.FaceRegistry is not None
        print("SNAPSHOT_OK")
        """)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=_ROOT,
        capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "SNAPSHOT_OK" in proc.stdout


def test_lazy_cv2_import_raises_clear_error_when_unavailable(monkeypatch):
    """When cv2 genuinely cannot be imported, only the decode path fails --
    with an explicit ImportError, never a silent fake success."""
    import src.face_registry as fr

    monkeypatch.setattr(fr, "_CV2", None)
    real_import = builtins.__import__

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cv2" or name.startswith("cv2."):
            raise ImportError(
                "simulated cloud: OpenCV unavailable (GUI build needs libGL)")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(ImportError) as ei:
        fr._import_cv2()
    assert "cv2" in str(ei.value) or "OpenCV" in str(ei.value)


def test_lazy_cv2_import_returns_real_cv2_when_available(monkeypatch):
    """When cv2 IS available, functions that need it still obtain the real
    module at execution time (no functionality removed)."""
    import src.face_registry as fr

    monkeypatch.setattr(fr, "_CV2", None)
    m = fr._import_cv2()
    assert hasattr(m, "imread")
    assert hasattr(m, "IMREAD_COLOR")


def test_register_one_decodes_image_through_lazy_cv2(monkeypatch, tmp_path):
    """The only runtime cv2 consumer (image decode during registry build)
    reaches OpenCV via the lazy getter -- never at module scope."""
    import numpy as np
    import src.face_registry as fr

    class FakeCV2:
        calls = 0

        @classmethod
        def imread(cls, path):
            cls.calls += 1
            return np.full((200, 200, 3), 255, dtype=np.uint8)

    monkeypatch.setattr(fr, "_import_cv2", lambda: FakeCV2)

    class _App:
        def get(self, img):
            face = type("F", (), {"bbox": None, "normed_embedding": None})()
            face.bbox = [0, 0, 160, 160]
            emb = np.ones(512, dtype=np.float32)
            face.normed_embedding = emb / np.linalg.norm(emb)
            return [face]

    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    (faces_dir / "EMP001.jpg").write_bytes(b"unused")
    monkeypatch.setattr(fr, "FACES_DIR", faces_dir)
    monkeypatch.setattr(fr, "EMBEDDINGS_FILE", tmp_path / "embeddings.pkl")

    reg = fr.FaceRegistry.__new__(fr.FaceRegistry)
    reg._init_insightface = lambda detect_size=640: None
    reg._app = _App()
    reg._load_or_build()

    assert FakeCV2.calls == 1, "image decode must go through the lazy getter"
    assert reg.employee_status().get("EMP001") == fr.ENROLLED


def test_enrollment_status_fragment_renders_without_cv2(monkeypatch, tmp_path):
    """Cloud simulation: cv2 blocked BEFORE face_registry is imported, then the
    real dashboard flow (admin -> Employees -> Validate enrollment) renders the
    enrollment status fragment — no ImportError, no FragmentHandledException
    (DIAG-27803) — and the module never pulled in OpenCV."""
    _enable_auth(monkeypatch)
    _apptest_env(monkeypatch, tmp_path)

    # Simulated Cloud interpreter: cv2 cannot be imported here.
    real_import = builtins.__import__

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cv2" or name.startswith("cv2."):
            raise ImportError(
                "simulated cloud: OpenCV unavailable (GUI build needs libGL)")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    # The fix means this import must succeed with cv2 blocked.
    import src.face_registry as fr
    assert fr._CV2 is None, "cv2 must not be loaded at module import time"
    monkeypatch.setattr(fr, "get_shared_registry",
                        lambda: _StubReg({"EMP900": "ENROLLED"}, 1, 1))
    monkeypatch.setattr(fr, "_REGISTRY_GLOBAL", None)
    fr.ENROLL_JOB.update({
        "state": "idle", "start": 0.0, "end": 0.0, "elapsed": 0.0,
        "status": {}, "enrolled": 0, "embeddings": 0,
        "message": "", "error_type": "",
    })

    at = _AppTest.from_file(str(_APP), default_timeout=90).run()
    assert not at.exception, at.exception
    _login(at, "admin", "ph66-admin")
    _nav_employees(at)
    assert not at.exception, at.exception

    # Drive the exact failing interaction: Validate enrollment -> fragment
    # renders the job result.  No ImportError / FragmentHandledException.
    [b for b in at.button if "Validate enrollment" in str(b.label)][0].click()
    at.run()
    at.run()
    ok = False
    for _ in range(60):
        if at.exception:
            break
        if "Validation finished" in " ".join(str(s.value) for s in at.success):
            ok = True
            break
        time.sleep(0.05)
        at.run()
    assert ok, "background validate never surfaced its result"
    assert not at.exception, at.exception
    assert fr._CV2 is None, "dashboard never imported Optical cv2"