"""Multi-person / multi-camera robustness regression suite.

Targets the "only one employee shows" + "Camera is not delivering frames"
reports with model-free tests.  No YOLO, no insightface, no webcams, no
network -- detector internals and health gates are stubbed with synthetic
numpy frames and ``unittest.mock``.

Covers:
* N persons -> N independent person detections (never first-face-only)
* N faces -> N independent identities (no label/embedding cross-talk)
* face -> person pairing uses face CENTRE + nearest person centroid
* per-person phone attribution (a phone near person B is not person A's)
* multi-employee independent trackers + independent states in live_state
* blank/dark or dead camera freezes state -- NEVER fabricates AWAY
* FROZEN_FRAME but bright stays usable (presence still monitored)
* camera recovery resumes normal tracking
* Unknown person is tracked separately, never becomes an employee
* security engine treats FROZEN/NO_FRAME as offline (failsafe) and
  LOW_FPS as online
* dashboard live table surfaces an Unknown row and distinguishes a
  dead/blank feed from a frozen feed
"""

import json
import time as _realtime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import main as m
import src.tracker as T
from src.detector import ActivityDetector, _containing_person, _phone_near_person
from src.face_registry import FaceRegistry
from src.security_engine import SecurityEngine

ROOT = Path(__file__).resolve().parents[1]


# ======================================================================
# Fixtures shared with the rest of the suite
# ======================================================================

class _FakeTime:
    now = 10_000.0

    @classmethod
    def time(cls) -> float:
        return cls.now


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
def fast_buffer(monkeypatch):
    monkeypatch.setattr(T, "SMOOTHING_BUFFER_SEC", 5.0)
    monkeypatch.setattr(T, "IDENTITY_STABILITY_FRAMES", 3)
    monkeypatch.setattr(T, "PRESENCE_PATIENCE_SEC", 8.0)
    return T.SMOOTHING_BUFFER_SEC


@pytest.fixture
def tmp_db(tmp_path):
    from src.database import get_connection, init_db
    conn = get_connection(tmp_path / "t.db")
    init_db(conn)
    yield conn
    conn.close()


# ======================================================================
# Detector-level: multi-face / multi-person independence
# ======================================================================

def _face(bbox, emp, score):
    f = MagicMock()
    f.bbox = np.array(bbox, dtype=float)
    f.normed_embedding = np.zeros(512, dtype=np.float32)
    return f, (emp, score)


def _detector(face_results, decisions):
    det = ActivityDetector.__new__(ActivityDetector)
    det.conf = 0.5
    det._half = False
    det._device = "cpu"
    det.model = MagicMock()
    det._face_registry = MagicMock()
    det._face_registry.app.get.return_value = face_results
    det._face_registry.identify.side_effect = decisions
    det._face_registry.threshold = 0.6
    return det


def test_two_faces_two_identified_employees():
    f1, l1 = _face((160, 60, 320, 140), "EMP001", 0.70)
    f2, l2 = _face((180, 300, 320, 400), "EMP004", 0.65)
    det = _detector([f1, f2], [l1, l2])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, fboxes, flabels, _, _ = det._process_frame(frame)
    assert flabels == [("EMP001", 0.70), ("EMP004", 0.65)]
    assert len(fboxes) == 2
    assert det._face_registry.identify.call_count == 2


def test_three_faces_three_independent_identities():
    faces = [_face((150, 40, 300, 140), "EMP001", 0.7)[0],
             _face((150, 160, 300, 260), "EMP004", 0.66)[0],
             _face((150, 290, 300, 400), "Unknown", 0.5)[0]]
    decisions = [("EMP001", 0.70), ("EMP004", 0.66), ("Unknown", 0.50)]
    det = _detector(faces, decisions)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, fboxes, flabels, _, present = det._process_frame(frame)
    assert [e for e, _ in flabels] == ["EMP001", "EMP004", "Unknown"]
    assert len(fboxes) == 3
    assert present is False  # no person boxes supplied


def test_four_persons_four_person_entries():
    det = _detector([], [])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = [(0, 0, 100, 240), (110, 0, 210, 240), (220, 0, 320, 240),
             (330, 0, 430, 240)]
    dets = det._process_frame(frame, phone_boxes=[], cam_id="cam_01",
                              person_boxes=boxes, person_confs=[0.9] * 4)
    persons = [d for d in dets if d.get("det_type") == "person"]
    assert len(persons) == 4
    assert all(d["emp_id"] == "__person__" for d in persons)
    assert {d["person_box"] for d in persons} == set(boxes)


def test_face_paired_to_own_person_box():
    pbox1 = (0, 0, 640, 240)
    pbox2 = (100, 240, 360, 470)
    f1, l1 = _face((160, 60, 320, 140), "EMP001", 0.70)
    f2, l2 = _face((180, 300, 320, 400), "EMP004", 0.65)
    det = _detector([f1, f2], [l1, l2])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = det._process_frame(frame, phone_boxes=[], cam_id="cam_01",
                              person_boxes=[pbox1, pbox2],
                              person_confs=[0.9, 0.9])
    by_id = {d["emp_id"]: d for d in dets if d.get("det_type") is None}
    assert by_id["EMP001"]["person_box"] == tuple(pbox1)
    assert by_id["EMP004"]["person_box"] == tuple(pbox2)


def test_face_association_nearest_centroid_not_first_in_list():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pbox_first = (0, 0, 640, 120)     # listed first, expanded box still covers face
    pbox_nearest = (100, 80, 320, 220)  # centroid closest to the face centre
    face = (150, 110, 250, 190)
    person = _containing_person(face, [pbox_first, pbox_nearest], frame.shape)
    assert person == tuple(pbox_nearest)


def test_phone_attributed_to_own_person_only():
    det = _detector([], [])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    person_a = (0, 0, 200, 400)
    person_b = (400, 80, 640, 460)
    phone = (480, 300, 540, 360)
    dets = det._process_frame(frame, phone_boxes=[phone], cam_id="cam_01",
                              person_boxes=[person_a, person_b],
                              person_confs=[0.9, 0.9])
    persons = {d["person_box"]: d for d in dets if d.get("det_type") == "person"}
    assert persons[tuple(person_a)]["phone"] is False
    assert persons[tuple(person_b)]["phone"] is True


def test_face_unmatched_person_present_no_identity_invented():
    f1, l1 = _face((160, 60, 320, 140), "Unknown", 0.5)
    det = _detector([f1], [l1])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    status, _, flabels, _, present = det._process_frame(
        frame, phone_boxes=[], person_boxes=[(0, 0, 640, 240)],
        person_confs=[0.9])
    assert flabels == [("Unknown", 0.5)]
    assert present is True
    assert "Unknown" in status            # monitored, but never an employee row
    assert "EMP001" not in status


def test_no_faces_person_present_still_drives_presence():
    det = _detector([], [])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, _, flabels, _, present = det._process_frame(
        frame, phone_boxes=[], person_boxes=[(0, 0, 640, 240)],
        person_confs=[0.9])
    assert flabels == []
    assert present is True  # someone is visibly there even if no face


# ======================================================================
# Tracker-level: multi-person independence + camera health gating
# ======================================================================

def test_two_employees_two_independent_tracks_two_states(fake_clock, recorder):
    mt = T.MultiTracker(None)
    dets = [
        {"emp_id": "EMP001", "present": True, "phone": False},
        {"emp_id": "EMP004", "present": True, "phone": True},
    ]
    mt.process_batch(dets, camera_online=True)
    assert sorted(mt.employee_ids) == ["EMP001", "EMP004"]
    assert mt.live_state("EMP001") == "ACTIVE"
    assert mt.live_state("EMP004") == "ON_PHONE"


def test_per_employee_phone_does_not_leak(fake_clock, recorder, fast_buffer):
    mt = T.MultiTracker(None)
    for _ in range(4):
        mt.process_batch([
            {"emp_id": "EMP001", "present": True, "phone": False},
            {"emp_id": "EMP004", "present": True, "phone": True},
        ], camera_online=True)
        _FakeTime.now += 1
    assert mt.live_state("EMP001") == "ACTIVE"
    assert mt.live_state("EMP004") == "ON_PHONE"


def test_blank_camera_freezes_state_never_away(fake_clock, recorder):
    dark_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    usable = m._usable_camera_frames(
        {"local_webcam": dark_frame},
        {"local_webcam": {"health": "ONLINE"}})
    assert usable == {}  # the daemon loop would set camera_online=False
    mt = T.MultiTracker(None)
    mt.process_batch([{"emp_id": "EMP001", "present": True}], camera_online=True)
    _FakeTime.now += 90
    mt.process_batch([], camera_online=bool(usable))
    assert mt.live_state("EMP001") == "ACTIVE"
    assert recorder == []  # no AWAY seconds recorded


def test_frozen_bright_camera_keeps_monitoring(fake_clock, recorder):
    bright = np.full((240, 320, 3), 120, dtype=np.uint8)
    usable = m._usable_camera_frames(
        {"local_webcam": bright},
        {"local_webcam": {"health": "FROZEN_FRAME"}})
    assert "local_webcam" in usable  # FROZEN but usable -> presence still works
    mt = T.MultiTracker(None)
    mt.process_batch([{"emp_id": "EMP001", "present": True}], camera_online=True)
    _FakeTime.now += 30
    mt.process_batch([{"emp_id": "EMP001", "present": True}],
                     camera_online=bool(usable))
    assert mt.live_state("EMP001") == "ACTIVE"


def test_offline_from_start_creates_no_tracker_no_away(fake_clock, recorder):
    mt = T.MultiTracker(None)
    mt.process_batch([], camera_online=False)
    assert mt.employee_ids == []  # never a phantom AWAY/Unknown tracker


def test_camera_recovery_resumes_tracking(fake_clock, recorder, fast_buffer):
    mt = T.MultiTracker(None)
    mt.process_batch([{"emp_id": "EMP001", "present": True}], camera_online=True)
    _FakeTime.now += 50
    mt.process_batch([], camera_online=False)          # camera dies
    assert mt.live_state("EMP001") == "ACTIVE"          # frozen, not AWAY
    _FakeTime.now += 50
    mt.process_batch([{"emp_id": "EMP001", "present": True}], camera_online=True)
    assert mt.live_state("EMP001") == "ACTIVE"          # resumed normally


def test_dedup_same_employee_across_cameras_single_tracker(fake_clock, recorder):
    mt = T.MultiTracker(None)
    mt.process_batch([
        {"emp_id": "EMP001", "present": True, "cam": "cam_01"},
        {"emp_id": "EMP001", "present": True, "cam": "cam_02"},
    ], camera_online=True)
    assert mt.employee_ids == ["EMP001"]


def test_unknown_tracked_separately_never_an_employee(fake_clock, recorder):
    mt = T.MultiTracker(None)
    mt.process_batch([{"emp_id": "Unknown", "present": True}],
                     camera_online=True)
    assert mt.employee_ids == ["Unknown"]
    assert mt.live_state("Unknown") == "ACTIVE"


# ======================================================================
# main gate + live_state payload
# ======================================================================

def test_usable_frames_drops_offline_camera():
    bright = np.full((240, 320, 3), 200, dtype=np.uint8)
    health = {"cam_01": {"health": "OFFLINE", "id": "cam_01"}}
    usable = m._usable_camera_frames({"cam_01": bright}, health)
    assert usable == {}
    assert health["cam_01"]["usable"] is False
    assert health["cam_01"]["unusable_reason"] == "OFFLINE"


def test_usable_frames_drops_blank_dark_feed_with_reason():
    dark = np.zeros((240, 320, 3), dtype=np.uint8)
    health = {"local_webcam": {"health": "ONLINE", "id": "local_webcam"}}
    usable = m._usable_camera_frames({"local_webcam": dark}, health)
    assert usable == {}
    assert health["local_webcam"]["usable"] is False
    assert health["local_webcam"]["unusable_reason"] == "DARK_BLANK_FRAME"
    assert health["local_webcam"]["frame_mean"] == 0.0


def test_usable_frames_keeps_frozen_and_low_fps_bright():
    bright = np.full((240, 320, 3), 120, dtype=np.uint8)
    health = {
        "frozen": {"health": "FROZEN_FRAME", "id": "frozen"},
        "slow": {"health": "LOW_FPS", "id": "slow"},
    }
    usable = m._usable_camera_frames({"frozen": bright, "slow": bright}, health)
    assert set(usable) == {"frozen", "slow"}
    assert health["frozen"]["usable"] is True


def test_usable_frames_annotations_on_health_dict():
    bright = np.full((240, 320, 3), 150, dtype=np.uint8)
    health = {"local_webcam": {"health": "ONLINE", "id": "local_webcam"}}
    m._usable_camera_frames({"local_webcam": bright}, health)
    assert health["local_webcam"]["usable"] is True
    assert health["local_webcam"]["frame_mean"] == 150.0


def _write_live_state(tmp_path, **kw):
    target = str(tmp_path / "live.json")
    old = m._LIVE_STATE_FILE
    m._LIVE_STATE_FILE = target
    try:
        m._write_live_state(
            kw.get("states", {}),
            kw.get("telemetry", {}),
            kw.get("cam_health", {}),
            kw.get("fps", 2.0),
            sources=kw.get("sources"),
            durations=kw.get("durations"),
            names=kw.get("names"),
            last_seen=kw.get("last_seen"),
        )
    finally:
        m._LIVE_STATE_FILE = old
    return json.loads(Path(target).read_text(encoding="utf-8"))


def test_live_state_independent_rows_and_unknown_excluded(tmp_path):
    now = _realtime.time()
    data = _write_live_state(
        tmp_path,
        states={"EMP001": "ACTIVE", "EMP004": "ON_PHONE", "Unknown": "ACTIVE"},
        telemetry={
            "EMP001": {"ACTIVE": 10.0},
            "EMP004": {"ON_PHONE": 5.0},
            "Unknown": {"ACTIVE": 2.0},
        },
        cam_health={"local_webcam": {"health": "ONLINE", "usable": True}},
        sources={"EMP001": "local_webcam", "Unknown": "local_webcam"},
        durations={"EMP001": 10.0, "EMP004": 5.0},
        names={"EMP001": "Yashmit", "EMP004": "Priya"},
        last_seen={"EMP001": now, "EMP004": now, "Unknown": now},
    )
    assert data["employees"]["EMP001"]["state"] == "ACTIVE"
    assert data["employees"]["EMP004"]["state"] == "ON_PHONE"
    assert "Unknown" not in data["employees"]       # never an employee row
    assert data["states"]["Unknown"] == "ACTIVE"     # tracked at top level
    assert data["last_seen"]["Unknown"] == now
    assert data["camera_health"]["local_webcam"]["usable"] is True


def test_live_state_never_reports_unobserved_employee_as_away(tmp_path):
    data = _write_live_state(
        tmp_path,
        states={"EMP001": "ACTIVE"},
        cam_health={"local_webcam": {"health": "ONLINE"}},
        names={"EMP001": "Yashmit", "EMP002": "Rahul", "EMP003": "Sneha"},
        last_seen={"EMP001": _realtime.time()},
    )
    assert data["employees"]["EMP002"]["state"] == "NOT_OBSERVED"
    assert data["employees"]["EMP003"]["state"] == "NOT_OBSERVED"
    assert data["employees"]["EMP002"]["session_sec"] == 0.0


# ======================================================================
# Security engine: FROZEN/NO_FRAME offline + LOW_FPS online (failsafe)
# ======================================================================

@pytest.fixture
def engine(tmp_db):
    enc = SecurityEngine(tmp_db)
    enc._cam_state.update("cam_01", True, now=0.0)
    return enc


def _sec_clock(monkeypatch):
    """Fake monotonic clock for the security engine.

    Starts at 0.0 — acceptable now that ``CameraStateManager.update`` uses
    ``now if now is not None else time.time()`` (0.0 was previously treated as
    "not provided", falling back to the real wall clock).
    """
    import src.security_engine as sec
    _FakeTime.now = 0.0
    monkeypatch.setattr(sec, "time", _FakeTime)
    return _FakeTime


def test_camera_state_update_honours_now_zero_clock():
    """`update(camera_id, online, now=0.0)` must use the provided clock.

    Regression: `now = now or time.time()` treated 0.0 as falsy and fell back
    to the real wall clock, so the offline trigger could never fire from a
    test clock (and any client passing now=0 in production would break).
    """
    from src.camera_state import CameraStateManager
    from src.domain import CAMERA_OFFLINE

    m = CameraStateManager(offline_trigger_sec=15.0)
    m.update("cam_01", True, now=0.0)   # goes online at t=0
    m.update("cam_01", False, now=0.0)  # dies at t=0 → offline_since=0.0
    events = m.update("cam_01", False, now=16.0)  # 16-0 = 16 ≥ 15 → fires
    assert any(e["event_type"] == CAMERA_OFFLINE for e in events)
    st = m.all_states()["cam_01"]
    assert st["offline_since"] == 0.0


def test_security_tick_frozen_drives_offline_failsafe(engine, monkeypatch):
    from src.domain import CAMERA_OFFLINE
    ft = _sec_clock(monkeypatch)
    engine.tick([], {"cam_01": {"health": "FROZEN_FRAME"}}, frames=None)
    ft.now += 16
    events = engine.tick([], {"cam_01": {"health": "FROZEN_FRAME"}}, frames=None)
    assert any(e["event_type"] == CAMERA_OFFLINE for e in events)


def test_security_tick_no_frame_drives_offline_failsafe(engine, monkeypatch):
    from src.domain import CAMERA_OFFLINE
    ft = _sec_clock(monkeypatch)
    engine.tick([], {"cam_01": {"health": "NO_FRAME"}}, frames=None)
    ft.now += 16
    events = engine.tick([], {"cam_01": {"health": "NO_FRAME"}}, frames=None)
    assert any(e["event_type"] == CAMERA_OFFLINE for e in events)


def test_security_tick_low_fps_stays_online(engine, monkeypatch):
    from src.domain import CAMERA_OFFLINE
    ft = _sec_clock(monkeypatch)
    events = []
    for _ in range(3):
        events += engine.tick([], {"cam_01": {"health": "LOW_FPS"}}, frames=None)
        ft.now += 20
    assert not any(e["event_type"] == CAMERA_OFFLINE for e in events)


def test_security_tick_online_stays_online(engine, monkeypatch):
    from src.domain import CAMERA_OFFLINE
    ft = _sec_clock(monkeypatch)
    for _ in range(2):
        engine.tick([], {"cam_01": {"health": "ONLINE"}}, frames=None)
        ft.now += 20
    events = engine.tick([], {"cam_01": {"health": "ONLINE"}}, frames=None)
    assert not any(e["event_type"] == CAMERA_OFFLINE for e in events)


# ======================================================================
# Dashboard: Unknown row + dead-vs-frozen camera messaging
# ======================================================================

def _app_segment(start) -> str:
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    return src.split(start)[1]


def test_live_table_source_shows_unknown_row_and_blank_feed_warning():
    seg = _app_segment("def tab_live_employees")
    seg = seg.split("def tab_camera_health")[0]
    assert "Unknown person" in seg
    assert "DARK_BLANK_FRAME" in seg
    assert "never counts as" in seg.lower() and "frozen" in seg.lower()


def test_live_table_source_distinguishes_frozen_vs_dead():
    seg = _app_segment("def tab_live_employees")
    seg = seg.split("def tab_camera_health")[0]
    assert "FROZEN_FRAME" in seg
    assert "Monitoring continues" in seg  # frozen-but-flowing is not "dead"


def test_camera_health_tab_source_reports_usable_and_frozen_notes():
    seg = _app_segment("def tab_camera_health")
    seg = seg.split("def tab_reports")[0]
    assert "Usable" in seg and "Reason" in seg
    assert "FROZEN_FRAME" in seg and "frozen" in seg.lower()
    assert "never" in seg.lower()  # offline/blank is never AWAY time


# ======================================================================
# EMP004 recognition diagnostics (registry top-k + tracker adoption)
# ======================================================================

def _reg(rows_by_id, threshold=0.6):
    """Build a FaceRegistry shell with synthetic unit-norm 512-D embeddings."""
    reg = FaceRegistry.__new__(FaceRegistry)
    ids, embs = [], []
    for eid, vec in rows_by_id.items():
        v = np.zeros(512, dtype=np.float32)
        v[: min(len(vec), 512)] = vec
        v = v / (np.linalg.norm(v) + 1e-9)
        embs.append(v)
        ids.append(eid)
    reg._employee_ids = ids
    reg._embeddings = np.stack(embs) if embs else np.empty((0, 512), np.float32)
    reg._by_employee = {eid: [i] for i, eid in enumerate(ids)}
    reg._file_status = {}
    reg._names = None
    reg.threshold = threshold
    return reg


def _q(vec):
    q = np.zeros(512, dtype=np.float32)
    q[: min(len(vec), 512)] = vec
    return q / (np.linalg.norm(q) + 1e-9)


def test_identify_k_returns_top_k_descending():
    reg = _reg({"EMP001": [1, 0, 0], "EMP002": [0, 1, 0], "EMP004": [0.5, 0.5, 0.4]})
    top = reg.identify_k(_q([0, 1, 0]), k=2)
    assert top[0][0] == "EMP002"
    assert top[0][1] > top[1][1]
    assert len(top) == 2


def test_identify_k_reports_argmax_even_below_threshold():
    # query is closest to EMP004 but at 0.55 < threshold 0.60: identify() -> Unknown,
    # yet identify_k must reveal that the near-miss identity is EMP004.
    # (axis 3 is a neutral filler so the remainder does not fall on EMP005.)
    reg = _reg({"EMP001": [1, 0, 0], "EMP004": [0, 1, 0], "EMP005": [0, 0, 1]})
    q = _q([0.0, 0.55, 0.05, 0.83])
    best_id, best_score = reg.identify(q)
    assert best_id == "Unknown"
    assert best_score < reg.threshold
    top = reg.identify_k(q, k=2)
    assert top[0][0] == "EMP004"
    assert 0.54 <= top[0][1] <= 0.56


def test_emp004_self_match_is_top_one_in_identify_k():
    reg = _reg({"EMP001": [1, 0, 0], "EMP002": [0.9, 0.1, 0], "EMP004": [0, 1, 0]})
    top = reg.identify_k(_q([0, 1, 0]), k=1)
    assert top[0] == ("EMP004", 1.0)


def test_emp004_separable_from_emp001_no_confusability():
    reg = _reg({"EMP001": [1, 0, 0], "EMP004": [0, 1, 0]})
    assert reg.confusability_report() == []
    embs = reg._embeddings
    off = embs @ embs.T
    off[range(len(embs)), range(len(embs))] = 0.0  # ignore self-similarity
    assert off.max() < 0.88  # identities are pairwise distinguishable


def test_face_debug_logs_candidate_name_and_runner_up(caplog, monkeypatch):
    import logging
    import src.detector as det_mod

    monkeypatch.setattr(det_mod, "FACE_DEBUG", True)
    monkeypatch.setattr(det_mod, "DETECTION_DEBUG", False)
    f1, _ = _face((160, 60, 320, 140), "EMP004", 0.57)
    det = _detector([f1], [("Unknown", 0.57)])
    det._face_registry.identify_k.return_value = [("EMP004", 0.57), ("EMP001", 0.20)]
    det._face_registry.employee_name.side_effect = {"EMP004": "Vishal",
                                                    "EMP001": "Yashmit Sharma"}.get
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with caplog.at_level(logging.INFO, logger="cctv.detector"):
        det._process_frame(frame, phone_boxes=[], person_boxes=[],
                           person_confs=[])
    line = next(r.getMessage() for r in caplog.records
                if "FACE_DEBUG" in r.getMessage())
    assert "candidate=EMP004" in line
    assert "name=Vishal" in line
    assert "similarity=0.5700" in line
    assert "runner_up=EMP001" in line
    assert "runner_up_sim=0.2000" in line
    assert "decision=UNKNOWN" in line  # diagnosed, honest: below 0.60


def test_tracker_eventually_adopts_emp004():
    st = T.SpatialTracker(adopt_frames=3)
    person = {"cam": "cam_01", "det_type": "person", "person_box": (0, 0, 100, 240),
              "person_conf": 0.9, "frame_w": 640, "frame_h": 480}
    face = {"cam": "cam_01", "face_box": (10, 20, 60, 100), "emp_id": "EMP004",
            "face_score": 0.67, "person_box": (0, 0, 100, 240),
            "frame_w": 640, "frame_h": 480}
    for i in range(3):
        st.process([person, face], now=1000.0 + i)
    tracks = st.active_tracks(1000.0 + 2)
    assert len(tracks) == 1
    assert tracks[0].identity == "EMP004"


def test_tracker_emp004_unknown_vote_never_demotes(fake_clock):
    st = T.SpatialTracker(adopt_frames=2)
    person = {"cam": "cam_01", "det_type": "person", "person_box": (0, 0, 100, 240),
              "person_conf": 0.9, "frame_w": 640, "frame_h": 480}
    face = {"cam": "cam_01", "face_box": (10, 20, 60, 100), "emp_id": "EMP004",
            "face_score": 0.67, "person_box": (0, 0, 100, 240),
            "frame_w": 640, "frame_h": 480}
    st.process([person, face], now=1000.0)
    st.process([person, face], now=1000.0 + 1)
    assert st.active_tracks(1000.0 + 1)[0].identity == "EMP004"
    unknown_face = dict(face, emp_id="Unknown", face_score=0.45)
    for i in range(4):
        st.process([person, unknown_face], now=1100.0 + i)
    assert st.active_tracks(1100.0 + 3)[0].identity == "EMP004"


def test_tracker_two_people_identities_do_not_leak():
    st = T.SpatialTracker(adopt_frames=3)
    p_left = (0, 0, 100, 260)
    p_right = (200, 0, 320, 260)
    left = [{"cam": "cam_01", "det_type": "person", "person_box": p_left,
             "person_conf": 0.9, "frame_w": 640, "frame_h": 480},
            {"cam": "cam_01", "face_box": (10, 30, 60, 120), "emp_id": "EMP001",
             "face_score": 0.72, "person_box": p_left, "frame_w": 640, "frame_h": 480}]
    right = [{"cam": "cam_01", "det_type": "person", "person_box": p_right,
              "person_conf": 0.9, "frame_w": 640, "frame_h": 480},
             {"cam": "cam_01", "face_box": (210, 30, 260, 120), "emp_id": "EMP004",
              "face_score": 0.69, "person_box": p_right, "frame_w": 640, "frame_h": 480}]
    for i in range(3):
        st.process(left + right, now=1200.0 + i)
    tracks = st.active_tracks(1200.0 + 2)
    by_bbox = {t.bbox: t.identity for t in tracks}
    assert by_bbox[tuple(p_left)] == "EMP001"
    assert by_bbox[tuple(p_right)] == "EMP004"


def test_live_state_emp004_name_vishal(tmp_path):
    now = _realtime.time()
    data = _write_live_state(
        tmp_path,
        states={"EMP004": "ACTIVE"},
        telemetry={"EMP004": {"ACTIVE": 5.0}},
        cam_health={"local_webcam": {"health": "ONLINE", "usable": True}},
        names={"EMP001": "Yashmit Sharma", "EMP002": "Rahul Kumar",
               "EMP003": "Priya Sharma", "EMP004": "Vishal"},
        last_seen={"EMP004": now},
    )
    assert data["employees"]["EMP004"]["employee_id"] == "EMP004"
    assert data["employees"]["EMP004"]["name"] == "Vishal"
    assert data["employees"]["EMP004"]["state"] == "ACTIVE"
    assert data["last_seen"]["EMP004"] == now
