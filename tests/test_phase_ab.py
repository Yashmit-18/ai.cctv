"""Phase A/B regression tests.

Covers the Phase A (spatial person tracking, motion, virtual-line entry/exit,
camera health) and Phase B (registry vocabulary, zone capacity, intrusion
cooldown, line-based LEFT debounce, UNEXPECTED_STAY) capabilities added on top
of the original platform, plus the new ``motion_events``/``entry_exit_events``
tables and the dashboard ``ai`` live payload.
"""

import json
import threading
import time

import numpy as np
import pytest

import config
import src.database as db
from src.camera import VideoCapture
from src.detector_registry import DetectorRegistry
from src.domain import (
    CAM_FROZEN,
    CAM_LOW_FPS,
    CAM_ONLINE,
    ENTRY_CROSSING,
    EXIT_CROSSING,
    LEFT,
    MOTION_DETECTED,
    UNEXPECTED_STAY,
    UNKNOWN_ID,
)
from src.entry_exit import EntryExitDetector, VirtualLine
from src.motion import MotionDetector
from src.security_engine import SecurityEngine
from src.tracker import SpatialTracker, Track
from src.zones import ZoneStore


def _person_det(cam, box, conf=0.9, phone=False, fw=200, fh=200):
    return {
        "cam": cam, "det_type": "person", "emp_id": "__person__",
        "person_box": tuple(box), "person_conf": conf, "phone": phone,
        "frame_w": fw, "frame_h": fh,
    }


def _face_det(cam, emp_id, face_box, person_box, fw=200, fh=200, score=0.9):
    return {
        "cam": cam, "emp_id": emp_id, "face_score": score,
        "face_box": tuple(face_box), "person_box": tuple(person_box),
        "frame_w": fw, "frame_h": fh,
    }


# ======================================================================
# Detector helpers
# ======================================================================

class _Box:
    def __init__(self, cls, xyxy):
        self.cls = np.array(cls)
        self.xyxy = np.array([list(xyxy)], dtype=float)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


def test_boxes_and_conf_defaults_when_no_conf_attr():
    from src.detector import _boxes_and_conf_of_class
    r = _Result([_Box(0, (1, 2, 3, 4)), _Box(1, (5, 6, 7, 8))])
    items = _boxes_and_conf_of_class(r, 0)
    assert items == [((1, 2, 3, 4), 1.0)]


def test_boxes_and_conf_reads_explicit_conf():
    from src.detector import _boxes_and_conf_of_class
    box = _Box(0, (1, 2, 3, 4))
    box.conf = np.array(0.813, dtype=float)
    r = _Result([box])
    items = _boxes_and_conf_of_class(r, 0)
    assert items == [((1, 2, 3, 4), 0.813)]


# ======================================================================
# Track: geometry and identity voting (A5/A6/A7)
# ======================================================================

def test_track_centroid_normalized():
    t = Track("t1", "c1", (40, 50, 60, 70), 0.9, 200, 200, 0.0)
    cx, cy = t.centroid_normalized
    assert cx == pytest.approx(0.25)
    assert cy == pytest.approx(0.3)


def test_track_identity_adopts_after_three_votes():
    t = Track("t1", "c1", (0, 0, 10, 10), 0.9, 100, 100, 0.0)
    assert t.identity is None
    for i in range(3):
        t.register_vote("EMP001", 0.9, float(i))
    assert t.identity == "EMP001"


def test_track_identity_never_demoted_to_unknown():
    t = Track("t1", "c1", (0, 0, 10, 10), 0.9, 100, 100, 0.0)
    for i in range(3):
        t.register_vote("EMP001", 0.9, float(i))
    for i in range(4, 8):
        t.register_vote(UNKNOWN_ID, 0.0, float(i))
    assert t.identity == "EMP001"


def test_track_identity_switch_needs_prior_run_dropout():
    t = Track("t1", "c1", (0, 0, 10, 10), 0.9, 100, 100, 0.0)
    for i in range(3):
        t.register_vote("EMP001", 0.9, float(i))
    assert t.identity == "EMP001"
    for i in range(2):
        t.register_vote("EMP002", 0.9, 100.0 + i)
    assert t.identity == "EMP001"
    t.register_vote("EMP002", 0.9, 103.0)
    assert t.identity == "EMP002"


def test_track_phone_seconds():
    t = Track("t1", "c1", (0, 0, 10, 10), 0.9, 100, 100, 100.0)
    assert t.phone_seconds(105.0) == 0.0
    t.update((0, 0, 10, 10), 0.9, phone=True, now=105.0)
    assert t.phone is True
    assert t.phone_seconds(111.0) == pytest.approx(6.0, abs=0.01)


# ======================================================================
# SpatialTracker integration (A5/A6/A7)
# ======================================================================

def test_single_person_keeps_one_track():
    st = SpatialTracker()
    tracks = st.process([_person_det("c1", (10, 10, 50, 50))], now=10.0)
    assert len(tracks) == 1
    tracks = st.process([_person_det("c1", (12, 12, 52, 52))], now=10.2)
    assert [t.track_id for t in tracks] == ["c1#1"]
    assert len(st.tracks) == 1


def test_two_persons_spawn_two_tracks():
    st = SpatialTracker()
    dets = [
        _person_det("c1", (10, 10, 40, 90)),
        _person_det("c1", (150, 10, 190, 90)),
    ]
    tracks = st.process(dets, now=1.0)
    assert len(tracks) == 2
    assert st.persons_by_camera() == {"c1": 2}


def test_face_votes_reinforce_identity():
    st = SpatialTracker()
    pdet = _person_det("c1", (10, 10, 90, 90))
    fdet = _face_det("c1", "EMP001", (30, 20, 70, 80), (10, 10, 90, 90))
    for i in range(3):
        st.process([pdet, fdet], now=10.0 + i * 0.25)
    track = st.active_tracks(11.0)[0]
    assert track.identity == "EMP001"


def test_phone_person_flagged_on_track():
    st = SpatialTracker()
    tracks = st.process([_person_det("c1", (10, 10, 50, 50), phone=True)],
                        now=10.0)
    assert tracks[0].phone is True
    assert tracks[0].phone_seconds(16.0) == pytest.approx(6.0, abs=0.01)


def test_prune_drops_stale_tracks():
    st = SpatialTracker(max_age_sec=3.0)
    st.process([_person_det("c1", (10, 10, 50, 50))], now=100.0)
    st.process([], now=104.0)
    assert st.snapshot(now=104.0)["count"] == 0


def test_snapshot_is_json_safe():
    st = SpatialTracker()
    st.process([_person_det("c1", (10, 10, 50, 50))], now=100.0)
    snap = st.snapshot(now=100.0)
    assert snap["count"] == 1
    row = snap["tracks"][0]
    assert {"track_id", "cam", "identity", "phone", "bbox", "trajectory"} <= set(row)
    json.dumps(snap)  # must serialize


# ======================================================================
# MotionDetector (A8)
# ======================================================================

def _frames(**kw):
    return {k: np.asarray(v, dtype=np.uint8) for k, v in kw.items()}


_base = np.zeros((240, 240, 3), dtype=np.uint8)
_move_a = np.full((240, 240, 3), 60, dtype=np.uint8)
_move_b = np.full((240, 240, 3), 90, dtype=np.uint8)
_move_c = np.full((240, 240, 3), 120, dtype=np.uint8)


def test_motion_first_frame_only_primes():
    md = MotionDetector(enabled=True, min_score=0.02, min_duration_sec=1.0,
                        cooldown_sec=100, downscale=0)
    status, events = md.process({"c1": _base}, ts=10.0)
    assert events == []
    assert status["c1"]["primed"] is True
    assert status["c1"]["motion"] is False


def test_motion_fires_after_duration_gate():
    md = MotionDetector(enabled=True, min_score=0.02, min_duration_sec=1.0,
                        cooldown_sec=100, downscale=0)
    md.process({"c1": _base}, ts=0.0)
    md.process({"c1": _move_a}, ts=0.5)
    status, events = md.process({"c1": _move_b}, ts=1.6)
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == MOTION_DETECTED
    assert ev["camera"] == "c1"
    assert ev["duration_seconds"] == pytest.approx(1.1, abs=0.01)
    assert ev["motion_score"] > 0


def test_continuous_motion_yields_single_event_per_run():
    md = MotionDetector(enabled=True, min_score=0.02, min_duration_sec=1.0,
                        cooldown_sec=100, downscale=0)
    md.process({"c1": _base}, ts=0.0)
    md.process({"c1": _move_a}, ts=0.5)
    _, evs = md.process({"c1": _move_b}, ts=1.6)
    assert len(evs) == 1
    md.process({"c1": _move_b}, ts=2.0)
    _, evs = md.process({"c1": _move_c}, ts=3.1)
    assert evs == []


def test_motion_cooldown_suppresses_then_refires():
    md = MotionDetector(enabled=True, min_score=0.02, min_duration_sec=0.0,
                        cooldown_sec=30, downscale=0)
    md.process({"c1": _base}, ts=0.0)
    _, evs = md.process({"c1": _move_a}, ts=5.0)
    assert len(evs) == 1
    _, evs = md.process({"c1": _move_b}, ts=10.0)
    assert evs == []
    _, evs = md.process({"c1": _move_c}, ts=41.0)
    assert len(evs) == 1


def test_motion_disabled_returns_empty():
    md = MotionDetector(enabled=False, min_score=0.02, min_duration_sec=0.0,
                        cooldown_sec=30, downscale=0)
    status, events = md.process({"c1": _move_a}, ts=1.0)
    assert events == []
    assert status == {}


def test_static_scene_resets_active_run():
    md = MotionDetector(enabled=True, min_score=0.02, min_duration_sec=1.0,
                        cooldown_sec=100, downscale=0)
    md.process({"c1": _base}, ts=0.0)
    md.process({"c1": _move_a}, ts=0.5)
    status, events = md.process({"c1": _move_a}, ts=1.0)
    assert events == []
    assert status["c1"]["motion"] is False


# ======================================================================
# EntryExitDetector (A10)
# ======================================================================

def _track(cx, cam="c1", identity=None, tid="t1", fw=200, fh=100):
    x1 = int(round(cx * fw - 5))
    x2 = int(round(cx * fw + 5))
    t = Track(tid, cam, (x1, 40, x2, 60), 0.9, fw, fh, 0.0)
    t.identity = identity
    return t


def _line(**kw):
    kw.setdefault("name", "main_door")
    kw.setdefault("cam", "c1")
    return VirtualLine(**kw)


def test_virtual_line_side_math():
    line = _line()
    assert line.side(0.3, 0.5) == "A"   # left-of directed line
    assert line.side(0.7, 0.5) == "B"   # right-of directed line


def test_virtual_line_default_mapping():
    line = _line()
    assert line.event_for("A_TO_B") == EXIT_CROSSING
    assert line.event_for("B_TO_A") == ENTRY_CROSSING
    assert line.event_for("A_TO_A") is None


def test_crossing_primes_first_frame_only():
    det = EntryExitDetector(lines=[_line()], debounce_sec=30)
    assert det.process([_track(0.3)], ts=100.0) == []
    assert det.process([_track(0.3)], ts=101.0) == []


def test_exit_crossing_emitted_once():
    det = EntryExitDetector(lines=[_line()], debounce_sec=30)
    det.process([_track(0.3)], ts=100.0)
    evs = det.process([_track(0.7)], ts=101.0)
    assert len(evs) == 1
    ev = evs[0]
    assert ev["event_type"] == EXIT_CROSSING
    assert ev["direction"] == "A_TO_B"
    assert ev["line"] == "main_door"
    assert ev["camera"] == "c1"
    assert ev["track_id"] == "t1"
    assert det.process([_track(0.7)], ts=102.0) == []


def test_entry_crossing_on_reverse_direction():
    det = EntryExitDetector(lines=[_line()], debounce_sec=0.1)
    det.process([_track(0.3)], ts=100.0)
    det.process([_track(0.7)], ts=101.0)
    evs = det.process([_track(0.3)], ts=101.2)
    assert len(evs) == 1
    assert evs[0]["event_type"] == ENTRY_CROSSING
    assert evs[0]["direction"] == "B_TO_A"


def test_debounce_suppresses_rapid_recrossing():
    det = EntryExitDetector(lines=[_line()], debounce_sec=30)
    det.process([_track(0.3)], ts=100.0)
    det.process([_track(0.7)], ts=101.0)
    assert det.process([_track(0.3)], ts=102.0) == []
    evs = det.process([_track(0.7)], ts=160.0)
    assert len(evs) == 1
    assert evs[0]["event_type"] == EXIT_CROSSING


def test_crossing_carries_employee_identity():
    det = EntryExitDetector(lines=[_line()], debounce_sec=30)
    det.process([_track(0.3, identity="EMP001")], ts=100.0)
    evs = det.process([_track(0.7, identity="EMP001")], ts=101.0)
    assert evs[0]["employee_id"] == "EMP001"


def test_no_lines_returns_empty():
    det = EntryExitDetector(lines=None, debounce_sec=30)
    assert det.process([_track(0.3)], ts=100.0) == []
    assert det.has_lines() is False


# ======================================================================
# Zone capacity (B5)
# ======================================================================

def test_zone_capacity_persisted_roundtrip(tmp_db):
    store = ZoneStore(tmp_db)
    zone = store.add(zone_name="server_room", camera="c1",
                     polygon=[[0, 0], [1, 0], [1, 1], [0, 1]], capacity=4)
    assert zone.capacity == 4
    assert store.get("server_room").capacity == 4
    assert store.list()[0].capacity == 4


def test_zone_capacity_defaults_to_none(tmp_db):
    store = ZoneStore(tmp_db)
    store.add(zone_name="plain", camera="c1")
    assert store.get("plain").capacity is None


def test_zone_capacity_db_level_upsert(tmp_db):
    store = ZoneStore(tmp_db)
    db.upsert_zone(tmp_db, {
        "zone_name": "db_zone", "camera": "c1",
        "polygon": "[[0,0],[1,0],[1,1],[0,1]]", "capacity": 7,
    })
    assert store.get("db_zone").capacity == 7


# ======================================================================
# Camera health (A12)
# ======================================================================

class _OpenCap:
    def isOpened(self):
        return True


def _fake_capture(**over):
    cap = VideoCapture.__new__(VideoCapture)
    cap._lock = threading.Lock()
    cap._running = True
    cap._cap = _OpenCap()
    cap._frame = np.zeros((10, 10, 3), dtype=np.uint8)
    cap._last_frame_ts = time.time()
    cap._reconnect_count = 0
    cap._frozen_streak = 0
    cap._frozen_since = 0.0
    cap._fps = 30.0
    for k, v in over.items():
        setattr(cap, k, v)
    return cap


def test_health_online_when_flowing():
    assert _fake_capture(_fps=30.0).health == CAM_ONLINE


def test_health_low_fps_when_slow_but_flowing():
    assert _fake_capture(_fps=2.0, _frozen_streak=0).health == CAM_LOW_FPS


def test_health_frozen_frame():
    cap = _fake_capture(_frozen_streak=30, _frozen_since=time.time() - 10,
                        _fps=30.0)
    assert cap.health == CAM_FROZEN


# ======================================================================
# Detector registry (B15)
# ======================================================================

def test_status_vocabulary_contains_phase_ab_statuses():
    vocab = DetectorRegistry.status_vocabulary()
    for status in ("AVAILABLE", "DISABLED", "DEGRADED",
                   "NOT_CONFIGURED", "FUTURE_MODEL_REQUIRED"):
        assert status in vocab["statuses"]


def test_system_capabilities_not_configured(tmp_db):
    reg = DetectorRegistry(tmp_db)
    reg.register_system_capabilities(cameras_available=False,
                                     entry_exit_lines=False,
                                     motion_enabled=False)
    assert reg.get("person_tracker")["status"] == "NOT_CONFIGURED"
    assert reg.get("motion_detector")["status"] == "DISABLED"
    assert reg.get("entry_exit_engine")["status"] == "NOT_CONFIGURED"


def test_system_capabilities_available(tmp_db):
    reg = DetectorRegistry(tmp_db)
    reg.register_system_capabilities(cameras_available=True,
                                     entry_exit_lines=True,
                                     motion_enabled=True)
    assert reg.get("person_tracker")["status"] == "AVAILABLE"
    assert reg.get("motion_detector")["status"] == "AVAILABLE"
    assert reg.get("entry_exit_engine")["status"] == "AVAILABLE"


def test_disable_sets_disabled_status(tmp_db):
    reg = DetectorRegistry(tmp_db)
    reg.register_system_capabilities(cameras_available=True,
                                     entry_exit_lines=True, motion_enabled=True)
    assert reg.disable("motion_detector") is True
    assert reg.get("motion_detector")["status"] == "DISABLED"
    assert reg.get("motion_detector")["enabled"] == 0


# ======================================================================
# New DB tables (A8/A10)
# ======================================================================

def test_motion_event_roundtrip(tmp_db):
    row_id = db.insert_motion_event(
        tmp_db, camera="c1", timestamp="2026-09-09 10:00:00",
        score=0.054, duration_seconds=1.2, bbox=(0.1, 0.2, 0.3, 0.4),
        event_id="sec-1")
    assert row_id > 0
    rows = db.query_motion_events(tmp_db, camera="c1")
    assert len(rows) == 1
    row = rows[0]
    assert row["event_id"] == "sec-1"
    assert row["score"] == pytest.approx(0.054)
    assert row["bbox_x1"] == pytest.approx(0.1)
    assert row["duration_seconds"] == pytest.approx(1.2)


def test_entry_exit_event_roundtrip(tmp_db):
    row_id = db.insert_entry_exit_event(
        tmp_db, camera="c1", timestamp="2026-09-09 10:30:00",
        event_type=ENTRY_CROSSING, line="main_door", direction="B_TO_A",
        track_id="c1#3", employee_id="EMP001", confidence=0.91,
        security_event_id=17)
    assert row_id > 0
    rows = db.query_entry_exit_events(tmp_db, track_id="c1#3")
    assert len(rows) == 1
    row = rows[0]
    assert row["event_type"] == ENTRY_CROSSING
    assert row["employee_id"] == "EMP001"
    assert row["security_event_id"] == 17


# ======================================================================
# SecurityEngine Phase A/B behaviours (B2/B6)
# ======================================================================

def _security_detection(**kw):
    det = {
        "cam": "cam_01",
        "emp_id": "EMP001",
        "face_box": (100, 100, 200, 200),
        "face_score": 0.95,
        "frame_width": 640,
        "frame_height": 480,
    }
    det.update(kw)
    return det


@pytest.fixture
def engine(tmp_db):
    eng = SecurityEngine(tmp_db)
    eng._cam_state.update("cam_01", True, now=0.0)
    return eng


def _intrusion_zones(engine):
    engine._zones.add(zone_name="server_room", camera="cam_01",
                      polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
                      allowed=["OTHER"], policy="INTRUSION")


def _intrusion_events(events):
    return [e for e in events if e.get("event_type") == "INTRUSION"]


def test_intrusion_fires_once_during_cooldown(engine):
    _intrusion_zones(engine)
    det = _security_detection(person_box=(300, 200, 400, 350))
    health = {"cam_01": {"health": "ONLINE"}}
    first = engine.tick([det], health)
    assert len(_intrusion_events(first)) == 1
    second = engine.tick([det], health)
    assert _intrusion_events(second) == []


def test_intrusion_refires_after_cooldown(engine, monkeypatch):
    _intrusion_zones(engine)
    det = _security_detection(emp_id="EMP009", person_box=(300, 200, 400, 350))
    health = {"cam_01": {"health": "ONLINE"}}
    engine.tick([det], health)
    engine.tick([det], health)
    monkeypatch.setattr(config, "INTRUSION_COOLDOWN_SEC", 0.0)
    third = engine.tick([det], health)
    assert len(_intrusion_events(third)) == 1


def test_process_event_assigns_event_id(engine):
    ev = engine._process_event({"event_type": "UNKNOWN_PRESENCE",
                                "camera": "cam_01", "confidence": 0.5,
                                "timestamp": "2026-09-09 10:00:00"})
    assert ev.get("event_id") is not None


def test_record_entry_exit_persists_and_emits_left(engine):
    result = engine.record_entry_exit({
        "event_type": EXIT_CROSSING,
        "camera": "cam_01", "line": "main_door", "direction": "A_TO_B",
        "track_id": "cam_01#5", "employee_id": "EMP001", "confidence": 0.9,
    })
    assert result is not None
    assert result["event_type"] == EXIT_CROSSING
    assert result.get("event_id") is not None
    assert engine._exit_left_at.get("EMP001") is not None
    lefts = engine._events.events(event_type=LEFT, limit=5)
    assert lefts


def test_left_is_debounced_after_exit_crossing(engine):
    engine.record_entry_exit({
        "event_type": EXIT_CROSSING,
        "camera": "cam_01", "line": "main_door", "direction": "A_TO_B",
        "track_id": "cam_01#5", "employee_id": "EMP001", "confidence": 0.9,
    })
    engine._present_prev = {"EMP001"}
    events = engine.tick([], {})
    assert [e for e in events if e.get("event_type") == "LEFT"] == []


def test_left_inferred_when_no_exit_crossing(engine):
    engine._present_prev = {"EMP001"}
    engine._exit_left_at["EMP001"] = 0.0
    events = engine.tick([], {})
    assert any(e.get("event_type") == LEFT and e.get("employee_id") == "EMP001"
               for e in events)


def test_left_inferred_fires_once_debounce_expires(engine, monkeypatch):
    engine.record_entry_exit({
        "event_type": EXIT_CROSSING,
        "camera": "cam_01", "line": "main_door", "direction": "A_TO_B",
        "track_id": "cam_01#5", "employee_id": "EMP001", "confidence": 0.9,
    })
    engine._present_prev = {"EMP001"}
    assert [e for e in engine.tick([], {}) if e.get("event_type") == "LEFT"] == []
    engine._present_prev = {"EMP001"}
    engine._exit_left_at["EMP001"] = time.time() - 60
    events = engine.tick([], {})
    assert any(e.get("event_type") == LEFT and e.get("employee_id") == "EMP001"
               for e in events)


# ----------------------------------------------------------------------
# UNEXPECTED_STAY (B6, opt-in)
# ----------------------------------------------------------------------

def _after_hours(monkeypatch):
    import src.security_engine as se
    monkeypatch.setattr(config, "UNEXPECTED_STAY_ENABLED", True)
    monkeypatch.setattr(se, "is_time_in_window",
                        lambda clock, start, end: False)


def test_unexpected_stay_emits_after_hours(engine, monkeypatch):
    _after_hours(monkeypatch)
    engine._crossing_arrivals["EMP001"] = time.time() - 3600
    events = engine.tick([], {})
    assert [e.get("event_type") for e in events
            if e.get("employee_id") == "EMP001"
            and e["event_type"] == UNEXPECTED_STAY]


def test_unexpected_stay_suppressed_inside_window(engine, monkeypatch):
    import src.security_engine as se
    monkeypatch.setattr(config, "UNEXPECTED_STAY_ENABLED", True)
    monkeypatch.setattr(se, "is_time_in_window",
                        lambda clock, start, end: True)
    engine._crossing_arrivals["EMP001"] = time.time() - 3600
    events = engine.tick([], {})
    assert UNEXPECTED_STAY not in [e.get("event_type") for e in events]


def test_unexpected_stay_disabled_by_default(engine):
    engine._crossing_arrivals["EMP001"] = time.time() - 3600
    events = engine.tick([], {})
    assert UNEXPECTED_STAY not in [e.get("event_type") for e in events]


def test_unexpected_stay_one_per_session(engine, monkeypatch):
    _after_hours(monkeypatch)
    engine._crossing_arrivals["EMP001"] = time.time() - 3600
    first = engine.tick([], {})
    assert UNEXPECTED_STAY in [e.get("event_type") for e in first]
    engine._present_prev = set()
    second = engine.tick([], {})
    assert UNEXPECTED_STAY not in [e.get("event_type") for e in second]


# ======================================================================
# dashboard / main wiring
# ======================================================================

def test_write_live_state_includes_ai_payload(tmp_path, monkeypatch):
    import main as m
    target = str(tmp_path / "live.json")
    monkeypatch.setattr(m, "_LIVE_STATE_FILE", target)
    m._write_live_state(
        {"EMP001": "PRESENT", "EMP002": "NOT_OBSERVED"},
        {"EMP001": {"ACTIVE": 5.0}},
        {"cam1": {"health": "ONLINE"}},
        fps=12.5,
        names={"EMP001": "Ada"},
        last_seen={"EMP001": "2026-09-09 09:00:00"},
        spatial={"count": 1, "tracks": [], "persons_by_camera": {"cam1": 1}},
        motion={"cam1": {"motion": False, "score": 0.0}},
        camera_selection={"mode": "AUTO", "source": "local"},
        ai_capabilities={"detectors": [
            {"detector_id": "motion", "name": "Motion", "status": "AVAILABLE"}
        ]},
    )
    with open(target, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["ai"]["spatial_tracks"]["count"] == 1
    assert data["ai"]["camera_selection"]["mode"] == "AUTO"
    assert data["ai"]["caps" if False else "capabilities"]["detectors"][0]["status"] == "AVAILABLE"
    assert data["employees"]["EMP001"]["name"] == "Ada"
    assert data["employees"]["EMP001"]["state"] == "PRESENT"
    assert data["employees"]["EMP002"]["state"] == "NOT_OBSERVED"


def test_registry_capabilities_handles_none(tmp_path, monkeypatch):
    import main as m
    assert m._registry_capabilities(None) is None


class _FakeRegistry:
    def list(self):
        return [
            {"detector_id": "person_tracker", "name": "Tracker",
             "status": "AVAILABLE", "status_detail": "ok", "enabled": 1,
             "event_types": "presence,tracking", "version": "ab"},
        ]


def test_registry_capabilities_payload():
    import main as m
    payload = m._registry_capabilities(_FakeRegistry())
    assert payload == {"detectors": [
        {"detector_id": "person_tracker", "name": "Tracker",
         "status": "AVAILABLE", "status_detail": "ok", "enabled": True,
         "event_types": "presence,tracking", "version": "ab"},
    ]}


def test_labels_for_skips_person_entries():
    import main as m
    cam_dets = [
        _person_det("c1", (10, 10, 40, 90)),
        _face_det("c1", "EMP001", (20, 30, 60, 80), (10, 10, 90, 90)),
    ]
    out = m._labels_for(cam_dets)
    assert len(out["face_boxes"]) == 1
    assert out["face_labels"] == [("EMP001", 0.9)]