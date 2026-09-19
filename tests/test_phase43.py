"""Phase 43 regression tests -- person side/back pose + phone attribution +
face-cadence + camera-lag diagnostics.

The Phase 43 audit found three root causes, all fixed in this phase:

1. PART B -- identity-coupled presence: the employee FSM only saw employees
   through *recognised faces*.  A side/back pose (no face) turned an employee
   ``present: False`` even though YOLO was detecting their body, so they went
   AWAY within ``AWAY_AFTER_SEC``.  Fix: ``SpatialTracker``'s fresh per-track
   claims (adopted identity + this-cycle body box) are bridged into the FSM
   (``main._merge_person_claims``) -- person != face, presence from the body.
2. PART D -- phone attribution was face-anchored (``_phone_near_face``).  A
   phone held at the body/side was lost for the employee.  Fix: the bridge
   attributes phone from the ONE spatial track that holds that identity
   (strictly per person, never a global flag).
3. PART G -- InsightFace ran every frame per camera (lag).  Fix: per-camera
   ``FACE_DETECT_CADENCE``; YOLO person+phone still runs every cycle.

All assertions use a fake wall clock (no wall-clock sleeps).  No models, no
network, no webcams.
"""

import time as _realtime  # noqa: F401  (module seam parity with tracker tests)

import numpy as np
import pytest

import main as m
import src.tracker as T
from src.tracker import MultiTracker, SpatialTracker


# ======================================================================
# Fixtures (same seams as test_phase41 / test_multiperson_pipeline)
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


@pytest.fixture
def tmp_db(tmp_path):
    from src.database import get_connection, init_db
    conn = get_connection(tmp_path / "t.db")
    init_db(conn)
    yield conn
    conn.close()


# ======================================================================
# Detection-dict builders (mirror detector._process_frame cam_id output)
# ======================================================================

def _person(box, cam="cam1"):
    return {"cam": cam, "det_type": "person", "emp_id": "__person__",
            "person_box": tuple(box), "person_conf": 0.9,
            "frame_w": 640, "frame_h": 480}


def _face(box, emp, cam="cam1", person_box=None, score=0.8):
    det = {"cam": cam, "emp_id": emp, "phone": False, "face_box": tuple(box),
           "face_score": score, "frame_w": 640, "frame_h": 480}
    if person_box is not None:
        det["person_box"] = tuple(person_box)
    return det


def _person_claim(emp, phone=False, cam="cam1"):
    return {"cam": cam, "emp_id": emp, "present": True,
            "phone": phone, "spatial_fallback": True}


def _cycle(spatial, tracker, dets, *, camera_online=True):
    """One daemon-loop iteration: spatial -> claims -> merge -> FSM."""
    dets = list(dets)
    spatial.process(dets)
    claims = spatial.claims()
    if claims:
        dets = m._merge_person_claims(dets, claims)
    tracker.process_batch(dets, camera_online=camera_online)


# ======================================================================
# PART B -- person != face: side/back pose never fakes AWAY
# ======================================================================

def _setup_spatial(adopt=1):
    return SpatialTracker(adopt_frames=adopt, switch_frames=adopt)


def test_side_pose_person_stays_active(fake_clock, recorder, instant_commit):
    """A person whose face is hidden (side/back) keeps EMP001 ACTIVE.

    The body box (YOLO person detection) and the spatial track's adopted
    identity -- not the face -- drive presence.  This is the primary
    regression for the report.
    """
    st = _setup_spatial()
    mt = MultiTracker(None)
    pbox = (100, 40, 300, 460)
    fbox = (150, 90, 250, 200)
    # Cycle 1: face visible -> identity adopted, FSM ACTIVE.
    _cycle(st, mt, [_person(pbox), _face(fbox, "EMP001", person_box=pbox)])
    assert mt.live_state("EMP001") == "ACTIVE"
    # Cycles 2..6: side/back pose -- body only, no face at all.
    for _ in range(5):
        _FakeTime.now += 1.0
        _cycle(st, mt, [_person(pbox)])
        assert mt.live_state("EMP001") == "ACTIVE", \
            "side/back pose must NOT go AWAY"
    assert recorder == []  # no ACTIVE->AWAY interval was ever logged


def test_person_leaving_with_claims_eventually_away(fake_clock, recorder,
                                                    instant_commit):
    """Claims stop the moment the body leaves -> absence recomputes normally."""
    st = _setup_spatial()
    mt = MultiTracker(None)
    pbox = (100, 40, 300, 460)
    fbox = (150, 90, 250, 200)
    _cycle(st, mt, [_person(pbox), _face(fbox, "EMP001", person_box=pbox)])
    _FakeTime.now += 1.0
    _cycle(st, mt, [_person(pbox)])
    assert mt.live_state("EMP001") == "ACTIVE"
    # EMP001's body gone; a different (unknown) person remains on camera, so
    # detections are non-empty and the FSM is not frozen.
    other = (400, 50, 600, 400)
    _FakeTime.now += 3.1
    _cycle(st, mt, [_person(other)])
    assert mt.live_state("EMP001") == "AWAY"
    assert recorder and recorder[-1][1] == "EMP001"
    assert recorder[-1][2] == "ACTIVE"


def test_person_detection_is_not_face_recognition(fake_clock, recorder,
                                                  instant_commit):
    """A plain body box without a face builds NO employee identity."""
    st = _setup_spatial()
    mt = MultiTracker(None)
    for _ in range(3):
        _cycle(st, mt, [_person((100, 40, 300, 460))])
        _FakeTime.now += 1.0
    assert st.claims() == {}                 # nothing adopted -> no claims
    assert mt.employee_ids == []             # never an employee row
    assert mt.live_states() == {}


def test_never_identified_person_stays_unknown_numbered(fake_clock, recorder,
                                                        instant_commit):
    """Unknown stays Unknown (existing numbering), no identity invented."""
    st = SpatialTracker(adopt_frames=2, switch_frames=2)
    p1 = (0, 40, 200, 460)
    p2 = (300, 40, 500, 460)
    uf = _face((60, 90, 140, 200), "Unknown", person_box=p1, score=0.44)
    for _ in range(3):
        st.process([_person(p1), _person(p2), uf], now=1000.0 + _)
    snap = st.snapshot(1000.0 + 2)
    labels = {t["identity_label"] for t in snap["tracks"]}
    assert labels <= {"Unknown", "Unknown 1", "Unknown 2"}
    assert st.claims() == {}                 # unresolved => never claimed


def test_face_loss_never_demotes_identity(fake_clock, recorder):
    """Adopted identity survives later Unknown-face votes (anti-flicker)."""
    st = SpatialTracker(adopt_frames=2, switch_frames=2)
    pbox = (100, 40, 300, 460)
    fbox = (150, 90, 250, 200)
    st.process([_person(pbox), _face(fbox, "EMP001", person_box=pbox)], now=1000.0)
    st.process([_person(pbox), _face(fbox, "EMP001", person_box=pbox)], now=1000.1)
    assert st.active_tracks(1000.1)[0].identity == "EMP001"
    unk = _face(fbox, "Unknown", person_box=pbox, score=0.42)
    for _ in range(4):
        st.process([_person(pbox), unk], now=1100.0 + _)
    tracks = st.active_tracks(1100.0 + 3)
    assert tracks[0].identity == "EMP001"    # identity never demoted
    assert st.claims() == {"EMP001": {"cam": "cam1", "phone": False}}


# ======================================================================
# PART D -- phone attribution is strictly per person
# ======================================================================

def test_per_person_phone_claim_only_holder(fake_clock, recorder,
                                            instant_commit):
    """Phone near EMP002's body puts EMP002 ON_PHONE, not EMP001."""
    st = _setup_spatial()
    mt = MultiTracker(tmp_db := None)
    pa, pb = (0, 40, 300, 460), (340, 40, 620, 460)
    fa, fb = (60, 90, 160, 210), (400, 90, 500, 210)
    _cycle(st, mt, [_person(pa), _person(pb),
                    _face(fa, "EMP001", person_box=pa),
                    _face(fb, "EMP002", person_box=pb)])
    assert st.claims() == {
        "EMP001": {"cam": "cam1", "phone": False},
        "EMP002": {"cam": "cam1", "phone": False},
    }
    # Same scene but a phone now sits near person B's body (side pose, faces
    # hidden).  Only EMP002's track carries the phone.
    pb_phone = _person(pb)
    pb_phone["phone"] = True
    pb_phone["phone_box"] = (500, 300, 570, 380)
    for _ in range(7):
        _FakeTime.now += 1.0
        _cycle(st, mt, [_person(pa), pb_phone])
    assert mt.live_state("EMP002") == "ON_PHONE"
    assert mt.live_state("EMP001") == "ACTIVE"


def test_unrelated_employee_unaffected_by_phone(fake_clock, recorder,
                                                instant_commit):
    """EMP001 stays ACTIVE while EMP002 is on the phone elsewhere."""
    st = _setup_spatial()
    mt = MultiTracker(None)
    pa, pb = (0, 40, 300, 460), (340, 40, 620, 460)
    fa, fb = (60, 90, 160, 210), (400, 90, 500, 210)
    _cycle(st, mt, [_person(pa), _person(pb),
                    _face(fa, "EMP001", person_box=pa),
                    _face(fb, "EMP002", person_box=pb)])
    for _ in range(7):
        _FakeTime.now += 1.0
        pd = _person(pb)
        pd["phone"] = True
        _cycle(st, mt, [_person(pa), pd])
    assert mt.live_state("EMP002") == "ON_PHONE"
    assert mt.live_state("EMP001") == "ACTIVE"


# ======================================================================
# PART C/E -- phone via the spatial bridge still honours the FSM rules
# ======================================================================

def test_side_pose_phone_triggers_on_phone(fake_clock, tmp_db, instant_commit):
    """Side-facing person holding a phone (no face at all) -> ON_PHONE >5s."""
    from src.database import query_security_events
    st = _setup_spatial()
    mt = MultiTracker(tmp_db)
    pbox = (100, 40, 300, 460)
    fbox = (150, 90, 250, 200)
    _cycle(st, mt, [_person(pbox), _face(fbox, "EMP001", person_box=pbox)])
    _FakeTime.now += 1.0
    for _ in range(6):
        _FakeTime.now += 1.0
        pd = _person(pbox)
        pd["phone"] = True
        _cycle(st, mt, [pd])
    assert mt.live_state("EMP001") == "ON_PHONE"
    events = query_security_events(tmp_db, event_type="PHONE_USE")
    assert len(events) == 1
    assert events[0]["employee_id"] == "EMP001"


def test_side_pose_phone_under_threshold_stays_active(fake_clock, recorder,
                                                      instant_commit):
    st = _setup_spatial()
    mt = MultiTracker(None)
    pbox, fbox = (100, 40, 300, 460), (150, 90, 250, 200)
    _cycle(st, mt, [_person(pbox), _face(fbox, "EMP001", person_box=pbox)])
    _FakeTime.now += 1.0
    for _ in range(4):
        _FakeTime.now += 1.0
        pd = _person(pbox)
        pd["phone"] = True
        _cycle(st, mt, [pd])
    assert mt.live_state("EMP001") == "ACTIVE"
    assert recorder == []


def test_claim_phone_one_alert_per_episode(fake_clock, tmp_db, instant_commit):
    from src.database import query_security_events
    st = _setup_spatial()
    mt = MultiTracker(tmp_db)
    pbox, fbox = (100, 40, 300, 460), (150, 90, 250, 200)
    _cycle(st, mt, [_person(pbox), _face(fbox, "EMP001", person_box=pbox)])
    _FakeTime.now += 1.0
    for _ in range(20):                      # phone use continues after alert
        _FakeTime.now += 1.0
        pd = _person(pbox)
        pd["phone"] = True
        _cycle(st, mt, [pd])
    assert mt.live_state("EMP001") == "ON_PHONE"
    assert len(query_security_events(tmp_db, event_type="PHONE_USE")) == 1


def test_claim_phone_new_episode_rearms(fake_clock, tmp_db, instant_commit):
    from src.database import query_security_events
    st = _setup_spatial()
    mt = MultiTracker(tmp_db)
    pbox, fbox = (100, 40, 300, 460), (150, 90, 250, 200)
    _cycle(st, mt, [_person(pbox), _face(fbox, "EMP001", person_box=pbox)])
    _FakeTime.now += 1.0
    for _ in range(6):                       # episode 1 -> ON_PHONE + alert 1
        _FakeTime.now += 1.0
        pd = _person(pbox)
        pd["phone"] = True
        _cycle(st, mt, [pd])
    assert mt.live_state("EMP001") == "ON_PHONE"
    _FakeTime.now += 2.1                     # phone put down > gap grace
    _cycle(st, mt, [_person(pbox)])          # recovery -> ACTIVE
    _FakeTime.now += 0.1
    _cycle(st, mt, [_person(pbox)])
    assert mt.live_state("EMP001") == "ACTIVE"
    _FakeTime.now += 1.0
    for _ in range(6):                       # episode 2 -> alert 2
        _FakeTime.now += 1.0
        pd = _person(pbox)
        pd["phone"] = True
        _cycle(st, mt, [pd])
    assert mt.live_state("EMP001") == "ON_PHONE"
    assert len(query_security_events(tmp_db, event_type="PHONE_USE")) == 2


def test_claim_presence_wall_clock_not_frame_count(fake_clock, recorder,
                                                   instant_commit):
    """Irregular frame gaps accumulate wall seconds for claim-based phone."""
    st = _setup_spatial()
    mt = MultiTracker(None)
    pbox, fbox = (100, 40, 300, 460), (150, 90, 250, 200)
    _cycle(st, mt, [_person(pbox), _face(fbox, "EMP001", person_box=pbox)])
    _FakeTime.now += 1.0
    gaps = [0.3, 0.4, 0.6, 0.5, 0.7, 0.5, 0.5, 1.0, 0.7, 0.5]
    for g in gaps:
        _FakeTime.now += g
        pd = _person(pbox)
        pd["phone"] = True
        _cycle(st, mt, [pd])
    assert mt.live_state("EMP001") == "ON_PHONE"  # sum of gaps > 5s


# ======================================================================
# PART H -- offline/freeze semantics with claims present
# ======================================================================

def test_claims_never_act_during_offline(fake_clock, recorder, instant_commit):
    """camera_online=False freezes the FSM even if claims are merged in."""
    st = _setup_spatial()
    mt = MultiTracker(None)
    pbox, fbox = (100, 40, 300, 460), (150, 90, 250, 200)
    _cycle(st, mt, [_person(pbox), _face(fbox, "EMP001", person_box=pbox)])
    _FakeTime.now += 90
    pd = _person(pbox)
    pd["phone"] = True
    _cycle(st, mt, [pd], camera_online=False)   # dead camera, claims present
    assert mt.live_state("EMP001") == "ACTIVE"   # frozen, never AWAY
    assert recorder == []                        # no fabricated AWAY/phone
    # Recovery restarts absence from a clean slate.
    _FakeTime.now += 1
    _cycle(st, mt, [_person(pbox)], camera_online=True)
    _FakeTime.now += 60
    _cycle(st, mt, [], camera_online=True)       # no dets => freeze, not AWAY
    assert mt.live_state("EMP001") == "ACTIVE"


def test_claim_produces_intervals_but_unknown_never_logged(fake_clock,
                                                           recorder,
                                                           instant_commit):
    """Claim-driven ACTIVE/ON_PHONE logs rows for EMP001 only."""
    st = SpatialTracker(adopt_frames=1, switch_frames=1)
    mt = MultiTracker(None)
    pbox, fbox = (100, 40, 300, 460), (150, 90, 250, 200)
    _cycle(st, mt, [_person(pbox), _face(fbox, "EMP001", person_box=pbox)])
    _FakeTime.now += 1.0
    for _ in range(6):
        _FakeTime.now += 1.0
        pd = _person(pbox)
        pd["phone"] = True
        _cycle(st, mt, [pd])
    assert recorder
    assert {r[1] for r in recorder} == {"EMP001"}
    assert {r[2] for r in recorder} <= {"ACTIVE", "ON_PHONE"}


# ======================================================================
# PART G -- face recognition cadence (lag fix) + skip-frames behaviour
# ======================================================================

class _Box:
    def __init__(self, cls, xyxy):
        self.cls = np.array(cls)
        self.xyxy = np.array([xyxy], dtype=float)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _Model:
    def __init__(self, phones: bool):
        self._phones = phones

    def predict(self, frames, **kwargs):
        out = []
        for _ in (frames if isinstance(frames, list) else [frames]):
            boxes = [_Box(0, (30, 30, 90, 90))]
            if self._phones:
                boxes.append(_Box(67, (180, 180, 210, 210)))
            out.append(_Result(boxes))
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
        f = _FakeFace()
        return [f]


class _FakeFace:
    bbox = np.array([120, 60, 220, 160], dtype=float)
    normed_embedding = np.zeros(512, dtype=np.float32)


def _det_stub(face_cadence=1, diag=False, phones=False):
    from src.detector import ActivityDetector
    det = ActivityDetector.__new__(ActivityDetector)
    det.conf = 0.3
    det._device = "cpu"
    det._half = False
    det.model = _Model(phones=phones)
    det._face_registry = _CountingReg()
    det._face_cadence = face_cadence
    det._face_ticks = {}
    det._phone_diag_on = diag
    det._diag_cycles = 0
    det._phone_stats = {}
    return det


def test_face_cadence_skips_recognition_frames():
    det = _det_stub(face_cadence=3)
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    for i in range(6):
        det.detect_batch({"cam1": frame})
        expected = 1 if i % 3 == 0 else 0  # ticks on cycle 1, 4
        assert det._face_registry.app.calls == sum(
            1 for j in range(i + 1) if j % 3 == 0), f"cycle {i + 1}"
    # Person presence is ALWAYS emitted, even on skip frames.
    det4 = _det_stub(face_cadence=3)
    det4.detect_batch({"cam1": frame})
    det4.detect_batch({"cam1": frame})
    det4.detect_batch({"cam1": frame})       # skip cycle
    nope = det4.detect_batch({"cam1": frame})
    presence = [d for d in nope if "person_present" in d]
    assert presence and presence[0]["person_present"] is True


def test_face_cadence_one_runs_faces_every_cycle():
    det = _det_stub(face_cadence=1)
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    det.detect_batch({"cam1": frame})
    det.detect_batch({"cam1": frame})
    assert det._face_registry.app.calls == 2


def test_face_reappears_on_next_tick_keeps_employee_present(fake_clock):
    """A cadence skip re-emits no face; body-driven claims hold the employee
    ACTIVE anyway; the next tick brings the known face back."""
    det = _det_stub(face_cadence=2)
    frame = np.zeros((360, 320, 3), dtype=np.uint8)
    st = _setup_spatial()
    mt = MultiTracker(None)
    _FakeTime.now = 10_000.0
    # Cycle 1 (cadence tick): the known face is emitted -> identity adopted.
    out1 = det.detect_batch({"cam1": frame})
    assert any(d.get("emp_id") == "EMP001" for d in out1)
    _cycle(st, mt, out1)
    assert mt.live_state("EMP001") == "ACTIVE"
    # Cycle 2 (skip): no face entries, but the bridge keeps the employee.
    _FakeTime.now += 0.5
    out2 = det.detect_batch({"cam1": frame})
    assert not any(d.get("face_box") is not None for d in out2)
    _cycle(st, mt, out2)
    assert mt.live_state("EMP001") == "ACTIVE"
    # Cycle 3 (tick): the known face reappears.
    _FakeTime.now += 0.5
    out3 = det.detect_batch({"cam1": frame})
    assert any(d.get("emp_id") == "EMP001" for d in out3)


# ======================================================================
# PART C -- phone diagnostics snapshot
# ======================================================================

def test_phone_diagnostics_snapshot_counts():
    det = _det_stub(diag=True, phones=True)
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    det.detect_batch({"cam1": frame})
    det.detect_batch({"cam1": frame})
    snap = det.diagnostics_snapshot()
    cam = snap["cameras"]["cam1"]
    assert snap["enabled"] is True
    assert cam["cycles"] == 2
    assert cam["person_dets"] == 2
    assert cam["phone_dets"] == 2
    assert cam["phone_conf_mean"] is not None
    assert cam["matched_to_person"] >= 0


def test_phone_diagnostics_disabled_by_default():
    det = _det_stub(diag=False)
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    det.detect_batch({"cam1": frame})
    assert det.diagnostics_snapshot() == {"enabled": False, "face_cadence": 1}


# ======================================================================
# PART B/D -- main._merge_person_claims unit behaviour
# ======================================================================

def test_merge_person_claims_is_additive_and_per_person():
    base = [_person((100, 40, 300, 460)), _person((340, 40, 620, 460))]
    merged = m._merge_person_claims(base, {
        "EMP001": {"cam": "cam1", "phone": False},
        "EMP002": {"cam": "cam1", "phone": True},
    })
    claims = [d for d in merged if d.get("spatial_fallback")]
    assert len(claims) == 2
    by_id = {d["emp_id"]: d for d in claims}
    assert by_id["EMP001"]["phone"] is False
    assert by_id["EMP002"]["phone"] is True
    # face-derived entries and __person__ entries are untouched.
    assert len([d for d in merged if d.get("det_type") == "person"]) == 2


def test_merge_person_claims_empty_is_identity():
    dets = [_person((100, 40, 300, 460))]
    assert m._merge_person_claims(dets, {}) is dets