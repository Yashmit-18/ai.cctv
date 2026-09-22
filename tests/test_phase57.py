"""Phase 57 regression tests -- employee-level identity margin (root cause).

Phase 56 real-world finding: EMP001 (6 enrolled embeddings) was always the
top face candidate (0.33-0.76) but NEVER CONFIRMED -- the margin gate compared
the top-1 and top-2 EMBEDDING ROWS, which were both EMP001's own enrollment
samples, producing margin 0.000-0.050 < FACE_MARGIN_MIN 0.06 even though the
identity was clear (the best OTHER employee scored far lower).

Phase 57 fix: rank EMPLOYEE identities, not embedding rows.  Each employee id
keeps its BEST embedding score (max aggregation); margin = best EMPLOYEE
score - second-best EMPLOYEE score.  Thresholds, candidate gate, candidate run
consistency, trusted-identity protection, Unknown behaviour, adoption,
productivity and camera outage semantics are all UNCHANGED -- only the margin
denominator is fixed.

No model/inference here -- synthetic unit-norm 512-D vectors with hand-picked
similarities, mirroring the Phase 44B determinism contract.
"""

import numpy as np
import pytest

import config
import main as m
import src.face_registry as fr
from src.camera_manager import MultiCameraManager
from src.domain import UNKNOWN_ID
from src.face_registry import (CANDIDATE_THRESHOLD, FaceRegistry,
                               RECOG_CANDIDATE, RECOG_CONFIRMED, RECOG_UNKNOWN,
                               RECOGNITION_THRESHOLD)
from src.tracker import MultiTracker, SpatialTracker, Track


# ======================================================================
# Fixtures (same seams as test_phase44b)
# ======================================================================

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
    _FakeTime.now = 10_000.0
    monkeypatch.setattr("src.tracker.time", _FakeTime)
    return _FakeTime


@pytest.fixture
def instant_commit(monkeypatch):
    monkeypatch.setattr("src.tracker.SMOOTHING_BUFFER_SEC", 0.0)
    monkeypatch.setattr("src.tracker.IDENTITY_STABILITY_FRAMES", 1)
    monkeypatch.setattr("src.tracker.AWAY_AFTER_SEC", 3.0)
    monkeypatch.setattr("src.tracker.PHONE_AFTER_SEC", 5.0)
    monkeypatch.setattr("src.tracker.PHONE_GAP_GRACE_SEC", 2.0)


@pytest.fixture
def recorder(monkeypatch):
    seen = []

    def fake_log(conn, ts, emp, state, dur, source=""):
        seen.append((ts, emp, state, dur, source))

    monkeypatch.setattr("src.tracker.log_interval", fake_log)
    return seen


# ======================================================================
# Build helpers (synthetic, deterministic, unit-norm 512-D)
# ======================================================================

def _q(*weights):
    """Query vector: q[i] = weight (only non-zero entries supplied)."""
    v = np.zeros(512, dtype=np.float32)
    for i, w in enumerate(weights):
        v[i] = float(w)
    return v


def _normalized(vec):
    v = np.asarray(vec, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def _reg(ids, emb_rows):
    """Bare FaceRegistry with id<->embedding pairing (multiple rows per id OK)."""
    r = FaceRegistry.__new__(FaceRegistry)
    r.threshold = RECOGNITION_THRESHOLD
    r._app = None
    r._employee_ids = list(ids)
    r._embeddings = np.asarray(emb_rows, dtype=np.float32)
    r._by_employee = {}
    for i, eid in enumerate(ids):
        r._by_employee.setdefault(eid, []).append(i)
    r._file_status = {}
    r._names = None
    return r


def _person(box, cam="cam1"):
    return {"cam": cam, "det_type": "person", "emp_id": "__person__",
            "person_box": tuple(box), "person_conf": 0.9,
            "frame_w": 640, "frame_h": 480, "phone": False}


def _face(box, emp, cam="cam1", person_box=None, score=0.8,
          status=None, cand_emp=None, cand_score=0.0, cand_margin=0.0):
    det = {"cam": cam, "emp_id": emp, "phone": False, "face_box": tuple(box),
           "face_score": score, "frame_w": 640, "frame_h": 480}
    if status is not None:
        det["recog_status"] = status
        if status == RECOG_CANDIDATE:
            det["recog_emp"] = cand_emp or emp
            det["recog_score"] = cand_score or score
            det["recog_margin"] = cand_margin
    if person_box is not None:
        det["person_box"] = tuple(person_box)
    return det


def _cycle(spatial, tracker, dets, *, camera_online=True):
    dets = list(dets)
    spatial.process(dets)
    claims = spatial.claims()
    if claims:
        dets = m._merge_person_claims(dets, claims)
    tracker.process_batch(dets, camera_online=camera_online)


def _make_track(adopt=2, switch=2, app_adopt=2):
    t = Track("t1", "cam1", (0, 0, 100, 200), 0.9, 640, 480, 0.0,
              adopt_frames=adopt, switch_frames=switch,
              appear_adopt_frames=app_adopt)
    return t


# ======================================================================
# TEST 1 -- same-employee duplicate embeddings must NOT deflate the margin
# ======================================================================

def test_57_same_employee_duplicates_do_not_deflate_margin():
    # EMP001 has two near-identical embeddings; EMP002 is clearly different.
    # Embedding-level top-2 would be EMP001(1.00) / EMP001(0.99) -> margin 0.01
    # -> never CONFIRMED.  Employee-level margin must be vs EMP002 instead.
    e0 = _q(1, 0, 0)
    e0b = _normalized(_q(0.99, 0.1, 0))          # same person, near-duplicate
    e2 = _q(0, 0, 1)                              # clearly different person
    reg = _reg(["EMP001", "EMP001", "EMP002"], [e0, e0b, e2])

    d = reg.identify_candidate(e0)                # EMP001-like query
    assert d["emp_id"] == "EMP001"
    # Employee-level runner-up must be EMP002 (a different identity), never the
    # near-identical second EMBEDDING of EMP001.
    assert d["second_id"] == "EMP002"
    assert d["margin"] == pytest.approx(1.0, abs=1e-3)   # 1.00 - employees max vs 0.0
    assert d["margin"] >= fr.MARGIN_MIN
    assert d["status"] == RECOG_CONFIRMED


def test_57_same_employee_duplicates_leave_score_and_identity_intact():
    # Even with a weaker 0.6x query the identity/margin semantics hold: the
    # best EMPLOYEE score is reported, duplicates do not leak into runner-up.
    e0 = _q(1, 0, 0)
    e0b = _normalized(_q(0.9, 0.2, 0))
    e2 = _q(0, 0, 1)
    reg = _reg(["EMP001", "EMP001", "EMP002"], [e0, e0b, e2])
    d = reg.identify_candidate(_normalized(_q(0.55, 0.15, 0.0)))
    assert d["emp_id"] == "EMP001"
    assert d["second_id"] == "EMP002"
    assert d["margin"] > fr.MARGIN_MIN


# ======================================================================
# TEST 2 -- EMP001 strong, EMP002 clearly lower -> correct employee margin
# ======================================================================

def test_57_clear_winner_employee_level_margin():
    # Query favours EMP001 at 0.80, EMP002 at 0.10 -> margin 0.70 -> CONFIRMED.
    e1 = _q(1, 0, 0)
    e2 = _q(0, 1, 0)
    reg = _reg(["EMP001", "EMP002"], [e1, e2])
    d = reg.identify_candidate(_normalized(_q(0.80, 0.10)))
    assert d["emp_id"] == "EMP001"
    assert d["second_id"] == "EMP002"
    assert d["score"] == pytest.approx(0.80 / np.linalg.norm(_q(0.80, 0.10)), abs=1e-3)
    expected_margin = (0.80 - 0.10) / np.linalg.norm(_q(0.80, 0.10))
    assert d["margin"] == pytest.approx(expected_margin, abs=1e-3)
    assert d["margin"] >= fr.MARGIN_MIN
    assert d["status"] == RECOG_CONFIRMED


def test_57_single_employee_gallery_full_score_margin():
    # Multiple EMP001 samples but NO other employee: margin = best score.
    e0 = _q(1, 0, 0)
    e0b = _normalized(_q(0.9, 0.2, 0))
    reg = _reg(["EMP001", "EMP001"], [e0, e0b])
    d = reg.identify_candidate(e0)
    assert d["emp_id"] == "EMP001"
    assert d["second_id"] is None
    assert d["margin"] == pytest.approx(1.0, abs=1e-3)
    assert d["status"] == RECOG_CONFIRMED


# ======================================================================
# TEST 3 -- genuinely ambiguous employees stay unconfirmed
# ======================================================================

def test_57_ambiguous_employees_stay_unconfirmed():
    # Two DIFFERENT employees within margin threshold: MUST remain CANDIDATE
    # even under employee-level semantics (both employees score ~0.60).
    e1 = _q(1, 0, 0)
    e2 = _normalized(_q(0.601, 0.799, 0.0))
    reg = _reg(["EMP004", "EMP003"], [e1, e2])
    q = _normalized(_q(0.61, 0.2921))
    d = reg.identify_candidate(q)
    assert d["emp_id"] == "EMP004"                    # best identity kept
    assert d["second_id"] == "EMP003"                 # a DIFFERENT employee
    assert d["status"] == RECOG_CANDIDATE             # NOT confirmed
    assert d["margin"] < fr.MARGIN_MIN


def test_57_identical_scores_two_employees_zero_margin():
    reg = _reg(["EMP003", "EMP004"], [_q(1, 0, 0), _q(0, 1, 0)])
    d = reg.identify_candidate(_q(0.85, 0.85))
    assert d["status"] == RECOG_CANDIDATE
    assert d["margin"] == pytest.approx(0.0)


# ======================================================================
# TEST 4 -- best below candidate floor -> no confirmation
# ======================================================================

def test_57_below_candidate_floor_is_unknown():
    e1 = _q(1, 0, 0)
    e2 = _q(0, 1, 0)
    reg = _reg(["EMP003", "EMP004"], [e1, e2])
    d = reg.identify_candidate(_q(0.30, 0.10))
    assert d["status"] == RECOG_UNKNOWN
    assert d["emp_id"] == "Unknown"
    assert d["margin"] == pytest.approx(0.0)


def test_57_empty_gallery_is_unknown():
    reg = _reg([], [])
    d = reg.identify_candidate(_q(1, 0, 0))
    assert d["status"] == RECOG_UNKNOWN
    assert d["emp_id"] == "Unknown"


# ======================================================================
# TEST 5 -- candidate run consistency remains intact
# ======================================================================

def test_57_candidate_adopts_resolved_track_after_consistency():
    # Candidate can only give an identity to a track that trusts nobody, after
    # adopt_frames consecutive runs -- unchanged from Phase 44B.
    t = _make_track(adopt=3, switch=3)
    for n in range(2):
        t.register_candidate_vote("EMP003", 0.54, float(n) + 1.0)
    assert t.identity in (None, UNKNOWN_ID)          # not yet adopted
    t.register_candidate_vote("EMP003", 0.55, 3.0)
    assert t.identity == "EMP003"
    assert t.identity_source == "candidate"


def test_57_candidate_promoted_to_face_authority_on_confirmed():
    t = _make_track(adopt=2, switch=2)
    for n in range(2):
        t.register_candidate_vote("EMP003", 0.55, float(n) + 1.0)
    assert t.identity_source == "candidate"
    t.register_vote("EMP003", 0.88, 10.0)
    assert t.identity == "EMP003"
    assert t.identity_source == "face"


# ======================================================================
# TEST 6 -- trusted identity cannot be demoted by weak appearance/ReID
# ======================================================================

def test_57_trusted_face_identity_never_demoted_by_appearance():
    t = _make_track(adopt=2, switch=2, app_adopt=2)
    for n in range(2):
        t.register_vote("EMP004", 0.92, float(n) + 1.0)
    assert t.identity == "EMP004"
    assert t.identity_source == "face"
    for n in range(10):
        t.register_appearance_vote("EMP003", 0.95, 100.0 + n)
    assert t.identity == "EMP004"
    assert t.identity_source == "face"


def test_57_candidates_never_override_trusted_identity():
    t = _make_track(adopt=2, switch=2)
    for n in range(2):
        t.register_vote("EMP004", 0.92, float(n) + 1.0)
    for n in range(8):
        t.register_candidate_vote("EMP003", 0.55, 100.0 + n)
    assert t.identity == "EMP004"
    assert t.identity_source == "face"


# ======================================================================
# TEST 7 -- Unknown stays Unknown when evidence is insufficient
# ======================================================================

def test_57_unknown_face_does_not_invent_identity(fake_clock, recorder,
                                                  instant_commit):
    st = SpatialTracker(adopt_frames=2, switch_frames=2)
    mt = MultiTracker(None)
    pbox = (100, 40, 300, 460)
    fbox = (150, 90, 250, 200)
    unk = _face(fbox, "Unknown", person_box=pbox, score=0.42,
                status=RECOG_UNKNOWN)
    for _ in range(3):
        _cycle(st, mt, [_person(pbox), unk])
        _FakeTime.now += 1.0
    assert st.claims() == {}                        # never an employee
    assert mt.employee_ids == []
    assert mt.live_states() == {}


# ======================================================================
# TEST 8 -- adoption behaviour remains intact
# ======================================================================

def test_57_confirmed_face_eventually_adopts(fake_clock):
    st = SpatialTracker(adopt_frames=3, switch_frames=3)
    pbox = (100, 40, 300, 460)
    fbox = (150, 90, 250, 200)
    face = _face(fbox, "EMP001", person_box=pbox, score=0.9,
                 status=RECOG_CONFIRMED)
    for i in range(3):
        st.process([_person(pbox), face], now=1000.0 + i)
    tracks = st.active_tracks(1000.0 + 2)
    assert tracks[0].identity == "EMP001"
    assert tracks[0].identity_source == "face"


# ======================================================================
# TEST 9 -- productivity semantics unchanged
# ======================================================================

def test_57_active_through_face_and_body(fake_clock, recorder, instant_commit):
    st = SpatialTracker(adopt_frames=2, switch_frames=2)
    mt = MultiTracker(None)
    pbox = (100, 40, 300, 460)
    fbox = (150, 90, 250, 200)
    face = _face(fbox, "EMP004", person_box=pbox, score=0.9,
                 status=RECOG_CONFIRMED)
    _cycle(st, mt, [_person(pbox), face])
    _FakeTime.now += 1.0
    _cycle(st, mt, [_person(pbox), face])
    assert mt.live_state("EMP004") == "ACTIVE"
    # EMP001 still balances productivity semantic: a CONFIRMED face -> ACTIVE,
    # then the employee leaves -> wall-clock AWAY after grace.
    face1 = _face(fbox, "EMP001", person_box=pbox, score=0.9,
                  status=RECOG_CONFIRMED)
    _cycle(st, mt, [_person(pbox), face1])
    _FakeTime.now += 1.0
    _cycle(st, mt, [_person(pbox), face1])
    assert mt.live_state("EMP001") == "ACTIVE"


# ======================================================================
# TEST 10 -- camera offline / frozen behaviour unchanged
# ======================================================================

def test_57_camera_offline_freezes_state_never_away(fake_clock, recorder):
    mt = MultiTracker(None)
    mt.process_batch([{"emp_id": "EMP001", "present": True}], camera_online=True)
    _FakeTime.now += 90
    mt.process_batch([], camera_online=False)        # camera dies
    assert mt.live_state("EMP001") == "ACTIVE"       # frozen, never AWAY
    assert recorder == []                            # no AWAY interval logged


def test_57_camera_frozen_still_keeps_presence(fake_clock, recorder):
    mt = MultiTracker(None)
    mt.process_batch([{"emp_id": "EMP001", "present": True}], camera_online=True)
    _FakeTime.now += 30
    mt.process_batch([{"emp_id": "EMP001", "present": True}],
                     camera_online=True)             # FROZEN but usable frame
    assert mt.live_state("EMP001") == "ACTIVE"


def test_57_offline_from_start_creates_no_fake_away(fake_clock, recorder):
    mt = MultiTracker(None)
    mt.process_batch([], camera_online=False)
    assert mt.employee_ids == []
    assert recorder == []