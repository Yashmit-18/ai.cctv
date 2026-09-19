"""Phase 44B regression tests -- identity cross-mapping fix + candidate gate.

Phase 44B fixes the real-world bug "Vishal sometimes displays/outputs as
Piyush".  The enrolled gallery is provably well-separated (real diagnostic run:
every employee self-matches at 1.0000, worst cross-identity similarity 0.1961
EMP002<->EMP003, no pair above the confusability ceiling) -- so the failure was
in the LIVE inference path: InsightFace only returned the argmax, a single
0.60 hard cutoff with NO margin check, and the tracker's run-voting let weak
0.6x observations of a distorted side/partial pose switch a trusted identity.

The fix layers a two-tier gate and a candidate channel:

1. **CONFIRMED** -- score >= 0.60 AND margin (best minus runner-up identity)
   >= 0.06.  This is the only thing that may adopt or switch a track identity.
2. **CANDIDATE**  -- score >= 0.50 but not CONFIRMED.  A candidate may only
   give an identity to a track that trusts NOBODY yet, after consecutive
   observations (temporal consistency); it can never demote, never switch and
   never override a trusted identity.
3. **UNKNOWN**    -- below the candidate floor (no recognition evidence).

Side/back-facing behaviour: person presence (YOLO body + spatial track) drives
presence, faces only drive identity, and face loss never demotes -- so a
back-facing employee stays ACTIVE and never fakes AWAY.  Performance:
face recognition stays cadence-thinned (default 5) while the person layer
keeps delivering presence every cycle; the camera pipeline serves the latest
frame only (no stale backlog); absence timing is wall-clock.

All assertions deterministic -- fake insightface, synthetic embeddings with
hand-picked similarities, a fake wall clock.  No real inference / network.
"""

import os
import sys
import time as _realtime  # noqa: F401  (module seam parity with tracker tests)
import types  # noqa: F401

import numpy as np
import pytest

import config
import main as m
import src.face_registry as fr
from src.camera_manager import MultiCameraManager
from src.detector import ActivityDetector
from src.domain import UNKNOWN_ID
from src.face_registry import (CANDIDATE_THRESHOLD, FaceRegistry,
                               RECOG_CANDIDATE, RECOG_CONFIRMED, RECOG_UNKNOWN,
                               RECOGNITION_THRESHOLD)
from src.tracker import MultiTracker, SpatialTracker, Track


# ======================================================================
# Fixtures (same seams as test_phase43/44)
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
# Synthetic registry / detection build helpers
# ======================================================================

def _ortho_basis(n):
    """n mutually-orthogonal unit embedding vectors (deterministic)."""
    return np.eye(512, dtype=np.float32)[:n].copy()


def _reg(ids, emb_rows):
    """Bare FaceRegistry with the given id<->embedding pairing (no model)."""
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


def _q(*weights):
    v = np.zeros(512, dtype=np.float32)
    for i, w in enumerate(weights):
        v[i] = float(w)
    return v


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


def _det_stub(face_cadence=5):
    det = ActivityDetector.__new__(ActivityDetector)
    det._face_cadence = face_cadence
    det._face_ticks = {}
    return det


# ======================================================================
# PART 1 -- identity / registry mapping (the image-level truth)
# ======================================================================

def test_44b_config_defaults_and_gate_shape():
    # Candidate is 0.50, confirmation stays 0.60, margin floor 0.06.
    assert config.FACE_CANDIDATE_THRESHOLD == pytest.approx(0.50)
    assert config.FACE_SIMILARITY_THRESHOLD == pytest.approx(0.60)
    assert config.FACE_MARGIN_MIN == pytest.approx(0.06)
    assert 0.0 <= CANDIDATE_THRESHOLD < RECOGNITION_THRESHOLD <= 1.0
    assert fr.MARGIN_MIN >= 0.0


def test_44b_self_match_is_confirmed_for_each_enrolled_employee():
    basis = _ortho_basis(4)
    reg = _reg(["EMP001", "EMP002", "EMP003", "EMP004"], basis)
    for i, eid in enumerate(["EMP001", "EMP002", "EMP003", "EMP004"]):
        dec = reg.identify_candidate(basis[i])
        assert dec["emp_id"] == eid
        assert dec["status"] == RECOG_CONFIRMED
        assert dec["score"] == pytest.approx(1.0)


def test_44b_vishal_piyush_mapping_is_identity_not_order():
    # EMP004 is Vishal, EMP003 is Piyush.  A Vishal-like query must map to
    # EMP004 regardless of gallery row order.
    basis = _ortho_basis(4)
    vishal = basis[3] * 0.9 + basis[0] * 0.1
    reg1 = _reg(["EMP001", "EMP002", "EMP003", "EMP004"], basis)
    reg2 = _reg(["EMP004", "EMP001", "EMP003", "EMP002"],
                basis[[3, 0, 2, 1]])
    d1 = reg1.identify_candidate(vishal)
    d2 = reg2.identify_candidate(vishal)
    assert d1["emp_id"] == "EMP004"
    assert d2["emp_id"] == "EMP004"
    assert d1["margin"] == pytest.approx(d2["margin"], abs=1e-6)
    assert d1["status"] == d2["status"] == RECOG_CONFIRMED


def test_44b_below_candidate_floor_is_unknown():
    basis = _ortho_basis(2)
    reg = _reg(["EMP003", "EMP004"], basis)
    dec = reg.identify_candidate(basis[0] * 0.3 + basis[1] * 0.1)
    assert dec["status"] == RECOG_UNKNOWN
    assert dec["emp_id"] == "Unknown"
    assert dec["margin"] == pytest.approx(0.0)


def test_44b_candidate_band_is_candidate_not_confirmed():
    # 0.55 >= candidate floor but < 0.60: tentative, NEVER a recognition.
    basis = _ortho_basis(2)
    reg = _reg(["EMP003", "EMP004"], basis)
    dec = reg.identify_candidate(basis[0] * 0.55 + basis[1] * 0.4)
    assert dec["emp_id"] == "EMP003"
    assert dec["status"] == RECOG_CANDIDATE
    assert dec["score"] == pytest.approx(0.55, abs=1e-4)


def test_44b_ambiguous_top_two_never_confirmed():
    # THE ROOT CAUSE GUARD: EMP004-like at 0.61 is > old cutoff, yet the
    # runner-up EMP003 is only 0.01 behind -- the gallery cannot separate this
    # frame, so this must NOT become a winner.  Old code: EMP004 would be
    # argmax >= 0.60 -> wrong on distorted frames.
    rows = np.array([[1.0, 0.0], [0.601, 0.799]], dtype=np.float32)
    reg = _reg(["EMP004", "EMP003"], np.pad(rows, ((0, 0), (0, 510))))
    q = _q(0.61, 0.2921)   # dot(EMP004)=0.61, dot(EMP003)=0.60
    dec = reg.identify_candidate(q)
    assert dec["emp_id"] == "EMP004"                  # best named (identity kept)
    assert dec["second_id"] == "EMP003"
    assert dec["status"] == RECOG_CANDIDATE           # but NOT confirmed
    assert dec["margin"] == pytest.approx(0.01, abs=1e-3)
    assert dec["margin"] < fr.MARGIN_MIN


def test_44b_confirmed_requires_clean_margin():
    reg = _reg(["EMP003", "EMP004"], _ortho_basis(2))
    # clear winner: 0.85 vs 0.05 -> CONFIRMED
    d = reg.identify_candidate(_q(0.85, 0.05))
    assert d["status"] == RECOG_CONFIRMED and d["emp_id"] == "EMP003"
    # identical scores -> zero margin -> NOT confirmed even though both >= 0.6
    d2 = reg.identify_candidate(_q(0.85, 0.85))
    assert d2["status"] == RECOG_CANDIDATE
    assert d2["margin"] == pytest.approx(0.0)


def test_44b_single_enroll_keeps_full_score_margin():
    # A one-gallery-entry registry: margin defaults to the best score, so a
    # strong single candidate still reaches CONFIRMED.
    reg = _reg(["EMP004"], np.eye(512, dtype=np.float32)[:1])
    d = reg.identify_candidate(np.eye(512, dtype=np.float32)[0])
    assert d["emp_id"] == "EMP004"
    assert d["status"] == RECOG_CONFIRMED
    assert d["second_id"] is None


def test_44b_employee_name_comes_from_db(tmp_path):
    reg = _reg(["EMP004"], np.eye(512, dtype=np.float32)[:1])
    assert reg.employee_name("EMP004") == "Vishal"
    assert reg.employee_name("EMP999") == ""


def test_44b_cache_pairing_guard_detects_corruption(monkeypatch, tmp_path):
    import pickle
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    emb_file = tmp_path / "embeddings.pkl"
    monkeypatch.setattr(fr, "FACES_DIR", faces_dir)
    monkeypatch.setattr(fr, "EMBEDDINGS_FILE", emb_file)
    # ids and rows out of sync (2 rows but 1 id) -> must refuse to serve it.
    emb_file.write_bytes(pickle.dumps({
        "version": fr.CACHE_VERSION,
        "employee_ids": ["EMP001"],
        "embeddings": [[0.1] * 512, [0.2] * 512],
        "by_employee": {"EMP001": [0, 1]},
        "file_status": {},
        "file_fingerprint": {},
    }))
    reg = FaceRegistry.__new__(FaceRegistry)
    reg.threshold = 0.6
    with pytest.raises(ValueError, match="pairing"):
        reg._load_cache()


# ======================================================================
# PART 2 -- match safety: candidates can never override a trusted identity
# ======================================================================

def test_44b_candidate_never_demotes_trusted_face_identity():
    t = _make_track(adopt=2, switch=2)
    for n in range(2):
        t.register_vote("EMP004", 0.92, float(n) + 1.0)
    assert t.identity == "EMP004"
    assert t.identity_source == "face"
    # A long run of weak EMP003 candidates must not touch the trusted identity.
    for n in range(8):
        t.register_candidate_vote("EMP003", 0.55, 100.0 + n)
    assert t.identity == "EMP004"
    assert t.identity_source == "face"


def test_44b_trusted_emp004_never_demoted_by_emp003_appearance():
    # Explicit Phase 47 Part-5 pair: EMP004 (Vishal) trusted via face, then a
    # long run of STRONG EMP003 (Piyush) appearance votes. The trusted face
    # identity must NEVER become EMP003 -- not even strong appearance votes
    # (the strictest bound; weak votes carry no weight at all by design).
    t = _make_track(adopt=2, switch=2, app_adopt=2)
    for n in range(2):
        t.register_vote("EMP004", 0.92, float(n) + 1.0)
    assert t.identity == "EMP004"
    assert t.identity_source == "face"
    for n in range(10):
        t.register_appearance_vote("EMP003", 0.95, 100.0 + n)
    assert t.identity == "EMP004"
    assert t.identity_source == "face"


def test_44b_trusted_emp003_never_demoted_by_emp004_appearance():
    t = _make_track(adopt=2, switch=2, app_adopt=2)
    for n in range(2):
        t.register_vote("EMP003", 0.92, float(n) + 1.0)
    assert t.identity == "EMP003"
    assert t.identity_source == "face"
    for n in range(10):
        t.register_appearance_vote("EMP004", 0.95, 100.0 + n)
    assert t.identity == "EMP003"
    assert t.identity_source == "face"


def test_44b_candidate_oscillation_cannot_flip_a_track():
    t = _make_track(adopt=2, switch=2)
    for n in range(2):
        t.register_vote("EMP004", 0.92, float(n) + 1.0)
    noise = ["EMP003", "EMP001", "EMP002"]
    for n in range(30):
        t.register_candidate_vote(noise[n % 3], 0.53 + 0.01 * n, 200.0 + n)
    assert t.identity == "EMP004"
    assert t.identity_source == "face"


def test_44b_candidate_never_switches_even_after_trusted_identity_leaves():
    t = _make_track(adopt=2, switch=2)
    for n in range(3):
        t.register_vote("EMP004", 0.92, float(n) + 1.0)
    assert t.identity == "EMP004"
    # EMP004's tracked person leaves; now only weak EMP003 candidates arrive.
    # Confirmed-face switching needs a CONFIRMED run; candidates get none.
    for n in range(6):
        t.register_candidate_vote("EMP003", 0.57, 300.0 + n)
    assert t.identity == "EMP004"     # candidate never upgrades a trust
    assert t.identity_source == "face"


def test_44b_candidate_adopts_unresolved_track_after_consistency():
    t = _make_track(adopt=3, switch=3)
    for n in range(2):
        t.register_candidate_vote("EMP003", 0.54, float(n) + 1.0)
    assert t.identity is None or t.identity == UNKNOWN_ID   # not yet
    t.register_candidate_vote("EMP003", 0.55, 3.0)
    assert t.identity == "EMP003"
    assert t.identity_source == "candidate"


def test_44b_candidate_promoted_to_face_authority_by_confirmed_match():
    t = _make_track(adopt=2, switch=2)
    for n in range(2):
        t.register_candidate_vote("EMP003", 0.55, float(n) + 1.0)
    assert t.identity_source == "candidate"
    t.register_vote("EMP003", 0.88, 10.0)      # face finally confirms
    assert t.identity == "EMP003"
    assert t.identity_source == "face"


def test_44b_spatial_candidate_channel_end_to_end():
    # detector emission -> _register_face_vote -> candidate store.
    st = SpatialTracker(adopt_frames=2, switch_frames=2)
    pbox = (100, 40, 300, 460)
    fbox = (150, 90, 250, 200)
    cand = _face(fbox, "Unknown", person_box=pbox, score=0.55,
                 status=RECOG_CANDIDATE, cand_emp="EMP004",
                 cand_score=0.55, cand_margin=0.12)
    for i in range(3):
        st.process([_person(pbox), cand], now=1000.0 + i)
    tracks = st.active_tracks(1000.0 + 2)
    assert tracks[0].identity == "EMP004"
    assert tracks[0].identity_source == "candidate"


# ======================================================================
# PART 3 -- side/back pose: never fakes AWAY, identity retained
# ======================================================================

def test_44b_side_back_pose_person_stays_active(fake_clock, recorder,
                                                instant_commit):
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
    # Back-facing: body only, no face at all.
    for _ in range(6):
        _FakeTime.now += 1.0
        _cycle(st, mt, [_person(pbox)])
        assert mt.live_state("EMP004") == "ACTIVE", \
            "back-facing employee must NOT go AWAY"
    assert recorder == []             # no ACTIVE->AWAY interval ever logged


def test_44b_known_identity_retained_through_temporary_face_loss(
        fake_clock, recorder, instant_commit):
    st = SpatialTracker(adopt_frames=2, switch_frames=2)
    mt = MultiTracker(None)
    pbox = (100, 40, 300, 460)
    fbox = (150, 90, 250, 200)
    face = _face(fbox, "EMP004", person_box=pbox, score=0.9,
                 status=RECOG_CONFIRMED)
    _cycle(st, mt, [_person(pbox), face])
    _FakeTime.now += 1.0
    _cycle(st, mt, [_person(pbox), face])
    labels = {t["identity_label"] for t in st.snapshot(1000.0)["tracks"]}
    assert labels == {"EMP004"}
    for _ in range(8):
        _FakeTime.now += 1.0
        _cycle(st, mt, [_person(pbox)])
        assert mt.live_state("EMP004") == "ACTIVE"
    labels = {t["identity_label"] for t in st.snapshot(1000.0)["tracks"]}
    assert labels == {"EMP004"}


def test_44b_person_only_never_invents_identity(fake_clock, recorder,
                                                instant_commit):
    st = SpatialTracker(adopt_frames=2, switch_frames=2)
    mt = MultiTracker(None)
    for _ in range(3):
        _cycle(st, mt, [_person((100, 40, 300, 460))])
        _FakeTime.now += 1.0
    assert st.claims() == {}
    assert mt.employee_ids == []
    assert mt.live_states() == {}


def test_44b_face_loss_never_demotes_adopted_identity():
    st = SpatialTracker(adopt_frames=2, switch_frames=2)
    pbox = (100, 40, 300, 460)
    fbox = (150, 90, 250, 200)
    face = _face(fbox, "EMP004", person_box=pbox, score=0.9)
    st.process([_person(pbox), face], now=1000.0)
    st.process([_person(pbox), face], now=1000.1)
    assert st.active_tracks(1000.1)[0].identity == "EMP004"
    unk = _face(fbox, "Unknown", person_box=pbox, score=0.42)
    for i in range(4):
        st.process([_person(pbox), unk], now=1100.0 + i)
    tracks = st.active_tracks(1100.0 + 3)
    assert tracks[0].identity == "EMP004"     # never demoted to Unknown
    assert st.claims() == {"EMP004": {"cam": "cam1", "phone": False}}


# ======================================================================
# PART 4 -- performance: cadence, latest-frame, wall-clock absence
# ======================================================================

def test_44b_cadence_5_default_thins_recognition_only():
    assert config.FACE_DETECT_CADENCE == 5
    det = _det_stub(face_cadence=5)
    ticks = [det._face_tick("cam_01") for _ in range(11)]
    assert ticks.count(True) == 3          # cycles 1, 6, 11
    assert ticks == [True] + [False] * 4 + [True] + [False] * 4 + [True]


def test_44b_latest_frame_only_no_stale_backlog():
    class _FreshCap:
        def __init__(self):
            self.n = 0
            self.last_frame_timestamp = _realtime.time()
            self.is_connected = True

        def read(self):
            self.n += 1
            self.last_frame_timestamp = _realtime.time()
            return f"frame-{self.n}"

        def health_info(self):
            return {}

    class _StaleCap(_FreshCap):
        def __init__(self):
            super().__init__()
            self.n = 1                              # already served one frame
            self.last_frame_timestamp = _realtime.time() - 900.0  # dead cam

        def read(self):
            # A dead camera keeps serving its last held frame: the read must
            # NOT bump the timestamp or latest_frames() could call it "fresh".
            return f"frame-{self.n}"

    fresh = _FreshCap()
    stale = _StaleCap()
    mgr = MultiCameraManager.__new__(MultiCameraManager)
    mgr._cameras = {"a": "a", "b": "b"}
    mgr._captures = {"a": fresh, "b": stale}
    mgr._lock = __import__("threading").Lock()
    mgr._max_batch = None

    first = mgr.get_latest_batch()      # newest for each stream, no backlog
    assert first == {"a": "frame-1", "b": "frame-1"}
    live = mgr.latest_frames()          # stale dead camera excluded
    assert set(live) == {"a"}
    again = mgr.get_latest_batch()      # "frame-2" dropped; only the newest
    assert again["a"] == "frame-3"
    assert fresh.n == 3                 # every poll re-reads the newest frame


def test_44b_absence_is_wall_clock_not_frame_count(fake_clock, recorder,
                                                   instant_commit):
    st = SpatialTracker(adopt_frames=2, switch_frames=2)
    mt = MultiTracker(None)
    pbox = (100, 40, 300, 460)
    fbox = (150, 90, 250, 200)
    face = _face(fbox, "EMP001", person_box=pbox, score=0.9,
                 status=RECOG_CONFIRMED)
    _cycle(st, mt, [_person(pbox), face])
    _FakeTime.now += 1.0
    _cycle(st, mt, [_person(pbox), face])
    assert mt.live_state("EMP001") == "ACTIVE"
    other = (400, 50, 600, 400)          # a DIFFERENT person on camera
    # 3 processed cycles but only +0.9 wall second: grace 3s not elapsed.
    for _ in range(3):
        _FakeTime.now += 0.3
        _cycle(st, mt, [_person(other)])
    assert mt.live_state("EMP001") == "ACTIVE", \
        "absence must be measured in wall seconds, not frame count"
    _FakeTime.now += 3.0                 # now past AWAY_AFTER_SEC=3
    _cycle(st, mt, [_person(other)])
    assert mt.live_state("EMP001") == "AWAY"