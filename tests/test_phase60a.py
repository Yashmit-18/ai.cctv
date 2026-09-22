# -*- coding: utf-8 -*-
"""Phase 60A -- OFFLINE OFFICE SIMULATION + DESK/CHAIR/EMPLOYEE CONTEXT.

Scope
-----
Phase 60A validates the complete *software-side* relationship:

    Employee -> Identity -> Spatial Track -> Desk/Zone -> Chair/Seat ->
    Office Schedule -> Employee Context -> Productivity/State semantics

There is NO physical IP CCTV/NVR in this phase.  Every observation is a
deterministic TEST observation built by ``src.simulation.office_context``
(``Observation`` / ``make_track`` / ``footpoint_bbox``) and fed into the REAL
modules (``src.seat_zones``, ``src.seat_chairs``, ``src.office_schedule``,
``src.tracker``, ``src.zones``, ``src.database``).  Nothing here claims real
camera validation.

Honest guarantees asserted (and nothing stronger):
  * a chair/zone NEVER becomes identity proof;
  * an Unknown occupant stays Unknown even in an assigned chair;
  * EMP002 in EMP001's chair stays EMP002 (neutral mismatch observation only);
  * an empty chair / empty zone is VACANT and never "employee AWAY";
  * a camera outage freezes desk state and the employee FSM -- no fake AWAY;
  * changing ``display_name`` never changes ``employee_id`` / ownership keys.

All test runs are labelled SIMULATED VALIDATION (no real NVR was used).
"""

import datetime as _dt
import json
import zoneinfo

import pytest

import config
from src.office_schedule import OfficeSchedule, parse_hhmm
from src.seat_chairs import (
    CHAIR_ASSIGNMENT_MISMATCH,
    CHAIR_IDENTITY_CONFIRMED,
    CHAIR_IDENTITY_UNKNOWN,
    CHAIR_OCCUPIED,
    CHAIR_VACATED,
    Chair,
    ChairStore,
    load_chair_defaults,
)
from src.seat_zones import (
    SEAT_ASSIGNMENT_MISMATCH,
    ZONE_IDENTITY_CONFIRMED,
    ZONE_IDENTITY_UNKNOWN,
    ZONE_OCCUPIED,
    ZONE_VACATED,
    SeatZone,
    SeatZoneStore,
    SeatZoneTracker,
)
from src import database as db
from src.simulation.office_context import (
    Observation,
    footpoint_bbox,
    make_track,
    simulate_step,
    ts_for_step,
)
from src.tracker import EmployeeTracker, Track
from src.zones import parse_polygon

# ----------------------------------------------------------------------
# Deterministic office layout (SIMULATED, never a real camera scene)
# ----------------------------------------------------------------------
# CAM01 -- lower half of the frame is split into two desks:
#   D01 (EMP001's desk)  x in [0.05, 0.45], y in [0.60, 1.00]
#   D02 (EMP002's desk)  x in [0.55, 0.95], y in [0.60, 1.00]
# plus a DISABLED full-frame zone (DXX) and a config-invalid no-polygon zone
# (DNN) so disabled/empty-polygon behaviour is exercised.
ZONE_D01 = SeatZone(
    zone_id="D01", camera_id="CAM01", name="Desk-01",
    polygon=[[0.05, 0.6], [0.45, 0.6], [0.45, 1.0], [0.05, 1.0]],
    enabled=True, assigned_employee_id="EMP001")
ZONE_D02 = SeatZone(
    zone_id="D02", camera_id="CAM01", name="Desk-02",
    polygon=[[0.55, 0.6], [0.95, 0.6], [0.95, 1.0], [0.55, 1.0]],
    enabled=True, assigned_employee_id="EMP002")
ZONE_DXX = SeatZone(
    zone_id="DXX", camera_id="CAM01", name="Disabled-zone",
    polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    enabled=False, assigned_employee_id=None)
ZONE_DNN = SeatZone(
    zone_id="DNN", camera_id="CAM01", name="Bad-polygon-zone",
    polygon=None, enabled=True, assigned_employee_id=None)

# Canonical desk footpoints: D01 left desk, D02 right desk, both on CAM01.
FD01 = (0.25, 0.85)
FD02 = (0.75, 0.85)


def _tracker(zones, *, confirm=5, clear=8, cooldown=0.0, conn=None,
             active_cameras=None) -> SeatZoneTracker:
    tr = SeatZoneTracker(conn=conn, confirm_frames=confirm,
                         clear_frames=clear, event_cooldown_sec=cooldown,
                         active_cameras=active_cameras)
    tr.set_zones(zones)
    return tr


def _all_snap(tr):
    return {z["zone_id"]: z for z in tr.snapshot()["zones"]}


def _is_away(e) -> bool:
    return "AWAY" in str(e.get("event_type", ""))

# ======================================================================
# 1. DESK / ZONE MEMBERSHIP + FOOTPOINT SEMANTICS  (Phase 60A objective 6)
# ======================================================================

def test_60a_footpoint_is_bottom_center():
    """The desk engine evaluates the track's *footpoint*, not its centroid."""
    b = footpoint_bbox(FD01[0], FD01[1])
    assert b[3] == int(FD01[1] * 480)          # bbox bottom row == foot y
    t = make_track("t1", "CAM01", FD01[0], FD01[1], "EMP001")
    fx, fy = t.footpoint_normalized
    assert fx == round(FD01[0], 4) and fy == round(FD01[1], 4)
    cx, cy = t.centroid_normalized
    assert cy < 0.6                            # centroid is above the desk zone
    assert fy >= 0.6                           # footpoint is inside it
    assert ZONE_D01.contains(fx, fy)
    assert not ZONE_D01.contains(cx, cy)


def test_60a_inside_assigned_desk_zone():
    tr = _tracker([ZONE_D01])
    t = make_track("t1", "CAM01", FD01[0], FD01[1], "EMP001")
    assert tr.track_zone("t1") is None
    for _ in range(5):
        tr.tick([t], usable_camera_ids={"CAM01"})
    assert tr.track_zone("t1") == "CAM01:D01"


def test_60a_outside_desk_zone_no_membership():
    tr = _tracker([ZONE_D01, ZONE_D02])
    t = make_track("t1", "CAM01", 0.25, 0.30, "EMP001")   # above both desks
    for _ in range(8):
        tr.tick([t], usable_camera_ids={"CAM01"})
    assert tr.track_zone("t1") is None
    assert _all_snap(tr)["D01"]["status"] == "VACANT"


def test_60a_zone_boundary_behavior():
    """Strict, deterministic membership around the polygon edge."""
    inside, outside = ZONE_D01.contains(0.25, 0.85), ZONE_D01.contains(0.25, 0.30)
    assert inside is True and outside is False
    # Epsilon behaviour across the right edge (x = 0.45) is strict.
    assert ZONE_D01.contains(0.4499, 0.85) is True
    assert ZONE_D01.contains(0.4501, 0.85) is False
    # Epsilon behaviour across the top edge (y = 0.60).
    assert ZONE_D01.contains(0.25, 0.6001) is True
    assert ZONE_D01.contains(0.25, 0.5999) is False
    # Same query twice is identical -- membership has no hidden state.
    assert ZONE_D01.contains(0.25, 0.85) is True


def test_60a_disabled_zone_excluded_and_labeled():
    tr = _tracker([ZONE_D01, ZONE_DXX])
    t = make_track("t1", "CAM01", 0.25, 0.85, "EMP001")
    for _ in range(5):
        tr.tick([t], usable_camera_ids={"CAM01"})
    snap = _all_snap(tr)
    assert snap["D01"]["status"] == "OCCUPIED"
    assert snap["DXX"]["status"] == "DISABLED"
    assert snap["DXX"]["current_identity"] is None


def test_60a_invalid_or_empty_polygon_never_occupies():
    with pytest.raises(ValueError):
        parse_polygon("[[0, 0], [1, 0]]")         # fewer than 3 points
    with pytest.raises(ValueError):
        parse_polygon("not-json")
    z = SeatZone(zone_id="DNN", camera_id="CAM01", polygon=None)
    assert z.contains(0.5, 0.5) is False          # no polygon -> never inside
    z2 = SeatZone(zone_id="DNN", camera_id="CAM01", polygon=[])
    assert z2.contains(0.5, 0.5) is False
    tr = _tracker([ZONE_DNN])
    t = make_track("t1", "CAM01", FD01[0], FD01[1], "EMP001")
    for _ in range(8):
        tr.tick([t], usable_camera_ids={"CAM01"})
    assert tr.track_zone("t1") is None            # no occupancy ever


def test_60a_wrong_camera_id_no_membership():
    """Identical polygon coordinates on the WRONG camera must not match."""
    tr = _tracker([ZONE_D01])
    t = make_track("t1", "CAM02", FD01[0], FD01[1], "EMP001")  # wrong camera
    for _ in range(8):
        tr.tick([t], usable_camera_ids={"CAM01", "CAM02"})
    assert tr.track_zone("t1") is None


def test_60a_unknown_zone_no_membership():
    tr = _tracker([ZONE_D01, ZONE_D02])
    t = make_track("t1", "CAM01", 0.5, 0.85)      # between the two desks
    for _ in range(8):
        tr.tick([t], usable_camera_ids={"CAM01"})
    assert tr.track_zone("t1") is None
    assert sum(1 for z in _all_snap(tr).values()
               if z["status"] == "OCCUPIED") == 0


def test_60a_zone_stability_transient_noise():
    """inside -> outside -> inside must not upset a stable zone."""
    tr = _tracker([ZONE_D01], confirm=5, clear=2)
    t = make_track("t1", "CAM01", FD01[0], FD01[1], "EMP001")
    for _ in range(5):
        tr.tick([t], usable_camera_ids={"CAM01"})
    assert _all_snap(tr)["D01"]["status"] == "OCCUPIED"
    events = []
    # A single noisy outside frame (1 of 2 clear-frames) is absorbed.
    events += tr.tick([make_track("t1", "CAM01", 0.25, 0.30, "EMP001")],
                      usable_camera_ids={"CAM01"})
    assert _all_snap(tr)["D01"]["status"] == "OCCUPIED"
    assert not any(e["event_type"] == ZONE_VACATED and e["zone_id"] == "CAM01:D01"
                   for e in events)
    # Person returns -- still the same stable occupancy episode.
    events += tr.tick([t], usable_camera_ids={"CAM01"})
    assert _all_snap(tr)["D01"]["status"] == "OCCUPIED"
    assert not any(e["event_type"] == ZONE_OCCUPIED and e["zone_id"] == "CAM01:D01"
                   for e in events)


# ======================================================================
# 2. CHAIR / SEAT VALIDATION  (Phase 60A objective 7)
# ======================================================================

def test_60a_chair_exists_enabled_and_scoped(tmp_db):
    st = ChairStore(tmp_db)
    st.add(chair_id="C01", camera_id="CAM01", zone_id="D01",
           label="Desk-01-chair", enabled=True, assigned_employee_id="EMP001")
    c = st.get("C01")
    assert c is not None
    assert c.enabled is True
    assert c.qualified_id == "CAM01:D01:C01"
    assert c.zone_id == "D01"                      # belongs to desk zone
    assert c.camera_id == "CAM01"                  # belongs to camera
    assert {x.chair_id for x in st.for_zone("D01")} == {"C01"}
    assert {x.chair_id for x in st.for_camera("CAM01")} == {"C01"}


def test_60a_chair_disabled_excluded(tmp_db):
    st = ChairStore(tmp_db)
    st.add(chair_id="C01", camera_id="CAM01", zone_id="D01")
    st.add(chair_id="C02", camera_id="CAM01", zone_id="D02")
    st.set_enabled("C02", False)
    assert st.get("C02").enabled is False
    enabled = st.list(enabled_only=True)
    assert {c.chair_id for c in enabled} == {"C01"}


def test_60a_chair_assigned_and_unassigned(tmp_db):
    st = ChairStore(tmp_db)
    st.add(chair_id="C01", camera_id="CAM01", zone_id="D01",
           assigned_employee_id="EMP001")
    st.add(chair_id="C02", camera_id="CAM01", zone_id="D02")   # unassigned
    assert st.get("C01").assigned_employee_id == "EMP001"
    assert st.get("C02").assigned_employee_id is None
    st.set_assignment("C02", "EMP002")
    assert st.get("C02").assigned_employee_id == "EMP002"
    st.set_assignment("C02", None)
    assert st.get("C02").assigned_employee_id is None


def test_60a_chair_missing_and_empty_config(tmp_db, tmp_path):
    st = ChairStore(tmp_db)
    assert st.get("NOPE") is None                  # missing chair config
    assert st.list() == []                         # empty chair configuration
    missing = tmp_path / "chairs-missing.json"
    assert load_chair_defaults(str(missing)) == []


# ======================================================================
# 3. IDENTITY SAFETY  (Phase 60A objective 8 -- chair is never identity)
# ======================================================================

def test_60a_identity_matches_assigned_chair_context(tmp_db):
    """A: EMP001 identity + EMP001 assigned chair -> consistent context."""
    tr = _tracker([ZONE_D01], conn=tmp_db)
    t = make_track("t1", "CAM01", FD01[0], FD01[1], "EMP001")
    events = []
    for _ in range(5):
        events += tr.tick([t], usable_camera_ids={"CAM01"})
    claim = tr.claims()[0]
    assert claim["employee_id"] == "EMP001"
    assert claim["identity_state"] == "CONFIRMED"
    assert claim["zone_id"] == "D01"
    assert any(e["event_type"] == ZONE_IDENTITY_CONFIRMED and
               e["employee_id"] == "EMP001" for e in events)


def test_60a_identity_chair_mismatch_is_neutral(tmp_db):
    """B: EMP002 identity in EMP001's chair -> neutral mismatch, no relabel."""
    tr = _tracker([ZONE_D01], conn=tmp_db)
    t = make_track("t1", "CAM01", FD01[0], FD01[1], "EMP002")
    events = []
    for _ in range(5):
        events += tr.tick([t], usable_camera_ids={"CAM01"})
    mis = [e for e in events if e["event_type"] == SEAT_ASSIGNMENT_MISMATCH]
    assert mis
    assert "assigned=EMP001" in mis[0]["details"] and "confirmed=EMP002" in \
        mis[0]["details"]
    for bad in ("intruder", "unauthorized", "trespass"):
        assert bad not in mis[0]["details"].lower()
    assert tr.claims()[0]["employee_id"] == "EMP002"   # identity upheld
    assert tr.snapshot()["zones"][0]["current_identity"] == "EMP002"


def test_60a_unknown_in_assigned_chair_stays_unknown(tmp_db):
    """C: UNKNOWN identity + assigned chair -> UNKNOWN remains UNKNOWN."""
    tr = _tracker([ZONE_D01], conn=tmp_db)
    t = make_track("t1", "CAM01", FD01[0], FD01[1], "Unknown")
    events = []
    for _ in range(5):
        events += tr.tick([t], usable_camera_ids={"CAM01"})
    assert any(e["event_type"] == ZONE_IDENTITY_UNKNOWN for e in events)
    claim = tr.claims()[0]
    assert claim["employee_id"] == "Unknown"
    assert claim["identity_state"] == "UNKNOWN"
    assert claim["employee_id"] != "EMP001"        # chair never bestows identity


def test_60a_unknown_and_empty_chair_context(tmp_db):
    """D: UNKNOWN + empty chair and UNKNOWN-alone -> never an employee."""
    tr = _tracker([ZONE_D01], conn=tmp_db)
    t = make_track("t1", "CAM01", FD01[0], FD01[1], "Unknown")
    for _ in range(5):
        tr.tick([t], usable_camera_ids={"CAM01"})
    assert tr.claims()[0]["employee_id"] == "Unknown"
    # after this person leaves, the zone is empty and VACANT, employee free
    for _ in range(10):
        tr.tick([], usable_camera_ids={"CAM01"})
    assert _all_snap(tr)["D01"]["status"] == "VACANT"
    assert tr.claims() == []


# ======================================================================
# 4. EMPLOYEE -> DESK -> CHAIR -> CAMERA MAPPING  (Phase 60A objective 9)
# ======================================================================

def test_60a_employee_desk_chair_camera_mapping(tmp_db):
    db.upsert_employee(tmp_db, "EMP001", name="A", display_name="A")
    db.upsert_employee(tmp_db, "EMP002", name="B", display_name="B")
    zs = SeatZoneStore(tmp_db)
    zs.add(zone_id="D01", camera_id="CAM01", name="Desk-01",
           polygon=[[0.05, 0.6], [0.45, 0.6], [0.45, 1.0], [0.05, 1.0]],
           assigned_employee_id="EMP001")
    zs.add(zone_id="D02", camera_id="CAM01", name="Desk-02",
           polygon=[[0.55, 0.6], [0.95, 0.6], [0.95, 1.0], [0.55, 1.0]],
           assigned_employee_id="EMP002")
    cs = ChairStore(tmp_db)
    cs.add(chair_id="C01", camera_id="CAM01", zone_id="D01",
           assigned_employee_id="EMP001")
    cs.add(chair_id="C02", camera_id="CAM01", zone_id="D02",
           assigned_employee_id="EMP002")

    tr = SeatZoneTracker(conn=tmp_db, store=zs, confirm_frames=5,
                         clear_frames=8, event_cooldown_sec=0.0,
                         active_cameras={"CAM01"})
    for _ in range(5):
        tr.tick([make_track("ta", "CAM01", FD01[0], FD01[1], "EMP001"),
                 make_track("tb", "CAM01", FD02[0], FD02[1], "EMP002")],
                usable_camera_ids={"CAM01"})

    snap = _all_snap(tr)
    assert snap["D01"]["current_identity"] == "EMP001"
    assert snap["D01"]["zone_id"] == "D01"
    assert snap["D02"]["current_identity"] == "EMP002"
    ch = {c.chair_id: c for c in cs.list()}
    assert ch["C01"].assigned_employee_id == "EMP001"
    assert ch["C01"].zone_id == "D01" and ch["C01"].camera_id == "CAM01"
    assert ch["C02"].assigned_employee_id == "EMP002"
    assert ch["C02"].zone_id == "D02" and ch["C02"].camera_id == "CAM01"
    # desk assignment is contextual (never identity)
    assert snap["D01"]["assigned_employee_id"] == "EMP001"


def test_60a_display_name_change_keeps_all_identity_keys(tmp_db):
    """Changing display_name must never alter employee_id / ownership."""
    db.upsert_employee(tmp_db, "EMP001", name="Sourabh", display_name="Sourabh")
    cs = ChairStore(tmp_db)
    cs.add(chair_id="C01", camera_id="CAM01", zone_id="D01",
           assigned_employee_id="EMP001")
    db.upsert_employee(tmp_db, "EMP001", display_name="Rahul")   # relabel
    emps = [e for e in db.list_employees(tmp_db)
            if e["employee_id"] == "EMP001"]
    assert len(emps) == 1 and emps[0]["employee_id"] == "EMP001"
    assert emps[0]["display_name"] == "Rahul"
    # ownership and chair/desk identity are untouched by the display change
    assert cs.get("C01").assigned_employee_id == "EMP001"
    assert cs.get("C01").zone_id == "D01" and cs.get("C01").camera_id == "CAM01"
    assert cs.get("C01").qualified_id == "CAM01:D01:C01"


# ======================================================================
# 5. OFFICE SCHEDULE  (Phase 60A objective 10)
# ======================================================================

TZ = zoneinfo.ZoneInfo(config.CCTV_OFFICE_TZ)


def _at(hour: int, minute: int = 0) -> _dt.datetime:
    return _dt.datetime(2026, 9, 22, hour, minute, tzinfo=TZ)


def test_60a_office_schedule_defaults():
    s = OfficeSchedule()
    assert str(s.tz) == config.CCTV_OFFICE_TZ == "Asia/Kolkata"
    assert s.start == "10:00" and s.end == "18:30"
    assert s.lunch_start == "14:00" and s.lunch_end == "14:35"
    assert s.is_office_hours(_at(10, 0)) is True
    assert s.is_office_hours(_at(18, 30)) is False


def test_60a_office_hours_boundaries():
    s = OfficeSchedule()
    assert s.is_office_hours(_at(9, 59)) is False
    assert s.is_office_hours(_at(10, 0)) is True
    assert s.is_office_hours(_at(10, 1)) is True
    assert s.is_office_hours(_at(18, 29)) is True
    assert s.is_office_hours(_at(18, 30)) is False
    assert s.is_office_hours(_at(18, 31)) is False


def test_60a_lunch_window_boundaries():
    s = OfficeSchedule()
    assert s.is_lunch_break(_at(13, 59)) is False
    assert s.is_lunch_break(_at(14, 0)) is True
    assert s.is_lunch_break(_at(14, 34)) is True
    assert s.is_lunch_break(_at(14, 35)) is False
    assert s.is_lunch_break(_at(14, 36)) is False
    assert s.is_monitoring_period(_at(13, 59)) is True   # pre-lunch
    assert s.is_monitoring_period(_at(14, 0)) is False   # lunch excluded


def test_60a_schedule_validation_rejects_bad_knobs():
    with pytest.raises(ValueError):
        parse_hhmm("24:00", knob="TEST")
    with pytest.raises(ValueError):
        OfficeSchedule(lunch_start="14:35", lunch_end="14:00")


# ======================================================================
# 6. THRESHOLD CONFIGURATION  (Phase 60A objective 11)
# ======================================================================

def test_60a_future_thresholds_centralised():
    assert float(config.FUTURE_AWAY_THRESHOLD_SECONDS) == 10.0
    assert float(config.FUTURE_PHONE_THRESHOLD_SECONDS) == 5.0
    assert float(config.FUTURE_TALKING_THRESHOLD_SECONDS) == 15.0


# ======================================================================
# 7. EMPLOYEE STATE SEMANTICS  (Phase 60A objectives 12/13)
# @description
# ======================================================================

class _FakeTime:
    now = 1_000_000.0

    @classmethod
    def time(cls) -> float:
        return cls.now

    @classmethod
    def strftime(cls, fmt, tt=None):
        import time as _real
        return _real.strftime(fmt, tt or _real.localtime(cls.now))

    @classmethod
    def localtime(cls, tt=None):
        import time as _real
        return _real.localtime(tt or cls.now)


@pytest.fixture
def fake_clock(monkeypatch):
    _FakeTime.now = 1_000_000.0
    monkeypatch.setattr("src.tracker.time", _FakeTime)
    return _FakeTime


def test_60a_healthy_absence_is_still_away(fake_clock, tmp_db):
    """The FSM only marks AWAY from the employee's own confirmed absence
    while the camera stays healthy -- never from a chair or a zone."""
    tr = EmployeeTracker(tmp_db, "EMP001")
    tr.process({"present": True}, camera_online=True)
    assert tr.committed_state == "ACTIVE"
    fake_clock.now += 30.0                        # genuinely gone (>AWAY_AFTER_SEC)
    tr.process({"present": False}, camera_online=True)
    assert tr.committed_state == "AWAY"


def test_60a_camera_failure_no_fake_away(fake_clock, tmp_db):
    """E/G: camera unavailable / no frames -> FSM freezes, never AWAY."""
    tr = EmployeeTracker(tmp_db, "EMP001")
    tr.process({"present": True}, camera_online=True)
    assert tr.committed_state == "ACTIVE"
    for _ in range(5):
        fake_clock.now += 30.0
        tr.process({"present": False}, camera_online=False)   # camera down
        assert tr.committed_state == "ACTIVE"                 # frozen
    assert tr.committed_state != "AWAY"


def test_60a_seat_layer_freeze_on_camera_outage():
    """Camera down -> desk zone freezes OCCUPIED, no VACATED, no AWAY."""
    tr = _tracker([ZONE_D01])
    t = make_track("t1", "CAM01", FD01[0], FD01[1], "EMP001")
    events = []
    for _ in range(5):
        events += tr.tick([t], usable_camera_ids={"CAM01"})
    assert _all_snap(tr)["D01"]["status"] == "OCCUPIED"
    for _ in range(200):
        events += tr.tick([], usable_camera_ids=set())       # dead camera
    snap = _all_snap(tr)["D01"]
    assert snap["status"] == "OCCUPIED"                      # never force-vacated
    assert not any(e["event_type"] == ZONE_VACATED for e in events)
    assert not any(_is_away(e) for e in events)


def test_60a_empty_chair_no_fake_away(tmp_db):
    """F: an empty chair is VACANT -- it never becomes employee AWAY."""
    tr = _tracker([ZONE_D01], conn=tmp_db)
    events = []
    for _ in range(20):
        events += tr.tick([], usable_camera_ids={"CAM01"})
    snap = _all_snap(tr)["D01"]
    assert snap["status"] == "VACANT"
    assert snap["current_identity"] is None
    assert tr.claims() == []
    assert not any(_is_away(e) for e in events)
    assert events == []                       # no transition -> no generated event


# ======================================================================
# 8. CAMERA CONTEXT  (Phase 60A objective 13)
# ======================================================================

def test_60a_camera_id_matching_context():
    tr = _tracker([ZONE_D01, ZONE_D02])
    t = make_track("t1", "CAM01", FD01[0], FD01[1], "EMP001")
    for _ in range(5):
        tr.tick([t], usable_camera_ids={"CAM01"})
    assert tr.track_zone("t1") == "CAM01:D01"


def test_60a_disabled_camera_context():
    tr = _tracker([ZONE_D01], active_cameras={"CAM02"})
    t = make_track("t1", "CAM01", FD01[0], FD01[1], "EMP001")
    for _ in range(6):
        tr.tick([t], usable_camera_ids={"CAM01"})
    snap = tr.snapshot()["zones"][0]
    assert snap["status"] == "CAMERA_DISABLED"
    assert snap["current_identity"] is None


def test_60a_missing_camera_context():
    tr = _tracker([ZONE_D01, ZONE_D02])
    t = make_track("t1", "CAM99", FD01[0], FD01[1], "EMP001")  # camera w/o zones
    for _ in range(6):
        tr.tick([t], usable_camera_ids={"CAM99", "CAM01"})
    assert tr.track_zone("t1") is None
    assert all(z["status"] == "VACANT" for z in tr.snapshot()["zones"])


# ======================================================================
# 9. MULTI-PERSON SIMULATION  (Phase 60A objective 14) -- SIMULATED
# ======================================================================

def test_60a_multi_person_both_mapped():
    tr = _tracker([ZONE_D01, ZONE_D02], confirm=5, clear=8)
    evs = []
    first_step = last_step = None
    for step in range(5):
        tracks, obs = simulate_step(step, [
            ("person_a", "EMP001", "CAM01", FD01[0], FD01[1]),
            ("person_b", "EMP002", "CAM01", FD02[0], FD02[1]),
        ])
        o1, o2 = obs
        if first_step is None:
            first_step = (o1, o2)
        last_step = (o1, o2)
        evs += tr.tick(tracks, usable_camera_ids={"CAM01"})
    snap = _all_snap(tr)
    assert snap["D01"]["status"] == "OCCUPIED"
    assert snap["D01"]["current_identity"] == "EMP001"
    assert snap["D02"]["status"] == "OCCUPIED"
    assert snap["D02"]["current_identity"] == "EMP002"
    # deterministic observations carry all the required fields
    asserted = last_step
    for o in asserted:
        assert o.employee_id in ("EMP001", "EMP002")
        assert o.camera_id == "CAM01"
        assert 0.0 <= o.fx <= 1.0 and 0.0 <= o.fy <= 1.0
    assert last_step[0].ts > first_step[0].ts       # monotonic simulated clock


def test_60a_multi_person_wrong_chair_context():
    """One employee intentionally in the other employee's desk/chair."""
    tr = _tracker([ZONE_D01, ZONE_D02], confirm=5, clear=8)
    evs = []
    for step in range(5):
        tracks, _ = simulate_step(step, [
            ("person_a", "EMP002", "CAM01", FD01[0], FD01[1]),  # EMP002 in D01
            ("person_b", "EMP001", "CAM01", FD02[0], FD02[1]),  # EMP001 in D02
        ])
        evs += tr.tick(tracks, usable_camera_ids={"CAM01"})
    snap = _all_snap(tr)
    assert snap["D01"]["current_identity"] == "EMP002"   # never relabelled
    assert snap["D02"]["current_identity"] == "EMP001"
    mis = [e for e in evs if e["event_type"] == SEAT_ASSIGNMENT_MISMATCH]
    assert mis
    for e in mis:
        assert e["employee_id"] in ("EMP001", "EMP002")  # confirmed occupant only


def test_60a_multi_person_unknown_next_to_assigned_chair():
    tr = _tracker([ZONE_D01, ZONE_D02], confirm=5, clear=8)
    for step in range(5):
        tracks, _ = simulate_step(step, [
            ("person_a", "EMP001", "CAM01", FD01[0], FD01[1]),
            ("person_x", "Unknown", "CAM01", FD02[0], FD02[1]),
        ])
        tr.tick(tracks, usable_camera_ids={"CAM01"})
    snap = _all_snap(tr)
    assert snap["D01"]["current_identity"] == "EMP001"
    assert snap["D02"]["current_identity"] == "Unknown"   # unknown stays unknown
    assert snap["D02"]["identity_state"] == "UNKNOWN"


def test_60a_two_people_in_same_zone():
    tr = _tracker([ZONE_D01, ZONE_D02], confirm=5, clear=8)
    for step in range(5):
        tracks, _ = simulate_step(step, [
            ("person_a", "EMP001", "CAM01", FD01[0], FD01[1]),
            ("person_b", "EMP002", "CAM01", 0.40, 0.85),      # both in D01
        ])
        tr.tick(tracks, usable_camera_ids={"CAM01"})
    snap = _all_snap(tr)["D01"]
    assert snap["status"] == "OCCUPIED"
    assert snap["persons"] == 2
    assert snap["current_identity"] in ("EMP001", "EMP002")   # first occupant


def test_60a_person_leaves_then_returns():
    tr = _tracker([ZONE_D01], confirm=5, clear=8)
    evs = []
    for step in range(5):
        tracks, _ = simulate_step(step, [("person_a", "EMP001", "CAM01",
                                          FD01[0], FD01[1])])
        evs += tr.tick(tracks, usable_camera_ids={"CAM01"})
    assert _all_snap(tr)["D01"]["status"] == "OCCUPIED"
    for _ in range(10):                                    # leaves completely
        evs += tr.tick([], usable_camera_ids={"CAM01"})
    snap = _all_snap(tr)["D01"]
    assert snap["status"] == "VACANT"
    assert snap["current_identity"] is None
    assert any(e["event_type"] == ZONE_VACATED for e in evs)
    # returns -> a fresh occupancy episode with the same stable identity
    evs = []
    for step in range(5):
        tracks, _ = simulate_step(step, [("person_a", "EMP001", "CAM01",
                                          FD01[0], FD01[1])])
        evs += tr.tick(tracks, usable_camera_ids={"CAM01"})
    snap = _all_snap(tr)["D01"]
    assert snap["status"] == "OCCUPIED"
    assert snap["current_identity"] == "EMP001"
    assert snap["episode"] == 2


# ======================================================================
# 10. TEMPORAL VALIDATION  (Phase 60A objective 15)
# ======================================================================

def test_60a_track_loss_then_reappear_same_episode():
    """Loss shorter than clear_frames keeps the stable occupancy episode."""
    tr = _tracker([ZONE_D01], confirm=5, clear=8)
    t = make_track("t1", "CAM01", FD01[0], FD01[1], "EMP001")
    for _ in range(5):
        tr.tick([t], usable_camera_ids={"CAM01"})
    assert _all_snap(tr)["D01"]["status"] == "OCCUPIED"
    # tracker loses the person for fewer frames than clear_frames
    for _ in range(5):
        tr.tick([], usable_camera_ids={"CAM01"})
    assert _all_snap(tr)["D01"]["status"] == "OCCUPIED"
    # reappears -- still OCCUPIED, no context reset
    tr.tick([t], usable_camera_ids={"CAM01"})
    assert _all_snap(tr)["D01"]["status"] == "OCCUPIED"
    assert _all_snap(tr)["D01"]["episode"] == 1


def test_60a_context_reset_after_clear():
    tr = _tracker([ZONE_D01], confirm=5, clear=2)
    t = make_track("t1", "CAM01", FD01[0], FD01[1], "EMP001")
    for _ in range(5):
        tr.tick([t], usable_camera_ids={"CAM01"})
    assert _all_snap(tr)["D01"]["status"] == "OCCUPIED"
    for _ in range(2):                                    # >= clear_frames
        tr.tick([], usable_camera_ids={"CAM01"})
    snap = _all_snap(tr)["D01"]
    assert snap["status"] == "VACANT"
    assert snap["current_identity"] is None
    assert snap["episode"] == 1
    tr.tick([t], usable_camera_ids={"CAM01"})             # reset context
    assert _all_snap(tr)["D01"]["status"] == "VACANT"     # needs fresh confirm
    for _ in range(4):
        tr.tick([t], usable_camera_ids={"CAM01"})
    snap = _all_snap(tr)["D01"]
    assert snap["status"] == "OCCUPIED"
    assert snap["episode"] == 2


# ======================================================================
# 11. PRIVACY / LOGGING SAFETY  (Phase 60A objective 16)
# ======================================================================

def test_60a_events_and_observations_privacy_safe(tmp_db):
    tr = _tracker([ZONE_D01, ZONE_D02], conn=tmp_db)
    observations = []
    for step in range(5):
        tracks, obs_list = simulate_step(step, [
            ("person_a", "EMP001", "CAM01", FD01[0], FD01[1]),
            ("person_b", "EMP002", "CAM01", FD02[0], FD02[1]),
        ])
        observations += obs_list
        tr.tick(tracks, usable_camera_ids={"CAM01"})
    rows = db.list_seat_events(tmp_db)
    assert rows
    allowed = {"id", "timestamp", "date", "event_type", "camera_id", "zone_id",
               "track_id", "employee_id", "details", "created_at"}
    assert set(rows[0].keys()) <= allowed
    joined = json.dumps(rows).lower()
    for secret in ("password", "rtsp://", "http://", "embedding", "face_box",
                   "person_box", "bbox", "credential"):
        assert secret not in joined
    # observations only carry ids/footpoint/timestamp -- no images or vectors
    for o in observations:
        assert isinstance(o, Observation)
        assert "embedding" not in o.__dict__ and "image" not in o.__dict__


# ======================================================================
# 12. ADMIN CONFIGURATION SURFACE  (Phase 60A objective 17)
# ======================================================================

def test_60a_admin_config_surface_represents_office_model(tmp_db):
    """Existing configuration surfaces can represent all office context."""
    db.upsert_employee(tmp_db, "EMP001", name="Sourabh", display_name="Sourabh",
                       active=True)
    emp = db.get_employee(tmp_db, "EMP001")
    assert emp["display_name"] == "Sourabh" and emp["active"] == 1
    db.set_employee_active(tmp_db, "EMP001", False)
    assert db.get_employee(tmp_db, "EMP001")["active"] == 0

    zs = SeatZoneStore(tmp_db)
    zs.add(zone_id="D01", camera_id="CAM01", name="Desk-01",
           polygon=[[0.05, 0.6], [0.45, 0.6], [0.45, 1.0], [0.05, 1.0]])
    zs.set_assignment("D01", "EMP001")
    assert zs.get("D01").assigned_employee_id == "EMP001"

    cs = ChairStore(tmp_db)
    cs.add(chair_id="C01", camera_id="CAM01", zone_id="D01")
    cs.set_assignment("C01", "EMP001")
    assert cs.get("C01").assigned_employee_id == "EMP001"

    s = OfficeSchedule()
    assert (s.start, s.end, s.lunch_start, s.lunch_end) == \
        (config.CCTV_OFFICE_START, config.CCTV_OFFICE_END,
         config.CCTV_LUNCH_START, config.CCTV_LUNCH_END)
    assert float(config.FUTURE_AWAY_THRESHOLD_SECONDS) == 10.0
    assert config.CHAIR_TRACKING_ENABLED in (True, False)
    assert int(config.CHAIR_CONFIRM_FRAMES) >= 1
    assert int(config.SEAT_ZONE_CONFIRM_FRAMES) >= 1


# ======================================================================
# 13. HONESTY MARKER
# ======================================================================

def test_60a_simulated_honesty_marker():
    """All Phase 60A validation is SIMULATED/OFFLINE -- no real CCTV/NVR."""
    assert True