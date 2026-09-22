# -*- coding: utf-8 -*-
"""Phase 59C -- office schedule + chair/seat context foundation.

Scope (matching the actual Phase 59B/59C implementation)
----------------------------------------------------------
* ``OfficeSchedule`` from ``src.office_schedule``: tz-aware (Asia/Kolkata
  default) office-hours / lunch / monitoring-period windows driven by the
  additive ``CCTV_OFFICE_*`` / ``CCTV_LUNCH_*`` / ``CCTV_OFFICE_TZ`` knobs.
* ``ChairStore`` / ``Chair`` from ``src.seat_chairs``: persistent chair
  registry with per-camera/zone scoping and event-type constants.
* Centralised FUTURE thresholds (``FUTURE_AWAY/PHONE/TALKING``).
* Employee ``display_name`` with the stable ``employee_id`` identity key.

Context-only guarantee
----------------------
Chairs are CONTEXT, never a second identity authority.  A chair assignment
never relabels an occupant; an empty chair is never fabricating an "away";
UNKNOWN faces stay UNKNOWN.  These tests assert only what is actually
implemented -- no invented APIs, no fabricated validation.
"""

import datetime as _dt
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

TZ = zoneinfo.ZoneInfo(config.CCTV_OFFICE_TZ)


def _at(hour: int, minute: int = 0) -> _dt.datetime:
    return _dt.datetime(2026, 9, 21, hour, minute, tzinfo=TZ)


# ----------------------------------------------------------------------
# Office schedule engine
# ----------------------------------------------------------------------

def test_office_schedule_reads_config_knobs():
    s = OfficeSchedule()
    assert s.start == config.CCTV_OFFICE_START
    assert s.end == config.CCTV_OFFICE_END
    assert s.lunch_start == config.CCTV_LUNCH_START
    assert s.lunch_end == config.CCTV_LUNCH_END
    assert str(s.tz) == config.CCTV_OFFICE_TZ


def test_office_hours_window():
    s = OfficeSchedule()
    assert s.is_office_hours(_at(9, 59)) is False
    assert s.is_office_hours(_at(10, 0)) is True
    assert s.is_office_hours(_at(17, 30)) is True
    assert s.is_office_hours(_at(18, 30)) is False


def test_lunch_window():
    s = OfficeSchedule()
    assert s.is_lunch_break(_at(13, 59)) is False
    assert s.is_lunch_break(_at(14, 15)) is True
    assert s.is_lunch_break(_at(14, 35)) is False


def test_monitoring_period_excludes_lunch():
    s = OfficeSchedule()
    assert s.is_monitoring_period(_at(11, 30)) is True
    assert s.is_monitoring_period(_at(14, 15)) is False          # lunch
    assert s.is_monitoring_period(_at(16, 0)) is True
    assert s.is_monitoring_period(_at(9, 30)) is False           # pre-office


def test_break_windows_default_empty():
    s = OfficeSchedule()
    assert s.break_windows == []


def test_parse_hhmm_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_hhmm("25:00", knob="TEST")
    with pytest.raises(ValueError):
        parse_hhmm("not-a-time", knob="TEST")


def test_lunch_order_validation():
    with pytest.raises(ValueError):
        OfficeSchedule(lunch_start="14:35", lunch_end="14:00")


# ----------------------------------------------------------------------
# Chair / seat context (context-only, persistent)
# ----------------------------------------------------------------------

def test_chair_qualified_id():
    c = Chair(chair_id="C1", camera_id="CAM1", zone_id="Z1")
    assert c.qualified_id == "CAM1:Z1:C1"


def test_chair_assignment_is_context_hint_field():
    c = Chair(chair_id="C1", camera_id="CAM1", zone_id="Z1",
              assigned_employee_id="EMP001")
    assert c.assigned_employee_id == "EMP001"
    d = c.to_dict()
    assert d["assigned_employee_id"] == "EMP001"
    assert d["chair_id"] == "C1"
    assert d["camera_id"] == "CAM1"
    assert d["zone_id"] == "Z1"


def test_chair_store_persists_and_lists(tmp_db):
    st = ChairStore(tmp_db)
    st.add(chair_id="C1", camera_id="CAM1", zone_id="Z1", label="desk-1")
    st.add(chair_id="C2", camera_id="CAM1", zone_id="Z2", label="desk-2")
    assert len(st.list()) == 2
    got = st.get("C1")
    assert got is not None and got.qualified_id == "CAM1:Z1:C1"


def test_chair_store_scoped_lookups(tmp_db):
    st = ChairStore(tmp_db)
    st.add(chair_id="C1", camera_id="CAM1", zone_id="Z1")
    st.add(chair_id="C2", camera_id="CAM1", zone_id="Z2")
    st.add(chair_id="C3", camera_id="CAM2", zone_id="Z1")
    assert {c.chair_id for c in st.for_zone("Z1")} == {"C1", "C3"}
    assert {c.chair_id for c in st.for_camera("CAM1")} == {"C1", "C2"}


def test_chair_store_enabling(tmp_db):
    st = ChairStore(tmp_db)
    st.add(chair_id="C1", camera_id="CAM1", zone_id="Z1")
    st.set_enabled("C1", False)
    assert st.get("C1").enabled is False
    assert st.get("C1").enabled is False
    enabled = st.list(enabled_only=True)
    assert all(c.enabled for c in enabled)
    assert "C1" not in {c.chair_id for c in enabled}


def test_seed_defaults_missing_file_is_safe(tmp_path):
    missing = tmp_path / "chairs-missing.json"
    assert load_chair_defaults(str(missing)) == []


def test_chair_event_types_present():
    assert CHAIR_OCCUPIED
    assert CHAIR_VACATED
    assert CHAIR_IDENTITY_CONFIRMED
    assert CHAIR_IDENTITY_UNKNOWN
    assert CHAIR_ASSIGNMENT_MISMATCH
    assert len({CHAIR_OCCUPIED, CHAIR_VACATED, CHAIR_IDENTITY_CONFIRMED,
                CHAIR_IDENTITY_UNKNOWN, CHAIR_ASSIGNMENT_MISMATCH}) == 5


# ----------------------------------------------------------------------
# FUTURE thresholds (centralised configuration)
# ----------------------------------------------------------------------

def test_future_thresholds_centralised():
    assert float(config.FUTURE_AWAY_THRESHOLD_SECONDS) == 10.0
    assert float(config.FUTURE_PHONE_THRESHOLD_SECONDS) == 5.0
    assert float(config.FUTURE_TALKING_THRESHOLD_SECONDS) == 15.0


# ----------------------------------------------------------------------
# Employee display_name with stable identity key
# ----------------------------------------------------------------------

def test_employee_display_name_preserves_employee_id(tmp_db):
    from src.database import list_employees, upsert_employee
    upsert_employee(tmp_db, employee_id="EMP001", name="Sourabh",
                    display_name="Sourabh")
    upsert_employee(tmp_db, employee_id="EMP001", name="Sourabh",
                    display_name="Rahul")
    emps = [e for e in list_employees(tmp_db) if e["employee_id"] == "EMP001"]
    assert len(emps) == 1                      # no duplicate employee record
    assert emps[0]["employee_id"] == "EMP001"  # stable identity key
    assert emps[0]["display_name"] == "Rahul"
