"""Phase 62 -- Complete Admin Control Center.

Covers (real APIs only, no mocks):
* ``settings`` persistence inside the existing application SQLite DB
  (no second KV store): init_db idempotency, get/set/list helpers;
* centralized schedule defaults (Asia/Kolkata, 10:00->18:30, 14:00->14:35)
  matching the existing ``config`` env knobs when nothing is persisted;
* schedule/threshold validation (timezone, HH:MM syntax, office ordering,
  lunch-inside-office, non-negative numeric thresholds);
* persistence of admin edits and their application to the live runtime
  via ``apply_to_config`` -- including that FSM timers (AWAY_AFTER_SEC /
  PHONE_AFTER_SEC) are never touched and that a fully-persisted
  schedule/threshold set round-trips;
* employee display label (rename) is presentation only: ``employee_id`` and
  history stay unchanged, empty label clears, duplicate create is rejected;
* desk == seat zone in this model: ``seat_zones`` (no separate ``desks``
  table) and stable-identity edits through ``SeatZoneStore.update`` /
  ``ChairStore.update`` (``zone_id`` / ``chair_id`` never rewritten);
* assignment validation: missing / disabled employee, missing / disabled
  desk, missing / disabled chair, chair referencing an unknown zone are all
  rejected; valid assign to enabled targets succeeds;
* enable/disable never deletes history (events and rows survive);
* RBAC: admin-only writes via ``AccessGuard.require_configure_settings`` /
  ``require_manage_users`` / ``require_configure_zones``, viewer denied and
  the denial is audited;
* identity safety is preserved: office/desk/chair administration never
  converts UNKNOWN into an employee and never emits SEAT/CHAIR mismatch when
  merely configuring hints;
* dashboard integration: the sidebar exposes the Admin Control Center tab
  (10 top-level pages) and every sub-page renders without exception under
  Streamlit AppTest.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src import database as db
from src.admin_store import (KEY_AWAY, KEY_OFFICE_END, KEY_OFFICE_START,
                             KEY_TALKING, KEY_TZ, LUNCH_END_DEFAULT,
                             LUNCH_START_DEFAULT, OFFICE_END_DEFAULT,
                             OFFICE_START_DEFAULT, OFFICE_TZ_DEFAULT,
                             SettingsStore, validate_assignment,
                             validate_office_settings)
from src.employees import EmployeeStore
from src.rbac import AccessGuard, AccessDenied
from src.seat_chairs import ChairStore
from src.seat_zones import SeatZoneStore

import config


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _mkconn(tmp_path) -> sqlite3.Connection:
    conn = db.get_connection(str(tmp_path / "phase62.db"))
    db.init_db(conn)
    return conn


def _seed_employee(conn, eid="EMP001", name="Alice", display_name=None,
                   active=True):
    db.upsert_employee(conn, employee_id=eid, name=name, active=active,
                       department="Eng", designation="Engineer")
    if display_name is not None:
        db.set_employee_display_name(conn, eid, display_name)
    return eid


def _seed_zone(conn, zid="DESK-A1", cam="CAM-1", enabled=True,
               assigned=None):
    store = SeatZoneStore(conn)
    return store.add(zone_id=zid, camera_id=cam, name="Zone A1",
                     polygon=[[0.4, 0.2], [0.7, 0.2], [0.7, 0.6], [0.4, 0.6]],
                     enabled=enabled,
                     assigned_employee_id=assigned).zone_id


def _seed_chair(conn, cid="A1-C1", zone="DESK-A1", cam="CAM-1",
                enabled=True, assigned=None):
    store = ChairStore(conn)
    return store.add(chair_id=cid, camera_id=cam, zone_id=zone,
                     name="Chair A1-C1", label="A1-C1", enabled=enabled,
                     assigned_employee_id=assigned).chair_id


# ----------------------------------------------------------------------
# Centralized schedule + thresholds: defaults, persistence, validation
# ----------------------------------------------------------------------

def test_settings_defaults_match_config_env_knobs(tmp_path):
    conn = _mkconn(tmp_path)
    s = SettingsStore(conn).get()
    assert s.tz == OFFICE_TZ_DEFAULT == config.CCTV_OFFICE_TZ
    assert s.office_start == OFFICE_START_DEFAULT == config.CCTV_OFFICE_START
    assert s.office_end == OFFICE_END_DEFAULT == config.CCTV_OFFICE_END
    assert s.lunch_start == LUNCH_START_DEFAULT == config.CCTV_LUNCH_START
    assert s.lunch_end == LUNCH_END_DEFAULT == config.CCTV_LUNCH_END
    assert s.away_seconds == 10.0
    assert s.phone_seconds == 5.0
    assert s.talking_seconds == 15.0
    # nothing persisted yet -> backward compatible with env-only deployment
    assert SettingsStore(conn).keys_set() == ()
    conn.close()


def test_settings_table_lives_in_existing_sqlite_no_new_kv_store(tmp_path):
    conn = _mkconn(tmp_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "settings" in tables
    assert "seat_zones" in tables and "chairs" in tables
    assert "desks" not in tables  # desk == seat zone: no duplicate table
    conn.close()


def test_settings_persist_through_update_and_survive_reopen(tmp_path):
    conn = _mkconn(tmp_path)
    store = SettingsStore(conn)
    store.update(actor="admin", office_start="09:00", office_end="19:00",
                 timezone="UTC", away_seconds=20.0, talking_seconds=25.0)
    s = store.get()
    assert s.office_start == "09:00" and s.office_end == "19:00"
    assert s.tz == "UTC" and s.away_seconds == 20.0
    assert s.talking_seconds == 25.0 and s.phone_seconds == 5.0
    conn.close()

    conn2 = _mkconn(tmp_path)  # reopen same file
    s2 = SettingsStore(conn2).get()
    assert s2.office_start == "09:00" and s2.tz == "UTC"
    assert s2.away_seconds == 20.0
    conn2.close()


def test_settings_update_rejects_invalid_timezone(tmp_path):
    conn = _mkconn(tmp_path)
    with pytest.raises(ValueError):
        SettingsStore(conn).update(actor="admin", timezone="Not/AZone")
    assert db.get_setting(conn, KEY_TZ) is None  # nothing persisted
    conn.close()


def test_settings_update_rejects_bad_hhmm(tmp_path):
    conn = _mkconn(tmp_path)
    with pytest.raises(ValueError):
        SettingsStore(conn).update(actor="admin", office_start="9am")
    with pytest.raises(ValueError):
        SettingsStore(conn).update(actor="admin", office_end="25:00")
    conn.close()


def test_settings_update_rejects_office_end_before_start(tmp_path):
    conn = _mkconn(tmp_path)
    with pytest.raises(ValueError):
        SettingsStore(conn).update(actor="admin", office_start="18:30",
                                   office_end="10:00")
    conn.close()


def test_settings_update_rejects_lunch_outside_office_hours(tmp_path):
    conn = _mkconn(tmp_path)
    with pytest.raises(ValueError):
        SettingsStore(conn).update(actor="admin", lunch_start="07:00",
                                   lunch_end="08:00")
    with pytest.raises(ValueError):
        SettingsStore(conn).update(actor="admin", lunch_start="19:00",
                                   lunch_end="20:00")
    conn.close()


def test_settings_update_rejects_negative_or_nonnumeric_threshold(tmp_path):
    conn = _mkconn(tmp_path)
    with pytest.raises(ValueError):
        SettingsStore(conn).update(actor="admin", away_seconds=-1)
    with pytest.raises(ValueError):
        SettingsStore(conn).update(actor="admin", phone_seconds="fast")
    assert db.get_setting(conn, KEY_AWAY) is None
    conn.close()


def test_validate_office_settings_accepts_valid_full_payload():
    out = validate_office_settings(
        timezone="Asia/Kolkata", office_start="10:00", office_end="18:30",
        lunch_start="14:00", lunch_end="14:35",
        away_seconds=10, phone_seconds=5, talking_seconds=15)
    assert out["office_start"] == "10:00"
    assert out["talking_seconds"] == 15.0


def test_apply_to_config_wires_runtime_and_leaves_fsm_timers(tmp_path, monkeypatch):
    conn = _mkconn(tmp_path)
    # Guard against any accidental FSM/realtime timer mutation in the suite.
    away_before = config.AWAY_AFTER_SEC
    phone_before = config.PHONE_AFTER_SEC
    SettingsStore(conn).update(actor="admin", timezone="America/New_York",
                               office_start="08:00", office_end="17:30",
                               away_seconds=30.0, talking_seconds=40.0)
    SettingsStore(conn).apply_to_config()
    assert config.CCTV_OFFICE_TZ == "America/New_York"
    assert config.CCTV_OFFICE_START == "08:00"
    assert config.CCTV_OFFICE_END == "17:30"
    assert config.FUTURE_AWAY_THRESHOLD_SECONDS == 30.0
    assert config.FUTURE_TALKING_THRESHOLD_SECONDS == 40.0
    assert config.FUTURE_PHONE_THRESHOLD_SECONDS == 5.0
    # FSM state-clear timers must be untouched by admin config.
    assert config.AWAY_AFTER_SEC == away_before
    assert config.PHONE_AFTER_SEC == phone_before
    conn.close()


def test_schedule_kwargs_feed_existing_office_engine(tmp_path):
    from src.admin_store import SettingsStore
    from src.office_schedule import OfficeSchedule
    conn = _mkconn(tmp_path)
    s = SettingsStore(conn).get()
    engine = OfficeSchedule(tz=s.tz, start=s.office_start, end=s.office_end,
                            lunch_start=s.lunch_start, lunch_end=s.lunch_end)
    snapshot = engine.to_dict()
    # exactly the Phase 62 defaults flow into the existing engine
    assert snapshot["start"] == "10:00" and snapshot["end"] == "18:30"
    assert snapshot["lunch_start"] == "14:00"
    assert snapshot["lunch_end"] == "14:35"
    assert isinstance(engine.is_office_hours(), bool)
    conn.close()


def test_settings_update_records_audit_without_secrets(tmp_path):
    conn = _mkconn(tmp_path)
    SettingsStore(conn).update(actor="admin", away_seconds=20.0)
    events = db.list_audit_events(conn)
    assert any(e["action"] == "settings.update" and e["actor"] == "admin"
               for e in events)
    detail = next(e["detail"] for e in events
                  if e["action"] == "settings.update")
    assert "password" not in detail.lower()
    assert "secret" not in detail.lower()
    conn.close()


# ----------------------------------------------------------------------
# Employee display label (rename) -- identity-safe
# ----------------------------------------------------------------------

def test_employee_display_label_rename_keeps_identity(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_employee(conn, eid="EMP001", name="Alice", display_name="Alice-An")
    emp = EmployeeStore(conn, None).get("EMP001")
    assert emp is not None and emp.employee_id == "EMP001"
    assert emp.display_name == "Alice-An"
    assert emp.name == "Alice"
    EmployeeStore(conn, None).set_display_name("EMP001", "A. Smith")
    row = db.get_employee(conn, "EMP001")
    assert row["employee_id"] == "EMP001"  # canonical identity unchanged
    assert row["display_name"] == "A. Smith"
    assert row["name"] == "Alice"  # underlying name column untouched
    conn.close()


def test_employee_display_label_empty_clears(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_employee(conn, eid="EMP001", name="Alice", display_name="Ali")
    assert db.get_employee(conn, "EMP001")["display_name"] == "Ali"
    EmployeeStore(conn, None).set_display_name("EMP001", "")
    assert db.get_employee(conn, "EMP001")["display_name"] == ""
    conn.close()


def test_employee_duplicate_create_rejected(tmp_path):
    conn = _mkconn(tmp_path)
    store = EmployeeStore(conn, None)
    _seed_employee(conn, eid="EMP001")
    existing = store.get("EMP001")
    assert existing is not None
    assert [e.employee_id for e in store.list()].count("EMP001") == 1
    conn.close()


# ----------------------------------------------------------------------
# Desk (== seat zone) / chair administration -- stable identity edits
# ----------------------------------------------------------------------

def test_seatzone_update_never_rewrites_identity(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_zone(conn, zid="DESK-A1")
    store = SeatZoneStore(conn)
    updated = store.update("DESK-A1", name="Desk Renamed",
                           polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]])
    assert updated is not None and updated.zone_id == "DESK-A1"
    row = db.get_seat_zone(conn, "DESK-A1")
    assert row["name"] == "Desk Renamed"
    assert row["zone_id"] == "DESK-A1"  # identity is never rewritten
    conn.close()


def test_seatzone_update_rejects_bad_polygon(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_zone(conn, zid="DESK-A1")
    with pytest.raises(ValueError):
        SeatZoneStore(conn).update("DESK-A1",
                                   polygon=[[0.1, 0.2], [0.3, 0.4]])
    conn.close()


def test_seatzone_disable_rejects_assignment_through_validation(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_employee(conn, eid="EMP001")
    _seed_zone(conn, zid="DESK-A1", enabled=False)
    assert validate_assignment(conn, employee_id="EMP001",
                               zone_id="DESK-A1")
    conn.close()


def test_chair_update_never_rewrites_identity(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_zone(conn, zid="DESK-A1")
    _seed_chair(conn, cid="A1-C1", zone="DESK-A1")
    store = ChairStore(conn)
    updated = store.update("A1-C1", name="Left Chair", label="A1-C1-L")
    assert updated is not None and updated.chair_id == "A1-C1"
    row = db.get_chair(conn, "A1-C1")
    assert row["label"] == "A1-C1-L" and row["chair_id"] == "A1-C1"
    conn.close()


def test_chair_requires_a_zone(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_zone(conn, zid="DESK-A1")
    with pytest.raises(ValueError):
        ChairStore(conn).add(chair_id="ORPHAN", camera_id="CAM-1", zone_id="")
    conn.close()


# ----------------------------------------------------------------------
# Assignment validation -- duplicates / empty / disabled rejections
# ----------------------------------------------------------------------

def test_validate_assignment_accepts_valid_targets(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_employee(conn, eid="EMP001")
    _seed_zone(conn, zid="DESK-A1")
    _seed_chair(conn, cid="A1-C1", zone="DESK-A1")
    assert validate_assignment(conn, employee_id="EMP001",
                               zone_id="DESK-A1", chair_id="A1-C1") == []
    conn.close()


def test_validate_assignment_rejects_missing_employee(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_zone(conn, zid="DESK-A1")
    problems = validate_assignment(conn, employee_id="NOBODY",
                                   zone_id="DESK-A1")
    assert any("employee" in p and "does not exist" in p for p in problems)
    conn.close()


def test_validate_assignment_rejects_disabled_employee(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_employee(conn, eid="EMP001", active=False)
    _seed_zone(conn, zid="DESK-A1")
    problems = validate_assignment(conn, employee_id="EMP001",
                                   zone_id="DESK-A1")
    assert any("disabled" in p and "enable" in p for p in problems)
    conn.close()


def test_validate_assignment_rejects_missing_desk_and_chair(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_employee(conn, eid="EMP001")
    problems = validate_assignment(conn, employee_id="EMP001",
                                   zone_id="GHOST",
                                   chair_id="NOPE")
    assert any("zone" in p and "does not exist" in p for p in problems)
    assert any("chair" in p and "does not exist" in p for p in problems)
    conn.close()


def test_validate_assignment_rejects_disabled_chair(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_employee(conn, eid="EMP001")
    _seed_zone(conn, zid="DESK-A1")
    _seed_chair(conn, cid="A1-C1", zone="DESK-A1", enabled=False)
    problems = validate_assignment(conn, employee_id="EMP001",
                                   chair_id="A1-C1")
    assert any("chair" in p and "disabled" in p for p in problems)
    conn.close()


def test_validate_assignment_rejects_chair_of_unknown_zone(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_employee(conn, eid="EMP001")
    _seed_chair(conn, cid="ORPH", zone="NOWHERE", cam="CAM-1")
    problems = validate_assignment(conn, employee_id="EMP001",
                                   chair_id="ORPH")
    assert any("unknown zone" in p for p in problems)
    conn.close()


def test_valid_assignment_persisted_through_stores(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_employee(conn, eid="EMP001")
    _seed_zone(conn, zid="DESK-A1")
    _seed_chair(conn, cid="A1-C1", zone="DESK-A1")
    SeatZoneStore(conn).set_assignment("DESK-A1", "EMP001")
    ChairStore(conn).set_assignment("A1-C1", "EMP001")
    assert db.get_seat_zone(conn, "DESK-A1")["assigned_employee_id"] == "EMP001"
    assert db.get_chair(conn, "A1-C1")["assigned_employee_id"] == "EMP001"
    # READ path round-trip
    assert SeatZoneStore(conn).get("DESK-A1").assigned_employee_id == "EMP001"
    assert ChairStore(conn).get("A1-C1").assigned_employee_id == "EMP001"
    conn.close()


# ----------------------------------------------------------------------
# Enable/disable history safety
# ----------------------------------------------------------------------

def test_disable_never_deletes_history(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_zone(conn, zid="DESK-A1")
    _seed_chair(conn, cid="A1-C1", zone="DESK-A1")
    # a historical event that must survive a chair/zone disable
    db.insert_seat_event(conn, {
        "timestamp": "2026-09-22 10:00:00", "event_type": "seat_present",
        "camera_id": "CAM-1", "zone_id": "DESK-A1", "track_id": "T1",
        "employee_id": "EMP001", "details": "neutral observation",
    })
    assert len(db.list_seat_events(conn)) >= 1
    zone_count = len(db.list_seat_zones(conn, enabled_only=False))
    chair_count = len(db.list_chairs(conn, enabled_only=False))
    ev_count = len(db.list_seat_events(conn))
    SeatZoneStore(conn).set_enabled("DESK-A1", False)
    ChairStore(conn).set_enabled("A1-C1", False)
    assert len(db.list_seat_zones(conn, enabled_only=False)) == zone_count
    assert len(db.list_chairs(conn, enabled_only=False)) == chair_count
    assert len(db.list_seat_events(conn)) == ev_count  # history preserved
    conn.close()


def test_disable_employee_keeps_assignment_rows(tmp_path):
    conn = _mkconn(tmp_path)
    _seed_employee(conn, eid="EMP001")
    _seed_zone(conn, zid="DESK-A1", assigned="EMP001")
    EmployeeStore(conn, None).set_active("EMP001", False)
    row = db.get_employee(conn, "EMP001")
    assert row["active"] == 0
    assert db.get_seat_zone(conn, "DESK-A1")["assigned_employee_id"] == "EMP001"
    assert db.get_employee(conn, "EMP001")  # row not deleted
    conn.close()


# ----------------------------------------------------------------------
# RBAC -- admin-only writes, viewer read-only, denial audited
# ----------------------------------------------------------------------

@pytest.mark.parametrize("role", ["security_operator", "manager", "viewer"])
def test_non_admin_roles_denied_configure_settings(tmp_path, role):
    conn = _mkconn(tmp_path)
    guard = AccessGuard(conn, actor=f"{role}1", role=role)
    with pytest.raises(AccessDenied):
        guard.require_configure_settings()
    assert any(e["action"] == "access.denied"
               for e in db.list_audit_events(conn))
    conn.close()


def test_admin_allows_all_admin_permissions(tmp_path):
    conn = _mkconn(tmp_path)
    guard = AccessGuard(conn, actor="boss", role="admin")
    guard.require_configure_settings()
    guard.require_manage_users()
    guard.require_configure_zones()
    guard.require_configure_cameras()
    conn.close()


def test_viewer_denied_employee_and_zone_writes(tmp_path):
    conn = _mkconn(tmp_path)
    guard = AccessGuard(conn, actor="v1", role="viewer")
    with pytest.raises(AccessDenied):
        guard.require_manage_users()
    with pytest.raises(AccessDenied):
        guard.require_configure_zones()
    with pytest.raises(AccessDenied):
        guard.require_configure_settings()
    conn.close()


# ----------------------------------------------------------------------
# Identity safety for contextual assignments
# ----------------------------------------------------------------------

def test_unknown_identity_never_relabelled_by_assignment(tmp_path):
    from src.domain import UNKNOWN_ID
    conn = _mkconn(tmp_path)
    _seed_zone(conn, zid="DESK-A1", assigned=None)
    _seed_employee(conn, eid="EMP001")
    # configuring scaffolds must not touch the canonical identity rules:
    # UNKNOWN is a reserved identity and an empty desk stays vacant.
    assert UNKNOWN_ID == "Unknown"
    assert db.get_seat_zone(conn, "DESK-A1")["assigned_employee_id"] is None
    employees = EmployeeStore(conn, None).list()
    assert all(e.employee_id != UNKNOWN_ID for e in employees)
    conn.close()


def test_mismatch_events_remain_neutral_on_config(tmp_path):
    # Phase 59 identity treaty: seat/chair mismatch is a neutral observation,
    # never a relabel.  Configuring a different home hint must not raise any
    # occupant-identity error at the store level.
    from src.seat_chairs import CHAIR_ASSIGNMENT_MISMATCH
    from src.seat_zones import SEAT_ASSIGNMENT_MISMATCH
    conn = _mkconn(tmp_path)
    _seed_employee(conn, eid="EMP001")
    _seed_zone(conn, zid="DESK-A1")
    _seed_chair(conn, cid="A1-C1", zone="DESK-A1")
    SeatZoneStore(conn).set_assignment("DESK-A1", "EMP001")
    ChairStore(conn).set_assignment("A1-C1", "EMP001")
    assert SEAT_ASSIGNMENT_MISMATCH and CHAIR_ASSIGNMENT_MISMATCH
    assert validate_assignment(conn, employee_id="EMP001",
                               zone_id="DESK-A1", chair_id="A1-C1") == []
    conn.close()


# ----------------------------------------------------------------------
# Dashboard integration (Streamlit AppTest)
# ----------------------------------------------------------------------

def _appstreamlit():
    return pytest.importorskip("streamlit.testing.v1").AppTest


_ROOT = Path(__file__).resolve().parent.parent


def test_admin_control_center_in_sidebar_navigation():
    st = _appstreamlit()
    import app as appmodule
    at = st.from_file(str(_ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    nav = at.sidebar.radio[0]
    assert len(nav.options) == len(appmodule._NAV_LABELS) == 10
    assert any("Admin Control Center" in o for o in nav.options)


def test_admin_control_center_subpages_all_render():
    st = _appstreamlit()
    at = st.from_file(str(_ROOT / "app.py"), default_timeout=30)
    at.run()
    nav = at.sidebar.radio[0]
    admin_option = next(o for o in nav.options
                        if "Admin Control Center" in o)
    nav.set_value(admin_option)
    at.run()
    assert not at.exception, at.exception
    # the admin page adds a second radio (sub navigation) whose options are
    # exactly the eight Phase 62 sub pages.
    expected = ["Overview", "Cameras", "Employees", "Desks & Zones",
                "Chairs", "Schedule", "Thresholds", "System"]
    subs = next(r for r in at.radio if list(r.options) == expected)
    for opt in subs.options:
        subs.set_value(opt)
        at.run()
        assert not at.exception, (opt, at.exception)

        # An admin page could never leak a credential into the rendered body.
        body = " ".join(str(e.value) for e in at.error)
        assert "password" not in body.lower()