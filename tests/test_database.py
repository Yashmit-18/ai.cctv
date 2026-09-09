"""Tests for the SQLite schema, migrations, CRUD and retention."""

import sqlite3
from datetime import date, timedelta

import pytest

from src import database as db
from src.database import (
    delete_employee,
    get_connection,
    get_employee,
    init_db,
    list_employees,
    log_interval,
    query_day_intervals,
    query_range_intervals,
    retention_cleanup,
    set_employee_active,
    set_employee_enrolled,
    upsert_employee,
)


# ---------------------------------------------------------------- schema / migration

def test_fresh_db_has_expected_columns(tmp_path):
    path = tmp_path / "fresh.db"
    conn = get_connection(str(path))
    init_db(conn)
    emp_cols = {r[1] for r in conn.execute("PRAGMA table_info(employees)").fetchall()}
    log_cols = {r[1] for r in conn.execute("PRAGMA table_info(activity_logs)").fetchall()}
    assert {"employee_id", "name", "department", "designation", "active",
            "enrolled", "created_at", "updated_at"} <= emp_cols
    assert {"timestamp", "date", "employee_id", "state",
            "duration_seconds", "source"} <= log_cols
    idx = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
    assert "idx_activity_emp_date" in idx
    conn.close()


def test_migration_of_legacy_schema_backfills_date(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            employee_id TEXT NOT NULL,
            state TEXT NOT NULL,
            duration_seconds REAL
        )
    """)
    conn.execute(
        "INSERT INTO activity_logs (timestamp, employee_id, state, duration_seconds) "
        "VALUES (?, ?, ?, ?)",
        ("2026-08-31 09:00:00", "EMP001", "ACTIVE", 60.0),
    )
    conn.commit()
    conn.close()

    conn = get_connection(str(path))   # re-open (WAL, etc.)
    init_db(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(activity_logs)").fetchall()}
    assert "date" in cols and "source" in cols
    row = conn.execute("SELECT date, source FROM activity_logs").fetchone()
    assert row[0] == "2026-08-31"
    assert row[1] is not None or row[1] == ""
    conn.close()


def test_init_db_is_idempotent(tmp_db):
    init_db(tmp_db)   # second run must not throw
    assert query_day_intervals(tmp_db, "2026-09-01") == []


def test_migrates_legacy_desk_employees_schema(tmp_path):
    """Phase-1 prototypes stored desk ROIs; init_db must rebuild that table so
    upsert_employee does not fail with 'no such column: department'."""
    path = tmp_path / "legacy_emp.db"
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL DEFAULT '',
            desk_x REAL, desk_y REAL, desk_w REAL, desk_h REAL
        )
    """)
    conn.execute("INSERT INTO employees (employee_id, name, desk_x, desk_y, desk_w, desk_h) "
                 "VALUES ('EMP001', 'Legacy User', 10, 20, 30, 40)")
    conn.commit()
    conn.close()

    conn = get_connection(str(path))
    init_db(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(employees)").fetchall()}
    assert {"employee_id", "name", "department", "designation",
            "active", "enrolled"} <= cols
    assert "desk_x" not in cols
    # Old values preserved, and new-schema writes now succeed:
    row = get_employee(conn, "EMP001")
    assert row["name"] == "Legacy User"
    assert upsert_employee(conn, "EMP001", department="Engineering") is False
    assert get_employee(conn, "EMP001")["department"] == "Engineering"
    # Ensure activity_logs migration still applies on the same DB:
    assert "date" in {r[1] for r in conn.execute("PRAGMA table_info(activity_logs)")}
    conn.close()


# ---------------------------------------------------------------- CRUD

def test_upsert_insert_and_update(tmp_db):
    assert upsert_employee(tmp_db, "EMP001", name="Alice", department="IT",
                           designation="Engineer") is True
    assert upsert_employee(tmp_db, "EMP001", name="Alice Jones") is False
    row = get_employee(tmp_db, "EMP001")
    assert row["name"] == "Alice Jones"
    assert row["department"] == "IT"      # partial update preserves other fields
    assert row["designation"] == "Engineer"


def test_upsert_empty_string_clears_field(tmp_db):
    upsert_employee(tmp_db, "EMP001", name="Alice", department="IT")
    upsert_employee(tmp_db, "EMP001", department="")
    assert get_employee(tmp_db, "EMP001")["department"] == ""
    assert get_employee(tmp_db, "EMP001")["name"] == "Alice"


def test_employee_active_enrolled_flags(tmp_db):
    upsert_employee(tmp_db, "EMP001")
    set_employee_enrolled(tmp_db, "EMP001", True)
    set_employee_active(tmp_db, "EMP001", False)
    emp = get_employee(tmp_db, "EMP001")
    assert emp["enrolled"] == 1 and emp["active"] == 0
    active_only = list_employees(tmp_db, include_inactive=False)
    assert active_only == []


def test_delete_employee(tmp_db):
    upsert_employee(tmp_db, "EMP001")
    delete_employee(tmp_db, "EMP001")
    assert get_employee(tmp_db, "EMP001") is None


def test_orders_by_employee_id(tmp_db):
    upsert_employee(tmp_db, "EMP09")
    upsert_employee(tmp_db, "EMP2")
    assert [e["employee_id"] for e in list_employees(tmp_db)] == ["EMP09", "EMP2"]


# ---------------------------------------------------------------- queries

def test_query_day_and_range_by_date_column(tmp_db):
    log_interval(tmp_db, "2026-08-31 23:00:00", "EMP001", "ACTIVE", 60, source="cam1")
    log_interval(tmp_db, "2026-09-01 00:30:00", "EMP001", "ACTIVE", 60, source="cam1")
    assert len(query_day_intervals(tmp_db, "2026-09-01")) == 1
    assert len(query_range_intervals(tmp_db, "2026-08-31", "2026-09-01")) == 2
    assert len(query_range_intervals(tmp_db, "2026-09-02", "2026-09-05")) == 0


def test_log_interval_sets_source_none_for_empty(tmp_db):
    log_interval(tmp_db, "2026-09-01 10:00:00", "EMP001", "ACTIVE", 10)
    row = tmp_db.execute(
        "SELECT employee_id, state, duration_seconds, source FROM activity_logs").fetchone()
    assert row[3] is None or row[3] == ""


# ---------------------------------------------------------------- retention

def test_retention_cleanup_removes_only_old(tmp_db):
    today = date.today()
    old = (today - timedelta(days=40)).isoformat()
    recent = today.isoformat()
    log_interval(tmp_db, f"{old} 10:00:00", "EMP001", "ACTIVE", 60)
    log_interval(tmp_db, f"{recent} 10:00:00", "EMP001", "ACTIVE", 60)
    removed = retention_cleanup(tmp_db, days=1)
    assert removed >= 1
    assert query_day_intervals(tmp_db, recent)  # recent row kept


def test_retention_cleanup_negative_is_safe_noop(tmp_db):
    assert retention_cleanup(tmp_db, days=-1) == 0