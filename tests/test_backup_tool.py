"""Tests for the WAL-safe backup tool."""

import os
import sqlite3

import pytest

from src.backup_tool import backup_all, backup_database, verify_integrity
from src import database as db


def test_backup_database_creates_file(tmp_db, tmp_path):
    db_path = tmp_db.execute("PRAGMA database_list").fetchone()[2]
    backup_dir = str(tmp_path / "backups")
    dst = backup_database(db_path, backup_dir)
    assert os.path.isfile(dst)
    assert verify_integrity(dst) == "ok"


def test_backup_includes_data(tmp_db, tmp_path):
    db_path = tmp_db.execute("PRAGMA database_list").fetchone()[2]
    # insert some data
    db.upsert_employee(tmp_db, "EMP001", name="Alice", department="Eng")
    backup_dir = str(tmp_path / "backups")
    dst = backup_database(db_path, backup_dir)
    conn = sqlite3.connect(dst)
    rows = conn.execute("SELECT COUNT(*) FROM employees WHERE employee_id='EMP001'").fetchone()
    conn.close()
    assert rows[0] == 1


def test_backup_all(tmp_path):
    project_root = tmp_path / "proj"
    (project_root / "data" / "database").mkdir(parents=True)
    (project_root / "data" / "faces").mkdir(parents=True)
    (project_root / "data" / "reports").mkdir(parents=True)
    db_path = str(project_root / "data" / "database" / "sessions.db")
    conn = sqlite3.connect(db_path)
    db.init_db(conn)
    db.upsert_employee(conn, "EMP001")
    conn.close()

    backup_dir = str(tmp_path / "bk")
    results = backup_all(str(project_root), backup_dir)
    assert "database" in results
    assert os.path.isfile(results["database"])
    assert verify_integrity(results["database"]) == "ok"


def test_verify_integrity_bad_file(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a database")
    assert verify_integrity(str(bad)) != "ok"
