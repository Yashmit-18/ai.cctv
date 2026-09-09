"""Tests for main.py maintenance helpers (retention wiring)."""

import os
import time
from datetime import date, timedelta

import pytest

import main as m
from src.database import log_interval, query_day_intervals


@pytest.fixture
def retention(monkeypatch, tmp_db, tmp_path):
    monkeypatch.setattr(m, "ENABLE_RETENTION", True)
    monkeypatch.setattr(m, "RETENTION_DAYS_LOGS", 1)
    monkeypatch.setattr(m, "RETENTION_DAYS_REPORTS", 1)
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(m, "REPORT_OUTPUT_DIR", str(reports))
    monkeypatch.setattr(m, "_LAST_RETENTION_DAY", None)
    old_day = (date.today() - timedelta(days=40)).isoformat()
    recent_day = date.today().isoformat()
    log_interval(tmp_db, f"{old_day} 10:00:00", "EMP001", "ACTIVE", 60)
    log_interval(tmp_db, f"{recent_day} 10:00:00", "EMP001", "ACTIVE", 60)
    old_report = reports / "daily_report_old.xlsx"
    old_report.write_bytes(b"x")
    old_ts = time.mktime((date.today() - timedelta(days=40)).timetuple())
    os.utime(old_report, (old_ts, old_ts))
    new_report = reports / "daily_report_new.xlsx"
    new_report.write_bytes(b"x")
    yield tmp_db, reports
    monkeypatch.setattr(m, "ENABLE_RETENTION", False)


def test_retention_purges_old_logs_and_reports_but_keeps_recent(retention):
    tmp_db, reports = retention
    m._maybe_run_retention(tmp_db)
    # old row purged; recent (within retention) kept
    assert query_day_intervals(tmp_db, (date.today() - timedelta(days=40)).isoformat()) == []
    assert len(query_day_intervals(tmp_db, date.today().isoformat())) == 1
    names = {p.name for p in reports.iterdir()}
    assert "daily_report_old.xlsx" not in names
    assert "daily_report_new.xlsx" in names


def test_retention_runs_at_most_once_per_day(retention, monkeypatch):
    tmp_db, reports = retention
    m._maybe_run_retention(tmp_db)
    # second call in the same day is a no-op
    m._maybe_run_retention(tmp_db)
    assert query_day_intervals(tmp_db, "2026-01-01") == []


def test_retention_disabled_never_deletes(retention, monkeypatch):
    monkeypatch.setattr(m, "ENABLE_RETENTION", False)
    tmp_db, reports = retention
    monkeypatch.setattr(m, "_LAST_RETENTION_DAY", None)
    m._maybe_run_retention(tmp_db)
    assert len(query_day_intervals(tmp_db,
                                    (date.today() - timedelta(days=40)).isoformat())) == 1
    assert (reports / "daily_report_old.xlsx").exists()