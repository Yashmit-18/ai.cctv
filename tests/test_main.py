"""Tests for main.py maintenance helpers (retention wiring)."""

import os
import time
from datetime import date, timedelta

import pytest

import main as m
from src.database import get_connection, init_db, log_interval, query_day_intervals, query_security_events
from src.domain import CAMERA_OFFLINE, CAMERA_RECOVERED


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


def test_all_camera_outage_emits_offline_then_recovery_from_daemon_loop(tmp_path, monkeypatch):
    """Health transitions run when no camera yields a usable frame.

    This drives the production loop with an empty frame batch throughout.  It
    verifies that the normal security FSM still persists CAMERA_OFFLINE after
    its threshold and CAMERA_RECOVERED once the health snapshot returns online.
    No detector is constructed and no employee state is fabricated.
    """
    db_path = tmp_path / "outage.db"
    conn = get_connection(str(db_path))
    init_db(conn)
    conn.close()

    class Clock:
        now = 0.0
        sleeps = 0

        @classmethod
        def time(cls):
            return cls.now

        @classmethod
        def sleep(cls, _seconds):
            cls.sleeps += 1
            if cls.sleeps >= 4:
                raise KeyboardInterrupt
            cls.now += 20.0

    class EmptyCameraPool:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def frame_interval(_rate):
            return 1.0

        def start(self):
            return self

        def stop(self):
            pass

        def camera_health(self):
            health = "ONLINE" if Clock.now in (0.0, 60.0) else "OFFLINE"
            return {"cam_01": {"health": health, "usable": False}}

        def latest_frames(self, **_kwargs):
            return {}

    class NoopScheduler:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(m, "DB_PATH", str(db_path))
    monkeypatch.setattr(m, "CAMERAS", {"cam_01": "rtsp://test"})
    monkeypatch.setattr(m, "MultiCameraManager", EmptyCameraPool)
    monkeypatch.setattr(m, "EODScheduler", NoopScheduler)
    monkeypatch.setattr(m, "_resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(m, "_write_live_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(m, "_shutdown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(m.time, "time", Clock.time)
    monkeypatch.setattr(m.time, "sleep", Clock.sleep)

    args = m.argparse.Namespace(
        validate_config=False, test_email=False, max_batch=None, device="cpu",
        source=None, headless=True, no_email=True,
    )
    m.run(args)

    check = get_connection(str(db_path))
    try:
        events = query_security_events(check, limit=20)
    finally:
        check.close()
    event_types = [event["event_type"] for event in events]
    assert CAMERA_OFFLINE in event_types
    assert CAMERA_RECOVERED in event_types
    assert not {"ACTIVE", "AWAY", "ON_PHONE", "ARRIVED", "LEFT"} & set(event_types)
