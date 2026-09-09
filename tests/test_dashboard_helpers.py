"""Pure-helper tests for the dashboard (Phase 9-11).

Only ``_live_overview_metrics`` is tested here (no Streamlit runtime needed).
The function must count OBSERVED employees only, exclude Unknown from the
employee-present count, and surface camera health unchanged.
"""

import pickle
from pathlib import Path

import pandas as pd

from app import (
    _faces_dir,
    _live_overview_metrics,
    cached_enrollment_status,
    enrollment_status_frame,
    save_enrollment_image,
)


def test_recognised_counts_observed_only():
    df = pd.DataFrame({
        "employee_id": ["EMP001", "EMP002", "EMP003"],
        "status": ["OBSERVED", "OBSERVED", "NOT OBSERVED"],
        "active_hours": [1.0, 2.0, 0.0],
        "phone_mins": [5, 10, 0],
        "away_mins": [0, 0, 0],
        "productive_pct": [80.0, 90.0, 0.0],
    })
    live = {"states": {"EMP001": "ACTIVE", "EMP002": "AWAY", "Unknown": "ACTIVE"},
            "camera_health": {"cam": {"health": "ONLINE"}}}
    m = _live_overview_metrics(df, live)
    assert m["recognised"] == 2          # NOT OBSERVED excluded
    assert m["present_now"] == 1         # EMP001 only (EMP002 AWAY, Unk excluded)
    assert m["unknown"] == 1
    assert m["cams_online"] == 1 and m["cams_total"] == 1


def test_empty_live_is_safe():
    df = pd.DataFrame()
    m = _live_overview_metrics(df, {})
    assert m["recognised"] == 0
    assert m["present_now"] == 0
    assert m["unknown"] == 0
    assert m["avg_pct"] == 0.0
    assert m["total_active"] == 0


def test_present_excludes_away():
    df = pd.DataFrame({
        "status": ["OBSERVED"], "active_hours": [1.0], "phone_mins": [0],
        "away_mins": [0], "productive_pct": [50.0],
    })
    m = _live_overview_metrics(df, {"states": {"EMP001": "AWAY",
                                               "EMP002": "ON_PHONE"}})
    assert m["present_now"] == 1
    assert m["unknown"] == 0


# ----------------------------------------------------------------------
# Grade 28: face-enrollment helpers
# ----------------------------------------------------------------------
def test_save_enrollment_image_writes_face_file(tmp_path, monkeypatch):
    faces_dir = tmp_path / "faces"
    monkeypatch.setitem(_faces_dir.__globals__, "FACES_DIR", str(faces_dir))
    ok, msg = save_enrollment_image("EMP010", "photo.jpg", b"\xff\xd8jpegdata")
    assert ok
    assert (faces_dir / "EMP010.jpg").exists()
    assert "EMP010.jpg" in msg


def test_save_enrollment_image_rejects_bad_input(tmp_path, monkeypatch):
    faces_dir = tmp_path / "faces"
    monkeypatch.setitem(_faces_dir.__globals__, "FACES_DIR", str(faces_dir))
    ok, msg = save_enrollment_image("EMP010", "photo.exe", b"x")
    assert not ok and "Unsupported image type" in msg
    ok, msg = save_enrollment_image("unknown", "photo.jpg", b"x")
    assert not ok and "not a valid employee id" in msg


def test_cached_enrollment_status_reads_embeddings_cache(tmp_path, monkeypatch):
    cache = tmp_path / "embeddings.pkl"
    cache.write_bytes(pickle.dumps({
        "version": 3,
        "file_status": {"EMP001": "ENROLLED", "EMP002": "NO_FACE"},
    }))
    monkeypatch.setattr("config.EMBEDDINGS_FILE", str(cache))
    status = cached_enrollment_status()
    assert status["EMP001"] == "ENROLLED"
    assert status["EMP002"] == "NO_FACE"


def test_cached_enrollment_status_missing_or_broken_cache_is_safe(tmp_path,
                                                                 monkeypatch):
    monkeypatch.setattr("config.EMBEDDINGS_FILE", str(tmp_path / "nope.pkl"))
    assert cached_enrollment_status() == {}
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle")
    monkeypatch.setattr("config.EMBEDDINGS_FILE", str(bad))
    assert cached_enrollment_status() == {}


def test_enrollment_status_frame_is_human_readable():
    df = enrollment_status_frame({"EMP001": "ENROLLED", "EMP002": "NO_FACE"})
    assert set(df["Employee ID"]) == {"EMP001", "EMP002"}
    assert "ENROLLED" in df["Status"].loc[0]
    assert "No face detected" in df["What this means"].loc[1]


def test_get_readonly_connection_sees_wal_rows(tmp_path):
    """Dashboard metrics helper must see rows committed under WAL mode even
    before any checkpoint.

    Regression: the old ``?mode=ro&immutable=1`` URI told SQLite the file is
    static, so it skipped the live WAL and returned only the occasionally
    checkpointed prefix -- stale metrics during the pilot (94% vs 69.8%).
    """
    import sqlite3

    import src.database as sdb
    from app import get_readonly_connection

    db = tmp_path / "live.db"
    writer = sqlite3.connect(db)
    writer.execute("PRAGMA journal_mode=wal")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    sdb.init_db(writer)
    writer.execute(
        "INSERT INTO activity_logs (timestamp, date, employee_id, state,"
        " duration_seconds, source) VALUES (?, ?, 'EMP001', 'ACTIVE', 10.0, '')",
        ("2026-01-05 10:00:00", "2026-01-05"),
    )
    writer.commit()
    try:
        reader = get_readonly_connection(str(db))
        assert reader is not None
        n = reader.execute(
            "SELECT COUNT(*) FROM activity_logs WHERE employee_id='EMP001'"
        ).fetchone()[0]
        assert n == 1, "WAL-committed row invisible to dashboard reader"

        from datetime import date

        from src.analytics import employee_day_metrics

        m = employee_day_metrics(reader, date(2026, 1, 5))
        emp = [x for x in m if x["employee_id"] == "EMP001"]
        assert emp and emp[0]["productive_seconds"] == 10.0

        import pytest

        with pytest.raises(sqlite3.DatabaseError):
            reader.execute(
                "INSERT INTO activity_logs (timestamp, employee_id, state,"
                " duration_seconds) VALUES ('2026-01-05 11:00:00', 'EMP001',"
                " 'ACTIVE', 1.0)"
            )
    finally:
        writer.close()