"""Tests for the event-driven evidence store."""

import os

import numpy as np
import pytest
from pathlib import Path

from src import database as db
from src.evidence import EvidenceStore
from src.domain import SEV_HIGH, SEV_LOW


@pytest.fixture
def evidence_dir(tmp_path):
    d = tmp_path / "evidence"
    d.mkdir()
    return str(d)


def _frame(value=128):
    return np.full((32, 32, 3), value, dtype=np.uint8)


def test_off_mode_writes_nothing(tmp_db, evidence_dir):
    store = EvidenceStore(tmp_db, mode="OFF", evidence_dir=evidence_dir)
    path = store.capture("INC-1", "CAMERA_OFFLINE", SEV_HIGH, "cam_01",
                         _frame())
    assert path is None
    assert os.listdir(evidence_dir) == []


def test_event_only_captures(tmp_db, evidence_dir):
    store = EvidenceStore(tmp_db, mode="EVENT_ONLY", evidence_dir=evidence_dir)
    path = store.capture("INC-1", "INTRUSION", SEV_HIGH, "cam_01", _frame())
    assert path is not None
    assert os.path.isfile(path)
    files = db.list_evidence_files(tmp_db, incident_id="INC-1")
    assert len(files) >= 1


def test_high_severity_only_filters(tmp_db, evidence_dir):
    store = EvidenceStore(tmp_db, mode="HIGH_SEVERITY_ONLY",
                          evidence_dir=evidence_dir)
    # LOW severity should not capture
    path = store.capture("INC-1", "ARRIVED", SEV_LOW, "cam_01", _frame())
    assert path is None
    # HIGH severity captures
    path2 = store.capture("INC-1", "CAMERA_OFFLINE", SEV_HIGH, "cam_01", _frame())
    assert path2 is not None


def test_metadata_json_written(tmp_db, evidence_dir):
    store = EvidenceStore(tmp_db, mode="EVENT_ONLY", evidence_dir=evidence_dir)
    store.capture("INC-1", "INTRUSION", SEV_HIGH, "cam_01", _frame(),
                  metadata={"foo": "bar"})
    files = db.list_evidence_files(tmp_db, incident_id="INC-1")
    kinds = {f["kind"] for f in files}
    assert "json" in kinds


def test_feed_buffer_no_crash(tmp_db, evidence_dir):
    store = EvidenceStore(tmp_db, mode="EVENT_ONLY", evidence_dir=evidence_dir,
                          max_buf_frames=5)
    for _ in range(10):
        store.feed_frame("cam_01", _frame())
    # buffer is bounded
    assert len(store._buffers["cam_01"]) <= 5
    path = store.capture("INC-1", "INTRUSION", SEV_HIGH, "cam_01", _frame())
    assert path is not None


def test_retention_cleanup(tmp_db, tmp_path, monkeypatch):
    from datetime import datetime, timedelta
    evidence_dir = tmp_path / "evidence"
    (evidence_dir / "2020-01-01" / "INC-old").mkdir(parents=True)
    (evidence_dir / "2020-01-01" / "INC-old" / "x.jpg").write_bytes(b"xx")
    store = EvidenceStore(tmp_db, mode="EVENT_ONLY", evidence_dir=str(evidence_dir),
                          retention_days=30)
    removed = store.retention_cleanup()
    assert removed >= 1
    assert not (evidence_dir / "2020-01-01").exists()


def test_storage_used_mb(tmp_db, tmp_path):
    evidence_dir = tmp_path / "evidence"
    (evidence_dir / "today").mkdir(parents=True)
    (evidence_dir / "today" / "x.jpg").write_bytes(b"x" * (1024 * 100))
    store = EvidenceStore(tmp_db, mode="EVENT_ONLY", evidence_dir=str(evidence_dir))
    mb = store.storage_used_mb()
    assert mb > 0


# ----------------------------------------------------------------------
# Phase 32 hardening
# ----------------------------------------------------------------------

def test_sha256_recorded(tmp_db, evidence_dir):
    store = EvidenceStore(tmp_db, mode="EVENT_ONLY", evidence_dir=evidence_dir)
    path = store.capture("INC-1", "INTRUSION", SEV_HIGH, "cam_01", _frame())
    assert path is not None
    rows = db.list_evidence_files(tmp_db, incident_id="INC-1")
    jpg_rows = [r for r in rows if r["kind"] == "jpeg"]
    assert jpg_rows, "expected a jpeg evidence row"
    assert jpg_rows[0]["sha256"]
    assert len(jpg_rows[0]["sha256"]) == 64


def test_camera_and_capture_type_recorded(tmp_db, evidence_dir):
    store = EvidenceStore(tmp_db, mode="EVENT_ONLY", evidence_dir=evidence_dir)
    store.capture("INC-1", "INTRUSION", SEV_HIGH, "cam_07", _frame())
    rows = db.list_evidence_files(tmp_db, incident_id="INC-1")
    jpgs = [r for r in rows if r["kind"] == "jpeg"]
    assert jpgs
    assert jpgs[0]["camera"] == "cam_07"
    assert jpgs[0]["capture_type"] in ("snapshot", "frame_buffer")


def test_path_safety_rejects_escape(tmp_db, tmp_path):
    evidence_dir = tmp_path / "evidence"
    store = EvidenceStore(tmp_db, mode="EVENT_ONLY", evidence_dir=str(evidence_dir))
    # A crafted "incident id" that tries to escape the evidence root must be
    # neutralized; capture must not write outside the root.
    evil = "../../escape"
    store.capture(evil, "INTRUSION", SEV_HIGH, "cam_01", _frame())
    escaped = tmp_path / "escape"
    # The hostile component was sanitized to a safe name inside the root.
    assert not escaped.exists()


def test_retention_deadline_set(tmp_db, evidence_dir):
    store = EvidenceStore(tmp_db, mode="EVENT_ONLY", evidence_dir=evidence_dir,
                          retention_days=14)
    store.capture("INC-1", "INTRUSION", SEV_HIGH, "cam_01", _frame())
    rows = db.list_evidence_files(tmp_db, incident_id="INC-1")
    assert rows and rows[0]["retention_deadline"]
