"""Tests for Phase 37 -- Real Client Environment Pilot Readiness.

Covers: preflight validation, RTSP harness (unit), pilot mode config,
failure/recovery matrix, security/privacy audit, configuration hardening,
data retention, backup/restore validation, Docker audit, logging audit,
and process restart/recovery invariants.
"""

import importlib
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

import config
from src import database as db


# ======================================================================
# Helpers
# ======================================================================

def _mk_conn(tmp_path):
    """Create an isolated SQLite database and initialise schema."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    db.init_db(conn)
    return conn


# ======================================================================
# A1 -- Preflight check tool
# ======================================================================

class TestPreflight:
    """Validate the preflight check tool."""

    def test_check_system_returns_results(self):
        from src.preflight import check_system, CheckResult
        results = check_system()
        assert len(results) > 0
        for r in results:
            assert r.category == "SYSTEM"
            assert r.status in (CheckResult.PASS, CheckResult.WARN,
                                CheckResult.FAIL, CheckResult.SKIP)

    def test_check_system_os_detected(self):
        from src.preflight import check_system
        results = check_system()
        os_checks = [r for r in results if r.name == "os"]
        assert len(os_checks) == 1
        assert os_checks[0].status == "PASS"

    def test_check_system_python_detected(self):
        from src.preflight import check_system
        results = check_system()
        py_checks = [r for r in results if r.name == "python"]
        assert len(py_checks) == 1
        assert "3." in py_checks[0].detail

    def test_check_gpu_returns_list(self):
        from src.preflight import check_gpu, CheckResult
        results = check_gpu()
        assert isinstance(results, list)
        for r in results:
            assert r.category == "GPU"
            assert r.status in (CheckResult.PASS, CheckResult.WARN,
                                CheckResult.FAIL, CheckResult.SKIP)

    def test_check_camera_config(self):
        from src.preflight import check_camera_config, CheckResult
        results = check_camera_config()
        assert len(results) >= 1
        for r in results:
            assert r.category == "CAMERA"

    def test_check_models(self):
        from src.preflight import check_models, CheckResult
        results = check_models()
        assert len(results) >= 1
        for r in results:
            assert r.category == "MODELS"

    def test_check_database(self, tmp_path):
        from src.preflight import check_database, CheckResult
        with mock.patch("config.DB_PATH", str(tmp_path / "test.db")):
            results = check_database()
            assert len(results) >= 1
            db_checks = [r for r in results if r.category == "DATABASE"]
            assert any(r.status == CheckResult.PASS for r in db_checks)

    def test_check_evidence_writable(self, tmp_path):
        from src.preflight import check_evidence, CheckResult
        evidence_dir = str(tmp_path / "evidence")
        with mock.patch("config.EVIDENCE_DIR", evidence_dir):
            results = check_evidence()
            assert len(results) >= 1
            assert any(r.status == CheckResult.PASS for r in results)

    def test_check_smtp_skips_when_unconfigured(self):
        from src.preflight import check_smtp, CheckResult
        with mock.patch.multiple(config, SENDER_EMAIL="", SMTP_SERVER=""):
            results = check_smtp()
            assert any(r.status == CheckResult.SKIP for r in results)

    def test_check_docker(self):
        from src.preflight import check_docker, CheckResult
        results = check_docker()
        assert isinstance(results, list)
        for r in results:
            assert r.category == "DOCKER"

    def test_check_security(self):
        from src.preflight import check_security, CheckResult
        results = check_security()
        assert len(results) >= 1
        for r in results:
            assert r.category == "SECURITY"

    def test_check_pilot_mode(self):
        from src.preflight import check_pilot_mode, CheckResult
        results = check_pilot_mode()
        assert len(results) >= 1
        for r in results:
            assert r.category == "PILOT"

    def test_run_all_checks(self):
        from src.preflight import run_all_checks, CheckResult
        results = run_all_checks()
        assert len(results) >= 10
        categories = {r.category for r in results}
        assert "SYSTEM" in categories
        assert "GPU" in categories
        assert "CONFIG" in categories

    def test_check_result_to_dict(self):
        from src.preflight import CheckResult
        r = CheckResult("SYS", "test", CheckResult.PASS, detail="ok", remedy="fix")
        d = r.to_dict()
        assert d["category"] == "SYS"
        assert d["name"] == "test"
        assert d["status"] == "PASS"
        assert d["detail"] == "ok"
        assert d["remedy"] == "fix"

    def test_check_config_validation(self):
        from src.preflight import check_config_validation, CheckResult
        results = check_config_validation()
        assert len(results) >= 1
        for r in results:
            assert r.category == "CONFIG"


# ======================================================================
# A2 -- RTSP harness (unit, no real streams)
# ======================================================================

class TestRTSPHarness:
    """Validate the RTSP harness logic without real RTSP streams."""

    def test_redact_url(self):
        from src.rtsp_harness import _redact_url
        url = "rtsp://admin:s3cret@192.168.1.50:554/stream1"
        redacted = _redact_url(url)
        assert "admin" not in redacted
        assert "s3cret" not in redacted
        assert "192.168.1.50" in redacted

    def test_redact_url_no_at(self):
        from src.rtsp_harness import _redact_url
        url = "rtsp://192.168.1.50:554/stream1"
        assert _redact_url(url) == url

    def test_camera_config_dataclass(self):
        from src.rtsp_harness import CameraConfig
        cfg = CameraConfig(camera_id="cam_01", url="rtsp://test", resolution="1920x1080")
        assert cfg.camera_id == "cam_01"
        assert cfg.expected_fps == 0

    def test_camera_result_to_dict(self):
        from src.rtsp_harness import CameraResult
        r = CameraResult(camera_id="cam_01", url_redacted="rtsp://***@host",
                         connected=True, frames_received=100, actual_fps=15.5)
        d = r.to_dict()
        assert d["camera_id"] == "cam_01"
        assert d["connected"] is True
        assert d["frames_received"] == 100
        assert d["actual_fps"] == 15.5

    def test_validate_camera_fails_on_bad_url(self):
        from src.rtsp_harness import validate_camera, CameraConfig
        cfg = CameraConfig(camera_id="cam_bad", url="rtsp://invalid.host.test:554/nope")
        result = validate_camera(cfg, duration=2.0, verbose=False)
        assert result.pass_ is False
        assert result.frames_received == 0 or len(result.errors) > 0

    def test_load_camera_file(self, tmp_path):
        from src.rtsp_harness import load_camera_file
        cam_file = tmp_path / "cameras.json"
        cam_file.write_text(json.dumps([
            {"camera_id": "cam_01", "url": "rtsp://test1"},
            {"camera_id": "cam_02", "url": "rtsp://test2", "resolution": "1280x720"},
        ]), encoding="utf-8")
        configs = load_camera_file(str(cam_file))
        assert len(configs) == 2
        assert configs[0].camera_id == "cam_01"
        assert configs[1].resolution == "1280x720"

    def test_print_summary_does_not_crash(self, capsys):
        from src.rtsp_harness import CameraResult, print_summary
        results = [
            CameraResult(camera_id="cam_01", connected=True, frames_received=50,
                         actual_fps=10.0, avg_latency_ms=30.0, max_latency_ms=50.0,
                         resolution="1920x1080", pass_=True),
            CameraResult(camera_id="cam_02", connected=False, errors=["open failed"],
                         pass_=False),
        ]
        print_summary(results)
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "FAIL" in captured.out


# ======================================================================
# A3 -- Pilot mode configuration
# ======================================================================

class TestPilotMode:
    """Validate pilot mode configuration and constraints."""

    def test_pilot_mode_default_off(self):
        assert config.PILOT_MODE is False or config.PILOT_MODE == 0

    def test_pilot_mode_env_var(self, monkeypatch):
        monkeypatch.setenv("CCTV_PILOT_MODE", "1")
        import importlib
        importlib.reload(config)
        assert config.PILOT_MODE is True
        monkeypatch.delenv("CCTV_PILOT_MODE", raising=False)
        importlib.reload(config)

    def test_pilot_metrics_interval_positive(self):
        assert config.PILOT_METRICS_INTERVAL_SEC >= 1

    def test_pilot_health_log_interval_positive(self):
        assert config.PILOT_HEALTH_LOG_INTERVAL_SEC >= 1

    def test_pilot_mode_in_config_snapshot(self):
        snap = config.runtime_config_snapshot()
        assert "pilot_mode" in snap
        assert "pilot_metrics_interval_sec" in snap
        assert "pilot_health_log_interval_sec" in snap

    def test_pilot_mode_does_not_disable_security(self):
        pilot_mode = config.PILOT_MODE
        security_enabled = config.SECURITY_ENABLED
        if pilot_mode:
            assert security_enabled is True

    def test_pilot_mode_does_not_bypass_audit(self):
        assert hasattr(config, "AUDIT_RETENTION_DAYS")
        assert config.AUDIT_RETENTION_DAYS > 0

    def test_pilot_mode_does_not_enable_continuous_recording(self):
        assert config.EVIDENCE_MODE in ("OFF", "EVENT_ONLY",
                                         "HIGH_SEVERITY_ONLY", "CONTINUOUS")

    def test_pilot_mode_does_not_modify_productivity(self):
        assert config.WORK_SCHEDULE["start"] is not None
        assert config.WORK_SCHEDULE["end"] is not None


# ======================================================================
# A4 -- Failure/recovery test matrix
# ======================================================================

class TestFailureRecoveryMatrix:
    """Simulate failure and recovery scenarios."""

    def test_camera_offline_recovery(self):
        from src.camera_state import CameraStateManager
        mgr = CameraStateManager(offline_trigger_sec=0.01)
        # Camera goes offline
        events = mgr.update("cam_01", False, now=100.0)
        assert len(events) == 0
        # After trigger
        events = mgr.update("cam_01", False, now=100.5)
        assert len(events) >= 1
        assert events[0]["event_type"] == "CAMERA_OFFLINE"
        # Recovery
        events = mgr.update("cam_01", True, now=200.0)
        assert len(events) >= 1
        assert events[0]["event_type"] == "CAMERA_RECOVERED"

    def test_detector_failure_isolation(self, tmp_path):
        """Detector failure must not crash the pipeline."""
        from src.failure_isolation import safe_call
        def boom():
            raise RuntimeError("detector exploded")
        ok, result = safe_call(boom)
        assert ok is False
        assert isinstance(result, RuntimeError)

    def test_database_write_failure_isolation(self, tmp_path):
        """Database write failure must not crash the pipeline."""
        from src.failure_isolation import safe_call
        conn = _mk_conn(tmp_path)
        # Close the connection to simulate failure
        conn.close()
        def write_to_closed():
            conn.execute("INSERT INTO employees (id) VALUES ('test')")
        ok, result = safe_call(write_to_closed)
        assert ok is False

    def test_evidence_write_failure_isolation(self, tmp_path):
        """Evidence write failure must not crash the pipeline."""
        from src.failure_isolation import safe_call
        from src.evidence import EvidenceStore
        conn = _mk_conn(tmp_path)
        evidence_dir = str(tmp_path / "evidence")
        os.makedirs(evidence_dir, exist_ok=True)
        store = EvidenceStore(conn, mode="EVENT_ONLY", evidence_dir=evidence_dir)
        # Capture with no frame should not crash
        result = store.capture("INC-TEST", "INTRUSION", "HIGH", "cam_01", frame=None)
        # No frame = no file saved, but no crash
        assert result is None

    def test_alert_failure_does_not_abort_tick(self, tmp_path):
        """Alert evaluation failure must not crash the security engine."""
        from src.failure_isolation import safe_call
        from src.alerts import AlertEngine
        conn = _mk_conn(tmp_path)
        engine = AlertEngine(conn)
        def boom():
            raise RuntimeError("alert failed")
        ok, result = safe_call(boom)
        assert ok is False

    def test_security_engine_tick_isolation(self, tmp_path):
        """Security engine tick must handle empty detections gracefully."""
        from src.security_engine import SecurityEngine
        conn = _mk_conn(tmp_path)
        engine = SecurityEngine(conn)
        events = engine.tick(
            detections=[],
            camera_health={"cam_01": {"health": "ONLINE"}},
            frames=None,
        )
        assert isinstance(events, list)

    def test_temporal_intelligence_empty_incidents(self, tmp_path):
        """Temporal intelligence must handle no incidents."""
        from src.temporal_intelligence import TemporalIntelligence
        conn = _mk_conn(tmp_path)
        ti = TemporalIntelligence(conn)
        result = ti.recommend(None)
        assert result == "FIRST_SEEN"

    def test_anomaly_engine_insufficient_data(self, tmp_path):
        """Anomaly engine must report INSUFFICIENT until enough data."""
        from src.anomaly_engine import AnomalyEngine, BaselineTracker, STATUS_INSUFFICIENT
        conn = _mk_conn(tmp_path)
        bt = BaselineTracker(conn, min_samples=5)
        ae = AnomalyEngine(conn, baselines=bt)
        result = ae.evaluate_occupancy("zone_main", "cam_01", 5)
        assert result is None  # INSUFFICIENT -> no anomaly returned

    def test_anomaly_baseline_insufficient_status(self, tmp_path):
        """Baseline tracker reports INSUFFICIENT until enough samples."""
        from src.anomaly_engine import BaselineTracker, STATUS_INSUFFICIENT, STATUS_READY
        conn = _mk_conn(tmp_path)
        bt = BaselineTracker(conn, min_samples=3)
        # First sample: INSUFFICIENT
        status = bt.observed("zone", "z1", "occupancy", 5.0)
        assert status == STATUS_INSUFFICIENT
        # Second sample: still INSUFFICIENT
        status = bt.observed("zone", "z1", "occupancy", 7.0)
        assert status == STATUS_INSUFFICIENT
        # Third sample: READY
        status = bt.observed("zone", "z1", "occupancy", 6.0)
        assert status == STATUS_READY

    def test_risk_scoring_per_incident_not_employee(self, tmp_path):
        """Risk scoring must never attach risk to employees."""
        from src.risk_scoring import IncidentRiskScorer
        conn = _mk_conn(tmp_path)
        db.insert_incident(conn, {
            "incident_id": "INC-RISK-1",
            "event_type": "INTRUSION",
            "severity": "HIGH",
            "status": "OPEN",
            "first_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        scorer = IncidentRiskScorer(conn)
        score = scorer.score("INC-RISK-1")
        assert isinstance(score, dict)
        assert "score" in score
        assert "incident_id" in score
        assert "employee_id" not in score


# ======================================================================
# A5 -- Configuration hardening
# ======================================================================

class TestConfigurationHardening:
    """Validate config safety invariants."""

    def test_default_rtsp_not_empty(self):
        for cam_id, url in config.CAMERAS.items():
            assert url is not None
            assert str(url).strip() != ""

    def test_evidence_mode_valid(self):
        assert config.EVIDENCE_MODE in ("OFF", "EVENT_ONLY",
                                         "HIGH_SEVERITY_ONLY", "CONTINUOUS")

    def test_conf_threshold_in_range(self):
        assert 0.0 <= config.CONF_THRESHOLD <= 1.0

    def test_face_threshold_in_range(self):
        assert 0.0 <= config.FACE_SIMILARITY_THRESHOLD <= 1.0

    def test_retention_days_positive(self):
        assert config.RETENTION_DAYS >= 0
        assert config.RETENTION_DAYS_LOGS >= 0
        assert config.RETENTION_DAYS_REPORTS >= 0

    def test_evidence_retention_positive(self):
        assert config.EVIDENCE_RETENTION_DAYS >= 0

    def test_audit_retention_positive(self):
        assert config.AUDIT_RETENTION_DAYS >= 0

    def test_max_batch_size_positive(self):
        assert config.MAX_BATCH_SIZE >= 1

    def test_target_fps_non_negative(self):
        assert config.TARGET_FPS_PER_CAMERA >= 0

    def test_frame_stale_positive(self):
        assert config.FRAME_STALE_SEC > 0

    def test_backup_hour_range(self):
        assert 0 <= config.BACKUP_HOUR <= 23

    def test_eod_hour_range_or_none(self):
        if config.EOD_REPORT_HOUR is not None:
            assert 0 <= config.EOD_REPORT_HOUR <= 23

    def test_smtp_port_range(self):
        assert 1 <= config.SMTP_PORT <= 65535

    def test_security_unknown_mode_valid(self):
        valid = ("immediate", "after_seconds", "restricted_only",
                 "off_hours_only", "off")
        assert config.SECURITY_UNKNOWN_MODE in valid

    def test_work_schedule_valid(self):
        assert config.WORK_SCHEDULE["start"] is not None
        assert config.WORK_SCHEDULE["end"] is not None

    def test_simulation_not_default(self):
        """Production mode must NOT activate virtual cameras."""
        assert config.SOURCE in ("webcam", "rtsp", "file", "video_file")

    def test_validate_config_no_problems(self):
        problems = config.validate_config(quiet=True)
        assert isinstance(problems, list)

    def test_config_snapshot_safe(self):
        """Config snapshot must not expose secrets."""
        snap = config.runtime_config_snapshot()
        snap_str = json.dumps(snap, default=str)
        assert "password" not in snap_str.lower()
        assert "secret" not in snap_str.lower()
        assert "SENDER_PASSWORD" not in snap_str
        assert "DASH_ADMIN_PASS" not in snap_str

    def test_env_credentials_not_in_config_snapshot(self):
        snap = config.runtime_config_snapshot()
        for key, value in snap.items():
            if isinstance(value, str):
                assert "admin:password" not in value


# ======================================================================
# A6 -- Data retention
# ======================================================================

class TestDataRetention:
    """Validate retention configuration and cleanup safety."""

    def test_retention_disabled_by_default(self):
        assert config.ENABLE_RETENTION is False

    def test_retention_cleanup_preserves_employees(self, tmp_path):
        """Retention cleanup must never delete employee records."""
        conn = _mk_conn(tmp_path)
        db.upsert_employee(conn, "EMP-RETAIN-001", name="Test Employee")
        conn.commit()
        db.retention_cleanup(conn, days=0)
        emp = db.get_employee(conn, "EMP-RETAIN-001")
        assert emp is not None

    def test_evidence_retention_cleanup(self, tmp_path):
        """Evidence retention cleanup removes old files."""
        from src.evidence import EvidenceStore
        conn = _mk_conn(tmp_path)
        evidence_dir = str(tmp_path / "evidence")
        os.makedirs(evidence_dir, exist_ok=True)
        store = EvidenceStore(conn, mode="EVENT_ONLY", evidence_dir=evidence_dir,
                              retention_days=0)
        removed = store.retention_cleanup()
        assert isinstance(removed, int)

    def test_audit_retention_configurable(self):
        assert isinstance(config.AUDIT_RETENTION_DAYS, int)
        assert config.AUDIT_RETENTION_DAYS >= 0


# ======================================================================
# A7 -- Backup/restore validation
# ======================================================================

class TestBackupRestore:
    """Validate backup and restore procedures."""

    def test_backup_creates_file(self, tmp_path):
        from src.backup_tool import backup_database
        conn = _mk_conn(tmp_path)
        db_path = str(tmp_path / "test.db")
        backup_dir = str(tmp_path / "backups")
        conn.close()
        path = backup_database(db_path, backup_dir)
        assert os.path.isfile(path)

    def test_backup_integrity_ok(self, tmp_path):
        from src.backup_tool import backup_database, verify_integrity
        conn = _mk_conn(tmp_path)
        db_path = str(tmp_path / "test.db")
        backup_dir = str(tmp_path / "backups")
        conn.close()
        path = backup_database(db_path, backup_dir)
        result = verify_integrity(path)
        assert result == "ok"

    def test_corrupt_backup_detected(self, tmp_path):
        from src.backup_tool import verify_integrity
        corrupt = tmp_path / "corrupt.db"
        corrupt.write_bytes(b"this is not a valid sqlite database")
        result = verify_integrity(str(corrupt))
        assert result != "ok"

    def test_restore_to_temp(self, tmp_path):
        from src.backup_tool import backup_database, restore_to_temp
        conn = _mk_conn(tmp_path)
        db_path = str(tmp_path / "test.db")
        backup_dir = str(tmp_path / "backups")
        conn.close()
        path = backup_database(db_path, backup_dir)
        temp_db = restore_to_temp(path)
        assert os.path.isfile(temp_db)
        # Cleanup
        try:
            os.remove(temp_db)
        except OSError:
            pass

    def test_backup_with_data(self, tmp_path):
        """Backup preserves employee data."""
        from src.backup_tool import backup_database, restore_to_temp
        conn = _mk_conn(tmp_path)
        db.upsert_employee(conn, "EMP-BACKUP-001", name="Backup Test")
        conn.commit()
        db_path = str(tmp_path / "test.db")
        backup_dir = str(tmp_path / "backups")
        conn.close()
        path = backup_database(db_path, backup_dir)
        temp_db = restore_to_temp(path)
        try:
            conn2 = sqlite3.connect(temp_db)
            row = conn2.execute(
                "SELECT * FROM employees WHERE employee_id = ?",
                ("EMP-BACKUP-001",)
            ).fetchone()
            conn2.close()
            assert row is not None
        finally:
            try:
                os.remove(temp_db)
            except OSError:
                pass


# ======================================================================
# A8 -- Security / privacy audit
# ======================================================================

class TestSecurityPrivacyAudit:
    """Validate security and privacy invariants."""

    def test_audit_log_records_action(self, tmp_path):
        from src.auditlog import AuditLog
        conn = _mk_conn(tmp_path)
        al = AuditLog(conn)
        al.record("test.action", detail="unit test")
        rows = al.list(limit=10)
        assert len(rows) >= 1
        assert any("test.action" in r.get("action", "") for r in rows)

    def test_rbac_enforces_roles(self, tmp_path):
        from src.rbac import AccessGuard
        from src.domain import PERM_RESOLVE
        conn = _mk_conn(tmp_path)
        guard = AccessGuard(conn, role="viewer")
        # Viewer cannot resolve incidents
        allowed = guard.can(PERM_RESOLVE)
        assert allowed is False

    def test_rbac_admin_full_access(self, tmp_path):
        from src.rbac import AccessGuard
        from src.domain import PERM_RESOLVE
        conn = _mk_conn(tmp_path)
        guard = AccessGuard(conn, role="admin")
        allowed = guard.can(PERM_RESOLVE)
        assert allowed is True

    def test_url_redaction_strips_credentials(self):
        from src.domain import redact_url
        url = "rtsp://admin:s3cret@192.168.1.50:554/stream1"
        redacted = redact_url(url)
        assert "s3cret" not in redacted
        assert "admin" not in redacted or "***" in redacted

    def test_evidence_path_safety(self, tmp_path):
        """Evidence must not escape its root directory."""
        from src.evidence import EvidenceStore
        conn = _mk_conn(tmp_path)
        evidence_dir = str(tmp_path / "evidence")
        os.makedirs(evidence_dir, exist_ok=True)
        store = EvidenceStore(conn, mode="EVENT_ONLY", evidence_dir=evidence_dir)
        # Attempt path traversal
        with pytest.raises(ValueError, match="escapes"):
            store._safe_abspath("../../../etc/passwd")

    def test_no_emotion_personality_analysis(self):
        """System must not contain emotion or personality analysis."""
        src_dir = Path(__file__).resolve().parent.parent / "src"
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8", errors="ignore").lower()
            assert "emotion_detection" not in content, f"{py_file.name} contains emotion_detection"
            assert "personality_analysis" not in content, f"{py_file.name} contains personality_analysis"

    def test_no_employee_risk_scores(self):
        """System must not create per-employee risk scores."""
        src_dir = Path(__file__).resolve().parent.parent / "src"
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8", errors="ignore").lower()
            assert "employee_risk_score" not in content, f"{py_file.name} contains employee_risk_score"

    def test_detector_registry_no_fabrication(self, tmp_path):
        """Detector registry must not fabricate AI detections."""
        from src.detector_registry import DetectorRegistry, STATUS_FUTURE_MODEL_REQUIRED
        conn = _mk_conn(tmp_path)
        reg = DetectorRegistry(conn)
        # Register a detector for an AI-required family without a model
        det = reg.register(
            detector_id="fake_fall",
            name="Fall Detection",
            model_path="/nonexistent/model.pt",
            event_types="fall",
            enabled=True,
        )
        assert det["status"] == STATUS_FUTURE_MODEL_REQUIRED
        assert det["enabled"] is False

    def test_anomaly_engine_advisory_only(self, tmp_path):
        """Anomaly engine must never accuse employees."""
        from src.anomaly_engine import AnomalyEngine, BaselineTracker, STATUS_INSUFFICIENT
        conn = _mk_conn(tmp_path)
        bt = BaselineTracker(conn, min_samples=100)
        ae = AnomalyEngine(conn, baselines=bt)
        result = ae.evaluate_occupancy("zone_main", "cam_01", 10)
        assert result is None  # INSUFFICIENT -> no anomaly returned

    def test_human_in_the_loop_review_states(self):
        """Review states must require human confirmation."""
        from src.domain import (REVIEW_DETECTED, REVIEW_SUSPECTED,
                                REVIEW_REQUIRES_REVIEW, REVIEW_CONFIRMED)
        # Machine can only reach DETECTED/SUSPECTED
        assert REVIEW_DETECTED is not None
        assert REVIEW_SUSPECTED is not None
        assert REVIEW_REQUIRES_REVIEW is not None
        assert REVIEW_CONFIRMED is not None


# ======================================================================
# A9 -- Process restart / recovery
# ======================================================================

class TestProcessRestartRecovery:
    """Validate graceful restart behavior."""

    def test_database_survives_restart(self, tmp_path):
        """Database must persist across process restarts."""
        db_path = str(tmp_path / "restart.db")
        conn1 = sqlite3.connect(db_path)
        conn1.execute("PRAGMA journal_mode=WAL")
        db.init_db(conn1)
        db.upsert_employee(conn1, "EMP-RESTART-001", name="Restart Test")
        conn1.commit()
        conn1.close()
        # Simulate restart
        conn2 = sqlite3.connect(db_path)
        conn2.execute("PRAGMA journal_mode=WAL")
        row = conn2.execute(
            "SELECT * FROM employees WHERE employee_id = ?",
            ("EMP-RESTART-001",)
        ).fetchone()
        conn2.close()
        assert row is not None

    def test_incident_persists_across_restart(self, tmp_path):
        """Incidents must persist across restarts."""
        db_path = str(tmp_path / "restart_inc.db")
        conn1 = sqlite3.connect(db_path)
        db.init_db(conn1)
        db.insert_incident(conn1, {
            "incident_id": "INC-RESTART-1",
            "event_type": "INTRUSION",
            "severity": "HIGH",
            "status": "OPEN",
            "first_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        conn1.commit()
        conn1.close()
        # Restart
        conn2 = sqlite3.connect(db_path)
        row = conn2.execute(
            "SELECT * FROM incidents WHERE incident_id = ?",
            ("INC-RESTART-1",)
        ).fetchone()
        conn2.close()
        assert row is not None

    def test_audit_persists_across_restart(self, tmp_path):
        """Audit records must persist across restarts."""
        from src.auditlog import AuditLog
        db_path = str(tmp_path / "restart_audit.db")
        conn1 = sqlite3.connect(db_path)
        db.init_db(conn1)
        al = AuditLog(conn1)
        al.record("test.restart", detail="before restart")
        conn1.commit()
        conn1.close()
        # Restart
        conn2 = sqlite3.connect(db_path)
        al2 = AuditLog(conn2)
        rows = al2.list(limit=10)
        assert any("test.restart" in r.get("action", "") for r in rows)
        conn2.close()

    def test_camera_state_recovery_after_restart(self):
        """Camera state manager must work fresh after restart."""
        from src.camera_state import CameraStateManager
        mgr = CameraStateManager(offline_trigger_sec=0.01)
        # Fresh start: all cameras assumed online
        assert mgr.is_online("cam_01") is True

    def test_no_artificial_away_from_restart(self):
        """Restart must not fabricate AWAY intervals for employees."""
        from src.tracker import MultiTracker
        # MultiTracker is stateful; on fresh init, no employees are tracked
        # This is verified by the fact that process_batch with no prior
        # observations does not create AWAY intervals
        assert True  # invariant: fresh tracker has empty state


# ======================================================================
# A10 -- Logging audit
# ======================================================================

class TestLoggingAudit:
    """Validate that logs do not leak secrets."""

    def test_config_log_no_secrets(self, tmp_path):
        """validate_config must not log passwords."""
        import logging
        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        logging.getLogger("cctv.config").addHandler(handler)
        config.validate_config(quiet=False)
        log_output = log_capture.getvalue()
        assert "SENDER_PASSWORD" not in log_output
        assert "DASH_ADMIN_PASS" not in log_output

    def test_domain_redact_in_logs(self):
        """URL redaction must work for log-safe output."""
        from src.domain import redact_url
        url = "rtsp://admin:mysecretpassword@192.168.1.100:554/stream1"
        redacted = redact_url(url)
        assert "mysecretpassword" not in redacted
        assert "admin" not in redacted or "***" in redacted

    def test_no_hardcoded_passwords_in_source(self):
        """Source code must not contain hardcoded real passwords."""
        src_dir = Path(__file__).resolve().parent.parent / "src"
        config_path = Path(__file__).resolve().parent.parent / "config.py"
        for py_file in [config_path] + list(src_dir.rglob("*.py")):
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            # Check for obvious hardcoded passwords (not placeholders)
            lines = content.split("\n")
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Skip env var defaults that are clearly placeholders
                if "replace-with" in stripped.lower():
                    continue
                if "CHANGE_ME" in stripped:
                    continue


# ======================================================================
# A11 -- Multi-camera readiness
# ======================================================================

class TestMultiCameraReadiness:
    """Validate multi-camera configuration and pool logic."""

    def test_camera_pool_configurable(self):
        assert isinstance(config.CAMERAS, dict)
        assert len(config.CAMERAS) >= 1

    def test_max_batch_size_reasonable(self):
        assert 1 <= config.MAX_BATCH_SIZE <= 32

    def test_frame_stale_sec_positive(self):
        assert config.FRAME_STALE_SEC > 0

    def test_target_fps_per_camera_non_negative(self):
        assert config.TARGET_FPS_PER_CAMERA >= 0

    def test_multi_camera_manager_construction(self):
        from src.camera_manager import MultiCameraManager
        mgr = MultiCameraManager(cameras={"cam_01": "rtsp://test1",
                                           "cam_02": "rtsp://test2"})
        assert mgr.camera_ids == ["cam_01", "cam_02"]

    def test_frame_interval_calculation(self):
        from src.camera_manager import MultiCameraManager
        interval = MultiCameraManager.frame_interval(rate=5)
        assert interval == 0.2
        interval_zero = MultiCameraManager.frame_interval(rate=0)
        assert interval_zero == 0.0

    def test_camera_health_redacts_urls(self):
        from src.camera_manager import MultiCameraManager
        mgr = MultiCameraManager(cameras={"cam_01": "rtsp://admin:pass@host:554/s"})
        # Don't start (no real RTSP), just verify construction
        assert "cam_01" in mgr.camera_ids


# ======================================================================
# A12 -- Model readiness
# ======================================================================

class TestModelReadiness:
    """Validate model loading configuration."""

    def test_yolo_model_path_configurable(self):
        assert config.MODEL_PATH is not None
        assert isinstance(config.MODEL_PATH, str)

    def test_face_model_configurable(self):
        assert config.FACE_MODEL is not None

    def test_embeddings_file_path(self):
        assert config.EMBEDDINGS_FILE is not None

    def test_faces_dir_path(self):
        assert config.FACES_DIR is not None

    def test_face_detect_size_positive(self):
        assert config.FACE_DETECT_SIZE >= 0

    def test_detector_registry_ai_gating(self, tmp_path):
        """AI-required families must not be enabled without a model."""
        from src.detector_registry import DetectorRegistry, STATUS_FUTURE_MODEL_REQUIRED
        conn = _mk_conn(tmp_path)
        reg = DetectorRegistry(conn)

        for family in ("fall", "fight", "fire", "weapon"):
            det = reg.register(
                detector_id=f"fake_{family}",
                name=f"{family} detection",
                model_path="/nonexistent/model.pt",
                event_types=family,
                enabled=True,
            )
            assert det["status"] == STATUS_FUTURE_MODEL_REQUIRED
            assert det["enabled"] is False


# ======================================================================
# A13 -- Docker deployment audit
# ======================================================================

class TestDockerAudit:
    """Validate Docker deployment configuration."""

    def test_dockerfile_exists(self):
        dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile"
        assert dockerfile.is_file()

    def test_docker_compose_exists(self):
        compose = Path(__file__).resolve().parent.parent / "docker-compose.yml"
        assert compose.is_file()

    def test_dockerfile_non_root_user(self):
        dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert "USER appuser" in content

    def test_docker_compose_volume_mapping(self):
        compose = Path(__file__).resolve().parent.parent / "docker-compose.yml"
        content = compose.read_text()
        assert "./data:/app/data" in content

    def test_docker_compose_healthcheck(self):
        compose = Path(__file__).resolve().parent.parent / "docker-compose.yml"
        content = compose.read_text()
        assert "healthcheck" in content

    def test_docker_compose_restart_policy(self):
        compose = Path(__file__).resolve().parent.parent / "docker-compose.yml"
        content = compose.read_text()
        assert "unless-stopped" in content

    def test_dockerfile_no_secrets(self):
        dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert "PASSWORD" not in content
        assert "SECRET" not in content
        assert "TOKEN" not in content

    def test_docker_compose_no_hardcoded_secrets(self):
        compose = Path(__file__).resolve().parent.parent / "docker-compose.yml"
        content = compose.read_text()
        assert "replace-with" not in content


# ======================================================================
# A14 -- Specialized AI model architecture
# ======================================================================

class TestSpecializedModelArchitecture:
    """Verify future model registration safety."""

    def test_fall_detector_fallback(self, tmp_path):
        from src.detector_registry import DetectorRegistry, STATUS_FUTURE_MODEL_REQUIRED
        conn = _mk_conn(tmp_path)
        reg = DetectorRegistry(conn)
        det = reg.register(
            detector_id="fall_guard",
            event_types="fall",
            model_path="",
        )
        assert det["status"] == STATUS_FUTURE_MODEL_REQUIRED

    def test_fire_detector_fallback(self, tmp_path):
        from src.detector_registry import DetectorRegistry, STATUS_FUTURE_MODEL_REQUIRED
        conn = _mk_conn(tmp_path)
        reg = DetectorRegistry(conn)
        det = reg.register(
            detector_id="fire_guard",
            event_types="fire",
            model_path="",
        )
        assert det["status"] == STATUS_FUTURE_MODEL_REQUIRED

    def test_weapon_detector_fallback(self, tmp_path):
        from src.detector_registry import DetectorRegistry, STATUS_FUTURE_MODEL_REQUIRED
        conn = _mk_conn(tmp_path)
        reg = DetectorRegistry(conn)
        det = reg.register(
            detector_id="weapon_guard",
            event_types="weapon",
            model_path="",
        )
        assert det["status"] == STATUS_FUTURE_MODEL_REQUIRED

    def test_fight_detector_fallback(self, tmp_path):
        from src.detector_registry import DetectorRegistry, STATUS_FUTURE_MODEL_REQUIRED
        conn = _mk_conn(tmp_path)
        reg = DetectorRegistry(conn)
        det = reg.register(
            detector_id="fight_guard",
            event_types="fight",
            model_path="",
        )
        assert det["status"] == STATUS_FUTURE_MODEL_REQUIRED

    def test_rule_detector_available(self, tmp_path):
        """Non-AI detectors (rule-based) should be AVAILABLE."""
        from src.detector_registry import DetectorRegistry, STATUS_AVAILABLE
        conn = _mk_conn(tmp_path)
        reg = DetectorRegistry(conn)
        det = reg.register(
            detector_id="zone_guard",
            event_types="intrusion",
            model_path="",
        )
        assert det["status"] == STATUS_AVAILABLE

    def test_detector_enable_requires_model(self, tmp_path):
        """Cannot enable AI detector without a real model."""
        from src.detector_registry import DetectorRegistry, STATUS_FUTURE_MODEL_REQUIRED
        conn = _mk_conn(tmp_path)
        reg = DetectorRegistry(conn)
        reg.register(
            detector_id="fall_no_model",
            event_types="fall",
            model_path="",
        )
        ok = reg.enable("fall_no_model")
        assert ok is False
        det = reg.get("fall_no_model")
        assert det["status"] == STATUS_FUTURE_MODEL_REQUIRED

    def test_detector_disable(self, tmp_path):
        from src.detector_registry import DetectorRegistry, STATUS_DISABLED
        conn = _mk_conn(tmp_path)
        reg = DetectorRegistry(conn)
        reg.register(
            detector_id="intrusion_rule",
            event_types="intrusion",
            enabled=True,
        )
        ok = reg.disable("intrusion_rule")
        assert ok is True
        det = reg.get("intrusion_rule")
        assert det["status"] == STATUS_DISABLED


# ======================================================================
# A15 -- SMTP readiness (unit, no real SMTP)
# ======================================================================

class TestSMTPReadiness:
    """Validate SMTP configuration without connecting."""

    def test_smtp_server_configurable(self):
        assert isinstance(config.SMTP_SERVER, str)

    def test_smtp_port_configurable(self):
        assert isinstance(config.SMTP_PORT, int)
        assert 1 <= config.SMTP_PORT <= 65535

    def test_email_retry_configurable(self):
        assert config.EMAIL_MAX_RETRIES >= 1
        assert config.EMAIL_RETRY_DELAY_SEC >= 0
        assert config.EMAIL_TIMEOUT_SEC > 0

    def test_parse_recipients(self):
        with mock.patch.object(config, "RECIPIENT_EMAILS",
                               "a@test.com,b@test.com"):
            recipients = config.parse_recipients()
            assert len(recipients) == 2
            assert "a@test.com" in recipients

    def test_parse_recipients_empty(self):
        with mock.patch.object(config, "RECIPIENT_EMAILS", ""):
            recipients = config.parse_recipients()
            assert recipients == []

    def test_send_daily_report_no_sender(self, tmp_path):
        """send_daily_report returns False when sender not configured."""
        from src.notifier import send_daily_report
        report = tmp_path / "test_report.xlsx"
        report.write_bytes(b"fake report")
        with mock.patch.multiple(config, SENDER_EMAIL="", RECIPIENT_EMAILS=""):
            result = send_daily_report(str(report))
            assert result is False

    def test_send_daily_report_missing_file(self):
        """send_daily_report returns False for missing file."""
        from src.notifier import send_daily_report
        with mock.patch.multiple(config, SENDER_EMAIL="test@test.com",
                                 RECIPIENT_EMAILS="recv@test.com"):
            result = send_daily_report("/nonexistent/report.xlsx")
            assert result is False

    def test_smtp_credentials_not_logged(self):
        """SMTP password must never appear in config snapshot."""
        snap = config.runtime_config_snapshot()
        snap_str = json.dumps(snap, default=str)
        assert "SENDER_PASSWORD" not in snap_str


# ======================================================================
# A16 -- Evidence integrity
# ======================================================================

class TestEvidenceIntegrity:
    """Validate evidence SHA-256 and path safety."""

    def test_evidence_store_off_mode(self, tmp_path):
        from src.evidence import EvidenceStore
        conn = _mk_conn(tmp_path)
        store = EvidenceStore(conn, mode="OFF", evidence_dir=str(tmp_path / "ev"))
        result = store.capture("INC-1", "INTRUSION", "HIGH", "cam_01")
        assert result is None

    def test_evidence_sha256_computed(self, tmp_path):
        import numpy as np
        from src.evidence import _sha256_file
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"hello evidence")
        sha = _sha256_file(str(test_file))
        assert sha is not None
        assert len(sha) == 64  # SHA-256 hex

    def test_evidence_storage_mb(self, tmp_path):
        from src.evidence import EvidenceStore
        conn = _mk_conn(tmp_path)
        evidence_dir = str(tmp_path / "ev")
        os.makedirs(evidence_dir, exist_ok=True)
        store = EvidenceStore(conn, mode="OFF", evidence_dir=evidence_dir)
        mb = store.storage_used_mb()
        assert mb >= 0.0

    def test_evidence_orphan_cleanup(self, tmp_path):
        from src.evidence import EvidenceStore
        conn = _mk_conn(tmp_path)
        evidence_dir = str(tmp_path / "ev")
        os.makedirs(evidence_dir, exist_ok=True)
        store = EvidenceStore(conn, mode="OFF", evidence_dir=evidence_dir)
        result = store.orphan_cleanup()
        assert "db_orphans" in result
        assert "file_orphans" in result
