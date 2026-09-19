"""Phase 54 -- defensive remediation regression tests.

Locks in the audit-derived fixes from PROJECT_FULL_SYSTEM_AUDIT_2026-09-19:

  M01  docker-compose dashboard default is AUTH-ON + FAIL-CLOSED (was
       unauthenticated open on 0.0.0.0:8501).
  M02  live-state writer logs failures (rate-limited) and recovery instead
       of silently swallowing every error -> stale dashboard with no signal.
  M04  ReID fallback is transparent: a requested-but-unloadable model
       records ``requested_model`` + ``load_error`` instead of vanishing.
  M05  advisory incidents self-close on lifecycle events (CAMERA_RECOVERED)
       and identical anomaly-pattern re-fires are deduped by cooldown.
  M06  rebuildable biometric caches are pruned only when retention is
       enabled AND the days window is > 0; enrollment images never touched.
  M07  report/EOD continuity: eod_state can never claim an xlsx that is not
       on disk, and on-disk reports without a state entry are surfaced.
"""

import json
import logging
import os
import time
from pathlib import Path

import pytest

import main
from src import database as db
from src.anomaly_engine import PATTERN_COOLDOWN_SEC, AnomalyEngine
from src.domain import INCIDENT_OPEN, INCIDENT_RESOLVED, SEV_HIGH, SEV_INFO
from src.incidents import IncidentEngine
from src.reid import AppearanceExtractor

MAIN_LOG = "cctv.main"


# ----------------------------------------------------------------------
# M01 - docker-compose secure-by-default dashboard auth
# ----------------------------------------------------------------------

def _compose_text() -> str:
    path = Path(__file__).resolve().parent.parent / "docker-compose.yml"
    return path.read_text(encoding="utf-8")


class TestM01ComposeAuthDefault:
    def test_dashboard_auth_defaults_to_on(self):
        text = _compose_text()
        assert "CCTV_DASH_AUTH=${CCTV_DASH_AUTH:-1}" in text
        assert "CCTV_DASH_AUTH=${CCTV_DASH_AUTH:-0}" not in text

    def test_fail_closed_default_and_forwarded(self):
        text = _compose_text()
        assert "CCTV_DASH_FAIL_CLOSED=${CCTV_DASH_FAIL_CLOSED:-1}" in text

    def test_session_and_optional_env_forwarded(self):
        text = _compose_text()
        assert "CCTV_DASH_SESSION_MINUTES=${CCTV_DASH_SESSION_MINUTES:-30}" in text
        assert "CCTV_DASH_PASS=${CCTV_DASH_PASS:-}" in text
        assert "CCTV_DASH_VIEWER_PASS=${CCTV_DASH_VIEWER_PASS:-}" in text

    def test_biometric_retention_env_wired_to_daemon(self):
        text = _compose_text()
        assert ("CCTV_RETENTION_BIOMETRICS_DAYS="
                "${CCTV_RETENTION_BIOMETRICS_DAYS:-0}") in text


# ----------------------------------------------------------------------
# M02 - live-state writer error visibility + recovery
# ----------------------------------------------------------------------

class TestM02LiveStateWriter:
    @pytest.fixture(autouse=True)
    def _guard(self, tmp_path):
        original = main._LIVE_STATE_FILE
        main._LIVE_STATE_FAILURES = 0
        main._LIVE_STATE_RECOVERED = False
        main._LIVE_STATE_LAST_ERROR_AT = 0.0
        main._LIVE_STATE_FILE = str(tmp_path / "live_state.json")
        yield
        main._LIVE_STATE_FILE = original  # never leak a stale tmp path
        main._LIVE_STATE_FAILURES = 0
        main._LIVE_STATE_RECOVERED = False
        main._LIVE_STATE_LAST_ERROR_AT = 0.0

    def test_failure_is_logged_not_swallowed(self, monkeypatch, caplog):
        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(main.json, "dump", boom)
        with caplog.at_level(logging.ERROR, logger=MAIN_LOG):
            main._write_live_state({}, {}, {}, 0.0)
        assert main._LIVE_STATE_FAILURES == 1
        assert main._LIVE_STATE_RECOVERED is False
        assert "live-state write FAILED" in caplog.text

    def test_recovery_is_logged_and_counter_reset(self, monkeypatch, caplog):
        boom = {"active": True}

        def flaky(*_args, **_kwargs):
            if boom["active"]:
                raise OSError("first write explodes")
            return None

        monkeypatch.setattr(main.json, "dump", flaky)
        with caplog.at_level(logging.ERROR, logger=MAIN_LOG):
            main._write_live_state({}, {}, {}, 0.0)
        boom["active"] = False
        with caplog.at_level(logging.WARNING, logger=MAIN_LOG):
            main._write_live_state({}, {}, {}, 0.0)
        assert main._LIVE_STATE_FAILURES == 0
        assert "recovered after 1 failures" in caplog.text

    def test_success_writes_snapshot(self, tmp_path):
        target = tmp_path / "live_state.json"
        main._LIVE_STATE_FILE = str(target)
        source_state = {"EMP001": "ACTIVE"}
        main._write_live_state(source_state, {}, {"CAM1": "ONLINE"}, 3.0)
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["states"] == source_state
        assert payload["camera_health"] == {"CAM1": "ONLINE"}

    def test_failure_preserves_last_good_snapshot(self, monkeypatch):
        main._write_live_state({"EMP001": "ACTIVE"},
                               {}, {"CAM1": "ONLINE"}, 3.0)

        def boom(*_args, **_kwargs):
            raise OSError("second write explodes")

        monkeypatch.setattr(main.json, "dump", boom)
        main._write_live_state({}, {}, {}, 0.0)
        with open(main._LIVE_STATE_FILE, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["states"] == {"EMP001": "ACTIVE"}
        assert main._LIVE_STATE_FAILURES == 1

    def test_repeated_failures_are_rate_limited(self, monkeypatch, caplog):
        def boom(*_args, **_kwargs):
            raise OSError("persistent fault")

        monkeypatch.setattr(main.json, "dump", boom)
        # First failure logs (with traceback); immediate repeats are silent.
        with caplog.at_level(logging.ERROR, logger=MAIN_LOG):
            main._write_live_state({}, {}, {}, 0.0)
            main._write_live_state({}, {}, {}, 0.0)
            main._write_live_state({}, {}, {}, 0.0)
        assert main._LIVE_STATE_FAILURES == 3
        assert caplog.text.count("live-state write FAILED") == 1


# ----------------------------------------------------------------------
# M04 - ReID fallback transparency
# ----------------------------------------------------------------------

class TestM04ReidTransparency:
    def test_no_model_uses_builtin_cleanly(self):
        extractor = AppearanceExtractor("")
        assert extractor.model_source == "built-in"
        assert extractor.requested_model == ""
        assert extractor.load_error is None

    def test_corrupt_checkpoint_records_error(self, tmp_path):
        bad = tmp_path / "osnet_x1_0.pth"
        bad.write_bytes(b"this is not a serialized torch checkpoint")
        extractor = AppearanceExtractor(str(bad))
        assert extractor.model_source == "built-in"
        assert extractor.checkpoint is None
        assert ("osnet_x1_0.pth" in extractor.requested_model)
        assert extractor.load_error  # captured, not swallowed

    def test_missing_requested_model_is_still_visible(self, tmp_path):
        missing = str(tmp_path / "does_not_exist.pth")
        extractor = AppearanceExtractor(missing)
        assert extractor.model_source == "built-in"
        assert extractor.requested_model.endswith("does_not_exist.pth")


# ----------------------------------------------------------------------
# M05 - incident advisory auto-close
# ----------------------------------------------------------------------

class TestM05AdvisoryAutoClose:
    def _bind_offline(self, tmp_db, severity=SEV_INFO, actor="system"):
        engine = IncidentEngine(tmp_db)
        ev = {
            "event_type": "CAMERA_OFFLINE", "timestamp": "2026-09-19 08:00:00",
            "camera": "CAM1", "zone": "Z1", "employee_id": "Unknown",
            "severity": severity,
        }
        inc_id = engine.bind(ev)
        return engine, inc_id

    def test_info_advisory_auto_closed_by_recovered(self, tmp_db):
        engine, inc_id = self._bind_offline(tmp_db, severity=SEV_INFO)
        assert engine.get(inc_id)["status"] == INCIDENT_OPEN
        ok = engine.auto_close_advisory(
            {"event_type": "CAMERA_RECOVERED", "timestamp": "2026-09-19 09:00:00",
             "incident_id": inc_id})
        assert ok is True
        assert engine.get(inc_id)["status"] == INCIDENT_RESOLVED

    def test_high_severity_incident_never_auto_closed(self, tmp_db):
        engine, inc_id = self._bind_offline(tmp_db, severity=SEV_HIGH)
        ok = engine.auto_close_advisory(
            {"event_type": "CAMERA_RECOVERED", "incident_id": inc_id})
        assert ok is False
        assert engine.get(inc_id)["status"] == INCIDENT_OPEN

    def test_non_self_closing_event_refused(self, tmp_db):
        engine, inc_id = self._bind_offline(tmp_db, severity=SEV_INFO)
        ok = engine.auto_close_advisory(
            {"event_type": "INTRUSION", "incident_id": inc_id})
        assert ok is False
        assert engine.get(inc_id)["status"] == INCIDENT_OPEN

    def test_already_resolved_noop(self, tmp_db):
        engine, inc_id = self._bind_offline(tmp_db, severity=SEV_INFO)
        assert engine.resolve(inc_id, actor="system") is True
        assert engine.auto_close_advisory(
            {"event_type": "CAMERA_RECOVERED", "incident_id": inc_id}) is False

    def test_auto_close_writes_audit_trail(self, tmp_db):
        engine, inc_id = self._bind_offline(tmp_db, severity=SEV_INFO)
        engine.auto_close_advisory(
            {"event_type": "CAMERA_RECOVERED", "incident_id": inc_id},
            actor="retention-bot")
        actions = [e["action"] for e in db.list_audit_events(tmp_db)]
        assert "incident.resolved" in actions


# ----------------------------------------------------------------------
# M05 - anomaly pattern cooldown dedup
# ----------------------------------------------------------------------

class TestM05AnomalyCooldown:
    def _insert_events(self, tmp_db, zone, count, event="AFTER_HOURS_ACTIVITY"):
        from datetime import date, timedelta
        for i in range(count):
            d = (date.today() - timedelta(days=i % 3)).strftime("%Y-%m-%d")
            db.insert_security_event(tmp_db, {
                "event_type": event, "severity": "MEDIUM",
                "camera": "cam_01", "zone": zone,
                "timestamp": f"{d} 22:00:00", "date": d})

    def test_no_prior_repeat_fires(self, tmp_db):
        ae = AnomalyEngine(tmp_db)
        assert ae._pattern_should_fire("Z1", 4) is True

    def test_unchanged_count_inside_cooldown_suppressed(self, tmp_db):
        ae = AnomalyEngine(tmp_db)
        self._insert_events(tmp_db, "Z1", 4)
        found = ae.detect_recurring_patterns(min_occurrences=4)
        assert len(found) == 1
        # Same observation immediately re-evaluated -> deduped.
        ae2 = AnomalyEngine(tmp_db)
        assert ae2._pattern_should_fire("Z1", 4) is False
        found2 = ae2.detect_recurring_patterns(min_occurrences=4)
        assert found2 == []

    def test_grown_count_refires_immediately(self, tmp_db):
        ae = AnomalyEngine(tmp_db)
        self._insert_events(tmp_db, "Z1", 4)
        assert len(ae.detect_recurring_patterns(min_occurrences=4)) == 1
        self._insert_events(tmp_db, "Z1", 6)  # now 10 events total
        ae2 = AnomalyEngine(tmp_db)
        assert ae2._pattern_should_fire("Z1", 10) is True
        assert len(ae2.detect_recurring_patterns(min_occurrences=4)) == 1

    def test_cooldown_expiry_allows_reminder(self, tmp_db):
        ae = AnomalyEngine(tmp_db)
        self._insert_events(tmp_db, "Z1", 4)
        assert len(ae.detect_recurring_patterns(min_occurrences=4)) == 1
        # Age the stored pattern beyond the cooldown window.
        tmp_db.execute(
            "UPDATE anomaly_events SET observed_at = datetime('now', 'localtime', ?) "
            "WHERE anomaly_type='REPEATED_PATTERN_DETECTED'",
            (f"-{int(PATTERN_COOLDOWN_SEC) + 60} seconds",))
        tmp_db.commit()
        ae2 = AnomalyEngine(tmp_db)
        assert ae2._pattern_should_fire("Z1", 4) is True


# ----------------------------------------------------------------------
# M06 - biometric cache retention
# ----------------------------------------------------------------------

def _set_mtime(path: Path, *, old: bool):
    stamp = time.time() - 366 * 24 * 3600 if old else time.time()
    os.utime(path, (stamp, stamp))


class _FakeEvidenceStore:
    def __init__(self, conn, **kwargs):
        self.removed = 0

    def retention_cleanup(self):
        return 0


class TestM06BiometricRetention:
    @pytest.fixture
    def retention_env(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(main, "ENABLE_RETENTION", True)
        monkeypatch.setattr(main, "RETENTION_DAYS_BIOMETRICS", 30)
        monkeypatch.setattr(main, "RETENTION_DAYS_LOGS", 365)
        monkeypatch.setattr(main, "RETENTION_DAYS_REPORTS", 365)
        monkeypatch.setattr(main, "REPORT_OUTPUT_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr(main, "_LAST_RETENTION_DAY", None)

        emb = tmp_path / "embeddings.pkl"
        appr = tmp_path / "appearance.pkl"
        old_emb = tmp_path / "old_embeddings.pkl"
        monkeypatch.setattr(config, "EMBEDDINGS_FILE", str(emb))
        monkeypatch.setattr(config, "APPEARANCE_EMBEDDINGS_FILE", str(appr))

        # Retention-adjacent stages are no-ops in this unit test.
        monkeypatch.setattr(main, "retention_cleanup", lambda *a, **k: 0)
        import src.database as sdb
        monkeypatch.setattr(sdb, "security_retention_cleanup",
                            lambda *a, **k: {"events": 0, "alerts": 0, "audit": 0})
        import src.evidence as sev
        monkeypatch.setattr(sev, "EvidenceStore", _FakeEvidenceStore)

        emb.write_bytes(b"face cache")
        appr.write_bytes(b"appearance cache")
        _set_mtime(emb, old=True)
        _set_mtime(appr, old=False)
        return emb, appr

    def test_stale_cache_pruned_fresh_kept(self, tmp_db, retention_env, caplog):
        emb, appr = retention_env
        assert emb.exists() and appr.exists()
        with caplog.at_level(logging.WARNING, logger=MAIN_LOG):
            main._maybe_run_retention(tmp_db)
        assert not emb.exists()
        assert appr.exists()
        assert "Retention (biometrics)" in caplog.text
        assert "embeddings.pkl" in caplog.text

    def test_disabled_retention_never_prunes(self, tmp_db, retention_env,
                                             monkeypatch):
        monkeypatch.setattr(main, "ENABLE_RETENTION", False)
        monkeypatch.setattr(main, "_LAST_RETENTION_DAY", None)
        emb, appr = retention_env
        main._maybe_run_retention(tmp_db)
        assert emb.exists()
        assert appr.exists()

    def test_zero_window_never_prunes(self, tmp_db, retention_env, monkeypatch):
        monkeypatch.setattr(main, "RETENTION_DAYS_BIOMETRICS", 0)
        monkeypatch.setattr(main, "_LAST_RETENTION_DAY", None)
        emb, appr = retention_env
        main._maybe_run_retention(tmp_db)
        assert emb.exists()
        assert appr.exists()


# ----------------------------------------------------------------------
# M07 - report/EOD continuity reconcile
# ----------------------------------------------------------------------

class TestM07EodReconcile:
    @pytest.fixture(autouse=True)
    def _paths(self, tmp_path, monkeypatch):
        state_file = tmp_path / "eod_state.json"
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        monkeypatch.setattr(main, "_EOD_STATE_FILE", str(state_file))
        monkeypatch.setattr(main, "REPORT_OUTPUT_DIR", str(reports_dir))
        self.state_file = state_file
        self.reports_dir = reports_dir

    def _write_state(self, payload):
        self.state_file.write_text(json.dumps(payload), encoding="utf-8")

    def test_mark_without_xlsx_flagged(self, caplog):
        self._write_state({"2026-09-01": {"report_generated": True}})
        with caplog.at_level(logging.WARNING, logger=MAIN_LOG):
            result = main._reconcile_eod_state()
        assert result == {"state_days": 1, "xlsx_days": 0}
        assert "2026-09-01" in caplog.text
        assert "report_generated" in caplog.text

    def test_mark_with_matching_xlsx_clean(self, caplog):
        (self.reports_dir / "daily_report_2026-09-01.xlsx").write_bytes(b"x")
        self._write_state({"2026-09-01": {"report_generated": True}})
        with caplog.at_level(logging.WARNING, logger=MAIN_LOG):
            main._reconcile_eod_state()
        assert "report_generated" not in caplog.text

    def test_orphan_xlsx_without_state_surfaced(self, caplog):
        (self.reports_dir / "daily_report_2026-09-02.xlsx").write_bytes(b"x")
        with caplog.at_level(logging.INFO, logger=MAIN_LOG):
            result = main._reconcile_eod_state()
        assert result == {"state_days": 0, "xlsx_days": 1}
        assert "2026-09-02" in caplog.text