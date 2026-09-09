"""Tests for Phase 32 camera intelligence & detector registry."""

import pytest

from src import database as db
from src.camera_intelligence import CameraIntelligence, SystemHealth
from src.detector_registry import DetectorRegistry


@pytest.fixture
def cams(tmp_db):
    db.upsert_camera_config(tmp_db, {
        "camera_id": "cam_01", "name": "Entrance", "location": "Lobby",
        "zone": "lobby", "source": "rtsp://x/1", "enabled": 1, "fps_target": 15,
    })
    db.upsert_camera_config(tmp_db, {
        "camera_id": "cam_02", "name": "Archive", "location": "Archive",
        "zone": "archive", "source": "rtsp://x/2", "enabled": 1, "fps_target": 15,
    })
    return tmp_db


def test_camera_status_configured(cams):
    ci = CameraIntelligence(cams)
    db.record_camera_health(cams, "cam_01",
                            {"health": "HEALTHY", "fps": 15.0, "tampered": 0})
    st = ci.camera_status("cam_01")
    assert st["configured"] is True
    assert st["healthy"] is True


def test_degraded_detects_tamper(cams):
    ci = CameraIntelligence(cams)
    db.record_camera_health(cams, "cam_01",
                            {"health": "HEALTHY", "fps": 15.0, "tampered": 1})
    degraded = ci.degraded_cameras()
    assert any(c["camera_id"] == "cam_01" and c["reason"] == "tampered"
               for c in degraded)


def test_degraded_low_fps(cams):
    ci = CameraIntelligence(cams)
    db.record_camera_health(cams, "cam_02",
                            {"health": "HEALTHY", "fps": 2.0, "tampered": 0})
    degraded = ci.degraded_cameras()
    assert any(c["camera_id"] == "cam_02" and "low_fps" in c["reason"]
               for c in degraded)


def test_availability_pct(cams):
    db.record_camera_health(cams, "cam_01", {"health": "HEALTHY"})
    db.record_camera_health(cams, "cam_01", {"health": "DEGRADED"})
    ci = CameraIntelligence(cams)
    assert 0.0 < ci.availability_pct("cam_01") < 100.0


def test_system_health_summary(cams):
    db.record_camera_health(cams, "cam_01",
                            {"health": "HEALTHY", "fps": 15.0})
    db.record_camera_health(cams, "cam_02",
                            {"health": "DEGRADED", "fps": 1.0})
    sh = SystemHealth(cams)
    s = sh.summary()
    assert s["camera_count"] == 2
    assert s["enabled_cameras"] == 2
    assert any(c["camera_id"] == "cam_02" for c in s["degraded"])


# ----------------------------------------------------------------------
# Detector registry (anti-fabrication)
# ----------------------------------------------------------------------

def test_register_logic_detector_available(tmp_db):
    reg = DetectorRegistry(tmp_db)
    det = reg.register(detector_id="motion", name="Motion",
                       event_types="MOTION_DETECTED")
    assert det["status"] == "AVAILABLE"
    assert det["enabled"] is True


def test_ai_detector_without_model_never_enabled(tmp_db):
    reg = DetectorRegistry(tmp_db)
    det = reg.register(detector_id="weapon", name="Weapon",
                       event_types="WEAPON_DETECTED", enabled=True)
    # Non-negotiable: no fabricated AI detections, and cannot auto-enable.
    assert det["status"] == "FUTURE_MODEL_REQUIRED"
    assert det["enabled"] is False


def test_enable_ai_without_model_blocked(tmp_db):
    reg = DetectorRegistry(tmp_db)
    reg.register(detector_id="fall", name="Fall", event_types="FALL_DETECTED")
    assert reg.enable("fall") is False
    assert reg.get("fall")["status"] == "FUTURE_MODEL_REQUIRED"


def test_enable_ai_with_real_model(tmp_db, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    reg = DetectorRegistry(tmp_db)
    reg.register(detector_id="fire", name="Fire", event_types="FIRE_DETECTED",
                 model_path=str(model))
    assert reg.enable("fire") is True
    assert reg.get("fire")["status"] == "AVAILABLE"


def test_disable_sets_unavailable(tmp_db):
    reg = DetectorRegistry(tmp_db)
    reg.register(detector_id="motion", name="Motion",
                 event_types="MOTION_DETECTED", enabled=True)
    assert reg.disable("motion") is True
    assert not reg.get("motion")["enabled"]  # integer 0 -> falsy
    # B15 status vocabulary: an explicit operator/config disable is DISABLED.
    assert reg.get("motion")["status"] == "DISABLED"
