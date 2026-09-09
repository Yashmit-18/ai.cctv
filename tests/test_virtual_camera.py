"""Unit tests for virtual camera sources + pool (Phase 36, simulation).

Verifies deterministic frame generation, camera-level fault injection
(frozen/dark/bright/noisy/disconnect/reconnect), and that ``VirtualCameraPool``
exposes the same batch/health interface the production pipeline consumes
including resource-limit fail-safe.
"""

import numpy as np
import pytest

from src.simulation.virtual_camera import (
    FAULT_DARK,
    FAULT_DISCONNECT,
    FAULT_FROZEN,
    ResourceLimitError,
    VirtualCamera,
    VirtualCameraPool,
)


@pytest.mark.simulation
def test_virtual_camera_frames_deterministic():
    a = VirtualCamera("cam_01", width=64, height=48, fps=10, seed=7)
    b = VirtualCamera("cam_01", width=64, height=48, fps=10, seed=7)
    f1 = a.next_frame(0)
    f2 = b.next_frame(0)
    assert f1.shape == (48, 64, 3)
    assert f1.dtype == np.uint8
    # same seed => identical baseline scene
    assert np.array_equal(f1, f2)


@pytest.mark.simulation
def test_virtual_camera_dark_and_bright():
    c = VirtualCamera("cam_01", width=64, height=48, fps=10, seed=1)
    c.schedule_fault(0, 5, FAULT_DARK)
    c.schedule_fault(10, 15, "bright")
    dark = c.next_frame(1)
    assert int(dark.mean()) < 20
    bright = c.next_frame(12)
    assert int(bright.mean()) > 200


@pytest.mark.simulation
def test_virtual_camera_frozen_identical():
    c = VirtualCamera("cam_01", width=64, height=48, fps=10, seed=2)
    c.schedule_fault(0, 10, FAULT_FROZEN)
    f1 = c.next_frame(1).copy()
    f2 = c.next_frame(3)
    assert np.array_equal(f1, f2)


@pytest.mark.simulation
def test_virtual_camera_disconnect_returns_none_and_reconnects():
    c = VirtualCamera("cam_01", width=64, height=48, fps=10, seed=3)
    c.schedule_fault(2, 6, FAULT_DISCONNECT)
    assert c.next_frame(1) is not None
    assert c.next_frame(3) is None      # offline window
    assert c.next_frame(5) is None
    assert c.next_frame(9) is not None  # recovered
    assert c.reconnect_count >= 1


@pytest.mark.simulation
def test_virtual_camera_rejects_bad_config():
    with pytest.raises(ValueError):
        VirtualCamera("cam_01", width=8, height=48)
    with pytest.raises(ValueError):
        VirtualCamera("cam_01", fps=0)


@pytest.mark.simulation
def test_pool_matches_production_interface():
    pool = VirtualCameraPool.build(2, seed=5, default_fps=10, width=64, height=48)
    pool.start()
    pool.step_cycle()
    batch = pool.get_latest_batch()
    assert list(batch.keys()) == ["cam_01", "cam_02"]
    assert all(f is not None for f in batch.values())
    health = pool.camera_health()
    assert all(h["health"] == "ONLINE" for h in health.values())
    assert pool.camera_ids == ["cam_01", "cam_02"]
    assert pool.online_count == 2
    assert pool.frame_interval(10) == pytest.approx(0.1)
    pool.stop()


@pytest.mark.simulation
def test_pool_disconnect_health_offline():
    pool = VirtualCameraPool([VirtualCamera("cam_01", width=64, height=48,
                                            fps=10, seed=1)])
    pool.start()
    pool._cameras["cam_01"].schedule_fault(0, 5, FAULT_DISCONNECT)
    pool.step_cycle()
    assert pool.camera_health()["cam_01"]["health"] == "OFFLINE"
    assert pool.latest_frames() == {}
    # online after recovery
    pool.step_cycle()
    pool.step_cycle()
    pool.step_cycle()
    pool.step_cycle()
    pool.step_cycle()
    pool.step_cycle()
    assert pool.latest_frames() != {}


@pytest.mark.simulation
def test_pool_enforces_max_cameras():
    with pytest.raises(ResourceLimitError):
        VirtualCameraPool.build(21)  # > default max_cameras=20


@pytest.mark.simulation
def test_build_rejects_empty_pool():
    with pytest.raises(ValueError):
        VirtualCameraPool.build(0)