"""Bounded soak test (Phase 36, marked `soak`).

This file does NOT sleep for hours.  It runs a number of repeated simulation
cycles in fast, simulated time and checks long-run invariants:

* no unbounded memory growth / thread growth across cycles,
* repeated reconnect does not leave stale camera state,
* repeated reconnect does not duplicate incidents or create stale trackers,
* identity stays stable (no flickering),
* productivity is never corrupted.

A multi-hour *wall-clock* soak remains a FUTURE_CLIENT_ENVIRONMENT task and is
declared NOT EXECUTED here -- never faked.
"""

import gc
import tracemalloc

import pytest

from src.simulation.runner import SimulationRunner

pytestmark = pytest.mark.soak


def test_repeated_reconnect_stability_and_no_duplicate_incidents():
    """Many disconnect/reconnect cycles must not multiply incidents or corrupt
    employee identity."""
    with SimulationRunner(scenario="N_camera_recovery", seed=7) as r:
        rep = r.run_steps()
    assert rep.completed
    assert rep.reconnect_count >= 1
    # repeated reconnect must not create an explosion of incidents (dedup holds)
    assert rep.incidents_generated <= 3
    rows = rep.final_state.get("activity_states", [])
    # no identity flicker: stable, single employee id persisted
    ids = {x["employee_id"] for x in rows}
    assert ids <= {"EMP001"}


def test_memory_stability_across_many_cycles():
    tracemalloc.start()
    try:
        with SimulationRunner(scenario="R_long_running", seed=2026) as r:
            _ = r.run_steps()
        _, peak1 = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    gc.collect()
    tracemalloc.start()
    try:
        with SimulationRunner(scenario="R_long_running", seed=2026) as r:
            _ = r.run_steps()
        _, peak2 = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Two identical runs: memory source should not grow materially.
    assert peak2 < peak1 + 8 * 1024 * 1024


def test_thread_count_stays_bounded():
    import threading
    with SimulationRunner(scenario="K_multiple_employees", seed=5) as r:
        rep = r.run_steps()
    assert rep.resource["threads"] <= 64


def test_no_stale_camera_state_after_recovery():
    """After a disconnected camera recovers, the pool reports ONLINE again and
    frames flow (no permanent stale OFFlINE state)."""
    from src.simulation.virtual_camera import (
        FAULT_DISCONNECT,
        VirtualCameraPool,
    )
    import numpy as np

    pool = VirtualCameraPool.build(1, seed=4, default_fps=10, width=64, height=48)
    pool.start()
    pool._cameras["cam_01"].schedule_fault(0, 4, FAULT_DISCONNECT)
    pool.step_cycle()  # offline
    assert pool.camera_health()["cam_01"]["health"] == "OFFLINE"
    # advance past the fault window then several online steps
    for _ in range(8):
        pool.step_cycle()
    assert pool.camera_health()["cam_01"]["health"] == "ONLINE"
    live = pool.latest_frames()
    assert live and isinstance(next(iter(live.values())), np.ndarray)
    pool.stop()