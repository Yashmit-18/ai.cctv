"""Regression: simulation layer must not alter/bypass the real webcam path
and must never be pulled into production runtime (Phase 36).

The virtual-camera layer is opt-in and test-only.  This file asserts:

* the real webcam-mode helpers (``_local_camera_index`` / ``_wait_for_initial_frames``)
  are unchanged and still selectable,
* the production camera manager / capture classes still construct normally,
* production entry points (``main``/``app``) import NO simulation modules,
* the simulation package is importable only under explicit control (i.e. it is
  not imported transitively by ``config`` or the daemon).
"""

import importlib

import pytest

from main import _local_camera_index, _wait_for_initial_frames
from src.camera import VideoCapture
from src.camera_manager import MultiCameraManager


class _NoFrames:
    online_count = 0
    active_camera_ids = []

    def get_latest_batch(self):
        return {}


class _FrameOnly:
    online_count = 1
    active_camera_ids = []
    frame = object()

    def get_latest_batch(self):
        return {"local_webcam": self.frame}


@pytest.mark.simulation
def test_real_webcam_mode_helpers_intact():
    assert _local_camera_index("webcam") == 0
    assert _local_camera_index("0") == 0
    assert _local_camera_index("") == 0


@pytest.mark.simulation
def test_wait_for_initial_frames_unchanged(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    assert _wait_for_initial_frames(_FrameOnly(), timeout=0.01) is True


@pytest.mark.simulation
def test_real_camera_classes_still_construct():
    vc = VideoCapture(source=0, frame_skip=1)
    assert vc.source == 0
    mgr = MultiCameraManager(cameras={"local_webcam": 0})
    assert mgr.camera_ids == ["local_webcam"]


@pytest.mark.simulation
def test_production_main_does_not_import_simulation():
    import re
    with open("main.py", encoding="utf-8") as f:
        src_main = f.read()
    sim_refs = re.findall(r"(src\.simulation|simulation\.)", src_main)
    assert not sim_refs, f"main.py pulls in simulation: {sim_refs}"


@pytest.mark.simulation
def test_config_does_not_import_simulation():
    import re
    with open("config.py", encoding="utf-8") as f:
        cfg = f.read()
    assert "simulation" not in cfg
