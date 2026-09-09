"""Virtual camera sources and a pooled camera manager for simulation (Phase 36).

``VirtualCamera`` produces real ``numpy`` BGR frames with an independent
resolution and FPS, and can deterministically inject camera-level faults:

* frozen feed (identical frames)
* dark / overexposed (bright) frames
* noisy frames
* disconnect / reconnect (with configurable outage window)

``VirtualCameraPool`` exposes the SAME interface the production pipeline
consumes (``latest_frames`` / ``get_latest_batch`` / ``camera_health`` /
``online_count`` / ``stop``) so the existing ingestion, health and tamper
machinery runs against virtual sources unchanged.

Simulation-only: A *virtual* camera is NOT a real RTSP camera.  Camera-level
faults here are explicit and opt-in; none can be triggered by production code.
Frame generation is deterministic under a fixed seed.
"""

from __future__ import annotations

import numpy as np

from src.domain import CAM_ONLINE, CAM_OFFLINE, CAM_RECONNECTING

# ----------------------------------------------------------------------
# Fault kinds (camera-level, deterministic when seeded)
# ----------------------------------------------------------------------
FAULT_NONE = "none"
FAULT_FROZEN = "frozen"
FAULT_DARK = "dark"
FAULT_BRIGHT = "bright"
FAULT_NOISY = "noisy"
FAULT_DISCONNECT = "disconnect"   # offline for a fixed outage window

# Hard cap on virtual cameras the simulator may construct at once (protects the
# host machine from a runaway simulation).
MAX_VIRTUAL_CAMERAS = 20


class ResourceLimitError(RuntimeError):
    """Raised when a simulation exceeds a configured / enforced resource limit.

    Used to guarantee the simulator cannot DOS the host machine (max cameras,
    max fps, max queue, max duration, max evidence).
    """


def _clamp01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


class VirtualCamera:
    """A deterministic synthetic frame source.

    Parameters
    ----------
    camera_id : str
        ``cam_01`` style id.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    fps : float
        Target frame rate for this camera.
    seed : int
        Deterministic RNG seed (reproducible frame content per run).
    """

    def __init__(self, camera_id: str, *, width: int = 640, height: int = 360,
                 fps: float = 15.0, seed: int = 0):
        self.camera_id = camera_id
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        if self.width < 16 or self.height < 16:
            raise ValueError("virtual camera resolution too small")
        if self.fps <= 0:
            raise ValueError("virtual camera fps must be > 0")
        self._rng = np.random.default_rng(seed)
        self._frame_index = 0
        self._frozen_frame: np.ndarray | None = None
        # Fault schedule: list of (start_step, end_step, kind). Seeded below.
        self._schedule: list[tuple[int, int, str]] = []
        self._disconnect_until = 0
        self._reconnects = 0
        self.telemetry_faults: list[str] = []

    # ------------------------------------------------------------------
    # Fault scheduling (deterministic, opt-in)
    # ------------------------------------------------------------------
    def schedule_fault(self, start_step: int, end_step: int,
                       kind: str = FAULT_DARK) -> None:
        if kind not in (FAULT_DARK, FAULT_BRIGHT, FAULT_NOISY, FAULT_FROZEN,
                        FAULT_DISCONNECT):
            raise ValueError(f"unsupported virtual fault kind: {kind!r}")
        if start_step < 0 or end_step < start_step:
            raise ValueError("invalid fault window")
        self._schedule.append((int(start_step), int(end_step), kind))

    @property
    def reconnect_count(self) -> int:
        return self._reconnects

    # ------------------------------------------------------------------
    # Frame generation
    # ------------------------------------------------------------------
    def _current_fault(self, step: int) -> str:
        for start, end, kind in self._schedule:
            if start <= step < end:
                return kind
        return FAULT_NONE

    def _advance_disconnect(self, step: int) -> bool:
        # A disconnect fault forces the camera offline until the end step;
        # on recovery we count a reconnect.
        fault = self._current_fault(step)
        if fault == FAULT_DISCONNECT and self._disconnect_until <= step:
            self._disconnect_until = step + 1_000_000  # stays off during window
        if fault == FAULT_DISCONNECT:
            return False  # offline -> no frame emitted
        if self._disconnect_until != 0:
            self._disconnect_until = 0
            self._reconnects += 1
            self.telemetry_faults.append("reconnect")
        return True

    def _num_scene_boxes(self) -> int:
        # A seeded, small, reproducible set of neutral office markers used to
        # give frames stable content.  Not a person detection: it is harmless
        # background texture (desk/stationary clutter) below detector
        # thresholds.  Faces/people are never synthesised here.
        return int(self._rng.integers(3, 7))

    def next_frame(self, step: int = 0) -> np.ndarray | None:
        """Return the frame for ``step``, or ``None`` while faulted offline."""
        if not self._advance_disconnect(step):
            fault = self._current_fault(step)
            if fault == FAULT_DISCONNECT:
                return None
        fault = self._current_fault(step)

        if fault == FAULT_DARK:
            frame = np.full((self.height, self.width, 3), 5, dtype=np.uint8)
            self.telemetry_faults.append("dark")
        elif fault == FAULT_BRIGHT:
            frame = np.full((self.height, self.width, 3), 245, dtype=np.uint8)
            self.telemetry_faults.append("bright")
        elif fault == FAULT_FROZEN:
            if self._frozen_frame is None:
                self._frozen_frame = self._base_frame()
            frame = self._frozen_frame.copy()
            self.telemetry_faults.append("frozen")
        elif fault == FAULT_NOISY:
            base = self._base_frame().astype(np.int16)
            noise = self._rng.integers(-40, 40, size=base.shape)
            frame = np.clip(base + noise, 0, 255).astype(np.uint8)
            self.telemetry_faults.append("noisy")
        else:
            frame = self._base_frame()

        self._frame_index += 1
        return frame

    def _base_frame(self) -> np.ndarray:
        """Seeded neutral office background (desks/walls) with slight motion."""
        h, w = self.height, self.width
        rng = self._rng
        base = rng.integers(95, 150, size=(h, w, 3)).astype(np.uint8)
        # a couple of desk rectangles as stable office texture
        for _ in range(self._num_scene_boxes()):
            bx = int(rng.integers(0, w - 60))
            by = int(rng.integers(0, h - 40))
            bw = int(rng.integers(40, 80))
            bh = int(rng.integers(24, 44))
            shade = int(rng.integers(60, 90))
            base[by:by + bh, bx:bx + bw] = (shade, shade, shade + 6)
        # slow-moving subtle texture shift per frame for determinism
        shift = int(self._frame_index % 4)
        if shift:
            base = np.roll(base, shift * 2, axis=1)
        return base


class VirtualCameraPool:
    """A pool of ``VirtualCamera`` exposing the production camera-pool API.

    Mirrors the methods the daemon/tests expect from ``MultiCameraManager``:
    ``latest_frames``, ``get_latest_batch``, ``camera_health``,
    ``online_count``, ``active_camera_ids``, ``is_connected``, ``stop``.
    """

    def __init__(self, cameras: list[VirtualCamera]):
        if not cameras:
            raise ValueError("VirtualCameraPool needs at least one camera")
        self._cameras = {c.camera_id: c for c in cameras}
        self._started = False
        self._step = 0

    @classmethod
    def build(cls, n: int, *, seed: int = 0, default_fps: float = 15.0,
              width: int = 640, height: int = 360) -> "VirtualCameraPool":
        """Build ``n`` virtual cameras (meeting a max-camera guard)."""
        if n < 1:
            raise ValueError("at least one virtual camera required")
        if n > MAX_VIRTUAL_CAMERAS:
            raise ResourceLimitError(
                f"max virtual cameras is {MAX_VIRTUAL_CAMERAS} (requested {n})")
        return cls([VirtualCamera(f"cam_{i:02d}", width=width, height=height,
                                  fps=default_fps, seed=seed + i)
                    for i in range(1, n + 1)])

    def start(self) -> "VirtualCameraPool":
        self._started = True
        return self

    def stop(self):
        self._started = False

    def step_cycle(self) -> int:
        """Advance one camera frame step; returns the new step index."""
        self._step += 1
        return self._step

    @property
    def step(self) -> int:
        return self._step

    @property
    def camera_ids(self) -> list[str]:
        return list(self._cameras.keys())

    @property
    def active_camera_ids(self) -> list[str]:
        return [cid for cid, c in self._cameras.items() if c.reconnect_count
                is not None and True]

    @property
    def online_count(self) -> int:
        health = self.camera_health()
        return sum(1 for h in health.values() if h.get("health") == CAM_ONLINE)

    def is_connected(self, cam_id: str) -> bool:
        c = self._cameras.get(cam_id)
        if not c:
            return False
        return self._current_fault(c) != FAULT_DISCONNECT

    def _current_fault(self, cam: VirtualCamera) -> str:
        for start, end, kind in cam._schedule:  # noqa: SLF001 (simulation-owned)
            if start <= self._step < end:
                return kind
        return FAULT_NONE

    # ------------------------------------------------------------------
    # Frame batch API (matches MultiCameraManager)
    # ------------------------------------------------------------------
    def get_latest_batch(self) -> dict[str, np.ndarray | None]:
        out: dict[str, np.ndarray | None] = {}
        for cid, cam in self._cameras.items():
            out[cid] = cam.next_frame(self._step)
        return out

    def latest_frames(self, limit: int | None = None) -> dict[str, np.ndarray]:
        batch = self.get_latest_batch()
        live = {cid: f for cid, f in batch.items() if f is not None}
        if limit is not None:
            live = {cid: live[cid] for cid in list(live)[:limit]}
        return live

    def frame_interval(self, rate: float = 15.0) -> float:
        return 1.0 / rate if rate and rate > 0 else 0.0

    # ------------------------------------------------------------------
    # Health API (matches MultiCameraManager.camera_health)
    # ------------------------------------------------------------------
    def camera_health(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for cid, cam in self._cameras.items():
            fault = self._current_fault(cam)
            if fault == FAULT_DISCONNECT:
                health = CAM_OFFLINE
            else:
                health = CAM_ONLINE
            out[cid] = {
                "id": cid,
                "source": f"virtual:{cid}",
                "health": health,
                "frames_read": cam._frame_index,  # noqa: SLF001
                "fps": round(cam.fps, 2),
                "reconnects": cam.reconnect_count,
                "last_frame": float(self._step),
                "connected": self.is_connected(cid),
                "virtual": True,
            }
        return out