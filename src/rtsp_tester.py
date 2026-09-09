"""RTSP multi-stream stress tester (Phase 7).

Spawns one daemon reader thread per camera in ``config.CAMERAS`` and
validates that all network sockets can be held open simultaneously.
Opens each stream with FFMPEG (TCP transport, buffer=1) but performs *no*
computer-vision inference and renders no video windows -- this deliberately
minimises memory pressure so we can isolate pure network/decoder capacity.

Prints a live console table (refreshed in place) showing each camera's
ID, stream resolution, connection status, and a rolling ping/latency.

Usage
-----
    python -m src.rtsp_tester              # uses all config.CAMERAS
    python -m src.rtsp_tester --duration 30
    python -m src.rtsp_tester --cameras cam_01 cam_02 cam_03
"""

import argparse
import logging
import os
import sys
import threading
import time

import cv2

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_ROOT)
    import config
    from config import CAMERAS, setup_logging
except Exception:  # pragma: no cover - only when invoked from another cwd
    raise

logger = logging.getLogger("cctv.rtsp_tester")

_READ_TIMEOUT = 5.0
_REFRESH_SEC = 1.0
_FALLBACK = (0, 0, "N/A")


class _Probe:
    """Per-camera probe state shared between reader thread and printer."""

    def __init__(self, cam_id: str, url: str):
        self.cam_id = cam_id
        self.url = url
        self.lock = threading.Lock()
        self.connected = False
        self.width = 0
        self.height = 0
        self.frames = 0
        self.ping_ms = 0.0
        self.error: str | None = None
        self._cap: cv2.VideoCapture | None = None
        self._running = False

    def _release(self):
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        with self.lock:
            self._cap = None
            self.connected = False


def _resolve_url(cam_id: str) -> str:
    return CAMERAS[cam_id]


def _open(probe: _Probe) -> bool:
    probe._release()
    cap = cv2.VideoCapture(probe.url, cv2.CAP_FFMPEG)
    if cap is None or not cap.isOpened():
        with probe.lock:
            probe.error = "open failed"
        return False
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    with probe.lock:
        probe._cap = cap
        probe.connected = True
        probe.width = w
        probe.height = h
        probe.error = None
    return True


def _timed_read(cap) -> tuple[bool, int, int, float]:
    """Non-blocking read with a hard timeout; returns (ret, w, h, ms)."""
    t0 = time.perf_counter()
    result: list = [False, 0, 0]
    holder: dict = {}

    def _do():
        ok, frame = cap.read()
        result[0] = ok
        if ok and frame is not None:
            holder["w"] = frame.shape[1]
            holder["h"] = frame.shape[0]
        return ok

    thr = threading.Thread(target=_do, daemon=True)
    thr.start()
    thr.join(timeout=_READ_TIMEOUT)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if thr.is_alive():
        return False, 0, 0, elapsed_ms
    return result[0], holder.get("w", 0), holder.get("h", 0), elapsed_ms


def _reader(probe: _Probe):
    probe._running = True
    backoff = 1.0
    _open(probe)
    while probe._running:
        with probe.lock:
            cap = probe._cap
        if cap is None or not cap.isOpened():
            # -- reconnect with exponential backoff --
            with probe.lock:
                probe.connected = False
            time.sleep(backoff)
            if _open(probe):
                backoff = 1.0
            else:
                backoff = min(backoff * 2.0, 30.0)
            continue

        ok, w, h, ms = _timed_read(cap)
        with probe.lock:
            if ok and w:
                probe.width, probe.height = w, h
                probe.frames += 1
                probe.ping_ms = ms
            else:
                probe.connected = False
                probe.error = "read failed"
        if not ok:
            probe._release()
            with probe.lock:
                probe.connected = False
    probe._running = False
    probe._release()


def _render(probes, duration: float):
    start = time.time()
    while time.time() - start < duration:
        print("\x1b[H\x1b[2J", end="")  # clear screen
        print(f"RTSP Stress Test  |  {len(probes)} cameras  |  "
              f"{time.time()-start:.0f}s / {duration:.0f}s\n")
        header = f"{'Cam':<8}{'Status':<9}{'Res':<14}{'Ping(ms)':<10}{'Frames':<8}"
        print(header)
        print("-" * len(header))
        for p in probes:
            with p.lock:
                status = "OK" if p.connected else ("STALL" if p.error else "DOWN")
                res = f"{p.width}x{p.height}" if p.width else "N/A"
                ping = f"{p.ping_ms:.0f}" if p.connected else "-"
                frames = p.frames
            bar = "  " if p.connected else "!!"
            print(f"{p.cam_id:<8}{status:<9}{res:<14}{ping:<10}{frames:<8}{bar}")
        print("Press Ctrl+C to exit")
        time.sleep(_REFRESH_SEC)


def run(args: argparse.Namespace | None = None):
    args = args or _parse_args()
    cam_ids = args.cameras or list(CAMERAS.keys())

    probes = [_Probe(cid, _resolve_url(cid)) for cid in cam_ids]
    if not probes:
        logger.error("No cameras selected. Add CAMERAS to config.py.")
        return 1

    threads = [threading.Thread(target=_reader, args=(p,), daemon=True) for p in probes]
    for t in threads:
        t.start()

    try:
        _render(probes, args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        for p in probes:
            p._running = False
        for t in threads:
            t.join(timeout=3)

    ok = sum(1 for p in probes if p.connected)
    print(f"\nResult: {ok}/{len(probes)} cameras holding steady.")
    return 0 if ok == len(probes) else 2


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTSP multi-stream stress tester")
    p.add_argument("--cameras", nargs="*", help="Specific camera IDs to test.")
    p.add_argument("--duration", type=float, default=30.0, help="Test duration (s).")
    return p.parse_args()


if __name__ == "__main__":
    setup_logging(logging.WARNING)
    sys.exit(run())
