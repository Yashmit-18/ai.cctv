"""RTSP client environment validation harness (Phase 37).

A controlled test harness for validating real RTSP camera connectivity in
a client environment.  Designed to be run by a field engineer or automated
CI before the first live pilot session.

Features
--------
* Connect to one or more RTSP cameras.
* Measure actual FPS and frame latency.
* Detect disconnects and recovery.
* Verify camera health state transitions.
* Never logs credentials.

Usage
-----
    python -m src.rtsp_harness --url rtsp://user:pass@192.168.1.100:554/stream1
    python -m src.rtsp_harness --camera-file cameras.json --duration 60
    python -m src.rtsp_harness --from-config   # uses config.CAMERAS
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
from pathlib import Path

logger = logging.getLogger("cctv.rtsp_harness")

_READ_TIMEOUT = 5.0
_STALE_FRAME_SEC = 10.0


@dataclass
class CameraConfig:
    """Single camera validation configuration."""
    camera_id: str
    url: str
    resolution: str = ""
    expected_fps: int = 0
    credentials_from_env: bool = False


@dataclass
class CameraResult:
    """Validation result for a single camera."""
    camera_id: str
    url_redacted: str = ""
    connected: bool = False
    frames_received: int = 0
    actual_fps: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    resolution: str = ""
    disconnects: int = 0
    recoveries: int = 0
    errors: list[str] = field(default_factory=list)
    health_transitions: list[str] = field(default_factory=list)
    duration_sec: float = 0.0
    pass_: bool = False

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "url_redacted": self.url_redacted,
            "connected": self.connected,
            "frames_received": self.frames_received,
            "actual_fps": round(self.actual_fps, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "max_latency_ms": round(self.max_latency_ms, 1),
            "resolution": self.resolution,
            "disconnects": self.disconnects,
            "recoveries": self.recoveries,
            "errors": self.errors,
            "health_transitions": self.health_transitions,
            "duration_sec": round(self.duration_sec, 1),
            "pass": self.pass_,
        }


def _redact_url(url: str) -> str:
    """Strip credentials from an RTSP URL for safe logging."""
    try:
        if "@" in url:
            protocol, rest = url.split("://", 1)
            _, host_part = rest.split("@", 1)
            return f"{protocol}://***@{host_part}"
    except (ValueError, IndexError):
        pass
    return url


def _timed_read(cap: cv2.VideoCapture) -> tuple[bool, float]:
    """Read a frame with a timeout, returning (ok, latency_ms)."""
    t0 = time.perf_counter()
    result: list = [False, None]

    def _do():
        if cap is not None:
            try:
                ret, frame = cap.read()
                result[0] = ret
                result[1] = frame
            except Exception:
                result[0] = False

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(timeout=_READ_TIMEOUT)
    latency = (time.perf_counter() - t0) * 1000.0
    if t.is_alive():
        return False, latency
    return result[0], latency


def validate_camera(config: CameraConfig, duration: float = 30.0,
                    verbose: bool = False) -> CameraResult:
    """Validate a single RTSP camera connection.

    Parameters
    ----------
    config : CameraConfig
        Camera URL and metadata.
    duration : float
        How long (seconds) to hold the connection and measure.
    verbose : bool
        Print live progress.
    """
    from src.domain import redact_url
    result = CameraResult(
        camera_id=config.camera_id,
        url_redacted=redact_url(config.url),
    )

    if verbose:
        print(f"\n[{config.camera_id}] Connecting to {result.url_redacted} ...")

    cap = cv2.VideoCapture(config.url, cv2.CAP_FFMPEG)
    if cap is None or not cap.isOpened():
        result.errors.append("Failed to open RTSP stream")
        if verbose:
            print(f"  FAILED: Could not open stream")
        return result

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

    result.connected = True
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result.resolution = f"{w}x{h}"
    if verbose:
        print(f"  Connected: {result.resolution}")

    start_time = time.time()
    fps_window_start = start_time
    fps_window_frames = 0
    latencies: list[float] = []
    prev_connected = True

    while (time.time() - start_time) < duration:
        ok, latency = _timed_read(cap)

        if ok:
            result.frames_received += 1
            fps_window_frames += 1
            latencies.append(latency)

            now = time.time()
            if now - fps_window_start >= 1.0:
                result.actual_fps = fps_window_frames / (now - fps_window_start)
                fps_window_frames = 0
                fps_window_start = now

            if not prev_connected:
                result.recoveries += 1
                result.health_transitions.append(
                    f"RECOVERED at {time.time() - start_time:.1f}s")
                prev_connected = True
                if verbose:
                    print(f"  RECOVERED at {time.time() - start_time:.1f}s")
        else:
            if prev_connected:
                result.disconnects += 1
                result.health_transitions.append(
                    f"DISCONNECTED at {time.time() - start_time:.1f}s")
                prev_connected = False
                if verbose:
                    print(f"  DISCONNECT at {time.time() - start_time:.1f}s")

        time.sleep(0.05)

    cap.release()
    result.duration_sec = time.time() - start_time

    if latencies:
        result.avg_latency_ms = sum(latencies) / len(latencies)
        result.max_latency_ms = max(latencies)

    # Pass criteria: connected, received frames, no unhandled errors
    result.pass_ = (
        result.connected
        and result.frames_received > 0
        and len(result.errors) == 0
    )

    if verbose:
        status = "PASS" if result.pass_ else "FAIL"
        print(f"  Result: {status} | frames={result.frames_received} "
              f"| fps={result.actual_fps:.1f} | "
              f"latency={result.avg_latency_ms:.0f}ms avg / "
              f"{result.max_latency_ms:.0f}ms max | "
              f"disconnects={result.disconnects}")

    return result


def validate_all(configs: list[CameraConfig], duration: float = 30.0,
                 verbose: bool = False) -> list[CameraResult]:
    """Validate multiple cameras concurrently."""
    threads: list[threading.Thread] = []
    results: dict[str, CameraResult] = {}

    def _run(cfg: CameraConfig):
        results[cfg.camera_id] = validate_camera(cfg, duration, verbose)

    for cfg in configs:
        t = threading.Thread(target=_run, args=(cfg,), daemon=True)
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=duration + 30)

    return [results[cfg.camera_id] for cfg in configs]


def load_camera_file(path: str) -> list[CameraConfig]:
    """Load camera configs from a JSON file.

    Expected format::

        [
            {"camera_id": "cam_01", "url": "rtsp://...", "resolution": "1920x1080"},
            ...
        ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    configs = []
    for entry in data:
        configs.append(CameraConfig(
            camera_id=entry.get("camera_id", "unknown"),
            url=entry.get("url", ""),
            resolution=entry.get("resolution", ""),
            expected_fps=entry.get("expected_fps", 0),
        ))
    return configs


def load_from_config() -> list[CameraConfig]:
    """Load cameras from config.CAMERAS."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config
    configs = []
    for cam_id, url in config.CAMERAS.items():
        configs.append(CameraConfig(
            camera_id=cam_id,
            url=url,
        ))
    return configs


def print_summary(results: list[CameraResult]):
    """Print a summary table."""
    from src.domain import redact_url
    print("\n" + "=" * 70)
    print("  RTSP Harness -- Validation Summary")
    print("=" * 70)
    header = f"{'Camera':<10}{'Status':<8}{'Res':<14}{'FPS':<8}{'Latency':<12}{'Frames':<8}{'DC':<5}"
    print(header)
    print("-" * len(header))
    all_pass = True
    for r in results:
        status = "PASS" if r.pass_ else "FAIL"
        if not r.pass_:
            all_pass = False
        lat = f"{r.avg_latency_ms:.0f}/{r.max_latency_ms:.0f}ms" if r.avg_latency_ms else "N/A"
        print(f"{r.camera_id:<10}{status:<8}{r.resolution:<14}"
              f"{r.actual_fps:<8.1f}{lat:<12}{r.frames_received:<8}{r.disconnects:<5}")
    print("-" * len(header))
    print(f"  Total: {len(results)} cameras | "
          f"Passed: {sum(1 for r in results if r.pass_)} | "
          f"Failed: {sum(1 for r in results if not r.pass_)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="RTSP Camera Validation Harness")
    parser.add_argument("--url", help="Single RTSP URL to test")
    parser.add_argument("--camera-id", default="cam_01", help="Camera ID for single URL")
    parser.add_argument("--camera-file", help="JSON file with camera configs")
    parser.add_argument("--from-config", action="store_true",
                        help="Load cameras from config.CAMERAS")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Test duration per camera (seconds)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    configs = []
    if args.url:
        configs.append(CameraConfig(camera_id=args.camera_id, url=args.url))
    elif args.camera_file:
        configs = load_camera_file(args.camera_file)
    elif args.from_config:
        configs = load_from_config()
    else:
        parser.error("Provide --url, --camera-file, or --from-config")

    if not configs:
        print("No cameras to test.")
        sys.exit(1)

    print(f"Testing {len(configs)} camera(s) for {args.duration}s each ...")
    results = validate_all(configs, args.duration, verbose=args.verbose)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print_summary(results)

    all_pass = all(r.pass_ for r in results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
