"""Phase 46 benchmark: current vs candidate phone/ReID models (Part 3/10).

Deterministic local benchmark using synthetic frames.  Labels every result
CODE VERIFIED / SIMULATED / MODEL LIMITATION / NOT VALIDATED -- never claims
real CCTV accuracy without a real camera feed.

Writes structured results to data/phase46_benchmark.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "phase46_benchmark.json"

# ---------------------------------------------------------------------------
# Synthetic frame builders (deterministic, no external assets)
# ---------------------------------------------------------------------------


def _synthetic_frame(w: int = 640, h: int = 480, seed: int = 42) -> np.ndarray:
    """Deterministic synthetic frame with a person + small phone blob."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(20, 60, (h, w, 3), dtype=np.uint8)
    # Person: large bright rectangle (upper body)
    frame[80:400, 200:440] = (120, 130, 140)
    # Phone: small bright blob near the person's hand
    frame[300:340, 380:420] = (200, 220, 240)
    return frame


def _synthetic_person_crop(seed: int = 1) -> np.ndarray:
    """Deterministic person crop for ReID embedding benchmark."""
    rng = np.random.default_rng(seed)
    crop = rng.integers(20, 60, (256, 128, 3), dtype=np.uint8)
    # Upper body (shirt colour)
    crop[0:160, :] = (90, 90, 200)
    # Lower body (pants colour)
    crop[160:, :] = (40, 40, 40)
    return crop


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _bench(fn, n: int = 5) -> dict:
    """Run fn n times, return ms stats.  fn must be deterministic."""
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "runs": n,
        "mean_ms": round(float(np.mean(times)), 2),
        "min_ms": round(float(np.min(times)), 2),
        "max_ms": round(float(np.max(times)), 2),
    }


def _model_info(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {"exists": False, "size_mb": None}
    return {"exists": True, "size_mb": round(os.path.getsize(path) / 1e6, 2)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    results: dict = {
        "label": "CODE VERIFIED / SIMULATED -- synthetic frames, no real camera",
        "date": "2026-09-12",
        "hardware": "CPU (local)",
        "phone": {},
        "reid": {},
    }

    # ---- Phone: current baseline YOLOv8n @640 (person+phone) -------------
    y8_path = str(ROOT / "models" / "yolov8n.pt")
    y11_path = str(ROOT / "models" / "yolo11n.pt")
    results["phone"]["yolov8n_baseline"] = _model_info(y8_path)
    results["phone"]["yolo11n_candidate"] = _model_info(y11_path)

    frame = _synthetic_frame()

    # YOLOv8n @640 (person+phone, every cycle)
    try:
        from ultralytics import YOLO
        y8 = YOLO(y8_path)
        # warmup
        y8.predict(frame, conf=0.4, imgsz=640, device="cpu", verbose=False)
        results["phone"]["yolov8n_640"] = _bench(
            lambda: y8.predict(frame, conf=0.4, imgsz=640, device="cpu",
                               verbose=False))
        # phone-only @1280 (dedicated phone pass)
        results["phone"]["yolov8n_1280_phone"] = _bench(
            lambda: y8.predict(frame, conf=0.4, classes=[67], imgsz=1280,
                               device="cpu", verbose=False))
    except Exception as exc:
        results["phone"]["yolov8n_error"] = str(exc)

    # YOLO11n @640 (person+phone)
    try:
        y11 = YOLO(y11_path)
        y11.predict(frame, conf=0.4, imgsz=640, device="cpu", verbose=False)
        results["phone"]["yolo11n_640"] = _bench(
            lambda: y11.predict(frame, conf=0.4, imgsz=640, device="cpu",
                                verbose=False))
        # phone-only @1280
        results["phone"]["yolo11n_1280_phone"] = _bench(
            lambda: y11.predict(frame, conf=0.4, classes=[67], imgsz=1280,
                                device="cpu", verbose=False))
    except Exception as exc:
        results["phone"]["yolo11n_error"] = str(exc)

    # ---- ReID: built-in descriptor vs (no ONNX model available) ----------
    from src.reid import AppearanceExtractor

    crop = _synthetic_person_crop()
    extractor = AppearanceExtractor(model_path="", imgsz=128, dim=256)
    results["reid"]["builtin_descriptor"] = {
        "source": extractor.model_source,
        "dim": 256,
        "imgsz": 128,
    }
    results["reid"]["builtin_embed_ms"] = _bench(lambda: extractor.embed(crop))

    # OSNet ONNX: not present locally -> document as NOT VALIDATED
    osnet_path = str(ROOT / "models" / "osnet_x1_0.onnx")
    results["reid"]["osnet_onnx"] = {
        **_model_info(osnet_path),
        "status": "NOT VALIDATED -- model file not present locally",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()