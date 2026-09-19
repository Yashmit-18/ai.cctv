"""Phase 49 benchmark: genuine ReID candidates vs built-in descriptor.

Deterministic offline benchmark on SYNTHETIC crops.  Every numeric result is
labelled SIMULATED (no real camera feed; thresholds/accuracy never validated on
real CCTV imagery).  Measures:

  * init latency + memory-relevant stats (param count) per candidate,
  * per-embed CPU latency (mean/min/max over N runs),
  * embedding dimension + model source actually selected by
    ``AppearanceExtractor``,
  * same-person stability (self-cosine incl. scale jitter) and
    different-person separation (cosine of distinct synthetic outfits).

Candidates: built-in descriptor, OSNet-x1_0, OSNet-x0_5 (lightweight),
OSNet-AIN-x1_0 (attention-IN) -- all MSMT17-trained, MIT-licensed weights in
``models/``.  Missing models are recorded as NOT VALIDATED, never fabricated.

Writes structured results to ``data/phase49_benchmark.json``.
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
OUT = ROOT / "data" / "phase49_benchmark.json"

MODEL_DIR = ROOT / "models"
CANDIDATES = {
    "osnet_x1_0": MODEL_DIR / "osnet_x1_0_msmt17.pth",
    "osnet_x0_5": MODEL_DIR / "osnet_x0_5_msmt17.pth",
    "osnet_ain_x1_0": MODEL_DIR / "osnet_ain_x1_0_msmt17.pth",
}

# ---------------------------------------------------------------------------
# Synthetic person crops (deterministic; no external assets)
# ---------------------------------------------------------------------------


def _synthetic_person_crop(seed: int, shirt_bgr=tuple,
                           scale: float = 1.0) -> np.ndarray:
    """Deterministic 256x128 person-like crop: darker background + clothing."""
    rng = np.random.default_rng(seed)
    crop = rng.integers(15, 45, (256, 128, 3), dtype=np.uint8)
    h, w, _ = crop.shape
    sh = int(round(h * 0.55 * scale)); sw = int(round(w * 0.9 * scale))
    y0 = (h - sh) // 2; x0 = (w - sw) // 2
    crop[y0:y0 + sh, x0:x0 + sw] = shirt_bgr
    crop[y0 + sh:y0 + sh + 6, max(0, x0):min(w, x0 + sw)] = (30, 30, 30)
    return np.ascontiguousarray(crop)


def _jitter(crop: np.ndarray, seed: int, amp: int = 6) -> np.ndarray:
    """Small deterministic photometric + spatial jitter (scale ~1.0 +- few px)."""
    rng = np.random.default_rng(seed)
    dx = int(rng.integers(-amp, amp + 1)); dy = int(rng.integers(-amp, amp + 1))
    m = np.zeros_like(crop)
    ry = slice(max(0, dy), min(crop.shape[0], crop.shape[0] + dy))
    rx = slice(max(0, dx), min(crop.shape[1], crop.shape[1] + dx))
    m[ry, rx] = crop[ry, rx]
    bright = rng.integers(-12, 13, size=3, dtype=np.int16)
    m = np.clip(m.astype(np.int16) + bright, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(m)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _bench(fn, n: int = 5) -> dict:
    """Run fn (deterministic) n times, return ms stats."""
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


def _model_info(path: Path) -> dict:
    if not path or not path.is_file():
        return {"exists": False, "size_mb": None}
    return {"exists": True, "size_mb": round(path.stat().st_size / 1e6, 2)}


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    results: dict = {
        "label": "CODE VERIFIED / SIMULATED -- synthetic person crops, CPU; "
                 "no real camera, thresholds NOT VALIDATED on real data",
        "date": "2026-09-19",
        "hardware": "CPU (local)",
        "candidates": {},
        "stability": {},
        "separation": {},
    }

    # ---- candidate presence --------------------------------------------
    for name, path in CANDIDATES.items():
        results["candidates"][name] = {"model": _model_info(path)}

    from src.reid import AppearanceExtractor

    # ---- built-in descriptor (production default) ----------------------
    crop_a = _synthetic_person_crop(seed=1, shirt_bgr=(90, 90, 200))
    try:
        be = AppearanceExtractor(model_path="", imgsz=128, dim=256)
        results["candidates"]["builtin_descriptor"] = {
            "source": be.model_source,
            "dim": int(be.feature_dim),
            "embed": _bench(lambda: be.embed(crop_a), n=10),
        }
    except Exception as exc:
        results["candidates"]["builtin_descriptor"] = {"error": str(exc)}

    # ---- OSNet candidates ----------------------------------------------
    for name, path in CANDIDATES.items():
        entry = {"source": "osnet", "dim": None, "params": None,
                 "init_ms": None, "embed": None, "status": "SIMULATED"}
        if not path.is_file():
            entry["status"] = "NOT VALIDATED -- model file not present locally"
            results["candidates"][name] = entry
            continue
        try:
            t0 = time.perf_counter()
            ex = AppearanceExtractor(model_path=str(path))
            entry["init_ms"] = round(
                (time.perf_counter() - t0) * 1000.0, 2)
            entry["source"] = ex.model_source
            entry["dim"] = int(ex.feature_dim)
            try:
                from torch import nn
                # param count via the extractor's loaded model avoids re-import
                model = getattr(ex, "_osnet", None).model
                entry["params"] = int(
                    sum(p.numel() for p in model.parameters()))
                _ = nn  # keep import referenced for pyflakes symmetry
            except Exception as exc:
                entry["params"] = f"unavailable: {exc}"
            entry["embed"] = _bench(lambda: ex.embed(crop_a), n=10)
        except Exception as exc:
            entry["status"] = f"error: {exc}"
        results["candidates"][name] = entry

    # ---- same-person stability + different-person separation -----------
    outfits = {
        "red_shirt": (60, 60, 210), "blue_shirt": (200, 90, 40),
        "green_shirt": (60, 170, 60), "dark_pants_1": (35, 35, 35),
        "dark_pants_2": (30, 30, 30),
    }
    for name, path in CANDIDATES.items():
        if not path.is_file():
            continue
        try:
            ex = AppearanceExtractor(model_path=str(path))
        except Exception:
            continue
        key = f"osnet:{name}"
        sim: dict = {}
        for cname, colour in outfits.items():
            a = _synthetic_person_crop(seed=3, shirt_bgr=colour)
            b = _jitter(a, seed=7)
            c = _synthetic_person_crop(
                seed=4, shirt_bgr=colour, scale=1.04)
            ea = ex.embed(a); eb = ex.embed(b); ec = ex.embed(c)
            if ea is not None and eb is not None:
                sim[f"{cname}_clean_jitter_cos"] = round(_cos(ea, eb), 4)
            if ea is not None and ec is not None:
                sim[f"{cname}_scale_jitter_cos"] = round(_cos(ea, ec), 4)
        results["stability"][key] = sim
        # different-person: distinct outfit colours, clean vs clean
        sep: dict = {}
        colours = list(outfits.values())
        for i in range(len(colours)):
            for j in range(i + 1, len(colours)):
                a = _synthetic_person_crop(seed=5, shirt_bgr=colours[i])
                b = _synthetic_person_crop(seed=6, shirt_bgr=colours[j])
                ea = ex.embed(a); eb = ex.embed(b)
                if ea is not None and eb is not None:
                    sep[f"{i}_{j}_cos"] = round(_cos(ea, eb), 4)
        results["separation"][key] = sep

    # Built-in descriptor separation for context
    try:
        be = AppearanceExtractor(model_path="", imgsz=128, dim=256)
        sep = {}
        colours = list(outfits.values())
        for i in range(len(colours)):
            for j in range(i + 1, len(colours)):
                a = _synthetic_person_crop(seed=5, shirt_bgr=colours[i])
                b = _synthetic_person_crop(seed=6, shirt_bgr=colours[j])
                ea = be.embed(a); eb = be.embed(b)
                if ea is not None and eb is not None:
                    sep[f"{i}_{j}_cos"] = round(_cos(ea, eb), 4)
        results["separation"]["builtin_descriptor"] = sep
        sim = {}
        for cname, colour in outfits.items():
            a = _synthetic_person_crop(seed=3, shirt_bgr=colour)
            b = _jitter(a, seed=7)
            ea = be.embed(a); eb = be.embed(b)
            if ea is not None and eb is not None:
                sim[f"{cname}_clean_jitter_cos"] = round(_cos(ea, eb), 4)
        results["stability"]["builtin_descriptor"] = sim
    except Exception:
        pass

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()