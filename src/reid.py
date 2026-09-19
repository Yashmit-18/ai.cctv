"""Appearance-based person re-identification (Phase 44).

Modules
-------
* ``AppearanceExtractor`` -- turns a person crop into a fixed-size, L2
  normalised embedding.  Default is a **built-in** lightweight descriptor
  (RGB histogram + spatial colour grid + grayscale cellisation + edge/silhouette
  energy).  When ``CCTV_REID_MODEL_PATH`` points to a trained model the built-in
  is replaced:
    * ``*.pth`` / ``*.pt`` -- a **trained OSNet** person-ReID backbone (MIT,
      vendored ``src.reid_osnet``), inputs 256x128, output dimension 512 (the
      Release 49 genuine-ReID upgrade; see the Phase 49 report).
    * ``*.onnx`` -- an OSNet-style ONNX export using the same 256x128
      preprocessing; a missing/unloadable model always falls back to the
      built-in descriptor (never a crash).
* ``AppearanceRegistry`` -- enrolls employees from full-body appearance images
  (``data/reid/EMP001.jpg``, ..., falls back to the face-enrollment images in
  ``data/faces`` when the reid directory is empty) and answers cosine-similarity
  identity queries with an ambiguity guard.
* ``python -m src.reid`` -- offline rebuild + integrity report.

Design rules (Phase 44)
-----------------------
* **Secondary signal.**  Appearance identity is used ONLY to resolve a track
  that face recognition cannot identify.  It can never outrank a face identity
  and is never adopted from weak or ambiguous matches -- the tracker applies
  the adoption policy, this module only scores matches.
* **No biometrics stored.**  Only the mean L2-normalised embedding per employee
  is cached to ``data/appearance.pkl``.  Raw person images are never written by
  this module.
* **Deterministic & CPU-cheap.**  The built-in descriptor has no external model
  dependency, so enrolment + runtime work out of the box.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

from config import (APPEARANCE_EMBEDDINGS_FILE, FACES_DIR, REID_DIR,
                    REID_EMBEDDING_DIM, REID_IMGSZ, REID_MARGIN_MIN,
                    REID_MODEL_PATH, REID_STRONG_THRESHOLD)

logger = logging.getLogger("cctv.reid")

# Phase 54 M16 -- rate-limited per-crop embedding-failure logging.  A busy
# scene with an unloadable model used to spam a full traceback per crop; the
# first failure still logs, repeats are throttled to once per 30 s.
_EMB_FAILURES = 0
_EMB_LAST_ERROR_AT = 0.0
_EMB_ERROR_RATE_LIMIT_SEC = 30.0

CACHE_VERSION = 2
#: Filenames like ``EMP001.jpg``, ``EMP001_2.jpg``, ``EMP002_back.jpg``.
_EMPLOYEE_ID_RE = "([A-Za-z]{2,10}[0-9]{2,6})"


# ---------------------------------------------------------------------------
# Descriptor / embedding extraction
# ---------------------------------------------------------------------------


def _resize_crop(crop: np.ndarray, side: int) -> np.ndarray:
    """Resize a BGR crop to ``side``x``side`` handling non-square input."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("empty person crop")
    scale = side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    import cv2 as _cv2
    resized = _cv2.resize(crop, (nw, nh), interpolation=_cv2.INTER_AREA)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    x0, y0 = (side - nw) // 2, (side - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


class AppearanceExtractor:
    """Person-crop -> L2-normalised embedding with an optional trained model.

    Parameters
    ----------
    model_path : str
        A trained ReID model path: ``*.pth``/``*.pt`` (OSNet torch weights) or
        ``*.onnx`` (OSNet-style ONNX export).  Empty/"" or unloadable -> the
        built-in descriptor (never a crash).
    imgsz : int
        Crop side used by the built-in descriptor's grid statistics.
    dim : int
        Output embedding dimension for the built-in / ONNX paths (OSNet torch
        weights always emit their own trained dimension, 512).

    Attributes
    ----------
    feature_dim : int
        The actual embedding dimension this extractor emits (256 built-in /
        ONNX, 512 for a torch OSNet).  Used to validate the appearance cache.
    """

    def __init__(self, model_path: str = "", imgsz: int = 128,
                 dim: int = 256):
        self.imgsz = int(imgsz)
        self.dim = int(dim)
        self.feature_dim = int(dim)
        self._session = None
        self._osnet = None
        self.model_source = "built-in"
        #: Phase 51 model identity: exact checkpoint filename + inferred variant.
        self.checkpoint = None
        self.model_variant = None
        # Phase 54 M04 -- fallback transparency.  A requested model that cannot
        # be loaded is recorded instead of silently vanishing: the daemon and
        # dashboard can then show *why* ReID is running its built-in descriptor
        # (missing file, corrupt checkpoint, torch unavailable, ...).
        self.requested_model = str(model_path) if model_path else ""
        self.load_error: str | None = None
        if model_path and os.path.isfile(model_path):
            if str(model_path).lower().endswith((".pth", ".pt")):
                self._load_torch_osnet(model_path)
            else:
                self._load_onnx(model_path)

    def _load_torch_osnet(self, model_path: str):
        """Load the vendored OSNet torch backend (Release 49 genuine ReID)."""
        self.load_error = None
        try:
            from src.reid_osnet import OsnetEmbedder
            self._osnet = OsnetEmbedder(model_path)
            self.feature_dim = self._osnet.feature_dim
            self.model_source = "osnet"
            self.checkpoint = os.path.basename(str(model_path))
            self.model_variant = getattr(self._osnet, "variant", None)
            logger.info("AppearanceExtractor using OSNet model %s "
                        "(variant=%s feature_dim=%d)", model_path,
                        self.model_variant, self.feature_dim)
        except Exception as exc:  # pragma: no cover - env/model dependent
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("AppearanceExtractor could not load OSNet %s (%s) -- "
                           "REQUESTED MODEL IS NOT ACTIVE; using built-in "
                           "descriptor. requested=%s load_error=%s",
                           model_path, exc, self.requested_model,
                           self.load_error)
            self._osnet = None
            self.model_source = "built-in"
            self.checkpoint = None
            self.model_variant = None

    def _load_onnx(self, model_path: str):
        self.load_error = None
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
            out = self._session.get_outputs()[0]
            self._session_out = out.name
            self.model_source = "onnx"
            self.checkpoint = os.path.basename(str(model_path))
            self.model_variant = None
            logger.info("AppearanceExtractor using ONNX model %s",
                        model_path)
        except Exception as exc:  # pragma: no cover - env dependent
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("AppearanceExtractor could not load %s (%s) -- "
                           "REQUESTED MODEL IS NOT ACTIVE; using built-in "
                           "descriptor. requested=%s load_error=%s",
                           model_path, exc, self.requested_model,
                           self.load_error)
            self._session = None
            self.model_source = "built-in"
            self.checkpoint = None
            self.model_variant = None

    # -- public ------------------------------------------------------------

    def embed(self, crop_bgr: np.ndarray):
        """Return the L2-normalised embedding, or None on any failure."""
        try:
            if crop_bgr is None or crop_bgr.size == 0:
                return None
            if self._osnet is not None:
                emb = self._osnet.extract(crop_bgr)
            elif self._session is not None:
                emb = self._embed_onnx(crop_bgr)
            else:
                emb = self._embed_builtin(crop_bgr)
            if emb is None:
                return None
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            if emb.size == 0 or not np.all(np.isfinite(emb)):
                return None
            norm = float(np.linalg.norm(emb))
            if norm <= 1e-8:
                return None
            return (emb / norm).astype(np.float32)
        except Exception:
            global _EMB_FAILURES, _EMB_LAST_ERROR_AT
            _EMB_FAILURES += 1
            now = time.time()
            if _EMB_FAILURES == 1 or (now - _EMB_LAST_ERROR_AT) >= _EMB_ERROR_RATE_LIMIT_SEC:
                _EMB_LAST_ERROR_AT = now
                logger.exception(
                    "appearance embedding failed (%d total; check requested "
                    "model/load_error on the extractor when ReID runs the "
                    "built-in descriptor)", _EMB_FAILURES)
            return None

    # -- built-in ----------------------------------------------------------

    def _embed_builtin(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Cheap, license-free, deterministic appearance descriptor.

        Concatenated and padded to ``self.dim``:
          * per-channel 16-bin RGB histogram flags          -> 48
          * 4x4 spatial grid per-channel mean colour        -> 48
          * 8x8 grayscale cell mean                         -> 64
          * 4x4 edge-energy grid (Sobel magnitude)          -> 16
        Total 176 features; padded to ``dim`` with zeros then L2-normalised
        by the caller.
        """
        side = self.imgsz
        import cv2 as _cv2
        tile = _resize_crop(crop_bgr, side)
        rgb = _cv2.cvtColor(tile, _cv2.COLOR_BGR2RGB).astype(np.float32)
        gray = _cv2.cvtColor(tile, _cv2.COLOR_BGR2GRAY).astype(np.float32)

        feats: list[np.ndarray] = []
        for c in range(3):
            hist, _ = np.histogram(rgb[..., c], bins=16, range=(0, 256))
            feats.append(hist.astype(np.float32) / max(1.0, hist.sum()))

        g = side // 4
        for c in range(3):
            grid = np.zeros(16, dtype=np.float32)
            for i in range(4):
                for j in range(4):
                    cell = rgb[i * g:(i + 1) * g, j * g:(j + 1) * g, c]
                    grid[i * 4 + j] = float(cell.mean()) if cell.size else 0.0
            feats.append(grid / 255.0)

        cell = side // 8
        gray_cells = np.zeros(64, dtype=np.float32)
        for i in range(8):
            for j in range(8):
                blk = gray[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
                gray_cells[i * 8 + j] = float(blk.mean()) if blk.size else 0.0
        feats.append(gray_cells / 255.0)

        gx = _cv2.Sobel(gray, _cv2.CV_32F, 1, 0, ksize=3)
        gy = _cv2.Sobel(gray, _cv2.CV_32F, 0, 1, ksize=3)
        mag = _cv2.magnitude(gx, gy)
        eg = side // 4
        edge = np.zeros(16, dtype=np.float32)
        for i in range(4):
            for j in range(4):
                blk = mag[i * eg:(i + 1) * eg, j * eg:(j + 1) * eg]
                edge[i * 4 + j] = float(blk.mean()) if blk.size else 0.0
        feats.append(edge / (255.0 * 4.0))

        vec = np.concatenate(feats)
        if vec.size < self.dim:
            vec = np.concatenate([vec, np.zeros(self.dim - vec.size,
                                                dtype=np.float32)])
        elif vec.size > self.dim:
            vec = vec[:self.dim]
        return vec

    # -- onnx --------------------------------------------------------------

    def _embed_onnx(self, crop_bgr: np.ndarray) -> np.ndarray:
        """OSNet-style ONNX export: 256x128 + torchreid normalisation."""
        from src.reid_osnet import (OSNET_INPUT_H, OSNET_INPUT_W,
                                    preprocess_osnet)
        inp = preprocess_osnet(crop_bgr, OSNET_INPUT_H, OSNET_INPUT_W)
        out = self._session.run([self._session_out], {
            self._session.get_inputs()[0].name: inp})[0]
        emb = np.asarray(out).reshape(-1)
        if emb.size != self.feature_dim:
            emb = np.resize(emb, self.feature_dim)
        return emb


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _employee_id_from_name(stem: str) -> str | None:
    """Map ``EMP001``, ``EMP001_2``, ``EMP002_back`` -> ``EMP001``."""
    head = stem.split("_")[0]
    digits = [ch for ch in head if ch.isdigit()]
    letters = [ch for ch in head if ch.isalpha()]
    if not digits or not letters:
        return None
    return ("".join(letters) + "".join(digits)).upper()


class AppearanceRegistry:
    """Employee appearance gallery with cached mean embeddings.

    Parameters mirror the face registry so the class can be constructed the
    same way everywhere: relatives use ``__new__`` in tests and are driven
    with explicit attributes.
    """

    def __init__(self):
        self.reid_dir = str(REID_DIR)
        self.fallback_dir = str(FACES_DIR)
        self.embeddings_file = str(APPEARANCE_EMBEDDINGS_FILE)
        self.extractor = AppearanceExtractor(
            model_path=str(REID_MODEL_PATH), imgsz=int(REID_IMGSZ),
            dim=int(REID_EMBEDDING_DIM))
        #: Actual embedding dimension produced by the extractor (512 for a
        #: trained OSNet).  A stale cache with mismatched dims is invalidated.
        self.embed_dim = int(getattr(self.extractor, "feature_dim",
                                     REID_EMBEDDING_DIM))
        self.strong_threshold = float(REID_STRONG_THRESHOLD)
        self.margin_min = float(REID_MARGIN_MIN)
        #: {employee_id: {"embedding": np.ndarray, "images": int, "source": str}}
        self._by_employee: dict = {}
        self._loaded_from = None

    # -- lifecycle ---------------------------------------------------------

    def ensure_built(self) -> dict:
        """Load the cached gallery or build it from images (images win)."""
        try:
            if self._by_employee:
                return {"status": "ready", "loaded_from": self._loaded_from}
            self._load_cache()
            if self._by_employee:
                return {"status": "cache", "loaded_from": self._loaded_from}
        except FileNotFoundError:
            self._by_employee = {}
        except Exception:
            logger.exception("appearance cache unreadable; rebuilding")
            self._by_employee = {}
        return {"status": "rebuilt", "loaded_from": self._build_from_images()}

    def rebuild(self) -> dict:
        """Force a rebuild from enrollment images and persist it."""
        src = self._build_from_images()
        self._save_cache()
        return {"status": "rebuilt", "loaded_from": src}

    def _build_from_images(self) -> str:
        dirs = [p for p in (self.reid_dir, self.fallback_dir) if Path(p).is_dir()]
        source = self.reid_dir if Path(self.reid_dir).is_dir() else \
            self.fallback_dir
        found: dict[str, list[np.ndarray]] = {}
        seen_files: dict[str, int] = {}
        for directory in dirs:
            for path in sorted(
                    list(Path(directory).glob("*.jpg")) +
                    list(Path(directory).glob("*.png"))):
                emp = _employee_id_from_name(path.stem)
                if emp is None:
                    continue
                img = self._load_image(str(path))
                if img is None:
                    continue
                emb = self.extractor.embed(img)
                if emb is None:
                    continue
                key = (emp, str(path))
                if key in seen_files:
                    continue
                seen_files[key] = 1
                found.setdefault(emp, []).append(emb)
        self._by_employee = {}
        for emp, embs in found.items():
            mean = np.mean(np.stack(embs), axis=0).astype(np.float32)
            norm = float(np.linalg.norm(mean))
            self._by_employee[emp] = {
                "embedding": (mean / norm).astype(np.float32) if norm > 1e-8
                else mean,
                "images": len(embs),
                "source": source,
            }
        return source

    @staticmethod
    def _load_image(path: str):
        import cv2 as _cv2
        try:
            img = _cv2.imread(path)
            if img is None:
                return None
            return img
        except Exception:
            return None

    def _load_cache(self):
        with open(self.embeddings_file, "rb") as fh:
            data = pickle.load(fh)
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
            raise FileNotFoundError("stale appearance cache")
        entries = data.get("by_employee", {})
        self._by_employee = {}
        for emp, entry in entries.items():
            emb = np.asarray(entry["embedding"], dtype=np.float32).reshape(-1)
            if emb.size == self.embed_dim:
                self._by_employee[emp] = {
                    "embedding": emb,
                    "images": int(entry.get("images", 0)),
                    "source": str(entry.get("source", "cache")),
                }
        if not self._by_employee:
            raise FileNotFoundError("empty appearance cache")
        self._loaded_from = "cache"

    def _save_cache(self):
        Path(self.embeddings_file).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{self.embeddings_file}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({
                "version": CACHE_VERSION,
                "embed_dim": self.embed_dim,
                "by_employee": self._by_employee,
            }, fh, protocol=4)
        os.replace(tmp, self.embeddings_file)

    # -- queries -----------------------------------------------------------

    def count_enrolled(self) -> int:
        return len(self._by_employee)

    def employee_status(self) -> dict:
        return {emp: "ENROLLED" for emp in sorted(self._by_employee)}

    def gallery(self) -> dict:
        """Embeddings keyed by employee (used by tests)."""
        return {
            emp: entry["embedding"]
            for emp, entry in self._by_employee.items()
        }

    def identify(self, emb) -> list:
        """Score one embedding against the gallery.

        Returns
        -------
        (employee_id, score, margin, strong, second_id)
          * ``employee_id`` -- ``"Unknown"`` when nobody reaches the strong
            bar (threshold + margin) or the gallery is empty.
          * ``score``       -- cosine similarity of the best match.
          * ``margin``      -- ``best - second_best`` similarity.
          * ``strong``      -- the tracker may only act on strong matches.
        """
        if emb is None or not self._by_employee:
            return ["Unknown", 0.0, 0.0, False, None]
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        nemb = float(np.linalg.norm(emb))
        if nemb <= 1e-8:
            return ["Unknown", 0.0, 0.0, False, None]
        emb = emb / nemb
        scored: list[tuple[str, float]] = []
        for emp, entry in self._by_employee.items():
            g = np.asarray(entry["embedding"], dtype=np.float32).reshape(-1)
            if g.size != emb.size:
                continue
            ng = float(np.linalg.norm(g))
            if ng <= 1e-8:
                continue
            sim = float(np.dot(emb, g / ng))
            scored.append((emp, sim))
        if not scored:
            return ["Unknown", 0.0, 0.0, False, None]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        top, best = scored[0]
        second = scored[1][1] if len(scored) > 1 else 0.0
        margin = best - second
        strong = best >= self.strong_threshold and margin >= self.margin_min
        second_id = scored[1][0] if len(scored) > 1 else None
        if not strong:
            return [top, best, margin, False, second_id]
        return [top, best, margin, True, second_id]


def rebuild_and_report() -> dict:
    """Offline helper used by ``python -m src.reid`` and the Launcher."""
    registry = AppearanceRegistry()
    result = registry.rebuild()
    return {
        "status": result["status"],
        "loaded_from": result["loaded_from"],
        "employees": registry.employee_status(),
        "enrolled": registry.count_enrolled(),
        "embed_dim": registry.embed_dim,
        "extractor": registry.extractor.model_source,
    }


if __name__ == "__main__":
    import sys
    report = rebuild_and_report()
    print("Appearance registry rebuilt:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    sys.exit(0 if report.get("enrolled", 0) else 1)