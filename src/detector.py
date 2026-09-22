"""Dynamic face-based detection engine with YOLO phone detection (Phase 6).

Features
--------
* Runs insightface to detect faces and extract embeddings per frame.
* Matches each detected face against the FaceRegistry (cosine similarity).
* Runs YOLO to detect cell phones and checks proximity to identified faces.
* Auto-detects CUDA / MPS / CPU and enables half-precision on GPU.
* No longer depends on static desk ROIs.
"""

import logging
import os
import time
from statistics import mean as _mean

import cv2
import numpy as np
from ultralytics import YOLO

import config  # noqa: E402  (Phase 63: live phone-pass knob reads)
from config import (
    CONF_THRESHOLD,
    DETECTION_DEBUG,
    FACE_DEBUG,
    LIVE_MIN_FACE_AREA,
    MODEL_PATH,
    PHONE_CLASS as CONFIG_PHONE_CLASS,
    PHONE_DETECT_CADENCE,
    PHONE_DIAG,
    PHONE_IMGSZ,
    PHONE_MODEL_PATH,
    PIPELINE_TIMING,
    REID_ENABLED,
    REID_CADENCE,
    REID_GATE_IOU,
)
from src.face_registry import (CANDIDATE_THRESHOLD, FaceRegistry, RECOG_CONFIRMED,
                               RECOG_CANDIDATE, RECOG_UNKNOWN)
from src.phone_detector import PhoneDetector
from src.reid import AppearanceRegistry

logger = logging.getLogger("cctv.detector")

PHONE_CLASS = 67
PERSON_CLASS = 0
# Classes we detect.  Person (0) improves presence handling so a visibly
# present-but-unidentifiable employee is not silently marked AWAY.
_DETECT_CLASSES = [PERSON_CLASS, PHONE_CLASS]
_FACE_EXPAND_RATIO = 1.5
_PHONE_PROXIMITY_RATIO = 0.4


# ------------------------------------------------------------------
# Device helpers
# ------------------------------------------------------------------

def _resolve_device(requested: str | None = None) -> str:
    """Return the best available inference device string.

    Priority:
      1. Explicitly requested device (``"cuda"``, ``"mps"``, ``"cpu"``).
      2. CUDA if ``torch.cuda.is_available()``.
      3. MPS if ``torch.backends.mps.is_available()`` (Apple Silicon).
      4. CPU fallback.
    """
    if requested:
        dev = requested.lower().strip()
        if dev in ("cuda", "gpu"):
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            return "cpu"
        if dev == "mps":
            try:
                import torch
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            logger.warning("MPS requested but unavailable; falling back to CPU.")
            return "cpu"
        return dev

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("CUDA device detected: %s", name)
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS device detected.")
            return "mps"
    except ImportError:
        pass

    logger.info("Using CPU for inference.")
    return "cpu"


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------

def _boxes_overlap(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _box_center(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _expand_box(box, frame_shape, ratio: float = 1.5) -> tuple[int, int, int, int]:
    """Expand a bounding box by *ratio* around its centre, clamped to frame."""
    h, w = frame_shape[:2]
    cx, cy = _box_center(box)
    bw = (box[2] - box[0]) * ratio
    bh = (box[3] - box[1]) * ratio
    x1 = max(0, int(cx - bw / 2))
    y1 = max(0, int(cy - bh / 2))
    x2 = min(w, int(cx + bw / 2))
    y2 = min(h, int(cy + bh / 2))
    return (x1, y1, x2, y2)


def _phone_near_face(face_box, phone_boxes, frame_shape) -> bool:
    """Check if any phone box overlaps or is near the expanded face box."""
    expanded = _expand_box(face_box, frame_shape, _FACE_EXPAND_RATIO)
    for phone in phone_boxes:
        if _boxes_overlap(expanded, phone):
            return True
        # Also check by centre-distance heuristic
        fx, fy = _box_center(expanded)
        px, py = _box_center(phone)
        face_w = expanded[2] - expanded[0]
        face_h = expanded[3] - expanded[1]
        dist = np.hypot(fx - px, fy - py)
        threshold = max(face_w, face_h) * _PHONE_PROXIMITY_RATIO
        if dist < threshold:
            return True
    return False


def _phone_near_person(person_box, phone_boxes, frame_shape,
                       ratio: float = _FACE_EXPAND_RATIO) -> tuple | None:
    """Return the first phone box overlapping/near the (expanded) person box.

    Unlike :func:`_phone_near_face` this associates the phone with the
    *person* directly (Phase A A7), so an unidentified person holding a phone
    is still attributed -- the result flows to the spatial track, not only to
    recognised employees.
    """
    if not phone_boxes:
        return None
    expanded = _expand_box(person_box, frame_shape, ratio)
    for phone in phone_boxes:
        if _boxes_overlap(expanded, phone):
            return tuple(phone)
        px, py = _box_center(phone)
        ex, ey = _box_center(expanded)
        dist = np.hypot(ex - px, ey - py)
        threshold = max(expanded[2] - expanded[0], expanded[3] - expanded[1]) * \
            _PHONE_PROXIMITY_RATIO
        if dist < threshold:
            return tuple(phone)
    return None


def _iou(a, b) -> float:
    """Intersection-over-union of two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _merge_phone_dets(base_phone_dets: list, enhanced: list) -> list:
    """Unify two ``[(box, conf)]`` phone lists, keeping enhanced duplicates out.

    The base (stock 640) pass and the Phase 44 high-resolution pass can both
    box the same phone; merging with an IoU gate prevents double counting
    while still adding phones only the high-resolution pass could see.
    """
    merged = list(base_phone_dets)
    base_boxes = [b for b, _ in base_phone_dets]
    for ebox, econ in enhanced:
        if any(_iou(base_box, ebox) > 0.3 for base_box in base_boxes):
            continue
        merged.append((ebox, econ))
    return merged


def _containing_person(face_box, person_boxes, frame_shape) -> tuple | None:
    """Associate a face with the nearest person box that plausibly contains it.

    Used to pair a recognised face with its person detection (Phase A A1), so
    downstream layers (zones, spatial tracker) reason about *persons* rather
    than faces alone.  The face CENTRE must fall inside the person's (1.5x
    expanded) box; when several persons satisfy containment the one with the
    closest centroid wins -- never first-in-list -- so a face in a crowd is
    attributed to its own person.
    """
    if not person_boxes:
        return None
    h, w = frame_shape[:2]
    fx, fy = _box_center(face_box)
    best = None
    best_d = None
    for pbox in person_boxes:
        cx, cy = _box_center(pbox)
        d = np.hypot(fx - cx, fy - cy)
        if best_d is not None and d >= best_d:
            continue
        exp = _expand_box(pbox, (h, w), _FACE_EXPAND_RATIO)
        inside = (pbox[0] <= fx <= pbox[2] and pbox[1] <= fy <= pbox[3]) or \
                 (exp[0] <= fx <= exp[2] and exp[1] <= fy <= exp[3])
        if inside:
            best, best_d = pbox, d
    return tuple(best) if best else None


# ------------------------------------------------------------------
# Main detector class
# ------------------------------------------------------------------

class ActivityDetector:
    """Face-based detection engine using insightface + YOLO.

    Parameters
    ----------
    model_path : str
        Path to the YOLO weights file.
    conf : float
        Minimum confidence threshold for YOLO detections.
    device : str | None
        Inference device (``"cuda"``, ``"mps"``, ``"cpu"``).  ``None`` for
        auto-detection.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        conf: float = CONF_THRESHOLD,
        device: str | None = None,
        face_detect_size: int | None = None,
        face_cadence: int = 1,
    ):
        self.conf = conf
        self._device = _resolve_device(device)
        self._half = self._device == "cuda"
        # Part G: run InsightFace once every ``face_cadence`` cycles per camera
        # while YOLO person+phone still runs every cycle.  1 = legacy behaviour
        # (faces every frame).  Never < 1.
        self._face_cadence = max(1, int(face_cadence if face_cadence else 1))
        self._face_ticks: dict[str, int] = {}
        # Part C: opt-in per-camera phone/person association measurements.
        self._phone_diag_on = bool(PHONE_DIAG)
        self._phone_stats: dict[str, dict] = {}
        self._diag_cycles = 0

        # Phase 45: opt-in per-cycle stage timings so ``main.py`` can publish
        # an itemized attribution of ``detect_ms`` (YOLO person / phone pass /
        # face / appearance ReID) to the dashboard.  Zero cost when
        # ``CCTV_PIPELINE_TIMING=0`` (the default); ``__new__``-constructed
        # test stubs simply report no timing.
        self._perf_on = bool(PIPELINE_TIMING)
        self._perf_raw: dict[str, float] = {
            "person_yolo_ms": 0.0, "phone_ms": 0.0,
            "face_ms": 0.0, "reid_ms": 0.0,
        }

        logger.info(
            "Loading YOLO model from %s  (device=%s, half=%s) ...",
            model_path, self._device, self._half,
        )
        # Phase 54 M03 -- fail-clear, never auto-download.  ``YOLO()`` would
        # silently pull missing weights from the network, breaking air-gapped
        # deployments and turning a missing model into an unbounded download
        # instead of a fast, honest startup failure.  A configured/derived
        # model path must already exist on disk:  vendor `models/*.pt` before
        # deploying (preflight models check FAILs when it is absent), or the
        # daemon refuses to start.
        if not model_path or not os.path.isfile(str(model_path)):
            if model_path:
                logger.error(
                    "YOLO weights not found at %r; refusing to auto-download. "
                    "Vendor the model file (see models/) or re-run "
                    "`python -m src.preflight` after obtaining weights.",
                    model_path)
            else:
                logger.error(
                    "No YOLO model path configured (MODEL_PATH empty); "
                    "refusing to start without weights.")
            raise RuntimeError(
                f"YOLO model unavailable at {model_path!r} (no auto-download; "
                f"vendor weights or fix CCTV_MODEL path / config.MODEL_PATH)")
        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            logger.error("YOLO model found but failed to load: %s", model_path)
            logger.error("Check the file is a valid, non-corrupt weights file "
                         "(re-vendor it if needed) and that ultralytics can "
                         "open it on this machine.")
            raise RuntimeError(f"YOLO model unavailable at {model_path!r}") from exc
        logger.info("YOLO model loaded successfully.")

        self._face_registry = FaceRegistry(
            detect_size=face_detect_size if face_detect_size else 640
        )

        # --- Phase 44: dedicated phone pass + appearance ReID ---
        # The phone pass reuses the already-loaded YOLO model (same weights,
        # phone-only class filter) unless the operator supplies a dedicated
        # phone model via CCTV_PHONE_MODEL_PATH.  It runs on its own
        # per-camera cadence at a higher input resolution (PHONE_IMGSZ) so
        # small phones finally get boxes the stock 640 pass misses.
        #
        # Phase 63: the pass knobs are read LIVE from the ``config`` module
        # (not bound at import) so an Admin Control Center override applied
        # by ``SettingsStore.apply_to_config()`` before construction is
        # honoured; untouched, they resolve to the env defaults below.
        _phone_class = int(getattr(config, "PHONE_CLASS", CONFIG_PHONE_CLASS)
                           or CONFIG_PHONE_CLASS)
        _phone_model_path = str(getattr(config, "PHONE_MODEL_PATH",
                                        PHONE_MODEL_PATH) or PHONE_MODEL_PATH)
        _phone_imgsz = int(getattr(config, "PHONE_IMGSZ", PHONE_IMGSZ)
                           or PHONE_IMGSZ)
        _phone_cadence = max(1, int(getattr(config, "PHONE_DETECT_CADENCE",
                                            PHONE_DETECT_CADENCE)
                                    or PHONE_DETECT_CADENCE))
        self._phone_detector = PhoneDetector(
            model=self.model,
            model_path=_phone_model_path,
            phone_class=_phone_class,
            conf=self.conf,
            imgsz=_phone_imgsz,
            device=self._device,
            half=self._half,
        )
        self._phone_cadence = _phone_cadence
        self._phone_ticks: dict[str, int] = {}
        # Appearance ReID is a secondary identity signal: it may only resolve
        # tracks face recognition cannot identify (enforced in tracker.py).
        self._reid_enabled = bool(REID_ENABLED)
        self._reid_registry = AppearanceRegistry()
        if self._reid_enabled:
            try:
                self._reid_registry.ensure_built()
            except Exception:  # never break the pipeline over appearance ReID
                logger.exception("appearance registry failed to load; reid off")
                self._reid_enabled = False
        self._reid_cadence = max(1, int(REID_CADENCE))
        self._reid_ticks: dict[str, int] = {}

    @property
    def device(self) -> str:
        return self._device

    @property
    def face_registry(self) -> FaceRegistry:
        return self._face_registry

    def detect(self, frame) -> dict:
        """Run face recognition + phone detection on a single frame.

        Returns
        -------
        dict with keys:
          ``"employees"``  -- {emp_id: {"present": bool, "phone_detected": bool}}
          ``"face_boxes"`` -- list of (x1,y1,x2,y2) for detected faces
          ``"face_labels"`` -- list of (emp_id, score) parallel to face_boxes
          ``"phone_boxes"`` -- list of (x1,y1,x2,y2) for detected phones
        """
        person_status, face_boxes, face_labels, phone_boxes, person_present = \
            self._process_frame(frame)
        return {
            "employees": person_status,
            "face_boxes": face_boxes,
            "face_labels": face_labels,
            "phone_boxes": phone_boxes,
            "person_present": person_present,
        }

    def detect_batch(self, frames: dict[str, object],
                     reid_gate_boxes: dict[str, list] | None = None) -> list[dict]:
        """Run batched inference across multiple camera frames (Phase 7).

        Parameters
        ----------
        frames : dict[str, object]
            ``{cam_id: frame}`` -- only cameras with a non-``None`` frame
            should be included.  Order is preserved.
        reid_gate_boxes : dict[str, list] | None
            OPT-IN (``CCTV_REID_GATE_RESOLVED=1``): ``{cam_id: [bbox, ...]}``
            person boxes resolved to face/trusted identities in the previous
            cycle, so the appearance pass skips them.  ``None``/missing =
            the appearance pass is never gated (production default).

        Returns
        -------
        list[dict]
            Flattened list of detections mapped by camera, one entry per
            identified employee per camera::

                [
                  {"cam": "cam_01", "emp_id": "EMP001", "phone": True},
                  {"cam": "cam_04", "emp_id": "EMP005", "phone": False},
                  ...
                ]

            Unrecognised faces are reported with ``emp_id == "Unknown"``.
        """
        if not frames:
            return []

        cam_ids = list(frames.keys())
        frame_list = [frames[cid] for cid in cam_ids]

        # Phase 45: reset the per-cycle stage timers (opt-in).
        perf = getattr(self, "_perf_on", False)
        if perf:
            raw = getattr(self, "_perf_raw", None) or {}
            for _k in ("person_yolo_ms", "phone_ms", "face_ms", "reid_ms"):
                raw[_k] = 0.0
            self._perf_raw = raw

        # --- Batched YOLO inference: one GPU call across all frames ---
        predict_kwargs = dict(
            conf=self.conf,
            classes=_DETECT_CLASSES,
            device=self._device,
            verbose=False,
        )
        if self._half:
            # Only request half precision when it is actually on; passing
            # half=False here triggers an Ultralytics deprecation warning on
            # every frame and floods the daemon log on CPU-only hosts.
            predict_kwargs["half"] = True
        _t_yolo = time.time() if perf else None
        results = self.model.predict(frame_list, **predict_kwargs)
        if _t_yolo is not None:
            self._perf_raw["person_yolo_ms"] += \
                (time.time() - _t_yolo) * 1000.0

        out: list[dict] = []
        self._diag_cycles = getattr(self, "_diag_cycles", 0) + 1
        for idx, cam_id in enumerate(cam_ids):
            frame = frames[cam_id]
            phone_dets = _boxes_and_conf_of_class(results[idx], PHONE_CLASS)
            person_dets = _boxes_and_conf_of_class(results[idx], PERSON_CLASS)
            # Phase 44: enhance phone recall with a cadence-gated, higher
            # resolution phone pass.  Dedupe against the base pass so a phone
            # is never counted twice, then flow the merged set downstream.
            phone_dets = self._enhance_phone_dets(frame, cam_id, phone_dets)
            # Phase 44: conservative appearance identity for unresolved tracks.
            reid_results = self._reid_for_camera(
                frame, cam_id, person_dets,
                gate_boxes=None if not reid_gate_boxes
                else reid_gate_boxes.get(cam_id)) \
                if person_dets else None
            # Part G: face recognition runs on a per-camera cadence; YOLO
            # person/phone detection always runs.  Person presence therefore
            # never depends on this cycle's face pass.
            run_faces = self._face_tick(cam_id)
            if getattr(self, "_phone_diag_on", False):
                self._record_phone_diag(cam_id, phone_dets, person_dets,
                                        frame.shape[:2], run_faces)
            out.extend(self._process_frame(
                frame,
                [b for b, _ in phone_dets],
                cam_id,
                person_boxes=[b for b, _ in person_dets],
                person_confs=[c for _, c in person_dets],
                run_faces=run_faces,
                reid_results=reid_results,
            ))
        return out

    def _face_tick(self, cam_id: str) -> bool:
        """Increment the per-camera face cadence counter; True -> run faces.

        Defensive against ``__new__``-constructed instances (tests): any
        missing attribute falls back to "faces every cycle".
        """
        cadence = getattr(self, "_face_cadence", 1) or 1
        if cadence < 1:
            cadence = 1
        ticks = getattr(self, "_face_ticks", None)
        if ticks is None:
            ticks = {}
            self._face_ticks = ticks
        n = ticks.get(cam_id, 0) + 1
        ticks[cam_id] = n
        return (n - 1) % cadence == 0

    # ------------------------------------------------------------------
    # Phase 44 -- phone enhancement pass
    # ------------------------------------------------------------------

    def _phone_tick(self, cam_id: str) -> bool:
        """Increment the per-camera phone-pass cadence counter (run or skip)."""
        cadence = getattr(self, "_phone_cadence", 1) or 1
        if cadence < 1:
            cadence = 1
        ticks = getattr(self, "_phone_ticks", None)
        if ticks is None:
            ticks = {}
            self._phone_ticks = ticks
        n = ticks.get(cam_id, 0) + 1
        ticks[cam_id] = n
        return (n - 1) % cadence == 0

    def _enhance_phone_dets(self, frame, cam_id: str,
                            base_phone_dets: list) -> list:
        """Run the dedicated phone pass on cadence and merge with the base set.

        Returns the base set untouched when the pass is unavailable, on a
        cadence skip, or when ``__new__``-constructed test stubs lack the
        Phase 44 attributes.
        """
        detector = getattr(self, "_phone_detector", None)
        if detector is None or not detector.ready:
            return list(base_phone_dets)
        if not self._phone_tick(cam_id):
            return list(base_phone_dets)
        try:
            _t = time.time() if getattr(self, "_perf_on", False) else None
            enhanced = detector.detect(frame)
            if _t is not None:
                self._perf_raw["phone_ms"] += (time.time() - _t) * 1000.0
        except Exception:
            logger.exception("dedicated phone pass failed; using base dets")
            return list(base_phone_dets)
        if not enhanced:
            return list(base_phone_dets)
        return _merge_phone_dets(base_phone_dets, enhanced)

    # ------------------------------------------------------------------
    # Phase 44 -- appearance ReID pass
    # ------------------------------------------------------------------

    def _reid_tick(self, cam_id: str) -> bool:
        """Increment the per-camera appearance cadence counter (run or skip)."""
        cadence = getattr(self, "_reid_cadence", 1) or 1
        if cadence < 1:
            cadence = 1
        ticks = getattr(self, "_reid_ticks", None)
        if ticks is None:
            ticks = {}
            self._reid_ticks = ticks
        n = ticks.get(cam_id, 0) + 1
        ticks[cam_id] = n
        return (n - 1) % cadence == 0

    def _extract_reid(self, frame, person_boxes, frame_shape,
                      gate_boxes: list | None = None) -> list[dict]:
        """Score every person box against the appearance gallery.

        ``gate_boxes`` (optional) -- person boxes from the PREVIOUS cycle that
        belong to a track already resolved to a face/trusted identity.  Since
        such a track can never be changed by appearance votes, those boxes skip
        the (potentially expensive) model inference entirely and emit an
        all-Unknown result that carries no appearance vote.  Available only via
        the opt-in ``CCTV_REID_GATE_RESOLVED`` path; defaults to no gating.

        Returns a list parallel to ``person_boxes``:
        ``{"emp", "score", "margin", "strong", "second"}``.  A missing/inactive
        registry yields all-Unknown entries (never a crash).
        """
        registry = getattr(self, "_reid_registry", None)
        if registry is None:
            return [
                {"emp": "Unknown", "score": 0.0, "margin": 0.0,
                 "strong": False, "second": None}
                for _ in person_boxes
            ]
        extractor = registry.extractor
        gate_iou = float(getattr(self, "_reid_gate_iou", REID_GATE_IOU))
        gated = bool(gate_boxes)
        results: list[dict] = []
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        _t = time.time() if getattr(self, "_perf_on", False) else None
        for box in person_boxes:
            x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
            x2 = min(fw, int(box[2])); y2 = min(fh, int(box[3]))
            emp, score, margin, strong, second = ("Unknown", 0.0, 0.0, False, None)
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                try:
                    skip = gated and any(
                        _box_iou((x1, y1, x2, y2), g) >= gate_iou
                        for g in gate_boxes)
                    if not skip:
                        crop = frame[y1:y2, x1:x2]
                        emb = extractor.embed(crop)
                        emp, score, margin, strong, second = \
                            registry.identify(emb)
                except Exception:
                    logger.exception("appearance reid failed for a person")
            results.append({
                "emp": emp,
                "score": round(float(score), 4),
                "margin": round(float(margin), 4),
                "strong": bool(strong),
                "second": second,
            })
        if _t is not None:
            self._perf_raw["reid_ms"] += (time.time() - _t) * 1000.0
        return results

    def _reid_for_camera(self, frame, cam_id: str, person_dets: list,
                         gate_boxes: list | None = None) \
            -> list[dict] | None:
        """Cadence-gated appearance pass for one camera's persons."""
        if not getattr(self, "_reid_enabled", False):
            return None
        if not self._reid_tick(cam_id):
            return None
        try:
            return self._extract_reid(frame, [b for b, _ in person_dets],
                                      frame.shape[:2], gate_boxes=gate_boxes)
        except Exception:
            logger.exception("appearance reid pass failed for %s", cam_id)
            return None

    def _record_phone_diag(self, cam_id: str, phone_dets: list,
                           person_dets: list, frame_shape, run_faces: bool):
        """Part C -- honest per-camera phone/person association measurements."""
        self._phone_stats = getattr(self, "_phone_stats", None) or {}
        st = self._phone_stats.setdefault(cam_id, {
            "cycles": 0, "person_dets": 0, "phone_dets": 0,
            "phone_conf": [], "phone_area": [], "matched": 0, "unmatched": 0,
            "face_frames": 0,
        })
        st["cycles"] += 1
        st["person_dets"] += len(person_dets)
        st["phone_dets"] += len(phone_dets)
        for box, conf in phone_dets:
            st["phone_conf"].append(round(float(conf), 4))
            st["phone_area"].append(int((box[2] - box[0]) * (box[3] - box[1])))
        st["phone_conf"] = st["phone_conf"][-200:]
        st["phone_area"] = st["phone_area"][-200:]
        if run_faces:
            st["face_frames"] += 1
        matched = 0
        if phone_dets and person_dets:
            phone_boxes = [b for b, _ in phone_dets]
            for pbox, _ in person_dets:
                if _phone_near_person(pbox, phone_boxes, frame_shape) is not None:
                    matched += 1
        st["matched"] += min(matched, len(phone_dets))
        st["unmatched"] += max(0, len(phone_dets) - matched)

    def diagnostics_snapshot(self) -> dict:
        """Part C -- rolling phone/person association statistics per camera.

        Only populated when ``CCTV_PHONE_DIAG=1``.  The numbers are the raw
        measurement pipeline used to diagnose weak phone detection; they never
        change detection behaviour.  The exact dict shape is part of the
        Phase 43 contract and is preserved verbatim.
        """
        if not self._phone_diag_on:
            return {"enabled": False, "face_cadence": getattr(self, "_face_cadence", 1)}
        cameras: dict[str, object] = {}
        for cam, st in (getattr(self, "_phone_stats", None) or {}).items():
            n_conf = len(st["phone_conf"])
            n_area = len(st["phone_area"])
            cameras[cam] = {
                "cycles": st["cycles"],
                "person_dets": st["person_dets"],
                "phone_dets": st["phone_dets"],
                "phone_conf_mean": round(_mean(st["phone_conf"]), 4) if n_conf else None,
                "phone_conf_min": min(st["phone_conf"]) if n_conf else None,
                "phone_area_mean": round(_mean(st["phone_area"]), 1) if n_area else None,
                "phone_area_min": min(st["phone_area"]) if n_area else None,
                "matched_to_person": st["matched"],
                "unmatched_to_person": st["unmatched"],
                "face_frames": st["face_frames"],
            }
        return {
            "enabled": True,
            "cycles": self._diag_cycles,
            "face_cadence": getattr(self, "_face_cadence", 1),
            "cameras": cameras,
        }

    def stage_timing(self) -> dict:
        """Phase 45 -- per-cycle itemized detect-stage timings (opt-in).

        Populated only when ``CCTV_PIPELINE_TIMING=1``: the cost of the four
        AI stages that make up ``detect_ms`` for the last ``detect_batch``
        call.  ``main.py`` reads it per cycle and folds it into its published
        EMA so dashboard lag is attributable to a concrete stage
        (person-YOLO / phone pass / face / appearance ReID).  When timing is
        off (or for ``__new__``-constructed test stubs) this returns the
        empty-safe zeroed values, never an error.
        """
        return dict(getattr(self, "_perf_raw", {}))

    def phase44_diagnostics(self) -> dict:
        """Phase 44 -- dedicated phone pass + appearance ReID live info.

        Separate from :meth:`diagnostics_snapshot` on purpose: the latter is a
        contractual Phase 43 surface that tests assert verbatim, while this is
        purely additive monitoring for the operator dashboard.
        """
        phase44: dict = {"phone_pass": None, "reid": {}}
        phone_det = getattr(self, "_phone_detector", None)
        if phone_det is not None:
            st = phone_det.stats()
            st["cadence"] = getattr(self, "_phone_cadence", 3)
            phase44["phone_pass"] = st
        reid_reg = getattr(self, "_reid_registry", None)
        reid: dict = {
            "enabled": bool(getattr(self, "_reid_enabled", False)),
            "cadence": getattr(self, "_reid_cadence", 3),
            "enrolled": 0,
            "extractor": None,
        }
        if reid_reg is not None:
            try:
                reid["enrolled"] = reid_reg.count_enrolled()
            except Exception:
                pass
            reid["extractor"] = getattr(reid_reg.extractor, "model_source",
                                        "built-in")
            # Phase 54 M04 -- fallback transparency: expose the requested model
            # and WHY the active extractor is not it (missing/corrupt/torch).
            reid["requested_model"] = getattr(reid_reg.extractor,
                                              "requested_model", "")
            reid["load_error"] = getattr(reid_reg.extractor, "load_error",
                                         None)
        phase44["reid"] = reid
        return phase44

    def _process_frame(self, frame, phone_boxes: list[tuple] | None = None,
                       cam_id: str | None = None,
                       person_boxes: list[tuple] | None = None,
                       person_confs: list[float] | None = None,
                       run_faces: bool = True,
                       reid_results: list[dict] | None = None) -> tuple:
        """Shared per-frame pipeline: faces -> embeddings -> phone/proximity.

        Parameters
        ----------
        frame
            Single BGR image.
        phone_boxes : list[tuple] | None
            Pre-computed phone detections (from a batched YOLO call).  When
            ``None``, runs its own YOLO inference on this frame.
        cam_id : str | None
            When provided, returns ``(emo_boxes)`` as a flat detection list.
        person_boxes : list[tuple] | None
            Pre-computed person detections (from a batched YOLO call); used
            for the conservative "person present but face unmatched" signal
            and for per-person detection entries (Phase A A1/A7).
        person_confs : list[float] | None
            Per-person YOLO confidences, parallel to ``person_boxes``.
        reid_results : list[dict] | None
            Optional appearance ReID output parallel to ``person_boxes``
            (``{"emp", "score", "margin", "strong", "second"}``).  Attached to
            each per-person detection entry so the spatial tracker can resolve
            identities face recognition cannot.

        Returns
        -------
        Tuple of ``(person_status, face_boxes, face_labels, phone_boxes)``
        in single-frame mode, or a list of detections when ``cam_id`` is set.
        """
        if phone_boxes is None:
            predict_kwargs = dict(
                conf=self.conf,
                classes=_DETECT_CLASSES,
                device=self._device,
                verbose=False,
            )
            if self._half:
                predict_kwargs["half"] = True
            results = self.model.predict(frame, **predict_kwargs)
            phone_dets = _boxes_and_conf_of_class(results[0], PHONE_CLASS)
            person_dets = _boxes_and_conf_of_class(results[0], PERSON_CLASS)
            phone_boxes = [b for b, _ in phone_dets]
            person_boxes = [b for b, _ in person_dets]
            person_confs = [c for _, c in person_dets]

        if person_boxes is None:
            person_boxes = []
        if person_confs is None:
            person_confs = [1.0] * len(person_boxes)

        # --- Step 1: Face detection + embedding via insightface ---
        # ``run_faces=False`` (Part G cadence skip) produces no face entries;
        # the per-person ``__person__`` entries below still carry presence and
        # phone, so employee state is driven by the spatial bridge, not by the
        # cadence decision.
        _t_face = time.time() if getattr(self, "_perf_on", False) and run_faces \
            else None
        face_results = self._face_registry.app.get(frame) if run_faces else []
        if _t_face is not None:
            self._perf_raw["face_ms"] += (time.time() - _t_face) * 1000.0

        face_boxes: list[tuple] = []
        face_labels: list[tuple[str, float]] = []
        face_statuses: list[str] = []
        face_candidates: list[dict | None] = []
        employee_status: dict[str, dict[str, bool]] = {}

        for face in face_results:
            bbox = tuple(round(v) for v in face.bbox)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            if LIVE_MIN_FACE_AREA > 0 and area < LIVE_MIN_FACE_AREA:
                logger.debug(
                    "face bbox=%s area=%d skipped (smaller than LIVE_MIN_FACE_AREA=%d)",
                    bbox, area, LIVE_MIN_FACE_AREA,
                )
                continue
            embedding = face.normed_embedding
            cand = None
            if hasattr(self._face_registry, "identify_candidate"):
                cand = self._face_registry.identify_candidate(embedding)
            if cand is None or not isinstance(cand, dict):
                # Legacy registry stub / Mock (tests): it only exposes
                # identify(); treat its result as a confirmed decision.
                emp_id, score = self._face_registry.identify(embedding)
                cand = {"emp_id": emp_id, "score": score, "margin": score,
                        "second_id": None,
                        "status": RECOG_CONFIRMED if emp_id != "Unknown"
                        else RECOG_UNKNOWN}
            emp_id, score, status = cand["emp_id"], cand["score"], cand["status"]
            # Phase 44B two-tier gate: only a CONFIRMED face becomes an
            # identity this frame.  A CANDIDATE (>= 0.50 but not a clean
            # winner) is surfaced separately so a track with no trusted
            # identity can adopt it after temporal consistency -- but it never
            # drives labels, presence-by-identity, phone attribution or the
            # tracker's face votes, and never demotes anyone.
            if status != RECOG_CONFIRMED:
                emp_id = "Unknown"
            decision = "MATCH" if status == RECOG_CONFIRMED else status
            logger.debug(
                "face bbox=%s score=%.4f conf_thr=%.4f cand_thr=%.4f "
                "margin=%.4f decision=%s target=%s",
                bbox, score, self._face_registry.threshold, CANDIDATE_THRESHOLD,
                cand["margin"], decision,
                cand["emp_id"] if status == RECOG_CONFIRMED else "Unknown",
            )
            if FACE_DEBUG:
                # Top-2 candidates (raw argmax, even below threshold) so a
                # near-miss like EMP004@0.57->CANDIDATE is visible instead of a
                # bare "Unknown" hiding which identity almost matched.
                rank = self._face_registry.identify_k(embedding, k=2)
                cid, cs = rank[0]
                rid = rank[1][0] if len(rank) > 1 else "-"
                rs = rank[1][1] if len(rank) > 1 else 0.0
                logger.info(
                    "FACE_DEBUG camera=%s face=%s bbox=%s area=%d candidate=%s "
                    "name=%s similarity=%.4f margin=%.4f runner_up=%s "
                    "runner_up_sim=%.4f conf_thr=%.4f cand_thr=%.4f decision=%s",
                    cam_id or "single", len(face_boxes), bbox, area, cid,
                    self._face_registry.employee_name(cid), cs, cand["margin"],
                    rid, rs, self._face_registry.threshold, CANDIDATE_THRESHOLD,
                    decision,
                )

            face_boxes.append(bbox)
            face_labels.append((emp_id, score))
            face_statuses.append(status)
            if status == RECOG_CANDIDATE:
                face_candidates.append({
                    "recog_emp": cand["emp_id"],
                    "recog_score": round(float(cand["score"]), 3),
                    "recog_margin": round(float(cand["margin"]), 3),
                    "recog_second": cand["second_id"],
                })
            else:
                face_candidates.append(None)

            if emp_id not in employee_status:
                employee_status[emp_id] = {"present": False, "phone_detected": False}
            employee_status[emp_id]["present"] = True

        if DETECTION_DEBUG:
            logger.info(
                "DETECTION_DEBUG camera=%s persons=%d raw_faces=%d identified_faces=%d phones=%d",
                cam_id or "single", len(person_boxes), len(face_results),
                len(face_labels), len(phone_boxes),
            )

        # --- Step 2: Proximity check: phone near each identified face ---
        for i, (emp_id, _score) in enumerate(face_labels):
            if emp_id == "Unknown":
                continue
            if _phone_near_face(face_boxes[i], phone_boxes, frame.shape):
                employee_status[emp_id]["phone_detected"] = True

        # --- Step 3: A person is visibly present but face none/unmatched.
        # Reported to the tracker so AWAY is not assumed just because a face
        # was not recognised this frame.  We intentionally *do not* invent an
        # identity for it -- conservative presence only.
        person_present = len(person_boxes) > 0

        # Always add Unknown if faces were seen (monitoring/debugging)
        if any(eid == "Unknown" for _, eid in face_labels):
            if "Unknown" not in employee_status:
                employee_status["Unknown"] = {"present": False, "phone_detected": False}
            employee_status["Unknown"]["present"] = True

        if cam_id is not None:
            frame_h, frame_w = frame.shape[:2]
            dets = []
            # Attach raw face geometry so the HUD can draw real boxes, plus
            # the containing person box and frame dims (Phase A A1).
            for i, (emp_id, score) in enumerate(face_labels):
                entry = {
                    "cam": cam_id,
                    "emp_id": emp_id,
                    "phone": False,
                    "face_box": face_boxes[i],
                    "face_score": round(float(score), 3),
                    "recog_status": face_statuses[i] if i < len(face_statuses)
                                   else RECOG_UNKNOWN,
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                    "frame_width": frame_w,
                    "frame_height": frame_h,
                }
                if i < len(face_candidates) and face_candidates[i] is not None:
                    entry.update(face_candidates[i])
                pbox = _containing_person(face_boxes[i], person_boxes, (frame_h, frame_w))
                if pbox:
                    entry["person_box"] = pbox
                if employee_status.get(emp_id, {}).get("phone_detected", False):
                    entry["phone"] = True
                dets.append(entry)
            # Emit one detection per person (Phase A A1) so downstream layers
            # count, track and attribute phones per *person*, not per frame.
            for i, (pbox, pconf) in enumerate(zip(person_boxes, person_confs)):
                phone_box = _phone_near_person(pbox, phone_boxes, (frame_h, frame_w))
                entry = {
                    "cam": cam_id,
                    "det_type": "person",
                    "emp_id": "__person__",
                    "person_box": pbox,
                    "person_conf": round(float(pconf), 3),
                    "phone": phone_box is not None,
                    "phone_box": phone_box,
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                    "frame_width": frame_w,
                    "frame_height": frame_h,
                }
                reid = reid_results[i] if reid_results and i < len(reid_results) \
                    else None
                if reid is not None:
                    entry["reid_emp"] = reid["emp"]
                    entry["reid_score"] = reid["score"]
                    entry["reid_margin"] = reid["margin"]
                    entry["reid_strong"] = bool(reid["strong"])
                dets.append(entry)
            # If a face was seen (even unmatched) it implies a person present.
            if not person_present and len(face_results) > 0:
                person_present = True
            dets.append({
                "cam": cam_id,
                "emp_id": "__person__",
                "phone": False,
                "person_present": person_present,
                "frame_w": frame_w,
                "frame_h": frame_h,
                "frame_width": frame_w,
                "frame_height": frame_h,
            })
            return dets

        return employee_status, face_boxes, face_labels, phone_boxes, person_present


def _boxes_and_conf_of_class(result, cls: int) -> list[tuple]:
    """Extract (box, confidence) pairs of a given YOLO class from a result."""
    items = []
    for box in result.boxes:
        cls_id = int(box.cls) if hasattr(box.cls, "item") else int(box.cls)
        if cls_id == cls:
            conf = getattr(box, "conf", 1.0)
            conf = float(conf.item()) if hasattr(conf, "item") else float(conf)
            items.append((
                tuple(round(v) for v in box.xyxy[0].tolist()),
                conf,
            ))
    return items


def _boxes_of_class(result, cls: int) -> list[tuple]:
    """Extract (x1,y1,x2,y2) boxes of a given YOLO class from a result."""
    return [b for b, _ in _boxes_and_conf_of_class(result, cls)]


def _box_iou(a, b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes (0.0 when either is degenerate)."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union



# Backward-compatible alias
Detector = ActivityDetector
