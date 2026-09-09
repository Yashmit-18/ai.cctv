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

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    CONF_THRESHOLD,
    DETECTION_DEBUG,
    FACE_DEBUG,
    LIVE_MIN_FACE_AREA,
    MODEL_PATH,
)
from src.face_registry import FaceRegistry

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
    ):
        self.conf = conf
        self._device = _resolve_device(device)
        self._half = self._device == "cuda"

        logger.info(
            "Loading YOLO model from %s  (device=%s, half=%s) ...",
            model_path, self._device, self._half,
        )
        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            logger.error("YOLO model not found or failed to load: %s", model_path)
            logger.error("Check that models/yolov8n.pt exists or run: "
                         "python -c 'from ultralytics import YOLO; YOLO(\"yolov8n.pt\")'")
            raise RuntimeError(f"YOLO model unavailable at {model_path!r}") from exc
        logger.info("YOLO model loaded successfully.")

        self._face_registry = FaceRegistry(
            detect_size=face_detect_size if face_detect_size else 640
        )

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

    def detect_batch(self, frames: dict[str, object]) -> list[dict]:
        """Run batched inference across multiple camera frames (Phase 7).

        Parameters
        ----------
        frames : dict[str, object]
            ``{cam_id: frame}`` -- only cameras with a non-``None`` frame
            should be included.  Order is preserved.

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
        results = self.model.predict(frame_list, **predict_kwargs)

        out: list[dict] = []
        for idx, cam_id in enumerate(cam_ids):
            frame = frames[cam_id]
            phone_dets = _boxes_and_conf_of_class(results[idx], PHONE_CLASS)
            person_dets = _boxes_and_conf_of_class(results[idx], PERSON_CLASS)
            out.extend(self._process_frame(
                frame,
                [b for b, _ in phone_dets],
                cam_id,
                person_boxes=[b for b, _ in person_dets],
                person_confs=[c for _, c in person_dets],
            ))
        return out

    def _process_frame(self, frame, phone_boxes: list[tuple] | None = None,
                       cam_id: str | None = None,
                       person_boxes: list[tuple] | None = None,
                       person_confs: list[float] | None = None) -> tuple:
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
        face_results = self._face_registry.app.get(frame)

        face_boxes: list[tuple] = []
        face_labels: list[tuple[str, float]] = []
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
            emp_id, score = self._face_registry.identify(embedding)
            decision = "MATCH" if emp_id != "Unknown" else "UNKNOWN"
            logger.debug(
                "face bbox=%s score=%.4f threshold=%.4f decision=%s -> %s",
                bbox, float(score), self._face_registry.threshold, decision, emp_id,
            )
            if FACE_DEBUG:
                # Top-2 candidates (raw argmax, even below threshold) so a
                # near-miss like EMP004@0.57->UNKNOWN is visible instead of a
                # bare "Unknown" hiding which identity almost matched.
                rank = self._face_registry.identify_k(embedding, k=2)
                cid, cs = rank[0]
                rid = rank[1][0] if len(rank) > 1 else "-"
                rs = rank[1][1] if len(rank) > 1 else 0.0
                logger.info(
                    "FACE_DEBUG camera=%s face=%s bbox=%s area=%d candidate=%s "
                    "name=%s similarity=%.4f runner_up=%s runner_up_sim=%.4f "
                    "threshold=%.4f decision=%s",
                    cam_id or "single", len(face_boxes), bbox, area, cid,
                    self._face_registry.employee_name(cid), cs, rid, rs,
                    self._face_registry.threshold, decision,
                )

            face_boxes.append(bbox)
            face_labels.append((emp_id, score))

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
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                    "frame_width": frame_w,
                    "frame_height": frame_h,
                }
                pbox = _containing_person(face_boxes[i], person_boxes, (frame_h, frame_w))
                if pbox:
                    entry["person_box"] = pbox
                if employee_status.get(emp_id, {}).get("phone_detected", False):
                    entry["phone"] = True
                dets.append(entry)
            # Emit one detection per person (Phase A A1) so downstream layers
            # count, track and attribute phones per *person*, not per frame.
            for pbox, pconf in zip(person_boxes, person_confs):
                phone_box = _phone_near_person(pbox, phone_boxes, (frame_h, frame_w))
                dets.append({
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
                })
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



# Backward-compatible alias
Detector = ActivityDetector
