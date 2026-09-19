"""Dedicated phone detector (Phase 44).

Modules
-------
* ``PhoneDetector`` -- a cadence-gated, higher-resolution phone-detection pass
  that plugs into the existing ``ActivityDetector``.  It can use a dedicated
  (ideally fine-tuned) YOLO phone model, or reuse the base YOLO model with a
  phone-only class filter at a higher input resolution (better small-object
  recall than the stock 640 pass).

Contract
--------
* Output is normalised to ``[(box, conf), ...]`` where every box is a plain
  ``(x1, y1, x2, y2)`` **int** tuple and every confidence a ``float`` --
  identical shape to the detector's ``_boxes_and_conf_of_class`` so
  downstream association logic (person/phone proximity) is unchanged.
* The class-to-phone mapping is explicit (``phone_class``).  COCO uses 67
  (``cell phone``); a fine-tuned single-class phone model commonly uses 0.
* Model selection is fail-safe: a configured-but-missing model path falls back
  to the base model with a warning -- never a silent crash and never a
  silently disabled phone check.
* The detector is inference-only and holds no per-person state; per-person
  phone attribution and temporal evidence remain in the spatial tracker.

Performance policy
------------------
Phone detection is intentionally *not* run on every frame of every camera.
``ActivityDetector`` gates this pass with ``PHONE_DETECT_CADENCE`` and the
tracker's per-track evidence window tolerates the resulting intermittent
observations.
"""

from __future__ import annotations

import logging
import os
from statistics import mean as _mean

logger = logging.getLogger("cctv.phone_detector")


def _normalize_box(box) -> tuple[int, int, int, int]:
    """Coerce a raw ultralytics box into a 4-int (x1,y1,x2,y2) tuple."""
    vals = [float(v) for v in box.xyxy[0].tolist()]
    return (round(vals[0]), round(vals[1]), round(vals[2]), round(vals[3]))


class PhoneDetector:
    """YOLO-based phone detector with normalised, measurable output.

    Parameters
    ----------
    model : object | None
        An existing (already loaded) YOLO model to reuse when no dedicated
        ``model_path`` is supplied.
    model_path : str
        Absolute path to a dedicated phone model.  Empty/"" or a path that
        does not exist -> reuse ``model`` (base model) with a warning.
    phone_class : int
        Class id treated as "phone".
    conf : float
        Confidence threshold for the phone pass (never silently lowered by
        this module).
    imgsz : int
        Input resolution for the phone pass (higher = better small phones).
    device : str
        Inference device string (``cuda`` / ``cpu`` / ...).
    half : bool
        Use half precision when the base detector runs with half precision.
    """

    def __init__(self, *, model=None, model_path: str = "",
                 phone_class: int = 67, conf: float = 0.4,
                 imgsz: int = 1280, device: str = "cpu", half: bool = False):
        self.phone_class = int(phone_class)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device
        self._half = bool(half)
        self._model = None
        self._source = "none"

        if model_path and os.path.isfile(model_path):
            try:
                from ultralytics import YOLO
                self._model = YOLO(model_path)
                self._source = str(model_path)
                logger.info("PhoneDetector using dedicated model %s "
                            "(class=%d imgsz=%d)", model_path, self.phone_class,
                            self.imgsz)
            except Exception as exc:
                logger.error("PhoneDetector failed to load %s (%s); "
                             "reusing base model.", model_path, exc)
                self._model = model
                self._source = "base-model"
        elif model is not None:
            if model_path:
                logger.warning(
                    "Phone model %r not found; reusing the base YOLO model "
                    "for the phone pass.", model_path)
            self._model = model
            self._source = "base-model"
        else:
            self._model = None
            self._source = "none"

        self._calls = 0
        self._boxes_seen = 0
        self._conf_window: list[float] = []

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def source(self) -> str:
        return self._source

    @property
    def ready(self) -> bool:
        """True when the phone pass can actually run (a model is available)."""
        return self._model is not None

    def stats(self) -> dict:
        """Rolling measurement for diagnostics (never changes behaviour)."""
        confs = self._conf_window[-200:]
        return {
            "calls": self._calls,
            "boxes_seen": self._boxes_seen,
            "conf_mean": round(_mean(confs), 4) if confs else None,
            "conf_min": min(confs) if confs else None,
            "source": self._source,
            "phone_class": self.phone_class,
            "imgsz": self.imgsz,
            "cadence": None,  # set by ActivityDetector in diagnostics
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame) -> list[tuple]:
        """Run the phone pass on one frame.

        Returns
        -------
        list[tuple]
            ``[(box, conf), ...]`` where ``box`` is a normalised int
            ``(x1, y1, x2, y2)`` and ``conf`` the YOLO confidence.  Empty list
            when no model is available.
        """
        if self._model is None or frame is None:
            return []
        kwargs = dict(
            conf=self.conf,
            classes=[self.phone_class],
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if self._half:
            kwargs["half"] = True
        results = self._model.predict(frame, **kwargs)
        boxes: list[tuple] = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf.item())
                boxes.append((_normalize_box(box), conf))
                self._conf_window.append(float(box.conf.item()))
        self._calls += 1
        self._boxes_seen += len(boxes)
        return boxes