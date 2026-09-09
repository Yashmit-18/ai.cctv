"""Interactive multi-desk ROI selector.

Draw multiple bounding boxes on the video feed, assign an Employee ID to each,
and save the combined presets to ``data/roi_presets.json``.

Hotkeys
-------
  c  - clear the current (uncommitted) rectangle
  n  - commit current box (prompts for Employee ID) and start a new one
  s  - save all committed desks to JSON and exit
  q  - quit without saving
"""

import json
import logging
import os
import sys

import cv2

from config import (
    DISPLAY_WIDTH,
    RTSP_URL,
    ROI_PRESETS_FILE,
    SOURCE,
    VIDEO_FILE_PATH,
    load_desk_rois,
)

logger = logging.getLogger("cctv.roi_selector")


class MultiROISelector:
    """Click-and-drag tool that accumulates multiple desk ROI boxes."""

    _PANEL_H = 60

    def __init__(self):
        self.desks: dict[str, dict[str, int]] = {}
        self.start_point: tuple[int, int] | None = None
        self.end_point: tuple[int, int] | None = None
        self.drawing = False
        self.frame = None
        self.display = None

    @staticmethod
    def _open_source():
        if SOURCE == "webcam":
            return cv2.VideoCapture(0)
        if SOURCE == "video_file":
            return cv2.VideoCapture(VIDEO_FILE_PATH)
        return cv2.VideoCapture(RTSP_URL)

    def _mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)

    def _scale_back(self, point):
        if self.display is None or self.frame is None:
            return point
        dh, dw = self.display.shape[:2]
        fh, fw = self.frame.shape[:2]
        return int(point[0] * fw / dw), int(point[1] * fh / dh)

    @staticmethod
    def _box_color(idx: int):
        palette = [
            (0, 255, 0), (255, 128, 0), (0, 200, 255), (255, 0, 255),
            (128, 255, 0), (0, 128, 255), (255, 255, 0), (128, 0, 255),
        ]
        return palette[idx % len(palette)]

    def _display_scales(self):
        if self.display is None or self.frame is None:
            return 1.0, 1.0
        return (
            self.display.shape[1] / self.frame.shape[1],
            self.display.shape[0] / self.frame.shape[0],
        )

    def _draw_committed(self, img):
        for idx, (emp_id, roi) in enumerate(self.desks.items()):
            sx, sy = self._display_scales()
            x1 = int(roi["x"] * sx)
            y1 = int(roi["y"] * sy)
            x2 = int((roi["x"] + roi["w"]) * sx)
            y2 = int((roi["y"] + roi["h"]) * sy)
            color = self._box_color(idx)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, emp_id, (x1, max(y1 - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

    def _draw_active(self, img):
        if self.drawing and self.start_point and self.end_point:
            cv2.rectangle(img, self.start_point, self.end_point, (255, 255, 255), 2)
        elif self.end_point and self.start_point and not self.drawing:
            cv2.rectangle(img, self.start_point, self.end_point, (0, 255, 0), 2)

    def _draw_panel(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], self._PANEL_H), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        count = len(self.desks)
        instructions = (
            f"Desks: {count}  |  Drag=draw  n=next  c=clear  s=save  q=quit"
        )
        cv2.putText(
            img, instructions, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )

    def run(self):
        cap = self._open_source()
        if not cap.isOpened():
            raise RuntimeError("Cannot open video source. Check config.py SOURCE.")

        self.desks = load_desk_rois()

        window = "Multi-ROI Selector"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, self._mouse)

        logger.info("==== Multi-Desk ROI Selector ====")
        logger.info("  Drag mouse to draw a rectangle around a desk.")
        logger.info("  Hotkeys: n=commit  c=clear  s=save+exit  q=quit")
        logger.info("  Loaded %d existing desk(s).", len(self.desks))

        while True:
            ret, self.frame = cap.read()
            if not ret:
                logger.warning("End of stream / connection lost.")
                break

            self.frame = cv2.resize(
                self.frame,
                (DISPLAY_WIDTH, int(DISPLAY_WIDTH * self.frame.shape[0] / self.frame.shape[1])),
            )
            self.display = self.frame.copy()

            self._draw_committed(self.display)
            self._draw_active(self.display)
            self._draw_panel(self.display)

            cv2.imshow(window, self.display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("n"):
                self._commit_box()
            elif key == ord("c"):
                self.start_point = None
                self.end_point = None
                self.drawing = False
                logger.info("Selection cleared.")
            elif key == ord("s"):
                self._save_all()
                break
            elif key in (ord("q"), 27):
                logger.info("Quit without saving.")
                break

        cap.release()
        cv2.destroyAllWindows()

    def _commit_box(self):
        if self.start_point is None or self.end_point is None:
            logger.info("No selection to commit. Draw a box first.")
            return

        x1, y1 = self._scale_back(self.start_point)
        x2, y2 = self._scale_back(self.end_point)
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)

        if w <= 0 or h <= 0:
            logger.info("Invalid selection (zero area). Try again.")
            return

        default_id = f"EMP{len(self.desks) + 1:03d}"
        employee_id = input(f"  Employee ID for this desk [{default_id}]: ").strip()
        if not employee_id:
            employee_id = default_id

        if employee_id in self.desks:
            overwrite = input(
                f"  '{employee_id}' already exists. Overwrite? (y/N): "
            ).strip().lower()
            if overwrite != "y":
                logger.info("Skipped.")
                self.start_point = None
                self.end_point = None
                return

        self.desks[employee_id] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        logger.info("Added %s: x=%d y=%d w=%d h=%d", employee_id, x, y, w, h)

        self.start_point = None
        self.end_point = None
        self.drawing = False

    def _save_all(self):
        if not self.desks:
            logger.info("No desks configured. Nothing to save.")
            return

        os.makedirs(os.path.dirname(ROI_PRESETS_FILE), exist_ok=True)
        with open(ROI_PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.desks, f, indent=4)

        logger.info("Saved %d desk(s) -> %s", len(self.desks), ROI_PRESETS_FILE)
        for emp_id, roi in self.desks.items():
            logger.info("  %s: x=%d y=%d w=%d h=%d", emp_id, roi["x"], roi["y"], roi["w"], roi["h"])


# Backward-compatible alias
ROISelector = MultiROISelector


if __name__ == "__main__":
    from config import setup_logging
    setup_logging()
    try:
        MultiROISelector().run()
    except Exception as err:
        logger.error("Error: %s", err)
        sys.exit(1)
