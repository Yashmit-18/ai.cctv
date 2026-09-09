"""Regression test for the batched detection path (detect_batch).

Found during live webcam smoke testing: ``detect_batch`` passed
``person_boxes=`` to ``_process_frame`` which did not accept it, crashing the
daemon on the very first frame.  Uses fake YOLO/insightface objects so no
model or device is instantiated.
"""

import numpy as np

from src.detector import ActivityDetector, PHONE_CLASS, PERSON_CLASS


class _Box:
    def __init__(self, cls, xyxy):
        self.cls = np.array(cls)
        self.xyxy = np.array([xyxy], dtype=float)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _Model:
    def __init__(self):
        self.calls = 0

    def predict(self, frames, **kwargs):
        self.calls += 1
        frame_list = frames if isinstance(frames, list) else [frames]
        out = []
        for _ in frame_list:
            out.append(_Result([
                _Box(PERSON_CLASS, (30, 30, 90, 90)),
                _Box(PHONE_CLASS, (180, 180, 210, 210)),
            ]))
        return out


class _NoFaces:
    def get(self, frame):
        return []


class _Reg:
    def __init__(self):
        self.app = _NoFaces()

    def identify(self, embedding):
        return ("Unknown", 0.0)


def _detector():
    det = ActivityDetector.__new__(ActivityDetector)
    det.conf = 0.3
    det._device = "cpu"
    det._half = False
    det.model = _Model()
    det._face_registry = _Reg()
    return det


def test_detect_batch_no_crash_and_person_signal():
    det = _detector()
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    dets = det.detect_batch({"cam1": frame})
    assert dets, "detect_batch returned nothing"
    presence = [d for d in dets if "person_present" in d]
    assert presence, "expected a __person__ presence entry"
    assert presence[0]["person_present"] is True
    persons = [d for d in dets if d.get("det_type") == "person"]
    assert persons, "Phase A: expected per-person detection entries"
    assert persons[0]["person_box"] is not None
    assert det.model.calls == 1  # single batched YOLO call for all frames


def test_detect_batch_multiple_frames_one_yolo_call():
    det = _detector()
    frames = {"cam1": np.zeros((100, 100, 3), dtype=np.uint8),
              "cam2": np.zeros((100, 100, 3), dtype=np.uint8)}
    dets = det.detect_batch(frames)
    presence = [d for d in dets if "person_present" in d]
    assert len(presence) == 2
    assert det.model.calls == 1


def test_detect_batch_empty_frames_is_noop():
    det = _detector()
    assert det.detect_batch({}) == []
    assert det.model.calls == 0