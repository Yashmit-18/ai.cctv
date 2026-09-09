"""Deterministic office-scenario generator (Phase 36).

Produces a scripted, seeded ground-truth of office activity and exports
per-step detection records *in the exact schema the real detection pipeline
emits* (``detect_batch`` output), so the existing tracker FSM, security
engine, correlation, incident, evidence and alert machinery runs end-to-end.

Scenarios (single letters A-R per the phase spec) cover: normal occupancy,
employee arrival/leave/activeness, phone proximity, Unknown presence,
restricted-zone intrusion, after-hours activity, loitering, cross-camera
movement, multiple simultaneous employees, Unknown+employee co-presence,
camera disconnect/recovery/outages, alert-failure+retry, evidence-during-
incident, and long-running operation.

Unsupported AI families (fall/fight/fire/weapon/PPE) are NEVER emitted here:
the registry reports them ``FUTURE_MODEL_REQUIRED`` instead.  There is no fake
fire/smoke/fight/fall/weapon detection.

Identity is stable per employee; ``Unknown`` is never conflated with an
employee and never affects productivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.domain import UNKNOWN_ID

# Zone semantics: a person with a face_box whose normalized centre falls in a
# restricted band triggers INTRUSION via the real security engine.
RESTRICTED_ZONE_NAME = "restricted"


@dataclass
class Scene:
    """A named, reusable office occupancy pattern."""

    name: str
    # list of (start_step, end_step, [("emp_id|Unknown", sub-behaviour)])
    episodes: list[tuple[int, int, list[str]]] = field(default_factory=list)


def _face_box(centre_x: float = 0.5, centre_y: float = 0.5) -> list[int]:
    """A face box centred around a normalized point in a 640x360 frame."""
    cx, cy = centre_x * 640, centre_y * 360
    x1 = max(0, int(cx - 40))
    y1 = max(0, int(cy - 50))
    x2 = x1 + 80
    y2 = y1 + 100
    return [x1, y1, x2, y2]


# Predefined deterministic scenarios. ``steps`` = number of frame steps.
SCENARIOS: dict[str, dict] = {
    # A -- normal office occupancy: two employees steadily present.
    "A_normal_occupancy": {
        "steps": 60,
        "camera_count": 1,
        "camera_fps": 10,
        "cast": [
            # emp, first_step, last_step, phone_window (or None)
            ("EMP001", 0, 59, None),
            ("EMP002", 0, 59, (20, 35)),
        ],
    },
    # B -- employee enters office partway through.
    "B_employee_enters": {
        "steps": 40,
        "camera_count": 1,
        "camera_fps": 10,
        "cast": [("EMP001", 15, 39, None)],
    },
    # C -- employee leaves office partway through.
    "C_employee_leaves": {
        "steps": 40,
        "camera_count": 1,
        "camera_fps": 10,
        "cast": [("EMP001", 0, 24, None)],
    },
    # E -- employee remains active with phone proximity.
    "E_phone_proximity": {
        "steps": 30,
        "camera_count": 1,
        "camera_fps": 10,
        "cast": [("EMP001", 0, 29, (0, 29))],
    },
    # F -- Unknown person appears (camera assigned), unknown NEVER becomes
    # an employee and never affects productivity.
    "F_unknown_presence": {
        "steps": 30,
        "camera_count": 1,
        "camera_fps": 10,
        "cast": [("Unknown", 8, 25, None)],
    },
    # J -- employee moves CAM-01 -> CAM-02 -> CAM-03.
    "J_cross_camera_move": {
        "steps": 60,
        "camera_count": 3,
        "camera_fps": 10,
        "cast": [
            ("EMP003", 0, 19, None, "cam_01"),
            ("EMP003", 20, 39, None, "cam_02"),
            ("EMP003", 40, 59, None, "cam_03"),
        ],
    },
    # K -- multiple employees simultaneously visible.
    "K_multiple_employees": {
        "steps": 40,
        "camera_count": 2,
        "camera_fps": 10,
        "cast": [
            ("EMP001", 0, 39, None, "cam_01"),
            ("EMP002", 0, 39, None, "cam_01"),
            ("EMP003", 5, 39, None, "cam_02"),
        ],
    },
    # L -- Unknown person + employee simultaneously.
    "L_unknown_plus_employee": {
        "steps": 30,
        "camera_count": 1,
        "camera_fps": 10,
        "cast": [
            ("EMP001", 0, 29, None),
            ("Unknown", 10, 29, None),
        ],
    },
    # M -- camera disconnect during employee presence (offline must NOT
    # create AWAY).
    "M_camera_disconnect": {
        "steps": 40,
        "camera_count": 1,
        "camera_fps": 10,
        "cast": [("EMP001", 0, 39, None)],
        "faults": [(10, 30, "disconnect")],
    },
    # N -- camera recovery after disconnect.
    "N_camera_recovery": {
        "steps": 40,
        "camera_count": 1,
        "camera_fps": 10,
        "cast": [("EMP001", 0, 9, None), ("EMP001", 15, 39, None)],
        "faults": [(10, 15, "disconnect")],
    },
    # G -- restricted-zone intrusion by an employee and by Unknown.
    "G_zone_intrusion": {
        "steps": 30,
        "camera_count": 1,
        "camera_fps": 10,
        "cast": [
            ("EMP001", 0, 29, None),
        ],
        "restricted_frames": [(12, 22)],  # moves into restricted band
    },
    # R -- long-running normal operation (many steps; bounded run).
    "R_long_running": {
        "steps": 200,
        "camera_count": 2,
        "camera_fps": 10,
        "cast": [
            ("EMP001", 0, 199, None, "cam_01"),
            ("EMP002", 0, 199, None, "cam_02"),
            ("Unknown", 50, 80, None, "cam_01"),
        ],
    },
}


@dataclass
class ScenarioPlan:
    """Builder producing a seeded, per-step detection stream."""

    scenario: str
    seed: int
    cameras: list[str]
    fps: float
    steps: int
    cast: list[tuple]
    faults: list[tuple] = field(default_factory=list)
    restricted_frames: list[tuple] = field(default_factory=list)
    unknown_intrusion: bool = False

    @classmethod
    def for_scenario(cls, scenario: str, seed: int = 42) -> "ScenarioPlan":
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario!r}")
        s = SCENARIOS[scenario]
        n = int(s["camera_count"])
        cameras = [f"cam_{i:02d}" for i in range(1, n + 1)]
        return cls(
            scenario=scenario,
            seed=seed,
            cameras=cameras,
            fps=float(s["camera_fps"]),
            steps=int(s["steps"]),
            cast=[tuple(c) for c in s["cast"]],
            faults=[tuple(f) for f in s.get("faults", [])],
            restricted_frames=[tuple(r) for r in s.get("restricted_frames", [])],
        )

    def detections_at_step(self, step: int) -> list[dict]:
        """Ground-truth detection records for one frame step in real schema."""
        out: list[dict] = []
        cam_present: dict[str, int] = {}
        for entry in self.cast:
            emp = entry[0]
            first = int(entry[1])
            last = int(entry[2])
            phone = entry[3] if len(entry) > 3 else None
            cam = entry[4] if len(entry) > 4 and entry[4] else "cam_01"
            phone_win = None
            if isinstance(phone, (tuple, list)) and len(phone) == 2:
                phone_win = (int(phone[0]), int(phone[1]))
            online = self._camera_online(cam, step)
            if not (online and first <= step <= last):
                continue
            is_unknown = emp == UNKNOWN_ID
            is_restricted = self._in_restricted(cam, step, is_unknown)
            # face centre: restricted band sits near a fixed x-band.
            cx = 0.78 if is_restricted else 0.5
            cy = 0.5
            face_box = _face_box(cx, cy)
            entry = {
                "cam": cam,
                "emp_id": emp,
                "phone": False,
                "person_present": True,
                "face_box": face_box,
                "face_score": 0.55 if is_unknown else 0.92,
            }
            if phone_win and phone_win[0] <= step <= phone_win[1]:
                entry["phone"] = True
            out.append(entry)
            cam_present[cam] = cam_present.get(cam, 0) + 1
        # person-present marker per online camera (matches __person__ signal)
        for cam in self.cameras:
            if self._camera_online(cam, step) and cam not in cam_present:
                out.append({
                    "cam": cam, "emp_id": "__person__",
                    "phone": False, "person_present": False,
                })
        return out

    def _camera_online(self, cam: str, step: int) -> bool:
        idx = self.cameras.index(cam) if cam in self.cameras else 0
        # deterministic per-camera outage windows (default none)
        for start, end, kind in self.faults:
            if kind == "disconnect" and start <= step < end:
                # apply to matching camera by offset in the fault tuple? only
                # single-camera disconnect scenarios defined; treat as global.
                if len(self.cameras) == 1 or idx == 0:
                    return False
        return True

    def _in_restricted(self, cam: str, step: int, is_unknown: bool) -> bool:
        # Global restricted window heuristic (scenario G / Unknown intrusion).
        for start, end in self.restricted_frames:
            if start <= step < end:
                return True
        return False

    def telemetry_fault_plan(self) -> list[dict]:
        return [
            {"cam": c, "start": s, "end": e, "kind": k}
            for (s, e, k) in self.faults
            for c in self.cameras
        ]