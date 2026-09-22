"""Deterministic OFFICE-CONTEXT observation generator (Phase 60A).

Context
-------
This module is SIMULATION-ONLY (like the rest of ``src.simulation``): it is
never imported by production runtime code (``main.py`` / ``app.py``) and it
produces **TEST OBSERVATIONS** only -- never real camera detections.

Purpose
-------
Phase 60A validates the software-side relationship between an employee, the
desk/seat zone, the chair and the office schedule *when valid tracking
observations are supplied*.  Physical IP CCTV/NVR is NOT available, so this
module packages deterministic, scripted footpoint observations into the exact
shape the real context layers consume:

* ``Track`` objects (``src.tracker.Track``) whose ``footpoint_normalized`` is
  bottom-centre of the bbox -- the point ``src.seat_zones`` evaluates;
* ``Observation`` records carrying ``track_id`` / ``employee_id`` /
  ``camera_id`` / ``footpoint`` / ``timestamp`` for assertion and reporting.

No tracking, identity, scheduling, chair or zone *logic* lives here -- that is
all in the real modules (``src.tracker``, ``src.seat_zones``,
``src.seat_chairs``, ``src.office_schedule``).  This generator only shapes the
input.  It is also deliberately free of any biometric, credential or image
payload so the simulated pipeline is privacy-safe by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.tracker import Track

# Normalised working resolution for the simulated frames.  The bbox builder
# converts a normalised footpoint into pixel coordinates inside this frame.
SIM_FRAME_W = 640
SIM_FRAME_H = 480


@dataclass(frozen=True)
class Observation:
    """One deterministic TEST observation (never a real camera result).

    ``ts`` is the simulated wall clock for the observation's step so tests
    (and event timestamps derived from it) stay fully deterministic.
    """

    step: int
    track_id: str
    employee_id: str
    camera_id: str
    fx: float
    fy: float
    ts: float


def footpoint_bbox(fx: float, fy: float, *, fw: int = SIM_FRAME_W,
                   fh: int = SIM_FRAME_H) -> tuple:
    """Build a person bbox whose bottom-centre (footpoint) sits at the
    normalised ``(fx, fy)``.

    A seated/monitored person's box centre can drift above their desk; the
    desk/chair layers evaluate the *standing point* instead.  This helper
    makes that property explicit and deterministic for every simulated track.
    """
    cx = int(fx * fw)
    y2 = int(fy * fh)
    y1 = int(0.1 * fh)
    return (cx - 80, y1, cx + 80, y2)


def make_track(track_id: str, camera_id: str, fx: float, fy: float,
               identity: str | None = None, *, fw: int = SIM_FRAME_W,
               fh: int = SIM_FRAME_H, now: float = 0.0) -> Track:
    """Create one deterministic spatial track with its footpoint at
    ``(fx, fy)`` on the given camera.

    ``identity`` is an *already-resolved* face identity (e.g. ``"EMP001"`` or
    ``"Unknown"``).  This generator never performs recognition -- the test
    supplies the identity the way the face pipeline hands it to the seat layer.
    """
    t = Track(str(track_id), str(camera_id),
              footpoint_bbox(fx, fy, fw=fw, fh=fh), 0.9, fw, fh, float(now))
    t.identity = identity
    return t


def ts_for_step(step: int, *, fps: float = 2.0,
                base: float = 1_000_000.0) -> float:
    """Deterministic simulated wall clock for a simulation step.

    ``base + step / fps`` -- strictly monotonic, repeatable across runs.
    """
    return base + float(step) / float(fps)


def simulate_step(step: int, placements, *, fps: float = 2.0,
                  fw: int = SIM_FRAME_W, fh: int = SIM_FRAME_H) -> tuple:
    """Fold scripted placements into this step's ``(tracks, observations)``.

    ``placements`` is an iterable of ``(track_id, employee_id, camera_id,
    fx, fy)`` tuples.  Returns deterministic ``list[Track]`` plus
    ``list[Observation]`` -- input for a single ``SeatZoneTracker.tick``.
    """
    now = ts_for_step(step, fps=fps)
    tracks: list[Track] = []
    observations: list[Observation] = []
    for tid, emp, cam, fx, fy in placements:
        tracks.append(make_track(tid, cam, fx, fy, emp, fw=fw, fh=fh, now=now))
        observations.append(Observation(
            step=step, track_id=str(tid), employee_id=str(emp),
            camera_id=str(cam), fx=float(fx), fy=float(fy), ts=now))
    return tracks, observations