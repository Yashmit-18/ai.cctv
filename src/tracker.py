"""Dynamic employee state tracker with smoothing buffer (Phase 6/7).

Split into two layers:

* ``EmployeeTracker``  -- single-employee FSM with a smoothing buffer that
  prevents rapid state-thrashing, camera-offline awareness, and presence
  patience so a temporarily-unreadable face is not instantly marked AWAY.
* ``MultiTracker``     -- manages a collection of ``EmployeeTracker``
  instances dynamically (one per recognised employee), logs transitions to
  SQLite, tracks pre-commit live states for the HUD, deduplicates
  multi-camera detections, and provides ``flush_all()`` for shutdown.

States
------
  ACTIVE    -- face detected in frame, no phone detected
  ON_PHONE  -- face detected, phone is near face/body
  AWAY      -- face NOT detected while at least one camera is online

Camera-offline semantics (CRITICAL)
-----------------------------------
The daemon passes ``camera_online`` into :meth:`process_batch`.  When *no*
camera is delivering frames the FSM is frozen -- employees are NOT advanced
to AWAY and no AWAY time is logged.  This prevents dead cameras from
silently becoming employee downtime.

Presence patience
-----------------
When a recognised employee stops being seen but a person is still visibly
present (person detection), the transition to AWAY is delayed by
``PRESENCE_PATIENCE_SEC`` rather than the short smoothing buffer, avoiding
ACTIVE -> AWAY flicker while the face is turned away / occluded.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from statistics import mean as _mean

from config import (
    IDENTITY_STABILITY_FRAMES,
    PRESENCE_PATIENCE_SEC,
    SMOOTHING_BUFFER_SEC,
    TRACK_DEBUG,
    TRACK_IOU_THRESHOLD,
    TRACK_MAX_AGE_SEC,
    TRACK_TRAJECTORY_LEN,
    IDENTITY_ADOPT_FRAMES,
    IDENTITY_SWITCH_FRAMES,
)
from src.database import log_interval
from src.domain import UNKNOWN_ID

logger = logging.getLogger("cctv.tracker")

ACTIVE = "ACTIVE"
ON_PHONE = "ON_PHONE"
AWAY = "AWAY"

_VALID_STATES = {ACTIVE, ON_PHONE, AWAY}


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


def _raw_state(status: dict) -> str:
    """Derive a single-employee raw state from the detector status dict."""
    present = status.get("present", False)
    phone = status.get("phone_detected", False)
    if not present:
        return AWAY
    if phone:
        return ON_PHONE
    return ACTIVE


# ======================================================================
# EmployeeTracker  (single-employee FSM)
# ======================================================================

class EmployeeTracker:
    """Finite State Machine for one employee with a smoothing buffer.

    The tracker holds a *candidate* state.  When the candidate differs from
    the committed state for at least the buffer duration the transition is
    committed (and logged to SQLite).
    """

    def __init__(self, conn, employee_id: str,
                 source: str = ""):
        self._conn = conn
        self.employee_id = employee_id
        self.source = source or ""
        self._state: str | None = None
        self._candidate: str | None = None
        self._since: float = time.time()
        self._pending_since: float = time.time()
        self._consecutive: int = 0
        self._ever_observed = False
        self.last_seen: float | None = None

    def process(self, status: dict, camera_online: bool = True,
                person_present: bool = False):
        """Advance the FSM with a new detector status dict.

        ``camera_online=False`` freezes the state machine: no state change
        and no time accounting, so a dead camera cannot fabricate AWAY time.
        """
        now = time.time()

        if not camera_online:
            # No usable frames this cycle -- do not accrue AWAY or shift state.
            # Two actions keep offline time OUT of employee productivity:
            #   * advance the current interval's start so the in-flight row
            #     never swallows the offline gap (audited in flush / commit),
            #   * reset the pending timer so a later return never commits
            #     immediately.
            if self._state is not None and self._since < now:
                self._since = now
            self._pending_since = now
            self._consecutive = 0
            return

        raw = _raw_state(status)
        if raw != AWAY:
            self._ever_observed = True
            self.last_seen = now

        # ----- First observation -------------------------------------------
        # Adopt the first observed state immediately.  This avoids both the
        # synthetic "starts as AWAY before first detection" artifact and the
        # need to log a None -> state transition.
        if self._state is None:
            self._state = raw
            self._candidate = raw
            self._since = now
            self._pending_since = now
            self._consecutive = 0
            return

        # ----- Presence patience: person visible but face unmatched ---------
        # While a person is physically on screen we do NOT assume the
        # employee is AWAY (face may be turned away / occluded).  Hold the
        # last known candidate until the person disappears.
        if raw == AWAY and person_present:
            self._pending_since = now
            self._candidate = self._state
            return

        # ----- Identity stability -----------------------------------------
        if raw == self._state:
            self._candidate = raw
            self._pending_since = now
            self._consecutive = 0
            return

        if raw == self._candidate:
            self._consecutive += 1
        else:
            self._candidate = raw
            self._consecutive = 1
            self._pending_since = now
            return

        # Buffer: widen to presence-patience when going AWAY while a person
        # is still visible (face temporarily unreadable).
        buffer = SMOOTHING_BUFFER_SEC
        if raw == AWAY and person_present:
            buffer = max(buffer, PRESENCE_PATIENCE_SEC)

        # Frame-count stability floor (defends against fast identity flicker).
        if IDENTITY_STABILITY_FRAMES > 0 and self._consecutive < IDENTITY_STABILITY_FRAMES:
            return

        if now - self._pending_since >= buffer:
            self._commit(raw, now)

    def _commit(self, raw: str, now: float):
        duration = now - self._since
        if self._state is not None and duration > 0:
            log_interval(
                self._conn,
                _iso(self._since),
                self.employee_id,
                self._state,
                round(duration, 2),
                source=self.source,
            )
            logger.debug("%s: %s -> %s (%.1fs)",
                         self.employee_id, self._state, raw, duration)
        elif self._state is None:
            logger.debug("%s: first observation -> %s", self.employee_id, raw)
        self._state = raw
        self._since = now
        self._candidate = raw
        self._consecutive = 0

    @property
    def current_state(self) -> str:
        return self._candidate or self._state or AWAY

    @property
    def committed_state(self) -> str:
        return self._state if self._state is not None else AWAY

    @property
    def duration_seconds(self) -> float:
        """Wall seconds of the current committed interval so far.

        Offline time is excluded because the interval start is advanced
        while the camera is offline (see :meth:`process`).
        """
        if self._state is None:
            return 0.0
        return max(0.0, time.time() - self._since)

    def flush(self):
        now = time.time()
        duration = now - self._since
        if self._state is not None and duration > 0:
            log_interval(
                self._conn,
                _iso(self._since),
                self.employee_id,
                self._state,
                round(duration, 2),
                source=self.source,
            )
        self._since = now


# ======================================================================
# MultiTracker  (dynamic collection manager)
# ======================================================================

class MultiTracker:
    """Manages ``EmployeeTracker`` instances created on-the-fly.

    Employees are discovered dynamically from face recognition.  Trackers are
    spawned the first time an employee_id appears in the detection output.
    :meth:`process_batch` deduplicates overlapping multi-camera detections so
    each tracker advances exactly once per cycle.
    """

    def __init__(self, conn):
        self._conn = conn
        self._trackers: dict[str, EmployeeTracker] = {}
        logger.info("MultiTracker initialised (dynamic face-based mode).")

    def _ensure_tracker(self, emp_id: str, source: str = "") -> EmployeeTracker:
        if emp_id not in self._trackers:
            self._trackers[emp_id] = EmployeeTracker(self._conn, emp_id, source=source)
            logger.info("New tracker created for %s", emp_id)
        return self._trackers[emp_id]

    @staticmethod
    def _deduplicate(detections: list[dict]) -> dict[str, dict[str, bool]]:
        """Merge per-camera detections into one status dict per employee."""
        merged: dict[str, dict[str, bool]] = {}
        for det in detections:
            emp_id = det.get("emp_id", "Unknown")
            entry = merged.setdefault(emp_id, {"present": False, "phone_detected": False})
            entry["present"] = entry["present"] or True
            if det.get("phone", False):
                entry["phone_detected"] = True
        return merged

    @staticmethod
    def _person_detected(detections: list[dict]) -> bool:
        """True if any camera present signal / person detection was emitted."""
        for det in detections:
            if det.get("emp_id") == "__person__" and det.get("person_present", False):
                return True
            if det.get("emp_id") == "Unknown" and det.get("person_present", False):
                return True
        return False

    def _process_one(self, emp_id: str, status: dict, camera_online: bool,
                     person_present: bool, source: str = ""):
        tracker = self._ensure_tracker(emp_id, source=source)
        tracker.process(status, camera_online=camera_online,
                        person_present=person_present)

    def process(self, multi_status: dict, camera_online: bool = True,
                person_present: bool = False):
        """Single-camera / batch-agnostic interface (backwards compatible)."""
        employees = multi_status.get("employees", {})
        self._process_statuses(employees, camera_online, person_present)

    def process_batch(self, detections: list[dict], camera_online: bool = True):
        """Advance all trackers from a flattened multi-camera detection list.

        ``camera_online`` should be ``False`` when no camera produced a live
        frame this cycle (see class docstring for semantics).
        """
        if not camera_online or not detections:
            # Freeze everything (no live frames) but preserve trackers.
            for tracker in self._trackers.values():
                tracker.process({"present": False}, camera_online=False)
            return

        person_present = self._person_detected(detections)
        merged = self._deduplicate(detections)

        all_emp_ids = set(self._trackers.keys()) | {
            e for e in merged.keys() if e != "__person__"}

        for emp_id in all_emp_ids:
            status = merged.get(emp_id, {"present": False, "phone_detected": False})
            source = {d.get("emp_id"): d.get("cam", "")
                      for d in detections if d.get("emp_id") != "__person__"}.get(emp_id, "")
            self._process_one(emp_id, status, camera_online, person_present, source)

    def _process_statuses(self, employees: dict, camera_online: bool,
                          person_present: bool):
        all_emp_ids = set(self._trackers.keys()) | set(employees.keys())
        for emp_id in all_emp_ids:
            status = employees.get(emp_id, {"present": False, "phone_detected": False})
            self._process_one(emp_id, status, camera_online, person_present)

    def live_states(self) -> dict[str, str]:
        return {eid: t.current_state for eid, t in self._trackers.items()}

    def live_state(self, employee_id: str) -> str | None:
        t = self._trackers.get(employee_id)
        return t.current_state if t else None

    def live_sources(self) -> dict[str, str]:
        return {eid: t.source for eid, t in self._trackers.items()}

    def live_durations(self) -> dict[str, float]:
        """Current committed-interval durations (seconds) per employee."""
        return {eid: t.duration_seconds for eid, t in self._trackers.items()}

    def live_last_seen(self) -> dict[str, float | None]:
        """Epoch timestamp of last positive observation per employee."""
        return {eid: t.last_seen for eid, t in self._trackers.items()}

    @property
    def employee_ids(self) -> list[str]:
        return list(self._trackers.keys())

    def flush_all(self):
        logger.info("Flushing all trackers ...")
        for tracker in self._trackers.values():
            tracker.flush()


# ======================================================================
# Backward-compatible alias (Phase 1/2 name)
# ======================================================================
class StateTracker:
    """Drop-in replacement for legacy ``StateTracker`` API."""

    def __init__(self, conn):
        self._conn = conn
        self._states: dict[str, str] = {}
        self._candidate: dict[str, str] = {}
        self._since: dict[str, float] = {}
        self._pending_since: dict[str, float] = {}

    def _ensure(self, employee_id: str, now: float):
        if employee_id not in self._states:
            self._states[employee_id] = AWAY
            self._since[employee_id] = now
            self._candidate[employee_id] = AWAY
            self._pending_since[employee_id] = now

    def process(self, employee_id: str, status: dict, camera_online: bool = True):
        now = time.time()
        self._ensure(employee_id, now)
        if not camera_online:
            self._pending_since[employee_id] = now
            return
        raw = _raw_state(status)
        current = self._states[employee_id]

        if raw == current:
            self._candidate[employee_id] = raw
            self._pending_since[employee_id] = now
            return

        if raw != self._candidate[employee_id]:
            self._candidate[employee_id] = raw
            self._pending_since[employee_id] = now
            return

        if now - self._pending_since[employee_id] >= SMOOTHING_BUFFER_SEC:
            duration = now - self._since[employee_id]
            log_interval(
                self._conn,
                _iso(self._since[employee_id]),
                employee_id,
                current,
                round(duration, 2),
                source="",
            )
            self._states[employee_id] = raw
            self._since[employee_id] = now

    def current_state(self, employee_id: str) -> str | None:
        if employee_id not in self._candidate:
            return self._states.get(employee_id)
        return self._candidate[employee_id]

    def flush(self, employee_id: str):
        now = time.time()
        self._ensure(employee_id, now)
        duration = now - self._since[employee_id]
        log_interval(
            self._conn,
            _iso(self._since[employee_id]),
            employee_id,
            self._states[employee_id],
            round(duration, 2),
            source="",
        )
        del self._states[employee_id]
        self._candidate.pop(employee_id, None)
        self._since.pop(employee_id, None)
        self._pending_since.pop(employee_id, None)


# ======================================================================
# Phase A -- Spatial person tracking (A5/A6)
# ======================================================================
# ``SpatialTracker`` adds what the identity FSM intentionally does not have:
# *spatial* continuity.  A :class:`Track` is one physical person instance on
# one camera, identified by a stable ``track_id`` plus a smoothed bounding
# box, trajectory, and (separately) a *temporally smoothed identity*.
#
# Identity rules (anti-flicker: EMP001 never flips to Unknown on one frame):
#   * a fresh track adopts the first identity with ``adopt_frames`` consecutive
#     votes;
#   * a KNOWN identity is never demoted to Unknown while the track is alive --
#     an unreadable face simply stops voting, the identity is kept;
#   * identity switches to a DIFFERENT known employee only after
#     ``switch_frames`` consecutive votes AND the previously adopted identity
#     has dropped out of the run entirely (i.e. that person effectively left
#     this track's space).
class Track:
    """One persistent spatial person instance (Phase A)."""

    def __init__(self, track_id: str, cam: str, bbox: tuple,
                 confidence: float, frame_w: int, frame_h: int, now: float,
                 adopt_frames: int | None = None,
                 switch_frames: int | None = None):
        self.track_id = track_id
        self.cam = cam
        self.frame_w = int(frame_w or 0) or 640
        self.frame_h = int(frame_h or 0) or 480
        self.bbox = tuple(bbox)
        self.confidence = float(confidence)
        self.first_seen = now
        self.last_seen = now
        self.identity: str | None = None
        self.identity_score = 0.0
        self.identity_updated: float | None = None
        self.phone = False
        self.phone_since: float | None = None
        self.trajectory: deque[tuple[float, float]] = deque()
        self._adopt_frames = int(adopt_frames) if adopt_frames is not None \
            else IDENTITY_ADOPT_FRAMES
        self._switch_frames = int(switch_frames) if switch_frames is not None \
            else IDENTITY_SWITCH_FRAMES
        self._runs: dict[str, int] = {}
        self._scores: dict[str, list[float]] = {}
        self.trajectory.append(self.centroid_normalized)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def centroid_normalized(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        cx = ((x1 + x2) / 2.0) / self.frame_w
        cy = ((y1 + y2) / 2.0) / self.frame_h
        return round(max(0.0, min(1.0, cx)), 4), round(max(0.0, min(1.0, cy)), 4)

    def update(self, bbox: tuple, confidence: float, phone: bool, now: float):
        """Merge a new person detection into this track (EMA smoothing)."""
        blend = 0.4
        new_x1, new_y1, new_x2, new_y2 = bbox
        self.bbox = (
            round(self.bbox[0] * (1 - blend) + new_x1 * blend),
            round(self.bbox[1] * (1 - blend) + new_y1 * blend),
            round(self.bbox[2] * (1 - blend) + new_x2 * blend),
            round(self.bbox[3] * (1 - blend) + new_y2 * blend),
        )
        self.confidence = float(confidence)
        self.last_seen = now
        if len(self.trajectory) >= TRACK_TRAJECTORY_LEN:
            self.trajectory.popleft()
        self.trajectory.append(self.centroid_normalized)
        if phone and not self.phone:
            self.phone_since = now
        self.phone = phone

    # ------------------------------------------------------------------
    # Identity voting (A6)
    # ------------------------------------------------------------------

    def register_vote(self, emp_id: str, score: float, now: float):
        """Record one frame's identity observation (consecutive-run voting)."""
        self._runs[emp_id] = self._runs.get(emp_id, 0) + 1
        for other in list(self._runs):
            if other != emp_id:
                self._runs[other] = 0
        bucket = self._scores.setdefault(emp_id, [])
        bucket.append(float(score))
        self._scores[emp_id] = bucket[-3:]
        self._decide(now)

    def _decide(self, now: float):
        known = {e: c for e, c in self._runs.items() if e != UNKNOWN_ID}
        if not known:
            # No positive identity in the current run: adopt Unknown on fresh
            # tracks, but NEVER demote an adopted identity (anti-flicker).
            if self.identity is None:
                self.identity = UNKNOWN_ID
                self.identity_updated = now
            return
        best = max(known, key=lambda e: (known[e], _mean(self._scores[e])))
        if best == self.identity:
            self.identity_score = round(_mean(self._scores[best]), 3)
            self.identity_updated = now
            return
        if self.identity is None or self.identity == UNKNOWN_ID:
            if known[best] >= self._adopt_frames:
                self._adopt(best, now)
            return
        # Switching from a KNOWN identity: require the new identity to hold a
        # consecutive run AND the previously adopted identity to have dropped
        # out completely (their person left this track).
        if known[best] >= self._switch_frames and self._runs.get(self.identity, 0) == 0:
            self._adopt(best, now)

    def _adopt(self, emp_id: str, now: float):
        old = self.identity
        self.identity = emp_id
        self.identity_score = round(_mean(self._scores[emp_id]), 3)
        self.identity_updated = now
        if TRACK_DEBUG:
            logger.info(
                "TRACK_DEBUG identity track=%s cam=%s %s -> %s score=%.3f",
                self.track_id, self.cam, old, emp_id, self.identity_score,
            )

    def phone_seconds(self, now: float) -> float:
        if not self.phone or self.phone_since is None:
            return 0.0
        return max(0.0, now - self.phone_since)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SpatialTracker:
    """Per-camera IoU person tracker with identity smoothing (Phase A A5/A6)."""

    def __init__(self, *, iou_threshold: float = TRACK_IOU_THRESHOLD,
                 max_age_sec: float = TRACK_MAX_AGE_SEC,
                 adopt_frames: int = IDENTITY_ADOPT_FRAMES,
                 switch_frames: int = IDENTITY_SWITCH_FRAMES,
                 debug: bool | None = None):
        self._iou_threshold = float(iou_threshold)
        self._max_age_sec = float(max_age_sec)
        self._adopt_frames = int(adopt_frames)
        self._switch_frames = int(switch_frames)
        self._debug = TRACK_DEBUG if debug is None else bool(debug)
        self._tracks: dict[str, Track] = {}
        self._next_id = 1
        self._last_persons_by_camera: dict[str, int] = {}
        logger.info(
            "SpatialTracker initialised (iou=%.2f max_age=%.1fs adopt=%d switch=%d)",
            self._iou_threshold, self._max_age_sec, adopt_frames, switch_frames,
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, detections: list[dict], now: float | None = None) -> list[Track]:
        """Advance tracking from one detection batch.  Returns active tracks."""
        now = time.time() if now is None else now

        persons_by_cam: dict[str, list[dict]] = {}
        faces_by_cam: dict[str, list[dict]] = {}
        for det in detections:
            cam = det.get("cam", "")
            if det.get("det_type") == "person" and det.get("person_box"):
                persons_by_cam.setdefault(cam, []).append(det)
            elif det.get("face_box") and det.get("emp_id"):
                faces_by_cam.setdefault(cam, []).append(det)

        self._last_persons_by_camera = {}
        for cam, p_dets in persons_by_cam.items():
            self._last_persons_by_camera[cam] = len(p_dets)
            self._associate_camera(cam, p_dets, faces_by_cam.get(cam, []), now)

        # Faces without any person detection on the camera this cycle can still
        # reinforce an existing track (identity vote).
        covered = set(persons_by_cam.keys())
        for cam, f_dets in faces_by_cam.items():
            if cam in covered:
                continue
            for fdet in f_dets:
                self._register_face_vote(cam, fdet, now)

        self.prune(now)
        return self.active_tracks(now)

    def _associate_camera(self, cam: str, p_dets: list[dict],
                          f_dets: list[dict], now: float):
        candidates = [t for t in self._tracks.values() if t.cam == cam]

        # Greedy IoU association: always attach the highest-IoU pair first.
        remaining_persons = list(p_dets)
        matched_tracks: set[str] = set()
        while remaining_persons and candidates:
            best_pair = None
            best_iou = None
            for t in candidates:
                if t.track_id in matched_tracks:
                    continue
                for pdet in remaining_persons:
                    iou = _iou(t.bbox, pdet["person_box"])
                    if best_iou is None or iou > best_iou:
                        best_iou = iou
                        best_pair = (t, pdet)
            if best_pair is None or best_iou < self._iou_threshold:
                break
            t, pdet = best_pair
            t.update(pdet["person_box"], pdet.get("person_conf", 1.0),
                     bool(pdet.get("phone", False)), now)
            matched_tracks.add(t.track_id)
            remaining_persons.remove(pdet)
            if self._debug:
                logger.info(
                    "TRACK_DEBUG update track=%s cam=%s bbox=%s phone=%s persons_left=%d",
                    t.track_id, cam, t.bbox, t.phone, len(remaining_persons),
                )

        # Unmatched persons spawn new tracks.
        for pdet in remaining_persons:
            track_id = f"{cam}#{self._next_id}"
            self._next_id += 1
            t = Track(track_id, cam, pdet["person_box"],
                      pdet.get("person_conf", 1.0),
                      pdet.get("frame_w", 0), pdet.get("frame_h", 0), now,
                      adopt_frames=self._adopt_frames,
                      switch_frames=self._switch_frames)
            t.phone = bool(pdet.get("phone", False))
            if t.phone:
                t.phone_since = now
            self._tracks[track_id] = t
            if self._debug:
                logger.info(
                    "TRACK_DEBUG new track=%s cam=%s bbox=%s phone=%s",
                    track_id, cam, t.bbox, t.phone,
                )

        # Identity votes from recognised faces on this camera.
        for fdet in f_dets:
            self._register_face_vote(cam, fdet, now)

    def _register_face_vote(self, cam: str, fdet: dict, now: float):
        t = self._face_track(cam, fdet)
        if t is None:
            return
        emp_id = fdet.get("emp_id") or UNKNOWN_ID
        t.register_vote(emp_id, fdet.get("face_score", 0.0), now)
        if self._debug and emp_id != UNKNOWN_ID:
            logger.info(
                "TRACK_DEBUG vote track=%s emp=%s score=%.3f -> identity=%s",
                t.track_id, emp_id, fdet.get("face_score", 0.0), t.identity,
            )

    def _face_track(self, cam: str, fdet: dict) -> Track | None:
        """Associate a face detection to the best track on the camera."""
        t_pool = [t for t in self._tracks.values() if t.cam == cam]
        if not t_pool:
            return None
        pbox = fdet.get("person_box")
        if pbox:
            t_pool = sorted(t_pool, key=lambda t: _iou(t.bbox, pbox), reverse=True)
            if _iou(t_pool[0].bbox, pbox) > 0.05:
                return t_pool[0]
        # Fall back: face centre inside an expanded track box.
        fw, fh = fdet.get("frame_w", 0) or 640, fdet.get("frame_h", 0) or 480
        fx1, fy1, fx2, fy2 = fdet["face_box"]
        fcx, fcy = ((fx1 + fx2) / 2.0) / fw, ((fy1 + fy2) / 2.0) / fh
        for t in t_pool:
            cx, cy = t.centroid_normalized
            bw, bh = (t.bbox[2] - t.bbox[0]) / fw, (t.bbox[3] - t.bbox[1]) / fh
            if abs(fcx - cx) <= bw and abs(fcy - cy) <= bh * 1.5:
                return t
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prune(self, now: float | None = None):
        now = time.time() if now is None else now
        dead = [tid for tid, t in self._tracks.items()
                if now - t.last_seen > self._max_age_sec]
        for tid in dead:
            t = self._tracks.pop(tid)
            if self._debug:
                logger.info(
                    "TRACK_DEBUG drop track=%s cam=%s age=%.1fs identity=%s",
                    tid, t.cam, now - t.first_seen, t.identity,
                )

    def active_tracks(self, now: float | None = None) -> list[Track]:
        now = time.time() if now is None else now
        return [t for t in self._tracks.values()
                if now - t.last_seen <= self._max_age_sec]

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def persons_by_camera(self) -> dict[str, int]:
        return dict(self._last_persons_by_camera)

    # ------------------------------------------------------------------
    # Serialisation (dashboard / live_state)
    # ------------------------------------------------------------------

    def snapshot(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        rows = []
        for t in self.active_tracks(now):
            rows.append({
                "track_id": t.track_id,
                "cam": t.cam,
                "identity": t.identity or UNKNOWN_ID,
                "identity_score": round(t.identity_score, 3),
                "phone": t.phone,
                "phone_sec": round(t.phone_seconds(now), 1),
                "bbox": [int(v) for v in t.bbox],
                "frame_w": t.frame_w,
                "frame_h": t.frame_h,
                "first_seen": round(t.first_seen, 2),
                "last_seen": round(t.last_seen, 2),
                "trajectory": [[float(x), float(y)] for x, y in t.trajectory],
            })
        return {
            "tracks": rows,
            "count": len(rows),
            "persons_by_camera": self.persons_by_camera(),
        }
