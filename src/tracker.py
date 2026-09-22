"""Dynamic employee state tracker with smoothing buffer (Phase 6/7).

Split into two layers:

* ``EmployeeTracker``  -- single-employee FSM with a smoothing buffer that
  prevents rapid state-thrashing, camera-offline awareness, and wall-clock
  absence/phone thresholds so timing is exact regardless of frame rate.
* ``MultiTracker``     -- manages a collection of ``EmployeeTracker``
  instances dynamically (one per recognised employee), logs transitions to
  SQLite, tracks pre-commit live states for the HUD, deduplicates
  multi-camera detections, and provides ``flush_all()`` for shutdown.

States
------
  ACTIVE    -- face detected in frame, no phone detected
  ON_PHONE  -- face detected, phone is near face/body (continuously for more
               than ``PHONE_AFTER_SEC`` wall seconds)
  AWAY      -- the employee's own presence has been absent for more than
               ``AWAY_AFTER_SEC`` wall seconds (grace 0..3s).  Per employee.

Camera-offline semantics (CRITICAL)
-----------------------------------
The daemon passes ``camera_online`` into :meth:`process_batch`.  When *no*
camera is delivering frames the FSM is frozen -- employees are NOT advanced
to AWAY and no AWAY time is logged.  This prevents dead cameras from
silently becoming employee downtime.

All absence/phone timing is wall-clock based (see ``AWAY_AFTER_SEC`` /
``PHONE_AFTER_SEC``); frame counts are never used as a proxy for seconds.
Absence is decided per employee from that employee's own last observation,
so one employee leaving never drags another into AWAY.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from statistics import mean as _mean

from config import (
    AWAY_AFTER_SEC,
    IDENTITY_STABILITY_FRAMES,
    PHONE_AFTER_SEC,
    PHONE_GAP_GRACE_SEC,
    PRESENCE_PATIENCE_SEC,
    SMOOTHING_BUFFER_SEC,
    TRACK_DEBUG,
    TRACK_IOU_THRESHOLD,
    TRACK_MAX_AGE_SEC,
    TRACK_TRAJECTORY_LEN,
    IDENTITY_ADOPT_FRAMES,
    IDENTITY_SWITCH_FRAMES,
    REID_ADOPT_FRAMES,
)
from src.database import insert_security_event, log_interval
from src.domain import UNKNOWN_ID
from src.face_registry import RECOG_CANDIDATE, RECOG_CONFIRMED, RECOG_UNKNOWN

logger = logging.getLogger("cctv.tracker")

ACTIVE = "ACTIVE"
ON_PHONE = "ON_PHONE"
AWAY = "AWAY"

_VALID_STATES = {ACTIVE, ON_PHONE, AWAY}


def _fsm_seconds(name: str, default: float) -> float:
    """Read a live FSM knob from the ``config`` module (Phase 63).

    Thresholds are read lazily -- never bound at import -- so an Admin
    Control Center override applied by ``SettingsStore.apply_to_config()``
    before this tracker is constructed is honoured.  Any read/parse failure
    falls back to the env default captured at import.
    """
    try:
        import config as _cfg
        return float(getattr(_cfg, name, default))
    except (TypeError, ValueError):
        return default


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
    """Finite State Machine for one employee.

    State transitions are driven by **wall-clock** timestamps, never by frame
    counts (the pipeline is CPU-bound and FPS varies):

    * ``AWAY``      -- the employee's own last confirmed presence is more than
                     ``AWAY_AFTER_SEC`` in the past.  Grace: 0..3s of absence
                     keeps the previous state.  A camera being offline never
                     advances absence (state freezes instead).
    * ``ON_PHONE``  -- a phone associated with this employee has been detected
                     continuously (gaps under ``PHONE_GAP_GRACE_SEC`` ignored)
                     for more than ``PHONE_AFTER_SEC``.  Exactly one PHONE_USE
                     security event is emitted when the episode crosses the
                     threshold.

    State priority: absence beats phone use (a person who leaves the camera is
    eventually AWAY, never ON_PHONE forever).  Each employee has their own
    tracker, so one leaving never drags another into AWAY.
    """

    def __init__(self, conn, employee_id: str,
                 source: str = "",
                 *, away_after_sec: float | None = None,
                 phone_after_sec: float | None = None,
                 phone_gap_grace_sec: float | None = None):
        self._conn = conn
        self.employee_id = employee_id
        self.source = source or ""
        # Phase 63: FSM thresholds resolve LIVE from ``config`` (admin
        # override wins when explicitly passed) instead of import-time
        # constants, so Admin → Thresholds takes real effect.
        self._away_after_sec = \
            float(away_after_sec) if away_after_sec is not None \
            else _fsm_seconds("AWAY_AFTER_SEC", AWAY_AFTER_SEC)
        self._phone_after_sec = \
            float(phone_after_sec) if phone_after_sec is not None \
            else _fsm_seconds("PHONE_AFTER_SEC", PHONE_AFTER_SEC)
        self._phone_gap_grace_sec = \
            float(phone_gap_grace_sec) if phone_gap_grace_sec is not None \
            else _fsm_seconds("PHONE_GAP_GRACE_SEC", PHONE_GAP_GRACE_SEC)
        self._state: str | None = None
        self._candidate: str | None = None
        self._since: float = time.time()
        self._pending_since: float = time.time()
        self._consecutive: int = 0
        self._ever_observed = False
        self.last_seen: float | None = None
        self._phone_since: float | None = None
        self._phone_last: float | None = None
        self._phone_alerted: bool = False

    def process(self, status: dict, camera_online: bool = True,
                person_present: bool = False):
        """Advance the FSM with a new detector status dict.

        ``camera_online=False`` freezes the state machine: no state change
        and no time accounting, so a dead camera cannot fabricate AWAY time.
        ``person_present`` is accepted for backward compatibility but is no
        longer used to hold an employee in ACTIVE -- absence is decided per
        employee by wall-clock since that employee's own last presence.
        """
        now = time.time()

        if not camera_online:
            # No usable frames this cycle -- do not accrue AWAY or shift state.
            # Two actions keep offline time OUT of employee productivity:
            #   * advance the current interval's start so the in-flight row
            #     never swallows the offline gap (audited in flush / commit),
            #   * reset the pending timer and last-seen so a later return
            #     never commits immediately and absence only starts counting
            #     again once valid frames resume.
            if self._state is not None and self._since < now:
                self._since = now
            self._pending_since = now
            self._consecutive = 0
            self.last_seen = now
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
            if raw == ON_PHONE:
                self._phone_since = now
                self._phone_last = now
                self._phone_alerted = False
            return

        # ----- Wall-clock absence (this employee's own presence) ------------
        # Absence is measured from that employee's last confirmed observation
        # (0..AWAY_AFTER_SEC grace), independent of frame rate and independent
        # of whether any *other* person is still on screen.  A leaving
        # employee becomes AWAY even when colleagues remain visible.
        if raw == AWAY:
            if self.last_seen is not None and now - self.last_seen >= self._away_after_sec:
                if self._state != AWAY:
                    self._commit(AWAY, now)
                self._candidate = AWAY
                self._pending_since = now
                self._consecutive = 0
                self._close_phone_run(now)
            else:
                # Within grace: hold the previous state (e.g. ACTIVE) so the
                # dashboard does not flicker to AWAY on a short gap.
                self._pending_since = now
            return

        # ----- Phone episode tracking --------------------------------------
        # A phone run starts at the first ON_PHONE observation.  Brief gaps
        # (``PHONE_GAP_GRACE_SEC``) between phone observations do not end the
        # episode, so one dropped frame never resets the timer.
        if raw == ON_PHONE:
            if self._phone_since is None:
                self._phone_since = now
                self._phone_last = now
                self._phone_alerted = False
            else:
                self._phone_last = now
            if self._state != ON_PHONE and now - self._phone_since >= self._phone_after_sec:
                # Crossed the continuous >5s threshold: commit ON_PHONE and
                # emit exactly ONE event for this episode.
                self._commit(ON_PHONE, now)
                if not self._phone_alerted:
                    self._phone_alerted = True
                    self._emit_phone_event(now)
                self._candidate = ON_PHONE
                self._pending_since = now
                self._consecutive = 0
                return
            # While the phone has not yet crossed the threshold, the visible
            # state must stay ACTIVE (candidate is held, below).
        else:
            if self._phone_since is not None and self._phone_last is not None:
                if now - self._phone_last >= self._phone_gap_grace_sec:
                    self._close_phone_run(now)

        # ----- Identity smoothing for non-absent raw states -----------------
        # Before the phone threshold has elapsed, ON_PHONE evidence is treated
        # as ACTIVE so the dashboard never flashes ON_PHONE early.
        effective_raw = raw
        if raw == ON_PHONE and self._state != ON_PHONE:
            effective_raw = ACTIVE

        if effective_raw == self._state:
            self._candidate = effective_raw
            self._pending_since = now
            self._consecutive = 0
            return

        if effective_raw == self._candidate:
            self._consecutive += 1
        else:
            self._candidate = effective_raw
            self._consecutive = 1
            self._pending_since = now
            return

        # Frame-count stability floor (defends against fast identity flicker).
        if IDENTITY_STABILITY_FRAMES > 0 and self._consecutive < IDENTITY_STABILITY_FRAMES:
            return

        if now - self._pending_since >= SMOOTHING_BUFFER_SEC:
            if effective_raw != ON_PHONE:
                # Leaving a phone episode closes it -- a later phone use is a
                # brand-new episode (own alert) rather than a continuation.
                self._close_phone_run(now)
            self._commit(effective_raw, now)

    def _close_phone_run(self, now: float):
        self._phone_since = None
        self._phone_last = None
        self._phone_alerted = False

    def _emit_phone_event(self, now: float):
        """Persist ONE PHONE_USE security event for a completed episode.

        Reuses the existing security-events infrastructure (no parallel alert
        system).  Factual metadata only -- no embeddings, faces or credentials.
        """
        if self._conn is None:
            return
        phone_since = self._phone_since if self._phone_since is not None else now
        insert_security_event(
            self._conn,
            {
                "event_type": "PHONE_USE",
                "timestamp": _iso(now),
                "camera": self.source or None,
                "employee_id": self.employee_id,
                "severity": "INFO",
                "duration_seconds": round(now - phone_since, 2),
                "status": "NEW",
                "details": (
                    f"Continuous phone use crossed threshold "
                    f"{self._phone_after_sec:g}s"
                ),
            },
        )

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

    def __init__(self, conn, *, away_after_sec: float | None = None,
                 phone_after_sec: float | None = None,
                 phone_gap_grace_sec: float | None = None):
        self._conn = conn
        self._trackers: dict[str, EmployeeTracker] = {}
        # Phase 63: optional admin-derived FSM thresholds forwarded to every
        # tracker.  ``None`` keeps the env/live-config default.
        self._away_after_sec = away_after_sec
        self._phone_after_sec = phone_after_sec
        self._phone_gap_grace_sec = phone_gap_grace_sec
        logger.info("MultiTracker initialised (dynamic face-based mode).")

    def _ensure_tracker(self, emp_id: str, source: str = "") -> EmployeeTracker:
        if emp_id not in self._trackers:
            self._trackers[emp_id] = EmployeeTracker(
                self._conn, emp_id, source=source,
                away_after_sec=self._away_after_sec,
                phone_after_sec=self._phone_after_sec,
                phone_gap_grace_sec=self._phone_gap_grace_sec)
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

        # Unknown/unmatched detections must NEVER become an employee tracker:
        # they would otherwise accrue state and log productivity records for
        # a person we could not identify.  They still count for presence via
        # ``_person_detected`` above.
        known = {e for e in merged.keys() if e not in ("Unknown", "UNKNOWN", "__person__")}
        all_emp_ids = set(self._trackers.keys()) | known

        for emp_id in all_emp_ids:
            status = merged.get(emp_id, {"present": False, "phone_detected": False})
            source = {d.get("emp_id"): d.get("cam", "")
                      for d in detections if d.get("emp_id") != "__person__"}.get(emp_id, "")
            self._process_one(emp_id, status, camera_online, person_present, source)

    def _process_statuses(self, employees: dict, camera_online: bool,
                          person_present: bool):
        known = {e for e in employees.keys()
                 if e not in ("Unknown", "UNKNOWN", "__person__")}
        all_emp_ids = set(self._trackers.keys()) | known
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
#
# Phase 44 appearance identity policy (secondary, conservative):
#   * appearance votes are registered ONLY for strong matches (similarity
#     threshold AND ambiguity margin) -- weak evidence carries no weight;
#   * appearance may adopt, but only on a track face recognition has NOT
#     resolved (identity None or Unknown) and only after ``appear_adopt_frames``
#     consecutive strong votes;
#   * a face identity OUTRANKS appearance: a matched face upgrades authority
#     immediately and a different matched face takes over after its own adopt
#     window -- appearance can never veto a face;
#   * appearance can switch only between two appearance-sourced identities
#     (previous occupant's run dropped out), and it can never demote anything.
class Track:
    """One persistent spatial person instance (Phase A)."""

    def __init__(self, track_id: str, cam: str, bbox: tuple,
                 confidence: float, frame_w: int, frame_h: int, now: float,
                 adopt_frames: int | None = None,
                 switch_frames: int | None = None,
                 appear_adopt_frames: int | None = None,
                 phone_window: int = 0, phone_min: int = 0):
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
        # Where the adopted identity came from: "face" (highest authority),
        # "appearance" (secondary -- only picked when face gave nothing), or
        # None (unresolved).
        self.identity_source: str | None = None
        # Phase 40 numbered-Unknown display label (per-track lifecycle).  When
        # this track is unresolved (identity None/"Unknown") the number yields
        # a stable "Unknown N" display label that never depends on the order of
        # detections within a frame or on the detection-array index.  It is a
        # DISPLAY-ONLY label -- the canonical identity semantics stay untouched.
        self.unknown_label: int | None = None
        self.phone = False
        self.phone_since: float | None = None
        # Phase 59 -- desk/seat zone context.  ``zone`` is the stable desk zone
        # this track currently occupies (``camera_id:zone_id`` when inside a
        # configured polygon, else None).  CONTEXT ONLY -- it never influences
        # identity decisions (see src.seat_zones).
        self.zone: str | None = None
        self.trajectory: deque[tuple[float, float]] = deque()
        self._adopt_frames = int(adopt_frames) if adopt_frames is not None \
            else IDENTITY_ADOPT_FRAMES
        self._switch_frames = int(switch_frames) if switch_frames is not None \
            else IDENTITY_SWITCH_FRAMES
        self._app_adopt_frames = \
            int(appear_adopt_frames) if appear_adopt_frames is not None \
            else int(REID_ADOPT_FRAMES)
        # Face identity votes (highest authority) -- consecutive-run voting.
        self._face_runs: dict[str, int] = {}
        self._face_scores: dict[str, list[float]] = {}
        # Phase 44B candidate votes -- tentative (>= FACE_CANDIDATE_THRESHOLD
        # but not CONFIRMED) face matches.  Kept in their OWN run/scores store
        # so a run of weak candidates can never reset the confirmed-face runs
        # and, via ``_decide``, can never overwrite or switch a trusted identity.
        self._cand_runs: dict[str, int] = {}
        self._cand_scores: dict[str, list[float]] = {}
        # Appearance identity votes (secondary) -- only STRONG matches are ever
        # registered by the caller, so weak evidence carries no identity weight.
        self._app_runs: dict[str, int] = {}
        self._app_scores: dict[str, list[float]] = {}
        # Phone evidence gate (Phase 44): reported phone = this frame's raw
        # observation AND at least ``_phone_min`` raw observations inside the
        # trailing ``_phone_window``.  ``phone_window <= 1`` keeps the legacy
        # immediate behaviour (used by old tests / callers that opt out).
        self._phone_window = int(phone_window)
        self._phone_min = int(phone_min)
        self._phone_evidence = deque(maxlen=self._phone_window) \
            if self._phone_window > 1 else None
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

    @property
    def footpoint_normalized(self) -> tuple[float, float]:
        """Bottom-centre of the bounding box, normalised to 0..1.

        Phase 59 desk/seat zones evaluate a person's *standing point* (where
        the person meets the desk/floor), not the visual box centre -- a seated
        person's box centre can drift above their seat while the footpoint
        stays anchored inside the desk zone.
        """
        x1, y1, x2, y2 = self.bbox
        fx = ((x1 + x2) / 2.0) / self.frame_w
        fy = y2 / self.frame_h
        return round(max(0.0, min(1.0, fx)), 4), round(max(0.0, min(1.0, fy)), 4)

    def update(self, bbox: tuple, confidence: float, phone: bool, now: float,
               reid_emp: str | None = None, reid_score: float = 0.0,
               reid_margin: float = 0.0, reid_strong: bool = False):
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
        self.observe(bool(phone), now)
        # Appearance identity vote: only STRONG matches are registered -- weak
        # appearance evidence can never add identity weight (Phase 44).
        if reid_emp and reid_emp != UNKNOWN_ID and reid_strong:
            self.register_appearance_vote(reid_emp, float(reid_score), now)

    def observe(self, raw_phone: bool, now: float):
        """Apply one raw phone observation through the evidence gate.

        With the Phase 44 gate engaged (``phone_window > 1``) a frame only
        *reports* phone when this frame is a raw detection AND at least
        ``phone_min`` raw detections occurred within the trailing window -- so
        one random false positive can never start a phone episode.  Recovery
        stays immediate because the report is gated by the current frame's raw
        value, not by a slow-decay smoother.
        """
        if self._phone_evidence is not None:
            self._phone_evidence.append(bool(raw_phone))
            reported = bool(raw_phone) and \
                sum(self._phone_evidence) >= self._phone_min
        else:
            reported = bool(raw_phone)
        if reported and not self.phone:
            self.phone_since = now
        self.phone = bool(reported)

    # ------------------------------------------------------------------
    # Identity voting (A6 + Phase 44 appearance votes)
    # ------------------------------------------------------------------

    def register_vote(self, emp_id: str, score: float, now: float):
        """Record one frame's FACE identity observation (consecutive-run voting).

        Face is the highest-authority signal; appearance votes can never
        override it (see ``_decide``).
        """
        self._face_runs[emp_id] = self._face_runs.get(emp_id, 0) + 1
        for other in list(self._face_runs):
            if other != emp_id:
                self._face_runs[other] = 0
        bucket = self._face_scores.setdefault(emp_id, [])
        bucket.append(float(score))
        self._face_scores[emp_id] = bucket[-3:]
        self._decide(now)

    def register_appearance_vote(self, emp_id: str, score: float, now: float):
        """Record one STRONG appearance observation (consecutive-run voting).

        The caller (detector bridge) only invokes this for STRONG matches
        (similarity threshold + ambiguity margin), so weak/ambiguous appearance
        evidence never registers identity weight.
        """
        self._app_runs[emp_id] = self._app_runs.get(emp_id, 0) + 1
        for other in list(self._app_runs):
            if other != emp_id:
                self._app_runs[other] = 0
        bucket = self._app_scores.setdefault(emp_id, [])
        bucket.append(float(score))
        self._app_scores[emp_id] = bucket[-3:]
        self._decide(now)

    def register_candidate_vote(self, emp_id: str, score: float, now: float):
        """Record one frame's CANDIDATE face observation (Phase 44B).

        Candidates are tentative matches (>= ``FACE_CANDIDATE_THRESHOLD`` but
        not CONFIRMED).  Their runs live in a separate store so they never
        interact with confirmed-face/appearance runs; ``_decide`` lets a run of
        candidates give an identity ONLY to a track that trusts nobody yet --
        they can never demote, switch or override anyone.
        """
        self._cand_runs[emp_id] = self._cand_runs.get(emp_id, 0) + 1
        for other in list(self._cand_runs):
            if other != emp_id:
                self._cand_runs[other] = 0
        bucket = self._cand_scores.setdefault(emp_id, [])
        bucket.append(float(score))
        self._cand_scores[emp_id] = bucket[-3:]
        self._decide(now)

    def _decide(self, now: float):
        # ----- Face branch (highest authority) ------------------------------
        # Unchanged Phase A semantics: consecutive-run voting, adopt on fresh
        # tracks, never demote an adopted identity, switches only after the
        # previous identity's run drops away entirely.
        face_known = {e: c for e, c in self._face_runs.items() if e != UNKNOWN_ID}
        if not face_known:
            # No positive identity in the current face run: adopt Unknown on
            # fresh tracks, but NEVER demote an adopted identity.
            if self.identity is None:
                self.identity = UNKNOWN_ID
                self.identity_updated = now
        else:
            best = max(face_known,
                       key=lambda e: (face_known[e], _mean(self._face_scores[e])))
            if best == self.identity:
                self.identity_score = round(_mean(self._face_scores[best]), 3)
                self.identity_updated = now
                if self.identity_source == "candidate":
                    # A CONFIRMED face for the candidate-held identity promotes
                    # it to full face authority so it gains the upgrade path
                    # and face-run switch semantics (Phase 44B).
                    self.identity_source = "face"
                return
            if self.identity is None or self.identity == UNKNOWN_ID:
                if face_known[best] >= self._adopt_frames:
                    self._adopt(best, now, "face")
                return
            if self.identity_source == "appearance":
                # Phase 44: a matched face OUTRANKS an appearance identity.
                # Same person -> immediate authority upgrade; different person
                # -> face's own adopt window (appearance yields when face
                # disagrees).
                if best == self.identity or face_known[best] >= self._adopt_frames:
                    self._adopt(best, now, "face")
                return
            # Switching from a KNOWN face identity: the new identity must hold
            # a consecutive run AND the previously adopted identity must have
            # dropped out completely (their person left this track).
            if face_known[best] >= self._switch_frames and \
                    self._face_runs.get(self.identity, 0) == 0:
                self._adopt(best, now, "face")

        # ----- Candidate branch (tentative face, lowest authority) ----------
        # Phase 44B: constituents/ambiguous matches -- score >= candidate
        # threshold but NOT CONFIRMED -- may only give an identity to a track
        # that currently trusts nobody, and only after ``_adopt_frames``
        # consecutive candidate observations (temporal consistency).  A
        # candidate can refresh itself and later be promoted by a CONFIRMED
        # face; it can NEVER demote, switch or override a trusted identity.
        app_known = {e: c for e, c in self._app_runs.items() if e != UNKNOWN_ID}
        cand_known = {e: c for e, c in self._cand_runs.items() if e != UNKNOWN_ID}
        if cand_known and not face_known and self.identity in (None, UNKNOWN_ID):
            best = max(cand_known,
                       key=lambda e: (cand_known[e], _mean(self._cand_scores[e])))
            if cand_known[best] >= self._adopt_frames:
                self.identity = best
                self.identity_score = round(_mean(self._cand_scores[best]), 3)
                self.identity_updated = now
                self.identity_source = "candidate"
                if TRACK_DEBUG:
                    logger.info(
                        "TRACK_DEBUG candidate_adopt track=%s cam=%s -> %s "
                        "score=%.3f runs=%d",
                        self.track_id, self.cam, best, self.identity_score,
                        cand_known[best],
                    )
        elif self.identity_source == "candidate" and \
                self.identity in cand_known and \
                self.identity not in face_known and \
                self.identity not in app_known:
            # Keep refreshing the candidate-held identity; face/app runs must
            # stay empty so the candidate does not shadow a stronger signal.
            self.identity_score = round(_mean(self._cand_scores[self.identity]), 3)
            self.identity_updated = now

        # ----- Appearance branch (secondary, conservative) ------------------
        # Runs only when face recognition cannot (or no longer can) identify
        # this track.  It can adopt, refresh, and -- strictly between two
        # appearance-sourced identities -- switch, but it can never touch a
        # face-sourced identity and never demotes anything.
        if not app_known:
            return
        app_best = max(app_known,
                       key=lambda e: (app_known[e], _mean(self._app_scores[e])))
        if self.identity is not None and self.identity != UNKNOWN_ID:
            if self.identity_source != "appearance":
                # Face-owned identity is immune to appearance evidence.
                return
            if app_best == self.identity:
                self.identity_score = round(_mean(self._app_scores[app_best]), 3)
                self.identity_updated = now
            elif app_known[app_best] >= self._switch_frames and \
                    self._app_runs.get(self.identity, 0) == 0:
                # Two appearance-sourced identities traded places on this track
                # (previous occupant genuinely left).
                self._adopt(app_best, now, "appearance")
            return
        if app_known[app_best] >= self._app_adopt_frames:
            self._adopt(app_best, now, "appearance")

    def _adopt(self, emp_id: str, now: float, source: str):
        old = self.identity
        self.identity = emp_id
        scores = self._face_scores if source == "face" else self._app_scores
        self.identity_score = round(_mean(scores[emp_id]), 3)
        self.identity_updated = now
        self.identity_source = source
        if TRACK_DEBUG:
            logger.info(
                "TRACK_DEBUG identity track=%s cam=%s %s -> %s source=%s score=%.3f",
                self.track_id, self.cam, old, emp_id, source, self.identity_score,
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
                 debug: bool | None = None,
                 phone_window: int = 0, phone_min: int = 0,
                 appear_adopt_frames: int = REID_ADOPT_FRAMES):
        self._iou_threshold = float(iou_threshold)
        self._max_age_sec = float(max_age_sec)
        self._adopt_frames = int(adopt_frames)
        self._switch_frames = int(switch_frames)
        self._debug = TRACK_DEBUG if debug is None else bool(debug)
        # Phase 44 phone evidence gate.  Explicitly requested by the daemon
        # pipeline (config values); defaults of 0/0 keep the legacy immediate
        # phone semantics for existing callers.
        self._phone_window = int(phone_window)
        self._phone_min = int(phone_min)
        self._appear_adopt_frames = int(appear_adopt_frames)
        self._tracks: dict[str, Track] = {}
        self._next_id = 1
        self._last_persons_by_camera: dict[str, int] = {}
        # Wall-clock of the most recent :meth:`process` call.  Used by
        # :meth:`claims` to distinguish tracks *updated this cycle* (fresh YOLO
        # person box) from tracks merely surviving within ``max_age_sec`` -- only
        # the former may vouch for an employee's presence (person != face).
        self._last_process_now: float | None = None
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
        self._last_process_now = now

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
        self._assign_unknown_labels(now)
        return self.active_tracks(now)

    def _assign_unknown_labels(self, now: float) -> list["Track"]:
        """Assign/refresh per-track numbered-Unknown display labels.

        Rules (Phase 40): the number is owned by the *spatial track*, never by
        the detection-array index, so a moving unresolved person keeps its
        ``Unknown N`` label; a track that momentarily disappears is preserved
        for ``max_age_sec`` and keeps its label; a track that adopts a KNOWN
        identity releases its number; a brand-new unresolved track receives
        the smallest number not currently in use by an active track, so it can
        never steal an active label.  Only tracks that are *conclusively*
        unresolved (identity has been decided to ``UNKNOWN_ID`` -- e.g. by an
        unmatched-face vote) are numbered; a pending track whose identity is
        still ``None`` (waiting to adopt a known face) is NOT numbered, so an
        about-to-be-known employee can never burn a label.  The canonical
        ``UNKNOWN_ID`` identity is untouched: ``Unknown N`` is purely a
        display label.
        """
        active = self.active_tracks(now)
        for t in active:
            if t.identity not in (None, UNKNOWN_ID) and t.unknown_label is not None:
                t.unknown_label = None
        unresolved = [t for t in active if t.identity == UNKNOWN_ID]
        used = {t.unknown_label for t in unresolved if t.unknown_label is not None}
        n = 1
        for t in unresolved:
            if t.unknown_label is not None:
                continue
            while n in used:
                n += 1
            t.unknown_label = n
            used.add(n)
        return active

    @staticmethod
    def _identity_label(identity: str | None, unknown_label: int | None) -> str:
        """Display label for a track identity.

        KNOWN identities display as the canonical employee id; unresolved
        tracks display as ``Unknown N`` (falling back to bare ``Unknown``
        before the tracker has issued a number).  Canonical identity is never
        replaced: this string is only for human display.
        """
        if identity is not None and identity != UNKNOWN_ID:
            return identity
        return f"Unknown {unknown_label}" if unknown_label else "Unknown"

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
            t.update(
                pdet["person_box"], pdet.get("person_conf", 1.0),
                bool(pdet.get("phone", False)), now,
                reid_emp=pdet.get("reid_emp"),
                reid_score=pdet.get("reid_score", 0.0),
                reid_margin=pdet.get("reid_margin", 0.0),
                reid_strong=bool(pdet.get("reid_strong", False)),
            )
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
                      switch_frames=self._switch_frames,
                      appear_adopt_frames=self._appear_adopt_frames,
                      phone_window=self._phone_window,
                      phone_min=self._phone_min)
            t.observe(bool(pdet.get("phone", False)), now)
            # Register the appearance vote on the CREATION frame too.  Face
            # votes already count on the frame a track is born (see
            # ``_register_face_vote`` below); without this a track would always
            # be one appearance vote behind its face-scored sibling, silently
            # inflating the effective adopt requirement by an extra frame.
            reid_emp = pdet.get("reid_emp")
            if reid_emp and reid_emp != UNKNOWN_ID and \
                    bool(pdet.get("reid_strong", False)):
                t.register_appearance_vote(
                    reid_emp, float(pdet.get("reid_score", 0.0)), now)
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
        # Phase 44B two-tier gate: only CONFIRMED faces cast identity votes.
        # CANDIDATE faces register in the separate candidate store via
        # ``register_candidate_vote`` (adoption only on tracks that trust
        # nobody yet); UNKNOWN faces cast nothing.
        status = fdet.get("recog_status")
        if status is None:
            # Legacy caller / test fixture: emp_id is already the gated
            # identity (or "Unknown"), treat it as a confirmed decision.
            status = (RECOG_CONFIRMED
                      if fdet.get("emp_id") not in (None, UNKNOWN_ID)
                      else RECOG_UNKNOWN)
        if status == RECOG_CONFIRMED:
            emp_id = fdet.get("emp_id") or UNKNOWN_ID
            t.register_vote(emp_id, fdet.get("face_score", 0.0), now)
            if self._debug and emp_id != UNKNOWN_ID:
                logger.info(
                    "TRACK_DEBUG vote track=%s emp=%s score=%.3f -> identity=%s",
                    t.track_id, emp_id, fdet.get("face_score", 0.0), t.identity,
                )
            return
        cand_emp = fdet.get("recog_emp")
        if status == RECOG_CANDIDATE and cand_emp and cand_emp != UNKNOWN_ID:
            t.register_candidate_vote(
                cand_emp, float(fdet.get("recog_score", 0.0)), now)
            if self._debug:
                logger.info(
                    "TRACK_DEBUG candidate track=%s emp=%s score=%.3f "
                    "margin=%.3f -> identity=%s",
                    t.track_id, cand_emp, fdet.get("recog_score", 0.0),
                    fdet.get("recog_margin", 0.0), t.identity,
                )
            return
        # UNKNOWN face (below candidate floor or explicit Unknown identity):
        # keep the legacy "unknown person seen" semantics -- a fresh track
        # adopts UNKNOWN_ID (numbered label), a trusted identity is NEVER
        # demoted by it.
        t.register_vote(UNKNOWN_ID, fdet.get("face_score", 0.0), now)
        if self._debug:
            logger.info(
                "TRACK_DEBUG unknown track=%s score=%.3f -> identity=%s",
                t.track_id, fdet.get("face_score", 0.0), t.identity,
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

    def claims(self) -> dict[str, dict[str, object]]:
        """Per-person presence + phone claims from this cycle's person detections.

        Returns ``{emp_id: {"cam": str, "phone": bool}}`` for every spatial
        track that (a) was matched to a real YOLO person box in the *most
        recent* :meth:`process` call and (b) carries an ADOPTED known identity.

        This is the Phase 43 "person != face" bridge.  It lets the employee
        FSM hold ACTIVE -- and attribute phone use -- from *body* persistence
        alone, even when no face is visible (side/back pose).  A track that is
        merely surviving (not updated this cycle) never claims presence, so a
        person who actually left stops being claimed and absence recomputes
        normally.  An unresolved track (identity ``None`` or ``UNKNOWN_ID``)
        is never claimed: it never invents an identity, preserving the
        "new-and-never-identified stays Unknown" rule.  Phone is attributed
        from the one track that holds that identity -- strictly per person,
        never a global flag.
        """
        if self._last_process_now is None:
            return {}
        floor = self._last_process_now - 1e-9
        out: dict[str, dict[str, object]] = {}
        for t in self._tracks.values():
            if t.identity in (None, UNKNOWN_ID) or t.last_seen < floor:
                continue
            out[t.identity] = {"cam": t.cam, "phone": bool(t.phone)}
        return out

    # ------------------------------------------------------------------
    # Serialisation (dashboard / live_state)
    # ------------------------------------------------------------------

    def snapshot(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        rows = []
        unknown_count = 0
        for t in self.active_tracks(now):
            is_unknown = t.identity is None or t.identity == UNKNOWN_ID
            if is_unknown:
                unknown_count += 1
            rows.append({
                "track_id": t.track_id,
                "cam": t.cam,
                "identity": t.identity or UNKNOWN_ID,
                "identity_label": self._identity_label(t.identity, t.unknown_label),
                "identity_source": t.identity_source,
                "unknown_label": t.unknown_label,
                "identity_score": round(t.identity_score, 3),
                "phone": t.phone,
                "phone_sec": round(t.phone_seconds(now), 1),
                "bbox": [int(v) for v in t.bbox],
                "frame_w": t.frame_w,
                "frame_h": t.frame_h,
                "first_seen": round(t.first_seen, 2),
                "last_seen": round(t.last_seen, 2),
                "trajectory": [[float(x), float(y)] for x, y in t.trajectory],
                "zone": t.zone,
            })
        return {
            "tracks": rows,
            "count": len(rows),
            "unknown_count": unknown_count,
            "persons_by_camera": self.persons_by_camera(),
        }
