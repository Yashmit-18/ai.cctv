"""Central security engine (Phase 31).

Orchestrates all security subsystems per batch cycle:
    - Unknown presence detection + configurable rules
    - After-hours activity detection
    - Zone-based intrusion detection
    - Loitering detection
    - Occupancy / crowd detection
    - Camera offline/recovered transitions
    - Camera tamper detection

Coordinates events -> incidents -> alerts -> evidence.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import numpy as np

import config
from src import database as db
from src.alerts import AlertEngine
from src.auditlog import AuditLog
from src.camera_state import CameraStateManager
from src.domain import (
    AFTER_HOURS_ACTIVITY,
    ARRIVED,
    AWAY,
    CAMERA_OFFLINE,
    CAMERA_RECOVERED,
    CAMERA_TAMPER_CLEARED,
    CAMERA_TAMPER_SUSPECTED,
    DETECTOR_UNAVAILABLE,
    ENTRY_CROSSING,
    EXIT_CROSSING,
    INTRUSION,
    LOITERING_SUSPECTED,
    LEFT,
    SEV_INFO,
    SEV_LOW,
    SEV_MEDIUM,
    UNEXPECTED_STAY,
    UNUSUAL_OCCUPANCY,
    UNKNOWN_ID,
    UNKNOWN_PRESENCE,
    is_time_in_window,
)
from src.correlation import EventCorrelator
from src.failure_isolation import safe_call
from src.evidence import EvidenceStore
from src.incidents import IncidentEngine
from src.security_events import EventStore, SecurityEvent
from src.zones import ZoneStore

logger = logging.getLogger("cctv.security")


class SecurityEngine:
    """Per-batch security orchestrator.

    Call ``tick()`` from the daemon loop after ``detect_batch()``.
    """

    def __init__(self, conn, *, zone_store: ZoneStore | None = None,
                 event_store: EventStore | None = None,
                 incident_engine: IncidentEngine | None = None,
                 alert_engine: AlertEngine | None = None,
                 camera_state: CameraStateManager | None = None,
                 tamper_monitor=None,
                 evidence_store: EvidenceStore | None = None,
                 audit_log: AuditLog | None = None,
                 correlator: EventCorrelator | None = None,
                 correlation_window_sec: float | None = None):
        self._conn = conn
        self._zones = zone_store or ZoneStore(conn)
        self._events = event_store or EventStore(conn)
        self._incidents = incident_engine or IncidentEngine(conn)
        self._alerts = alert_engine or AlertEngine(conn)
        self._cam_state = camera_state or CameraStateManager(
            offline_trigger_sec=config.OFFLINE_TRIGGER_SEC)
        self._tamper = tamper_monitor
        self._evidence = evidence_store or EvidenceStore(
            conn, mode=config.EVIDENCE_MODE,
            evidence_dir=config.EVIDENCE_DIR,
            pre_sec=config.EVIDENCE_PRE_SEC,
            post_sec=config.EVIDENCE_POST_SEC,
            max_buf_frames=config.EVIDENCE_BUF_FRAMES,
            retention_days=config.EVIDENCE_RETENTION_DAYS,
            max_mb=config.EVIDENCE_MAX_MB,
        )
        self._audit = audit_log or AuditLog(conn)
        self._correlator = correlator or EventCorrelator(
            conn, incident_engine=self._incidents,
            window_sec=correlation_window_sec or config.CORRELATION_WINDOW_SEC)

        # Unknown presence tracking per camera
        self._unknown_first_seen: dict[str, float] = {}
        # Loitering tracking: (camera, zone) -> first-in-epoch
        self._zone_entry: dict[tuple[str, str], float] = {}
        # Occupancy tracking per zone
        self._zone_occupancy: dict[str, int] = {}
        self._last_occupancy_event: dict[str, float] = {}
        # Employee arrived/left tracking
        self._employee_first_seen: dict[str, float] = {}
        self._employee_last_seen: dict[str, float] = {}
        # Employees present in the previous tick (for LEFT emission, B6).
        self._present_prev: set[str] = set()
        # Rate limit for DETECTOR_UNAVAILABLE (last emission epoch).
        self._detector_down_last_event = 0.0

        # Phase A/B hardening (B2, B6, A10).
        # Intrusion cooldown per (camera, zone): while a person stays inside a
        # restricted zone, no INTRUSION event is re-emitted until the configured
        # cooldown elapses -- prevents DB/alert spam every detection tick.
        self._last_intrusion: dict[tuple[str, str], float] = {}
        # EXIT_CROSSING-based LEFT timestamps per employee (debounce against
        # the disappearance-derived LEFT so one departure yields one event).
        self._exit_left_at: dict[str, float] = {}
        # Line-crossing arrival bookkeeping for the opt-in UNEXPECTED_STAY
        # heuristic (B6): employee who crossed INTO the site but never crossed
        # back out before the office window ended.  Populated ONLY from
        # ENTRY/EXIT_CROSSING events -- never from generic appearance.
        self._crossing_arrivals: dict[str, float] = {}

        # Phase 33 -- optional advisory intelligence services (lazy-built,
        # isolated by safe_call in run_intelligence_cycle).
        self._intelligence = None
        self._intel_cycle = 0

    def tick(self, detections: list[dict], camera_health: dict,
             frames: dict | None = None, *,
             detector_unavailable: bool = False) -> list[dict]:
        """Run one security evaluation cycle.

        ``detections``: flat detection list from detector.detect_batch().
        ``camera_health``: camera_id -> health info dict.
        ``frames``: camera_id -> frame (for tamper/evidence).
        ``detector_unavailable``: True when the person-detection model is not
            running (loaded) this cycle.  When True the engine treats an empty
            ``detections`` list as *blind* rather than *empty premises* and
            emits a high-severity DETECTOR_UNAVAILABLE event so a person can
            never walk in during detector failure without an alert.

        Returns all events generated in this cycle.
        """
        if not config.SECURITY_ENABLED:
            return []

        all_events: list[dict] = []
        present_now: set[str] = set()
        now = time.time()
        clock_min = datetime.now().hour * 60 + datetime.now().minute
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- 1. Camera health transitions ---
        for cam_id, health_info in camera_health.items():
            # ONLINE/LOW_FPS = truly delivering usable content.  NO_FRAME,
            # FROZEN_FRAME, RECONNECTING and OFFLINE must be treated as NOT
            # online so a dead/stale/frozen feed becomes CAMERA_OFFLINE instead
            # of silently pretending the scene is empty-but-guarded.
            is_online = health_info.get("health") in ("ONLINE", "LOW_FPS")
            cam_events = self._cam_state.update(cam_id, is_online, now)
            for ev in cam_events:
                ev["timestamp"] = now_iso
                all_events.append(self._process_event(ev))

        # --- 2. Camera tamper detection ---
        if self._tamper and frames:
            for cam_id, frame in frames.items():
                if frame is not None:
                    tamper_events = self._tamper.analyze(cam_id, frame, now)
                    for ev in tamper_events:
                        ev["timestamp"] = now_iso
                        all_events.append(self._process_event(ev))

        # --- 3. Evidence frame feeding ---
        if frames:
            for cam_id, frame in frames.items():
                self._evidence.feed_frame(cam_id, frame)

        # --- 4. Per-detection security analysis ---
        person_count_by_camera: dict[str, int] = {}
        person_count_by_zone: dict[str, int] = {}
        person_present_by_camera: set[str] = set()

        # --- 4*. Fail-closed: detector down = blind, not empty premises. ---
        # With no model loaded the engine has no ground truth on person
        # presence.  Emit a high-severity DETECTOR_UNAVAILABLE event (rate
        # limited) instead of silently trusting an empty detection list, so a
        # person can never walk in unnoticed during a detector outage.  We do
        # NOT fabricate person presence from camera-online status here.
        if detector_unavailable:
            self._eval_detector_unavailable(now, now_iso, all_events)

        for det in detections:
            if det.get("emp_id") == "__person__":
                cam = det.get("cam", "")
                person_count_by_camera[cam] = person_count_by_camera.get(cam, 0) + 1
                person_present_by_camera.add(cam)
                continue

            emp_id = det.get("emp_id", UNKNOWN_ID)
            cam = det.get("cam", "")
            face_box = det.get("face_box")
            face_score = det.get("face_score", 0.0)

            # Recognized/unknown faces also indicate a person is present
            person_count_by_camera[cam] = person_count_by_camera.get(cam, 0) + 1
            person_present_by_camera.add(cam)

            # Camera must be online for zone/intrusion analysis
            cam_online = self._cam_state.is_online(cam)

            # --- 4a. Unknown presence ---
            if emp_id == UNKNOWN_ID:
                unknown_events = self._eval_unknown(cam, now, now_iso, face_score)
                all_events.extend(unknown_events)

            # --- 4b. Employee arrived/left ---
            if emp_id != UNKNOWN_ID:
                present_now.add(emp_id)
                self._eval_employee_presence(emp_id, now, now_iso, all_events)

            # --- 4c. Zone evaluation (only if camera online + person has position) ---
            if cam_online and face_box:
                x, y = _box_center_normalized(face_box, det)
                zone_hits = self._zones.evaluate_presence(
                    cam, x, y, emp_id, clock_min, now_iso)
                for hit in zone_hits:
                    zone = hit["zone"]
                    zname = zone.zone_name
                    person_count_by_zone[zname] = person_count_by_zone.get(zname, 0) + 1

                    if hit["intrusion"]:
                        # Track zone entry for loitering
                        key = (cam, zname)
                        if key not in self._zone_entry:
                            self._zone_entry[key] = now

                        # Gate repeated events: while the person stays inside,
                        # INTRUSION is re-emitted only after the configured
                        # cooldown (B2 hardening / anti-spam).
                        cooldown = max(
                            float(getattr(config, "INTRUSION_COOLDOWN_SEC", 0.0)), 0.0)
                        if now - self._last_intrusion.get(key, 0.0) >= cooldown:
                            self._last_intrusion[key] = now
                            intr_event = {
                                "event_type": INTRUSION,
                                "camera": cam,
                                "zone": zname,
                                "employee_id": emp_id,
                                "confidence": min(1.0, face_score) if face_score else 0.8,
                                "timestamp": now_iso,
                            }
                            all_events.append(self._process_event(intr_event))
                    else:
                        # Person is allowed; clear loitering entry
                        key = (cam, zname)
                        self._zone_entry.pop(key, None)

        # --- 4d. Loitering detection ---
        loiter_events = self._eval_loitering(now, now_iso)
        all_events.extend(loiter_events)

        # --- 4e. Occupancy detection ---
        occ_events = self._eval_occupancy(person_count_by_zone, now, now_iso)
        all_events.extend(occ_events)

        # --- 5. After-hours detection (only if people detected and camera online) ---
        if person_present_by_camera and not is_time_in_window(
                clock_min, config.SECURITY_OFFICE_START, config.SECURITY_OFFICE_END):
            for cam_id in person_present_by_camera:
                if self._cam_state.is_online(cam_id):
                    ah_event = {
                        "event_type": AFTER_HOURS_ACTIVITY,
                        "camera": cam_id,
                        "confidence": 0.9,
                        "timestamp": now_iso,
                    }
                    all_events.append(self._process_event(ah_event))

        # --- 6. Emit LEFT for employees present last cycle but absent now.
        # An EXIT_CROSSING already emitted LEFT (B6/A10); the appearance-based
        # fallback here is suppressed briefly so one departure yields one event.
        for emp_id in sorted(self._present_prev - present_now):
            if now - self._exit_left_at.get(emp_id, 0.0) < \
                    float(getattr(config, "LEFT_DEBOUNCE_SEC", 10.0)):
                continue
            all_events.append({
                "event_type": LEFT,
                "employee_id": emp_id,
                "confidence": 1.0,
                "timestamp": now_iso,
            })
        self._present_prev = present_now

        # --- 7. Opt-in B6 heuristic: crossed in via a line, never crossed out.
        self._eval_unexpected_stay(now, now_iso, clock_min, all_events)

        return all_events

    def _eval_detector_unavailable(self, now: float, now_iso: str,
                                   all_events: list[dict]) -> None:
        """Emit a rate-limited DETECTOR_UNAVAILABLE event (fail-closed)."""
        cooldown = max(getattr(config, "SECURITY_DETECTOR_COOLDOWN_SEC", 60.0), 5.0)
        if now - self._detector_down_last_event < cooldown:
            return
        self._detector_down_last_event = now
        all_events.append({
            "event_type": DETECTOR_UNAVAILABLE,
            "confidence": 1.0,
            "timestamp": now_iso,
        })

    def _eval_unknown(self, camera: str, now: float, now_iso: str,
                      face_score: float) -> list[dict]:
        """Evaluate unknown presence rules."""
        events: list[dict] = []
        mode = config.SECURITY_UNKNOWN_MODE

        if mode == "off":
            return events

        first = self._unknown_first_seen.get(camera)
        if first is None:
            self._unknown_first_seen[camera] = now
            first = now

        duration = now - first

        fire = False
        if mode == "immediate":
            fire = True
        elif mode == "after_seconds":
            fire = duration >= config.SECURITY_UNKNOWN_AFTER_SEC
        elif mode == "restricted_only":
            # Check if any restricted zone on this camera
            zones = self._zones.for_camera(camera)
            restricted = any(z.policy != "BYPASS" for z in zones)
            if restricted:
                fire = duration >= config.SECURITY_UNKNOWN_AFTER_SEC
        elif mode == "off_hours_only":
            clock_min = datetime.now().hour * 60 + datetime.now().minute
            if not is_time_in_window(clock_min, config.SECURITY_OFFICE_START,
                                     config.SECURITY_OFFICE_END):
                fire = duration >= config.SECURITY_UNKNOWN_AFTER_SEC

        if fire:
            # Cooldown / suppress repeats
            if config.SECURITY_UNKNOWN_SUPPRESS_REPEATS:
                last = self._events.events(
                    event_type=UNKNOWN_PRESENCE, camera=camera, limit=1)
                if last:
                    from datetime import datetime as dt
                    try:
                        last_ts = dt.strptime(last[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
                        elapsed = (dt.now() - last_ts).total_seconds()
                        if elapsed < config.SECURITY_UNKNOWN_COOLDOWN_SEC:
                            return events
                    except (ValueError, KeyError):
                        pass

            events.append({
                "event_type": UNKNOWN_PRESENCE,
                "camera": camera,
                "confidence": min(1.0, face_score) if face_score else 0.7,
                "duration_seconds": round(duration, 1),
                "timestamp": now_iso,
            })

        return events

    def _eval_employee_presence(self, emp_id: str, now: float, now_iso: str,
                                all_events: list[dict]) -> None:
        """Track employee first-seen / last-seen and emit ARRIVED/LEFT."""
        first = self._employee_first_seen.get(emp_id)
        if first is None:
            self._employee_first_seen[emp_id] = now
            all_events.append({
                "event_type": ARRIVED,
                "employee_id": emp_id,
                "confidence": 1.0,
                "timestamp": now_iso,
            })
        self._employee_last_seen[emp_id] = now

    def record_entry_exit(self, ev: dict, now: float | None = None) -> dict | None:
        """Persist a track-based line-crossing event (A10/B6).

        Call from the daemon with each event produced by
        :class:`~src.entry_exit.EntryExitDetector`.  Events are recorded
        through the normal security pipeline (incident binding + alerts +
        evidence).  An ``EXIT_CROSSING`` for a known employee also emits
        ``LEFT`` (and is bookkept so the appearance-based LEFT is debounced);
        ``ENTRY_CROSSING`` primes the opt-in UNEXPECTED_STAY heuristic.
        """
        ev_type = ev.get("event_type")
        if ev_type not in (ENTRY_CROSSING, EXIT_CROSSING):
            return None
        now = time.time() if now is None else now
        # Timestamp must be resolved before any derived event (e.g. the LEFT
        # for an EXIT_CROSSING) is persisted, or the DB NOT NULL constraint
        # on security_events.timestamp trips.
        ev.setdefault("timestamp", time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(now)))

        emp_id = ev.get("employee_id")
        if emp_id:
            if ev_type == ENTRY_CROSSING:
                self._crossing_arrivals.setdefault(emp_id, now)
            elif ev_type == EXIT_CROSSING:
                self._crossing_arrivals.pop(emp_id, None)
                self._exit_left_at[emp_id] = now
                left_ev = {
                    "event_type": LEFT,
                    "employee_id": emp_id,
                    "camera": ev.get("camera"),
                    "confidence": ev.get("confidence", 1.0),
                    "timestamp": ev.get("timestamp"),
                }
                self._process_event(left_ev)

        ev.setdefault("track_id", ev.get("track_id"))
        return self._process_event(ev)

    def _eval_unexpected_stay(self, now: float, now_iso: str, clock_min: int,
                              all_events: list[dict]) -> None:
        """Opt-in B6: crossed in but never crossed out before the window ended."""
        if not getattr(config, "UNEXPECTED_STAY_ENABLED", False):
            return
        if is_time_in_window(clock_min, config.SECURITY_OFFICE_START,
                             config.SECURITY_OFFICE_END):
            return
        for emp_id, arrived in list(self._crossing_arrivals.items()):
            # One LOW-severity event per employee-session; never accusatory.
            self._crossing_arrivals.pop(emp_id, None)
            all_events.append(self._process_event({
                "event_type": UNEXPECTED_STAY,
                "employee_id": emp_id,
                "confidence": 0.5,
                "duration_seconds": round(now - arrived, 1),
                "timestamp": now_iso,
            }))

    def _eval_loitering(self, now: float, now_iso: str) -> list[dict]:
        """Check zone entries for loitering."""
        events: list[dict] = []
        for (cam, zone_name), entry_time in list(self._zone_entry.items()):
            duration = now - entry_time
            if duration >= config.LOITERING_SEC:
                events.append({
                    "event_type": LOITERING_SUSPECTED,
                    "camera": cam,
                    "zone": zone_name,
                    "confidence": min(1.0, duration / (config.LOITERING_SEC * 2)),
                    "duration_seconds": round(duration, 1),
                    "timestamp": now_iso,
                })
                # Don't re-fire for 5 minutes
                self._zone_entry[(cam, zone_name)] = now + 300
        return events

    def _eval_occupancy(self, person_count_by_zone: dict[str, int],
                        now: float, now_iso: str) -> list[dict]:
        """Check zone occupancy against crowd threshold."""
        events: list[dict] = []
        for zone_name, count in person_count_by_zone.items():
            if count > config.CROWD_THRESHOLD:
                last = self._last_occupancy_event.get(zone_name, 0.0)
                if (now - last) >= config.OCCUPANCY_COOLDOWN_SEC:
                    events.append({
                        "event_type": UNUSUAL_OCCUPANCY,
                        "zone": zone_name,
                        "person_count": count,
                        "confidence": min(1.0, count / (config.CROWD_THRESHOLD * 2)),
                        "timestamp": now_iso,
                    })
                    self._last_occupancy_event[zone_name] = now
        return events

    def _process_event(self, ev: dict) -> dict:
        """Persist event, bind to incident, dispatch alerts, capture evidence."""
        from src.domain import DEFAULT_EVENT_SEVERITY
        ev_type = ev.get("event_type", "UNKNOWN")
        sev = (ev.get("severity") or DEFAULT_EVENT_SEVERITY.get(ev_type, "MEDIUM")).upper()
        ev["severity"] = sev

        # Record event
        event_id = self._events.record_dict(dict(ev))
        ev["event_id"] = event_id

        correlate_reason = None
        # Bind to an incident -- prefer a correlated open incident when this
        # event is genuinely related to one (person-threat family or a camera
        # lifecycle), otherwise use canonical dedup.
        if self._correlator:
            corr = self._correlator.correlate(ev, event_id)
            if corr:
                ev["incident_id"] = corr["incident_id"]
                correlate_reason = corr["reason"]
                inc_id = self._incidents.attach(
                    ev, event_id, correlation_reason=correlate_reason)
            else:
                inc_id = self._incidents.bind(ev)
        else:
            inc_id = self._incidents.bind(ev)
        ev["incident_id"] = inc_id

        # Back-link every event to its incident (investigation timeline) and
        # persist the correlation reason for explainable investigation.
        db.set_event_incident(self._conn, event_id, inc_id)
        db.link_incident_event(
            self._conn, inc_id, event_id, ev_type,
            ev.get("camera"), ev.get("zone"), correlate_reason,
        )

        # Structured log
        logger.info(
            "[SECURITY] %s camera=%s zone=%s severity=%s incident=%s confidence=%.2f",
            ev_type, ev.get("camera", "-"), ev.get("zone", "-"),
            sev, inc_id, ev.get("confidence", 0.0),
        )

        # Dispatch alerts (isolated: a notifier failure must not abort the tick)
        safe_call(self._alerts.evaluate, dict(ev))

        # Capture evidence for events -- mode-aware (B7).  The EvidenceStore
        # applies OFF / HIGH_SEVERITY_ONLY / EVENT_ONLY semantics internally, so
        # we must NOT hard-gate on HIGH+ here or EVENT_ONLY could never record
        # a low-severity event.  Isolated so a failure never aborts the tick.
        cam = ev.get("camera")
        if cam:
            safe_call(
                self._evidence.capture,
                inc_id, ev_type, sev, cam,
                metadata={"event": dict(ev)})

        return ev

    def run_intelligence_cycle(self) -> dict:
        """Phase 33 advisory intelligence refresh (temporal, anomaly, risk).

        Called from the daemon loop after ``tick()``.  Every step is isolated so
        a failure in one advisory subsystem can never break the main pipeline
        (graceful degradation).  Steps never touch productivity/employee data.

        Returns a dict of per-step results (intended for observability).
        """
        results: dict = {}
        try:
            from src.temporal_intelligence import TemporalIntelligence
            ti = TemporalIntelligence(
                self._conn,
                repeat_threshold=config.TEMPORAL_REPEAT_THRESHOLD,
                escalate_count=config.TEMPORAL_ESCALATE_COUNT,
                silence_days=config.TEMPORAL_SILENCE_DAYS,
            )
            results["temporal"] = len(ti.refresh_all(actor="daemon"))
        except Exception as exc:
            logger.warning("intelligence.temporal failed (isolated): %s", exc)
            results["temporal_error"] = str(exc)

        # Throttle the heavier advisory steps to every N ticks.
        self._intel_cycle += 1
        if self._intel_cycle % 60 == 0:
            try:
                from src.risk_scoring import IncidentRiskScorer
                if config.RISK_ENABLED:
                    scorer = IncidentRiskScorer(self._conn)
                    scored = 0
                    for inc in self._incidents.list(status="OPEN", limit=100):
                        scorer.score(inc["incident_id"], actor="daemon")
                        scored += 1
                    results["risk_scored"] = scored
            except Exception as exc:
                logger.warning("intelligence.risk failed (isolated): %s", exc)
                results["risk_error"] = str(exc)
            try:
                from src.anomaly_engine import AnomalyEngine, BaselineTracker
                ae = AnomalyEngine(
                    self._conn,
                    baselines=BaselineTracker(self._conn,
                                              min_samples=config.ANOMALY_MIN_SAMPLES),
                )
                results["patterns"] = len(ae.detect_recurring_patterns(actor="daemon"))
            except Exception as exc:
                logger.warning("intelligence.pattern failed (isolated): %s", exc)
                results["pattern_error"] = str(exc)
        return results

    def flush(self) -> None:
        """Finalize pending state on shutdown."""
        pass  # Evidence buffers are bounded by maxlen; no flush needed

    def snapshot(self) -> dict:
        """Return a summary dict for the dashboard."""
        return {
            "cameras_online": sum(
                1 for s in self._cam_state.all_states().values() if s["online"]),
            "cameras_offline": sum(
                1 for s in self._cam_state.all_states().values() if not s["online"]),
            "open_incidents": self._incidents.open_count(),
            "high_severity_open": self._incidents.high_severity_open_count(),
            "evidence_mode": config.EVIDENCE_MODE,
            "evidence_storage_mb": self._evidence.storage_used_mb(),
            "camera_states": self._cam_state.all_states(),
        }


def _box_center_normalized(face_box, detection) -> tuple[float, float]:
    """Extract normalized center point from a face/person box."""
    if not face_box or len(face_box) < 4:
        return 0.5, 0.5  # fallback: center of frame
    x1, y1, x2, y2 = face_box[:4]
    # face_box is in pixel coords; we don't know frame size here,
    # so return the box center as-is (caller should normalize if needed)
    # For zone evaluation, we use the detection's person box if available
    person_box = detection.get("person_box")
    if person_box and len(person_box) >= 4:
        px1, py1, px2, py2 = person_box[:4]
        cx = (px1 + px2) / 2
        cy = (py1 + py2) / 2
        # Normalize if we have frame dimensions
        fw = detection.get("frame_width", 640)
        fh = detection.get("frame_height", 480)
        return cx / fw, cy / fh
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    fw = detection.get("frame_width", 640)
    fh = detection.get("frame_height", 480)
    return cx / fw, cy / fh
