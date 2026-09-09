"""Deterministic simulation runner driving the REAL pipeline under load (Phase 36).

``SimulationRunner`` runs a scenario end-to-end against an isolated database
and the real ingestion/tracking/security/incident/evidence/alert machinery:

    VirtualCameraPool (frames + health)
                         │
        ScenarioPlan (detections, real detect_batch schema)
                         │
                         ▼
        MultiTracker.process_batch + SecurityEngine.tick   (REAL code)
                         │
                         ▼
        incidents / alerts / evidence / camera-health       (REAL code)

The runner never fabricates AI detections, never touches the production
database, and enforces configurable resource limits so it cannot DOS the host.
All timings are real wall-clock measurements.  Simulation data stays in the
isolated database and a temp evidence dir, never in production stores.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from src import database as db
from src.alerts import AlertEngine
from src.auditlog import AuditLog
from src.camera_state import CameraStateManager
from src.correlation import EventCorrelator
from src.evidence import EvidenceStore
from src.incidents import IncidentEngine
from src.security_engine import SecurityEngine
from src.tracker import MultiTracker
from src.zones import ZoneStore

from .report import SimulationReport, Timer, describe
from .scenario import ScenarioPlan
from .virtual_camera import ResourceLimitError, VirtualCameraPool

# Default resource limits (overridable) -- prevent host DOS.
DEFAULT_LIMITS: dict[str, int] = {
    "max_cameras": 20,
    "max_fps_total": 600,
    "max_steps": 200_000,
    "max_evidence_files": 500,
    "max_workers": 16,
}


def _count_rows(conn, table: str) -> int:
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except Exception:
        return 0


class SimulationRunner:
    """Runs one deterministic scenario against the real pipeline."""

    def __init__(self, *, scenario: str, seed: int = 42, limits: dict | None = None):
        self.scenario = scenario
        self.seed = seed
        self.plan = ScenarioPlan.for_scenario(scenario, seed=seed)
        self.limits = {**DEFAULT_LIMITS, **(limits or {})}
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._evidence_dir: str = ""
        self._started = False

    def _enforce(self, key: str, value: int) -> None:
        limit = int(self.limits.get(key, 0))
        if limit and value > limit:
            raise ResourceLimitError(
                f"{key} limit exceeded: {value} > {limit} (simulation aborted)")

    # ------------------------------------------------------------------
    # Setup / teardown (isolated DB + temp evidence)
    # ------------------------------------------------------------------
    def _setup(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="cctv_sim_")
        self._evidence_dir = os.path.join(self._tmpdir.name, "evidence")
        os.makedirs(self._evidence_dir, exist_ok=True)
        db_path = os.path.join(self._tmpdir.name, "sim.db")
        self.conn = db.get_connection(db_path)
        db.init_db(self.conn)
        self._started = True

    def _teardown(self) -> None:
        if self._started:
            try:
                self.conn.close()
            except Exception:
                pass
        if self._tmpdir:
            self._tmpdir.cleanup()
        self._started = False

    def __enter__(self) -> "SimulationRunner":
        self._setup()
        return self

    def __exit__(self, *exc) -> None:
        self._teardown()

    # ------------------------------------------------------------------
    # Evidence dir (for scenario Q: evidence generated during an incident)
    # ------------------------------------------------------------------
    @property
    def evidence_dir(self) -> str:
        return self._evidence_dir

    def run_steps(self, max_steps: int | None = None,
                  run_real_duration: float = 0.0,
                  on_step=None) -> SimulationReport:
        """Drive the scenario through the real pipeline.

        ``max_steps`` bounds the number of simulated frame steps.  Actual
        processing is bounded by ``run_real_duration`` (wall clock) so a
        "soak" run is still time-bounded; both are enforced.

        Returns a structured ``SimulationReport`` of genuinely measured data.
        """
        self._enforce("max_cameras", len(self.plan.cameras))
        self._enforce("max_fps_total", int(self.plan.fps) * len(self.plan.cameras))
        self._enforce("max_steps", max_steps or self.plan.steps)

        if not self._started:
            self._setup()

        pool = VirtualCameraPool.build(
            len(self.plan.cameras), seed=self.seed,
            default_fps=self.plan.fps, width=320, height=180,
        ).start()
        # wire camera-level disconnect faults from the scenario plan
        for cam in pool._cameras.values():  # noqa: SLF001
            for (s, e, kind) in self.plan.faults:
                cam.schedule_fault(s, e, kind)

        tracker = MultiTracker(self.conn)
        cam_state = CameraStateManager()
        evidence = EvidenceStore(
            self.conn, mode="EVENT_ONLY",
            evidence_dir=self._evidence_dir,
            pre_sec=0.5, post_sec=1.0, max_buf_frames=8,
            retention_days=30, max_mb=64,
        )
        incidents = IncidentEngine(self.conn)
        correlator = EventCorrelator(
            self.conn, incident_engine=incidents, window_sec=60)
        security = SecurityEngine(
            self.conn,
            camera_state=cam_state,
            evidence_store=evidence,
            incident_engine=incidents,
            correlator=correlator,
            alert_engine=AlertEngine(self.conn),
            audit_log=AuditLog(self.conn),
        )
        # seed one restricted zone for intrusion scenarios
        zones = ZoneStore(self.conn)
        if self._has_restricted(self.plan):
            zones.add(
                zone_name="restricted",
                camera="",
                polygon=[(0.70, 0.30), (0.85, 0.30), (0.85, 0.72), (0.70, 0.72)],
                allowed=["EMP999"],  # nobody allowed -> intrusion for anyone
                policy="INTRUSION",
            )

        report = SimulationReport(
            scenario=self.scenario, seed=self.seed,
            num_cameras=len(self.plan.cameras),
            duration_sec=0.0, fps_target=int(self.plan.fps),
        )
        report.faults_injected = [f"{k} cam:{s}-{e}" for (s, e, k) in self.plan.faults]

        batch_lat: list[float] = []
        detect_lat: list[float] = []
        db_lat: list[float] = []
        evidence_lat: list[float] = []
        alert_lat: list[float] = []

        steps = min(max_steps or self.plan.steps, self.plan.steps)
        start_wall = time.time()
        processed = 0
        dropped = 0
        errors: list[str] = []
        reconnect_total = 0

        for step in range(steps):
            if run_real_duration and (time.time() - start_wall) >= run_real_duration:
                break
            pool.step_cycle()
            frames = pool.latest_frames()
            dets = self.plan.detections_at_step(step)

            with Timer() as t:
                tracker.process_batch(dets, camera_online=bool(frames))
            batch_lat.append(t.elapsed_ms())

            health = pool.camera_health()
            reconnect_total += sum(cam.reconnect_count for cam in pool._cameras.values())

            with Timer() as t2:
                try:
                    events = security.tick(
                        detections=dets, camera_health=health, frames=frames)
                except Exception as exc:  # fail-safe per the real daemon
                    errors.append(f"step {step}: security.tick: {exc}")
                    events = []
            for ev in events:
                detect_lat.append(t2.elapsed_ms())
                break

            evidence_lat.append(_evidence_timing(evidence))
            alert_lat.append(_alert_timing(self.conn))
            db_lat.append(_db_timing(self.conn))

            if frames:
                processed += int(sum(1 for f in frames.values() if f is not None))
            dropped += max(0, len(self.plan.cameras) - len(frames))

            report.events_generated = _count_rows(self.conn, "security_events")
            report.incidents_generated = _count_rows(self.conn, "incidents")
            alert_count = _count_rows(self.conn, "alerts")
            report.alerts_generated = max(report.alerts_generated, alert_count)
            report.evidence_generated = _count_rows(self.conn, "evidence_files")

            self._enforce("max_evidence_files", report.evidence_generated)
            if report.evidence_generated > int(self.limits.get("max_evidence_files", 0)) \
                    and self.limits.get("max_evidence_files"):
                break

            if on_step:
                on_step(step)

        report.steps = steps
        report.processed_frames = processed
        report.dropped_frames = dropped
        report.reconnect_count = reconnect_total
        report.duration_sec = round(time.time() - start_wall, 6)
        report.errors = errors
        report.latency = {
            "batch_ms": describe(batch_lat),
            "pipeline_ms": describe(detect_lat),
            "database_ms": describe(db_lat),
            "evidence_ms": describe(evidence_lat),
            "alert_ms": describe(alert_lat),
        }
        report.resource = {
            "evidence_files_on_disk": _count_files(self._evidence_dir),
            "evidence_db_rows": report.evidence_generated,
            "threads": _thread_count(),
        }

        # Persist the real tracker interval state so activity_logs reflects
        # the actual state-machine outcome (authoritative productivity signal).
        try:
            tracker.flush_all()
        except Exception:
            pass

        report.final_state = self._final_state(tracker, security, self.conn)

        pool.stop()
        report.completed = (len(errors) == 0)
        return report

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _has_restricted(plan: ScenarioPlan) -> bool:
        return bool(plan.restricted_frames)

    @staticmethod
    def _final_state(tracker, security, conn=None) -> dict:
        state: dict = {"live_states": tracker.live_states()}
        try:
            state["live_durations"] = tracker.live_durations()
        except Exception:
            pass
        if conn is not None:
            try:
                state["activity_states"] = [
                    dict(r) for r in conn.execute(
                        "SELECT employee_id, state, duration FROM activity_logs").fetchall()]
            except Exception:
                state["activity_states"] = []
        return state

    def resource_url(self) -> str:
        return f"virtual:{self.scenario}"


def _count_files(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    total = 0
    for _root, _dirs, files in os.walk(path):
        total += len(files)
    return total


def _thread_count() -> int:
    try:
        import threading
        return threading.active_count()
    except Exception:
        return 0


def _db_timing(conn) -> float:
    with Timer() as t:
        try:
            conn.execute("SELECT COUNT(*) FROM security_events").fetchone()
        except Exception:
            pass
    return t.elapsed_ms()


def _evidence_timing(evidence: EvidenceStore) -> float:
    with Timer() as t:
        try:
            evidence.storage_used_mb()
        except Exception:
            pass
    return t.elapsed_ms()


def _alert_timing(conn) -> float:
    with Timer() as t:
        try:
            db.query_alerts(conn, limit=5)
        except Exception:
            pass
    return t.elapsed_ms()