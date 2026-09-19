"""CCTV Employee Productivity Tracker -- Production Orchestrator (Phase 7).

Pipeline:
  Video Capture (MultiCameraManager pool)
    -> GPU Batch Inference (ActivityDetector.detect_batch)
    -> Multi-Tracker FSM w/ dedup (MultiTracker.process_batch)
    -> SQLite -> Excel -> Email.

CLI Flags
---------
  --source          Override the multi-camera pool with a single local source
                    (e.g. 'webcam' or '0'). Drives webcam index 0 instead of
                    the configured config.CAMERAS.
  --headless        Run without the cv2 GUI window (daemon mode)
  --device          Inference device (cuda | mps | cpu); auto-detect if omitted
  --max-batch N     Frames per GPU inference call (default from config, 4)
  --validate-config Check configuration and exit 0/1 without starting streams
  --test-email      Send a test report email and exit
  --no-email        Run with the EOD scheduler but never send emails

Hotkeys (live HUD only)
-----------------------
  q / ESC  Graceful shutdown: flush pending logs + save + email report.
  r        On-demand report snapshot (stream keeps running).
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np

import config
from config import (
    AUTO_BACKUP_DAILY,
    AUDIT_RETENTION_DAYS,
    BACKUP_HOUR,
    CAMERAS,
    CONF_THRESHOLD,
    DB_PATH,
    ENABLE_RETENTION,
    EOD_REPORT_HOUR,
    EVIDENCE_RETENTION_DAYS,
    MAX_BATCH_SIZE,
    MODEL_PATH,
    PROJECT_ROOT,
    REPORT_OUTPUT_DIR,
    RETENTION_DAYS_BIOMETRICS,
    RETENTION_DAYS_LOGS,
    RETENTION_DAYS_REPORTS,
    STARTUP_FRAME_WAIT_SEC,
    WORK_SCHEDULE,
    setup_logging,
    validate_config,
)
from src.camera_manager import MultiCameraManager
from src.database import (get_connection, init_db, insert_entry_exit_event,
                          insert_motion_event, retention_cleanup)
from src.domain import (  # noqa: F401 (re-exported for health constants)
    CAM_FROZEN,
    CAM_LOW_FPS,
    CAM_NO_FRAME,
    CAM_OFFLINE,
    CAM_ONLINE,
    CAM_RECONNECTING,
)
from src.detector import ActivityDetector, _resolve_device
from src.employees import EmployeeStore
from src.entry_exit import EntryExitDetector, load_lines
from src.motion import MotionDetector
from src.notifier import send_daily_report
from src.reporter import generate_daily_report
from src.security_engine import SecurityEngine
from src.tamper_monitor import TamperMonitor
from src.tracker import ACTIVE, AWAY, MultiTracker, ON_PHONE, SpatialTracker

logger = logging.getLogger("cctv.main")

WINDOW_NAME = "CCTV Productivity Monitor (Multi-Camera Face Recognition)"

STATE_COLORS: dict[str, tuple[int, int, int]] = {
    ACTIVE:    (0, 200, 0),
    ON_PHONE:  (0, 0, 230),
    AWAY:      (0, 210, 230),
}

_DASH_W = 320
_DASH_PAD = 10

_MONTAGE_COLS = 3
_MONTAGE_W = 480


# ======================================================================
# CLI parsing
# ======================================================================

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CCTV Employee Productivity Tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run as a headless background service (no GUI window).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu", "auto"],
        default="auto",
        help="Inference device. 'auto' picks the best available GPU/CPU.",
    )
    parser.add_argument(
        "--test-email",
        action="store_true",
        help="Send a test email via SMTP and exit.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override camera pool with a single local source. "
             "'webcam' (or a device index like '0', '1') drives that video "
             "device.  Other non-numeric values are treated as an alias for "
             "webcam 0 (e.g. --source laptop).  When set, 'config.CAMERAS' "
             "is ignored and a single-camera manager is used.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit (no camera/daemon start).",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="Never send email (report still generated on shutdown/EOD).",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=None,
        help="Override MAX_BATCH_SIZE for this run.",
    )
    return parser.parse_args(argv)


# ======================================================================
# HUD rendering
# ======================================================================

def _montage_grid(count: int) -> tuple[int, int]:
    """Return (cols, rows) for a camera-count-appropriate grid."""
    cols = 3 if count > 6 else 2
    rows = (count + cols - 1) // cols
    return cols, rows


def build_montage(cams_frame, labels, states, cam_health=None) -> "cv2.Mat | None":
    """Compose a tiled grid of the latest camera frames.

    ``labels`` maps camera_id -> dict with ``"face_boxes"`` and
    ``"face_labels"`` (raw-frame coordinates) drawn on each tile.
    ``states`` maps emp_id -> state for per-face HUD colours.
    ``cam_health`` optionally maps camera_id -> health-info dict to draw a
    status badge on each tile.
    """
    items = [(cid, f) for cid, f in cams_frame.items() if f is not None]
    if not items:
        return None
    cam_health = cam_health or {}

    cols, rows = _montage_grid(len(items))
    tw = _MONTAGE_W
    th = int(tw * 0.6)
    pad = 8
    canvas = np.zeros((rows * (th + pad), cols * (tw + pad), 3), dtype=np.uint8)

    health_colors = {
        "ONLINE": (0, 200, 0),
        "NO_FRAME": (0, 165, 255),
        "RECONNECTING": (0, 140, 255),
        "OFFLINE": (0, 0, 255),
    }

    for i, (cam_id, frame) in enumerate(items):
        r, c = divmod(i, cols)
        x0 = c * (tw + pad)
        y0 = r * (th + pad)
        scaled = cv2.resize(frame, (tw, th))
        canvas[y0:y0 + th, x0:x0 + tw] = scaled

        # Face overlays (frame-local coords -> tile-local coords)
        tile = labels.get(cam_id, {})
        for box, (emp_id, _score) in zip(
            tile.get("face_boxes", []), tile.get("face_labels", [])
        ):
            state = states.get(emp_id, AWAY)
            color = STATE_COLORS.get(state, (200, 200, 200))
            sx = tw / frame.shape[1]
            sy = th / frame.shape[0]
            bx = (int(box[0] * sx) + x0, int(box[1] * sy) + y0,
                  int(box[2] * sx) + x0, int(box[3] * sy) + y0)
            cv2.rectangle(canvas, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
            label_text = f"[{emp_id}: {state}]"
            cv2.putText(canvas, label_text, (bx[0], max(bx[1] - 6, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        cv2.putText(canvas, cam_id, (x0 + 6, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Health badge (top-right)
        info = cam_health.get(cam_id, {})
        health = info.get("health", "ONLINE")
        hcol = health_colors.get(health, (200, 200, 200))
        badge = f"{health}{' ' + str(info.get('reconnects', 0)) + 'x' if info.get('reconnects') else ''}"
        (tw_, th_) = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        bx2 = x0 + tw - tw_ - 6
        cv2.putText(canvas, badge, (bx2, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hcol, 1, cv2.LINE_AA)

    return canvas


def draw_dashboard(display, employee_ids, states, telemetry, fps, cam_online, cam_total):
    h, w = display.shape[:2]
    x0 = w - _DASH_W
    overlay = display.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    y = 28
    line_h = 26
    cx = x0 + _DASH_PAD

    cv2.putText(display, "DASHBOARD", (cx, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    y += line_h + 4
    cv2.line(display, (cx, y), (x0 + _DASH_W - _DASH_PAD, y), (100, 100, 100), 1)
    y += 8

    if not employee_ids:
        cv2.putText(display, "No employees detected", (cx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y += line_h

    for emp_id in employee_ids:
        state = states.get(emp_id, AWAY)
        color = STATE_COLORS.get(state, (200, 200, 200))
        t = telemetry.get(emp_id, {ACTIVE: 0.0, ON_PHONE: 0.0, AWAY: 0.0})

        cv2.putText(display, f"{emp_id}", (cx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2, cv2.LINE_AA)
        y += line_h - 4
        cv2.putText(display, f"  State: {state}", (cx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += line_h - 6
        cv2.putText(
            display,
            f"  Active: {t[ACTIVE]/3600:.2f}h  Phone: {t[ON_PHONE]/60:.1f}m  Away: {t[AWAY]/60:.1f}m",
            (cx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        y += line_h - 4
        cv2.line(display, (cx, y), (x0 + _DASH_W - _DASH_PAD, y), (60, 60, 60), 1)
        y += 8

    y += 4
    cv2.putText(display, f"FPS: {fps:.1f}  Cameras: {cam_online}/{cam_total}",
                (cx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y += line_h
    status_color = (0, 200, 0) if cam_online == cam_total and cam_total else (0, 165, 255)
    cv2.putText(display, f"Camera health: {cam_online} online / {cam_total - cam_online} degraded",
                (cx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)
    y += line_h
    cv2.putText(display, "q/ESC: quit  r: report", (cx, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)


# ======================================================================
# Live-state snapshot (consumed by the Streamlit dashboard)
# ======================================================================

_LIVE_STATE_FILE = os.path.join(config.PROJECT_ROOT, "data", "live_state.json")


# Phase 54 -- runtime visibility (M02).  A failed live-state write used to be
# swallowed silently, leaving the dashboard stale with no signal.  The writer
# now logs the first failure immediately (traceback) and then rate-limited
# while the outage persists; a successful write after failures logs the
# recovery once.  The writer is still best-effort: it never raises and never
# blocks the loop.
_LIVE_STATE_FAILURES = 0
_LIVE_STATE_RECOVERED = False
_LIVE_STATE_LAST_ERROR_AT = 0.0
_LIVE_STATE_ERROR_RATE_LIMIT_SEC = 30.0


def _write_live_state(states: dict, telemetry: dict, cam_health: dict, fps: float,
                      sources: dict | None = None, durations: dict | None = None,
                      security: dict | None = None, names: dict | None = None,
                      last_seen: dict | None = None,
                      last_detection_sec: float | None = None,
                      spatial: dict | None = None, motion: dict | None = None,
                      camera_selection: dict | None = None,
                      ai_capabilities: dict | None = None,
                      phone_diag: dict | None = None,
                      timing: dict | None = None,
                      reid: dict | None = None):
    """Persist a small JSON snapshot for the dashboard live view.

    Best-effort: never raises, never blocks.  On failure the previous snapshot
    is left intact (dashboard handles absence/staleness) and the error is
    logged loudly rather than swallowed (Phase 54 M02).

    When *names* and *last_seen* are provided the payload includes a rich
    ``employees`` mapping keyed by employee id.  Each entry carries the human
    name (resolved from the DB, never from the face filename), the current
    state, session duration, active/phone/away seconds, camera source, and
    the last time the employee was observed.  Employees on the roster who have
    never been observed report ``NOT_OBSERVED`` (never a fabricated AWAY/0s
    session).  The ``Unknown`` sentinel is excluded from this mapping -- the
    dashboard tracks it via ``states`` directly.
    """
    global _LIVE_STATE_FAILURES, _LIVE_STATE_RECOVERED, _LIVE_STATE_LAST_ERROR_AT
    try:
        all_states = states or {}
        all_telemetry = telemetry or {}
        all_sources = sources or {}
        all_durations = durations or {}
        name_map = names or {}
        seen = last_seen or {}

        # Build per-employee detail.  Unknown is tracked separately by the
        # dashboard (it never has a stable employee name or DB record).
        employees_detail: dict[str, dict] = {}
        if name_map is not None:
            # Collect all known employee ids: every tracker (including
            # Unknown) plus every name-map entry.
            all_emp_ids = set(all_states.keys()) | set(name_map.keys())
            for eid in sorted(all_emp_ids):
                if eid in ("Unknown", "UNKNOWN"):
                    continue
                emp_telem = all_telemetry.get(eid, {})
                st = all_states.get(eid)
                if not st:
                    # Never-observed roster member -> explicit NOT_OBSERVED;
                    # an employee observed this session but since departed is
                    # AWAY, never a fabricated blank state.
                    st = AWAY if seen.get(eid) else "NOT_OBSERVED"
                employees_detail[eid] = {
                    "employee_id": eid,
                    "name": name_map.get(eid, "") or eid,
                    "state": st,
                    "session_sec": all_durations.get(eid, 0.0),
                    "active_sec": emp_telem.get("ACTIVE", 0.0),
                    "phone_sec": emp_telem.get("ON_PHONE", 0.0),
                    "away_sec": emp_telem.get("AWAY", 0.0),
                    "source": all_sources.get(eid, ""),
                    "last_seen": seen.get(eid),
                }

        payload = {
            "updated": time.time(),
            "iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "states": all_states,
            "telemetry": all_telemetry,
            "sources": all_sources,
            "durations": all_durations,
            "last_seen": seen,
            "employees": employees_detail,
            "camera_health": cam_health,
            "fps": round(fps, 2),
            "last_detection_sec": last_detection_sec,
            "security": security or {},
            "ai": {
                "spatial_tracks": spatial or {},
                "motion": motion or {},
                "camera_selection": camera_selection or {},
                "capabilities": ai_capabilities or {},
                "phone_diag": phone_diag or {},
                "pipeline_timing": timing or {},
                "reid": reid or {},
            },
        }
        os.makedirs(os.path.dirname(_LIVE_STATE_FILE), exist_ok=True)
        tmp = _LIVE_STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, _LIVE_STATE_FILE)
    except Exception:
        _LIVE_STATE_FAILURES += 1
        now = time.time()
        # First failure always carries a traceback; repeats are rate-limited so
        # a persistent fault can never flood the log, only report degradation.
        rate_ok = (now - _LIVE_STATE_LAST_ERROR_AT) >= _LIVE_STATE_ERROR_RATE_LIMIT_SEC
        if _LIVE_STATE_FAILURES == 1 or rate_ok:
            _LIVE_STATE_LAST_ERROR_AT = now
            logger.error(
                "live-state write FAILED (%d consecutive; dashboard will go "
                "stale until this recovers); last good snapshot preserved",
                _LIVE_STATE_FAILURES,
                exc_info=(_LIVE_STATE_FAILURES == 1))
        _LIVE_STATE_RECOVERED = False
    else:
        if _LIVE_STATE_FAILURES > 0:
            logger.warning(
                "live-state write recovered after %d failures.",
                _LIVE_STATE_FAILURES)
        _LIVE_STATE_FAILURES = 0


def _maybe_log_pilot_metrics(cam_health: dict, fps: float, detector: "ActivityDetector | None"):
    """Phase 37 -- periodic operational health log for the advisory pilot.

    Only runs when PILOT_MODE is enabled.  Emits structured, human-readable
    operational metrics (camera online count, FPS, YOLO detector state, model
    device) to the log so operators can confirm the platform is healthy and
    that nothing is silently reduced to CPU / no-model during the pilot.
    """
    if not config.PILOT_MODE:
        return
    try:
        online = sum(1 for h in cam_health.values() if h.get("connected"))
        total = len(cam_health)
        det_state = "running"
        det_device = "n/a"
        if detector is not None:
            det_state = "running" if getattr(detector, "model", None) is not None else "no-model"
            det_device = str(getattr(detector, "device", "n/a"))
        log.info(
            "PILOT metrics: cameras=%s/%s fps=%.2f detector=%s device=%s",
            online, total, fps, det_state, det_device,
        )
    except Exception:  # pragma: no cover -- best-effort logging
        pass


def _local_camera_index(source: str) -> int:
    """Map ``--source`` values to a video device index.

    * ``webcam`` / ``local`` / ``laptop`` -> 0
    * a numeric string (``"0"``, ``"2"``) -> that index
    * anything else -> 0 (treated as a descriptive alias)

    ``auto`` (runtime probing) is handled by ``_auto_detect_camera_index``.
    """
    s = str(source).strip().lower()
    if s in ("webcam", "local", "laptop", "usb"):
        return 0
    if s.isdigit():
        return int(s)
    return 0


def _auto_detect_camera_index(max_probe: int = 5) -> int:
    """Probe local video devices ``0..max_probe-1`` and pick the first usable one.

    Local device indices shift across reboots and DroidCam reconnects (observed:
    index 0 returned only black frames while the live feed landed on index 1),
    so ``--source auto`` scans indices and returns the first that delivers a
    frame whose mean brightness is at least ``BLACK_FRAME_MEAN``.  Raises
    ``RuntimeError`` when no usable device is found.
    """
    for idx in range(max_probe):
        cap = cv2.VideoCapture(idx)
        try:
            if not cap.isOpened():
                logger.warning("Auto camera index: probe %d -> not open", idx)
                continue
            ok, frame = cap.read()
            ok, frame = cap.read() if ok else (ok, frame)
            if not ok or frame is None:
                continue
            mean = float(np.mean(frame))
            if mean >= config.BLACK_FRAME_MEAN:
                logger.info("Auto camera index: probe %d -> mean=%.1f (usable)", idx, mean)
                return idx
            logger.warning("Auto camera index: probe %d -> mean=%.1f (dark; skipping)",
                           idx, mean)
        finally:
            cap.release()
    raise RuntimeError(
        f"No usable camera index found in 0..{max_probe - 1}. "
        "Check DroidCam is connected and the lens is uncovered."
    )


def _wait_for_initial_frames(cams: "MultiCameraManager | None",
                             timeout: float) -> bool:
    """Wait up to ``timeout`` seconds for a local (webcam) source to connect.

    Returns True as soon as any camera reports a connected backend *or*
    delivers a frame.  Connection is the meaningful signal: opening a webcam
    can take several seconds on some machines, and a camera that can never
    open (busy / missing / permission-blocked) never reports connected, so
    the probe fails fast.  Used only for local sources -- RTSP streams open
    asynchronously and must not block a 30-second staggered rollout.
    """
    waited = 0.0
    step = 0.5
    while waited < timeout:
        if cams.active_camera_ids:
            return True
        if any(f is not None for f in cams.get_latest_batch().values()):
            return True
        time.sleep(step)
        waited += step
    return False


def _local_source_message(source: str) -> str:
    return (
        "Unable to open webcam. Check camera permissions or whether another "
        "application is using the camera.  Try a different device index:  "
        "  python main.py --source 1\n"
        "  or auto-detection:  python main.py --source auto"
    )


def _usable_camera_frames(frames: dict, health: dict[str, dict]) -> dict:
    """Drop frames that must not drive employee state.

    A camera can deliver *fresh* frames that are still unusable for honest
    presence:

    * health is OFFLINE / RECONNECTING / NO_FRAME -> nothing fresh to read;
    * the frame is blank/dark (mean brightness below ``BLACK_FRAME_MEAN``) --
      a disconnected DroidCam or covered lens must NEVER produce AWAY time.

    ``FROZEN_FRAME`` (content unchanged for a sustained period) and
    ``LOW_FPS`` stay usable: a person sitting still keeps their ACTIVE state,
    and FROZEN describes scene motion, not a broken camera.  Usability is
    annotated back onto the health snapshot (``usable`` / ``frame_mean`` /
    ``unusable_reason``) so the dashboard can explain the state honestly.
    """
    black_mean = config.BLACK_FRAME_MEAN
    good_health = {CAM_ONLINE, CAM_LOW_FPS, CAM_FROZEN}
    usable: dict = {}
    for cid, frame in list(frames.items()):
        h = health.get(cid) or {}
        if h.get("health") not in good_health:
            h["usable"] = False
            h["unusable_reason"] = h.get("health") or CAM_NO_FRAME
            continue
        if frame is None:
            h["usable"] = False
            h["unusable_reason"] = CAM_NO_FRAME
            continue
        mean: float | None = None
        if black_mean > 0:
            try:
                mean = float(np.mean(frame))
            except Exception:
                mean = None
        if mean is not None:
            h["frame_mean"] = round(mean, 2)
        if mean is not None and mean < black_mean:
            h["usable"] = False
            h["unusable_reason"] = "DARK_BLANK_FRAME"
            continue
        h["usable"] = True
        usable[cid] = frame
    return usable


def _print_demo_banner(cameras: dict, device: str, enrolled: int,
                       face_images: int, target_fps: int) -> None:
    """Print a concise, demo-friendly startup block (webcam mode only).

    Deliberately free of secrets: ``cameras`` values are device indices in
    this mode, never URLs.
    """
    cam_id = next(iter(cameras))
    index = cameras[cam_id]
    enrolled_label = enrolled if enrolled else "0"
    print("-" * 52)
    print("CCTV Employee Productivity Tracker -- DEMO MODE")
    print("-" * 52)
    print(f"Camera:            {cam_id}")
    print(f"Video device:      {index}")
    print(f"Inference device:  {device}")
    print(f"Target FPS:        {target_fps}")
    print(f"Employees enrolled:{enrolled_label}  (face images on disk: {face_images})")
    if not enrolled:
        print("Enrollment:  none enrolled yet -- add photos in data/faces and")
        print("             run:  python -m src.face_registry")
    print("-" * 52)
    print("Live. Press Ctrl+C to stop gracefully.  Dashboard: streamlit run app.py")


def _count_face_images() -> int:
    try:
        return len([p for p in Path(config.FACES_DIR).glob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    except OSError:
        return 0


# ======================================================================
# Report + email helpers
# ======================================================================

_EOD_STATE_FILE = os.path.join(config.PROJECT_ROOT, "data", "eod_state.json")


def _load_eod_state() -> dict:
    try:
        with open(_EOD_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_eod_state(state: dict):
    try:
        os.makedirs(os.path.dirname(_EOD_STATE_FILE), exist_ok=True)
        with open(_EOD_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except OSError as exc:
        logger.warning("Could not persist EOD state: %s", exc)


def _report_done_for(day: str, email: bool) -> bool:
    state = _load_eod_state()
    entry = state.get(day, {})
    if email:
        return bool(entry.get("email_sent"))
    return bool(entry.get("report_generated")) and bool(entry.get("email_sent", False))


def _mark_report_done(day: str, generated: bool, email: bool):
    state = _load_eod_state()
    entry = state.setdefault(day, {})
    if generated:
        entry["report_generated"] = True
    if email:
        entry["email_sent"] = True
    _save_eod_state(state)


_LAST_RETENTION_DAY: str | None = None


def _maybe_run_retention(conn):
    """Opt-in retention: runs at most once per day.

    Never deletes unless ``CCTV_ENABLE_RETENTION=1`` -- the daemon must not
    silently destroy history.
    """
    global _LAST_RETENTION_DAY
    if not ENABLE_RETENTION:
        return
    today = date.today().isoformat()
    if _LAST_RETENTION_DAY == today:
        return
    _LAST_RETENTION_DAY = today

    try:
        removed = retention_cleanup(conn, RETENTION_DAYS_LOGS)
        if removed:
            logger.info("Retention: purged %d activity rows older than %d days.",
                        removed, RETENTION_DAYS_LOGS)
    except Exception as exc:  # noqa: BLE001 - retention must never kill the loop
        logger.warning("Retention (logs) failed: %s", exc)

    try:
        cutoff = date.today().toordinal() - RETENTION_DAYS_REPORTS
        removed_reports = 0
        for name in os.listdir(REPORT_OUTPUT_DIR):
            if not name.endswith(".xlsx"):
                continue
            path = os.path.join(REPORT_OUTPUT_DIR, name)
            try:
                if date.fromtimestamp(os.path.getmtime(path)).toordinal() < cutoff:
                    os.remove(path)
                    removed_reports += 1
            except OSError:
                continue
        if removed_reports:
            logger.info("Retention: pruned %d report files older than %d days.",
                        removed_reports, RETENTION_DAYS_REPORTS)
    except OSError as exc:
        logger.warning("Retention (reports) skipped: %s", exc)

    # Security-domain retention (events/alerts/audit) -- incidences are never
    # auto-purged (they are the investigation record).
    try:
        from src.database import security_retention_cleanup
        removed_sec = security_retention_cleanup(
            conn,
            events_days=RETENTION_DAYS_LOGS,
            alerts_days=RETENTION_DAYS_LOGS,
            audit_days=AUDIT_RETENTION_DAYS,
        )
        if any(removed_sec.values()):
            logger.info("Security retention: purged %s.", removed_sec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Retention (security) skipped: %s", exc)

    # Evidence-file retention (file-based, within the evidence root only).
    try:
        from src.evidence import EvidenceStore
        ev_store = EvidenceStore(
            conn,
            evidence_dir=os.path.join(PROJECT_ROOT, "data", "evidence"),
            retention_days=EVIDENCE_RETENTION_DAYS,
        )
        removed_ev = ev_store.retention_cleanup()
        if removed_ev:
            logger.info("Evidence retention: removed %d dirs.", removed_ev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Retention (evidence) skipped: %s", exc)

    # Phase 54 -- biometric cache retention (M06).  Only the *rebuildable*
    # mean-embedding caches (``data/embeddings.pkl`` face, ``data/appearance.pkl``
    # ReID) are pruned when ``CCTV_RETENTION_BIOMETRICS_DAYS>0`` AND retention is
    # enabled.  Enrollment images under ``data/faces/`` / ``data/reid/`` are the
    # source of truth and are NEVER touched -- a cache removed here is rebuilt
    # from those images on the next build.
    if RETENTION_DAYS_BIOMETRICS > 0:
        try:
            cutoff = date.today().toordinal() - RETENTION_DAYS_BIOMETRICS
            pruned = []
            for cache_path in (config.EMBEDDINGS_FILE, config.APPEARANCE_EMBEDDINGS_FILE):
                if not os.path.isfile(cache_path):
                    continue
                try:
                    if date.fromtimestamp(os.path.getmtime(cache_path)).toordinal() < cutoff:
                        os.remove(cache_path)
                        pruned.append(os.path.basename(cache_path))
                except OSError:
                    continue
            if pruned:
                logger.warning(
                    "Retention (biometrics): pruned rebuildable caches older "
                    "than %d days: %s (enrollment images in data/faces and "
                    "data/reid are untouched and can rebuild them)",
                    RETENTION_DAYS_BIOMETRICS, ", ".join(pruned))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retention (biometrics) skipped: %s", exc)


_LAST_BACKUP_DAY: str | None = None


def _maybe_run_daily_backup():
    """Opt-in automatic backup: runs at most once per day (Phase 35 wiring).

    Only runs when ``CCTV_AUTO_BACKUP_DAILY=1``. Uses the existing WAL-safe
    ``backup_all()``. A single failed backup is logged and never kills the loop.
    """
    global _LAST_BACKUP_DAY
    if not AUTO_BACKUP_DAILY:
        return
    if datetime.now().hour != BACKUP_HOUR:
        return
    today = date.today().isoformat()
    if _LAST_BACKUP_DAY == today:
        return
    _LAST_BACKUP_DAY = today
    try:
        from src.backup_tool import backup_all
        results = backup_all(PROJECT_ROOT)
        logger.info("Auto-backup complete: %s", sorted(results.keys()))
    except Exception as exc:  # noqa: BLE001 - backup must never kill the loop
        logger.warning("Auto-backup failed: %s", exc)


def generate_and_maybe_email(conn, hint: str, email_on_complete: bool = False,
                             day: date | None = None):
    day = day or date.today()
    try:
        p = generate_daily_report(conn, day)
        # Phase 54 M07 -- report/EOD continuity.  eod_state must never claim a
        # day's report exists unless the xlsx is really on disk (the historical
        # 09-01 "EOD entry without a file" gap).  A report path that does not
        # exist is treated as a failed generation and left unmarked.
        if not (p and os.path.isfile(p)):
            logger.error("Report generation (%s) produced no file (%r); EOD "
                         "state for %s NOT marked done.", hint, p,
                         day.isoformat())
            return None
        logger.info("Report generated (%s) -> %s", hint, p)
        _mark_report_done(day.isoformat(), True, False)
        sent = False
        if email_on_complete:
            sent = send_daily_report(p)
            if sent:
                _mark_report_done(day.isoformat(), True, True)
                logger.info("EOD email sent for %s", day.isoformat())
            else:
                logger.warning("EOD email FAILED for %s (will retry on next cycle)",
                               day.isoformat())
        return p
    except Exception as err:
        logger.error("Report generation FAILED (%s): %s", hint, err)
        return None


def _reconcile_eod_state() -> dict:
    """Phase 54 M07 -- log report/EOD-state continuity mismatches.

    Read-only reconciler: it never fabricates a missing xlsx and never edits
    (or deletes) eod_state.  It surfaces two historical gap classes so an
    operator sees them instead of silent divergence:

    * a day marked ``report_generated`` in ``eod_state.json`` with no matching
      ``data/reports/daily_report_<day>.xlsx``;
    * an on-disk report with no ``eod_state`` entry (on-demand generation).

    Runs once at daemon startup.
    """
    state = _load_eod_state()
    xlsx_days: set[str] = set()
    try:
        for name in os.listdir(REPORT_OUTPUT_DIR):
            if name.startswith("daily_report_") and name.endswith(".xlsx"):
                xlsx_days.add(name[len("daily_report_"):-len(".xlsx")])
    except OSError as exc:
        logger.warning("EOD reconcile skipped (reports dir unreadable): %s", exc)
        return {"state_days": len(state), "xlsx_days": 0}
    for day in sorted(state):
        if state.get(day, {}).get("report_generated") and day not in xlsx_days:
            logger.warning(
                "EOD reconcile: eod_state marks %s report_generated but "
                "daily_report_%s.xlsx is missing (pruned or failed write); "
                "state kept as-is, file not fabricated.", day, day)
    for day in sorted(xlsx_days):
        if not state.get(day, {}).get("report_generated"):
            logger.info(
                "EOD reconcile: daily_report_%s.xlsx exists with no eod_state "
                "entry (on-demand run); a future run marks it.", day)
    logger.debug("EOD reconcile: %d state days, %d xlsx days.",
                 len(state), len(xlsx_days))
    return {"state_days": len(state), "xlsx_days": len(xlsx_days)}


# ======================================================================
# Background EOD scheduler (Phase 14 -- duplicates + restart safe)
# ======================================================================

class EODScheduler:
    """Daily report/email scheduler with duplicate prevention.

    Deterministic rules:
    * A report is generated exactly once per calendar day.
    * If the daemon starts after the configured EOD hour and that day's
      report was not already generated, it is generated immediately
      (missed-window recovery).
    * Email is sent at most once per day; restarting does not resend.
    """

    def __init__(self, conn_factory, enabled: bool = True, hour: int = 19,
                 email_on_complete: bool = False):
        self._conn_factory = conn_factory
        self._enabled = enabled
        self._hour = hour
        self._email_on_complete = email_on_complete
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self):
        if not self._enabled:
            logger.info("EOD scheduler disabled (EOD_REPORT_HOUR is None).")
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("EOD report scheduled at %02d:00 daily.", self._hour)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    @staticmethod
    def _day_done(day: date, email: bool) -> bool:
        state = _load_eod_state()
        entry = state.get(day.isoformat(), {})
        if email:
            return bool(entry.get("email_sent"))
        return bool(entry.get("report_generated"))

    def _run(self):
        import datetime as _dt

        while not self._stop_event.is_set():
            now = datetime.now()

            # -- Missed-window recovery on (re)start ----------------------
            # If we've passed today's EOD time and today's report is missing,
            # generate immediately.  Prevents silent data loss after restart.
            if (now.hour * 60 + now.minute) >= self._hour * 60 and not self._day_done(date.today(), self._email_on_complete):
                self._run_eod(date.today())

            target = now.replace(hour=self._hour, minute=0, second=0, microsecond=0)
            if target <= now:
                target += _dt.timedelta(days=1)
            wait_secs = (target - now).total_seconds()
            elapsed = 0.0
            while elapsed < wait_secs and not self._stop_event.is_set():
                chunk = min(5.0, wait_secs - elapsed)
                self._stop_event.wait(timeout=chunk)
                elapsed += chunk
            if self._stop_event.is_set():
                break

            conn = self._conn_factory()
            try:
                self._run_eod(date.today())
            finally:
                conn.close()

    def _run_eod(self, report_day: date):
        day = report_day.isoformat()
        if self._day_done(report_day, self._email_on_complete):
            logger.info("EOD report for %s already generated; skipping.", day)
            return
        conn = self._conn_factory()
        try:
            generate_and_maybe_email(
                conn, "eod-scheduled", email_on_complete=self._email_on_complete,
                day=report_day)
        finally:
            try:
                conn.close()
            except Exception:
                pass


# ======================================================================
# Main run loop
# ======================================================================

def _step_detector(detector, frames, load_detector,
                   reid_gate_boxes: dict[str, list] | None = None):
    """Load-or-detect for one frame-processing step.

    Always returns ``(detector, detections)`` with ``detections`` defined -- a
    guaranteed-safe empty list at minimum.  This covers the iteration where the
    detector is first (successfully) loaded: it does NOT skip ``detect_batch``
    on that same iteration (pre-fix, the first successful load left
    ``detections`` unbound, raising ``UnboundLocalError`` at
    ``tracker.process_batch``).

    ``load_detector`` is a callable returning a ready detector (or ``None`` on
    failure).  A detector-load failure keeps the FSM frozen (empty detections);
    a ``detect_batch`` failure is isolated and degrades to empty detections --
    neither fabricates employee absence.

    ``reid_gate_boxes`` (optional) is forwarded to
    ``detector.detect_batch(reid_gate_boxes=...)`` -- the opt-in event-driven
    ReID cadence (``CCTV_REID_GATE_RESOLVED=1``).
    """
    detections: list[dict] = []
    if detector is None:
        detector = load_detector()
    if detector is not None:
        try:
            detections = detector.detect_batch(
                frames, reid_gate_boxes=reid_gate_boxes)
        except Exception as err:
            logger.exception("Detector batch failed (isolated): %s", err)
            logger.warning("Detection degraded for this step; "
                           "retrying next batch. Employees stay "
                           "frozen -- no fake absence.")
            detections = []
    return detector, detections


def _merge_person_claims(detections: list[dict],
                         claims: dict[str, dict[str, object]]) -> list[dict]:
    """Attach spatial per-person presence+phone claims into the detection list.

    Phase 43 Part B/D bridge.  ``claims`` maps an ADOPTED spatial identity
    (from YOLO *body* persistence -- any pose, face visible or not) to its
    presence and phone flags.  Each claim becomes a minimal detection entry so
    ``MultiTracker.process_batch``/``EmployeeTracker`` see the employee as
    present and (when the specific track reports a phone) on-phone.

    Semantics preserved:
    * person != face -- presence comes from the body box, not a recognised face;
    * phone is attributed from the ONE track that holds that identity (per
      person, never a global ``phone_detected`` flag);
    * a claim never invents an identity (unresolved tracks are not claimed);
    * claims are additive -- the tracker dedup merges/ORs them with any
      face-derived statuses already present this cycle.
    """
    if not claims:
        return detections
    out = list(detections)
    for emp_id, claim in claims.items():
        out.append({
            "emp_id": emp_id,
            "cam": claim.get("cam", ""),
            "present": True,
            "phone": bool(claim.get("phone", False)),
            "spatial_fallback": True,
        })
    return out


def run(args: argparse.Namespace | None = None):
    args = args or parse_args()

    if args.validate_config:
        problems = validate_config(quiet=False)
        print("Configuration validation:", "OK" if not problems else "FAILED")
        for p in problems:
            print("  -", p)
        sys.exit(0 if not problems else 1)

    if args.test_email:
        test_report = generate_daily_report(get_connection(DB_PATH), date.today())
        sent = send_daily_report(test_report)
        sys.exit(0 if sent else 1)

    if args.max_batch is not None:
        if args.max_batch < 1:
            logger.error("--max-batch must be >= 1.")
            return
        batch_size = args.max_batch
    else:
        batch_size = MAX_BATCH_SIZE

    # Fail-fast on obvious config errors before starting streams.
    problems = validate_config()
    if problems:
        logger.error("Configuration problems detected; fix before running:")
        for p in problems:
            logger.error("  - %s", p)
        return

    device = None if args.device == "auto" else args.device
    resolved_device = _resolve_device(device)

    # -- Local single-source override ---------------------------------
    # If --source is provided (e.g. 'webcam' or '0'), bypass the configured
    # multi-camera pool and drive the requested video device as a single
    # logical camera.
    is_local = bool(args.source)
    # Phase 45: single local sources get the faster loop cap (see
    # config.effective_target_fps -- always overridable by CCTV_TARGET_FPS).
    _target_fps_eff = config.effective_target_fps(local_source=is_local)
    if args.source:
        raw_source = str(args.source).strip().lower()
        if raw_source == "auto":
            try:
                cam_index = _auto_detect_camera_index()
                logger.info("--source auto selected video device %d", cam_index)
            except RuntimeError as exc:
                logger.error(str(exc))
                print(_local_source_message(args.source))
                sys.exit(3)
            cameras: dict[str, str | int] = {"local_webcam": cam_index}
        else:
            cam_index = _local_camera_index(args.source)
            cameras = {"local_webcam": cam_index}
        camera_label = f"local source '{args.source}' (video device {cam_index})"
    elif not CAMERAS:
        logger.error(
            "No cameras configured. Set CCTV_CAM_<NN>_URL in .env for an RTSP "
            "pool, or run `main.py --source <index>` (e.g. 0) for a single "
            "local webcam, then validate with `python -m src.preflight` "
            "before retrying this run.")
        return
    else:
        cameras = CAMERAS
        camera_label = f"{len(CAMERAS)} cameras ({', '.join(CAMERAS)})"

    logger.info("Headless=%s | Device=%s | Batch=%d | %s",
                args.headless, resolved_device, batch_size, camera_label)

    cams = MultiCameraManager(cameras=cameras, max_batch=batch_size).start()
    if is_local and STARTUP_FRAME_WAIT_SEC > 0:
        if not _wait_for_initial_frames(cams, STARTUP_FRAME_WAIT_SEC):
            cams.stop()
            logger.error(_local_source_message(args.source))
            print(_local_source_message(args.source))
            sys.exit(3)
    conn = get_connection(DB_PATH)
    init_db(conn)
    _maybe_run_retention(conn)   # opt-in; applies once per day (see env)
    _reconcile_eod_state()        # Phase 54 M07 -- report/EOD continuity log
    tracker = MultiTracker(conn)

    # -- Phase A: spatial person tracking (A5/A6), motion (A8), entry/exit
    # (A10).  Strictly additive layers over the detector's per-person entries;
    # failures here degrade gracefully without touching the employee FSM.
    spatial = SpatialTracker(
        phone_window=config.PHONE_EVIDENCE_WINDOW,
        phone_min=config.PHONE_EVIDENCE_MIN,
        appear_adopt_frames=config.REID_ADOPT_FRAMES,
    )
    motion = MotionDetector(
        enabled=config.MOTION_ENABLED,
        min_score=config.MOTION_MIN_SCORE,
        min_duration_sec=config.MOTION_MIN_DURATION_SEC,
        cooldown_sec=config.MOTION_COOLDOWN_SEC,
        downscale=config.MOTION_DOWNSCALE,
    )
    entry_exit = EntryExitDetector(
        lines=load_lines(config.ENTRY_EXIT_LINES_FILE),
        debounce_sec=config.ENTRY_EXIT_DEBOUNCE_SEC,
    )

    # A13 -- camera selection telemetry (honest: config mode + resolved input).
    if is_local:
        camera_selection = {
            "mode": config.CAMERA_MODE,
            "source": raw_source,
            "camera_ids": list(cameras.keys()),
            "device_index": cam_index if isinstance(cam_index, int) else None,
        }
    else:
        camera_selection = {
            "mode": config.CAMERA_MODE,
            "source": "config.CAMERAS",
            "camera_ids": list(cameras.keys()),
        }
    employees = EmployeeStore(conn, WORK_SCHEDULE)
    employee_names: dict[str, str] = {
        e.employee_id: (e.name or "").strip()
        for e in employees.list()
        if e.employee_id not in ("Unknown", "UNKNOWN")
    }

    # -- Phase 31: Security engine -------------------------------------------
    tamper_mon = TamperMonitor(
        persist_sec=config.TAMPER_PERSIST_SEC,
        dark_mean=config.TAMPER_DARK_MEAN,
        bright_mean=config.TAMPER_BRIGHT_MEAN,
        frozen_diff=config.TAMPER_FROZEN_DIFF,
        low_std=config.TAMPER_LOW_STD,
        enable_dark=config.TAMPER_ENABLE_DARK,
    )
    security = SecurityEngine(conn, tamper_monitor=tamper_mon)
    from src.zones import seed_zones
    seed_zones(conn, security._zones, config.SECURITY_ZONES_FILE)
    from src.auditlog import AuditLog
    audit = AuditLog(conn)
    audit.record("daemon.start", detail=f"device={resolved_device} batch={batch_size}")

    if is_local:
        _print_demo_banner(
            cameras, resolved_device,
            enrolled=sum(1 for e in employees.list() if e.enrolled),
            face_images=_count_face_images(),
            target_fps=_target_fps_eff,
        )
        _resolution_reported = False

    detector = None

    # Detector factory for the per-step load-or-detect helper.  Constructs the
    # real ActivityDetector (YOLO + InsightFace) and reconciles the enrolled
    # flag; returns None (and keeps retrying next step) on any load failure.
    def _load_detector():
        try:
            d = ActivityDetector(
                model_path=MODEL_PATH,
                conf=CONF_THRESHOLD,
                device=resolved_device,
                face_detect_size=config.FACE_DETECT_SIZE,
                face_cadence=config.FACE_DETECT_CADENCE,
            )
            try:
                employees.sync_enrolled(d.face_registry.enrolled_employee_ids)
            except Exception:
                logger.debug("Enrolled-flag sync skipped (DB busy?)",
                             exc_info=True)
            return d
        except Exception as err:
            logger.error("Model load failed: %s", err)
            logger.error("Did you run: pip install ultralytics torch insightface ?")
            return None

    telemetry: dict[str, dict[str, float]] = {}
    last_state: dict[str, str | None] = {}

    last_frame_labels: dict[str, dict] = {}
    caps_registered = False
    registry = None

    fps_frames = 0
    fps_start = time.time()
    fps = 0.0
    last_processed = None
    dark_frames: dict[str, int] = {}
    step_count = 0
    last_detection_at = None
    # Part F: opt-in per-stage loop timing, smoothed as an EMA and published
    # into live_state["ai"]["pipeline_timing"] so dashboard lag is attributable
    # to the right stage.  Zero overhead when CCTV_PIPELINE_TIMING=0.
    timing_ema: dict[str, float] = {}
    timing_n = 0

    interval = MultiCameraManager.frame_interval(_target_fps_eff)

    scheduler = EODScheduler(
        conn_factory=lambda: get_connection(DB_PATH),
        enabled=EOD_REPORT_HOUR is not None,
        hour=EOD_REPORT_HOUR or 19,
        email_on_complete=not args.no_email,
    )
    scheduler.start()

    use_gui = not args.headless
    if use_gui:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # SecurityEngine and AuditLog are created above (lines 663/667) and must be
    # retained for the whole run. Phase 35: removed the erroneous `= None`
    # overwrite that previously nulled the security engine and audit log, which
    # disabled security processing in production.
    try:
        while True:
            # Part F timing anchors (opt-in).  t_capture/t_detect/t_state track
            # the major stages; live-state seconds are measured around the write.
            t_total = time.time()
            t_capture = t_total
            t_detect = t_capture
            t_state = t_detect
            # Phase 45: per-cycle itemized windows folded into the published
            # EMA when CCTV_PIPELINE_TIMING=1 (zero cost otherwise).
            t_track0 = t_track1 = t_capture
            t_db0 = t_db1 = t_capture
            t_events0 = t_events1 = t_capture
            # -- Round-robin throttle + usable-camera gating ----------------
            # Pull at most ``batch_size`` fresh frames, then drop any that must
            # not drive employee state: dead/no-frame health and blank/dark
            # content.  A dead/blank camera freezes states -- it never
            # fabricates AWAY time (and never reports "nobody present").
            health = cams.camera_health()
            frames = cams.latest_frames(limit=batch_size)
            frames = _usable_camera_frames(frames, health)
            camera_online = bool(frames)
            t_capture = time.time()

            # -- CAMERA_DEBUG: structured per-loop diagnostics ---------------
            # Rate-limited so the log stays readable; gated on the same
            # usable/ annotations the pipeline acted on.  Never logs creds.
            if config.CAMERA_DEBUG or config.DETECTION_DEBUG:
                step_count += 1
                if step_count % 5 == 1 or not camera_online:
                    for cid, h in health.items():
                        fr = frames.get(cid)
                        w = hgt = None
                        if fr is not None:
                            try:
                                hgt, w = fr.shape[:2]
                            except Exception:
                                pass
                        frame_age = 0.0
                        if h.get("last_frame"):
                            frame_age = time.time() - h["last_frame"]
                        mode = camera_selection.get("mode", config.CAMERA_MODE)
                        ridx = camera_selection.get("device_index")
                        ridx_s = str(ridx) if ridx is not None else h.get("source")
                        logger.info(
                            "CAMERA_DEBUG: cam=%s source=%s mode=%s resolved_index=%s "
                            "opened=%s frame_received=%s frame_width=%s "
                            "frame_height=%s frame_age=%.2f input_fps=%s "
                            "processing_fps=%.2f camera_online=%s usable=%s",
                            cid, h.get("source"), mode, ridx_s,
                            h.get("connected"), fr is not None, w, hgt,
                            frame_age, h.get("fps"), fps, camera_online,
                            h.get("usable"),
                        )

            # -- Dead/blank-feed watchdog ------------------------------------
            # Any camera stuck on DARK_BLANK_FRAME is surfaced loudly.  It is
            # the same signal that froze the states above, so the message
            # exactly matches what the pipeline did.
            usable_ids = {cid for cid, h in health.items() if h.get("usable")}
            for cid, h in health.items():
                if cid in usable_ids:
                    dark_frames[cid] = 0
                    continue
                if h.get("unusable_reason") != "DARK_BLANK_FRAME":
                    continue
                dark_frames[cid] = dark_frames.get(cid, 0) + 1
                n = dark_frames[cid]
                if n in (1, 15, 45, 90, 150, 300) or n % 600 == 0:
                    logger.warning(
                        "Camera %s is delivering only dark/blank frames "
                        "(mean %.1f < %.1f for ~%d frames). DroidCam may be "
                        "disconnected or the lens covered. This is not employee "
                        "absence and counts no AWAY time; employee states are "
                        "frozen until a usable frame arrives. Try `--source auto` "
                        "or `--source <index>`.",
                        cid, h.get("frame_mean", 0.0), config.BLACK_FRAME_MEAN, n,
                    )

            # Keep the security health FSM ticking even when every feed is
            # unusable.  Detection is deliberately skipped in that case, and
            # the employee tracker remains frozen below.
            detections = []
            tracks = []
            if frames:
                # -- Phase 49 OPT-IN event-driven ReID cadence ----------------
                # When CCTV_REID_GATE_RESOLVED=1 the appearance pass skips
                # person boxes already resolved to a face identity (previous
                # spatial cycle).  A face-resolved track can never be changed
                # by appearance votes, so that model inference is pure waste.
                # Default OFF: production behaviour is unchanged until the
                # real-camera latency measurements justify enabling it.
                reid_gate_boxes = None
                if config.REID_GATE_RESOLVED and spatial is not None:
                    reid_gate_boxes = {}
                    try:
                        for _t in spatial.active_tracks():
                            if getattr(_t, "identity", None) in (None, "Unknown"):
                                continue
                            if getattr(_t, "identity_source", None) != "face":
                                continue
                            bb = getattr(_t, "bbox", None)
                            cam = getattr(_t, "cam", None)
                            if bb is not None and cam is not None:
                                reid_gate_boxes.setdefault(cam, []).append(
                                    tuple(float(v) for v in bb))
                    except Exception:
                        logger.exception("reid gate lookup failed; running "
                                         "ungated this cycle")
                        reid_gate_boxes = None
                # Backend load-or-detect for this step.  Guarantees a defined
                # (possibly empty) `detections` list every iteration, including
                # the first successful detector load.  `detector` is rebound to
                # the returned value so a just-created detector persists.
                detector, detections = _step_detector(
                    detector, frames, _load_detector,
                    reid_gate_boxes=reid_gate_boxes)

                # -- Phase A (spatial) runs FIRST so per-person claims can feed
                # the employee FSM below.  Still strictly isolated: a failure
                # here only costs this cycle's claims, never the pipeline.
                try:
                    t_track0 = time.time()
                    spatial.process(detections)
                    tracks = spatial.active_tracks()
                except Exception as _st:
                    logger.warning("Spatial tracking skipped (isolated): %s", _st)

                # -- Part B/D bridge: body-level presence, not face-level. ----
                # A known employee whose face is hidden (side/back pose) still
                # has a fresh spatial track holding their adopted identity.
                # Merge that track's presence + phone into the detection list
                # BEFORE the FSM consumes it -- person != face, and phone is
                # attributed from that ONE track only (strictly per person).
                try:
                    claims = spatial.claims() if spatial is not None else {}
                except Exception:
                    claims = {}
                if claims:
                    detections = _merge_person_claims(detections, claims)

                tracker.process_batch(detections, camera_online=camera_online)
                t_track1 = time.time()

                # -- Phase A: motion + line crossings (advisory) ------------
                # Strictly additive/advisory; isolated so a Phase A failure can
                # never degrade the employee pipeline.  Motion events are stored
                # independently; crossings feed the security engine (B6/A10).
                try:
                    t_db0 = time.time()
                    _, motion_events = motion.process(frames)
                    for mev in motion_events:
                        try:
                            insert_motion_event(
                                conn,
                                camera=mev["camera"],
                                timestamp=mev["timestamp"],
                                score=mev.get("motion_score", 0.0),
                                duration_seconds=mev.get("duration_seconds", 0.0),
                                bbox=mev.get("bbox"),
                            )
                        except Exception as _me:
                            logger.debug("Motion persist skipped (isolated): %s", _me)
                    for xev in entry_exit.process(tracks):
                        try:
                            sec_ev = security.record_entry_exit(dict(xev))
                            insert_entry_exit_event(
                                conn,
                                camera=xev["camera"],
                                timestamp=xev["timestamp"],
                                event_type=xev["event_type"],
                                line=xev["line"],
                                direction=xev["direction"],
                                track_id=xev["track_id"],
                                employee_id=xev.get("employee_id"),
                                confidence=xev.get("confidence", 0.0),
                                security_event_id=(sec_ev or {}).get("event_id"),
                            )
                        except Exception as _xe:
                            logger.debug("Entry/exit persist skipped (isolated): %s", _xe)

                    # B15 -- expose Phase A/B capability status once per run.
                    if registry is None:
                        from src.detector_registry import DetectorRegistry
                        registry = DetectorRegistry(conn)
                    if not caps_registered:
                        registry.register_system_capabilities(
                            cameras_available=bool(frames),
                            entry_exit_lines=entry_exit.has_lines(),
                            motion_enabled=config.MOTION_ENABLED,
                        )
                        caps_registered = True
                except Exception as _pa:
                    logger.warning("Phase A processing skipped (isolated): %s", _pa)

                if any(d.get("person_present") for d in detections):
                    last_detection_at = time.time()

                # -- Phase 33: advisory intelligence (temporal/anomaly/risk) --
                # Isolated inside the engine (never breaks the main pipeline).
                try:
                    security.run_intelligence_cycle()
                except Exception:
                    logger.debug("Intelligence cycle skipped (isolated).",
                                 exc_info=True)

                if is_local and not _resolution_reported:
                    h, w = next(iter(frames.values())).shape[:2]
                    print(f"Camera feed online: {w}x{h}  "
                          f"(~{_target_fps_eff} fps processed)")
                    _resolution_reported = True

                # Reconcile discovered employees into the metadata store.
                discovered = [d["emp_id"] for d in detections
                              if d["emp_id"] not in ("__person__",)]
                if discovered:
                    try:
                        employees.ensure_known(
                            discovered,
                            enrolled_ids=detector.face_registry.enrolled_employee_ids,
                        )
                    except Exception:
                        logger.debug("Employee reconcile skipped (DB busy?)", exc_info=True)
                t_db1 = time.time()

                # Cache per-camera face overlays for the montage
                for cam_id in frames:
                    cam_dets = [d for d in detections if d["cam"] == cam_id]
                    last_frame_labels[cam_id] = _labels_for(cam_dets)
            else:
                # No *usable* live frames -- freeze timers so a dead or
                # blank/frozen camera can never accrue AWAY time.
                last_processed = time.time()

            t_detect = time.time()

            # -- Phase 31: Security engine tick --
            # Camera health must be evaluated even when no feed supplies a
            # usable frame; otherwise an all-camera outage can never reach the
            # CAMERA_OFFLINE transition.  Do not call a missing detector a
            # detector failure while there is no frame to inspect: that would
            # change the established all-camera-outage event semantics.
            try:
                t_events0 = time.time()
                security.tick(
                    detections=detections,
                    camera_health=health,
                    frames=frames,
                    detector_unavailable=bool(frames) and (
                        detector is None or getattr(detector, "model", None) is None),
                )
                t_events1 = time.time()
            except Exception as err:
                logger.exception("Security tick failed (isolated): %s", err)

            states = tracker.live_states()
            employee_ids = tracker.employee_ids

            now = time.time()
            for emp_id in employee_ids:
                if emp_id not in telemetry:
                    telemetry[emp_id] = {ACTIVE: 0.0, ON_PHONE: 0.0, AWAY: 0.0}
                    last_state[emp_id] = None
                cur = states.get(emp_id, AWAY)
                if last_state[emp_id] is not None and last_processed is not None:
                    dt = now - last_processed
                    telemetry[emp_id][cur] = telemetry[emp_id].get(cur, 0.0) + dt
                last_state[emp_id] = cur
            last_processed = now
            t_state = time.time()

            fps_frames += 1
            if now - fps_start >= 1.0:
                fps = fps_frames / (now - fps_start)
                fps_frames = 0
                fps_start = now

            _maybe_log_pilot_metrics(health, fps, detector)

            t_live0 = time.time()
            phone_diag = (detector.diagnostics_snapshot()
                          if detector is not None
                          and hasattr(detector, "diagnostics_snapshot")
                          else {"enabled": False})
            reid_payload = None
            if detector is not None and hasattr(detector, "phase44_diagnostics"):
                try:
                    reid_payload = detector.phase44_diagnostics().get("reid")
                except Exception:
                    reid_payload = None

            _write_live_state(
                tracker.live_states(), telemetry,
                health, fps,
                sources=tracker.live_sources(),
                durations=tracker.live_durations(),
                security=(security.snapshot() if security else None),
                names=employee_names,
                last_seen=tracker.live_last_seen(),
                last_detection_sec=
                    (time.time() - last_detection_at) if last_detection_at else None,
                spatial=spatial.snapshot(),
                motion=motion.snapshot(),
                camera_selection=camera_selection,
                ai_capabilities=(_registry_capabilities(registry)
                                 if registry is not None else None),
                phone_diag=phone_diag,
                timing=timing_ema,
                reid=reid_payload,
            )
            t_live = time.time()

            # Part F: update the per-stage EMA (published next cycle).
            if config.PIPELINE_TIMING:
                _loop_total = time.time()
                timing_n += 1
                _k = 1.0 / timing_n
                _stage = (detector.stage_timing()
                          if detector is not None
                          and hasattr(detector, "stage_timing")
                          else {})
                _max_age_ms = 0.0
                for _hc in health.values():
                    _last = _hc.get("last_frame")
                    if _last:
                        _max_age_ms = max(_max_age_ms,
                                          (time.time() - _last) * 1000.0)
                _detect = t_detect - t_capture
                _cap = t_capture - t_total
                _state = t_state - t_detect
                _live = t_live - t_live0
                _sub = (_cap + _state + _live
                        + (_stage.get("person_yolo_ms", 0.0)
                           + _stage.get("phone_ms", 0.0)
                           + _stage.get("face_ms", 0.0)
                           + _stage.get("reid_ms", 0.0)
                           + (t_track1 - t_track0)
                           + (t_db1 - t_db0)
                           + (t_events1 - t_events0)) * 0.001)
                for _key, _v in {
                    "capture_ms": _cap * 1000.0,
                    "person_yolo_ms": _stage.get("person_yolo_ms", 0.0),
                    "phone_ms": _stage.get("phone_ms", 0.0),
                    "face_ms": _stage.get("face_ms", 0.0),
                    "reid_ms": _stage.get("reid_ms", 0.0),
                    "tracking_ms": (t_track1 - t_track0) * 1000.0,
                    "db_ms": (t_db1 - t_db0) * 1000.0,
                    "events_ms": (t_events1 - t_events0) * 1000.0,
                    "detect_ms": _detect * 1000.0,
                    "state_ms": _state * 1000.0,
                    "live_state_ms": _live * 1000.0,
                    "other_ms": max(0.0, (_loop_total - t_total) - _sub)
                                * 1000.0,
                    "frame_age_ms": _max_age_ms,
                    "total_ms": (_loop_total - t_total) * 1000.0,
                }.items():
                    if _key in timing_ema:
                        timing_ema[_key] += (_v - timing_ema[_key]) * _k
                    else:
                        timing_ema[_key] = _v

            _maybe_run_retention(conn)   # once per day; no-op most seconds

            _maybe_run_daily_backup()   # opt-in; at BACKUP_HOUR once per day

            if use_gui:
                cam_health = cams.camera_health()
                montage = build_montage(frames, last_frame_labels, states, cam_health)
                online_ct = cams.online_count
                if montage is not None:
                    draw_dashboard(montage, employee_ids, states, telemetry,
                                   fps, online_ct, len(cam_health))
                    cv2.imshow(WINDOW_NAME, montage)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("r"):
                    generate_and_maybe_email(conn, "on-demand", email_on_complete=not args.no_email)
            else:
                cv2.waitKey(1)

            if interval > 0:
                time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as err:
        logger.exception("Unexpected error in main loop: %s", err)
    finally:
        scheduler.stop()
        if use_gui:
            cv2.destroyAllWindows()
        _shutdown(conn, tracker, cams, email_on_complete=not args.no_email,
                  security=security, audit=audit)


def _registry_capabilities(registry) -> dict | None:
    """B15 -- JSON-safe registry view for the dashboard (isolated)."""
    try:
        if registry is None:
            return None
        return {"detectors": [
            {
                "detector_id": d.get("detector_id"),
                "name": d.get("name", ""),
                "status": d.get("status", "UNAVAILABLE"),
                "status_detail": d.get("status_detail", ""),
                "enabled": bool(d.get("enabled")),
                "event_types": d.get("event_types", ""),
                "version": d.get("version", "") or "",
            }
            for d in registry.list()
        ]}
    except Exception:
        return None


def _labels_for(cam_dets):
    """Map a camera's flat detection entries into HUD face overlay data.

    Each recognised detection carries ``face_box`` + ``face_label`` (for
    matched / unmatched faces).  ``__person__`` entries carry no box and are
    used only for presence, so they are skipped here.
    """
    boxes = []
    labels = []
    seen = set()
    for det in cam_dets:
        if det.get("emp_id") == "__person__":
            continue
        fb = det.get("face_box")
        if not fb:
            continue
        emp_id = det.get("emp_id", "Unknown")
        score = det.get("face_score", 0.0)
        key = (tuple(fb), emp_id)
        if key in seen:
            continue
        seen.add(key)
        boxes.append(tuple(fb))
        labels.append((emp_id, score))
    return {"face_boxes": boxes, "face_labels": labels}


# ======================================================================
# Shutdown
# ======================================================================

def _shutdown(conn, tracker: MultiTracker, cams: MultiCameraManager,
              email_on_complete: bool = True, security=None, audit=None):
    logger.info("Shutting down gracefully ...")
    try:
        if security:
            security.flush()
        tracker.flush_all()
    finally:
        if audit:
            audit.record("daemon.shutdown")
        generate_and_maybe_email(conn, "final", email_on_complete=email_on_complete)
        if cams:
            cams.stop()
        conn.close()
    logger.info("Done. Goodbye.")


# ======================================================================

if __name__ == "__main__":
    setup_logging()
    _args = parse_args()
    run(_args)
