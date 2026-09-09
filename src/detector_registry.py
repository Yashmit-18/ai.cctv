"""Detector registry / plugin architecture (Phase 32).

A registry of computer-vision detectors.  The registry never fabricates AI
detections and never silently enables a model:

* Detectors whose event family requires a real trained model
  (fall / fight / fire / weapon) are registered as ``FUTURE_MODEL_REQUIRED``
  unless an actual, verifiable model file is configured.
* A detector is only reportable as ``AVAILABLE`` when its model path exists
  on disk and it is enabled.
* State transitions are explicit (``register``, ``enable``, ``disable``,
  ``set_status``) and audited.

This replaces the hard-coded detector list in ``ActivityDetector`` with a
config-driven, pluggable registry while keeping every existing detector's
privacy/naming guarantees intact.
"""

from __future__ import annotations

import logging
import os

from src import database as db

logger = logging.getLogger("cctv.detectors")

# Event families that MUST NOT be claimed without a real model.
AI_REQUIRED_FAMILIES = {"fall", "fight", "fire", "weapon"}

# B15 -- status vocabulary.  A detector is exactly one of these at a time; the
# dashboard renders the vocabulary verbatim so operators never see invented
# states.
STATUS_AVAILABLE = "AVAILABLE"                  # enabled + ready now
STATUS_DISABLED = "DISABLED"                    # operator/config disabled
STATUS_DEGRADED = "DEGRADED"                    # enabled but underperforming
STATUS_NOT_CONFIGURED = "NOT_CONFIGURED"        # registered, no input wired
STATUS_FUTURE_MODEL_REQUIRED = "FUTURE_MODEL_REQUIRED"
STATUS_UNAVAILABLE = "UNAVAILABLE"              # DEFAULT legacy/db fallback

STATUS_VOCABULARY: tuple[str, ...] = (
    STATUS_AVAILABLE,
    STATUS_DISABLED,
    STATUS_DEGRADED,
    STATUS_NOT_CONFIGURED,
    STATUS_FUTURE_MODEL_REQUIRED,
    STATUS_UNAVAILABLE,
)


class DetectorRegistry:
    """Config-driven registry of detectors with safety invariants."""

    def __init__(self, conn):
        self._conn = conn

    def register(self, *, detector_id: str, name: str = "", version: str = "",
                 model_path: str = "", event_types: str = "",
                 conf_threshold: float | None = None,
                 persist_sec: float | None = None,
                 cooldown_sec: float | None = None,
                 camera_scope: str = "", zone_scope: str = "",
                 enabled: bool = True) -> dict:
        """Register (upsert) a detector.

        Safety: an AI-required family without a verifiable model file is stored
        as ``FUTURE_MODEL_REQUIRED`` and force-disabled, never auto-used.
        """
        present_model = bool(model_path) and os.path.isfile(model_path)
        is_ai = self._needs_ai_model(event_types)
        if is_ai and not present_model:
            enabled = False
            status = STATUS_FUTURE_MODEL_REQUIRED
            detail = "requires real model; not enabled (no AI fabrication)"
        elif is_ai and present_model:
            status = STATUS_AVAILABLE if enabled else STATUS_DISABLED
            detail = "model present"
        else:
            status = STATUS_AVAILABLE if enabled else STATUS_DISABLED
            detail = "rule/logic detector"
        det = {
            "detector_id": detector_id,
            "name": name, "version": version, "enabled": enabled,
            "model_path": model_path, "event_types": event_types,
            "conf_threshold": conf_threshold, "persist_sec": persist_sec,
            "cooldown_sec": cooldown_sec, "camera_scope": camera_scope,
            "zone_scope": zone_scope, "status": status,
            "status_detail": detail,
        }
        db.upsert_detector(self._conn, det)
        logger.info(
            "[DETECTOR] %s registered status=%s enabled=%s",
            detector_id, status, enabled)
        return det

    def _needs_ai_model(self, event_types: str) -> bool:
        lowered = (event_types or "").lower()
        return any(f in lowered for f in AI_REQUIRED_FAMILIES)

    def enable(self, detector_id: str, *, require_model: bool = True) -> bool:
        det = db.get_detector(self._conn, detector_id)
        if not det:
            return False
        if require_model and self._needs_ai_model(det.get("event_types", "")):
            model = det.get("model_path") or ""
            if not (model and os.path.isfile(model)):
                db.set_detector_status(
                    self._conn, detector_id, STATUS_FUTURE_MODEL_REQUIRED,
                    detail="cannot enable: real model required")
                return False
        db.upsert_detector(self._conn, {**det, "enabled": True,
                                        "status": STATUS_AVAILABLE})
        return True

    def disable(self, detector_id: str) -> bool:
        det = db.get_detector(self._conn, detector_id)
        if not det:
            return False
        db.upsert_detector(self._conn, {**det, "enabled": False,
                                        "status": STATUS_DISABLED})
        return True

    # ------------------------------------------------------------------
    # Phase A/B -- system capability registration (A5/A8/A10, B15)
    # ------------------------------------------------------------------

    def register_system_capabilities(self, *, cameras_available: bool = True,
                                     entry_exit_lines: bool = False,
                                     motion_enabled: bool = True) -> list[dict]:
        """Register the Phase A/B capabilities for the operator dashboard.

        The registry stays honest: no camera input -> ``NOT_CONFIGURED``;
        feature switched off in config -> ``DISABLED``; missing virtual lines
        file -> ``NOT_CONFIGURED`` with an actionable detail string.
        """
        caps = []
        caps.append(self.register(
            detector_id="person_tracker",
            name="Person Tracker (A5/A6)",
            event_types="presence,tracking", enabled=cameras_available,
            camera_scope="*",
        ))
        if not cameras_available:
            self.set_status("person_tracker", STATUS_NOT_CONFIGURED,
                            "no camera source is producing frames")
        else:
            self.set_status("person_tracker", STATUS_AVAILABLE,
                            "tracking active on available cameras")

        caps.append(self.register(
            detector_id="motion_detector",
            name="Motion Detector (A8)",
            event_types="motion", enabled=motion_enabled, camera_scope="*",
        ))
        self.set_status(
            "motion_detector",
            STATUS_AVAILABLE if motion_enabled else STATUS_DISABLED,
            "MOTION_ENABLED=true" if motion_enabled else "MOTION_ENABLED=false",
        )

        caps.append(self.register(
            detector_id="entry_exit_engine",
            name="Entry/Exit Line Engine (A10)",
            event_types="entry,exit", enabled=entry_exit_lines, camera_scope="*",
        ))
        self.set_status(
            "entry_exit_engine",
            STATUS_AVAILABLE if entry_exit_lines else STATUS_NOT_CONFIGURED,
            "virtual lines configured" if entry_exit_lines
            else "no entry/exit lines file (ENTRY_EXIT_LINES_FILE)",
        )
        return caps

    @staticmethod
    def status_vocabulary() -> dict:
        """B15 -- machine + human vocabulary for detector status."""
        return {
            "statuses": list(STATUS_VOCABULARY),
            "available": STATUS_AVAILABLE,
            "disabled": STATUS_DISABLED,
            "degraded": STATUS_DEGRADED,
            "not_configured": STATUS_NOT_CONFIGURED,
            "future_model_required": STATUS_FUTURE_MODEL_REQUIRED,
        }

    def set_status(self, detector_id: str, status: str,
                   detail: str = "") -> None:
        db.set_detector_status(self._conn, detector_id, status, detail)

    def list(self, *, enabled_only: bool = False) -> list[dict]:
        return db.list_detectors(self._conn, enabled_only=enabled_only)

    def get(self, detector_id: str) -> dict | None:
        return db.get_detector(self._conn, detector_id)

    # ------------------------------------------------------------------
    # Phase 33 -- detector/model health & versioning (B9)
    # ------------------------------------------------------------------

    def record_run(self, detector_id: str, *, ok: bool, error: str = "",
                   last_inference: str | None = None) -> None:
        """Record the outcome of one detector run (last-run/last-inference)."""
        db.record_detector_run(self._conn, detector_id, ok=ok, error=error,
                               last_inference=last_inference)

    def health(self, detector_id: str) -> dict | None:
        """Transparent health summary for one detector."""
        det = db.get_detector(self._conn, detector_id)
        if not det:
            return None
        return {
            "detector_id": detector_id,
            "status": det.get("status"),
            "enabled": bool(det.get("enabled")),
            "version": det.get("version") or "",
            "model_path": det.get("model_path") or "",
            "last_run_at": det.get("last_run_at"),
            "last_inference_at": det.get("last_inference_at"),
            "last_run_ok": bool(det.get("last_run_ok")),
            "last_error": det.get("last_error") or "",
            "inference_count": int(det.get("inference_count") or 0),
        }

    def health_summary(self) -> dict:
        """Aggregate detector health for System Health 2.0."""
        rows = self.list()
        available = [d for d in rows if d.get("status") == STATUS_AVAILABLE]
        failing = [d for d in rows if d.get("enabled") and not d.get("last_run_ok")]
        future = [d for d in rows if d.get("status") == STATUS_FUTURE_MODEL_REQUIRED]
        return {
            "total": len(rows),
            "available": len(available),
            "failing": len(failing),
            "future_model_required": len(future),
        }

