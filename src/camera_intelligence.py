"""Camera intelligence & system health (Phase 32).

Aggregates camera configuration + health telemetry into operational insights:
availability %, degraded-camera detection, downtime, tamper flags and a
system-health summary.

Safety contract
---------------
Camera failure is reported AS CAMERA FAILURE only.  It is NEVER interpreted
as employee absence, and it never alters productivity data.
"""

from __future__ import annotations

import config
from src import database as db


class CameraIntelligence:
    """Operational insights over camera config + health."""

    def __init__(self, conn):
        self._conn = conn

    def list_cameras(self, *, enabled_only: bool = True) -> list[dict]:
        return db.list_camera_configs(self._conn, enabled_only=enabled_only)

    def camera_status(self, camera_id: str) -> dict:
        """Return the latest health snapshot for one camera (or None)."""
        cfg = db.get_camera_config(self._conn, camera_id)
        hist = db.camera_health_history(self._conn, camera_id, limit=1)
        latest = hist[0] if hist else None
        return {
            "camera_id": camera_id,
            "configured": cfg is not None,
            "enabled": bool(cfg.get("enabled", 1)) if cfg else False,
            "healthy": bool(latest and latest.get("health") == "HEALTHY"),
            "health": (latest or {}).get("health") or "UNKNOWN",
            "fps": (latest or {}).get("fps"),
            "reconnects": (latest or {}).get("reconnects", 0),
            "tampered": bool((latest or {}).get("tampered", 0)),
            "downtime_sec": db.camera_downtime_total(self._conn, camera_id),
        }

    def overview(self) -> list[dict]:
        """Combine config + latest health for every configured camera."""
        out = []
        for cfg in self.list_cameras(enabled_only=False):
            cid = cfg["camera_id"]
            status = self.camera_status(cid)
            status.update({
                "name": cfg.get("name", ""),
                "location": cfg.get("location", ""),
                "zone": cfg.get("zone", ""),
            })
            out.append(status)
        return out

    def degraded_cameras(self, *, max_fps: float = 7.0) -> list[dict]:
        """Cameras that are offline, tampered, or running below target FPS."""
        degraded = []
        for cam in self.overview():
            reason = None
            if not cam["healthy"] and cam["health"] != "UNKNOWN":
                reason = f"health={cam['health']}"
            elif cam["tampered"]:
                reason = "tampered"
            elif cam["fps"] is not None and cam["fps"] < max_fps:
                reason = f"low_fps={cam['fps']}"
            if reason:
                cam["reason"] = reason
                degraded.append(cam)
        return degraded

    def availability_pct(self, camera_id: str | None = None,
                         window_sec: float = 86400.0) -> float:
        """Fraction of sampled records that were healthy (all cameras)."""
        if camera_id:
            records = db.camera_health_history(
                self._conn, camera_id, limit=100000)
        else:
            rows = self._conn.execute(
                "SELECT * FROM camera_health LIMIT ?", (100000,)).fetchall()
            cols = [d[1] for d in
                    self._conn.execute(
                        "PRAGMA table_info(camera_health)").fetchall()]
            records = [dict(zip(cols, r)) for r in rows]
        if not records:
            return 100.0
        healthy = sum(1 for rec in records if rec.get("health") == "HEALTHY")
        total = len(records)
        return round(100.0 * healthy / total, 2) if total else 100.0

    # ------------------------------------------------------------------
    # Phase 33 -- camera reliability score (B7) & coverage (B8)
    # ------------------------------------------------------------------

    def reliability_score(self, camera_id: str) -> dict:
        """Transparent 0..100 reliability score for one camera.

        Components (documented, sum-weighted):
        * availability % (0.5)
        * reconnects penalty (0.2)  -- more reconnects = lower
        * tamper-free (0.2)         -- tampered camera = lower
        * fps adequacy (0.1)        -- below target = lower
        The components are returned so the number is auditable.  Reliability is
        an operational metric about the camera, never about employees.
        """
        cfg = db.get_camera_config(self._conn, camera_id)
        avail = self.availability_pct(camera_id)
        hist = db.camera_health_history(self._conn, camera_id, limit=1000)
        reconnects = sum(int(r.get("reconnects") or 0) for r in hist)
        latest = hist[0] if hist else None
        tampered = bool(latest and latest.get("tampered"))
        fps = latest.get("fps") if latest else None

        # availability component (0..1)
        comp_avail = avail / 100.0
        # reconnects penalty
        comp_reconn = max(0.0, 1.0 - min(1.0, reconnects / 10.0))
        # tamper component
        comp_tamper = 0.0 if tampered else 1.0
        # fps adequacy
        target = cfg.get("fps_target") or config.RELIABILITY_FPS_TARGET
        if fps is None:
            comp_fps = 0.5
        else:
            comp_fps = min(1.0, fps / max(1, target))

        score = round(
            0.5 * comp_avail + 0.2 * comp_reconn + 0.2 * comp_tamper +
            0.1 * comp_fps, 2,
        ) * 100
        return {
            "camera_id": camera_id,
            "score": round(score, 1),
            "components": {
                "availability": round(avail, 2),
                "reconnects": reconnects,
                "tampered": tampered,
                "fps": fps,
                "fps_target": target,
            },
        }

    def coverage_analysis(self) -> dict:
        """Camera coverage analysis across configured cameras (B8).

        A camera with no health samples, a disabled camera, or a camera not in
        the topology is flagged as a ``COVERAGE GAP`` -- an operational gap,
        **not** a security failure and never an accusation.  Uses
        ``camera_config.coverage_note`` if the operator annotated it.
        """
        from src.camera_topology import CameraTopology
        topo = CameraTopology(self._conn)
        in_topology = set(topo.connected_cameras())
        gaps: list[dict] = []
        covered = 0
        for cfg in self.list_cameras(enabled_only=False):
            cid = cfg["camera_id"]
            if not cfg.get("enabled", 1):
                gaps.append({
                    "camera_id": cid, "coverage_note": cfg.get("coverage_note") or "",
                    "reason": "camera disabled",
                })
                continue
            hist = db.camera_health_history(self._conn, cid, limit=1)
            if not hist:
                gaps.append({
                    "camera_id": cid, "coverage_note": cfg.get("coverage_note") or "",
                    "reason": "no health samples (COVERAGE GAP)",
                })
                continue
            covered += 1
            if cid not in in_topology:
                gaps.append({
                    "camera_id": cid, "coverage_note": cfg.get("coverage_note") or "",
                    "reason": "not in camera topology graph",
                })
        return {
            "configured": len(self.list_cameras(enabled_only=False)),
            "covered": covered,
            "gaps": gaps,
            "gap_count": len(gaps),
            "topology_edges": topo.summary()["edge_count"],
        }


class SystemHealth:
    """Coarse but honest system-health summary."""

    def __init__(self, conn):
        self._conn = conn

    def summary(self) -> dict:
        cams = CameraIntelligence(self._conn)
        enabled = cams.list_cameras(enabled_only=True)
        return {
            "camera_count": len(cams.list_cameras(enabled_only=False)),
            "enabled_cameras": len(enabled),
            "degraded": cams.degraded_cameras(),
            "availability_pct": cams.availability_pct(),
            "total_downtime_sec": db.camera_downtime_total(self._conn),
        }

    def summary_v2(self) -> dict:
        """System Health 2.0 (Phase 33, C1).

        Aggregates cameras, detectors, alerts, evidence and DB size into one
        honest health picture with an overall status string
        (HEALTHY / DEGRADED / FAILED / NOT_CONFIGURED).
        """
        from src.detector_registry import DetectorRegistry, STATUS_AVAILABLE
        from src.alerts import AlertEngine

        cams = CameraIntelligence(self._conn)
        records = cams.overview()
        degraded = cams.degraded_cameras()
        availability = cams.availability_pct()

        detectors = DetectorRegistry(self._conn)
        det_rows = detectors.list()
        available_dets = [d for d in det_rows if d.get("status") == STATUS_AVAILABLE]
        degraded_dets = [d for d in det_rows if d.get("enabled") and not d.get("last_run_ok")]

        # Alert health (B6): delivery success across non-escalated history.
        alert_rows = db.query_alerts_filtered(self._conn, limit=100000)
        total_alerts = len(alert_rows)
        failed_alerts = sum(1 for a in alert_rows
                            if a.get("status") in ("FAILED", "RETRYING"))
        escalated = sum(1 for a in alert_rows if a.get("status") == "ESCALATED")
        alert_delivery_pct = (100.0 * (total_alerts - failed_alerts) / total_alerts) \
            if total_alerts else 100.0

        # DB size + growth (A7).
        db_size_mb = 0.0
        try:
            info = self._conn.execute("PRAGMA page_count").fetchone()
            page_size = self._conn.execute("PRAGMA page_size").fetchone()
            if info and page_size:
                db_size_mb = round(float(info[0] * page_size[0]) / (1024 * 1024), 2)
        except Exception:
            pass

        # Evidence (A5).
        from src.evidence_access import EvidenceAccess
        ev_rows = db.list_evidence_files(self._conn)
        evidence_count = len(ev_rows)
        missing_files = 0
        for r in ev_rows:
            import os
            if r.get("available", 1) and r.get("path") and not os.path.isfile(r.get("path")):
                missing_files += 1

        problems: list[str] = []
        if degraded:
            problems.append(f"{len(degraded)} camera(s) degraded")
        if availability < 90:
            problems.append(f"camera availability {availability}% < 90%")
        if failed_alerts:
            problems.append(f"{failed_alerts} failed alert(s)")
        if missing_files:
            problems.append(f"{missing_files} missing evidence file(s)")
        if degraded_dets:
            problems.append(f"{len(degraded_dets)} detector(s) failing")

        if not records:
            status = "NOT_CONFIGURED"
        elif problems:
            status = "FAILED" if availability < 60 else "DEGRADED"
        else:
            status = "HEALTHY"

        return {
            "status": status,
            "camera_count": len(records),
            "degraded_cameras": len(degraded),
            "availability_pct": availability,
            "detector_count": len(det_rows),
            "available_detectors": len(available_dets),
            "degraded_detectors": len(degraded_dets),
            "alert_delivery_pct": round(alert_delivery_pct, 2),
            "escalated_alerts": escalated,
            "evidence_count": evidence_count,
            "missing_evidence": missing_files,
            "db_size_mb": db_size_mb,
            "problems": problems,
        }
