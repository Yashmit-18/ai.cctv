"""Security-risk scoring for incidents (Phase 33, B4).

Scores **incidents** (not people) so that operators can triage an investigation
queue.  The score is 0..1 and is the weighted combination of transparent,
documented components -- severity, corroboration (multiple linked event types),
repeat frequency, and cross-camera corroboration.

Hard rules
----------
* The score applies to an **incident**, never to an employee.  There is no
  per-person risk score anywhere in the system (forbidden by policy).
* Components are stored verbatim so the number is fully auditable.
* The score is advisory meta-data; it cannot resolve/dismiss/accuse, and it
  never alters productivity or employee state.
"""

from __future__ import annotations

import json
import logging

import config
from src import database as db

logger = logging.getLogger("cctv.risk")

# Component weights (transparent, sum == 1.0).
SEVERITY_WEIGHT = 0.5
CORROBORATION_WEIGHT = 0.25
REPEAT_WEIGHT = 0.15
CROSS_CAMERA_WEIGHT = 0.10

# Phase 34 bounded context-factor bonuses (each small, capped to keep <= 1.0).
_F_ZONE_SENSITIVE = 0.10
_F_AFTER_HOURS = 0.10
_F_EVIDENCE = 0.05
_F_LOW_CONFIDENCE = 0.05

_SEV_TO_VALUE = {
    "INFO": 0.2, "LOW": 0.3, "MEDIUM": 0.5, "HIGH": 0.8, "CRITICAL": 1.0,
}


class IncidentRiskScorer:
    """Computes a transparent 0..1 score for one incident."""

    def __init__(self, conn):
        self._conn = conn

    def score(self, incident_id: str, *, actor: str = "system") -> dict | None:
        inc = db.get_incident(self._conn, incident_id)
        if not inc:
            return None

        severity_value = _SEV_TO_VALUE.get(inc.get("severity") or "INFO", 0.5)
        # Corroboration: number of distinct linked event types.
        events = db.list_incident_events(self._conn, incident_id)
        distinct_types = {e.get("event_type") for e in events if e.get("event_type")}
        distinct_types.add(inc.get("event_type"))  # the incident's own type
        corroboration = min(1.0, len(distinct_types) / 3.0)

        occurrences = max(1, int(inc.get("occurrences") or 1))
        repeat = min(1.0, (occurrences - 1) / 5.0)

        # Cross-camera corroboration: distinct cameras linked to the incident.
        cameras = {e.get("camera") for e in events if e.get("camera")}
        cameras.add(inc.get("camera"))
        cross = 1.0 if len(cameras) >= 2 else 0.0

        score = (
            SEVERITY_WEIGHT * severity_value
            + CORROBORATION_WEIGHT * corroboration
            + REPEAT_WEIGHT * repeat
            + CROSS_CAMERA_WEIGHT * cross
        )

        # --- Phase 34: transparent context factors (bounded bonus) ---------
        # Each is a small, documented additive term.  A human can read the
        # ``components`` and ``rationale`` to explain exactly why a score is
        # what it is.  They never apply to a person -- only to this incident.
        ctx: dict[str, float] = {"zone_sensitive": 0.0, "after_hours": 0.0,
                                 "evidence": 0.0, "low_confidence": 0.0}
        if config.RISK_CONTEXT_FACTORS:
            ctx["zone_sensitive"] = _F_ZONE_SENSITIVE \
                if _zone_sensitive(self._conn, inc.get("zone")) else 0.0
            ctx["after_hours"] = _F_AFTER_HOURS \
                if _has_after_hours(events, inc.get("event_type")) else 0.0
            ctx["evidence"] = _F_EVIDENCE \
                if _evidence_present(self._conn, incident_id) else 0.0
            conf = inc.get("confidence")
            ctx["low_confidence"] = _F_LOW_CONFIDENCE \
                if conf is not None and float(conf) < 0.5 else 0.0
            for v in ctx.values():
                score += v

        score = round(min(1.0, max(0.0, score)), 3)

        components = {
            "severity": round(severity_value, 3),
            "corroboration_types": len(distinct_types),
            "occurrences": occurrences,
            "cross_camera": len(cameras) >= 2,
            **{k: round(float(v), 3) for k, v in ctx.items()},
        }
        ctx_txt = " ".join(f"{k}={v:.2f}" for k, v in ctx.items() if v)
        rationale = (
            f"score={score:.2f} = 0.5*sev({severity_value:.2f}) + "
            f"0.25*corrob({corroboration:.2f}) + 0.15*repeat({repeat:.2f}) + "
            f"0.10*cross({cross:.2f})"
            + (f" + context({ctx_txt})" if ctx_txt else "")
        )
        db.upsert_incident_risk(self._conn, incident_id, score,
                                components=json.dumps(components),
                                rationale=rationale)
        db.audit(self._conn, "incident.risk_scored", actor=actor,
                 resource=incident_id, detail=f"score={score}")
        return {
            "incident_id": incident_id,
            "score": score,
            "components": components,
            "rationale": rationale,
        }

    def top(self, limit: int = 50) -> list[dict]:
        return db.list_incident_risk(self._conn, limit=limit)

    def list(self, limit: int = 200) -> list[dict]:
        """Canonical risk-listing view joined with incident context.

        Each row carries the incident's ``severity``/``status``/``review_state``
        so operators can filter to open, non-dismissed incidents.  This is the
        API the dashboard risk panel uses.
        """
        return db.list_incident_risk(self._conn, limit=limit)

    def get_for(self, incident_id: str) -> dict | None:
        return db.get_incident_risk(self._conn, incident_id)


def recommend_review(score: float) -> str:
    """Map a score to an operator-facing triage hint (advisory only)."""
    if score >= 0.8:
        return "REVIEW_PRIORITY"
    if score >= 0.5:
        return "REVIEW"
    return "LOW_PRIORITY"


def _zone_sensitive(conn, zone: str | None) -> bool:
    """True when a known zone exists and is not `BYPASS` (i.e. sensitive)."""
    if not zone:
        return False
    try:
        rows = conn.execute(
            "SELECT alert_policy FROM security_zones WHERE zone_name = ?",
            (zone,)).fetchall()
    except Exception:
        return False
    for r in rows:
        return bool(r[0] and r[0] != "BYPASS")
    return False


def _has_after_hours(events: list[dict], event_type: str | None) -> bool:
    if event_type == "AFTER_HOURS_ACTIVITY":
        return True
    from src.domain import AFTER_HOURS_ACTIVITY
    return any(e.get("event_type") == AFTER_HOURS_ACTIVITY for e in events)


def _evidence_present(conn, incident_id: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM evidence_files WHERE incident_id = ?",
        (incident_id,)).fetchone()
    return bool(row and row[0] > 0)
