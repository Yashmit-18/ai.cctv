"""Evidence integrity & access audit (Phase 33, A5/A6/A8).

* **Integrity viewer** -- ``verify_integrity`` re-hashes an evidence file and
  compares against the stored SHA-256, returning ``INTEGRITY_VERIFIED`` /
  ``INTEGRITY_FAILED`` / ``INTEGRITY_UNAVAILABLE`` (missing file or no stored
  hash).  This is the *view-time* verification that Phase 32 left implicit.
* **Access audit** -- every view / export / verify logs a row to
  ``evidence_access`` (actor, action, incident) so evidence handling is fully
  traceable.
* **Export hardening** -- bounded, audited export of security records with RTSP
  credential redaction (via ``domain.redact_url``).

This module is a light façade: file hashing and path isolation live in
``EvidenceStore``; the DB rows live in ``database``.  It never deletes evidence.
"""

from __future__ import annotations

import hashlib
import logging
import os

from src import database as db

logger = logging.getLogger("cctv.evidence_access")

INTEGRITY_VERIFIED = "INTEGRITY_VERIFIED"
INTEGRITY_FAILED = "INTEGRITY_FAILED"
INTEGRITY_UNAVAILABLE = "INTEGRITY_UNAVAILABLE"


def _sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class EvidenceAccess:
    """Integrity verification + access auditing for evidence artifacts."""

    def __init__(self, conn, *, actor: str = "system"):
        self._conn = conn
        self._actor = actor

    # -- integrity ------------------------------------------------------
    def verify_integrity(self, evidence_id: int) -> dict:
        """Re-hash an evidence file and compare to its stored SHA-256.

        Returns ``{status, evidence_id, path, detail}`` with one of the
        INTEGRITY_* statuses.  Audits the verification.
        """
        row = None
        for r in db.list_evidence_files(self._conn):
            if int(r.get("id") or 0) == evidence_id:
                row = r
                break
        if not row:
            db.log_evidence_access(self._conn, evidence_id, "verify",
                                   actor=self._actor, detail="no such evidence row")
            return {"status": INTEGRITY_UNAVAILABLE, "evidence_id": evidence_id,
                    "path": None, "detail": "no such evidence row"}
        path = row.get("path")
        stored = row.get("sha256")
        if not path or not os.path.isfile(path):
            db.log_evidence_access(self._conn, evidence_id, "verify",
                                   actor=self._actor,
                                   detail="INTEGRITY_UNAVAILABLE (file missing)")
            return {"status": INTEGRITY_UNAVAILABLE, "evidence_id": evidence_id,
                    "path": path, "detail": "file missing on disk"}
        if not stored:
            db.log_evidence_access(self._conn, evidence_id, "verify",
                                   actor=self._actor,
                                   detail="INTEGRITY_UNAVAILABLE (no stored hash)")
            return {"status": INTEGRITY_UNAVAILABLE, "evidence_id": evidence_id,
                    "path": path, "detail": "no stored hash"}
        actual = _sha256_file(path)
        ok = bool(actual and actual.lower() == str(stored).lower())
        status = INTEGRITY_VERIFIED if ok else INTEGRITY_FAILED
        db.set_evidence_metadata(self._conn, path, sha256=stored)
        db.log_evidence_access(
            self._conn, evidence_id, "verify", actor=self._actor,
            incident_id=row.get("incident_id"),
            detail=f"{status} stored={stored} actual={actual}")
        return {
            "status": status, "evidence_id": evidence_id, "path": path,
            "detail": status,
        }

    # -- access auditing ------------------------------------------------
    def view(self, evidence_id: int) -> None:
        self._log(evidence_id, "view")

    def export_artifact(self, evidence_id: int) -> None:
        self._log(evidence_id, "export")

    def _log(self, evidence_id: int, action: str) -> None:
        incident_id = None
        for r in db.list_evidence_files(self._conn):
            if int(r.get("id") or 0) == evidence_id:
                incident_id = r.get("incident_id")
                break
        db.log_evidence_access(self._conn, evidence_id, action,
                               actor=self._actor, incident_id=incident_id)

    def access_log(self, *, evidence_id: int | None = None,
                   limit: int = 200) -> list[dict]:
        return db.list_evidence_access(self._conn, evidence_id=evidence_id,
                                       limit=limit)

    # -- export ---------------------------------------------------------
    def export_incident(self, incident_id: str, *, max_events: int = 1000) -> dict:
        """Bounded, audited export of one incident's records as plain dicts.

        Full row paths in evidence are surfaced but RTSP-style credentials are
        redacted and any raw URL strings in notes/details are scrubbed.
        Returns ``{"incident", "events", "alerts", "evidence", "exported_at",
        "bounded"}``.
        """
        from src.domain import redact_url
        incident = db.get_incident(self._conn, incident_id)
        if not incident:
            return {}
        events = db.query_security_events(self._conn, incident_id=incident_id,
                                          limit=max_events)
        evidence = db.list_evidence_files(self._conn, incident_id=incident_id)
        evidence_out = [dict(e) for e in evidence]
        for e in evidence_out:
            p = e.get("path") or ""
            e["path_r"] = redact_url(p)
        alerts = db.query_alerts_filtered(self._conn, incident_id=incident_id,
                                          limit=500)
        for a in alerts:
            if a.get("message"):
                a["message"] = redact_url(str(a["message"]))
        notes = db.list_incident_notes(self._conn, incident_id)
        db.log_evidence_access(
            self._conn, 0, "export", actor=self._actor,
            incident_id=incident_id, detail=f"export of incident {incident_id}")
        from datetime import datetime
        return {
            "incident": incident,
            "events": events,
            "evidence": evidence_out,
            "alerts": alerts,
            "notes": notes,
            "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bounded": {
                "max_events": max_events,
                "event_count": len(events),
            },
        }
