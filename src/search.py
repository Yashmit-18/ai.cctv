"""Structured security search & filtering (Phase 32).

Provides a unified, read-only ``SecuritySearch`` facade over incidents,
events, alerts and evidence with safe parameterized filters (type, severity,
camera, zone, employee, incident, review state, date range, free-text).  All
SQL is constructed from an allow-listed column map and bound parameters; no
user-controlled string is ever interpolated into SQL.

This module never mutates data -- it is a query/filter layer used by the
investigation workflow and dashboard.
"""

from __future__ import annotations

from src import database as db


class SecuritySearch:
    """Read-only structured search across security records."""

    # Allow-listed filterable columns per entity (safe raw SQL names only).
    _INCIDENT_FILTERS = {
        "event_type", "severity", "status", "camera", "zone", "employee_id",
        "review_state",
    }
    _EVENT_FILTERS = {
        "event_type", "severity", "camera", "zone", "employee_id", "incident_id",
    }
    _ALERT_FILTERS = {"event_type", "severity", "status", "camera", "incident_id"}
    _EVIDENCE_FILTERS = {"incident_id", "camera", "capture_type", "available"}

    def __init__(self, conn):
        self._conn = conn

    def _build(self, table: str, filters: dict, allowed: set[str],
               date_col: str, limit: int = 500, order: str = "DESC",
               offset: int = 0) -> list[dict]:
        clauses: list[str] = []
        params: list = []
        for key, value in (filters or {}).items():
            if key == "date_from":
                clauses.append(f"{date_col} >= ?"); params.append(value)
            elif key == "date_to":
                clauses.append(f"{date_col} <= ?"); params.append(value)
            elif key in allowed and value not in (None, ""):
                clauses.append(f"{key} = ?"); params.append(value)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT * FROM {table} " + where +
            f" ORDER BY {date_col} {order} LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        rows = self._conn.execute(sql, params).fetchall()
        cols = [d[1] for d in
                self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
        out = [dict(zip(cols, r)) for r in rows]
        return out

    def _text_search(self, table: str, columns: list[str], term: str,
                     date_col: str, limit: int = 200, offset: int = 0) -> list[dict]:
        like = f"%{term}%"
        conds = " OR ".join(f"{c} LIKE ?" for c in columns)
        sql = (
            f"SELECT * FROM {table} WHERE ({conds}) "
            f"ORDER BY {date_col} DESC LIMIT ? OFFSET ?"
        )
        rows = self._conn.execute(sql, [like] * len(columns) + [limit, offset]).fetchall()
        cols = [d[1] for d in
                self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
        return [dict(zip(cols, r)) for r in rows]

    def search_incidents(self, *, filters: dict | None = None,
                         text: str | None = None, limit: int = 200,
                         offset: int = 0) -> list[dict]:
        return self._build(
            "incidents", filters, self._INCIDENT_FILTERS,
            "created_at", limit=limit, order="DESC", offset=offset)

    def search_incidents_text(self, term: str, limit: int = 200, offset: int = 0) -> list[dict]:
        """Free-text over incident id + notes (investigation notes search)."""
        return self._text_search(
            "incidents", ["incident_id", "notes", "event_type"], term,
            "created_at", limit=limit, offset=offset)

    def search_events(self, *, filters: dict | None = None,
                      limit: int = 500, offset: int = 0) -> list[dict]:
        return self._build(
            "security_events", filters, self._EVENT_FILTERS,
            "timestamp", limit=limit, order="DESC", offset=offset)

    def search_alerts(self, *, filters: dict | None = None,
                      limit: int = 200, offset: int = 0) -> list[dict]:
        return self._build(
            "alerts", filters, self._ALERT_FILTERS,
            "created_at", limit=limit, order="DESC", offset=offset)

    def search_evidence(self, *, filters: dict | None = None,
                        limit: int = 300, offset: int = 0) -> list[dict]:
        return self._build(
            "evidence_files", filters, self._EVIDENCE_FILTERS,
            "captured_at", limit=limit, order="DESC", offset=offset)

    # ------------------------------------------------------------------
    # Composite investigation helper
    # ------------------------------------------------------------------
    def incident_timeline(self, incident_id: str) -> dict:
        """Return one incident plus its related events, alerts and evidence.

        Death/simplest useful shape for the investigation view: every related
        record grouped by entity.
        """
        incident = db.get_incident(self._conn, incident_id)
        if not incident:
            return {}
        return {
            "incident": incident,
            "events": db.list_incident_events(self._conn, incident_id),
            "related_events": db.query_security_events(
                self._conn, incident_id=incident_id),
            "alerts": db.query_alerts_filtered(
                self._conn, incident_id=incident_id),
            "evidence": db.list_evidence_files(self._conn, incident_id=incident_id),
            "notes": db.list_incident_notes(self._conn, incident_id),
        }

    def timeline_v2(self, incident_id: str, *, page: int = 0,
                    page_size: int = 50) -> dict:
        """Investigation Timeline 2.0 (Phase 33, A4).

        Deterministic, DB-backed, chronological, paginated.  Merges the
        incident's related events, alerts, evidence and notes into one ordered
        stream of ``{ts, kind, label, detail}`` rows (oldest first).  Includes
        pagination metadata so the UI can page through long timelines.
        """
        incident = db.get_incident(self._conn, incident_id)
        if not incident:
            return {"incident_id": incident_id, "rows": [], "page": page,
                    "page_size": page_size, "total": 0}

        entries: list[dict] = []
        # Related events (the granular underlying records).
        for ev in db.query_security_events(self._conn, incident_id=incident_id,
                                           limit=10000):
            entries.append({
                "ts": ev.get("timestamp"),
                "kind": "EVENT",
                "label": ev.get("event_type"),
                "detail": f"camera={ev.get('camera','-')} zone={ev.get('zone','-')} "
                          f"sev={ev.get('severity','-')}",
            })
        # Alerts fired for this incident.
        for al in db.query_alerts_filtered(self._conn, incident_id=incident_id,
                                           limit=500):
            entries.append({
                "ts": al.get("created_at"),
                "kind": "ALERT",
                "label": (al.get("rule_name") or "alert"),
                "detail": f"{al.get('status','-')} {al.get('channel','-')}",
            })
        # Evidence captured.
        for ev in db.list_evidence_files(self._conn, incident_id=incident_id):
            entries.append({
                "ts": ev.get("captured_at"),
                "kind": "EVIDENCE",
                "label": ev.get("capture_type") or ev.get("kind"),
                "detail": f"camera={ev.get('camera','-')}",
            })
        # Investigation notes (operator narrative).
        for nt in db.list_incident_notes(self._conn, incident_id):
            entries.append({
                "ts": nt.get("created_at"),
                "kind": "NOTE",
                "label": f"note by {nt.get('author','system')}",
                "detail": (nt.get("note") or "")[:200],
            })

        # Deterministic chronological sort (oldest first; stable tie-break).
        entries.sort(key=lambda e: (e.get("ts") or "", e["kind"], e["label"]))
        total = len(entries)
        start = page * page_size
        return {
            "incident": {
                "incident_id": incident_id,
                "event_type": incident.get("event_type"),
                "status": incident.get("status"),
                "temporal_state": incident.get("temporal_state"),
                "severity": incident.get("severity"),
            },
            "rows": entries[start:start + page_size],
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": (total // page_size) + (1 if total % page_size else 0),
        }
