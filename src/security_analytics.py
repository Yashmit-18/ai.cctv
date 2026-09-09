"""Security analytics & operations metrics (Phase 32).

Read-only, aggregate analytics over security incidents, events, alerts and
evidence.  These are OPERATIONAL/security-health metrics (volumes, hotspots,
funnels, degradation).  They deliberately do NOT profile individual employees
or use security data to alter productivity conclusions.
"""

from __future__ import annotations

from datetime import datetime, timedelta


class SecurityAnalytics:
    """Aggregate operational analytics (read-only)."""

    def __init__(self, conn):
        self._conn = conn

    def _count(self, table: str, column: str, where: str = "",
               params: tuple = ()) -> list[dict]:
        sql = f"SELECT {column} AS k, COUNT(*) AS n FROM {table}"
        if where:
            sql += " WHERE " + where
        sql += f" GROUP BY {column} ORDER BY n DESC"
        return [{"key": r[0], "count": r[1]}
                for r in self._conn.execute(sql, params).fetchall()]

    def incident_by_status(self) -> list[dict]:
        return self._count("incidents", "status")

    def incident_by_severity(self) -> list[dict]:
        return self._count("incidents", "severity")

    def incident_by_type(self) -> list[dict]:
        return self._count("incidents", "event_type")

    def incidents_by_day(self, n_days: int = 14) -> list[dict]:
        """Incident counts per calendar day (label/value)."""
        since = (datetime.now() - timedelta(days=n_days)).strftime("%Y-%m-%d")
        rows = self._conn.execute(
            "SELECT substr(first_seen, 1, 10) AS d, COUNT(*) AS n "
            "FROM incidents WHERE first_seen >= ? "
            "GROUP BY d ORDER BY d", (since,)
        ).fetchall()
        return [{"day": r[0], "count": r[1]} for r in rows]

    def hotspots_by_zone(self) -> list[dict]:
        """Security hotspots: incidents grouped by zone, most frequent first."""
        return self._count("incidents", "zone")

    def events_by_type(self, limit: int = 20) -> list[dict]:
        return self._count("security_events", "event_type")[:limit]

    def alert_funnel(self) -> list[dict]:
        return self._count("alerts", "status")

    def evidence_totals(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*) AS files, "
            "COALESCE(SUM(size_bytes), 0) AS bytes FROM evidence_files"
        ).fetchone()
        files = int(row[0] or 0)
        bytes_ = int(row[1] or 0)
        return {"files": files, "bytes": bytes_, "mb": round(bytes_ / (1024 * 1024), 2)}

    def incidents_open(self) -> int:
        r = self._conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE status = 'OPEN'").fetchone()
        return int(r[0] or 0)

    def alerts_pending(self) -> int:
        r = self._conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE status IN ('FAILED','PENDING')"
        ).fetchone()
        return int(r[0] or 0)

    def snapshot(self) -> dict:
        """One combined operational snapshot for a dashboard."""

        def _k(rows):
            return [{"key": r["key"], "count": r["count"]} for r in rows]

        return {
            "incidents": {
                "by_status": _k(self.incident_by_status()),
                "by_severity": _k(self.incident_by_severity()),
                "by_type": _k(self.incident_by_type()),
                "open": self.incidents_open(),
                "by_day": self.incidents_by_day(),
            },
            "hotspots_by_zone": _k(self.hotspots_by_zone()),
            "alerts": {
                "by_status": _k(self.alert_funnel()),
                "pending": self.alerts_pending(),
            },
            "evidence": self.evidence_totals(),
            "events_by_type": _k(self.events_by_type()),
        }
