"""Alert engine (Phase 31).

Pluggable channel adapters: log, dashboard (alerts table), email.
Rule matching with cooldown and deduplication.
"""

from __future__ import annotations

import logging
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText

import config
from src import database as db
from src.security_events import severity_ge, severity_to_index

logger = logging.getLogger("cctv.alerts")


class AlertEngine:
    """Evaluates security events against alert rules and dispatches.

    Phase 32 hardening: full alert lifecycle (PENDING/DELIVERED/FAILED/
    RETRYING/ACKNOWLEDGED/RESOLVED/ESCALATED), automatic retry of failed
    deliveries, and escalation of un-addressed high-severity alerts.
    """

    def __init__(self, conn, *, max_retries: int = 3,
                 retry_delay_sec: int = 60,
                 escalation_sec: int = 900):
        self._conn = conn
        self._last_fire: dict[str, float] = {}  # rule_name -> last-fire epoch
        self._max_retries = max_retries
        self._retry_delay_sec = retry_delay_sec
        self._escalation_sec = escalation_sec

    def evaluate(self, event_dict: dict) -> list[dict]:
        """Check event against all enabled rules; dispatch matching ones.

        Returns list of dispatched alert dicts.
        """
        rules = db.list_alert_rules(self._conn)
        dispatched: list[dict] = []
        now = time.time()
        now_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for rule in rules:
            if not rule.get("enabled", 1):
                continue

            # Event type match
            rule_event = (rule.get("event_type") or "").strip()
            if rule_event and rule_event != event_dict.get("event_type"):
                continue

            # Severity match
            event_sev = event_dict.get("severity") or "INFO"
            min_sev = rule.get("min_severity") or "INFO"
            if not severity_ge(event_sev, min_sev):
                continue

            # Schedule window
            start = rule.get("schedule_start")
            end = rule.get("schedule_end")
            if start and end:
                from src.domain import is_time_in_window
                now_min = datetime.now().hour * 60 + datetime.now().minute
                if not is_time_in_window(now_min, start, end):
                    continue

            # Cooldown
            rule_name = rule.get("rule_name", "")
            cooldown = rule.get("cooldown_sec", 300)
            last = self._last_fire.get(rule_name, 0.0)
            if (now - last) < cooldown:
                continue

            # Dispatch to channels
            channels = [c.strip() for c in (rule.get("channels") or "log").split(",")]
            for ch in channels:
                alert = {
                    "created_at": now_dt,
                    "rule_name": rule_name,
                    "event_type": event_dict.get("event_type"),
                    "severity": event_sev,
                    "camera": event_dict.get("camera"),
                    "zone": event_dict.get("zone"),
                    "employee_id": event_dict.get("employee_id"),
                    "incident_id": event_dict.get("incident_id"),
                    "channel": ch,
                    "message": self._build_message(event_dict, rule),
                }
                delivered, error = self._dispatch(ch, alert)
                alert["delivered"] = delivered
                alert["status"] = "DELIVERED" if delivered else "FAILED"
                alert["attempts"] = 1
                alert["error"] = error
                alert["last_attempt_epoch"] = now
                _id = db.insert_alert(self._conn, alert)
                db.set_alert_status(
                    self._conn, _id, alert["status"],
                    error=error, attempts=1, last_attempt_epoch=now)
                dispatched.append(alert)

            self._last_fire[rule_name] = now

        return dispatched

    def retry_pending(self) -> list[dict]:
        """Re-attempt delivery of FAILED/RETRYING alerts that have retries left.

        Returns the list of alerts retried this cycle.
        """
        now = time.time()
        retried: list[dict] = []
        rows = db.query_alerts_filtered(self._conn, status="FAILED", limit=500)
        # Coalesce on id presence via direct query of attempts column.
        for row in rows:
            alert_id = row.get("id")
            if not alert_id:
                continue
            attempts = int(row.get("attempts") or 1)
            if attempts > self._max_retries:
                continue
            last = float(row.get("last_attempt_epoch") or 0.0)
            if (now - last) < self._retry_delay_sec:
                continue
            delivered, error = self._dispatch(row.get("channel") or "log", row)
            new_attempts = attempts + 1
            if delivered:
                db.set_alert_status(
                    self._conn, alert_id, "DELIVERED", error="",
                    attempts=new_attempts, last_attempt_epoch=now)
            else:
                db.set_alert_status(
                    self._conn, alert_id, "FAILED", error=error,
                    attempts=new_attempts, last_attempt_epoch=now)
            row["delivered"] = delivered
            row["attempts"] = new_attempts
            retried.append(row)
        return retried

    def escalate_stale(self) -> list[dict]:
        """Escalate un-addressed HIGH/CRITICAL alerts older than threshold.

        Acknowledged / resolved / escalated alerts are never touched.  Returns
        list of escalated alert ids.
        """
        escalated: list[dict] = []
        now = time.time()
        for alert in db.query_alerts_filtered(self._conn, limit=1000):
            status = alert.get("status") or "SENT"
            if status in ("ACKNOWLEDGED", "RESOLVED", "ESCALATED"):
                continue
            sev = alert.get("severity") or "INFO"
            if sev not in ("HIGH", "CRITICAL"):
                continue
            last = float(alert.get("last_attempt_epoch") or 0.0) or \
                _parse_dt_epoch(alert.get("created_at"))
            if (now - last) < self._escalation_sec:
                continue
            db.escalate_alert(self._conn, alert.get("id"))
            escalated.append(alert)
        return escalated

    def acknowledge(self, alert_id: int) -> None:
        db.acknowledge_alert(self._conn, alert_id)

    def resolve(self, alert_id: int) -> None:
        db.resolve_alert(self._conn, alert_id)

    def escalate(self, alert_id: int) -> None:
        db.escalate_alert(self._conn, alert_id)

    # ------------------------------------------------------------------
    # Phase 33 -- incident-level alert intelligence (B5) & health (B6) 
    # ------------------------------------------------------------------

    def incident_alert(self, incident: dict, *, min_severity: str = "MEDIUM",
                       dedup_sec: int = 3600) -> dict | None:
        """Fire ONE alert per incident (not per event) to avoid spam.

        ``incident`` is an incident row.  If an alert for this incident already
        exists and is within ``dedup_sec`` of the last, it is suppressed
        (returns None) -- this is the incident-level dedup that prevents alert
        storms from correlated event bursts.  Returns the created alert dict or
        None when suppressed / below the severity floor.
        """
        from src.security_events import severity_ge
        sev = incident.get("severity") or "INFO"
        if not severity_ge(sev, min_severity):
            return None
        # Dedup: no more than one alert per incident within the window.
        existing = db.query_alerts_filtered(self._conn, incident_id=incident.get("incident_id"),
                                            limit=1)
        if existing:
            try:
                last_dt = datetime.strptime(str(existing[0].get("created_at"))[:19],
                                           "%Y-%m-%d %H:%M:%S")
                now_dt = datetime.now()
                if (now_dt - last_dt).total_seconds() < dedup_sec:
                    return None
            except (ValueError, TypeError):
                pass
        alert = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rule_name": "incident-alert",
            "event_type": incident.get("event_type"),
            "severity": sev,
            "camera": incident.get("camera"),
            "zone": incident.get("zone"),
            "employee_id": incident.get("employee_id"),
            "incident_id": incident.get("incident_id"),
            "channel": "dashboard",
            "message": (f"INCIDENT {incident.get('incident_id')} "
                        f"{incident.get('event_type')} sev={sev} "
                        f"occurrences={incident.get('occurrences','?')}"),
        }
        alert["delivered"] = True
        alert["status"] = "DELIVERED"
        alert["attempts"] = 1
        alert["error"] = ""
        alert["last_attempt_epoch"] = time.time()
        _id = db.insert_alert(self._conn, alert)
        db.set_alert_status(self._conn, _id, "DELIVERED", attempts=1,
                            last_attempt_epoch=time.time())
        return alert

    def health(self) -> dict:
        """Alert delivery health snapshot (B6) for System Health 2.0."""
        rows = db.query_alerts_filtered(self._conn, limit=100000)
        total = len(rows)
        by_status: dict[str, int] = {}
        for a in rows:
            s = a.get("status") or "SENT"
            by_status[s] = by_status.get(s, 0) + 1
        failed = by_status.get("FAILED", 0) + by_status.get("RETRYING", 0)
        escalated = by_status.get("ESCALATED", 0)
        delivery_pct = (100.0 * (total - failed) / total) if total else 100.0
        return {
            "total": total,
            "by_status": by_status,
            "failed": failed,
            "escalated": escalated,
            "delivery_pct": round(delivery_pct, 2),
        }

    def _dispatch(self, channel: str, alert: dict) -> tuple[bool, str]:
        """Send to a specific channel.  Returns (success, error)."""
        if channel == "log":
            logger.info(
                "[ALERT] %s severity=%s camera=%s zone=%s incident=%s",
                alert.get("event_type", "?"),
                alert.get("severity", "?"),
                alert.get("camera", "?"),
                alert.get("zone", "-"),
                alert.get("incident_id", "-"),
            )
            return True, ""
        if channel == "dashboard":
            # Already persisted via insert_alert; dashboard reads the table
            return True, ""
        if channel == "email":
            # Honest email delivery adapter.  If SMTP is not configured this
            # returns a clear failure so the alert is marked FAILED (never a
            # false "sent").  The SMTP password is never logged.
            recipients = config.parse_recipients()
            if not config.SENDER_EMAIL or not recipients:
                return False, "email not configured (SENDER_EMAIL / RECIPIENT_EMAILS)"
            try:
                _send_alert_email(
                    subject=(f"[CCTV ALERT] {alert.get('event_type', 'event')} "
                             f"({alert.get('severity', '?')})"),
                    body=_email_body(alert),
                    recipients=recipients,
                    timeout=config.EMAIL_TIMEOUT_SEC,
                )
                return True, ""
            except (smtplib.SMTPException, OSError) as exc:
                logger.error("[ALERT] email delivery failed: %s",
                             type(exc).__name__)
                return False, f"email delivery failed: {type(exc).__name__}"
        logger.debug("[ALERT] unknown channel: %s", channel)
        return False, "unknown channel"

    def _build_message(self, event: dict, rule: dict) -> str:
        parts = [event.get("event_type", "EVENT")]
        if event.get("camera"):
            parts.append(f"camera={event['camera']}")
        if event.get("zone"):
            parts.append(f"zone={event['zone']}")
        if event.get("employee_id"):
            parts.append(f"employee={event['employee_id']}")
        parts.append(f"severity={event.get('severity', '?')}")
        return " ".join(parts)


def _email_body(alert: dict) -> str:
    """Build a plain-text email body for an alert (no secrets)."""
    lines = [
        "Security alert from the CCTV monitoring system.",
        "",
        f"Event:      {alert.get('event_type', '?')}",
        f"Severity:   {alert.get('severity', '?')}",
        f"Camera:     {alert.get('camera') or '-'}",
        f"Zone:       {alert.get('zone') or '-'}",
        f"Employee:   {alert.get('employee_id') or '-'}",
        f"Incident:   {alert.get('incident_id') or '-'}",
        f"Time:       {alert.get('created_at') or '-'}",
        "",
        alert.get("message") or "No additional details.",
        "",
        "--\nAutomated message from CCTV Employee Productivity Monitoring System.",
    ]
    return "\n".join(lines)


def _send_alert_email(*, subject: str, body: str, recipients: list[str],
                      timeout: float) -> None:
    """Send a plain-text alert email over SMTP (TLS when not port 25).

    Raises ``smtplib.SMTPException`` / ``OSError`` on delivery failure.  The
    SMTP password is used only for authentication and never logged.
    """
    msg = MIMEText(body, "plain")
    msg["From"] = config.SENDER_EMAIL
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT,
                      timeout=timeout) as server:
        server.ehlo()
        if config.SMTP_PORT != 25:
            server.starttls()
            server.ehlo()
        if config.SENDER_EMAIL and config.SENDER_PASSWORD:
            server.login(config.SENDER_EMAIL, config.SENDER_PASSWORD)
        server.sendmail(config.SENDER_EMAIL, recipients, msg.as_string())


def _parse_dt_epoch(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.strptime(str(value)[:19], "%Y-%m-%d %H:%M:%S").timestamp()
    except (ValueError, TypeError):
        return 0.0
