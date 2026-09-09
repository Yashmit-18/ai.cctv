"""Automated EOD report email dispatcher via SMTP (Phase 13).

Features
--------
* SMTP TLS + authentication + multiple recipients.
* Attachment (the daily ``.xlsx`` report).
* Configurable retry with delay: the daemon does **not** endlessly retry --
  up to ``EMAIL_MAX_RETRIES`` attempts with ``EMAIL_RETRY_DELAY_SEC`` between
  them, then the delivery is marked failed.
* Success/failure status is tracked and readable by the dashboard so email
  failures are visible, not silent.
* SMTP password is never logged.
"""

import logging
import os
import smtplib
import threading
import time
from datetime import date
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import (
    EMAIL_MAX_RETRIES,
    EMAIL_RETRY_DELAY_SEC,
    EMAIL_TIMEOUT_SEC,
    SENDER_EMAIL,
    SENDER_PASSWORD,
    SMTP_PORT,
    SMTP_SERVER,
    parse_recipients,
)

logger = logging.getLogger("cctv.notifier")

# ----------------------------------------------------------------------
# Exposed delivery status (readable by the dashboard)
# ----------------------------------------------------------------------
_LOCK = threading.Lock()
_LAST_STATUS: dict = {
    "ok": None,          # True/False/None (no attempt yet)
    "report": None,
    "attempts": 0,
    "message": "",
    "at": None,
}


def last_email_status() -> dict:
    """Thread-safe snapshot of the last delivery attempt."""
    with _LOCK:
        return dict(_LAST_STATUS)


def _set_status(**kw):
    with _LOCK:
        _LAST_STATUS.update(kw)
        _LAST_STATUS["at"] = time.strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------

def _build_summary_html(report_path: str) -> str:
    basename = os.path.basename(report_path)
    date_part = basename.replace("daily_report_", "").replace(".xlsx", "")
    return (
        "<html><body>"
        "<h2>CCTV Productivity Tracker -- Daily Report</h2>"
        f"<p>Please find attached the productivity report for "
        f"<strong>{date_part}</strong>.</p>"
        "<p>This report was generated automatically by the CCTV "
        "Employee Productivity Monitoring System.</p>"
        "<hr>"
        "<p style='font-size:12px;color:#888;'>This is an automated message. "
        "Do not reply directly.</p>"
        "</body></html>"
    )


def _build_summary_text(report_path: str) -> str:
    basename = os.path.basename(report_path)
    date_part = basename.replace("daily_report_", "").replace(".xlsx", "")
    return (
        f"CCTV Productivity Tracker -- Daily Report\n"
        f"Report date: {date_part}\n\n"
        "Please find the daily productivity report attached.\n\n"
        "--\nAutomated message from CCTV Employee Productivity Monitoring System."
    )


def _send_once(report_path: str, recipients: list[str], timeout: float) -> None:
    """Perform a single SMTP send.  Raises on failure (no retry here).

    ``report_path`` may be empty/None current to send a minimal test message
    without an attachment.
    """
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = (
        f"CCTV Productivity Report -- {date.today().strftime('%Y-%m-%d')}"
    )

    if report_path:
        msg.attach(MIMEText(_build_summary_html(report_path), "html"))
        msg.attach(MIMEText(_build_summary_text(report_path), "plain"))
    else:
        msg.attach(MIMEText(
            "<html><body><h3>CCTV Productivity Monitor</h3>"
            "<p>This is a test email confirming SMTP delivery works.</p>"
            "</body></html>", "html"))
        msg.attach(MIMEText(
            "CCTV Productivity Monitor -- test email confirming SMTP delivery.", "plain"))

    if report_path:
        with open(report_path, "rb") as f:
            part = MIMEBase("application",
                            "vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{os.path.basename(report_path)}"',
        )
        msg.attach(part)

    smtp = smtplib.SMTP_SSL if SMTP_PORT == 465 else smtplib.SMTP
    with smtp(SMTP_SERVER, SMTP_PORT, timeout=timeout) as server:
        server.ehlo()
        if SMTP_PORT not in (465, 25):
            server.starttls()
            server.ehlo()
        if SENDER_EMAIL and SENDER_PASSWORD:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipients, msg.as_string())


def send_daily_report(
    report_path: str,
    recipient_emails: list[str] | None = None,
    max_retries: int | None = None,
    retry_delay: float | None = None,
) -> bool:
    """Send the Excel report to configured recipients with bounded retries.

    Returns
    -------
    bool -- True if delivered (eventually), False otherwise.
    """
    recipients = recipient_emails or parse_recipients()
    max_retries = EMAIL_MAX_RETRIES if max_retries is None else max_retries
    retry_delay = EMAIL_RETRY_DELAY_SEC if retry_delay is None else retry_delay

    if not SENDER_EMAIL or not recipients:
        msg = "Email not sent: SENDER_EMAIL or RECIPIENT_EMAILS not configured."
        logger.warning(msg)
        _set_status(ok=False, report=report_path, attempts=0, message=msg)
        return False

    if not os.path.isfile(report_path):
        msg = f"Report file not found: {report_path}"
        logger.error(msg)
        _set_status(ok=False, report=report_path, attempts=0, message=msg)
        return False

    attempts = 0
    last_exc = None
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            logger.info(
                "Sending report to %s (attempt %d/%d) ...",
                recipients, attempt, max_retries,
            )
            _send_once(report_path, recipients, EMAIL_TIMEOUT_SEC)
            logger.info("Report email sent successfully to %s", recipients)
            _set_status(ok=True, report=report_path, attempts=attempts,
                        message="sent")
            return True
        except Exception as exc:  # noqa: BLE001 - report structured failure
            last_exc = exc
            retryable = isinstance(exc, (smtplib.SMTPException, OSError))
            logger.error("Email attempt %d failed: %s", attempt,
                         type(exc).__name__)
            # Non-retryable (e.g. auth/local file) -> bail early.
            if not retryable:
                break
            if attempt < max_retries:
                logger.info("Retrying in %.1fs ...", retry_delay)
                time.sleep(retry_delay)

    msg = f"Email failed after {attempts} attempt(s): {last_exc}"
    logger.error(msg)
    _set_status(ok=False, report=report_path, attempts=attempts, message=msg)
    return False


def send_test_email(report_path: str | None = None) -> bool:
    """Send a minimal test (no attachment required if none given)."""
    recipients = parse_recipients()
    if not SENDER_EMAIL or not recipients:
        msg = "Email not sent: SENDER_EMAIL or RECIPIENT_EMAILS not configured."
        logger.warning(msg)
        _set_status(ok=False, report=report_path or "", attempts=0, message=msg)
        return False
    attempts = 0
    last_exc = None
    for attempt in range(1, max(1, EMAIL_MAX_RETRIES) + 1):
        attempts = attempt
        try:
            _send_once(report_path or "", recipients, EMAIL_TIMEOUT_SEC)
            logger.info("Test email sent successfully to %s", recipients)
            _set_status(ok=True, report=report_path or "", attempts=attempts,
                        message="sent")
            return True
        except Exception as exc:  # noqa: BLE001 - report structured failure
            last_exc = exc
            retryable = isinstance(exc, (smtplib.SMTPException, OSError))
            logger.error("Test email attempt %d failed: %s", attempt,
                         type(exc).__name__)
            if not retryable or attempt >= max(1, EMAIL_MAX_RETRIES):
                break
            time.sleep(EMAIL_RETRY_DELAY_SEC)
    msg = f"Test email failed after {attempts} attempt(s): {last_exc}"
    logger.error(msg)
    _set_status(ok=False, report=report_path or "", attempts=attempts, message=msg)
    return False
