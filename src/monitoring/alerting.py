"""
Alerting system — dispatches alerts through configured channels.
Supports logging, email, Slack webhook, and PagerDuty.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Alert:
    """Represents a monitoring alert."""
    level: str          # INFO | WARNING | CRITICAL
    title: str
    message: str
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class AlertManager:
    """Dispatches monitoring alerts to configured channels.

    Channels configured via env vars or constructor arguments.

    Example:
        >>> am = AlertManager(channels=["log", "slack"])
        >>> am.send(Alert(level="WARNING", title="Drift Detected", message="20% of features drifted"))
    """

    def __init__(
        self,
        channels: Optional[list[str]] = None,
        slack_webhook_url: Optional[str] = None,
        email_recipients: Optional[list[str]] = None,
        min_level: str = "WARNING",
    ) -> None:
        self.channels = channels or ["log"]
        self.slack_webhook_url = slack_webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.email_recipients = email_recipients or []
        self.min_level = min_level

        self._level_order = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}

    def send(self, alert: Alert) -> None:
        """Send alert to all configured channels.

        Args:
            alert: Alert object with level, title, message, metadata.
        """
        if self._level_order.get(alert.level, 0) < self._level_order.get(self.min_level, 1):
            return  # Below minimum level

        for channel in self.channels:
            try:
                if channel == "log":
                    self._send_log(alert)
                elif channel == "slack":
                    self._send_slack(alert)
                elif channel == "email":
                    self._send_email(alert)
                elif channel == "pagerduty":
                    self._send_pagerduty(alert)
                else:
                    log.warning(f"Unknown alert channel: {channel}")
            except Exception as e:
                log.error(f"Failed to send alert via '{channel}': {e}")

    def _send_log(self, alert: Alert) -> None:
        msg = f"[ALERT:{alert.level}] {alert.title} — {alert.message}"
        if alert.level == "CRITICAL":
            log.critical(msg)
        elif alert.level == "WARNING":
            log.warning(msg)
        else:
            log.info(msg)

    def _send_slack(self, alert: Alert) -> None:
        if not self.slack_webhook_url:
            log.warning("Slack webhook URL not configured.")
            return

        import urllib.request, json
        payload = {
            "text": f"*[{alert.level}]* {alert.title}\n{alert.message}",
            "attachments": [{
                "color": "#ff0000" if alert.level == "CRITICAL" else "#ffaa00",
                "text": json.dumps(alert.metadata, indent=2),
            }],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.slack_webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            log.info(f"Slack alert sent: {resp.status}")

    def _send_email(self, alert: Alert) -> None:
        import smtplib
        from email.mime.text import MIMEText

        smtp_host = os.getenv("SMTP_HOST", "localhost")
        smtp_port = int(os.getenv("SMTP_PORT", 25))
        sender = os.getenv("ALERT_EMAIL_FROM", "automl@alerts.local")

        body = f"{alert.title}\n\n{alert.message}\n\n{alert.metadata}"
        msg = MIMEText(body)
        msg["Subject"] = f"[AutoMLPro:{alert.level}] {alert.title}"
        msg["From"] = sender
        msg["To"] = ", ".join(self.email_recipients)

        with smtplib.SMTP(smtp_host, smtp_port, timeout=5) as smtp:
            smtp.sendmail(sender, self.email_recipients, msg.as_string())
            log.info("Email alert sent.")

    def _send_pagerduty(self, alert: Alert) -> None:
        pd_key = os.getenv("PAGERDUTY_ROUTING_KEY", "")
        if not pd_key:
            log.warning("PagerDuty routing key not configured.")
            return

        import json, urllib.request
        payload = {
            "routing_key": pd_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"{alert.title}: {alert.message}",
                "severity": alert.level.lower(),
                "custom_details": alert.metadata,
            },
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            "https://events.pagerduty.com/v2/enqueue",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            log.info(f"PagerDuty event sent: {resp.status}")
