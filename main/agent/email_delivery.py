"""Send agent answers via SMTP (optional — configured in .env)."""

from __future__ import annotations

import os
import re
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _smtp_configured() -> bool:
    return bool(
        os.environ.get("SMTP_HOST")
        and os.environ.get("SMTP_FROM")
    )


def send_answer_email(to_email: str, subject: str, body: str) -> dict[str, Any]:
    """Send plain-text email. Returns ok/status/error_message."""
    to_email = (to_email or "").strip()
    subject = (subject or "Financial research answer").strip()[:200]
    body = (body or "").strip()

    if not EMAIL_RE.match(to_email):
        return {
            "tool": "send_email",
            "ok": False,
            "status": "invalid_email",
            "error_message": f"Invalid email address: {to_email!r}",
        }
    if not body:
        return {
            "tool": "send_email",
            "ok": False,
            "status": "empty_body",
            "error_message": "Email body is empty.",
        }
    if not _smtp_configured():
        return {
            "tool": "send_email",
            "ok": False,
            "status": "not_configured",
            "error_message": (
                "SMTP is not configured. Set SMTP_HOST, SMTP_FROM, and optionally "
                "SMTP_USER, SMTP_PASSWORD, SMTP_PORT in .env."
            ),
        }

    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    from_addr = os.environ["SMTP_FROM"]
    user = os.environ.get("SMTP_USER") or from_addr
    password = os.environ.get("SMTP_PASSWORD", "")
    use_tls = os.environ.get("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_email
    msg.set_content(body)

    try:
        if port == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=context) as server:
                if password:
                    server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as server:
                if use_tls:
                    server.starttls(context=ssl.create_default_context())
                if password:
                    server.login(user, password)
                server.send_message(msg)
        return {
            "tool": "send_email",
            "ok": True,
            "status": "sent",
            "to_email": to_email,
            "subject": subject,
            "body_chars": len(body),
        }
    except Exception as exc:
        return {
            "tool": "send_email",
            "ok": False,
            "status": "error",
            "to_email": to_email,
            "subject": subject,
            "error_message": str(exc),
        }
