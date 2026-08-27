"""Email delivery of a rendered Session as HTML."""

from __future__ import annotations

import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

import markdown

from vocab_learner.config import Settings
from vocab_learner.logging_setup import get_logger

log = get_logger(__name__)


class EmailConfigError(ValueError):
    """Raised when email is requested but config is incomplete."""


class EmailSendError(RuntimeError):
    """Raised when SMTP delivery fails after config checks pass."""


@dataclass(frozen=True)
class EmailSettings:
    """The subset of Settings needed for email — extracted so we validate once."""

    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    email_from: str
    email_to: str

    @classmethod
    def from_settings(cls, settings: Settings) -> EmailSettings:
        missing: list[str] = []
        if not settings.smtp_host:
            missing.append("VOCAB_LEARNER_SMTP_HOST")
        if not settings.smtp_user:
            missing.append("VOCAB_LEARNER_SMTP_USER")
        if not settings.smtp_password:
            missing.append("VOCAB_LEARNER_SMTP_PASSWORD")
        if not settings.email_from:
            missing.append("VOCAB_LEARNER_EMAIL_FROM")
        if not settings.email_to:
            missing.append("VOCAB_LEARNER_EMAIL_TO")
        if missing:
            raise EmailConfigError("Email requested but missing config: " + ", ".join(missing))
        # mypy: after the check above, all values are non-None
        return cls(
            smtp_host=settings.smtp_host,  # type: ignore[arg-type]
            smtp_port=settings.smtp_port,
            smtp_user=settings.smtp_user,  # type: ignore[arg-type]
            smtp_password=settings.smtp_password,  # type: ignore[arg-type]
            email_from=settings.email_from,  # type: ignore[arg-type]
            email_to=settings.email_to,  # type: ignore[arg-type]
        )


class EmailSender:
    """
    Sends a rendered Session as an HTML email. Markdown source is attached
    for archival. Fails loud (raises) — the caller decides whether to soft-fail.
    """

    def __init__(self, email_settings: EmailSettings) -> None:
        self._cfg = email_settings

    def send_file(self, markdown_path: Path) -> None:
        """
        Send a previously-generated session Markdown file. Subject is
        derived from the filename since we no longer have the Session object.
        """
        if not markdown_path.exists():
            raise FileNotFoundError(markdown_path)

        markdown_source = markdown_path.read_text(encoding="utf-8")
        html_body = self._markdown_to_html(markdown_source)
        subject = self._subject_from_filename(markdown_path)

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self._cfg.email_from
        message["To"] = self._cfg.email_to
        message.set_content(
            f"Vocabulary session from {markdown_path.name}. "
            "HTML version in body; full Markdown attached."
        )
        message.add_alternative(html_body, subtype="html")
        message.add_attachment(
            markdown_source.encode("utf-8"),
            maintype="text",
            subtype="markdown",
            filename=markdown_path.name,
        )

        log.info(
            "email_sending",
            to=self._cfg.email_to,
            subject=subject,
            file=str(markdown_path),
        )

        try:
            with smtplib.SMTP(self._cfg.smtp_host, self._cfg.smtp_port) as smtp:
                smtp.starttls()
                smtp.login(self._cfg.smtp_user, self._cfg.smtp_password)
                smtp.send_message(message)
        except (smtplib.SMTPException, OSError) as e:
            log.error("email_send_failed", error=str(e))
            raise EmailSendError(f"SMTP delivery failed: {e}") from e

        log.info("email_sent", to=self._cfg.email_to)

    @staticmethod
    def _subject_from_filename(path: Path) -> str:
        """
        Extract a human-readable subject from a session filename.
        Expects the standard `session_YYYYMMDD_HHMMSS.md` format;
        falls back to the raw stem if it doesn't match.
        """
        stem = path.stem  # session_20260826_203242
        if stem.startswith("session_"):
            return f"Vocab Session — {stem.removeprefix('session_')}"
        return f"Vocab Session — {stem}"

    @staticmethod
    def _markdown_to_html(source: str) -> str:
        """Convert Markdown to a self-contained HTML document with inline styling."""
        body = markdown.markdown(
            source,
            extensions=["extra", "sane_lists"],
        )
        # Email clients strip <style> blocks inconsistently — inline the essentials.
        # This keeps the CSS minimal and portable.
        body_style = (
            "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', "
            "Helvetica, Arial, sans-serif; "
            "line-height: 1.6; "
            "color: #24292f; "
            "max-width: 720px; "
            "margin: 0 auto; "
            "padding: 24px;"
        )
        return (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head>\n"
            '<meta charset="utf-8">\n'
            "<title>Vocabulary Session</title>\n"
            "</head>\n"
            f'<body style="{body_style}">\n'
            f"{body}\n"
            "</body>\n"
            "</html>\n"
        )
