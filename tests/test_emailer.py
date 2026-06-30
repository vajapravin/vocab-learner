"""Tests for email sending with mocked SMTP."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vocab_learner.config import Settings
from vocab_learner.emailer import (
    EmailConfigError,
    EmailSender,
    EmailSendError,
    EmailSettings,
)


def _base_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "openai_api_key": "sk-test",
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "smtp_user": "sender@example.com",
        "smtp_password": "app-password",
        "email_from": "sender@example.com",
        "email_to": "recipient@example.com",
        "_env_file": None,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


class TestEmailSettings:
    def test_from_settings_returns_populated_when_complete(self) -> None:
        settings = _base_settings()
        cfg = EmailSettings.from_settings(settings)

        assert cfg.smtp_host == "smtp.example.com"
        assert cfg.smtp_port == 587
        assert cfg.email_to == "recipient@example.com"

    def test_from_settings_raises_when_incomplete(self) -> None:
        settings = _base_settings(smtp_password=None, email_to=None)

        with pytest.raises(EmailConfigError) as excinfo:
            EmailSettings.from_settings(settings)

        # Both missing vars should be mentioned in the error
        msg = str(excinfo.value)
        assert "SMTP_PASSWORD" in msg
        assert "EMAIL_TO" in msg


class TestEmailSender:
    @pytest.fixture
    def markdown_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "session_test.md"
        p.write_text("# Test Session\n\n## Words\n\n### 1. alpha\n\nA test word.\n")
        return p

    def test_sends_email_with_expected_headers(self, markdown_file: Path) -> None:
        cfg = EmailSettings.from_settings(_base_settings())
        sender = EmailSender(cfg)

        with patch("smtplib.SMTP") as smtp_class:
            smtp_instance = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp_instance

            sender.send_file(markdown_file)

        # Verify SMTP flow
        smtp_class.assert_called_once_with("smtp.example.com", 587)
        smtp_instance.starttls.assert_called_once()
        smtp_instance.login.assert_called_once_with("sender@example.com", "app-password")
        smtp_instance.send_message.assert_called_once()

        # Inspect the message that was sent
        sent_message = smtp_instance.send_message.call_args[0][0]
        assert sent_message["From"] == "sender@example.com"
        assert sent_message["To"] == "recipient@example.com"
        assert sent_message["Subject"].startswith("Vocab Session —")

    def test_email_contains_html_and_markdown_attachment(self, markdown_file: Path) -> None:
        cfg = EmailSettings.from_settings(_base_settings())
        sender = EmailSender(cfg)

        with patch("smtplib.SMTP") as smtp_class:
            smtp_instance = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp_instance
            sender.send_file(markdown_file)

        sent_message = smtp_instance.send_message.call_args[0][0]

        # Collect content-types across all parts
        content_types = {part.get_content_type() for part in sent_message.walk()}
        assert "text/plain" in content_types  # fallback
        assert "text/html" in content_types  # main body
        assert "text/markdown" in content_types  # attachment

    def test_smtp_failure_raises_send_error(self, markdown_file: Path) -> None:
        import smtplib

        cfg = EmailSettings.from_settings(_base_settings())
        sender = EmailSender(cfg)

        with patch("smtplib.SMTP") as smtp_class:
            smtp_class.side_effect = smtplib.SMTPException("connection refused")

            with pytest.raises(EmailSendError, match="SMTP delivery failed"):
                sender.send_file(markdown_file)

    def test_html_body_contains_session_content(self, markdown_file: Path) -> None:
        """The Markdown-to-HTML conversion should preserve the content."""
        cfg = EmailSettings.from_settings(_base_settings())
        sender = EmailSender(cfg)

        with patch("smtplib.SMTP") as smtp_class:
            smtp_instance = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp_instance
            sender.send_file(markdown_file)

        sent_message = smtp_instance.send_message.call_args[0][0]

        # Find the HTML part
        html_part = next(
            (part for part in sent_message.walk() if part.get_content_type() == "text/html"),
            None,
        )
        assert html_part is not None
        html_content = html_part.get_content()
        assert "<h1>Test Session</h1>" in html_content
        assert "<h3>1. alpha</h3>" in html_content


class TestSubjectDerivation:
    def test_standard_session_filename(self, tmp_path: Path) -> None:
        cfg = EmailSettings.from_settings(_base_settings())
        sender = EmailSender(cfg)

        session_file = tmp_path / "session_20260826_203242.md"
        session_file.write_text("# Test")

        with patch("smtplib.SMTP") as smtp_class:
            smtp_instance = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp_instance
            sender.send_file(session_file)

        subject = smtp_instance.send_message.call_args[0][0]["Subject"]
        assert subject == "Vocab Session — 20260826_203242"

    def test_nonstandard_filename_falls_back_to_stem(self, tmp_path: Path) -> None:
        cfg = EmailSettings.from_settings(_base_settings())
        sender = EmailSender(cfg)

        session_file = tmp_path / "my_custom_session.md"
        session_file.write_text("# Test")

        with patch("smtplib.SMTP") as smtp_class:
            smtp_instance = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp_instance
            sender.send_file(session_file)

        subject = smtp_instance.send_message.call_args[0][0]["Subject"]
        assert subject == "Vocab Session — my_custom_session"
