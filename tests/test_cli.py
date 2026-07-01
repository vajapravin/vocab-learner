"""Tests for the CLI entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vocab_learner.cli import (
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_PIPELINE_ERROR,
    EXIT_USER_ERROR,
    build_parser,
    main,
)
from vocab_learner.llm.protocol import LLMError
from vocab_learner.pipeline import PipelineResult


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any real env keys and set known test values."""
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("VOCAB_LEARNER_LLM_PROVIDER", "openai")


@pytest.fixture
def real_image(tmp_path: Path) -> Path:
    """A tiny but valid JPEG on disk that the CLI can find."""
    p = tmp_path / "page.jpg"
    p.write_bytes(
        bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000"
            "ffdb004300080606070605080707070909080a0c"
            "140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20"
            "242e2720222c231c1c2837292c30313434341f27"
            "393d38323c2e333432ffc00011080001000103012200021101031101"
            "ffc4001f0000010501010101010100000000000000000102030405060708090a0b"
            "ffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f0243362728209"
            "0a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa"
            "ffda000c03010002110311003f00fbd0a28a2803ffd9"
        )
    )
    return p


class TestParser:
    def test_requires_image(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run"])

    def test_accepts_image_only(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "some/path.jpg"])
        assert args.image == Path("some/path.jpg")
        assert args.output_dir is None
        assert args.quiet is False

    def test_accepts_all_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--log-level",
                "DEBUG",
                "--quiet",
                "run",
                "page.jpg",
                "--output-dir",
                "/tmp/out",
            ]
        )
        assert args.command == "run"
        assert args.image == Path("page.jpg")
        assert args.output_dir == Path("/tmp/out")
        assert args.log_level == "DEBUG"
        assert args.quiet is True

    def test_rejects_invalid_log_level(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "page.jpg", "--log-level", "SHOUT"])


class TestMain:
    def test_missing_image_returns_user_error(
        self, clean_env: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["run", str(tmp_path / "does_not_exist.jpg")])
        assert rc == EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "Image not found" in captured.err

    def test_directory_instead_of_file_returns_user_error(
        self, clean_env: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["run", str(tmp_path)])
        assert rc == EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "Not a file" in captured.err

    def test_happy_path(
        self,
        clean_env: None,
        real_image: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """
        Patch Pipeline.run to avoid real LLM calls but exercise the whole
        CLI pipeline construction.
        """
        fake_output = tmp_path / "output" / "session_test.md"
        fake_result = MagicMock(spec=PipelineResult)
        fake_result.output_path = fake_output

        with patch("vocab_learner.cli.Pipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.run.return_value = fake_result
            rc = main(["run", str(real_image), "--output-dir", str(tmp_path / "output")])

        assert rc == EXIT_OK
        captured = capsys.readouterr()
        assert str(fake_output) in captured.out

    def test_pipeline_error_returns_pipeline_exit_code(
        self,
        clean_env: None,
        real_image: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with patch("vocab_learner.cli.Pipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.run.side_effect = LLMError("simulated")
            rc = main(["run", str(real_image)])

        assert rc == EXIT_PIPELINE_ERROR
        captured = capsys.readouterr()
        assert "simulated" in captured.err

    def test_keyboard_interrupt_returns_130(
        self,
        clean_env: None,
        real_image: Path,
    ) -> None:
        with patch("vocab_learner.cli.Pipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.run.side_effect = KeyboardInterrupt()
            rc = main(["run", str(real_image)])

        assert rc == EXIT_INTERRUPTED

    def test_success_prints_only_path_to_stdout(
        self,
        clean_env: None,
        real_image: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The stdout contract: one line, one path. That's it."""
        fake_output = tmp_path / "session_test.md"
        fake_result = MagicMock(spec=PipelineResult)
        fake_result.output_path = fake_output

        with patch("vocab_learner.cli.Pipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.run.return_value = fake_result
            main(["run", str(real_image)])

        captured = capsys.readouterr()
        # Exactly one non-empty stdout line
        lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert lines == [str(fake_output)]


class TestSendCommand:
    def test_send_missing_file_returns_user_error(
        self, clean_env: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["send", str(tmp_path / "nope.md")])
        assert rc == EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()

    def test_send_wrong_extension_returns_user_error(
        self, clean_env: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        wrong = tmp_path / "notes.txt"
        wrong.write_text("hi")

        rc = main(["send", str(wrong)])
        assert rc == EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert ".md" in captured.err

    def test_send_missing_email_config_returns_user_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Strip everything, then patch load_settings to also skip .env
        for _key in list(monkeypatch.__dict__.get("_setitem", [])):
            pass
        for key in (
            "VOCAB_LEARNER_SMTP_HOST",
            "VOCAB_LEARNER_SMTP_USER",
            "VOCAB_LEARNER_SMTP_PASSWORD",
            "VOCAB_LEARNER_EMAIL_FROM",
            "VOCAB_LEARNER_EMAIL_TO",
            "ANTHROPIC_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("VOCAB_LEARNER_LLM_PROVIDER", "openai")

        # Bypass .env file loading entirely for this test
        from vocab_learner.config import Settings

        def fake_load() -> Settings:
            return Settings(
                openai_api_key="sk-test",
                _env_file=None,
            )  # type: ignore[call-arg]

        monkeypatch.setattr("vocab_learner.cli.load_settings", fake_load)

        session_file = tmp_path / "session_test.md"
        session_file.write_text("# Test")

        rc = main(["send", str(session_file)])
        assert rc == EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "Email config error" in captured.err

    def test_send_happy_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Complete email config
        for key in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "EMAIL_FROM", "EMAIL_TO"):
            monkeypatch.setenv(f"VOCAB_LEARNER_{key}", f"test-{key}")
        monkeypatch.setenv("VOCAB_LEARNER_SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("VOCAB_LEARNER_EMAIL_FROM", "from@test.com")
        monkeypatch.setenv("VOCAB_LEARNER_EMAIL_TO", "to@test.com")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        session_file = tmp_path / "session_20260101_120000.md"
        session_file.write_text("# Test Session\n\n## Words\n\n### 1. alpha\n")

        with patch("smtplib.SMTP") as smtp_class:
            smtp_instance = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp_instance
            rc = main(["send", str(session_file)])

        assert rc == EXIT_OK
        captured = capsys.readouterr()
        assert f"Sent: {session_file}" in captured.out


class TestDailyCommand:
    """Tests for the `daily` subcommand."""

    def test_missing_file_silent_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Scheduled runs should not exit-with-error on missing files."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        monkeypatch.setenv("VOCAB_LEARNER_INBOX_DIR", str(inbox))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("VOCAB_LEARNER_LLM_PROVIDER", "openai")

        rc = main(["daily", "--date", "2026-01-15"])

        assert rc == EXIT_OK
        captured = capsys.readouterr()
        # Nothing printed to stdout when file is missing
        assert captured.out == ""

    def test_missing_file_strict_returns_user_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Manual --strict invocations should exit-with-error on missing files."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        monkeypatch.setenv("VOCAB_LEARNER_INBOX_DIR", str(inbox))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("VOCAB_LEARNER_LLM_PROVIDER", "openai")

        rc = main(["daily", "--date", "2026-01-15", "--strict"])

        assert rc == EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()

    def test_invalid_date_returns_user_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("VOCAB_LEARNER_LLM_PROVIDER", "openai")

        rc = main(["daily", "--date", "not-a-date"])

        assert rc == EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "Invalid --date" in captured.err

    def test_happy_path_processes_and_emails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """File exists → pipeline runs → email sent → exit 0."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        image = inbox / "IMG_2026-01-15.jpg"
        image.write_bytes(b"\xff\xd8\xff")  # minimal JPEG magic bytes

        output_dir = tmp_path / "output"

        for key, val in {
            "VOCAB_LEARNER_INBOX_DIR": str(inbox),
            "VOCAB_LEARNER_OUTPUT_DIR": str(output_dir),
            "OPENAI_API_KEY": "sk-test",
            "VOCAB_LEARNER_LLM_PROVIDER": "openai",
            "VOCAB_LEARNER_SMTP_HOST": "smtp.test.com",
            "VOCAB_LEARNER_SMTP_USER": "u",
            "VOCAB_LEARNER_SMTP_PASSWORD": "p",
            "VOCAB_LEARNER_EMAIL_FROM": "from@test.com",
            "VOCAB_LEARNER_EMAIL_TO": "to@test.com",
        }.items():
            monkeypatch.setenv(key, val)

        # Mock the whole pipeline and SMTP
        fake_output = output_dir / "session_test.md"
        fake_output.parent.mkdir(parents=True, exist_ok=True)
        fake_output.write_text("# Test Session")

        fake_result = MagicMock()
        fake_result.output_path = fake_output
        fake_result.session.session_id = "session_test"

        with (
            patch("vocab_learner.cli.Pipeline") as mock_pipeline_cls,
            patch("smtplib.SMTP") as smtp_class,
        ):
            mock_pipeline_cls.return_value.run.return_value = fake_result
            smtp_instance = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp_instance

            rc = main(["daily", "--date", "2026-01-15"])

        assert rc == EXIT_OK
        captured = capsys.readouterr()
        assert str(fake_output) in captured.out
        smtp_instance.send_message.assert_called_once()

    def test_email_failure_does_not_fail_the_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Pipeline succeeded → primary work is done → email failure is logged, not fatal."""
        import smtplib

        inbox = tmp_path / "inbox"
        inbox.mkdir()
        image = inbox / "IMG_2026-01-15.jpg"
        image.write_bytes(b"\xff\xd8\xff")

        for key, val in {
            "VOCAB_LEARNER_INBOX_DIR": str(inbox),
            "VOCAB_LEARNER_OUTPUT_DIR": str(tmp_path / "output"),
            "OPENAI_API_KEY": "sk-test",
            "VOCAB_LEARNER_LLM_PROVIDER": "openai",
            "VOCAB_LEARNER_SMTP_HOST": "smtp.test.com",
            "VOCAB_LEARNER_SMTP_USER": "u",
            "VOCAB_LEARNER_SMTP_PASSWORD": "p",
            "VOCAB_LEARNER_EMAIL_FROM": "from@test.com",
            "VOCAB_LEARNER_EMAIL_TO": "to@test.com",
        }.items():
            monkeypatch.setenv(key, val)

        fake_output = tmp_path / "output" / "session_test.md"
        fake_output.parent.mkdir(parents=True, exist_ok=True)
        fake_output.write_text("# Test")

        fake_result = MagicMock()
        fake_result.output_path = fake_output
        fake_result.session.session_id = "session_test"

        with (
            patch("vocab_learner.cli.Pipeline") as mock_pipeline_cls,
            patch("smtplib.SMTP") as smtp_class,
        ):
            mock_pipeline_cls.return_value.run.return_value = fake_result
            smtp_class.side_effect = smtplib.SMTPException("mailbox full")

            rc = main(["daily", "--date", "2026-01-15"])

        # Pipeline succeeded → EXIT_OK even though email failed
        assert rc == EXIT_OK
        captured = capsys.readouterr()
        assert "Email delivery failed" in captured.err
