"""Command-line entry point for vocab-learner."""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from vocab_learner.config import (
    Settings,  # local for type narrowing
    load_settings,
)
from vocab_learner.emailer import (
    EmailConfigError,
    EmailSender,
    EmailSendError,
    EmailSettings,
)
from vocab_learner.llm.factory import build_llm_client
from vocab_learner.llm.protocol import LLMError
from vocab_learner.logging_setup import configure_logging, get_logger
from vocab_learner.pipeline import Pipeline

# Exit codes
EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_PIPELINE_ERROR = 2
EXIT_EMAIL_ERROR = 3
EXIT_INTERRUPTED = 130

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vocab-learner",
        description=(
            "Turn a scanned dictionary page into a vocabulary study session, "
            "and optionally email it as HTML."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override the configured log level.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress logs entirely (still prints results to stdout).",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="<command>",
    )

    # `run` subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Run the extraction/teaching/story pipeline on a dictionary image.",
    )
    run_parser.add_argument(
        "image",
        type=Path,
        help="Path to a dictionary page image (JPEG or PNG).",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the session Markdown. Defaults to Settings.output_dir.",
    )

    # `send` subcommand
    send_parser = subparsers.add_parser(
        "send",
        help="Email an existing session Markdown file as HTML.",
    )
    send_parser.add_argument(
        "session_file",
        type=Path,
        help="Path to a session Markdown file previously produced by `run`.",
    )

    # `daily` subcommand
    daily_parser = subparsers.add_parser(
        "daily",
        help=(
            "Process today's dictionary image and email the result. "
            "Designed to run as a scheduled task."
        ),
    )
    daily_parser.add_argument(
        "--date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Process the file for a specific date instead of today.",
    )
    daily_parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit with an error if the target file is missing. "
            "Default (for scheduled runs) is silent skip."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return EXIT_USER_ERROR

    log_level = args.log_level or settings.log_level
    if args.quiet:
        log_level = "ERROR"
    configure_logging(log_level)

    if args.command == "run":
        return _cmd_run(args, settings)
    if args.command == "send":
        return _cmd_send(args, settings)
    if args.command == "daily":
        return _cmd_daily(args, settings)

    # argparse's `required=True` on subparsers prevents this, but mypy
    # doesn't know that.
    parser.print_help(sys.stderr)
    return EXIT_USER_ERROR


def _cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    """Run the pipeline on an image and write a session Markdown."""
    assert isinstance(settings, Settings)

    if args.output_dir is not None:
        settings = settings.model_copy(update={"output_dir": args.output_dir})

    if not args.image.exists():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return EXIT_USER_ERROR
    if not args.image.is_file():
        print(f"Not a file: {args.image}", file=sys.stderr)
        return EXIT_USER_ERROR

    try:
        client = build_llm_client(settings)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return EXIT_USER_ERROR

    pipeline = Pipeline(client=client, settings=settings)

    try:
        result = pipeline.run(args.image)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (LLMError, RuntimeError, FileNotFoundError, ValueError) as e:
        log.error("pipeline_failed", error=str(e))
        print(f"Pipeline error: {e}", file=sys.stderr)
        return EXIT_PIPELINE_ERROR

    print(result.output_path)
    return EXIT_OK


def _cmd_send(args: argparse.Namespace, settings: object) -> int:
    """Email a previously-generated session Markdown file."""
    from vocab_learner.config import Settings

    assert isinstance(settings, Settings)

    if not args.session_file.exists():
        print(f"Session file not found: {args.session_file}", file=sys.stderr)
        return EXIT_USER_ERROR
    if not args.session_file.is_file():
        print(f"Not a file: {args.session_file}", file=sys.stderr)
        return EXIT_USER_ERROR
    if args.session_file.suffix != ".md":
        print(
            f"Expected a .md file; got {args.session_file.suffix}",
            file=sys.stderr,
        )
        return EXIT_USER_ERROR

    try:
        email_settings = EmailSettings.from_settings(settings)
    except EmailConfigError as e:
        print(f"Email config error: {e}", file=sys.stderr)
        return EXIT_USER_ERROR

    sender = EmailSender(email_settings)

    try:
        sender.send_file(args.session_file)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except EmailSendError as e:
        log.error("email_send_error", error=str(e))
        print(f"Email delivery failed: {e}", file=sys.stderr)
        return EXIT_EMAIL_ERROR

    print(f"Sent: {args.session_file}")
    return EXIT_OK


def _cmd_daily(args: argparse.Namespace, settings: Settings) -> int:
    """
    Process today's (or a specified date's) dictionary image and email
    the resulting session. Designed to be triggered by an external scheduler.
    """
    target_date = _resolve_target_date(args.date)
    if target_date is None:
        print(
            f"Invalid --date value: {args.date!r}. Expected YYYY-MM-DD.",
            file=sys.stderr,
        )
        return EXIT_USER_ERROR

    filename = f"IMG_{target_date.isoformat()}.jpg"
    image_path = settings.inbox_dir / filename

    log.info(
        "daily_start",
        target_date=target_date.isoformat(),
        expected_file=str(image_path),
    )

    if not image_path.exists():
        log.info("daily_file_not_found", path=str(image_path))
        if args.strict:
            print(f"File not found: {image_path}", file=sys.stderr)
            return EXIT_USER_ERROR
        # Silent skip for scheduled runs
        return EXIT_OK

    # Run the pipeline (reuses the same code path as `run`)
    try:
        client = build_llm_client(settings)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return EXIT_USER_ERROR

    pipeline = Pipeline(client=client, settings=settings)

    try:
        result = pipeline.run(image_path)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (LLMError, RuntimeError, FileNotFoundError, ValueError) as e:
        log.error("daily_pipeline_failed", error=str(e))
        print(f"Pipeline error: {e}", file=sys.stderr)
        return EXIT_PIPELINE_ERROR

    # Email the result. Failure logs but doesn't fail the run.
    try:
        email_settings = EmailSettings.from_settings(settings)
        sender = EmailSender(email_settings)
        sender.send_file(result.output_path)
    except EmailConfigError as e:
        log.error("daily_email_config_error", error=str(e))
        print(f"Email skipped: {e}", file=sys.stderr)
    except EmailSendError as e:
        log.error("daily_email_send_error", error=str(e))
        print(f"Email delivery failed: {e}", file=sys.stderr)

    log.info(
        "daily_done",
        session_id=result.session.session_id,
        output=str(result.output_path),
    )
    print(result.output_path)
    return EXIT_OK


def _resolve_target_date(date_str: str | None) -> date | None:
    """Return today's date if date_str is None, else parse it. None on parse failure."""
    if date_str is None:
        return datetime.now().date()
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        return None


if __name__ == "__main__":
    sys.exit(main())
