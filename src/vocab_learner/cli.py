"""Command-line entry point for vocab-learner."""

from __future__ import annotations

import argparse
import sys
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


if __name__ == "__main__":
    sys.exit(main())
