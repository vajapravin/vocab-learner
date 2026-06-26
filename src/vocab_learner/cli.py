"""Command-line entry point for vocab-learner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vocab_learner.config import load_settings
from vocab_learner.llm.factory import build_llm_client
from vocab_learner.llm.protocol import LLMError
from vocab_learner.logging_setup import configure_logging, get_logger
from vocab_learner.pipeline import Pipeline

# Exit codes
EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_PIPELINE_ERROR = 2
EXIT_INTERRUPTED = 130

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vocab-learner",
        description=(
            "Turn a scanned dictionary page into a vocabulary study session "
            "(extraction, teaching cards, and a reinforcement story)."
        ),
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to a dictionary page image (JPEG or PNG).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the session Markdown. Defaults to Settings.output_dir.",
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
        help="Suppress logs entirely (still prints the output path on success).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Entry point. Returns an exit code; also usable programmatically
    (pass argv, capture return).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load settings early so we fail fast on missing keys / bad env
    try:
        settings = load_settings()
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return EXIT_USER_ERROR

    # Apply CLI overrides
    if args.output_dir is not None:
        settings = settings.model_copy(update={"output_dir": args.output_dir})
    log_level = args.log_level or settings.log_level
    if args.quiet:
        log_level = "ERROR"
    configure_logging(log_level)

    # Validate inputs
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

    # Success — the only thing we print to stdout is the output path,
    # so users can pipe it: `vocab-learner page.jpg | xargs open`
    print(result.output_path)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
