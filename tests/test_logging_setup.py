"""Regression tests for stream separation in logging."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_logs_go_to_stderr_not_stdout(tmp_path: Path) -> None:
    """
    Structured logs must not pollute stdout. Downstream tooling (jq, JSON
    parsers, unix pipes) relies on stdout being clean.
    """
    script = tmp_path / "emit.py"
    script.write_text(
        textwrap.dedent(
            """
            from vocab_learner.logging_setup import configure_logging, get_logger
            configure_logging("INFO")
            log = get_logger("test")
            log.info("test_event", key="value")
            print("STDOUT_MARKER")
            """
        )
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "STDOUT_MARKER"
    assert "test_event" in result.stderr
    assert "test_event" not in result.stdout