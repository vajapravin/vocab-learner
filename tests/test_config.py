"""Tests for Settings loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vocab_learner.config import Settings


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any real env vars that could bleed into the test."""
    for key in list(os.environ):
        if key.startswith("VOCAB_LEARNER_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.setenv("VOCAB_LEARNER_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("VOCAB_LEARNER_EXTRACTION_MODEL", "claude-haiku-4-5")

    # Disable .env file discovery for this test
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.anthropic_api_key == "sk-test-123"
    assert settings.log_level == "DEBUG"
    assert settings.extraction_model == "claude-haiku-4-5"
    assert settings.teaching_model == "claude-sonnet-5"  # default
    assert isinstance(settings.output_dir, Path)


def test_settings_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    with pytest.raises(ValueError):
        Settings(_env_file=None)  # type: ignore[call-arg]
