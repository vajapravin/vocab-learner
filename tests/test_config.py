"""Tests for Settings loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vocab_learner.config import Settings


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any real env vars that could bleed into the test."""
    for key in list(os.environ):
        if key.startswith("VOCAB_LEARNER_") or key in {
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
        }:
            monkeypatch.delenv(key, raising=False)


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.setenv("VOCAB_LEARNER_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("VOCAB_LEARNER_EXTRACTION_MODEL", "gpt-4.1")

    # Disable .env file discovery for this test
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.anthropic_api_key == "sk-test-123"
    assert settings.log_level == "DEBUG"
    assert settings.extraction_model == "gpt-4.1"
    assert settings.teaching_model == "gpt-4.1"  # default
    assert isinstance(settings.output_dir, Path)


def test_settings_no_keys_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None
