"""Tests for the LLM client factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from vocab_learner.config import Settings
from vocab_learner.llm.anthropic_client import AnthropicClient
from vocab_learner.llm.factory import build_llm_client
from vocab_learner.llm.openai_client import OpenAIClient


def test_factory_returns_openai_client() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key="sk-test",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = build_llm_client(settings)
    assert isinstance(client, OpenAIClient)


def test_factory_returns_anthropic_client() -> None:
    settings = Settings(
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-test",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = build_llm_client(settings)
    assert isinstance(client, AnthropicClient)


def test_factory_openai_without_key_raises() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key=None,
        _env_file=None,  # type: ignore[call-arg]
    )
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_llm_client(settings)


def test_factory_anthropic_without_key_raises() -> None:
    settings = Settings(
        llm_provider="anthropic",
        anthropic_api_key=None,
        _env_file=None,  # type: ignore[call-arg]
    )
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_llm_client(settings)


def test_factory_returns_huggingface_client(tmp_path: Path) -> None:
    from vocab_learner.llm.huggingface_client import HuggingFaceClient

    settings = Settings(
        llm_provider="huggingface",
        huggingface_api_key="hf_test_key",
        sqlite_db_path=tmp_path / "test.db",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = build_llm_client(settings)
    assert isinstance(client, HuggingFaceClient)


def test_factory_huggingface_without_key_raises() -> None:
    settings = Settings(
        llm_provider="huggingface",
        huggingface_api_key=None,
        hf_token=None,
        huggingface_api_keys=None,
        _env_file=None,  # type: ignore[call-arg]
    )
    with pytest.raises(ValueError, match="HUGGINGFACE_API_KEY"):
        build_llm_client(settings)
