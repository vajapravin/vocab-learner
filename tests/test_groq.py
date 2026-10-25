"""Tests for Groq multi-model client, SQLite rate limit tracker, and key management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import openai
import pytest
from pydantic import BaseModel

from vocab_learner.config import Settings
from vocab_learner.llm.groq_client import GroqClient
from vocab_learner.llm.groq_models import (
    GroqModelInfo,
    get_groq_model_pool,
    get_groq_text_models,
    get_groq_vision_models,
)


class SampleResponse(BaseModel):
    name: str
    score: int


def test_groq_models_pool() -> None:
    pool = get_groq_model_pool()
    assert len(pool) >= 5
    vision = get_groq_vision_models(pool)
    text = get_groq_text_models(pool)
    assert len(vision) >= 2
    assert len(text) >= 5
    assert all(isinstance(m, GroqModelInfo) for m in pool)


def test_load_groq_keys_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_primary")
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_extra1, gsk_extra2")
    monkeypatch.setenv("GROQ_API_KEY_1", "gsk_numbered1")

    settings = Settings(
        llm_provider="groq",
        sqlite_db_path=tmp_path / "db.sqlite",
        _env_file=None,  # type: ignore[call-arg]
    )
    keys = GroqClient._load_keys(settings)
    assert "gsk_primary" in keys
    assert "gsk_extra1" in keys
    assert "gsk_extra2" in keys
    assert "gsk_numbered1" in keys


def test_groq_complete_structured_success(tmp_path: Path) -> None:
    settings = Settings(
        llm_provider="groq",
        groq_api_key="gsk_test_dummy",
        sqlite_db_path=tmp_path / "test.db",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = GroqClient(settings=settings)

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"name": "groq_word", "score": 98}'))]
    mock_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=30, total_tokens=130)

    for c in client._clients.values():
        c.chat.completions.create = MagicMock(return_value=mock_resp)  # type: ignore[method-assign]

    res = client.complete_structured(
        model="llama-3.3-70b-versatile",
        system_prompt="System prompt",
        user_prompt="User prompt",
        response_model=SampleResponse,
    )
    assert res.name == "groq_word"
    assert res.score == 98

    # Verify usage was recorded in SQLite
    with client._tracker._get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM token_usage WHERE model_id = ?",
            ("llama-3.3-70b-versatile",),
        ).fetchone()
        assert row is not None
        assert row["total_tokens"] == 130


def test_groq_complete_structured_rate_limit_failover(tmp_path: Path) -> None:
    settings = Settings(
        llm_provider="groq",
        groq_api_key="gsk_test_dummy",
        sqlite_db_path=tmp_path / "test.db",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = GroqClient(settings=settings)

    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content='{"name": "failover_success", "score": 95}'))
    ]
    mock_resp.usage = MagicMock(prompt_tokens=40, completion_tokens=15, total_tokens=55)

    error_429 = openai.RateLimitError(
        message="Too many requests - rate limit reached",
        response=MagicMock(status_code=429, headers={}),
        body={"error": {"message": "Rate limit reached"}},
    )

    for c in client._clients.values():
        c.chat.completions.create = MagicMock(  # type: ignore[method-assign]
            side_effect=[error_429, mock_resp]
        )

    res = client.complete_structured(
        model="llama-3.3-70b-versatile",
        system_prompt="System",
        user_prompt="User",
        response_model=SampleResponse,
    )
    assert res.name == "failover_success"
    assert res.score == 95

    # Failing model marked in cooldown
    is_ok, reason = client._tracker.is_within_limit("llama-3.3-70b-versatile")
    assert is_ok is False
    assert "cooldown" in str(reason)


def test_groq_complete_vision_structured(tmp_path: Path) -> None:
    settings = Settings(
        llm_provider="groq",
        groq_api_key="gsk_test_dummy",
        sqlite_db_path=tmp_path / "test.db",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = GroqClient(settings=settings)

    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content='{"name": "vision_extracted", "score": 100}'))
    ]
    mock_resp.usage = MagicMock(prompt_tokens=300, completion_tokens=40, total_tokens=340)

    for c in client._clients.values():
        c.chat.completions.create = MagicMock(return_value=mock_resp)  # type: ignore[method-assign]

    res = client.complete_vision_structured(
        model="llama-3.2-11b-vision-preview",
        system_prompt="System",
        user_prompt="Extract image",
        image_bytes=b"sample_image_data",
        image_media_type="image/jpeg",
        response_model=SampleResponse,
    )
    assert res.name == "vision_extracted"
    assert res.score == 100


def test_groq_clean_json_markdown_fences() -> None:
    raw = '```json\n{\n  "name": "clean",\n  "score": 50\n}\n```'
    cleaned = GroqClient._clean_json(raw)
    assert cleaned == '{\n  "name": "clean",\n  "score": 50\n}'
