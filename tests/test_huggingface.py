"""Tests for Hugging Face multi-model architecture, SQLite token tracker, and key management."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import openai
import pytest
from pydantic import BaseModel

from vocab_learner.config import Settings
from vocab_learner.llm.huggingface_client import HuggingFaceClient
from vocab_learner.llm.models_registry import (
    ModelInfo,
    get_model_pool,
    get_text_models,
    get_vision_models,
)
from vocab_learner.llm.token_tracker import TokenTracker


class SampleResponse(BaseModel):
    name: str
    score: int


# --- Model Registry Tests ---------------------------------------------------


def test_models_registry_has_30_models() -> None:
    pool = get_model_pool()
    assert len(pool) == 30
    vision = get_vision_models(pool)
    text = get_text_models(pool)
    assert len(vision) >= 10
    assert len(text) == 30
    assert all(isinstance(m, ModelInfo) for m in pool)


# --- SQLite TokenTracker Tests -----------------------------------------------


def test_token_tracker_within_limit_and_exceeded(tmp_path: Path) -> None:
    db_file = tmp_path / "test_tracker.db"
    tracker = TokenTracker(
        db_path=db_file,
        tpm_limit=5000,
        rpm_limit=3,
        daily_limit=20000,
    )

    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Initially within limit
    is_ok, reason = tracker.is_within_limit(model, estimated_tokens=1000)
    assert is_ok is True
    assert reason is None

    # Record usage
    tracker.record_usage(
        model_id=model,
        key_hash="key1",
        prompt_tokens=2500,
        completion_tokens=2000,
        total_tokens=4500,
    )

    # Next call estimating 1000 would exceed TPM limit (4500 + 1000 > 5000)
    is_ok, reason = tracker.is_within_limit(model, estimated_tokens=1000)
    assert is_ok is False
    assert "TPM limit exceeded" in str(reason)

    # Small call estimating 200 is still under TPM (4500 + 200 <= 5000)
    is_ok, reason = tracker.is_within_limit(model, estimated_tokens=200)
    assert is_ok is True


def test_token_tracker_cooldown_and_failover(tmp_path: Path) -> None:
    db_file = tmp_path / "test_tracker.db"
    tracker = TokenTracker(db_path=db_file)
    model1 = "meta-llama/Llama-3.1-8B-Instruct"
    model2 = "meta-llama/Llama-3.3-70B-Instruct"

    # Mark model1 in cooldown (e.g. 429 error)
    tracker.mark_cooldown(model1, cooldown_seconds=60.0, error_message="HTTP 429 Too Many Requests")

    # model1 is now not available
    is_ok, reason = tracker.is_within_limit(model1)
    assert is_ok is False
    assert "cooldown" in str(reason)

    # Selection automatically switches to model2!
    chosen, switched = tracker.select_available_model(
        candidate_models=[model1, model2],
        preferred_model=model1,
    )
    assert chosen == model2
    assert switched is True


def test_token_tracker_storage_pruning(tmp_path: Path) -> None:
    db_file = tmp_path / "test_storage.db"
    tracker = TokenTracker(db_path=db_file)
    model = "test-model"

    now = time.time()
    old_time = now - 90000.0  # > 24 hours ago

    # Insert an old event directly
    tracker.record_usage(
        model_id=model,
        key_hash="old_key",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        now=old_time,
    )

    # Insert a new event (triggers automatic prune of >24h old events)
    tracker.record_usage(
        model_id=model,
        key_hash="new_key",
        prompt_tokens=200,
        completion_tokens=100,
        total_tokens=300,
        now=now,
    )

    # Verify only recent records remain in SQLite
    with tracker._get_connection() as conn:
        rows = conn.execute("SELECT * FROM token_usage").fetchall()
        assert len(rows) == 1
        assert rows[0]["key_hash"] == "new_key"


# --- Key Management from .env Tests -----------------------------------------


def test_load_keys_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_primary")
    monkeypatch.setenv("HUGGINGFACE_API_KEYS", "hf_extra1, hf_extra2")
    monkeypatch.setenv("HUGGINGFACE_API_KEY_1", "hf_numbered1")
    monkeypatch.setenv("HF_TOKEN_2", "hf_numbered2")

    settings = Settings(
        llm_provider="huggingface",
        sqlite_db_path=tmp_path / "db.sqlite",
        _env_file=None,  # type: ignore[call-arg]
    )
    keys = HuggingFaceClient._load_keys(settings)
    assert "hf_primary" in keys
    assert "hf_extra1" in keys
    assert "hf_extra2" in keys
    assert "hf_numbered1" in keys
    assert "hf_numbered2" in keys


# --- HuggingFaceClient Execution & Switching Tests --------------------------


def test_complete_structured_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = Settings(
        llm_provider="huggingface",
        huggingface_api_key="hf_test_dummy",
        sqlite_db_path=tmp_path / "test.db",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = HuggingFaceClient(settings=settings)

    # Mock the internal OpenAI client response
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"name": "test_word", "score": 95}'))]
    mock_resp.usage = MagicMock(prompt_tokens=120, completion_tokens=35, total_tokens=155)

    for c in client._clients.values():
        c.chat.completions.create = MagicMock(return_value=mock_resp)  # type: ignore[method-assign]

    res = client.complete_structured(
        model="meta-llama/Llama-3.1-8B-Instruct",
        system_prompt="System prompt",
        user_prompt="User prompt",
        response_model=SampleResponse,
    )
    assert res.name == "test_word"
    assert res.score == 95

    # Verify usage was recorded in SQLite
    with client._tracker._get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM token_usage WHERE model_id = ?",
            ("meta-llama/Llama-3.1-8B-Instruct",),
        ).fetchone()
        assert row is not None
        assert row["total_tokens"] == 155


def test_complete_structured_automatic_switching_on_limit(tmp_path: Path) -> None:
    settings = Settings(
        llm_provider="huggingface",
        huggingface_api_key="hf_test_dummy",
        sqlite_db_path=tmp_path / "test.db",
        tpm_limit=500,  # small limit
        _env_file=None,  # type: ignore[call-arg]
    )
    client = HuggingFaceClient(settings=settings)

    preferred_model = "meta-llama/Llama-3.1-8B-Instruct"

    # Fill up the preferred model's limit in SQLite
    client._tracker.record_usage(
        model_id=preferred_model,
        key_hash="hash",
        prompt_tokens=300,
        completion_tokens=300,
        total_tokens=600,
    )

    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content='{"name": "switched_word", "score": 88}'))
    ]
    mock_resp.usage = MagicMock(prompt_tokens=50, completion_tokens=20, total_tokens=70)

    for c in client._clients.values():
        c.chat.completions.create = MagicMock(return_value=mock_resp)  # type: ignore[method-assign]

    # Calling with preferred_model should switch to an available LLM
    res = client.complete_structured(
        model=preferred_model,
        system_prompt="System",
        user_prompt="User",
        response_model=SampleResponse,
    )
    assert res.name == "switched_word"

    # Inspect SQLite: usage should be recorded for the switched model, NOT the preferred model
    with client._tracker._get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT model_id FROM token_usage WHERE total_tokens = 70"
        ).fetchall()
        assert len(rows) == 1
        switched_to = rows[0]["model_id"]
        assert switched_to != preferred_model


def test_complete_structured_429_failover(tmp_path: Path) -> None:
    settings = Settings(
        llm_provider="huggingface",
        huggingface_api_key="hf_test_dummy",
        sqlite_db_path=tmp_path / "test.db",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = HuggingFaceClient(settings=settings)

    # First call returns 429 RateLimitError; second call succeeds
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='{"name": "recovered", "score": 99}'))]
    mock_resp.usage = MagicMock(prompt_tokens=40, completion_tokens=15, total_tokens=55)

    error_429 = openai.RateLimitError(
        message="Too many requests - rate limit exceeded",
        response=MagicMock(status_code=429, headers={}),
        body={"error": {"message": "Rate limit reached"}},
    )

    for c in client._clients.values():
        c.chat.completions.create = MagicMock(  # type: ignore[method-assign]
            side_effect=[error_429, mock_resp]
        )

    res = client.complete_structured(
        model="meta-llama/Llama-3.1-8B-Instruct",
        system_prompt="System",
        user_prompt="User",
        response_model=SampleResponse,
    )
    assert res.name == "recovered"
    assert res.score == 99

    # The failing model should now be marked in cooldown in SQLite
    is_ok, reason = client._tracker.is_within_limit("meta-llama/Llama-3.1-8B-Instruct")
    assert is_ok is False
    assert "cooldown" in str(reason)


def test_complete_vision_structured(tmp_path: Path) -> None:
    settings = Settings(
        llm_provider="huggingface",
        huggingface_api_key="hf_test_dummy",
        sqlite_db_path=tmp_path / "test.db",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = HuggingFaceClient(settings=settings)

    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content='{"name": "vision_word", "score": 100}'))
    ]
    mock_resp.usage = MagicMock(prompt_tokens=500, completion_tokens=50, total_tokens=550)

    for c in client._clients.values():
        c.chat.completions.create = MagicMock(return_value=mock_resp)  # type: ignore[method-assign]

    res = client.complete_vision_structured(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        system_prompt="System",
        user_prompt="User",
        image_bytes=b"fake_jpeg_bytes",
        image_media_type="image/jpeg",
        response_model=SampleResponse,
    )
    assert res.name == "vision_word"
    assert res.score == 100


def test_clean_json_markdown_fences() -> None:
    raw = '```json\n{\n  "name": "fenced",\n  "score": 10\n}\n```'
    cleaned = HuggingFaceClient._clean_json(raw)
    assert cleaned == '{\n  "name": "fenced",\n  "score": 10\n}'
