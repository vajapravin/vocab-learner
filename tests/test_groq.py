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
    assert len(pool) >= 4
    vision = get_groq_vision_models(pool)
    text = get_groq_text_models(pool)
    assert len(vision) >= 1
    assert len(text) >= 3
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
        model="openai/gpt-oss-120b",
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
            ("openai/gpt-oss-120b",),
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
        model="openai/gpt-oss-120b",
        system_prompt="System",
        user_prompt="User",
        response_model=SampleResponse,
    )
    assert res.name == "failover_success"
    assert res.score == 95

    # Failing model marked in cooldown
    is_ok, reason = client._tracker.is_within_limit("openai/gpt-oss-120b")
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
        model="qwen/qwen3.8-27b",
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


def test_groq_try_parse_strips_schema_metadata(tmp_path: Path) -> None:
    settings = Settings(
        llm_provider="groq",
        groq_api_key="gsk_test_dummy",
        sqlite_db_path=tmp_path / "test.db",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = GroqClient(settings=settings)
    raw = (
        '{"name": "test_word", "score": 100, '
        '"required": ["name", "score"], "$schema": "http://json-schema.org"}'
    )
    parsed, err = client._try_parse(raw, SampleResponse)
    assert err is None
    assert parsed is not None
    assert parsed.name == "test_word"
    assert parsed.score == 100


def test_groq_extract_retry_delay() -> None:
    # 1. From error message
    err1 = openai.RateLimitError(
        message="Rate limit reached on TPM. Please try again in 3.42s.",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    assert GroqClient._extract_retry_delay(err1) == 3.42

    # 2. From retry-after header
    err2 = openai.RateLimitError(
        message="Too many requests",
        response=MagicMock(status_code=429, headers={"retry-after": "5.5"}),
        body=None,
    )
    assert GroqClient._extract_retry_delay(err2) == 5.5

    # 3. None when missing
    err3 = openai.RateLimitError(
        message="Too many requests without time hint",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    assert GroqClient._extract_retry_delay(err3) is None


def test_groq_retry_wait_jitter_randomness() -> None:
    """Random jitter must vary across concurrent requests to avoid collisions."""
    base_wait = 2.0
    delays = [GroqClient._calculate_jittered_delay(base_wait) for _ in range(20)]
    # All delays should be >= base_wait
    assert all(d >= base_wait for d in delays)
    # Delays must not all be identical (random noise added)
    assert len(set(delays)) > 10


def test_groq_per_model_limits_compound(tmp_path: Path) -> None:
    pool = get_groq_model_pool()
    compound = next((m for m in pool if m.model_id == "groq/compound"), None)
    assert compound is not None
    assert compound.rpm_limit == 30
    assert compound.rpd_limit == 250
    assert compound.tpm_limit == 70000
    assert compound.tpd_limit == 500000

    settings = Settings(
        llm_provider="groq",
        groq_api_key="gsk_test_dummy",
        sqlite_db_path=tmp_path / "test_compound.db",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = GroqClient(settings=settings)

    # Check registered limits in SQLite
    limits = client._tracker.get_or_create_limits("groq/compound")
    assert limits["rpm_limit"] == 30
    assert limits["rpd_limit"] == 250
    assert limits["tpm_limit"] == 70000
    assert limits["tpd_limit"] == 500000


def test_groq_rpd_limit_exceeded_switches_model(tmp_path: Path) -> None:
    """When a model exceeds its daily request limit (RPD), it switches to an available model."""
    import time

    settings = Settings(
        llm_provider="groq",
        groq_api_key="gsk_test_dummy",
        sqlite_db_path=tmp_path / "test_rpd.db",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = GroqClient(settings=settings)

    # groq/compound has RPD=250. Let's record 250 requests for it in the last 24 hours.
    now = time.time()
    for i in range(250):
        client._tracker.record_usage(
            model_id="groq/compound",
            key_hash="hash",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            now=now - (i * 10),
        )

    # Request 251 should fail RPD limit
    is_ok, reason = client._tracker.is_within_limit("groq/compound", now=now)
    assert is_ok is False
    assert "RPD limit exceeded" in str(reason)

    # When candidate models are [groq/compound, llama-3.1-8b-instant], selection switches!
    chosen, switched = client._tracker.select_available_model(
        candidate_models=["groq/compound", "llama-3.1-8b-instant"],
        preferred_model="groq/compound",
        now=now,
    )
    assert chosen == "llama-3.1-8b-instant"
    assert switched is True

    # Usage summary verification
    usage = client._tracker.get_model_usage("groq/compound", now=now)
    assert usage["rpd_used"] == 250
    assert usage["rpd_limit"] == 250


def test_tpm_sliding_window_wait_calculation(tmp_path: Path) -> None:
    """When TPM limit is exceeded, get_seconds_until_available calculates exact wait time."""
    import time

    settings = Settings(
        llm_provider="groq",
        groq_api_key="gsk_test_dummy",
        sqlite_db_path=tmp_path / "test_wait.db",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = GroqClient(settings=settings)
    now = time.time()

    # Record 6441 tokens for model at now - 45s (limit is 8000)
    client._tracker.record_usage(
        model_id="openai/gpt-oss-120b",
        key_hash="hash",
        prompt_tokens=3000,
        completion_tokens=3441,
        total_tokens=6441,
        now=now - 45.0,
    )

    # An estimated request of 2000 tokens: 6441 + 2000 = 8441 > 8000!
    is_ok, reason = client._tracker.is_within_limit(
        "openai/gpt-oss-120b", estimated_tokens=2000, now=now
    )
    assert is_ok is False
    assert "TPM limit exceeded" in str(reason)

    # Wait until now - 45.0 + 60.0 = 15.0 seconds from now!
    wait_time = client._tracker.get_seconds_until_available(
        "openai/gpt-oss-120b", estimated_tokens=2000, now=now
    )
    assert 14.5 <= wait_time <= 15.5


def test_select_available_model_load_balancing(tmp_path: Path) -> None:
    """select_available_model distributes load to models with lower token usage."""
    import time

    settings = Settings(
        llm_provider="groq",
        groq_api_key="gsk_test_dummy",
        sqlite_db_path=tmp_path / "test_balance.db",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = GroqClient(settings=settings)
    now = time.time()

    # Model A has 3000 tokens used in the last 60s
    client._tracker.record_usage(
        model_id="openai/gpt-oss-120b",
        key_hash="hash",
        prompt_tokens=1500,
        completion_tokens=1500,
        total_tokens=3000,
        now=now - 10.0,
    )
    # Model B has 0 tokens used

    # Even if Model A is preferred, Model B has lower usage and is selected to balance load!
    chosen, switched = client._tracker.select_available_model(
        candidate_models=["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        preferred_model="openai/gpt-oss-120b",
        estimated_tokens=1000,
        now=now,
    )
    assert chosen == "openai/gpt-oss-20b"
    assert switched is True


def test_groq_json_generation_failure_switches_model(tmp_path: Path) -> None:
    """When a model fails with 400 'Failed to generate JSON', it marks cooldown and switches."""
    settings = Settings(
        llm_provider="groq",
        groq_api_key="gsk_test_dummy",
        sqlite_db_path=tmp_path / "test_json_fail.db",
        _env_file=None,  # type: ignore[call-arg]
    )
    client = GroqClient(settings=settings)

    call_count = {"model_a": 0, "model_b": 0}

    def mock_create(*args, **kwargs):  # type: ignore[no-untyped-def]
        model = kwargs.get("model")
        if model == "allam-2-7b":
            call_count["model_a"] += 1
            # Groq 400 error
            mock_resp = MagicMock()
            mock_resp.status_code = 400
            err_dict = {
                "error": {
                    "message": "Failed to generate JSON. Please adjust your prompt.",
                    "code": "json_validate_failed",
                }
            }
            raise openai.BadRequestError(
                message="Error code: 400 - " + str(err_dict),
                response=mock_resp,
                body=err_dict,
            )
        else:
            call_count["model_b"] += 1
            mock_choice = MagicMock()
            mock_choice.message.content = '{"name": "Alice", "score": 99}'
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            mock_resp.usage = MagicMock(prompt_tokens=50, completion_tokens=20, total_tokens=70)
            return mock_resp

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = mock_create
    client._clients["gsk_test_dummy"] = mock_openai

    client._text_models = ["allam-2-7b", "openai/gpt-oss-120b"]

    res = client._execute_with_switching(
        requested_model="allam-2-7b",
        system_prompt="sys",
        user_content=[{"type": "text", "text": "usr"}],
        response_model=SampleResponse,
        max_tokens=500,
        is_vision=False,
        estimated_tokens=500,
    )

    assert res.name == "Alice"
    assert res.score == 99
    # Verify allam-2-7b was tried, failed, and openai/gpt-oss-120b took over
    assert call_count["model_a"] >= 1
    assert call_count["model_b"] == 1
    # Verify allam-2-7b is marked in cooldown
    is_ok, reason = client._tracker.is_within_limit("allam-2-7b")
    assert is_ok is False
    assert "Model in cooldown" in str(reason)
