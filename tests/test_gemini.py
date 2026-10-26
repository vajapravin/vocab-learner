"""Tests for the Gemini LLM client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from google.genai import errors
from pydantic import BaseModel

from vocab_learner.llm.gemini_client import GeminiClient
from vocab_learner.llm.protocol import LLMClient, LLMError, LLMValidationError


class SampleModel(BaseModel):
    name: str
    count: int


def test_gemini_client_satisfies_protocol() -> None:
    client = GeminiClient(api_key="test-key")
    assert isinstance(client, LLMClient)


def test_gemini_client_complete_structured_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GeminiClient(api_key="test-key")
    mock_resp = MagicMock()
    mock_resp.text = '{"name": "vocabulary", "count": 5}'

    mock_generate = MagicMock(return_value=mock_resp)
    monkeypatch.setattr(client._client.models, "generate_content", mock_generate)

    res = client.complete_structured(
        model="gemini-2.5-flash",
        system_prompt="system",
        user_prompt="user",
        response_model=SampleModel,
    )

    assert res.name == "vocabulary"
    assert res.count == 5
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["contents"] == ["user"]
    config = call_kwargs["config"]
    assert config.response_mime_type == "application/json"


def test_gemini_client_strips_markdown_code_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GeminiClient(api_key="test-key")
    mock_resp = MagicMock()
    mock_resp.text = '```json\n{"name": "fenced", "count": 9}\n```'

    mock_generate = MagicMock(return_value=mock_resp)
    monkeypatch.setattr(client._client.models, "generate_content", mock_generate)

    res = client.complete_structured(
        model="gemini-2.5-flash",
        system_prompt="system",
        user_prompt="user",
        response_model=SampleModel,
    )

    assert res.name == "fenced"
    assert res.count == 9


def test_gemini_client_complete_vision_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GeminiClient(api_key="test-key")
    mock_resp = MagicMock()
    mock_resp.text = '{"name": "vision_word", "count": 1}'

    mock_generate = MagicMock(return_value=mock_resp)
    monkeypatch.setattr(client._client.models, "generate_content", mock_generate)

    res = client.complete_vision_structured(
        model="gemini-2.5-flash",
        system_prompt="system",
        user_prompt="describe image",
        image_bytes=b"fake_image_bytes",
        image_media_type="image/jpeg",
        response_model=SampleModel,
    )

    assert res.name == "vision_word"
    assert res.count == 1
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args.kwargs
    contents = call_kwargs["contents"]
    assert len(contents) == 2
    # First part is the image part
    assert contents[0].inline_data.data == b"fake_image_bytes"
    assert contents[0].inline_data.mime_type == "image/jpeg"
    assert contents[1] == "describe image"


def test_gemini_client_repair_retry_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GeminiClient(api_key="test-key")
    bad_resp = MagicMock()
    bad_resp.text = '{"name": "missing_count"}'  # Missing required 'count'

    good_resp = MagicMock()
    good_resp.text = '{"name": "repaired", "count": 42}'

    mock_generate = MagicMock(side_effect=[bad_resp, good_resp])
    monkeypatch.setattr(client._client.models, "generate_content", mock_generate)

    res = client.complete_structured(
        model="gemini-2.5-flash",
        system_prompt="system",
        user_prompt="user",
        response_model=SampleModel,
    )

    assert res.name == "repaired"
    assert res.count == 42
    assert mock_generate.call_count == 2
    # Verify the repair attempt contents include the error message
    second_call_contents = mock_generate.call_args_list[1].kwargs["contents"]
    assert len(second_call_contents) == 2
    assert "Your previous response could not be parsed" in second_call_contents[1]


def test_gemini_client_repair_retry_exhausted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GeminiClient(api_key="test-key")
    bad_resp = MagicMock()
    bad_resp.text = '{"bad": "json"}'

    mock_generate = MagicMock(return_value=bad_resp)
    monkeypatch.setattr(client._client.models, "generate_content", mock_generate)

    with pytest.raises(LLMValidationError, match="failed validation after retry"):
        client.complete_structured(
            model="gemini-2.5-flash",
            system_prompt="system",
            user_prompt="user",
            response_model=SampleModel,
        )

    assert mock_generate.call_count == 2


def test_gemini_client_api_error_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GeminiClient(api_key="test-key")
    mock_generate = MagicMock(side_effect=errors.APIError(400, "Invalid API key"))
    monkeypatch.setattr(client._client.models, "generate_content", mock_generate)

    with pytest.raises(LLMError, match="Gemini API error"):
        client.complete_structured(
            model="gemini-2.5-flash",
            system_prompt="system",
            user_prompt="user",
            response_model=SampleModel,
        )
