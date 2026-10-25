"""Tests for the local GGUF LLM client and OCR handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from vocab_learner.config import Settings
from vocab_learner.llm.factory import build_llm_client
from vocab_learner.llm.local_client import LocalLlamaClient
from vocab_learner.llm.protocol import LLMError


class SampleModel(BaseModel):
    name: str
    count: int


def test_factory_returns_local_client(tmp_path: Path) -> None:
    fake_model = tmp_path / "model.gguf"
    fake_model.write_text("fake gguf content")

    settings = Settings(
        llm_provider="local",
        local_model_path=fake_model,
        _env_file=None,  # type: ignore[call-arg]
    )
    client = build_llm_client(settings)
    assert isinstance(client, LocalLlamaClient)


def test_local_client_missing_model_raises(tmp_path: Path) -> None:
    non_existent = tmp_path / "does_not_exist.gguf"
    with pytest.raises(FileNotFoundError, match="Local GGUF model not found"):
        LocalLlamaClient(model_path=non_existent)


def test_local_client_complete_structured(tmp_path: Path) -> None:
    fake_model = tmp_path / "test.gguf"
    fake_model.write_text("fake gguf")

    client = LocalLlamaClient(model_path=fake_model)

    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"name": "test_word", "count": 42}'}}]
    }
    client._llm = mock_llm

    res = client.complete_structured(
        model="qwen2.5-7b-instruct",
        system_prompt="system",
        user_prompt="user",
        response_model=SampleModel,
    )
    assert res.name == "test_word"
    assert res.count == 42


def test_local_client_try_parse_markdown_fence(tmp_path: Path) -> None:
    fake_model = tmp_path / "test.gguf"
    fake_model.write_text("fake gguf")

    client = LocalLlamaClient(model_path=fake_model)

    fenced = '```json\n{"name": "apple", "count": 5}\n```'
    parsed, err = client._try_parse(fenced, SampleModel)
    assert err is None
    assert parsed is not None
    assert parsed.name == "apple"
    assert parsed.count == 5


def test_local_client_complete_vision_structured_uses_ocr(tmp_path: Path) -> None:
    fake_model = tmp_path / "test.gguf"
    fake_model.write_text("fake gguf")

    client = LocalLlamaClient(model_path=fake_model)

    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"name": "ocr_word", "count": 1}'}}]
    }
    client._llm = mock_llm
    client._run_ocr = MagicMock(return_value="mocked ocr text line")  # type: ignore[method-assign]

    res = client.complete_vision_structured(
        model="qwen2.5-7b-instruct",
        system_prompt="sys",
        user_prompt="usr",
        image_bytes=b"dummy_image",
        image_media_type="image/jpeg",
        response_model=SampleModel,
    )
    assert res.name == "ocr_word"
    assert res.count == 1
    client._run_ocr.assert_called_once_with(b"dummy_image")


def test_local_client_ocr_missing_binary_raises(tmp_path: Path) -> None:
    fake_model = tmp_path / "test.gguf"
    fake_model.write_text("fake gguf")

    client = LocalLlamaClient(
        model_path=fake_model, tesseract_cmd="nonexistent_tesseract_binary_xyz"
    )

    with pytest.raises(LLMError, match="Tesseract OCR executable"):
        client._run_ocr(b"dummy image bytes")
