"""Tests for Extractor using a fake LLM client."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from vocab_learner.config import Settings
from vocab_learner.extractor import Extractor
from vocab_learner.models import DerivedForm, ExtractionResult, VocabEntry

T = TypeVar("T", bound=BaseModel)


class FakeLLMClient:
    """Satisfies the LLMClient protocol structurally. No real API calls."""

    def __init__(self, canned_result: ExtractionResult) -> None:
        self.canned_result = canned_result
        self.calls: list[dict[str, Any]] = []

    def complete_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_tokens: int = 4096,
    ) -> T:
        raise NotImplementedError

    def complete_vision_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        image_media_type: str,
        response_model: type[T],
        max_tokens: int = 4096,
    ) -> T:
        self.calls.append(
            {
                "model": model,
                "image_bytes_len": len(image_bytes),
                "image_media_type": image_media_type,
            }
        )
        return self.canned_result  # type: ignore[return-value]


@pytest.fixture
def tiny_jpeg(tmp_path: Path) -> Path:
    # Smallest valid JPEG possible (baseline, 1x1 white pixel).
    payload = bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000"
        "ffdb004300080606070605080707070909080a0c"
        "140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20"
        "242e2720222c231c1c2837292c30313434341f27"
        "393d38323c2e333432ffc00011080001000103012200021101031101"
        "ffc4001f0000010501010101010100000000000000000102030405060708090a0b"
        "ffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f0243362728209"
        "0a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa"
        "ffda000c03010002110311003f00fbd0a28a2803ffd9"
    )
    p = tmp_path / "tiny.jpg"
    p.write_bytes(payload)
    return p


@pytest.fixture
def project_dirs(tmp_path: Path) -> Path:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "extraction.md").write_text("test prompt")
    return tmp_path


def _make_settings(root: Path) -> Settings:
    return Settings(
        anthropic_api_key="sk-test",
        prompts_dir=root / "prompts",
        output_dir=root / "output",
        fixtures_dir=root / "fixtures",
        _env_file=None,  # type: ignore[call-arg]
    )


def _canned_result() -> ExtractionResult:
    return ExtractionResult(
        entries=[
            VocabEntry(
                headword="achieve",
                part_of_speech="v.t.",
                pronunciation_guide="(અચી'વ્)",
                derived_forms=[DerivedForm(word="achievement", part_of_speech="n.")],
                sense_count=2,
                raw_block="achieve, v.t. ...",
            )
        ],
        page_identifier="12",
        source_image_path=Path("placeholder"),
        model_used="placeholder",
        extracted_at=datetime.now(UTC),
    )


def test_extract_calls_client_with_image(tiny_jpeg: Path, project_dirs: Path) -> None:
    fake = FakeLLMClient(_canned_result())
    settings = _make_settings(project_dirs)

    ext = Extractor(client=fake, settings=settings)
    result = ext.extract(tiny_jpeg)

    assert len(fake.calls) == 1
    assert fake.calls[0]["image_media_type"] == "image/jpeg"
    assert fake.calls[0]["image_bytes_len"] > 0
    assert result.entries[0].headword == "achieve"


def test_extract_overwrites_provenance_fields(tiny_jpeg: Path, project_dirs: Path) -> None:
    """LLM output for source_image_path and model_used must be discarded."""
    fake = FakeLLMClient(_canned_result())
    settings = _make_settings(project_dirs)

    ext = Extractor(client=fake, settings=settings)
    result = ext.extract(tiny_jpeg)

    assert result.source_image_path == tiny_jpeg
    assert result.model_used == settings.extraction_model


def test_extract_raises_on_missing_file(project_dirs: Path) -> None:
    fake = FakeLLMClient(_canned_result())
    settings = _make_settings(project_dirs)
    ext = Extractor(client=fake, settings=settings)

    with pytest.raises(FileNotFoundError):
        ext.extract(Path("/nonexistent/file.jpg"))


def test_extract_raises_on_unsupported_media_type(tmp_path: Path, project_dirs: Path) -> None:
    bogus = tmp_path / "notes.txt"
    bogus.write_text("hi")

    fake = FakeLLMClient(_canned_result())
    settings = _make_settings(project_dirs)
    ext = Extractor(client=fake, settings=settings)

    with pytest.raises(ValueError, match="Unsupported image type"):
        ext.extract(bogus)


def test_extract_prompt_missing_raises(tmp_path: Path) -> None:
    settings = Settings(
        anthropic_api_key="sk-test",
        prompts_dir=tmp_path / "does-not-exist",
        _env_file=None,  # type: ignore[call-arg]
    )
    fake = FakeLLMClient(_canned_result())
    with pytest.raises(FileNotFoundError, match="Extraction prompt not found"):
        Extractor(client=fake, settings=settings)
