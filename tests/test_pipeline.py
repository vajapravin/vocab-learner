"""Tests for the end-to-end Pipeline orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
from pydantic import BaseModel

from vocab_learner.config import Settings
from vocab_learner.models import (
    ExtractionResult,
    Story,
    TeachingCard,
    VocabEntry,
    WordConnections,
)
from vocab_learner.pipeline import Pipeline

T = TypeVar("T", bound=BaseModel)


class RoutingFakeClient:
    """
    Fake LLMClient that returns different responses depending on the
    response_model requested. Lets us simulate the three-stage pipeline
    end-to-end without touching a real API.
    """

    def __init__(
        self,
        extraction: ExtractionResult,
        teaching_cards: list[TeachingCard],
        story: Story,
    ) -> None:
        self._extraction = extraction
        self._teaching_cards = teaching_cards
        self._story = story
        self._card_iter = iter(teaching_cards)
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({"stage": "vision", "response_model": response_model})
        return cast(T, self._extraction)

    def complete_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_tokens: int = 4096,
    ) -> T:
        self.calls.append(
            {"stage": "text", "response_model": response_model, "prompt": user_prompt}
        )
        if response_model is Story:
            return cast(T, self._story)
        # Teaching cards are consumed one per call
        return cast(T, next(self._card_iter))


@pytest.fixture
def project_dirs(tmp_path: Path) -> Path:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "extraction.md").write_text("extract prompt")
    (tmp_path / "prompts" / "teaching.md").write_text("teach prompt")
    (tmp_path / "prompts" / "story.md").write_text("story prompt")
    (tmp_path / "output").mkdir()
    return tmp_path


def _make_settings(root: Path) -> Settings:
    return Settings(
        openai_api_key="sk-test",
        prompts_dir=root / "prompts",
        output_dir=root / "output",
        _env_file=None,
    )  # type: ignore[call-arg]


def _tiny_jpeg(tmp_path: Path) -> Path:
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
    p = tmp_path / "page.jpg"
    p.write_bytes(payload)
    return p


def _make_extraction(*headwords: str) -> ExtractionResult:
    return ExtractionResult(
        entries=[VocabEntry(headword=w, raw_block=f"{w}, ...") for w in headwords],
        page_identifier="12",
        source_image_path=Path("placeholder"),
        model_used="fake",
        extracted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _make_card(word: str) -> TeachingCard:
    return TeachingCard(
        word=word,
        simple_meaning="m",
        part_of_speech="noun",
        pronunciation_easy="p",
        example_sentence="e",
        real_life_context="c",
        connections=WordConnections(),
        memory_hook="h",
        personal_usage_pattern="I ___.",
    )


def _make_story() -> Story:
    return Story(title="t", body="alpha and beta", words_used=[], coverage=0.0)


class TestPipelineRun:
    def test_happy_path_produces_session_and_file(self, project_dirs: Path) -> None:
        image = _tiny_jpeg(project_dirs)
        settings = _make_settings(project_dirs)

        client = RoutingFakeClient(
            extraction=_make_extraction("alpha", "beta"),
            teaching_cards=[_make_card("alpha"), _make_card("beta")],
            story=_make_story(),
        )
        pipeline = Pipeline(client=client, settings=settings)

        result = pipeline.run(image)

        assert result.session.session_id.startswith("session_")
        assert len(result.session.teaching_cards) == 2
        assert result.output_path.exists()
        assert result.output_path.suffix == ".md"

    def test_output_file_contains_rendered_markdown(self, project_dirs: Path) -> None:
        image = _tiny_jpeg(project_dirs)
        settings = _make_settings(project_dirs)

        client = RoutingFakeClient(
            extraction=_make_extraction("alpha"),
            teaching_cards=[_make_card("alpha")],
            story=_make_story(),
        )
        pipeline = Pipeline(client=client, settings=settings)

        result = pipeline.run(image)
        content = result.output_path.read_text(encoding="utf-8")

        assert "# Vocabulary Session — Page 12" in content
        assert "### 1. alpha" in content

    def test_zero_extracted_entries_raises(self, project_dirs: Path) -> None:
        image = _tiny_jpeg(project_dirs)
        settings = _make_settings(project_dirs)

        client = RoutingFakeClient(
            extraction=_make_extraction(),  # empty
            teaching_cards=[],
            story=_make_story(),
        )
        pipeline = Pipeline(client=client, settings=settings)

        with pytest.raises(RuntimeError, match="zero entries"):
            pipeline.run(image)

    def test_session_id_format(self, project_dirs: Path) -> None:
        image = _tiny_jpeg(project_dirs)
        settings = _make_settings(project_dirs)

        client = RoutingFakeClient(
            extraction=_make_extraction("alpha"),
            teaching_cards=[_make_card("alpha")],
            story=_make_story(),
        )
        pipeline = Pipeline(client=client, settings=settings)

        result = pipeline.run(image)

        # session_YYYYMMDD_HHMMSS = 22 chars total
        assert len(result.session.session_id) == len("session_20260826_143012")
        assert result.session.session_id[8:16].isdigit()

    def test_pipeline_reusable_across_runs(self, project_dirs: Path) -> None:
        """Two run() calls should produce two distinct session IDs and files."""

        image = _tiny_jpeg(project_dirs)
        settings = _make_settings(project_dirs)

        def make_client() -> RoutingFakeClient:
            return RoutingFakeClient(
                extraction=_make_extraction("alpha"),
                teaching_cards=[_make_card("alpha")],
                story=_make_story(),
            )

        pipeline = Pipeline(client=make_client(), settings=settings)
        pipeline.run(image)
