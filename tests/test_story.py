"""Tests for StoryGenerator using a fake LLM client."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
from pydantic import BaseModel

from vocab_learner.config import Settings
from vocab_learner.models import Story, TeachingCard, WordConnections
from vocab_learner.story import StoryGenerator

T = TypeVar("T", bound=BaseModel)


class FakeLLMClient:
    def __init__(self, response: Story) -> None:
        self.response = response
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
        self.calls.append({"model": model, "user_prompt": user_prompt})
        return cast(T, self.response)

    def complete_vision_structured(self, **kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.fixture
def project_dirs(tmp_path: Path) -> Path:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "story.md").write_text("test story prompt")
    return tmp_path


def _make_settings(root: Path) -> Settings:
    return Settings(
        openai_api_key="sk-test",
        prompts_dir=root / "prompts",
        _env_file=None,
    )  # type: ignore[call-arg]


def _make_card(word: str) -> TeachingCard:
    return TeachingCard(
        word=word,
        simple_meaning="meaning",
        part_of_speech="noun",
        pronunciation_easy="pron",
        example_sentence="example",
        real_life_context="context",
        connections=WordConnections(),
        memory_hook="hook",
        personal_usage_pattern="pattern ___",
    )


class TestGenerate:
    def test_returns_story_for_cards(self, project_dirs: Path) -> None:
        canned = Story(
            title="A Test Story",
            body="Alice had an achievement. She reached an accord with Bob.",
            words_used=["achieve", "accord"],
            coverage=1.0,
        )
        fake = FakeLLMClient(canned)
        gen = StoryGenerator(client=fake, settings=_make_settings(project_dirs))

        cards = [_make_card("achieve"), _make_card("accord")]
        story = gen.generate(cards)

        assert story.title == "A Test Story"
        assert len(fake.calls) == 1

    def test_overwrites_llm_reported_coverage_with_ground_truth(self, project_dirs: Path) -> None:
        """LLM claims full coverage but only one word appears in the body."""
        canned = Story(
            title="Half Truth",
            body="Alice made an achievement today.",  # only "achieve" present
            words_used=["achieve", "accord"],  # LLM lies
            coverage=1.0,  # LLM lies
        )
        fake = FakeLLMClient(canned)
        gen = StoryGenerator(client=fake, settings=_make_settings(project_dirs))

        cards = [_make_card("achieve"), _make_card("accord")]
        story = gen.generate(cards)

        assert story.words_used == ["achieve"]
        assert story.coverage == 0.5

    def test_derived_forms_count(self, project_dirs: Path) -> None:
        """'accountable' in body should count as 'account' being used."""
        canned = Story(
            title="Derived",
            body="She was fully accountable for the outcome.",
            words_used=[],
            coverage=0.0,
        )
        fake = FakeLLMClient(canned)
        gen = StoryGenerator(client=fake, settings=_make_settings(project_dirs))

        cards = [_make_card("account")]
        story = gen.generate(cards)

        assert story.words_used == ["account"]
        assert story.coverage == 1.0

    def test_word_boundary_prevents_false_positive(self, project_dirs: Path) -> None:
        """'massacre' in body should NOT count as 'acre' being used."""
        canned = Story(
            title="No False Match",
            body="The battle was a massacre.",
            words_used=[],
            coverage=0.0,
        )
        fake = FakeLLMClient(canned)
        gen = StoryGenerator(client=fake, settings=_make_settings(project_dirs))

        cards = [_make_card("acre")]
        story = gen.generate(cards)

        assert story.words_used == []
        assert story.coverage == 0.0

    def test_raises_on_empty_cards(self, project_dirs: Path) -> None:
        fake = FakeLLMClient(Story(title="x", body="x", words_used=[], coverage=0.0))
        gen = StoryGenerator(client=fake, settings=_make_settings(project_dirs))

        with pytest.raises(ValueError, match="zero teaching cards"):
            gen.generate([])
