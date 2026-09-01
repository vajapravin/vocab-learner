"""Tests for Teacher using a fake LLM client."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
from pydantic import BaseModel

from vocab_learner.config import Settings
from vocab_learner.llm.protocol import LLMError
from vocab_learner.models import (
    DerivedForm,
    TeachingCard,
    VocabEntry,
    WordConnections,
)
from vocab_learner.teacher import Teacher

T = TypeVar("T", bound=BaseModel)


class FakeLLMClient:
    """
    Fake LLMClient that returns pre-programmed responses. Configure by
    passing either a single card (returned every call) or a callable that
    takes the user_prompt and returns a card (or raises).
    """

    def __init__(
        self,
        response: TeachingCard | Any,
    ) -> None:
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
        self.calls.append(
            {"model": model, "user_prompt": user_prompt, "response_model": response_model}
        )
        if callable(self.response):
            result = self.response(user_prompt)
            if isinstance(result, Exception):
                raise result
            return cast(T, result)
        return cast(T, self.response)

    def complete_vision_structured(self, **kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.fixture
def project_dirs(tmp_path: Path) -> Path:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "teaching.md").write_text("test teaching prompt")
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
        simple_meaning="A test meaning.",
        part_of_speech="noun",
        pronunciation_easy="test",
        example_sentence="This is a test.",
        real_life_context="Used in tests.",
        connections=WordConnections(),
        memory_hook="Remember: this is a test.",
        personal_usage_pattern="I test ___ every day.",
    )


def _make_entry(word: str) -> VocabEntry:
    return VocabEntry(
        headword=word,
        part_of_speech="v.t.",
        derived_forms=[DerivedForm(word=f"{word}ment", part_of_speech="n.")],
        raw_block=f"{word}, v.t. ...",
    )


class TestTeachOne:
    def test_produces_card_for_entry(self, project_dirs: Path) -> None:
        fake = FakeLLMClient(_make_card("achieve"))
        teacher = Teacher(client=fake, settings=_make_settings(project_dirs))

        card = teacher.teach_one(_make_entry("achieve"))

        assert card.word == "achieve"
        assert len(fake.calls) == 1

    def test_corrects_word_mismatch(self, project_dirs: Path) -> None:
        """If the LLM returns a different headword, Teacher forces it back."""
        fake = FakeLLMClient(_make_card("achievement"))
        teacher = Teacher(client=fake, settings=_make_settings(project_dirs))

        card = teacher.teach_one(_make_entry("achieve"))

        assert card.word == "achieve"

    def test_user_prompt_contains_headword_and_pos(self, project_dirs: Path) -> None:
        fake = FakeLLMClient(_make_card("achieve"))
        teacher = Teacher(client=fake, settings=_make_settings(project_dirs))

        teacher.teach_one(_make_entry("achieve"))

        prompt = fake.calls[0]["user_prompt"]
        assert "achieve" in prompt
        assert "v.t." in prompt
        assert "achievement" in prompt  # derived form

    def test_gujarati_meaning_passed_to_prompt(self, project_dirs: Path) -> None:
        """The extracted Gujarati is included in the teaching prompt."""
        fake = FakeLLMClient(_make_card("achieve"))
        teacher = Teacher(client=fake, settings=_make_settings(project_dirs))

        entry = VocabEntry(
            headword="achieve",
            raw_block="achieve, v.t. ...",
            gujarati_meaning="પ્રાપ્ત કરવું",
        )
        teacher.teach_one(entry)

        prompt = fake.calls[0]["user_prompt"]
        assert "પ્રાપ્ત કરવું" in prompt
        assert "copy this into the card unchanged" in prompt

    def test_gujarati_meaning_absent_from_prompt_when_null(self, project_dirs: Path) -> None:
        """No Gujarati line appears in the prompt when the field is null."""
        fake = FakeLLMClient(_make_card("achieve"))
        teacher = Teacher(client=fake, settings=_make_settings(project_dirs))

        teacher.teach_one(_make_entry("achieve"))  # no gujarati_meaning

        prompt = fake.calls[0]["user_prompt"]
        assert "Gujarati meaning" not in prompt

    def test_gujarati_meaning_overwritten_from_extraction(self, project_dirs: Path) -> None:
        """If the LLM alters Gujarati, extraction's version wins."""
        # LLM returns card with wrong Gujarati (paraphrased or reformatted)
        drifted_card = _make_card("achieve")
        drifted_card = drifted_card.model_copy(
            update={"gujarati_meaning": "કંઈક પ્રાપ્ત કરવું"}  # "extra" text
        )
        fake = FakeLLMClient(drifted_card)
        teacher = Teacher(client=fake, settings=_make_settings(project_dirs))

        entry = VocabEntry(
            headword="achieve",
            raw_block="achieve, v.t. ...",
            gujarati_meaning="પ્રાપ્ત કરવું",  # original source
        )
        card = teacher.teach_one(entry)

        # Extraction's version wins, LLM's drift is discarded
        assert card.gujarati_meaning == "પ્રાપ્ત કરવું"


class TestTeachAll:
    def test_teaches_every_entry(self, project_dirs: Path) -> None:
        def responder(prompt: str) -> TeachingCard:
            # Return a card whose word matches whichever headword is in the prompt.
            for candidate in ("alpha", "beta", "gamma"):
                if candidate in prompt:
                    return _make_card(candidate)
            raise AssertionError(f"unexpected prompt: {prompt}")

        fake = FakeLLMClient(responder)
        teacher = Teacher(client=fake, settings=_make_settings(project_dirs))

        entries = [_make_entry("alpha"), _make_entry("beta"), _make_entry("gamma")]
        cards = teacher.teach_all(entries)

        assert len(cards) == 3
        assert [c.word for c in cards] == ["alpha", "beta", "gamma"]

    def test_soft_fails_on_single_error(self, project_dirs: Path) -> None:
        """One LLMError shouldn't abort the whole batch."""

        def responder(prompt: str) -> TeachingCard | Exception:
            if "beta" in prompt:
                return LLMError("simulated failure")
            for candidate in ("alpha", "gamma"):
                if candidate in prompt:
                    return _make_card(candidate)
            raise AssertionError(f"unexpected prompt: {prompt}")

        fake = FakeLLMClient(responder)
        teacher = Teacher(client=fake, settings=_make_settings(project_dirs))

        entries = [_make_entry("alpha"), _make_entry("beta"), _make_entry("gamma")]
        cards = teacher.teach_all(entries)

        assert len(cards) == 2
        assert [c.word for c in cards] == ["alpha", "gamma"]


def test_missing_prompt_raises(tmp_path: Path) -> None:
    settings = Settings(
        openai_api_key="sk-test",
        prompts_dir=tmp_path / "does-not-exist",
        _env_file=None,
    )  # type: ignore[call-arg]
    fake = FakeLLMClient(_make_card("x"))
    with pytest.raises(FileNotFoundError, match="Teaching prompt not found"):
        Teacher(client=fake, settings=settings)
