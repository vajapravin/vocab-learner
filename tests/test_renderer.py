"""Tests for the Markdown renderer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from vocab_learner.models import (
    DerivedForm,
    ExtractionResult,
    Session,
    Story,
    TeachingCard,
    VocabEntry,
    WordConnections,
)
from vocab_learner.renderer import MarkdownRenderer


def _make_session() -> Session:
    return Session(
        session_id="session_20260826_143012",
        extraction=ExtractionResult(
            entries=[
                VocabEntry(headword="achieve", raw_block="achieve, v.t. ..."),
                VocabEntry(
                    headword="account",
                    derived_forms=[DerivedForm(word="accountable")],
                    raw_block="account, n. ...",
                ),
            ],
            page_identifier="12",
            source_image_path=Path("fixtures/page_012.jpg"),
            model_used="gpt-4.1",
            extracted_at=datetime(2026, 8, 26, 14, 30, 0, tzinfo=UTC),
        ),
        teaching_cards=[
            TeachingCard(
                word="achieve",
                simple_meaning="To successfully complete a goal.",
                part_of_speech="verb (transitive)",
                pronunciation_easy="uh-CHEEV",
                pronunciation_ipa="/əˈtʃiːv/",  # noqa: RUF001
                example_sentence="She achieved her lifelong dream.",
                real_life_context="In discussions of goals, careers, and personal milestones.",
                connections=WordConnections(
                    synonyms=["accomplish", "attain"],
                    antonyms=["fail"],
                    collocations=["achieve a goal", "achieve success"],
                    related_words=["achievement", "achievable"],
                ),
                memory_hook="Picture crossing a finish line — you achieve the race.",
                personal_usage_pattern="I achieved ___ after ___.",
            ),
            TeachingCard(
                word="account",
                simple_meaning="A record of money spent or received.",
                part_of_speech="noun",
                pronunciation_easy="uh-KOUNT",
                pronunciation_ipa=None,  # optional field
                example_sentence="Please open a new account.",
                real_life_context="At banks and in financial paperwork.",
                connections=WordConnections(),  # all empty
                memory_hook="Think of counting money into an account.",
                personal_usage_pattern="I keep an account of ___.",
            ),
        ],
        story=Story(
            title="A Small Victory",
            body="Sara opened a new account and achieved her savings goal.",
            words_used=["achieve", "account"],
            coverage=1.0,
        ),
        created_at=datetime(2026, 8, 26, 14, 30, 12, tzinfo=UTC),
    )


class TestRender:
    def test_produces_non_empty_markdown(self) -> None:
        renderer = MarkdownRenderer()
        output = renderer.render(_make_session())

        assert output
        assert output.endswith("\n")

    def test_header_contains_session_metadata(self) -> None:
        renderer = MarkdownRenderer()
        output = renderer.render(_make_session())

        assert "# Vocabulary Session — Page 12" in output
        assert "session_20260826_143012" in output
        assert "gpt-4.1" in output
        assert "2 of 2 extracted" in output

    def test_story_section_present(self) -> None:
        renderer = MarkdownRenderer()
        output = renderer.render(_make_session())

        assert "## Story: A Small Victory" in output
        assert "Sara opened a new account" in output
        assert "(100%)" in output

    def test_all_cards_rendered_with_numbers(self) -> None:
        renderer = MarkdownRenderer()
        output = renderer.render(_make_session())

        assert "### 1. achieve" in output
        assert "### 2. account" in output

    def test_pronunciation_ipa_omitted_when_null(self) -> None:
        renderer = MarkdownRenderer()
        output = renderer.render(_make_session())

        # Card 1 has IPA and should include the separator
        assert "uh-CHEEV · /əˈtʃiːv/" in output  # noqa: RUF001
        # Card 2 has no IPA — no leftover separator
        assert "uh-KOUNT · " not in output
        assert "uh-KOUNT\n" in output or "uh-KOUNT\n\n" in output

    def test_empty_connections_omit_the_section(self) -> None:
        renderer = MarkdownRenderer()
        output = renderer.render(_make_session())

        # Card 1 has connections
        assert "**Synonyms:** accomplish, attain" in output
        # Card 2 has none — no bullet with empty content anywhere
        assert "**Synonyms:** \n" not in output
        assert "**Synonyms:**\n" not in output

    def test_story_appears_after_all_word_cards(self) -> None:
        """Story is the reward at the end, not the intro."""
        renderer = MarkdownRenderer()
        output = renderer.render(_make_session())

        last_card_pos = output.rfind("### 2. account")
        story_pos = output.find("## Story:")

        assert last_card_pos != -1
        assert story_pos != -1
        assert story_pos > last_card_pos, "Story should appear after the final word card"


class TestRenderToFile:
    def test_writes_file_and_returns_path(self, tmp_path: Path) -> None:
        renderer = MarkdownRenderer()
        target = tmp_path / "session.md"

        result = renderer.render_to_file(_make_session(), target)

        assert result == target
        assert target.exists()
        content = target.read_text(encoding="utf-8")
        assert "# Vocabulary Session — Page 12" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        renderer = MarkdownRenderer()
        target = tmp_path / "outputs" / "sessions" / "s.md"

        renderer.render_to_file(_make_session(), target)

        assert target.exists()


class TestEmptyCards:
    def test_no_cards_produces_placeholder(self) -> None:
        session = _make_session()
        session_no_cards = session.model_copy(update={"teaching_cards": []})

        renderer = MarkdownRenderer()
        output = renderer.render(session_no_cards)

        assert "_No teaching cards in this session._" in output


@pytest.mark.parametrize(
    "field_check,expected",
    [
        ("**Meaning:**", "To successfully complete a goal."),
        ("**Example:**", "She achieved her lifelong dream."),
        ("**Where you'll see it:**", "In discussions of goals"),
        ("**Memory hook:**", "Picture crossing a finish line"),
        ("**Try it yourself:**", "I achieved ___ after ___."),
    ],
)
def test_teaching_fields_appear_in_output(field_check: str, expected: str) -> None:
    """Each teaching card field lands somewhere in the Markdown."""
    renderer = MarkdownRenderer()
    output = renderer.render(_make_session())
    assert field_check in output
    assert expected in output
