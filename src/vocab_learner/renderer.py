"""Markdown renderer for a completed Session."""

from __future__ import annotations

from pathlib import Path

from vocab_learner.logging_setup import get_logger
from vocab_learner.models import Session, TeachingCard, WordConnections

log = get_logger(__name__)


class MarkdownRenderer:
    """
    Renders a Session as human-readable Markdown. Pure formatting — no LLM
    calls, no I/O beyond the final write.
    """

    def render(self, session: Session) -> str:
        """Produce the full Markdown document as a string."""
        parts: list[str] = []
        parts.append(self._render_header(session))
        parts.append(self._render_words(session.teaching_cards))
        parts.append(self._render_story(session))
        return "\n\n".join(parts).rstrip() + "\n"

    def render_to_file(self, session: Session, output_path: Path) -> Path:
        """Render and write to disk. Returns the path written to."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.render(session)
        output_path.write_text(content, encoding="utf-8")
        log.info(
            "session_rendered",
            session_id=session.session_id,
            path=str(output_path),
            bytes=len(content),
        )
        return output_path

    # -- section renderers ---------------------------------------------------

    def _render_header(self, session: Session) -> str:
        ex = session.extraction
        page = ex.page_identifier or "unknown"
        return "\n".join(
            [
                f"# Vocabulary Session — Page {page}",
                "",
                f"**Source:** `{ex.source_image_path}`  ",
                f"**Session:** `{session.session_id}`  ",
                f"**Date:** {session.created_at.strftime('%Y-%m-%d %H:%M UTC')}  ",
                f"**Model:** {ex.model_used}  ",
                f"**Words taught:** {len(session.teaching_cards)} of {len(ex.entries)} extracted",
                "",
                "---",
            ]
        )

    def _render_story(self, session: Session) -> str:
        story = session.story
        coverage_pct = round(story.coverage * 100)
        return "\n".join(
            [
                f"## Story: {story.title}",
                "",
                f"_Words used: {len(story.words_used)} / "
                f"{len(session.teaching_cards)} ({coverage_pct}%)_",
                "",
                story.body,
            ]
        )

    def _render_words(self, cards: list[TeachingCard]) -> str:
        if not cards:
            return "## Words\n\n_No teaching cards in this session._\n\n---"

        blocks: list[str] = ["## Words"]
        for i, card in enumerate(cards, start=1):
            blocks.append(self._render_card(i, card))
        return "\n\n".join(blocks) + "\n\n---"

    def _render_card(self, index: int, card: TeachingCard) -> str:
        pronunciation = card.pronunciation_easy
        if card.pronunciation_ipa:
            pronunciation = f"{card.pronunciation_easy} · {card.pronunciation_ipa}"

        lines = [
            f"### {index}. {card.word} _({card.part_of_speech})_",
            "",
            f"**Pronunciation:** {pronunciation}",
            "",
            f"**Meaning:** {card.simple_meaning}",
        ]

        if card.gujarati_meaning:
            lines.extend(
                [
                    "",
                    f"**Gujarati:** {card.gujarati_meaning}",
                ]
            )

        lines.extend(
            [
                "",
                f'**Example:** _"{card.example_sentence}"_',
                "",
                f"**Where you'll see it:** {card.real_life_context}",
                "",
                f"**Memory hook:** {card.memory_hook}",
            ]
        )

        connections_block = self._render_connections(card.connections)
        if connections_block:
            lines.append("")
            lines.append("**Word connections:**")
            lines.append(connections_block)

        lines.append("")
        lines.append(f"**Try it yourself:** {card.personal_usage_pattern}")

        return "\n".join(lines)

    @staticmethod
    def _render_connections(connections: WordConnections) -> str:
        """Return a bullet list of non-empty connection categories, or empty string."""
        items: list[tuple[str, list[str]]] = [
            ("Synonyms", connections.synonyms),
            ("Antonyms", connections.antonyms),
            ("Collocations", connections.collocations),
            ("Word family", connections.related_words),
        ]
        rendered = [f"- **{label}:** {', '.join(values)}" for label, values in items if values]
        return "\n".join(rendered)
