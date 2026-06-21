"""LLM-based teacher: VocabEntry -> TeachingCard."""

from __future__ import annotations

from vocab_learner.config import Settings
from vocab_learner.llm.protocol import LLMClient, LLMError
from vocab_learner.logging_setup import get_logger
from vocab_learner.models import TeachingCard, VocabEntry

log = get_logger(__name__)


class Teacher:
    """Turns extracted vocabulary entries into rich teaching cards, one per word."""

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        path = self._settings.prompts_dir / "teaching.md"
        if not path.exists():
            raise FileNotFoundError(f"Teaching prompt not found: {path}")
        return path.read_text(encoding="utf-8")

    def teach_one(self, entry: VocabEntry) -> TeachingCard:
        """Produce a single teaching card for one vocabulary entry."""
        user_prompt = self._render_user_prompt(entry)
        log.info("teaching_start", word=entry.headword)

        card = self._client.complete_structured(
            model=self._settings.teaching_model,
            system_prompt=self._prompt,
            user_prompt=user_prompt,
            response_model=TeachingCard,
            max_tokens=self._settings.max_tokens,
        )

        # Guarantee the card's word matches the input, in case the LLM
        # rephrased it. This is a light integrity check, not a validation
        # of teaching quality.
        if card.word.strip().lower() != entry.headword.strip().lower():
            log.warning(
                "teaching_word_mismatch",
                expected=entry.headword,
                got=card.word,
            )
            card = card.model_copy(update={"word": entry.headword})

        log.info("teaching_done", word=entry.headword)
        return card

    def teach_all(self, entries: list[VocabEntry]) -> list[TeachingCard]:
        """
        Produce teaching cards for every entry. Fails soft: if teaching one
        word fails, log the failure and continue with the rest. Returns only
        successfully-taught cards.

        Rationale: at MVP scale (30 words per page), losing one card to a
        transient LLM issue is preferable to losing the whole session.
        """
        cards: list[TeachingCard] = []
        for entry in entries:
            try:
                cards.append(self.teach_one(entry))
            except LLMError as e:
                log.error(
                    "teaching_failed_for_word",
                    word=entry.headword,
                    error=str(e),
                )
        log.info(
            "teaching_batch_done",
            requested=len(entries),
            succeeded=len(cards),
        )
        return cards

    def _render_user_prompt(self, entry: VocabEntry) -> str:
        """Format one VocabEntry as the user message for the teaching LLM."""
        lines = [
            f"Headword: {entry.headword}",
        ]
        if entry.part_of_speech:
            lines.append(f"Part of speech (as printed): {entry.part_of_speech}")
        if entry.pronunciation_guide:
            lines.append(
                f"Pronunciation guide (as printed, Gujarati script): {entry.pronunciation_guide}"
            )
        if entry.derived_forms:
            forms = ", ".join(
                f"{f.word}{f' ({f.part_of_speech})' if f.part_of_speech else ''}"
                for f in entry.derived_forms
            )
            lines.append(f"Derived forms: {forms}")
        lines.append(f"Sense count in source: {entry.sense_count}")
        lines.append("Source block (may contain Gujarati; ignore non-English text):")
        lines.append(entry.raw_block)
        return "\n".join(lines)
