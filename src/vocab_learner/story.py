"""LLM-based story generator: TeachingCard list -> one cohesive story."""

from __future__ import annotations

import re

from vocab_learner.config import Settings
from vocab_learner.llm.protocol import LLMClient
from vocab_learner.logging_setup import get_logger
from vocab_learner.models import Story, TeachingCard

log = get_logger(__name__)


class StoryGenerator:
    """
    Produces one short story from a list of teaching cards. The story tries
    to use as many of the session's words as possible without straining prose
    quality.
    """

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        path = self._settings.prompts_dir / "story.md"
        if not path.exists():
            raise FileNotFoundError(f"Story prompt not found: {path}")
        return path.read_text(encoding="utf-8")

    def generate(self, cards: list[TeachingCard]) -> Story:
        if not cards:
            raise ValueError("Cannot generate a story from zero teaching cards.")

        words = [card.word for card in cards]
        user_prompt = self._render_user_prompt(words)
        log.info("story_start", word_count=len(words))

        story = self._client.complete_structured(
            model=self._settings.teaching_model,
            system_prompt=self._prompt,
            user_prompt=user_prompt,
            response_model=Story,
            max_tokens=self._settings.max_tokens,
        )

        # Ground-truth verification: the LLM's self-reported words_used and
        # coverage are advisory. Recompute both from the actual body text.
        verified_words = self._words_present_in(story.body, words)
        verified_coverage = len(verified_words) / len(words)

        if set(verified_words) != set(story.words_used):
            log.warning(
                "story_words_used_mismatch",
                claimed=sorted(story.words_used),
                verified=sorted(verified_words),
            )

        story = story.model_copy(
            update={
                "words_used": verified_words,
                "coverage": round(verified_coverage, 3),
            }
        )

        log.info(
            "story_done",
            word_count=len(words),
            words_used=len(verified_words),
            coverage=story.coverage,
        )
        return story

    def _render_user_prompt(self, words: list[str]) -> str:
        return (
            f"Total vocabulary words in this session: {len(words)}\n"
            f"Words to use as naturally as possible:\n" + "\n".join(f"  - {w}" for w in words)
        )

    @staticmethod
    def _words_present_in(body: str, target_words: list[str]) -> list[str]:
        """
        Return the subset of `target_words` whose headword or a simple
        morphological variant appears in `body`, case-insensitively.

        This is intentionally lenient: matches the headword's stem so that
        "achievement" counts as a use of "achieve", "accountable" as "account".
        Uses word-boundary regex to avoid false positives ("acre" matching "massacre").
        """
        body_lower = body.lower()
        present: list[str] = []
        for word in target_words:
            stem = word.lower().rstrip("e")  # crude but effective: achieve -> achiev
            # Word boundary + stem + optional continuation (able, ment, ing, ed, s, ...)
            pattern = rf"\b{re.escape(stem)}[a-z]*\b"
            if re.search(pattern, body_lower):
                present.append(word)
        return present
