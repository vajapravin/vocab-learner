"""End-to-end orchestrator: image path -> rendered session on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vocab_learner.config import Settings
from vocab_learner.extractor import Extractor
from vocab_learner.llm.protocol import LLMClient
from vocab_learner.logging_setup import get_logger
from vocab_learner.models import Session
from vocab_learner.renderer import MarkdownRenderer
from vocab_learner.story import StoryGenerator
from vocab_learner.teacher import Teacher
from vocab_learner.utils import utcnow

log = get_logger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Return value of Pipeline.run(). Both the data and where it landed."""

    session: Session
    output_path: Path


class Pipeline:
    """
    Orchestrates the full extraction -> teaching -> story -> render flow
    for one image. Reusable across many run() calls.

    Fails loud when a whole stage produces nothing usable; delegates
    within-stage resilience to the stages themselves.
    """

    def __init__(
        self,
        client: LLMClient,
        settings: Settings,
        renderer: MarkdownRenderer | None = None,
    ) -> None:
        self._settings = settings
        self._extractor = Extractor(client=client, settings=settings)
        self._teacher = Teacher(client=client, settings=settings)
        self._story_gen = StoryGenerator(client=client, settings=settings)
        self._renderer = renderer or MarkdownRenderer()

    def run(self, image_path: Path) -> PipelineResult:
        session_id = self._new_session_id()
        log.info("pipeline_start", session_id=session_id, image=str(image_path))

        extraction = self._extractor.extract(image_path)
        if not extraction.entries:
            raise RuntimeError(
                f"Extraction produced zero entries for {image_path}. Aborting — nothing to teach."
            )

        cards = self._teacher.teach_all(extraction.entries)
        if not cards:
            raise RuntimeError(
                f"Teaching produced zero cards from {len(extraction.entries)} "
                "extracted entries. Aborting — nothing to build a story from."
            )

        story = self._story_gen.generate(cards)

        session = Session(
            session_id=session_id,
            extraction=extraction,
            teaching_cards=cards,
            story=story,
        )

        output_path = self._settings.output_dir / f"{session_id}.md"
        self._renderer.render_to_file(session, output_path)

        log.info(
            "pipeline_done",
            session_id=session_id,
            extracted=len(extraction.entries),
            taught=len(cards),
            coverage=story.coverage,
            output=str(output_path),
        )
        return PipelineResult(session=session, output_path=output_path)

    @staticmethod
    def _new_session_id() -> str:
        return "session_" + utcnow().strftime("%Y%m%d_%H%M%S")
