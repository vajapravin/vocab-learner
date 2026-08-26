"""Smoke test: extraction -> teaching -> story, printed as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vocab_learner.config import load_settings
from vocab_learner.extractor import Extractor
from vocab_learner.llm.factory import build_llm_client
from vocab_learner.logging_setup import configure_logging
from vocab_learner.story import StoryGenerator
from vocab_learner.teacher import Teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Teach only the first N extracted entries (cheap smoke runs).",
    )
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings.log_level)

    client = build_llm_client(settings)
    extractor = Extractor(client=client, settings=settings)
    teacher = Teacher(client=client, settings=settings)
    story_gen = StoryGenerator(client=client, settings=settings)

    extraction = extractor.extract(args.image)
    entries = extraction.entries
    if args.limit is not None:
        entries = entries[: args.limit]

    cards = teacher.teach_all(entries)
    story = story_gen.generate(cards)

    output = {
        "page": extraction.page_identifier,
        "extracted": len(extraction.entries),
        "taught": len(cards),
        "story": story.model_dump(mode="json"),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
