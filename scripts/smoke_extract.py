"""Throwaway smoke test: run the extractor on one image and print JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vocab_learner.config import load_settings
from vocab_learner.extractor import Extractor
from vocab_learner.llm.factory import build_llm_client
from vocab_learner.logging_setup import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings.log_level)

    client = build_llm_client(settings)
    extractor = Extractor(client=client, settings=settings)
    result = extractor.extract(args.image)

    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
