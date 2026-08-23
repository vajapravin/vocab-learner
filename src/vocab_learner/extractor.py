"""Vision-LLM-based extractor: image -> ExtractionResult."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from vocab_learner.config import Settings
from vocab_learner.llm.protocol import LLMClient
from vocab_learner.logging_setup import get_logger
from vocab_learner.models import ExtractionResult

log = get_logger(__name__)

SUPPORTED_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/gif"})


class Extractor:
    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        path = self._settings.prompts_dir / "extraction.md"
        if not path.exists():
            raise FileNotFoundError(f"Extraction prompt not found: {path}")
        return path.read_text(encoding="utf-8")

    def extract(self, image_path: Path) -> ExtractionResult:
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        media_type, _ = mimetypes.guess_type(image_path)
        if media_type not in SUPPORTED_MEDIA_TYPES:
            raise ValueError(f"Unsupported image type: {media_type} ({image_path})")

        image_bytes = image_path.read_bytes()
        log.info(
            "extraction_start",
            image=str(image_path),
            bytes=len(image_bytes),
            model=self._settings.extraction_model,
        )

        result = self._client.complete_vision_structured(
            model=self._settings.extraction_model,
            system_prompt=self._prompt,
            user_prompt="Extract the dictionary entries from this page.",
            image_bytes=image_bytes,
            image_media_type=media_type,
            response_model=ExtractionResult,
            max_tokens=self._settings.max_tokens,
        )

        # Overwrite provenance fields with ground truth
        result = result.model_copy(
            update={
                "source_image_path": image_path,
                "model_used": self._settings.extraction_model,
            }
        )

        log.info(
            "extraction_done",
            entries=len(result.entries),
            page=result.page_identifier,
        )
        return result
