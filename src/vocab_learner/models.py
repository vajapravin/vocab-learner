"""Pydantic data contracts for pipeline stages."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core.utils import _utcnow


class DerivedForm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word: str = Field(..., min_length=1)
    part_of_speech: str | None = None


class VocabEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headword: str = Field(..., min_length=1)
    part_of_speech: str | None = None
    pronunciation_guide: str | None = Field(
        default=None,
        description="Parenthesized transliteration as printed in the dictionary.",
    )
    derived_forms: list[DerivedForm] = Field(default_factory=list)
    sense_count: int = Field(default=1, ge=1)
    raw_block: str = Field(
        ...,
        description="Full raw text of this entry block from the source page. For debugging.",
    )


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[VocabEntry]
    page_identifier: str | None = None
    source_image_path: Path
    model_used: str
    extracted_at: datetime = Field(default_factory=_utcnow)
