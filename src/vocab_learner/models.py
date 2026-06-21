"""Pydantic data contracts for pipeline stages."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from vocab_learner.utils import utcnow


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
        description="Full raw text of this entry block from the source page.",
    )


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[VocabEntry]
    page_identifier: str | None = None
    source_image_path: Path
    model_used: str
    extracted_at: datetime = Field(default_factory=utcnow)


class WordConnections(BaseModel):
    model_config = ConfigDict(extra="forbid")

    synonyms: list[str] = Field(default_factory=list)
    antonyms: list[str] = Field(default_factory=list)
    collocations: list[str] = Field(
        default_factory=list,
        description="Common multi-word phrases the headword appears in.",
    )
    related_words: list[str] = Field(
        default_factory=list,
        description="Word family members, cognates, morphologically related terms.",
    )


class TeachingCard(BaseModel):
    """One vocabulary teaching card in the 9-section format."""

    model_config = ConfigDict(extra="forbid")

    word: str = Field(..., min_length=1)
    simple_meaning: str = Field(..., min_length=1)
    part_of_speech: str = Field(..., min_length=1)
    pronunciation_easy: str = Field(
        ...,
        min_length=1,
        description="Human-readable pronunciation guide, e.g. 'uh-KOM-uh-dayt'.",
    )
    pronunciation_ipa: str | None = Field(
        default=None,
        description="IPA transcription, if known. Optional.",
    )
    example_sentence: str = Field(..., min_length=1)
    real_life_context: str = Field(
        ...,
        min_length=1,
        description="Where and how this word is naturally used.",
    )
    connections: WordConnections
    memory_hook: str = Field(
        ...,
        min_length=1,
        description="Short memorable association, analogy, or mental image.",
    )
    personal_usage_pattern: str = Field(
        ...,
        min_length=1,
        description="Fill-in-the-blank sentence pattern learners can adapt.",
    )
