"""Tests for Pydantic data contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vocab_learner.models import DerivedForm, ExtractionResult, VocabEntry


class TestVocabEntry:
    def test_minimal_valid_entry(self) -> None:
        entry = VocabEntry(headword="achieve", raw_block="achieve, v.t. ...")
        assert entry.headword == "achieve"
        assert entry.part_of_speech is None
        assert entry.derived_forms == []
        assert entry.sense_count == 1

    def test_full_entry_with_derived_forms(self) -> None:
        entry = VocabEntry(
            headword="achieve",
            part_of_speech="v.t.",
            pronunciation_guide="(અચી'વ્)",
            derived_forms=[
                DerivedForm(word="achievable", part_of_speech="a."),
                DerivedForm(word="achievement", part_of_speech="n."),
            ],
            sense_count=2,
            raw_block="achieve, v.t. ... achievable, a. ... achievement, n. ...",
        )
        assert len(entry.derived_forms) == 2
        assert entry.derived_forms[0].word == "achievable"

    def test_empty_headword_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VocabEntry(headword="", raw_block="x")

    def test_extra_fields_rejected(self) -> None:
        """LLM must not sneak unknown fields past validation."""
        with pytest.raises(ValidationError):
            VocabEntry.model_validate(
                {
                    "headword": "achieve",
                    "raw_block": "x",
                    "hallucinated_field": "oops",
                }
            )

    def test_sense_count_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            VocabEntry(headword="achieve", raw_block="x", sense_count=0)

    def test_gujarati_meaning_optional_and_defaults_to_none(self) -> None:
        """gujarati_meaning is optional (deferred Phase 2 field)."""
        entry = VocabEntry(headword="achieve", raw_block="achieve, v.t. ...")
        assert entry.gujarati_meaning is None

    def test_gujarati_meaning_preserved_verbatim(self) -> None:
        """When present, the field stores the source string as-is, no processing."""
        entry = VocabEntry(
            headword="achieve",
            raw_block="achieve, v.t. પ્રાપ્ત કરવું; (2) સફળતાપૂર્વક પૂરું કરવું",
            gujarati_meaning="પ્રાપ્ત કરવું; (2) સફળતાપૂર્વક પૂરું કરવું",
        )
        assert entry.gujarati_meaning == "પ્રાપ્ત કરવું; (2) સફળતાપૂર્વક પૂરું કરવું"


class TestExtractionResult:
    def test_roundtrip_json(self) -> None:
        result = ExtractionResult(
            entries=[VocabEntry(headword="achieve", raw_block="x")],
            page_identifier="12",
            source_image_path=Path("fixtures/page_012.jpg"),
            model_used="claude-sonnet-5",
        )
        dumped = result.model_dump_json()
        restored = ExtractionResult.model_validate_json(dumped)
        assert restored.entries[0].headword == "achieve"
        assert restored.page_identifier == "12"

    def test_empty_entries_allowed(self) -> None:
        """A blank page is a valid extraction result, just an empty one."""
        result = ExtractionResult(
            entries=[],
            source_image_path=Path("fixtures/blank.jpg"),
            model_used="claude-sonnet-5",
        )
        assert result.entries == []
