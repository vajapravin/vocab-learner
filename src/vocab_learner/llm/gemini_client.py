"""Gemini implementation of LLMClient using google-genai."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from vocab_learner.llm.protocol import LLMError, LLMValidationError
from vocab_learner.logging_setup import get_logger

T = TypeVar("T", bound=BaseModel)

log = get_logger(__name__)

GeminiContentPart = types.Part | str


class GeminiClient:
    """
    Gemini-backed LLMClient using the google-genai SDK.
    Uses structured output configuration (response_mime_type="application/json"
    and response_schema) plus a schema hint in system instructions, validates
    via Pydantic, and retries once on validation failure.
    """

    def __init__(self, api_key: str, timeout_seconds: float = 120.0) -> None:
        timeout_ms = int(timeout_seconds * 1000)
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

    def complete_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_tokens: int = 4096,
    ) -> T:
        return self._call(
            model=model,
            system_prompt=system_prompt,
            contents=[user_prompt],
            response_model=response_model,
            max_tokens=max_tokens,
        )

    def complete_vision_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        image_media_type: str,
        response_model: type[T],
        max_tokens: int = 4096,
    ) -> T:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=image_media_type)
        return self._call(
            model=model,
            system_prompt=system_prompt,
            contents=[image_part, user_prompt],
            response_model=response_model,
            max_tokens=max_tokens,
        )

    # -- internals -----------------------------------------------------------

    def _call(
        self,
        *,
        model: str,
        system_prompt: str,
        contents: list[GeminiContentPart],
        response_model: type[T],
        max_tokens: int,
    ) -> T:
        schema_hint = self._schema_hint(response_model)
        system_full = f"{system_prompt}\n\n{schema_hint}"

        raw_text = self._send(
            model=model,
            system=system_full,
            contents=contents,
            max_tokens=max_tokens,
        )
        parsed, first_error = self._try_parse(raw_text, response_model)
        if parsed is not None:
            return parsed

        log.warning("llm_validation_failed_first_attempt", error=str(first_error))

        repair_text = (
            "Your previous response could not be parsed as valid JSON "
            "matching the required schema. Error:\n\n"
            f"{first_error}\n\n"
            "Return ONLY the corrected JSON. No prose, no code fences."
        )
        repair_contents: list[GeminiContentPart] = [*contents, repair_text]
        raw_text = self._send(
            model=model,
            system=system_full,
            contents=repair_contents,
            max_tokens=max_tokens,
        )
        parsed, final_error = self._try_parse(raw_text, response_model)
        if parsed is not None:
            return parsed

        log.error("llm_validation_failed_final", error=str(final_error))
        raise LLMValidationError(f"LLM output failed validation after retry: {final_error}")

    def _send(
        self,
        *,
        model: str,
        system: str,
        contents: list[GeminiContentPart],
        max_tokens: int,
    ) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=contents,  # type: ignore[arg-type]
                config=config,
            )
        except errors.APIError as e:
            log.error("gemini_api_error", error=str(e))
            raise LLMError(f"Gemini API error: {e}") from e

        text = response.text or ""
        log.debug("llm_raw_response", chars=len(text))
        return text.strip()

    def _try_parse(self, raw: str, response_model: type[T]) -> tuple[T | None, Exception | None]:
        try:
            data = self._parse_json(raw)
            parsed: T = response_model.model_validate(data)
            return parsed, None
        except (json.JSONDecodeError, ValidationError) as e:
            return None, e

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            raise json.JSONDecodeError("Expected JSON object at top level", raw, 0)
        return result

    @staticmethod
    def _schema_hint(model: type[BaseModel]) -> str:
        schema = model.model_json_schema()
        return (
            "You must respond with a single JSON object matching this JSON Schema "
            "exactly. Do not include any prose or markdown. Return only raw JSON.\n\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```"
        )
