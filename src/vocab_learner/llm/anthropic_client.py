"""Anthropic implementation of LLMClient."""

from __future__ import annotations

import base64
import json
from typing import Any, TypeVar

import anthropic
from anthropic.types import (
    ImageBlockParam,
    Message,
    TextBlock,
    TextBlockParam,
)
from pydantic import BaseModel, ValidationError

from vocab_learner.llm.protocol import LLMError, LLMValidationError
from vocab_learner.logging_setup import get_logger

T = TypeVar("T", bound=BaseModel)

log = get_logger(__name__)


class AnthropicClient:
    """
    Thin wrapper around the Anthropic SDK that:
      - forces JSON output via a schema hint appended to the system prompt
      - parses + validates against a Pydantic model
      - retries once on validation failure, feeding the error back to the model
    """

    def __init__(self, api_key: str, timeout_seconds: float = 120.0) -> None:
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)

    def complete_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_tokens: int = 4096,
    ) -> T:
        text_block: TextBlockParam = {"type": "text", "text": user_prompt}
        return self._call(
            model=model,
            system_prompt=system_prompt,
            user_content=[text_block],
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
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        image_block: ImageBlockParam = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,  # type: ignore[typeddict-item]
                "data": b64,
            },
        }
        text_block: TextBlockParam = {"type": "text", "text": user_prompt}
        content: list[TextBlockParam | ImageBlockParam] = [image_block, text_block]

        return self._call(
            model=model,
            system_prompt=system_prompt,
            user_content=content,
            response_model=response_model,
            max_tokens=max_tokens,
        )

    # -- internals -----------------------------------------------------------

    def _call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: list[TextBlockParam | ImageBlockParam],
        response_model: type[T],
        max_tokens: int,
    ) -> T:
        schema_hint = self._schema_hint(response_model)
        system_full = f"{system_prompt}\n\n{schema_hint}"

        raw_text = self._send(
            model=model,
            system=system_full,
            content=user_content,
            max_tokens=max_tokens,
        )
        parsed, first_error = self._try_parse(raw_text, response_model)
        if parsed is not None:
            return parsed

        log.warning("llm_validation_failed_first_attempt", error=str(first_error))

        repair_block: TextBlockParam = {
            "type": "text",
            "text": (
                "Your previous response could not be parsed as valid JSON "
                "matching the required schema. Error:\n\n"
                f"{first_error}\n\n"
                "Return ONLY the corrected JSON. No prose, no code fences."
            ),
        }
        repair_content: list[TextBlockParam | ImageBlockParam] = [
            *user_content,
            repair_block,
        ]
        raw_text = self._send(
            model=model,
            system=system_full,
            content=repair_content,
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
        content: list[TextBlockParam | ImageBlockParam],
        max_tokens: int,
    ) -> str:
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.APIError as e:
            log.error("anthropic_api_error", error=str(e))
            raise LLMError(f"Anthropic API error: {e}") from e

        text = self._extract_text(response)
        log.debug("llm_raw_response", chars=len(text))
        return text

    @staticmethod
    def _extract_text(response: Message) -> str:
        parts: list[str] = [
            block.text for block in response.content if isinstance(block, TextBlock)
        ]
        return "".join(parts).strip()

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
            "exactly. Do not include any prose, markdown, or code fences. "
            "Return only raw JSON.\n\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```"
        )

    def _try_parse(self, raw: str, response_model: type[T]) -> tuple[T | None, Exception | None]:
        try:
            data = self._parse_json(raw)
            parsed: T = response_model.model_validate(data)
            return parsed, None
        except (json.JSONDecodeError, ValidationError) as e:
            return None, e
