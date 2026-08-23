"""OpenAI implementation of LLMClient."""

from __future__ import annotations

import base64
import json
from typing import Any, TypeVar

import openai
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
)
from pydantic import BaseModel, ValidationError

from vocab_learner.llm.protocol import LLMError, LLMValidationError
from vocab_learner.logging_setup import get_logger

T = TypeVar("T", bound=BaseModel)

log = get_logger(__name__)

UserContentPart = ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam


class OpenAIClient:
    """
    OpenAI-backed LLMClient. Uses Chat Completions with JSON mode + a schema hint
    in the system prompt, then Pydantic-validates. Retries once on validation failure.
    """

    def __init__(self, api_key: str, timeout_seconds: float = 120.0) -> None:
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout_seconds)

    def complete_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_tokens: int = 4096,
    ) -> T:
        text_part: ChatCompletionContentPartTextParam = {
            "type": "text",
            "text": user_prompt,
        }
        return self._call(
            model=model,
            system_prompt=system_prompt,
            user_content=[text_part],
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
        data_url = f"data:{image_media_type};base64,{b64}"

        image_part: ChatCompletionContentPartImageParam = {
            "type": "image_url",
            "image_url": {"url": data_url, "detail": "high"},
        }
        text_part: ChatCompletionContentPartTextParam = {
            "type": "text",
            "text": user_prompt,
        }
        return self._call(
            model=model,
            system_prompt=system_prompt,
            user_content=[image_part, text_part],
            response_model=response_model,
            max_tokens=max_tokens,
        )

    # -- internals -----------------------------------------------------------

    def _call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: list[UserContentPart],
        response_model: type[T],
        max_tokens: int,
    ) -> T:
        schema_hint = self._schema_hint(response_model)
        system_full = f"{system_prompt}\n\n{schema_hint}"

        raw_text = self._send(
            model=model,
            system=system_full,
            user_content=user_content,
            max_tokens=max_tokens,
        )
        parsed, first_error = self._try_parse(raw_text, response_model)
        if parsed is not None:
            return parsed

        log.warning("llm_validation_failed_first_attempt", error=str(first_error))

        repair_part: ChatCompletionContentPartTextParam = {
            "type": "text",
            "text": (
                "Your previous response could not be parsed as valid JSON "
                "matching the required schema. Error:\n\n"
                f"{first_error}\n\n"
                "Return ONLY the corrected JSON. No prose, no code fences."
            ),
        }
        repair_content: list[UserContentPart] = [*user_content, repair_part]
        raw_text = self._send(
            model=model,
            system=system_full,
            user_content=repair_content,
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
        user_content: list[UserContentPart],
        max_tokens: int,
    ) -> str:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except openai.APIError as e:
            log.error("openai_api_error", error=str(e))
            raise LLMError(f"OpenAI API error: {e}") from e

        choice = response.choices[0]
        text = choice.message.content or ""
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
