"""Hugging Face multi-model LLMClient with 30 LLMs and SQLite token limit tracking."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from typing import TypeVar

import openai
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
)
from pydantic import BaseModel, ValidationError

from vocab_learner.config import Settings
from vocab_learner.llm.models_registry import (
    ModelInfo,
    get_model_pool,
    get_text_models,
    get_vision_models,
)
from vocab_learner.llm.protocol import LLMError, LLMValidationError
from vocab_learner.llm.token_tracker import TokenTracker
from vocab_learner.logging_setup import get_logger

T = TypeVar("T", bound=BaseModel)
log = get_logger(__name__)

UserContentPart = ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam


class HuggingFaceClient:
    """
    Hugging Face multi-model LLM client supporting 30 different models.
    All API keys are managed in .env.
    Checks token and rate limits in SQLite before every request, and automatically
    switches to an available LLM whose limit is not exceeded.
    """

    def __init__(
        self,
        settings: Settings,
        tracker: TokenTracker | None = None,
        models_pool: list[ModelInfo] | None = None,
    ) -> None:
        self._settings = settings
        self._base_url = settings.huggingface_base_url
        self._timeout = settings.request_timeout_seconds

        # Load all API keys from .env / settings
        self._keys = self._load_keys(settings)
        if not self._keys:
            raise ValueError(
                "No Hugging Face API keys found. Set HUGGINGFACE_API_KEY, HF_TOKEN, "
                "or HUGGINGFACE_API_KEYS in .env."
            )
        self._key_index = 0

        # Initialize SQLite token tracker
        self._tracker = tracker or TokenTracker(
            db_path=settings.sqlite_db_path,
            tpm_limit=settings.tpm_limit,
            rpm_limit=settings.rpm_limit,
            rpd_limit=settings.rpd_limit,
            tpd_limit=settings.daily_token_limit,
        )

        # 30 models pool
        self._models_pool = models_pool or get_model_pool()
        self._vision_models = [m.model_id for m in get_vision_models(self._models_pool)]
        self._text_models = [m.model_id for m in get_text_models(self._models_pool)]

        # Map of active OpenAI clients keyed by API key
        self._clients: dict[str, openai.OpenAI] = {
            key: openai.OpenAI(base_url=self._base_url, api_key=key, timeout=self._timeout)
            for key in self._keys
        }

    @staticmethod
    def _load_keys(settings: Settings) -> list[str]:
        """Load all Hugging Face keys from settings and .env."""
        keys: list[str] = []

        if settings.huggingface_api_key:
            keys.append(settings.huggingface_api_key.strip())
        if settings.hf_token:
            keys.append(settings.hf_token.strip())
        if settings.huggingface_api_keys:
            for k in re.split(r"[,\s]+", settings.huggingface_api_keys.strip()):
                if k:
                    keys.append(k)

        # Also search environment for HUGGINGFACE_API_KEY_1..30 or HF_TOKEN_1..30
        for i in range(1, 31):
            env_key = os.environ.get(f"HUGGINGFACE_API_KEY_{i}") or os.environ.get(f"HF_TOKEN_{i}")
            if env_key and env_key.strip():
                keys.append(env_key.strip())

        # Deduplicate preserving order
        unique_keys: list[str] = []
        for k in keys:
            if k not in unique_keys:
                unique_keys.append(k)
        return unique_keys

    def _next_key(self) -> str:
        """Rotate to the next API key in the pool."""
        key = self._keys[self._key_index % len(self._keys)]
        self._key_index += 1
        return key

    @staticmethod
    def _key_hash(key: str) -> str:
        """Create a shortened hash of the API key for ledger logging."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]

    # -- Public LLMClient Protocol Methods -----------------------------------

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
        return self._execute_with_switching(
            requested_model=model,
            system_prompt=system_prompt,
            user_content=[text_part],
            response_model=response_model,
            max_tokens=max_tokens,
            is_vision=False,
            estimated_tokens=(len(system_prompt) + len(user_prompt)) // 4 + max_tokens,
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
        return self._execute_with_switching(
            requested_model=model,
            system_prompt=system_prompt,
            user_content=[image_part, text_part],
            response_model=response_model,
            max_tokens=max_tokens,
            is_vision=True,
            # Vision token estimation: image (~1200) + text prompt + max_tokens
            estimated_tokens=1200 + (len(system_prompt) + len(user_prompt)) // 4 + max_tokens,
        )

    # -- Switching and Execution Engine ---------------------------------------

    def _execute_with_switching(
        self,
        *,
        requested_model: str,
        system_prompt: str,
        user_content: list[UserContentPart],
        response_model: type[T],
        max_tokens: int,
        is_vision: bool,
        estimated_tokens: int,
    ) -> T:
        """
        Check token limits before calling each request and switch LLM whose
        limit is not exceeded. Retries with failover if 429/quota error occurs.
        """
        candidate_models = self._vision_models if is_vision else self._text_models

        # If requested_model is an OpenAI/Anthropic placeholder or not in candidate list,
        # fallback to candidate list's primary model
        preferred = (
            requested_model
            if requested_model in candidate_models
            else (candidate_models[0] if candidate_models else requested_model)
        )

        max_attempts = min(5, len(candidate_models))
        tried_models: set[str] = set()

        for attempt in range(max_attempts):
            # Check SQLite token limit and select an LLM whose limit is not exceeded!
            remaining_candidates = [m for m in candidate_models if m not in tried_models]
            if not remaining_candidates:
                remaining_candidates = candidate_models

            selected_model, switched = self._tracker.select_available_model(
                candidate_models=remaining_candidates,
                estimated_tokens=estimated_tokens,
                preferred_model=preferred if preferred not in tried_models else None,
            )
            tried_models.add(selected_model)

            api_key = self._next_key()
            key_hash = self._key_hash(api_key)
            client = self._clients[api_key]

            log.info(
                "llm_request_dispatched",
                model=selected_model,
                requested=requested_model,
                switched=switched,
                key=key_hash,
                is_vision=is_vision,
                attempt=attempt + 1,
            )

            try:
                result, tokens_used = self._call_model(
                    client=client,
                    model=selected_model,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    response_model=response_model,
                    max_tokens=max_tokens,
                )
                # Successfully completed: record actual token usage in SQLite
                self._tracker.record_usage(
                    model_id=selected_model,
                    key_hash=key_hash,
                    prompt_tokens=tokens_used.get("prompt_tokens", estimated_tokens // 2),
                    completion_tokens=tokens_used.get("completion_tokens", estimated_tokens // 2),
                    total_tokens=tokens_used.get("total_tokens", estimated_tokens),
                )
                return result

            except openai.RateLimitError as e:
                log.warning(
                    "llm_rate_limit_exceeded_switching_llm",
                    model=selected_model,
                    error=str(e),
                    attempt=attempt + 1,
                )
                # Mark model in cooldown in SQLite so subsequent requests immediately skip it
                self._tracker.mark_cooldown(
                    model_id=selected_model,
                    cooldown_seconds=60.0,
                    error_message=str(e),
                )
                # Continue loop to switch to next LLM
                continue

            except openai.APIStatusError as e:
                # Catch 429 Too Many Requests, 402 Payment Required/no credits, or quota/limit error
                err_text = str(e).lower()
                is_quota_or_credits = (
                    e.status_code in {402, 429}
                    or "credit" in err_text
                    or "payment" in err_text
                    or "rate" in err_text
                    or "limit" in err_text
                    or "quota" in err_text
                )
                if is_quota_or_credits:
                    log.warning(
                        "llm_credits_or_rate_limit_exceeded_switching_llm",
                        model=selected_model,
                        error=str(e),
                        status_code=e.status_code,
                    )
                    # For 402 (no credits), mark longer cooldown (1 hour)
                    cooldown = 3600.0 if e.status_code == 402 or "credit" in err_text else 60.0
                    self._tracker.mark_cooldown(
                        model_id=selected_model,
                        cooldown_seconds=cooldown,
                        error_message=str(e),
                    )
                    continue

                # Other non-rate-limit API error
                log.error("llm_api_status_error", model=selected_model, error=str(e))
                raise LLMError(f"Hugging Face API error ({selected_model}): {e}") from e

            except LLMValidationError:
                # Validation retry already handled in _call_model; re-raise
                raise

            except Exception as e:
                # Network or unexpected error: mark cooldown and attempt switch
                err_str = str(e).lower()
                if (
                    "429" in err_str
                    or "402" in err_str
                    or "credit" in err_str
                    or "quota" in err_str
                    or "rate limit" in err_str
                ):
                    log.warning("llm_error_rate_limited", model=selected_model, error=str(e))
                    cooldown = 3600.0 if "402" in err_str or "credit" in err_str else 60.0
                    self._tracker.mark_cooldown(
                        model_id=selected_model,
                        cooldown_seconds=cooldown,
                        error_message=str(e),
                    )
                    continue
                log.error("llm_unexpected_error", model=selected_model, error=str(e))
                raise LLMError(f"Hugging Face unexpected error ({selected_model}): {e}") from e

        raise LLMError(
            f"All {max_attempts} attempted Hugging Face models exceeded limits, require credits, "
            f"or failed. Attempted models: {list(tried_models)}. "
            "Please check that models in models_registry.py are free, or add pre-paid credits."
        )

    def _call_model(
        self,
        *,
        client: openai.OpenAI,
        model: str,
        system_prompt: str,
        user_content: list[UserContentPart],
        response_model: type[T],
        max_tokens: int,
    ) -> tuple[T, dict[str, int]]:
        schema_hint = self._schema_hint(response_model)
        system_full = f"{system_prompt}\n\n{schema_hint}"

        raw_text, usage = self._send(
            client=client,
            model=model,
            system=system_full,
            user_content=user_content,
            max_tokens=max_tokens,
        )
        parsed, first_error = self._try_parse(raw_text, response_model)
        if parsed is not None:
            return parsed, usage

        log.warning("llm_validation_failed_first_attempt", model=model, error=str(first_error))

        repair_part: ChatCompletionContentPartTextParam = {
            "type": "text",
            "text": (
                "Your previous response could not be parsed as valid JSON "
                "matching the required schema. Error:\n\n"
                f"{first_error}\n\n"
                "Return ONLY the corrected JSON object. No prose, no markdown fences."
            ),
        }
        repair_content: list[UserContentPart] = [*user_content, repair_part]
        raw_text, repair_usage = self._send(
            client=client,
            model=model,
            system=system_full,
            user_content=repair_content,
            max_tokens=max_tokens,
        )

        # Aggregate tokens
        usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + repair_usage.get(
            "prompt_tokens", 0
        )
        usage["completion_tokens"] = usage.get("completion_tokens", 0) + repair_usage.get(
            "completion_tokens", 0
        )
        usage["total_tokens"] = usage.get("total_tokens", 0) + repair_usage.get("total_tokens", 0)

        parsed, final_error = self._try_parse(raw_text, response_model)
        if parsed is not None:
            return parsed, usage

        log.error("llm_validation_failed_final", model=model, error=str(final_error))
        raise LLMValidationError(f"LLM output failed validation after retry: {final_error}")

    def _send(
        self,
        *,
        client: openai.OpenAI,
        model: str,
        system: str,
        user_content: list[UserContentPart],
        max_tokens: int,
    ) -> tuple[str, dict[str, int]]:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        # Some open-source models work better without response_format json_object parameter
        # Try with json_object first, fallback to standard if model rejects it
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except (openai.BadRequestError, openai.APIError) as e:
            if "response_format" in str(e).lower() or "json_object" in str(e).lower():
                log.info("retrying_without_response_format", model=model)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
            else:
                raise

        choice = response.choices[0]
        text = choice.message.content or ""
        tokens_info = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0)
            if response.usage
            else len(system) // 4,
            "completion_tokens": getattr(response.usage, "completion_tokens", 0)
            if response.usage
            else len(text) // 4,
            "total_tokens": getattr(response.usage, "total_tokens", 0)
            if response.usage
            else (len(system) + len(text)) // 4,
        }
        return text.strip(), tokens_info

    def _try_parse(self, raw: str, response_model: type[T]) -> tuple[T | None, Exception | None]:
        cleaned = self._clean_json(raw)
        try:
            parsed = response_model.model_validate_json(cleaned)
            return parsed, None
        except (ValidationError, ValueError) as e:
            # Fallback: extract json object {...} from text
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                try:
                    parsed = response_model.model_validate_json(match.group(1))
                    return parsed, None
                except Exception:
                    pass
            return None, e

    @staticmethod
    def _clean_json(raw: str) -> str:
        """Strip markdown fences (```json ... ```) and surrounding whitespace."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _schema_hint(response_model: type[BaseModel]) -> str:
        schema = response_model.model_json_schema()
        return (
            "Output format requirement: you must respond with a JSON object conforming "
            f"to this JSON schema:\n{json.dumps(schema, indent=2)}\n"
            "Return ONLY the JSON object. Do not wrap it in markdown fences (```json)."
        )
