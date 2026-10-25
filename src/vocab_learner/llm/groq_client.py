"""Groq multi-model LLMClient with SQLite token limit tracking and automatic switching."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import time
from typing import TypeVar

import openai
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
)
from pydantic import BaseModel, ValidationError

from vocab_learner.config import Settings
from vocab_learner.llm.groq_models import (
    GroqModelInfo,
    get_groq_model_pool,
    get_groq_text_models,
    get_groq_vision_models,
)
from vocab_learner.llm.protocol import LLMError, LLMValidationError
from vocab_learner.llm.token_tracker import TokenTracker
from vocab_learner.logging_setup import get_logger

T = TypeVar("T", bound=BaseModel)
log = get_logger(__name__)

UserContentPart = ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam


class GroqClient:
    """
    Groq multi-model LLM client supporting vision and high-speed text models.
    All API keys are managed in .env.
    Checks token and rate limits in SQLite before every request, and automatically
    switches to an available Groq model whose limit is not exceeded.
    """

    def __init__(
        self,
        settings: Settings,
        tracker: TokenTracker | None = None,
        models_pool: list[GroqModelInfo] | None = None,
    ) -> None:
        self._settings = settings
        self._base_url = settings.groq_base_url
        self._timeout = settings.request_timeout_seconds

        # Load all API keys from .env / settings
        self._keys = self._load_keys(settings)
        if not self._keys:
            raise ValueError("No Groq API keys found. Set GROQ_API_KEY or GROQ_API_KEYS in .env.")
        self._key_index = 0

        # Initialize SQLite token tracker
        self._tracker = tracker or TokenTracker(
            db_path=settings.sqlite_db_path,
            tpm_limit=settings.tpm_limit,
            rpm_limit=settings.rpm_limit,
            rpd_limit=settings.rpd_limit,
            tpd_limit=settings.daily_token_limit,
        )

        # Map of active OpenAI clients (max_retries=0 enables instant model switching)
        self._clients: dict[str, openai.OpenAI] = {
            key: openai.OpenAI(
                base_url=self._base_url,
                api_key=key,
                timeout=self._timeout,
                max_retries=0,
            )
            for key in self._keys
        }

        # Groq models pool - dynamically filter by active models on the Groq account
        self._models_pool = models_pool or get_groq_model_pool()
        if models_pool is None:
            primary_client = next(iter(self._clients.values()))
            try:
                active_ids = {m.id for m in primary_client.models.list().data}
                filtered = [m for m in self._models_pool if m.model_id in active_ids]
                if filtered:
                    self._models_pool = filtered
            except Exception as e:
                log.debug("groq_models_list_filter_skipped", error=str(e))

        self._vision_models = [m.model_id for m in get_groq_vision_models(self._models_pool)]
        self._text_models = [m.model_id for m in get_groq_text_models(self._models_pool)]

        # Register per-model limits in SQLite and unblock any false-alarm cooldowns
        for m in self._models_pool:
            self._tracker.register_model_limits(
                model_id=m.model_id,
                rpm_limit=m.rpm_limit,
                rpd_limit=m.rpd_limit,
                tpm_limit=m.tpm_limit,
                tpd_limit=m.tpd_limit,
            )
            self._tracker.clear_cooldown(m.model_id)

    @staticmethod
    def _load_keys(settings: Settings) -> list[str]:
        """Load all Groq keys from settings and .env."""
        keys: list[str] = []

        if settings.groq_api_key:
            keys.append(settings.groq_api_key.strip())
        if settings.groq_api_keys:
            for k in re.split(r"[,\s]+", settings.groq_api_keys.strip()):
                if k:
                    keys.append(k)

        # Also search environment for GROQ_API_KEY_1..30
        for i in range(1, 31):
            env_key = os.environ.get(f"GROQ_API_KEY_{i}")
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
        # Realistic token estimation: prompt tokens + expected completion
        # Structured cards are typically ~300-600 tokens
        expected_output = min(max_tokens, 600)
        estimated = (len(system_prompt) + len(user_prompt)) // 4 + expected_output
        return self._execute_with_switching(
            requested_model=model,
            system_prompt=system_prompt,
            user_content=[text_part],
            response_model=response_model,
            max_tokens=max_tokens,
            is_vision=False,
            estimated_tokens=estimated,
        )

    @staticmethod
    def _optimize_image_for_groq(image_bytes: bytes, max_dim: int = 1600) -> tuple[bytes, str]:
        """
        Resize and compress large images to prevent HTTP 413 (Payload Too Large).
        Ensures max dimension <= max_dim and payload size stays well below Groq API limits.
        """
        try:
            import io

            from PIL import Image

            img: Image.Image = Image.open(io.BytesIO(image_bytes))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            if max(img.size) > max_dim or len(image_bytes) > 1_500_000:
                if max(img.size) > max_dim:
                    ratio = max_dim / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return buf.getvalue(), "image/jpeg"
        except Exception as e:
            log.warning("image_optimization_skipped", error=str(e))
        return image_bytes, "image/jpeg"

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
        optimized_bytes, media_type = self._optimize_image_for_groq(image_bytes)
        b64 = base64.standard_b64encode(optimized_bytes).decode("ascii")
        data_url = f"data:{media_type};base64,{b64}"

        image_part: ChatCompletionContentPartImageParam = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
        text_part: ChatCompletionContentPartTextParam = {
            "type": "text",
            "text": user_prompt,
        }
        # Groq enforces 1000 OTPM on qwen vision; clamp max_tokens <= 600 to avoid OTPM 429
        clamped_max_tokens = min(max_tokens, 600)
        estimated = 1200 + (len(system_prompt) + len(user_prompt)) // 4 + clamped_max_tokens

        return self._execute_with_switching(
            requested_model=model,
            system_prompt=system_prompt,
            user_content=[image_part, text_part],
            response_model=response_model,
            max_tokens=clamped_max_tokens,
            is_vision=True,
            estimated_tokens=estimated,
        )

    # -- Jitter and Delay Helpers ---------------------------------------------

    @staticmethod
    def _extract_retry_delay(err: Exception) -> float | None:
        """Extract retry wait time in seconds from RateLimitError headers or message."""
        response = getattr(err, "response", None)
        if response is not None and hasattr(response, "headers"):
            retry_header = response.headers.get("retry-after")
            if retry_header:
                try:
                    return float(retry_header)
                except ValueError:
                    pass

        msg = str(err).lower()
        match_sec = re.search(r"try again in\s+([\d\.]+)\s*s", msg)
        if match_sec:
            try:
                return float(match_sec.group(1))
            except ValueError:
                pass

        match_min = re.search(r"try again in\s+(\d+)m([\d\.]+)s", msg)
        if match_min:
            try:
                return float(match_min.group(1)) * 60 + float(match_min.group(2))
            except ValueError:
                pass

        return None

    @staticmethod
    def _calculate_jittered_delay(
        base_wait: float,
        min_noise: float = 0.25,
        max_noise: float = 1.75,
    ) -> float:
        """
        Add a small amount of random delay (noise / jitter) to the wait time so multiple
        concurrent requests don't all retry at the exact same millisecond.
        """
        proportional_noise = max(0.0, base_wait) * random.uniform(0.05, 0.15)
        random_noise = random.uniform(min_noise, max_noise)
        return round(base_wait + proportional_noise + random_noise, 3)

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
        Check token limits before calling each request and switch Groq model whose
        limit is not exceeded. Retries with failover if 429/quota error occurs.
        """
        candidate_models = self._vision_models if is_vision else self._text_models

        preferred = (
            requested_model
            if requested_model in candidate_models
            else (candidate_models[0] if candidate_models else requested_model)
        )

        max_attempts = 4 if len(candidate_models) == 1 else min(5, len(candidate_models))
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

            # If the selected model is temporarily over limits, wait for window to free tokens
            is_ok, limit_reason = self._tracker.is_within_limit(
                selected_model,
                estimated_tokens=estimated_tokens,
            )
            if not is_ok:
                wait_sec = self._tracker.get_seconds_until_available(
                    selected_model,
                    estimated_tokens=estimated_tokens,
                )
                max_pacing_wait = 65.0 if len(candidate_models) == 1 else 30.0
                if 0.0 < wait_sec <= max_pacing_wait:
                    jittered_wait = self._calculate_jittered_delay(wait_sec)
                    log.info(
                        "groq_tpm_pacing_wait_for_window_reset",
                        model=selected_model,
                        wait_seconds=jittered_wait,
                        base_wait=round(wait_sec, 2),
                        reason=limit_reason,
                    )
                    time.sleep(jittered_wait)

            log.info(
                "groq_request_dispatched",
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
                err_msg = str(e)
                log.warning(
                    "groq_rate_limit_exceeded_switching_model",
                    model=selected_model,
                    error=err_msg,
                    attempt=attempt + 1,
                )

                retry_delay = self._extract_retry_delay(e)
                # Only wait and retry the same model if there are no other candidate models
                # available (e.g. vision tasks where only 1 model exists).
                # If other candidates exist, immediately mark cooldown and failover!
                can_wait_and_retry = len(candidate_models) == 1 and attempt < (max_attempts - 1)
                if can_wait_and_retry:
                    fallback_delay = self._tracker.get_seconds_until_available(
                        selected_model,
                        estimated_tokens=estimated_tokens,
                    )
                    base_delay = (
                        retry_delay
                        if retry_delay is not None
                        else (fallback_delay if fallback_delay > 0.0 else (2.0 * (attempt + 1)))
                    )
                    jittered_wait = self._calculate_jittered_delay(base_delay)
                    log.info(
                        "groq_waiting_with_jitter_before_retry",
                        model=selected_model,
                        base_wait=base_delay,
                        total_wait=jittered_wait,
                        attempt=attempt + 1,
                    )
                    time.sleep(jittered_wait)
                    # Allow retrying this model
                    tried_models.discard(selected_model)
                    continue

                # Add a small random delay (noise) before switching so concurrent
                # requests don't all hit the next model simultaneously
                stagger_noise = random.uniform(0.15, 0.75)
                time.sleep(stagger_noise)

                # Mark model in cooldown in SQLite so subsequent requests immediately skip it
                self._tracker.mark_cooldown(
                    model_id=selected_model,
                    cooldown_seconds=60.0,
                    error_message=err_msg,
                )
                continue

            except openai.APIStatusError as e:
                err_text = str(e).lower()
                is_rate_or_quota = (
                    e.status_code in {429, 402}
                    or "rate" in err_text
                    or "limit" in err_text
                    or "quota" in err_text
                )
                is_unavailable = e.status_code == 404 or (
                    e.status_code == 400
                    and (
                        "decommissioned" in err_text
                        or "not exist" in err_text
                        or "json_validate_failed" in err_text
                        or "failed to validate json" in err_text
                        or "failed to generate json" in err_text
                    )
                )
                if is_rate_or_quota or is_unavailable:
                    is_permanent = (
                        "decommissioned" in err_text
                        or "not exist" in err_text
                        or "json_validate_failed" in err_text
                        or "failed to validate json" in err_text
                        or "failed to generate json" in err_text
                    )
                    cooldown = 86400.0 if is_permanent else 60.0
                    log.warning(
                        "groq_status_error_switching_model",
                        model=selected_model,
                        error=str(e),
                        status_code=e.status_code,
                    )
                    if is_rate_or_quota:
                        time.sleep(random.uniform(0.15, 0.75))
                    self._tracker.mark_cooldown(
                        model_id=selected_model,
                        cooldown_seconds=cooldown,
                        error_message=str(e),
                    )
                    continue

                log.error("groq_api_status_error", model=selected_model, error=str(e))
                raise LLMError(f"Groq API error ({selected_model}): {e}") from e

            except LLMValidationError as e:
                log.warning(
                    "groq_validation_error_switching_model",
                    model=selected_model,
                    error=str(e),
                )
                self._tracker.mark_cooldown(
                    model_id=selected_model,
                    cooldown_seconds=86400.0,
                    error_message=str(e),
                )
                continue

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                    log.warning("groq_error_rate_limited", model=selected_model, error=str(e))
                    self._tracker.mark_cooldown(
                        model_id=selected_model,
                        cooldown_seconds=60.0,
                        error_message=str(e),
                    )
                    continue
                log.error("groq_unexpected_error", model=selected_model, error=str(e))
                raise LLMError(f"Groq unexpected error ({selected_model}): {e}") from e

        raise LLMError(
            f"All {max_attempts} attempted Groq models exceeded limits or failed. "
            f"Attempted models: {list(tried_models)}"
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

        log.warning("groq_validation_failed_first_attempt", model=model, error=str(first_error))

        repair_part: ChatCompletionContentPartTextParam = {
            "type": "text",
            "text": (
                "Your previous JSON response failed validation:\n"
                f"Previous output: {raw_text}\n\n"
                f"Validation error: {first_error}\n\n"
                "Return ONLY the corrected JSON object matching the required schema. "
                "No markdown fences."
            ),
        }
        # Keep only text parts from user_content to avoid re-uploading heavy images against ITPM
        text_only_parts: list[UserContentPart] = [
            p for p in user_content if p.get("type") == "text"
        ]
        repair_content: list[UserContentPart] = [*text_only_parts, repair_part]
        raw_text, repair_usage = self._send(
            client=client,
            model=model,
            system=system_full,
            user_content=repair_content,
            max_tokens=max_tokens,
        )

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

        log.error("groq_validation_failed_final", model=model, error=str(final_error))
        raise LLMValidationError(f"Groq LLM output failed validation after retry: {final_error}")

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

        # Clamp max_tokens to fit within Groq model OTPM limits (qwen: 1000 OTPM limit)
        effective_max_tokens = max_tokens
        if "qwen" in model.lower():
            effective_max_tokens = min(max_tokens, 800)
        else:
            effective_max_tokens = min(max_tokens, 2048)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=effective_max_tokens,
                response_format={"type": "json_object"},
            )
        except openai.RateLimitError as e:
            err_msg = str(e).lower()
            is_otpm = (
                "output tokens per minute" in err_msg
                or "otpm" in err_msg
                or "reduce max_tokens" in err_msg
            )
            if is_otpm:
                log.info("groq_clamping_max_tokens_on_otpm_limit", model=model)
                retry_wait = self._extract_retry_delay(e)
                wait_time = (
                    self._calculate_jittered_delay(retry_wait)
                    if retry_wait is not None
                    else random.uniform(1.0, 3.0)
                )
                time.sleep(wait_time)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=400,
                    response_format={"type": "json_object"},
                )
            else:
                raise
        except (openai.BadRequestError, openai.APIError) as e:
            err_str = str(e).lower()
            if (
                "response_format" in err_str
                or "json_object" in err_str
                or "json_validate_failed" in err_str
                or "failed to validate json" in err_str
                or "failed to generate json" in err_str
            ):
                log.info("retrying_without_response_format", model=model, error=str(e))
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=effective_max_tokens,
                    )
                except Exception as retry_err:
                    log.warning(
                        "retry_without_response_format_failed",
                        model=model,
                        error=str(retry_err),
                    )
                    raise e from retry_err
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
        # Attempt direct JSON deserialization and metadata cleanup
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                # Strip schema metadata keys that some LLMs inadvertently copy
                for meta_key in ("required", "$defs", "$schema"):
                    if meta_key in data and meta_key not in response_model.model_fields:
                        data.pop(meta_key, None)
                parsed = response_model.model_validate(data)
                return parsed, None
        except Exception:
            pass

        try:
            parsed = response_model.model_validate_json(cleaned)
            return parsed, None
        except (ValidationError, ValueError) as e:
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    if isinstance(data, dict):
                        for meta_key in ("required", "$defs", "$schema"):
                            if meta_key in data and meta_key not in response_model.model_fields:
                                data.pop(meta_key, None)
                        parsed = response_model.model_validate(data)
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
