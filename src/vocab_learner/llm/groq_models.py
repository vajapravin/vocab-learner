"""Groq models registry with capabilities and rate limit specifications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroqModelInfo:
    """Metadata for a model in the Groq inference pool."""

    model_id: str
    supports_vision: bool
    context_length: int = 131072
    default_tpm_limit: int = 30000
    default_rpm_limit: int = 30
    default_daily_limit: int = 500000


# Curated catalog of Groq models (vision and text)
GROQ_MODELS_POOL: list[GroqModelInfo] = [
    # --- Vision Models (Multimodal: image + text input) ---
    GroqModelInfo(
        model_id="llama-3.2-11b-vision-preview",
        supports_vision=True,
        context_length=131072,
        default_tpm_limit=7000,
        default_rpm_limit=30,
    ),
    GroqModelInfo(
        model_id="llama-3.2-90b-vision-preview",
        supports_vision=True,
        context_length=131072,
        default_tpm_limit=7000,
        default_rpm_limit=30,
    ),
    GroqModelInfo(
        model_id="qwen/qwen3.8-27b",
        supports_vision=True,
        context_length=131072,
        default_tpm_limit=30000,
        default_rpm_limit=30,
    ),
    # --- Text Models ---
    GroqModelInfo(
        model_id="llama-3.3-70b-versatile",
        supports_vision=False,
        context_length=131072,
        default_tpm_limit=30000,
        default_rpm_limit=30,
    ),
    GroqModelInfo(
        model_id="llama-3.1-8b-instant",
        supports_vision=False,
        context_length=131072,
        default_tpm_limit=30000,
        default_rpm_limit=30,
    ),
    GroqModelInfo(
        model_id="deepseek-r1-distill-llama-70b",
        supports_vision=False,
        context_length=131072,
        default_tpm_limit=30000,
        default_rpm_limit=30,
    ),
    GroqModelInfo(
        model_id="mixtral-8x7b-32768",
        supports_vision=False,
        context_length=32768,
        default_tpm_limit=30000,
        default_rpm_limit=30,
    ),
    GroqModelInfo(
        model_id="gemma2-9b-it",
        supports_vision=False,
        context_length=8192,
        default_tpm_limit=15000,
        default_rpm_limit=30,
    ),
    GroqModelInfo(
        model_id="openai/gpt-oss-120b",
        supports_vision=False,
        context_length=131072,
        default_tpm_limit=30000,
        default_rpm_limit=30,
    ),
    GroqModelInfo(
        model_id="openai/gpt-oss-20b",
        supports_vision=False,
        context_length=131072,
        default_tpm_limit=30000,
        default_rpm_limit=30,
    ),
]

_GROQ_BY_ID = {m.model_id: m for m in GROQ_MODELS_POOL}


def get_groq_model_pool(custom_models: list[str] | None = None) -> list[GroqModelInfo]:
    """Return the configured pool of Groq models, optionally filtered or extended."""
    if not custom_models:
        return list(GROQ_MODELS_POOL)

    result: list[GroqModelInfo] = []
    for model_id in custom_models:
        clean_id = model_id.strip()
        if not clean_id:
            continue
        if clean_id in _GROQ_BY_ID:
            result.append(_GROQ_BY_ID[clean_id])
        else:
            is_vision = "vision" in clean_id.lower() or "-vl" in clean_id.lower()
            result.append(GroqModelInfo(model_id=clean_id, supports_vision=is_vision))
    return result if result else list(GROQ_MODELS_POOL)


def get_groq_vision_models(pool: list[GroqModelInfo] | None = None) -> list[GroqModelInfo]:
    """Return Groq models capable of vision (image) input."""
    source = pool if pool is not None else GROQ_MODELS_POOL
    return [m for m in source if m.supports_vision]


def get_groq_text_models(pool: list[GroqModelInfo] | None = None) -> list[GroqModelInfo]:
    """Return Groq models capable of text tasks."""
    return list(pool if pool is not None else GROQ_MODELS_POOL)
