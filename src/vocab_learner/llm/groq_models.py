"""Groq models registry with capabilities and rate limit specifications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroqModelInfo:
    """Metadata for a model in the Groq inference pool."""

    model_id: str
    supports_vision: bool
    context_length: int = 131072
    rpm_limit: int = 30  # requests per minute
    rpd_limit: int = 1000  # requests per day (daily equivalent of RPM)
    tpm_limit: int = 30000  # tokens per minute
    tpd_limit: int = 500000  # tokens per day (daily equivalent of TPM)

    # Backward compatibility properties
    @property
    def default_rpm_limit(self) -> int:
        return self.rpm_limit

    @property
    def default_rpd_limit(self) -> int:
        return self.rpd_limit

    @property
    def default_tpm_limit(self) -> int:
        return self.tpm_limit

    @property
    def default_daily_limit(self) -> int:
        return self.tpd_limit


# Curated catalog of Groq models (vision and text) with per-model limits
GROQ_MODELS_POOL: list[GroqModelInfo] = [
    # --- Vision & Multimodal Model ---
    GroqModelInfo(
        model_id="qwen/qwen3.8-27b",
        supports_vision=True,
        context_length=131072,
        rpm_limit=30,
        rpd_limit=1000,
        tpm_limit=7000,
        tpd_limit=200000,
    ),
    # --- High-Capability Text Models ---
    GroqModelInfo(
        model_id="openai/gpt-oss-120b",
        supports_vision=False,
        context_length=131072,
        rpm_limit=30,
        rpd_limit=1000,
        tpm_limit=8000,
        tpd_limit=200000,
    ),
    GroqModelInfo(
        model_id="openai/gpt-oss-20b",
        supports_vision=False,
        context_length=131072,
        rpm_limit=30,
        rpd_limit=14400,
        tpm_limit=8000,
        tpd_limit=500000,
    ),
    GroqModelInfo(
        model_id="allam-2-7b",
        supports_vision=False,
        context_length=8192,
        rpm_limit=30,
        rpd_limit=1000,
        tpm_limit=8000,
        tpd_limit=200000,
    ),
    # --- Compound / Agentic Models ---
    GroqModelInfo(
        model_id="groq/compound",
        supports_vision=False,
        context_length=131072,
        rpm_limit=30,
        rpd_limit=250,
        tpm_limit=70000,
        tpd_limit=500000,
    ),
    GroqModelInfo(
        model_id="groq/compound-mini",
        supports_vision=False,
        context_length=131072,
        rpm_limit=30,
        rpd_limit=250,
        tpm_limit=70000,
        tpd_limit=500000,
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
