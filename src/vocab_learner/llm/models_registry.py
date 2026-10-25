"""30 Hugging Face models registry with capabilities and rate limit specifications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for an LLM in the Hugging Face router pool."""

    model_id: str
    supports_vision: bool
    context_length: int = 131072
    default_tpm_limit: int = 30000
    default_rpm_limit: int = 30
    default_daily_limit: int = 500000


# Exactly 30 distinct live models available on Hugging Face router:
# 15 Vision-capable models (modalities: ['text', 'image'])
# 15 Text-specialized models (modalities: ['text'])
HUGGINGFACE_MODELS_POOL: list[ModelInfo] = [
    # --- Vision Models (supports both image extraction and text) ---
    ModelInfo(
        model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        supports_vision=True,
        context_length=131072,
    ),
    ModelInfo(
        model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        supports_vision=True,
        context_length=1048576,
    ),
    ModelInfo(
        model_id="Qwen/Qwen2.5-VL-72B-Instruct",
        supports_vision=True,
        context_length=131072,
    ),
    ModelInfo(
        model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
        supports_vision=True,
        context_length=131072,
    ),
    ModelInfo(
        model_id="Qwen/Qwen3.5-9B",
        supports_vision=True,
        context_length=131072,
    ),
    ModelInfo(
        model_id="Qwen/Qwen3.5-27B",
        supports_vision=True,
        context_length=131072,
    ),
    ModelInfo(
        model_id="deepseek-ai/DeepSeek-V4.1-Flash",
        supports_vision=True,
        context_length=131072,
    ),
    ModelInfo(
        model_id="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        supports_vision=True,
        context_length=131072,
    ),
    ModelInfo(
        model_id="google/gemma-3-4b-it",
        supports_vision=True,
        context_length=131072,
    ),
    ModelInfo(
        model_id="google/gemma-3-12b-it",
        supports_vision=True,
        context_length=131072,
    ),
    ModelInfo(
        model_id="google/gemma-3-27b-it",
        supports_vision=True,
        context_length=131072,
    ),
    ModelInfo(
        model_id="CohereLabs/aya-vision-32b",
        supports_vision=True,
        context_length=131072,
    ),
    ModelInfo(
        model_id="CohereLabs/command-a-vision-07-2025",
        supports_vision=True,
        context_length=131072,
    ),
    ModelInfo(
        model_id="zai-org/GLM-4.6V-Flash",
        supports_vision=True,
        context_length=131072,
    ),
    ModelInfo(
        model_id="zai-org/GLM-5.3-Flash",
        supports_vision=True,
        context_length=131072,
    ),
    # --- Text Models ---
    ModelInfo(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="NousResearch/Hermes-3-Llama-3.1-70B",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="Qwen/Qwen3-8B",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="Qwen/Qwen3-14B",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="deepseek-ai/DeepSeek-V3",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="deepseek-ai/DeepSeek-R1",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="microsoft/phi-4",
        supports_vision=False,
        context_length=131072,
    ),
    ModelInfo(
        model_id="CohereLabs/c4ai-command-r-08-2024",
        supports_vision=False,
        context_length=131072,
    ),
]

_POOL_BY_ID = {m.model_id: m for m in HUGGINGFACE_MODELS_POOL}


def get_model_pool(custom_models: list[str] | None = None) -> list[ModelInfo]:
    """Return the configured pool of models, optionally filtered or extended."""
    if not custom_models:
        return list(HUGGINGFACE_MODELS_POOL)

    result: list[ModelInfo] = []
    for model_id in custom_models:
        clean_id = model_id.strip()
        if not clean_id:
            continue
        if clean_id in _POOL_BY_ID:
            result.append(_POOL_BY_ID[clean_id])
        else:
            # Custom model: default to text, supports vision if 'vision' or 'vl' in name
            is_vision = "vision" in clean_id.lower() or "-vl" in clean_id.lower()
            result.append(ModelInfo(model_id=clean_id, supports_vision=is_vision))
    return result if result else list(HUGGINGFACE_MODELS_POOL)


def get_vision_models(pool: list[ModelInfo] | None = None) -> list[ModelInfo]:
    """Return only models capable of vision (image) input."""
    source = pool if pool is not None else HUGGINGFACE_MODELS_POOL
    return [m for m in source if m.supports_vision]


def get_text_models(pool: list[ModelInfo] | None = None) -> list[ModelInfo]:
    """Return all models capable of text tasks (all models in pool)."""
    return list(pool if pool is not None else HUGGINGFACE_MODELS_POOL)
