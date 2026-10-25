"""Factory: return the LLMClient implementation for the configured provider."""

from __future__ import annotations

from vocab_learner.config import Settings
from vocab_learner.llm.anthropic_client import AnthropicClient
from vocab_learner.llm.groq_client import GroqClient
from vocab_learner.llm.huggingface_client import HuggingFaceClient
from vocab_learner.llm.openai_client import OpenAIClient
from vocab_learner.llm.protocol import LLMClient


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "groq":
        keys = settings.get_groq_keys()
        if not keys:
            raise ValueError(
                "LLM provider is 'groq' but no Groq API keys are set. "
                "Set GROQ_API_KEY or GROQ_API_KEYS in .env."
            )
        return GroqClient(settings=settings)
    if settings.llm_provider == "huggingface":
        keys = settings.get_huggingface_keys()
        if not keys:
            raise ValueError(
                "LLM provider is 'huggingface' but no Hugging Face API keys are set. "
                "Set HUGGINGFACE_API_KEY, HF_TOKEN, or HUGGINGFACE_API_KEYS in .env."
            )
        return HuggingFaceClient(settings=settings)
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("LLM provider is 'anthropic' but ANTHROPIC_API_KEY is not set.")
        return AnthropicClient(
            api_key=settings.anthropic_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("LLM provider is 'openai' but OPENAI_API_KEY is not set.")
        return OpenAIClient(
            api_key=settings.openai_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
    # mypy: Literal["anthropic", "openai", "huggingface", "groq"] is exhausted; this is unreachable
    raise ValueError(f"Unknown provider: {settings.llm_provider}")
