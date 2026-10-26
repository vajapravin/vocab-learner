"""Factory: return the LLMClient implementation for the configured provider."""

from __future__ import annotations

from vocab_learner.config import Settings
from vocab_learner.llm.anthropic_client import AnthropicClient
from vocab_learner.llm.gemini_client import GeminiClient
from vocab_learner.llm.local_client import LocalLlamaClient
from vocab_learner.llm.openai_client import OpenAIClient
from vocab_learner.llm.protocol import LLMClient


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("LLM provider is 'anthropic' but ANTHROPIC_API_KEY is not set.")
        return AnthropicClient(
            api_key=settings.anthropic_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("LLM provider is 'gemini' but GEMINI_API_KEY is not set.")
        return GeminiClient(
            api_key=settings.gemini_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("LLM provider is 'openai' but OPENAI_API_KEY is not set.")
        return OpenAIClient(
            api_key=settings.openai_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
    if settings.llm_provider == "local":
        return LocalLlamaClient(
            model_path=settings.local_model_path,
            n_gpu_layers=settings.local_gpu_layers,
            n_ctx=settings.local_context_length,
            tesseract_cmd=settings.tesseract_cmd,
        )
    # mypy: Literal["anthropic", "openai", "local", "gemini"] is exhausted; this is unreachable
    raise ValueError(f"Unknown provider: {settings.llm_provider}")
