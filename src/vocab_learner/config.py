"""Typed application configuration loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VOCAB_LEARNER_",
        case_sensitive=False,
        extra="ignore",
    )

    # Anthropic key uses its own env var name, no prefix
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")

    log_level: str = "INFO"
    extraction_model: str = "claude-sonnet-5"
    teaching_model: str = "claude-sonnet-5"
    output_dir: Path = Path("./output")
    fixtures_dir: Path = Path("./fixtures")
    prompts_dir: Path = Path("./prompts")

    # LLM defaults
    max_tokens: int = 4096
    request_timeout_seconds: float = 120.0


def load_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
