"""Typed application configuration loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VOCAB_LEARNER_",
        case_sensitive=False,
        extra="ignore",
    )

    # Provider selection
    llm_provider: Provider = "openai"

    # Provider-specific keys (each optional; the selected provider's key is required
    # at client-construction time, not at settings-load time)
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    log_level: str = "INFO"
    extraction_model: str = "gpt-4.1"
    teaching_model: str = "gpt-4.1"
    output_dir: Path = Path("./output")
    fixtures_dir: Path = Path("./fixtures")
    prompts_dir: Path = Path("./prompts")

    max_tokens: int = 4096
    request_timeout_seconds: float = 120.0


def load_settings() -> Settings:
    return Settings()
