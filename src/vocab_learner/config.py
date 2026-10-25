"""Typed application configuration loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai", "huggingface", "groq"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VOCAB_LEARNER_",
        case_sensitive=False,
        extra="ignore",
    )

    # Provider selection
    llm_provider: Provider = "groq"

    # Provider-specific keys (each optional; the selected provider's key is required
    # at client-construction time, not at settings-load time)
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    huggingface_api_key: str | None = Field(default=None, alias="HUGGINGFACE_API_KEY")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    huggingface_api_keys: str | None = Field(default=None, alias="HUGGINGFACE_API_KEYS")
    huggingface_base_url: str = Field(
        default="https://router.huggingface.co/v1", alias="HUGGINGFACE_BASE_URL"
    )
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_api_keys: str | None = Field(default=None, alias="GROQ_API_KEYS")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")

    # SQLite minimal token tracking and rate limiting
    sqlite_db_path: Path = Path("./output/llm_usage.db")
    tpm_limit: int = 30000
    rpm_limit: int = 30
    daily_token_limit: int = 500000

    log_level: str = "INFO"
    extraction_model: str = "gpt-4.1"
    teaching_model: str = "gpt-4.1"
    output_dir: Path = Path("./output")
    inbox_dir: Path = Path("./inbox")
    fixtures_dir: Path = Path("./fixtures")
    prompts_dir: Path = Path("./prompts")

    max_tokens: int = 4096
    request_timeout_seconds: float = 120.0

    # Email (all optional; validated at EmailSender construction, not at load)
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    email_from: str | None = None
    email_to: str | None = None

    def get_huggingface_keys(self) -> list[str]:
        """Collect all Hugging Face keys from settings and environment variables."""
        import os
        import re

        keys: list[str] = []
        if self.huggingface_api_key:
            keys.append(self.huggingface_api_key.strip())
        if self.hf_token:
            keys.append(self.hf_token.strip())
        if self.huggingface_api_keys:
            for k in re.split(r"[,\s]+", self.huggingface_api_keys.strip()):
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

    def get_groq_keys(self) -> list[str]:
        """Collect all Groq keys from settings and environment variables."""
        import os
        import re

        keys: list[str] = []
        if self.groq_api_key:
            keys.append(self.groq_api_key.strip())
        if self.groq_api_keys:
            for k in re.split(r"[,\s]+", self.groq_api_keys.strip()):
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


def load_settings() -> Settings:
    return Settings()
