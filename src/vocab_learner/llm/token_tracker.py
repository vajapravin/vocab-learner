"""SQLite-backed minimal token and rate limit tracker for LLM switching."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from vocab_learner.logging_setup import get_logger

log = get_logger(__name__)

# Default limits
DEFAULT_TPM_LIMIT = 30000  # tokens per minute
DEFAULT_RPM_LIMIT = 30  # requests per minute
DEFAULT_DAILY_LIMIT = 500000  # tokens per 24 hours
DEFAULT_COOLDOWN_SECONDS = 60.0  # cooldown when 429 is encountered


class TokenTracker:
    """
    Tracks token usage and rate limits per model in a lightweight SQLite database.
    Keeps minimal data storage by pruning history older than 24 hours on every write.
    """

    def __init__(
        self,
        db_path: Path,
        tpm_limit: int = DEFAULT_TPM_LIMIT,
        rpm_limit: int = DEFAULT_RPM_LIMIT,
        daily_limit: int = DEFAULT_DAILY_LIMIT,
    ) -> None:
        self.db_path = db_path
        self.default_tpm_limit = tpm_limit
        self.default_rpm_limit = rpm_limit
        self.default_daily_limit = daily_limit

        self._ensure_dir()
        self._init_db()

    def _ensure_dir(self) -> None:
        if self.db_path != Path(":memory:"):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            # Enable WAL mode and auto vacuum for minimal footprint and concurrency
            if str(self.db_path) != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA auto_vacuum=INCREMENTAL;")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_limits (
                    model_id TEXT PRIMARY KEY,
                    tpm_limit INTEGER NOT NULL,
                    rpm_limit INTEGER NOT NULL,
                    daily_limit INTEGER NOT NULL,
                    cooldown_until REAL NOT NULL DEFAULT 0.0,
                    last_error TEXT,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    timestamp REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_token_usage_model_time
                ON token_usage(model_id, timestamp);
                """
            )
            conn.commit()

    def get_or_create_limits(self, model_id: str) -> dict[str, Any]:
        """Fetch model limits and cooldown from SQLite, or insert defaults."""
        now = time.time()
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM model_limits WHERE model_id = ?",
                (model_id,),
            ).fetchone()

            if row:
                return dict(row)

            # Insert default row
            conn.execute(
                """
                INSERT INTO model_limits (
                    model_id, tpm_limit, rpm_limit, daily_limit,
                    cooldown_until, last_error, updated_at
                ) VALUES (?, ?, ?, ?, 0.0, NULL, ?)
                """,
                (
                    model_id,
                    self.default_tpm_limit,
                    self.default_rpm_limit,
                    self.default_daily_limit,
                    now,
                ),
            )
            conn.commit()
            return {
                "model_id": model_id,
                "tpm_limit": self.default_tpm_limit,
                "rpm_limit": self.default_rpm_limit,
                "daily_limit": self.default_daily_limit,
                "cooldown_until": 0.0,
                "last_error": None,
                "updated_at": now,
            }

    def is_within_limit(
        self,
        model_id: str,
        estimated_tokens: int = 1000,
        now: float | None = None,
    ) -> tuple[bool, str | None]:
        """
        Check whether calling `model_id` with `estimated_tokens` stays within limits.
        Returns (is_available, reason_if_exceeded).
        """
        current_time = now if now is not None else time.time()
        limits = self.get_or_create_limits(model_id)

        # 1. Check cooldown (e.g. from a prior 429 response)
        cooldown_until = limits.get("cooldown_until", 0.0)
        if current_time < cooldown_until:
            wait_sec = round(cooldown_until - current_time, 1)
            return (
                False,
                f"Model in cooldown for {wait_sec}s due to error: {limits.get('last_error')}",
            )

        with self._get_connection() as conn:
            # 2. Check 60-second window TPM and RPM
            window_60s = current_time - 60.0
            row_60s = conn.execute(
                """
                SELECT COALESCE(SUM(total_tokens), 0) AS sum_tokens, COUNT(*) AS req_count
                FROM token_usage
                WHERE model_id = ? AND timestamp >= ?
                """,
                (model_id, window_60s),
            ).fetchone()

            used_tokens_60s = row_60s["sum_tokens"] if row_60s else 0
            req_count_60s = row_60s["req_count"] if row_60s else 0

            tpm_limit = limits["tpm_limit"]
            if (used_tokens_60s + estimated_tokens) > tpm_limit:
                return (
                    False,
                    f"TPM limit exceeded: {used_tokens_60s} + {estimated_tokens} > {tpm_limit}",
                )

            rpm_limit = limits["rpm_limit"]
            if (req_count_60s + 1) > rpm_limit:
                return (
                    False,
                    f"RPM limit exceeded: {req_count_60s} + 1 > {rpm_limit}",
                )

            # 3. Check 24-hour window daily limit
            window_24h = current_time - 86400.0
            row_24h = conn.execute(
                """
                SELECT COALESCE(SUM(total_tokens), 0) AS sum_tokens
                FROM token_usage
                WHERE model_id = ? AND timestamp >= ?
                """,
                (model_id, window_24h),
            ).fetchone()

            used_tokens_24h = row_24h["sum_tokens"] if row_24h else 0
            daily_limit = limits["daily_limit"]
            if (used_tokens_24h + estimated_tokens) > daily_limit:
                err_msg = (
                    f"Daily token limit exceeded: {used_tokens_24h} + {estimated_tokens} > "
                    f"{daily_limit}"
                )
                return (False, err_msg)

        return (True, None)

    def select_available_model(
        self,
        candidate_models: list[str],
        estimated_tokens: int = 1000,
        preferred_model: str | None = None,
        now: float | None = None,
    ) -> tuple[str, bool]:
        """
        Check candidate models and return the first model whose limit is NOT exceeded.
        Returns (chosen_model, switched_from_preferred).
        """
        current_time = now if now is not None else time.time()

        # Check preferred model first if valid
        if preferred_model and preferred_model in candidate_models:
            is_ok, reason = self.is_within_limit(
                preferred_model,
                estimated_tokens=estimated_tokens,
                now=current_time,
            )
            if is_ok:
                return (preferred_model, False)
            log.info(
                "preferred_llm_limit_exceeded_switching",
                preferred=preferred_model,
                reason=reason,
            )

        # Iterate through candidate models to find one whose limit is not exceeded
        reasons: dict[str, str] = {}
        for candidate in candidate_models:
            if candidate == preferred_model:
                continue
            is_ok, reason = self.is_within_limit(
                candidate,
                estimated_tokens=estimated_tokens,
                now=current_time,
            )
            if is_ok:
                log.info(
                    "switched_to_available_llm",
                    model=candidate,
                    switched_from=preferred_model,
                )
                return (candidate, True)
            if reason:
                reasons[candidate] = reason

        # If all candidates are over limit, choose fallback model
        fallback = candidate_models[0] if candidate_models else "meta-llama/Llama-3.1-8B-Instruct"
        log.warning(
            "all_llms_limit_exceeded_using_fallback",
            fallback=fallback,
            candidate_count=len(candidate_models),
        )
        return (fallback, preferred_model != fallback)

    def record_usage(
        self,
        model_id: str,
        key_hash: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int | None = None,
        now: float | None = None,
    ) -> None:
        """Record token usage event in SQLite and prune older records for minimal storage."""
        current_time = now if now is not None else time.time()
        actual_total = (
            total_tokens if total_tokens is not None else (prompt_tokens + completion_tokens)
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO token_usage (
                    model_id, key_hash, prompt_tokens, completion_tokens, total_tokens, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    key_hash,
                    prompt_tokens,
                    completion_tokens,
                    actual_total,
                    current_time,
                ),
            )

            # Minimal data storage: keep only the last 24 hours of usage events
            prune_cutoff = current_time - 86400.0
            conn.execute(
                "DELETE FROM token_usage WHERE timestamp < ?",
                (prune_cutoff,),
            )
            conn.commit()

    def mark_cooldown(
        self,
        model_id: str,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        error_message: str | None = None,
        now: float | None = None,
    ) -> None:
        """Mark a model in cooldown (e.g. after HTTP 429 response) in SQLite."""
        current_time = now if now is not None else time.time()
        cooldown_until = current_time + cooldown_seconds

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO model_limits (
                    model_id, tpm_limit, rpm_limit, daily_limit,
                    cooldown_until, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    cooldown_until = excluded.cooldown_until,
                    last_error = excluded.last_error,
                    updated_at = excluded.updated_at
                """,
                (
                    model_id,
                    self.default_tpm_limit,
                    self.default_rpm_limit,
                    self.default_daily_limit,
                    cooldown_until,
                    error_message,
                    current_time,
                ),
            )
            conn.commit()

        log.warning(
            "model_marked_cooldown",
            model=model_id,
            cooldown_seconds=cooldown_seconds,
            error=error_message,
        )

    def clear_cooldown(self, model_id: str) -> None:
        """Clear cooldown for a model in SQLite."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE model_limits SET cooldown_until = 0.0, last_error = NULL "
                "WHERE model_id = ?",
                (model_id,),
            )
            conn.commit()
