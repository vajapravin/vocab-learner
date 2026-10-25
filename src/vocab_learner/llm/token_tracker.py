"""SQLite-backed minimal token and rate limit tracker for LLM switching."""

from __future__ import annotations

import random
import sqlite3
import time
from pathlib import Path
from typing import Any

from vocab_learner.logging_setup import get_logger

log = get_logger(__name__)

# Default limits
DEFAULT_TPM_LIMIT = 30000  # tokens per minute
DEFAULT_RPM_LIMIT = 30  # requests per minute
DEFAULT_RPD_LIMIT = 1000  # requests per day (daily equivalent of RPM)
DEFAULT_TPD_LIMIT = 500000  # tokens per 24 hours (daily equivalent of TPM)
DEFAULT_DAILY_LIMIT = DEFAULT_TPD_LIMIT  # for backward compatibility
DEFAULT_COOLDOWN_SECONDS = 60.0  # cooldown when 429 is encountered


class TokenTracker:
    """
    Tracks token usage and rate limits per model in a lightweight SQLite database.
    Manages:
      - RPM (Requests Per Minute)
      - TPM (Tokens Per Minute)
      - RPD (Requests Per Day - daily equivalent of RPM)
      - TPD (Tokens Per Day - daily equivalent of TPM)
    Keeps minimal data storage by pruning history older than 24 hours on every write.
    """

    def __init__(
        self,
        db_path: Path,
        tpm_limit: int = DEFAULT_TPM_LIMIT,
        rpm_limit: int = DEFAULT_RPM_LIMIT,
        rpd_limit: int = DEFAULT_RPD_LIMIT,
        tpd_limit: int = DEFAULT_TPD_LIMIT,
        daily_limit: int | None = None,
    ) -> None:
        self.db_path = db_path
        self.default_tpm_limit = tpm_limit
        self.default_rpm_limit = rpm_limit
        self.default_rpd_limit = rpd_limit
        self.default_tpd_limit = daily_limit if daily_limit is not None else tpd_limit
        self.default_daily_limit = self.default_tpd_limit

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
                    rpd_limit INTEGER NOT NULL DEFAULT 1000,
                    tpd_limit INTEGER NOT NULL DEFAULT 500000,
                    cooldown_until REAL NOT NULL DEFAULT 0.0,
                    last_error TEXT,
                    updated_at REAL NOT NULL
                );
                """
            )

            # Auto-migrate columns if table existed without rpd_limit or tpd_limit
            cols = {
                row["name"] for row in conn.execute("PRAGMA table_info(model_limits);").fetchall()
            }
            if "rpd_limit" not in cols:
                conn.execute(
                    "ALTER TABLE model_limits ADD COLUMN rpd_limit INTEGER NOT NULL DEFAULT 1000;"
                )
            if "tpd_limit" not in cols:
                conn.execute(
                    "ALTER TABLE model_limits ADD COLUMN tpd_limit INTEGER NOT NULL DEFAULT 500000;"
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

    def register_model_limits(
        self,
        model_id: str,
        rpm_limit: int,
        rpd_limit: int,
        tpm_limit: int,
        tpd_limit: int,
    ) -> None:
        """Register or update limits for a specific model in SQLite."""
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO model_limits (
                    model_id, tpm_limit, rpm_limit, daily_limit,
                    rpd_limit, tpd_limit, cooldown_until, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0.0, NULL, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    tpm_limit = excluded.tpm_limit,
                    rpm_limit = excluded.rpm_limit,
                    daily_limit = excluded.daily_limit,
                    rpd_limit = excluded.rpd_limit,
                    tpd_limit = excluded.tpd_limit,
                    updated_at = excluded.updated_at
                """,
                (model_id, tpm_limit, rpm_limit, tpd_limit, rpd_limit, tpd_limit, now),
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
                    rpd_limit, tpd_limit, cooldown_until, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0.0, NULL, ?)
                """,
                (
                    model_id,
                    self.default_tpm_limit,
                    self.default_rpm_limit,
                    self.default_tpd_limit,
                    self.default_rpd_limit,
                    self.default_tpd_limit,
                    now,
                ),
            )
            conn.commit()
            return {
                "model_id": model_id,
                "tpm_limit": self.default_tpm_limit,
                "rpm_limit": self.default_rpm_limit,
                "daily_limit": self.default_tpd_limit,
                "rpd_limit": self.default_rpd_limit,
                "tpd_limit": self.default_tpd_limit,
                "cooldown_until": 0.0,
                "last_error": None,
                "updated_at": now,
            }

    def get_model_usage(
        self,
        model_id: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Return current usage and configured limits for a model across 60s and 24h."""
        current_time = now if now is not None else time.time()
        limits = self.get_or_create_limits(model_id)

        with self._get_connection() as conn:
            # 60s window
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

            # 24h window
            window_24h = current_time - 86400.0
            row_24h = conn.execute(
                """
                SELECT COALESCE(SUM(total_tokens), 0) AS sum_tokens, COUNT(*) AS req_count
                FROM token_usage
                WHERE model_id = ? AND timestamp >= ?
                """,
                (model_id, window_24h),
            ).fetchone()
            used_tokens_24h = row_24h["sum_tokens"] if row_24h else 0
            req_count_24h = row_24h["req_count"] if row_24h else 0

        return {
            "model_id": model_id,
            "rpm_used": req_count_60s,
            "rpm_limit": limits.get("rpm_limit", self.default_rpm_limit),
            "tpm_used": used_tokens_60s,
            "tpm_limit": limits.get("tpm_limit", self.default_tpm_limit),
            "rpd_used": req_count_24h,
            "rpd_limit": limits.get("rpd_limit", self.default_rpd_limit),
            "tpd_used": used_tokens_24h,
            "tpd_limit": limits.get("tpd_limit")
            or limits.get("daily_limit", self.default_tpd_limit),
            "cooldown_until": limits.get("cooldown_until", 0.0),
        }

    def is_within_limit(
        self,
        model_id: str,
        estimated_tokens: int = 1000,
        now: float | None = None,
    ) -> tuple[bool, str | None]:
        """
        Check whether calling `model_id` with `estimated_tokens` stays within limits:
          1. Cooldown from prior 429/402 errors
          2. RPM (Requests Per Minute)
          3. TPM (Tokens Per Minute)
          4. RPD (Requests Per Day)
          5. TPD (Tokens Per Day)
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

            rpm_limit = limits.get("rpm_limit", self.default_rpm_limit)
            if (req_count_60s + 1) > rpm_limit:
                return (
                    False,
                    f"RPM limit exceeded: {req_count_60s} + 1 > {rpm_limit} req/min",
                )

            tpm_limit = limits.get("tpm_limit", self.default_tpm_limit)
            if (used_tokens_60s + estimated_tokens) > tpm_limit:
                msg = (
                    f"TPM limit exceeded: {used_tokens_60s} + {estimated_tokens} > "
                    f"{tpm_limit} tokens/min"
                )
                return (False, msg)

            # 3. Check 24-hour window RPD (daily requests) and TPD (daily tokens)
            window_24h = current_time - 86400.0
            row_24h = conn.execute(
                """
                SELECT COALESCE(SUM(total_tokens), 0) AS sum_tokens, COUNT(*) AS req_count
                FROM token_usage
                WHERE model_id = ? AND timestamp >= ?
                """,
                (model_id, window_24h),
            ).fetchone()

            used_tokens_24h = row_24h["sum_tokens"] if row_24h else 0
            req_count_24h = row_24h["req_count"] if row_24h else 0

            rpd_limit = limits.get("rpd_limit", self.default_rpd_limit)
            if (req_count_24h + 1) > rpd_limit:
                return (
                    False,
                    f"RPD limit exceeded: {req_count_24h} + 1 > {rpd_limit} req/day",
                )

            tpd_limit = limits.get("tpd_limit") or limits.get("daily_limit", self.default_tpd_limit)
            if (used_tokens_24h + estimated_tokens) > tpd_limit:
                err_msg = (
                    f"TPD limit exceeded: {used_tokens_24h} + {estimated_tokens} > "
                    f"{tpd_limit} tokens/day"
                )
                return (False, err_msg)

        return (True, None)

    def get_seconds_until_available(
        self,
        model_id: str,
        estimated_tokens: int = 1000,
        now: float | None = None,
    ) -> float:
        """Calculate estimated wait time in seconds until `model_id` has room

        for `estimated_tokens` within its limits (cooldown, 60s TPM/RPM, 24h TPD/RPD).
        """
        current_time = now if now is not None else time.time()
        limits = self.get_or_create_limits(model_id)

        # 1. Cooldown wait
        cooldown_until = limits.get("cooldown_until", 0.0)
        cooldown_wait = max(0.0, cooldown_until - current_time)

        # 2. Sliding 60-second window TPM & RPM wait
        window_60s = current_time - 60.0
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, total_tokens
                FROM token_usage
                WHERE model_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC
                """,
                (model_id, window_60s),
            ).fetchall()

            used_tokens_60s = sum(r["total_tokens"] for r in rows)
            req_count_60s = len(rows)

            tpm_wait = 0.0
            tpm_limit = limits.get("tpm_limit", self.default_tpm_limit)
            if (used_tokens_60s + estimated_tokens) > tpm_limit:
                needed_free = (used_tokens_60s + estimated_tokens) - tpm_limit
                accum = 0
                for r in rows:
                    accum += int(r["total_tokens"])
                    if accum >= needed_free:
                        tpm_wait = max(0.0, float(r["timestamp"]) + 60.0 - current_time)
                        break

            rpm_wait = 0.0
            rpm_limit = limits.get("rpm_limit", self.default_rpm_limit)
            if (req_count_60s + 1) > rpm_limit and rows:
                oldest = rows[0]
                rpm_wait = max(0.0, float(oldest["timestamp"]) + 60.0 - current_time)

        return float(max(cooldown_wait, tpm_wait, rpm_wait))

    def select_available_model(
        self,
        candidate_models: list[str],
        estimated_tokens: int = 1000,
        preferred_model: str | None = None,
        now: float | None = None,
    ) -> tuple[str, bool]:
        """
        Check candidate models and return the model whose limit is NOT exceeded.
        When multiple models are available, prioritizes the model with the lowest
        recent 60-second token usage to balance load evenly and avoid hitting TPM limits.
        Returns (chosen_model, switched_from_preferred).
        """
        current_time = now if now is not None else time.time()

        # Gather status and 60-second usage for all candidate models
        available_candidates: list[tuple[str, int]] = []
        for candidate in candidate_models:
            is_ok, _ = self.is_within_limit(
                candidate,
                estimated_tokens=estimated_tokens,
                now=current_time,
            )
            if is_ok:
                usage = self.get_model_usage(candidate, now=current_time)
                available_candidates.append((candidate, usage["tpm_used"]))

        if available_candidates:
            # Sort by lowest 60s token usage; prioritize preferred_model if tied
            available_candidates.sort(
                key=lambda item: (item[1], 0 if item[0] == preferred_model else 1)
            )
            best_model = available_candidates[0][0]
            switched = (preferred_model is not None) and (best_model != preferred_model)
            if switched:
                log.info(
                    "switched_to_least_loaded_llm",
                    model=best_model,
                    preferred=preferred_model,
                    tokens_used_60s=available_candidates[0][1],
                )
            return (best_model, switched)

        # If ALL candidates are over limit, choose the candidate that will be ready soonest
        wait_times = [
            (
                c,
                self.get_seconds_until_available(
                    c, estimated_tokens=estimated_tokens, now=current_time
                ),
            )
            for c in candidate_models
        ]
        wait_times.sort(key=lambda item: item[1])
        fallback = wait_times[0][0] if wait_times else candidate_models[0]
        min_wait = round(wait_times[0][1] + 5.0, 2) if wait_times else 5.0
        log.warning(
            "all_llms_limit_exceeded_using_fallback",
            fallback=fallback,
            candidate_count=len(candidate_models),
            min_wait_seconds=min_wait,
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
        jitter_seconds: float | None = None,
    ) -> None:
        """Mark a model in cooldown (e.g. after HTTP 429 response) in SQLite.

        Adds a small amount of random delay (noise / jitter) to cooldown_seconds
        so multiple concurrent requests don't all retry at the exact same millisecond.
        """
        current_time = now if now is not None else time.time()
        # Add random jitter (1.0 to 4.0s) unless explicitly overridden
        noise = jitter_seconds if jitter_seconds is not None else random.uniform(1.0, 4.0)
        effective_cooldown = cooldown_seconds + noise
        cooldown_until = current_time + effective_cooldown

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
            cooldown_seconds=round(effective_cooldown, 2),
            jitter=round(noise, 2),
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
