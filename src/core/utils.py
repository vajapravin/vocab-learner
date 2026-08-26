from datetime import UTC, datetime


def _utcnow() -> datetime:
    return datetime.now(UTC)
