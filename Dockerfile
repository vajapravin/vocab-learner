FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_EXTRA_INDEX_URL="https://abetlen.github.io/llama-cpp-python/whl/cpu"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    build-essential cmake \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# Copy only manifests first for layer caching
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project || uv sync --no-install-project

# Source is bind-mounted in dev; installed as editable
COPY . .
RUN uv sync

ENV PATH="/opt/venv/bin:$PATH"

CMD ["bash"]
