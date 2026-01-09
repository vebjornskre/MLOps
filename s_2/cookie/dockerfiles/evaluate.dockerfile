# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY src/cookie_test/data/ src/cookie_test/data/
COPY README.md README.md
COPY models/ models/
COPY models/trained_model.pth models/trained_model.pth
COPY reports/figures/ reports/figures/

WORKDIR /
RUN uv sync --locked --no-cache --no-install-project

# ALWAYS ADD THIS, MAKES THE DOCKER BUILD WAY FASTER
# AFTER THE FIRST ONE
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

ENTRYPOINT ["uv", "run", "src/cookie_test/evaluate.py"]
