FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies first so this layer is cached across code edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY floorplan_seg ./floorplan_seg

ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "floorplan_seg"]
CMD ["--help"]
