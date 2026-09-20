FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# HOME and UV_CACHE_DIR point at /tmp so the image also runs under an arbitrary
# --user uid that has no passwd entry (see docker-compose.yml).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install runtime dependencies first so this layer is cached across code edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY floorplan_seg ./floorplan_seg

# Run unprivileged so files written to the bind-mounted ./output belong to
# uid 1000 (the first user on most Linux hosts) rather than to root.
RUN useradd --system --uid 1000 --no-create-home --shell /usr/sbin/nologin app
USER app

ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "floorplan_seg"]
CMD ["--help"]
