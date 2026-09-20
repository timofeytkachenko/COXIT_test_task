FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# HOME points at /tmp so the image also runs under an arbitrary --user uid that
# has no passwd entry (see docker-compose.yml).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install runtime dependencies first so this layer is cached across code edits.
# The uv cache is only needed while syncing; leaving it behind would ship
# ~200 MB of root-owned files that a non-root runtime user cannot open.
COPY pyproject.toml uv.lock ./
RUN UV_CACHE_DIR=/tmp/uv-cache-build \
        uv sync --frozen --no-dev --no-install-project \
    && rm -rf /tmp/uv-cache-build

# Guard anyone who invokes uv inside the container: it must not recreate a
# root-owned cache that a non-root --user could then not open.
ENV UV_NO_CACHE=1

COPY floorplan_seg ./floorplan_seg

# Run unprivileged so files written to the bind-mounted ./output belong to
# uid 1000 (the first user on most Linux hosts) rather than to root.
RUN useradd --system --uid 1000 --no-create-home --shell /usr/sbin/nologin app
USER app

# Call the venv interpreter directly: `uv run` is not needed once the
# environment is synced, and it would try to initialise a cache at runtime.
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["python", "-m", "floorplan_seg"]
CMD ["--help"]
