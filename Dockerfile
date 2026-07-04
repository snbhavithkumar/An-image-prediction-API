# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into a prefix so we can copy just the packages to the runtime image
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system deps only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — principle of least privilege
RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source — owned by non-root user
COPY --chown=appuser:appgroup app/ app/
COPY --chown=appuser:appgroup .env.example .env

# Create models directory for custom weights (mount a volume here in production)
RUN mkdir -p app/models && chown -R appuser:appgroup app/models

USER appuser

EXPOSE 8000

# Kubernetes/ECS liveness + readiness in one
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=90s \
    --retries=3 \
    CMD curl --silent --fail http://localhost:8000/api/v1/health || exit 1

# Exec form — uvicorn receives SIGTERM cleanly for graceful shutdown.
# Workers=1: PyTorch models are not fork-safe.
# Use horizontal scaling (multiple containers) for throughput.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "h11", \
     "--access-log", \
     "--proxy-headers"]
