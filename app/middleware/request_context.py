"""
RequestContextMiddleware
───────────────────────
For every HTTP request this middleware:

1. Generates (or propagates) a unique request ID from the X-Request-ID header.
2. Injects it into the logging context var so every log line is correlated.
3. Logs the incoming request and outgoing response at INFO level.
4. Attaches timing, rate-limit, and request-ID response headers.
5. Records HTTP metrics (count + latency) via Prometheus.
"""
from __future__ import annotations

import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.logging import get_logger, request_id_ctx
from app.core.metrics import HTTP_LATENCY, HTTP_REQUESTS

logger = get_logger("middleware")

# Endpoints we don't want to log as noisy access entries
_SILENT_PATHS = {"/api/v1/health", "/metrics", "/favicon.ico"}


class RequestContextMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # ── 1. Request ID ─────────────────────────────────────────────────────
        request_id = (
            request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )
        token = request_id_ctx.set(request_id)

        start = time.perf_counter()

        # ── 2. Log incoming request ───────────────────────────────────────────
        if request.url.path not in _SILENT_PATHS:
            logger.info(
                "→ %s %s",
                request.method,
                request.url.path,
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": self._client_ip(request),
                    "user_agent": request.headers.get("User-Agent", ""),
                },
            )

        # ── 3. Process request ────────────────────────────────────────────────
        try:
            response: Response = await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled exception in request", exc_info=exc)
            raise
        finally:
            request_id_ctx.reset(token)

        elapsed = time.perf_counter() - start

        # ── 4. Metrics ────────────────────────────────────────────────────────
        endpoint = request.url.path
        HTTP_REQUESTS.labels(  # type: ignore[union-attr]
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()
        HTTP_LATENCY.labels(method=request.method, endpoint=endpoint).observe(elapsed)  # type: ignore[union-attr]

        # ── 5. Response headers ───────────────────────────────────────────────
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.2f}"

        # Propagate rate-limit headers set by the security dependency
        if hasattr(request.state, "rate_limit_remaining"):
            response.headers["X-RateLimit-Limit"] = str(request.state.rate_limit_limit)
            response.headers["X-RateLimit-Remaining"] = str(request.state.rate_limit_remaining)

        # ── 6. Log outgoing response ──────────────────────────────────────────
        if request.url.path not in _SILENT_PATHS:
            logger.info(
                "← %s %s %d (%.0fms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed * 1000,
                extra={
                    "status_code": response.status_code,
                    "duration_ms": round(elapsed * 1000, 2),
                },
            )

        return response

    @staticmethod
    def _client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
