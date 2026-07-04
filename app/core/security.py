"""
Security layer: API-key authentication + sliding-window rate limiting.

API Key Auth
────────────
Pass `X-API-Key: <key>` header. If API_KEYS is empty in .env, auth is disabled
(useful for local dev). In production, set at least one key.

Rate Limiting
─────────────
In-process sliding-window limiter keyed by (api_key or client IP).
For multi-instance deployments, swap _RateLimiter with a Redis-backed
implementation — the interface is identical.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Deque, Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security.api_key import APIKeyHeader

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("security")

# ── API Key ───────────────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(
    name=settings.API_KEY_HEADER,
    auto_error=False,  # We handle the 401 ourselves for a richer error body
)


async def verify_api_key(
    api_key: Optional[str] = Security(_api_key_header),
) -> Optional[str]:
    """
    FastAPI dependency. Returns the validated key, or None if auth is disabled.
    Raises HTTP 401 when auth is enabled and the key is missing/invalid.
    """
    if not settings.auth_enabled:
        return None

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing API key. Supply it via the '{settings.API_KEY_HEADER}' header.",
            headers={"WWW-Authenticate": settings.API_KEY_HEADER},
        )

    if api_key not in settings.API_KEYS:
        logger.warning("Invalid API key attempt", extra={"key_prefix": api_key[:8]})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": settings.API_KEY_HEADER},
        )

    return api_key


# ── Sliding-window rate limiter ───────────────────────────────────────────────

class _SlidingWindowRateLimiter:
    """
    Per-identity sliding-window counter.

    Timestamps of recent requests are kept in a deque; on each call, entries
    older than the window are dropped. If the remaining count exceeds the limit,
    the request is rejected.

    Memory: O(rate_limit_requests) per unique identity — negligible.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._buckets: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def is_allowed(self, identity: str) -> tuple[bool, int]:
        """
        Returns (allowed, remaining_requests).
        Thread-safe via asyncio lock.
        """
        async with self._lock:
            now = time.monotonic()
            cutoff = now - self._window
            bucket = self._buckets[identity]

            # Evict expired timestamps
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            remaining = max(0, self._max - len(bucket))
            if len(bucket) >= self._max:
                return False, 0

            bucket.append(now)
            return True, remaining - 1


_limiter = _SlidingWindowRateLimiter(
    max_requests=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
)


def _client_identity(request: Request, api_key: Optional[str]) -> str:
    """Prefer API key as identity; fall back to IP address."""
    if api_key:
        return f"key:{api_key[:16]}"
    forwarded_for = request.headers.get("X-Forwarded-For")
    ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.client.host if request.client else "unknown")
    return f"ip:{ip}"


async def rate_limit(
    request: Request,
    api_key: Optional[str] = Depends(verify_api_key),
) -> None:
    """
    FastAPI dependency. Raises HTTP 429 when the caller exceeds their quota.
    Adds standard RateLimit-* headers to every response.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    identity = _client_identity(request, api_key)
    allowed, remaining = await _limiter.is_allowed(identity)

    # Attach headers so clients can implement back-off
    request.state.rate_limit_remaining = remaining
    request.state.rate_limit_limit = settings.RATE_LIMIT_REQUESTS

    if not allowed:
        logger.warning("Rate limit exceeded", extra={"identity": identity})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {settings.RATE_LIMIT_REQUESTS} requests "
                f"per {settings.RATE_LIMIT_WINDOW_SECONDS}s."
            ),
            headers={
                "Retry-After": str(settings.RATE_LIMIT_WINDOW_SECONDS),
                "X-RateLimit-Limit": str(settings.RATE_LIMIT_REQUESTS),
                "X-RateLimit-Remaining": "0",
            },
        )
