"""
Prediction result cache keyed by SHA-256 of the raw image bytes.

Identical images (same bytes) → instant response from cache, zero GPU usage.
Uses an in-process LRU with a configurable TTL so stale entries expire.

Design notes:
- Thread-safe: asyncio is single-threaded but the lock prevents races if you
  ever switch to a multi-worker setup.
- Swap-friendly: the CacheBackend protocol makes it trivial to plug in Redis
  for multi-instance deployments without changing call sites.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from typing import List, Optional, Protocol

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import CACHE_HITS, CACHE_MISSES
from app.schemas.prediction import PredictionResult

logger = get_logger("cache")


# ── Protocol ──────────────────────────────────────────────────────────────────

class CacheBackend(Protocol):
    async def get(self, key: str) -> Optional[List[PredictionResult]]: ...
    async def set(self, key: str, value: List[PredictionResult]) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def clear(self) -> None: ...
    async def size(self) -> int: ...


# ── In-process LRU implementation ─────────────────────────────────────────────

class InMemoryLRUCache:
    """
    Thread-safe LRU cache with TTL expiry.

    Entry format: (inserted_at_unix, List[PredictionResult])
    """

    def __init__(self, max_size: int, ttl_seconds: int) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[float, List[PredictionResult]]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[List[PredictionResult]]:
        async with self._lock:
            if key not in self._store:
                CACHE_MISSES.inc()  # type: ignore[union-attr]
                return None

            inserted_at, value = self._store[key]
            if time.monotonic() - inserted_at > self._ttl:
                del self._store[key]
                CACHE_MISSES.inc()  # type: ignore[union-attr]
                logger.debug("Cache entry expired", extra={"key": key[:16]})
                return None

            # Move to end (most recently used)
            self._store.move_to_end(key)
            CACHE_HITS.inc()  # type: ignore[union-attr]
            logger.debug("Cache hit", extra={"key": key[:16]})
            return value

    async def set(self, key: str, value: List[PredictionResult]) -> None:
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.monotonic(), value)
            if len(self._store) > self._max_size:
                evicted = self._store.popitem(last=False)
                logger.debug("Cache evicted LRU entry", extra={"key": str(evicted[0])[:16]})

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    async def size(self) -> int:
        async with self._lock:
            return len(self._store)


# ── Helpers ───────────────────────────────────────────────────────────────────

def image_cache_key(image_bytes: bytes) -> str:
    """Deterministic cache key: hex SHA-256 of the raw image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()


# ── Module-level singleton ────────────────────────────────────────────────────

prediction_cache: CacheBackend = InMemoryLRUCache(
    max_size=settings.CACHE_MAX_SIZE,
    ttl_seconds=settings.CACHE_TTL_SECONDS,
)
