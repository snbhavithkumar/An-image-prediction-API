"""
Prometheus metrics exposed at GET /metrics.

Tracked:
  - http_requests_total          (counter)   labelled by method, endpoint, status
  - http_request_duration_seconds (histogram) p50/p95/p99 latency per endpoint
  - inference_duration_seconds   (histogram)  pure model forward-pass time
  - inference_batch_size         (histogram)  items per batch when batching is on
  - cache_hits_total             (counter)    result cache hits
  - cache_misses_total           (counter)    result cache misses
  - model_load_duration_seconds  (gauge)      how long startup model load took
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

from app.core.config import settings

# ── Registry ──────────────────────────────────────────────────────────────────
# Use a custom registry so we don't accidentally expose process/platform metrics
# to external callers (those stay on the default registry).
_registry = CollectorRegistry(auto_describe=True) if _PROMETHEUS_AVAILABLE else None


def _make_counter(name: str, doc: str, labels: list[str] | None = None) -> "Counter | _NoOpMetric":
    if not _PROMETHEUS_AVAILABLE or not settings.METRICS_ENABLED:
        return _NoOpMetric()
    return Counter(name, doc, labels or [], registry=_registry)


def _make_histogram(name: str, doc: str, labels: list[str] | None = None, buckets: list[float] | None = None) -> "Histogram | _NoOpMetric":
    if not _PROMETHEUS_AVAILABLE or not settings.METRICS_ENABLED:
        return _NoOpMetric()
    kwargs = {"registry": _registry, "labelnames": labels or []}
    if buckets:
        kwargs["buckets"] = buckets
    return Histogram(name, doc, **kwargs)


def _make_gauge(name: str, doc: str) -> "Gauge | _NoOpMetric":
    if not _PROMETHEUS_AVAILABLE or not settings.METRICS_ENABLED:
        return _NoOpMetric()
    return Gauge(name, doc, registry=_registry)


# ── Metric definitions ────────────────────────────────────────────────────────

HTTP_REQUESTS = _make_counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

HTTP_LATENCY = _make_histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

INFERENCE_DURATION = _make_histogram(
    "inference_duration_seconds",
    "Model forward-pass duration in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

INFERENCE_BATCH_SIZE = _make_histogram(
    "inference_batch_size",
    "Number of images processed per batch",
    buckets=[1, 2, 4, 8, 16, 32],
)

CACHE_HITS = _make_counter("cache_hits_total", "Prediction cache hits")
CACHE_MISSES = _make_counter("cache_misses_total", "Prediction cache misses")

MODEL_LOAD_DURATION = _make_gauge(
    "model_load_duration_seconds",
    "Time taken to load the model at startup",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@contextmanager
def track_inference_time() -> Generator[None, None, None]:
    """Context manager that records how long the wrapped block takes."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if _PROMETHEUS_AVAILABLE and settings.METRICS_ENABLED:
            INFERENCE_DURATION.observe(elapsed)  # type: ignore[union-attr]


def metrics_output() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    if not _PROMETHEUS_AVAILABLE or not settings.METRICS_ENABLED:
        return b"# metrics disabled\n", "text/plain"
    return generate_latest(_registry), CONTENT_TYPE_LATEST  # type: ignore[return-value]


# ── No-op fallback ────────────────────────────────────────────────────────────

class _NoOpMetric:
    """Swallows all metric calls when prometheus_client is not installed."""
    def labels(self, **_: object) -> "_NoOpMetric":
        return self
    def inc(self, *_: object, **__: object) -> None: ...
    def observe(self, *_: object, **__: object) -> None: ...
    def set(self, *_: object, **__: object) -> None: ...
