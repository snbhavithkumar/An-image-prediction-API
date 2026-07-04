"""
Unit tests for the ML engine, cache, rate limiter, and configuration.

None of these tests require a GPU or internet connection.
The model is replaced with a tiny fake that returns predictable logits.
"""
from __future__ import annotations

import asyncio
import io
import time
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from app.core.cache import InMemoryLRUCache, image_cache_key
from app.core.security import _SlidingWindowRateLimiter
from app.schemas.prediction import PredictionResult
from app.services.ml_engine import MLEngine
from tests.conftest import make_image_bytes


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_rgb_bytes(w: int = 224, h: int = 224) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(0, 128, 255)).save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def _make_loaded_engine(num_classes: int = 1000, hot_class: int = 42) -> MLEngine:
    """Return an MLEngine whose model is replaced with a minimal fake."""
    engine = MLEngine()
    engine._is_loaded = True
    engine._labels = [f"class_{i}" for i in range(num_classes)]

    # Deterministic fake model: class `hot_class` gets logit 10, rest 0
    logits = torch.zeros(1, num_classes)
    logits[0, hot_class] = 10.0

    mock_model = MagicMock()
    mock_model.return_value = logits
    engine._model = mock_model
    return engine


# ── Preprocessing ─────────────────────────────────────────────────────────────

class TestPreprocess:

    def setup_method(self):
        self.engine = MLEngine()

    def test_output_shape(self):
        t = self.engine.preprocess(make_rgb_bytes())
        assert t.shape == (1, 3, 224, 224)

    def test_output_dtype(self):
        t = self.engine.preprocess(make_rgb_bytes())
        assert t.dtype == torch.float32

    def test_normalised_values_not_raw_uint8(self):
        t = self.engine.preprocess(make_rgb_bytes())
        assert t.min().item() < 1.0

    def test_corrupt_bytes_raise_value_error(self):
        with pytest.raises(ValueError, match="Could not decode"):
            self.engine.preprocess(b"not-an-image")

    def test_small_image_resized(self):
        t = self.engine.preprocess(make_rgb_bytes(50, 50))
        assert t.shape == (1, 3, 224, 224)

    def test_wide_image_centre_cropped(self):
        t = self.engine.preprocess(make_rgb_bytes(640, 480))
        assert t.shape == (1, 3, 224, 224)

    def test_grayscale_converted_to_rgb(self):
        buf = io.BytesIO()
        Image.new("L", (224, 224)).save(buf, format="JPEG")
        t = self.engine.preprocess(buf.getvalue())
        assert t.shape == (1, 3, 224, 224)


# ── Synchronous forward pass (_forward) ──────────────────────────────────────

class TestForward:

    def test_top_class_is_hottest_logit(self):
        engine = _make_loaded_engine(hot_class=42)
        tensor = torch.zeros(1, 3, 224, 224)
        results = engine._forward(tensor)
        assert len(results) == 1
        assert results[0][0].label == "class_42"

    def test_confidence_sums_below_one_for_top_k(self):
        engine = _make_loaded_engine()
        tensor = torch.zeros(1, 3, 224, 224)
        results = engine._forward(tensor)
        total = sum(r.confidence for r in results[0])
        # Top-K confidences don't have to sum to 1, but must be ≤ 1
        assert total <= 1.0 + 1e-6

    def test_top_confidence_dominates(self):
        """class_42 logit=10 vs all-zeros → softmax should give >0.99."""
        engine = _make_loaded_engine(hot_class=42)
        tensor = torch.zeros(1, 3, 224, 224)
        results = engine._forward(tensor)
        assert results[0][0].confidence > 0.99

    def test_returns_top_k_items(self):
        from app.core.config import settings
        engine = _make_loaded_engine()
        tensor = torch.zeros(1, 3, 224, 224)
        results = engine._forward(tensor)
        assert len(results[0]) == settings.TOP_K_PREDICTIONS

    def test_confidence_within_unit_interval(self):
        engine = _make_loaded_engine()
        tensor = torch.zeros(1, 3, 224, 224)
        results = engine._forward(tensor)
        for r in results[0]:
            assert 0.0 <= r.confidence <= 1.0

    def test_not_loaded_predict_raises(self):
        engine = MLEngine()  # _is_loaded=False
        with pytest.raises(RuntimeError, match="not loaded"):
            asyncio.get_event_loop().run_until_complete(engine.predict(make_rgb_bytes()))


# ── Batch inference (async) ───────────────────────────────────────────────────

class TestAsyncPredict:

    def test_predict_returns_list_of_results(self):
        engine = _make_loaded_engine()
        loop = asyncio.new_event_loop()
        engine.start_batch_worker(loop)
        try:
            result = loop.run_until_complete(engine.predict(make_rgb_bytes()))
        finally:
            loop.run_until_complete(engine.shutdown())
            loop.close()
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], PredictionResult)


# ── Cache ─────────────────────────────────────────────────────────────────────

class TestInMemoryLRUCache:

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_set_and_get(self):
        cache = InMemoryLRUCache(max_size=10, ttl_seconds=60)
        data = [PredictionResult(label="cat", confidence=0.99)]
        self._run(cache.set("key1", data))
        result = self._run(cache.get("key1"))
        assert result is not None
        assert result[0].label == "cat"

    def test_missing_key_returns_none(self):
        cache = InMemoryLRUCache(max_size=10, ttl_seconds=60)
        result = self._run(cache.get("does-not-exist"))
        assert result is None

    def test_expired_entry_returns_none(self):
        cache = InMemoryLRUCache(max_size=10, ttl_seconds=0)  # TTL = 0 → immediate expiry
        data = [PredictionResult(label="dog", confidence=0.5)]
        self._run(cache.set("key2", data))
        time.sleep(0.01)
        result = self._run(cache.get("key2"))
        assert result is None

    def test_lru_eviction(self):
        cache = InMemoryLRUCache(max_size=2, ttl_seconds=60)
        data = lambda label: [PredictionResult(label=label, confidence=0.5)]
        self._run(cache.set("a", data("a")))
        self._run(cache.set("b", data("b")))
        self._run(cache.set("c", data("c")))  # should evict "a"
        assert self._run(cache.get("a")) is None
        assert self._run(cache.get("b")) is not None
        assert self._run(cache.get("c")) is not None

    def test_clear(self):
        cache = InMemoryLRUCache(max_size=10, ttl_seconds=60)
        self._run(cache.set("k", [PredictionResult(label="x", confidence=0.1)]))
        self._run(cache.clear())
        assert self._run(cache.get("k")) is None

    def test_size(self):
        cache = InMemoryLRUCache(max_size=10, ttl_seconds=60)
        self._run(cache.set("a", [PredictionResult(label="a", confidence=0.1)]))
        self._run(cache.set("b", [PredictionResult(label="b", confidence=0.2)]))
        assert self._run(cache.size()) == 2

    def test_cache_key_is_deterministic(self):
        img = make_image_bytes()
        assert image_cache_key(img) == image_cache_key(img)

    def test_different_images_have_different_keys(self):
        img1 = make_image_bytes(width=100)
        img2 = make_image_bytes(width=200)
        assert image_cache_key(img1) != image_cache_key(img2)


# ── Rate limiter ──────────────────────────────────────────────────────────────

class TestRateLimiter:

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_allows_requests_within_limit(self):
        limiter = _SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            allowed, _ = self._run(limiter.is_allowed("user1"))
            assert allowed is True

    def test_blocks_when_limit_exceeded(self):
        limiter = _SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            self._run(limiter.is_allowed("user2"))
        allowed, remaining = self._run(limiter.is_allowed("user2"))
        assert allowed is False
        assert remaining == 0

    def test_different_identities_are_independent(self):
        limiter = _SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        self._run(limiter.is_allowed("alice"))
        allowed_alice, _ = self._run(limiter.is_allowed("alice"))
        allowed_bob, _ = self._run(limiter.is_allowed("bob"))
        assert allowed_alice is False
        assert allowed_bob is True

    def test_remaining_decrements(self):
        limiter = _SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
        _, r1 = self._run(limiter.is_allowed("u"))
        _, r2 = self._run(limiter.is_allowed("u"))
        assert r2 < r1
