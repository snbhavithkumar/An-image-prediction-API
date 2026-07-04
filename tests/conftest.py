"""
Shared pytest fixtures.

Design principles:
- Tests never download model weights (no GPU/internet required in CI).
- The mock engine mirrors the real MLEngine interface exactly.
- Auth and rate-limiting are disabled by default; individual tests opt in.
- Image helpers produce minimal valid files for each supported format.
"""
from __future__ import annotations

import io
from typing import AsyncGenerator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from PIL import Image

from app.schemas.prediction import PredictionResult

# ── Image helpers ─────────────────────────────────────────────────────────────

def make_image_bytes(width: int = 224, height: int = 224, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(100, 149, 237)).save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


def make_corrupt_bytes() -> bytes:
    return b"this is definitely not an image"


def make_oversized_bytes() -> bytes:
    # Repeat a real image until it exceeds MAX_IMAGE_SIZE_MB
    base = make_image_bytes(width=512, height=512)
    return base * 25  # ~25× ensures it's > 10 MB


# ── Mock ML engine ────────────────────────────────────────────────────────────

DEFAULT_PREDICTIONS: List[PredictionResult] = [
    PredictionResult(label="golden_retriever",       confidence=0.9423),
    PredictionResult(label="labrador_retriever",     confidence=0.0341),
    PredictionResult(label="cocker_spaniel",         confidence=0.0098),
    PredictionResult(label="flat_coated_retriever",  confidence=0.0072),
    PredictionResult(label="curly_coated_retriever", confidence=0.0051),
]


@pytest.fixture()
def mock_engine():
    """
    A MagicMock that behaves like a fully-loaded MLEngine.
    `predict` is an AsyncMock so it can be awaited.
    """
    engine = MagicMock()
    engine.is_loaded = True
    engine.labels = [f"class_{i}" for i in range(1000)]
    engine.predict = AsyncMock(return_value=DEFAULT_PREDICTIONS)
    engine.predict_batch = AsyncMock(return_value=[DEFAULT_PREDICTIONS, DEFAULT_PREDICTIONS])
    engine.version = MagicMock(
        name="vit_b_16",
        architecture="Vision Transformer (ViT-B/16)",
        weights_sha256=None,
        custom_weights=False,
        num_classes=1000,
        device="cpu",
    )
    return engine


# ── Test client ───────────────────────────────────────────────────────────────

@pytest.fixture()
def client(mock_engine):
    """
    FastAPI TestClient with:
    - Real ML engine replaced by mock (no GPU/weights needed).
    - Cache disabled so each test gets fresh inference.
    - Auth and rate limiting disabled.
    - Lifespan skipped (no model download on startup).
    """
    from app.api.deps import get_ml_engine
    from app.core.config import settings
    from app.main import app

    app.dependency_overrides[get_ml_engine] = lambda: mock_engine

    # Patch the module-level singleton so health check and middleware see it
    with (
        patch("app.api.v1.endpoints.ml_engine", mock_engine),
        patch("app.main.ml_engine", mock_engine),
        patch.object(settings, "CACHE_ENABLED", False),
        patch.object(settings, "RATE_LIMIT_ENABLED", False),
        patch.object(settings, "API_KEYS", []),
    ):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    app.dependency_overrides.clear()


@pytest.fixture()
def authed_client(mock_engine):
    """Client with a valid API key pre-set in headers."""
    from app.api.deps import get_ml_engine
    from app.core.config import settings
    from app.main import app

    app.dependency_overrides[get_ml_engine] = lambda: mock_engine

    with (
        patch("app.api.v1.endpoints.ml_engine", mock_engine),
        patch("app.main.ml_engine", mock_engine),
        patch.object(settings, "CACHE_ENABLED", False),
        patch.object(settings, "RATE_LIMIT_ENABLED", False),
        patch.object(settings, "API_KEYS", ["test-key-abc123"]),
    ):
        with TestClient(
            app,
            raise_server_exceptions=True,
            headers={"X-API-Key": "test-key-abc123"},
        ) as c:
            yield c

    app.dependency_overrides.clear()


# ── Image fixtures ────────────────────────────────────────────────────────────

@pytest.fixture()
def jpeg_bytes() -> bytes:
    return make_image_bytes(fmt="JPEG")


@pytest.fixture()
def png_bytes() -> bytes:
    return make_image_bytes(fmt="PNG")


@pytest.fixture()
def webp_bytes() -> bytes:
    return make_image_bytes(fmt="WEBP")
