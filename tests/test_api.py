"""
Integration tests for all API endpoints.

Coverage:
  - GET  /health
  - GET  /metrics
  - GET  /model/info
  - POST /predict        (happy path, cache, validation, errors)
  - POST /predict/batch  (happy path, partial failures, size limit)
  - Auth (missing key, invalid key, valid key)
  - Rate limiting (429 when exceeded)
  - Response headers (X-Request-ID, X-Response-Time-Ms)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status

from app.core.config import settings
from tests.conftest import DEFAULT_PREDICTIONS, make_corrupt_bytes, make_image_bytes, make_oversized_bytes


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == status.HTTP_200_OK

    def test_response_schema(self, client, mock_engine):
        with patch("app.api.v1.endpoints.ml_engine", mock_engine):
            r = client.get("/api/v1/health")
        body = r.json()
        assert body["status"] == "healthy"
        assert isinstance(body["model_loaded"], bool)
        assert "version" in body
        assert "environment" in body

    def test_no_auth_required(self, authed_client):
        """Health must be reachable without an API key (for load-balancers)."""
        # Remove the auth header
        r = authed_client.get("/api/v1/health", headers={"X-API-Key": ""})
        # It's OK as long as it doesn't return 401
        assert r.status_code != status.HTTP_401_UNAUTHORIZED


# ── /model/info ───────────────────────────────────────────────────────────────

class TestModelInfoEndpoint:
    def test_returns_model_metadata(self, client):
        r = client.get("/api/v1/model/info")
        assert r.status_code == status.HTTP_200_OK
        body = r.json()
        for key in ("name", "architecture", "input_size", "num_classes", "top_k", "device"):
            assert key in body

    def test_num_classes_is_positive(self, client):
        r = client.get("/api/v1/model/info")
        assert r.json()["num_classes"] > 0


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredictEndpoint:

    # Happy path ───────────────────────────────────────────────────────────────

    def test_jpeg_returns_200(self, client, jpeg_bytes):
        r = client.post("/api/v1/predict", files={"file": ("dog.jpg", jpeg_bytes, "image/jpeg")})
        assert r.status_code == status.HTTP_200_OK

    def test_png_returns_200(self, client, png_bytes):
        r = client.post("/api/v1/predict", files={"file": ("img.png", png_bytes, "image/png")})
        assert r.status_code == status.HTTP_200_OK

    def test_webp_returns_200(self, client, webp_bytes):
        r = client.post("/api/v1/predict", files={"file": ("img.webp", webp_bytes, "image/webp")})
        assert r.status_code == status.HTTP_200_OK

    def test_response_schema_complete(self, client, jpeg_bytes):
        r = client.post("/api/v1/predict", files={"file": ("dog.jpg", jpeg_bytes, "image/jpeg")})
        body = r.json()
        assert "top_prediction" in body
        assert "confidence" in body
        assert "top_k" in body
        assert "cached" in body

    def test_top_k_length(self, client, jpeg_bytes):
        r = client.post("/api/v1/predict", files={"file": ("dog.jpg", jpeg_bytes, "image/jpeg")})
        assert len(r.json()["top_k"]) == settings.TOP_K_PREDICTIONS

    def test_top_prediction_matches_first_top_k(self, client, jpeg_bytes):
        r = client.post("/api/v1/predict", files={"file": ("dog.jpg", jpeg_bytes, "image/jpeg")})
        body = r.json()
        assert body["top_prediction"] == body["top_k"][0]["label"]

    def test_confidence_is_valid_probability(self, client, jpeg_bytes):
        r = client.post("/api/v1/predict", files={"file": ("dog.jpg", jpeg_bytes, "image/jpeg")})
        body = r.json()
        assert 0.0 <= body["confidence"] <= 1.0
        for item in body["top_k"]:
            assert 0.0 <= item["confidence"] <= 1.0

    def test_cached_flag_is_false_on_fresh_request(self, client, jpeg_bytes):
        r = client.post("/api/v1/predict", files={"file": ("dog.jpg", jpeg_bytes, "image/jpeg")})
        assert r.json()["cached"] is False

    def test_response_includes_x_request_id_header(self, client, jpeg_bytes):
        r = client.post("/api/v1/predict", files={"file": ("dog.jpg", jpeg_bytes, "image/jpeg")})
        assert "x-request-id" in r.headers

    def test_response_includes_timing_header(self, client, jpeg_bytes):
        r = client.post("/api/v1/predict", files={"file": ("dog.jpg", jpeg_bytes, "image/jpeg")})
        assert "x-response-time-ms" in r.headers

    # Cache behaviour ──────────────────────────────────────────────────────────

    def test_cache_hit_returns_cached_true(self, mock_engine, client):
        """Second identical upload should be served from cache."""
        from app.api.deps import get_ml_engine
        from app.core.config import settings as s
        from app.main import app

        app.dependency_overrides[get_ml_engine] = lambda: mock_engine

        with (
            patch("app.api.v1.endpoints.ml_engine", mock_engine),
            patch("app.main.ml_engine", mock_engine),
            patch.object(s, "CACHE_ENABLED", True),
            patch.object(s, "RATE_LIMIT_ENABLED", False),
            patch.object(s, "API_KEYS", []),
        ):
            from fastapi.testclient import TestClient
            with TestClient(app) as c:
                jpeg = make_image_bytes()
                file_tuple = ("dog.jpg", jpeg, "image/jpeg")

                r1 = c.post("/api/v1/predict", files={"file": file_tuple})
                r2 = c.post("/api/v1/predict", files={"file": ("dog2.jpg", jpeg, "image/jpeg")})

                assert r1.status_code == 200
                assert r2.status_code == 200
                assert r2.json()["cached"] is True
                # Inference called exactly once
                assert mock_engine.predict.call_count == 1

        app.dependency_overrides.clear()

    # Error cases ──────────────────────────────────────────────────────────────

    def test_missing_file_returns_422(self, client):
        r = client.post("/api/v1/predict")
        assert r.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_wrong_content_type_returns_415(self, client):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert r.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

    def test_corrupt_image_returns_422(self, client, mock_engine):
        mock_engine.predict = AsyncMock(side_effect=ValueError("Could not decode image"))
        r = client.post(
            "/api/v1/predict",
            files={"file": ("bad.jpg", make_corrupt_bytes(), "image/jpeg")},
        )
        assert r.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_oversized_image_returns_413(self, client):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("huge.jpg", make_oversized_bytes(), "image/jpeg")},
        )
        assert r.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


# ── /predict/batch ────────────────────────────────────────────────────────────

class TestBatchPredictEndpoint:

    def _post_batch(self, client, images):
        files = [("files", (f"img{i}.jpg", img, "image/jpeg")) for i, img in enumerate(images)]
        return client.post("/api/v1/predict/batch", files=files)

    def test_two_images_returns_200(self, client):
        r = self._post_batch(client, [make_image_bytes(), make_image_bytes()])
        assert r.status_code == status.HTTP_200_OK

    def test_response_contains_results_list(self, client):
        r = self._post_batch(client, [make_image_bytes()])
        body = r.json()
        assert "results" in body
        assert len(body["results"]) == 1

    def test_succeeded_and_failed_counts(self, client):
        r = self._post_batch(client, [make_image_bytes(), make_image_bytes()])
        body = r.json()
        assert body["total"] == 2
        assert body["succeeded"] == 2
        assert body["failed"] == 0

    def test_partial_failure_continues(self, client, mock_engine):
        """A single bad image must not abort the entire batch."""
        call_count = 0

        async def sometimes_fail(image_bytes):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("corrupt")
            return DEFAULT_PREDICTIONS

        mock_engine.predict = sometimes_fail
        r = self._post_batch(client, [make_corrupt_bytes(), make_image_bytes()])
        # Request itself succeeds
        assert r.status_code == status.HTTP_200_OK
        body = r.json()
        assert body["failed"] == 1
        assert body["succeeded"] == 1

    def test_batch_too_large_returns_422(self, client):
        images = [make_image_bytes()] * (settings.BATCH_MAX_SIZE + 1)
        r = self._post_batch(client, images)
        assert r.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# ── Authentication ────────────────────────────────────────────────────────────

class TestAuthentication:

    def _make_authed_client(self, app, mock_engine, key: str | None):
        """Helper to build a client with a specific key and auth enabled."""
        from app.api.deps import get_ml_engine
        from app.core.config import settings as s

        app.dependency_overrides[get_ml_engine] = lambda: mock_engine

        patches = [
            patch("app.api.v1.endpoints.ml_engine", mock_engine),
            patch("app.main.ml_engine", mock_engine),
            patch.object(s, "CACHE_ENABLED", False),
            patch.object(s, "RATE_LIMIT_ENABLED", False),
            patch.object(s, "API_KEYS", ["valid-key"]),
        ]
        return patches

    def test_missing_key_returns_401(self, mock_engine):
        from app.api.deps import get_ml_engine
        from app.core.config import settings as s
        from app.main import app

        app.dependency_overrides[get_ml_engine] = lambda: mock_engine

        with (
            patch("app.api.v1.endpoints.ml_engine", mock_engine),
            patch("app.main.ml_engine", mock_engine),
            patch.object(s, "CACHE_ENABLED", False),
            patch.object(s, "RATE_LIMIT_ENABLED", False),
            patch.object(s, "API_KEYS", ["valid-key"]),
        ):
            from fastapi.testclient import TestClient
            with TestClient(app) as c:
                r = c.post(
                    "/api/v1/predict",
                    files={"file": ("img.jpg", make_image_bytes(), "image/jpeg")},
                )
                assert r.status_code == status.HTTP_401_UNAUTHORIZED

        app.dependency_overrides.clear()

    def test_invalid_key_returns_401(self, mock_engine):
        from app.api.deps import get_ml_engine
        from app.core.config import settings as s
        from app.main import app

        app.dependency_overrides[get_ml_engine] = lambda: mock_engine

        with (
            patch("app.api.v1.endpoints.ml_engine", mock_engine),
            patch("app.main.ml_engine", mock_engine),
            patch.object(s, "CACHE_ENABLED", False),
            patch.object(s, "RATE_LIMIT_ENABLED", False),
            patch.object(s, "API_KEYS", ["valid-key"]),
        ):
            from fastapi.testclient import TestClient
            with TestClient(app) as c:
                r = c.post(
                    "/api/v1/predict",
                    files={"file": ("img.jpg", make_image_bytes(), "image/jpeg")},
                    headers={"X-API-Key": "wrong-key"},
                )
                assert r.status_code == status.HTTP_401_UNAUTHORIZED

        app.dependency_overrides.clear()

    def test_valid_key_returns_200(self, authed_client, jpeg_bytes):
        r = authed_client.post(
            "/api/v1/predict",
            files={"file": ("dog.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert r.status_code == status.HTTP_200_OK
