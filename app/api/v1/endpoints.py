"""
API v1 endpoints
────────────────
GET  /health          — liveness + readiness
GET  /metrics         — Prometheus scrape endpoint
GET  /model/info      — model version & architecture metadata
POST /predict         — single-image classification
POST /predict/batch   — multi-image classification (up to BATCH_MAX_SIZE)
"""
from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import Response

from app.api.deps import get_ml_engine, require_auth_and_rate_limit
from app.core.cache import image_cache_key, prediction_cache
from app.core.config import settings
from app.core.logging import get_logger, request_id_ctx
from app.core.metrics import metrics_output
from app.schemas.prediction import (
    BatchPredictionItem,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from app.services.ml_engine import MLEngine

logger = get_logger("endpoints")
router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
MAX_BYTES = settings.MAX_IMAGE_SIZE_MB * 1_048_576

_COMMON_ERRORS = {
    401: {"model": ErrorResponse, "description": "Invalid or missing API key"},
    413: {"model": ErrorResponse, "description": "File exceeds size limit"},
    415: {"model": ErrorResponse, "description": "Unsupported media type"},
    422: {"model": ErrorResponse, "description": "Image could not be decoded"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    503: {"model": ErrorResponse, "description": "Model not ready"},
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile, data: bytes) -> None:
    """Raise appropriate HTTP errors for bad uploads."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Accepted: {', '.join(sorted(ALLOWED_TYPES))}",
        )
    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {len(data) / 1_048_576:.1f} MB exceeds the {settings.MAX_IMAGE_SIZE_MB} MB limit.",
        )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Observability"],
    # Health checks must NOT require auth so load-balancers can reach them.
)
async def health_check() -> HealthResponse:
    """
    Liveness + readiness probe in one endpoint.
    `model_loaded: false` → the pod is live but not yet ready for traffic.
    """
    cache_size = await prediction_cache.size() if settings.CACHE_ENABLED else None
    return HealthResponse(
        status="healthy",
        model_loaded=ml_engine.is_loaded,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        cache_size=cache_size,
    )


# ── GET /metrics ──────────────────────────────────────────────────────────────

@router.get(
    "/metrics",
    summary="Prometheus metrics",
    tags=["Observability"],
    include_in_schema=not settings.is_production,  # Hide from public docs in prod
)
async def prometheus_metrics() -> Response:
    """Prometheus scrape endpoint. Point your Prometheus server here."""
    body, content_type = metrics_output()
    return Response(content=body, media_type=content_type)


# ── GET /model/info ───────────────────────────────────────────────────────────

@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    tags=["Model"],
    dependencies=[Depends(require_auth_and_rate_limit)],
)
async def model_info(engine: MLEngine = Depends(get_ml_engine)) -> ModelInfoResponse:
    """Returns architecture, device, weight fingerprint, and class count."""
    v = engine.version
    return ModelInfoResponse(
        name=v.name if v else settings.MODEL_NAME,
        architecture=v.architecture if v else "Vision Transformer (ViT-B/16)",
        input_size=settings.MODEL_INPUT_SIZE,
        num_classes=len(engine.labels),
        top_k=settings.TOP_K_PREDICTIONS,
        device=v.device if v else "unknown",
        custom_weights=v.custom_weights if v else False,
    )


# ── POST /predict ─────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify an image",
    description=(
        "Upload a JPEG, PNG, WebP, BMP, or TIFF image. "
        "Returns the top predicted class and ranked top-K list. "
        "Results are cached by image hash — identical uploads return instantly."
    ),
    responses=_COMMON_ERRORS,
    tags=["Prediction"],
    dependencies=[Depends(require_auth_and_rate_limit)],
)
async def predict_image(
    request: Request,
    file: UploadFile = File(..., description="Image to classify"),
    engine: MLEngine = Depends(get_ml_engine),
) -> PredictionResponse:
    request_id = request_id_ctx.get()
    image_bytes = await file.read()
    _validate_upload(file, image_bytes)

    # ── Cache lookup ──────────────────────────────────────────────────────────
    if settings.CACHE_ENABLED:
        cache_key = image_cache_key(image_bytes)
        cached = await prediction_cache.get(cache_key)
        if cached is not None:
            logger.info("Cache hit for prediction", extra={"filename": file.filename})
            return PredictionResponse(
                request_id=request_id,
                top_prediction=cached[0].label,
                confidence=cached[0].confidence,
                top_k=cached,
                cached=True,
                inference_ms=None,
            )

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info(
        "Running inference",
        extra={"filename": file.filename, "size_kb": round(len(image_bytes) / 1024, 1)},
    )

    t0 = time.perf_counter()
    try:
        top_k = await engine.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    inference_ms = (time.perf_counter() - t0) * 1000

    # ── Cache store ───────────────────────────────────────────────────────────
    if settings.CACHE_ENABLED:
        await prediction_cache.set(cache_key, top_k)

    return PredictionResponse(
        request_id=request_id,
        top_prediction=top_k[0].label,
        confidence=top_k[0].confidence,
        top_k=top_k,
        cached=False,
        inference_ms=round(inference_ms, 2),
    )


# ── POST /predict/batch ───────────────────────────────────────────────────────

@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Classify multiple images in one request",
    description=(
        f"Upload up to {settings.BATCH_MAX_SIZE} images in a single multipart request. "
        "Images are processed as a single GPU batch for maximum throughput. "
        "Per-image errors are reported inline — the request itself never fails "
        "due to a single bad image."
    ),
    responses=_COMMON_ERRORS,
    tags=["Prediction"],
    dependencies=[Depends(require_auth_and_rate_limit)],
)
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(..., description=f"Up to {settings.BATCH_MAX_SIZE} images"),
    engine: MLEngine = Depends(get_ml_engine),
) -> BatchPredictionResponse:
    request_id = request_id_ctx.get()

    if len(files) > settings.BATCH_MAX_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch size {len(files)} exceeds maximum of {settings.BATCH_MAX_SIZE}.",
        )

    t0 = time.perf_counter()
    results: List[BatchPredictionItem] = []

    for file in files:
        image_bytes = await file.read()

        # Per-item validation (doesn't abort the whole batch)
        try:
            _validate_upload(file, image_bytes)
        except HTTPException as exc:
            results.append(BatchPredictionItem(
                filename=file.filename or "unknown",
                top_prediction="",
                confidence=0.0,
                top_k=[],
                error=exc.detail,
            ))
            continue

        # Cache check per item
        top_k = None
        if settings.CACHE_ENABLED:
            cache_key = image_cache_key(image_bytes)
            top_k = await prediction_cache.get(cache_key)

        if top_k is None:
            try:
                top_k = await engine.predict(image_bytes)
                if settings.CACHE_ENABLED:
                    await prediction_cache.set(cache_key, top_k)
            except (ValueError, Exception) as exc:
                results.append(BatchPredictionItem(
                    filename=file.filename or "unknown",
                    top_prediction="",
                    confidence=0.0,
                    top_k=[],
                    error=str(exc),
                ))
                continue

        results.append(BatchPredictionItem(
            filename=file.filename or "unknown",
            top_prediction=top_k[0].label,
            confidence=top_k[0].confidence,
            top_k=top_k,
            error=None,
        ))

    succeeded = sum(1 for r in results if r.error is None)
    return BatchPredictionResponse(
        request_id=request_id,
        results=results,
        total=len(files),
        succeeded=succeeded,
        failed=len(files) - succeeded,
        inference_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
