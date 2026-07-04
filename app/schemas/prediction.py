from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Core building blocks ──────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """A single class label with its probability."""
    label: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability (0–1)")


# ── Single-image prediction ───────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Response for POST /predict (single image)."""
    request_id: Optional[str] = Field(None, description="Echo of X-Request-ID header")
    top_prediction: str = Field(..., description="Highest-confidence label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_k: List[PredictionResult] = Field(..., description="Top-K predictions, highest first")
    cached: bool = Field(False, description="True when result was served from cache")
    inference_ms: Optional[float] = Field(None, description="Model forward-pass time (ms); null on cache hit")

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "top_prediction": "golden_retriever",
                "confidence": 0.9423,
                "top_k": [
                    {"label": "golden_retriever", "confidence": 0.9423},
                    {"label": "labrador_retriever", "confidence": 0.0341},
                ],
                "cached": False,
                "inference_ms": 38.4,
            }
        }
    }


# ── Batch prediction ──────────────────────────────────────────────────────────

class BatchPredictionItem(BaseModel):
    """Result for one image within a batch."""
    filename: str
    top_prediction: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_k: List[PredictionResult]
    error: Optional[str] = Field(None, description="Per-image error message; null on success")


class BatchPredictionResponse(BaseModel):
    """Response for POST /predict/batch."""
    request_id: Optional[str] = None
    results: List[BatchPredictionItem]
    total: int = Field(..., description="Total images submitted")
    succeeded: int = Field(..., description="Images successfully classified")
    failed: int = Field(..., description="Images that produced errors")
    inference_ms: float = Field(..., description="Total batch inference time (ms)")


# ── Model information ─────────────────────────────────────────────────────────

class ModelInfoResponse(BaseModel):
    """Response for GET /model/info."""
    name: str
    architecture: str
    input_size: int
    num_classes: int
    top_k: int
    device: str
    custom_weights: bool = Field(..., description="True if custom weights were loaded")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "vit_b_16",
                "architecture": "Vision Transformer (ViT-B/16)",
                "input_size": 224,
                "num_classes": 1000,
                "top_k": 5,
                "device": "cpu",
                "custom_weights": False,
            }
        }
    }


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    environment: str
    cache_size: Optional[int] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0",
                "environment": "production",
                "cache_size": 42,
            }
        }
    }


# ── Errors ────────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    request_id: Optional[str] = None
    error: ErrorDetail

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "error": {
                    "code": "UNSUPPORTED_MEDIA_TYPE",
                    "message": "Unsupported file type 'application/pdf'.",
                },
            }
        }
    }
