"""
Application factory and lifespan.

Startup sequence
────────────────
1. Configure structured logging.
2. Load ML model weights into memory (blocks until done).
3. Start the async batch-inference worker.
4. (Optional) Initialise Sentry for error tracking.

Shutdown sequence
─────────────────
1. Drain and cancel the batch worker.
2. Release model from memory.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.middleware.request_context import RequestContextMiddleware
from app.services.ml_engine import ml_engine

configure_logging()
logger = get_logger("main")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info(
        "Starting %s v%s [%s]",
        settings.APP_NAME,
        settings.VERSION,
        settings.ENVIRONMENT,
    )

    # Optional Sentry integration
    if settings.SENTRY_DSN:
        try:
            import sentry_sdk
            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                environment=settings.ENVIRONMENT,
                release=settings.VERSION,
                traces_sample_rate=0.1,
            )
            logger.info("Sentry initialised")
        except ImportError:
            logger.warning("sentry-sdk not installed; skipping Sentry init")

    # Load model (blocking — intentional)
    ml_engine.load()

    # Start the async batch worker
    loop = asyncio.get_event_loop()
    ml_engine.start_batch_worker(loop)

    logger.info("API is ready to serve traffic")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down — draining batch worker")
    await ml_engine.shutdown()
    logger.info("Shutdown complete")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description=(
            "## Image Prediction API\n\n"
            "Production-grade image classification powered by **Vision Transformer (ViT-B/16)**.\n\n"
            "### Key capabilities\n"
            "- Single and batch image classification\n"
            "- SHA-256 content-addressed result caching\n"
            "- Async micro-batch GPU inference queue\n"
            "- API-key auth + sliding-window rate limiting\n"
            "- Prometheus metrics at `/api/v1/metrics`\n"
            "- Structured JSON logging with request-ID correlation\n"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        # In production, restrict docs to internal networks via a reverse proxy
        # rather than disabling them here.
    )

    # ── Middleware (outermost → innermost) ────────────────────────────────────
    # Order matters: RequestContext must wrap everything so request-IDs are
    # available in all subsequent middleware and handlers.

    app.add_middleware(RequestContextMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred. Please try again.",
                }
            },
        )

    return app


app = create_app()
