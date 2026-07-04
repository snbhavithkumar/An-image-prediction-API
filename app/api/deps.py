"""
FastAPI dependencies — the single source of truth for cross-cutting concerns.

Why dependency injection?
─────────────────────────
• Testability: swap any dep in tests via app.dependency_overrides.
• Composability: stack auth + rate-limit + model-ready as a chain.
• Clarity: each endpoint declares exactly what it needs.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from app.core.security import rate_limit, verify_api_key
from app.services.ml_engine import MLEngine, ml_engine


# ── Model availability ────────────────────────────────────────────────────────

def get_ml_engine() -> MLEngine:
    """
    Provides the shared MLEngine singleton.
    Returns HTTP 503 if the model hasn't finished loading yet —
    this protects against requests arriving during a slow startup.
    """
    if not ml_engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Please retry in a few seconds.",
            headers={"Retry-After": "5"},
        )
    return ml_engine


# ── Combined auth + rate-limit guard ─────────────────────────────────────────
# This is the dependency most endpoints should use.
# It chains: API-key verification → sliding-window rate limit.

async def require_auth_and_rate_limit(
    request: Request,
    api_key: Optional[str] = Depends(verify_api_key),
    _rl: None = Depends(rate_limit),
) -> Optional[str]:
    """Returns the validated API key (or None if auth is disabled)."""
    return api_key
