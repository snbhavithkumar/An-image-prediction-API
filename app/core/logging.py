"""
Structured JSON logging with request-ID correlation.

Every log line emitted during a request includes the request_id so you can
filter a single request across thousands of lines in Datadog / CloudWatch /
GCP Logging without a trace back-end.
"""
import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any, Dict, Optional

from app.core.config import settings

# ── Request-ID context var ────────────────────────────────────────────────────
# Set by RequestContextMiddleware; read by the JSON formatter below.
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """
    Emits one JSON object per log line — machine-readable, grep-friendly,
    and compatible with every major log aggregation platform.
    """

    RESERVED = {"message", "timestamp", "level", "logger", "request_id"}

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self._iso(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
            "environment": settings.ENVIRONMENT,
            "version": settings.VERSION,
        }

        # Merge any extra= kwargs the caller passed
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in self.RESERVED:
                payload[key] = val

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)

    @staticmethod
    def _iso(ts: float) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root.handlers.clear()
    root.addHandler(handler)

    # Quieten chatty third-party loggers
    for noisy in ("uvicorn.access", "PIL", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("image_prediction_api").info(
        "Logging initialised", extra={"log_level": settings.LOG_LEVEL}
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"image_prediction_api.{name}")
