from functools import lru_cache
from typing import List, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "Image Prediction API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    # ── API ───────────────────────────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ORIGINS: List[str] = ["*"]

    # ── Auth ──────────────────────────────────────────────────────────────────
    API_KEY_HEADER: str = "X-API-Key"
    # Comma-separated list of valid API keys; empty = auth disabled
    API_KEYS: List[str] = []

    # ── Rate limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 60       # requests per window
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL_PATH: str = "app/models/best_model.pth"
    MODEL_NAME: str = "vit_b_16"        # used as model registry key
    MODEL_INPUT_SIZE: int = 224
    TOP_K_PREDICTIONS: int = 5
    INFERENCE_TIMEOUT_SECONDS: float = 30.0

    # ── Async batch inference ─────────────────────────────────────────────────
    BATCH_MAX_SIZE: int = 8             # images per batch
    BATCH_WAIT_MS: int = 20            # max ms to wait before flushing a batch

    # ── Uploads ───────────────────────────────────────────────────────────────
    MAX_IMAGE_SIZE_MB: int = 10

    # ── Cache ─────────────────────────────────────────────────────────────────
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 300        # 5 min result cache per image hash
    CACHE_MAX_SIZE: int = 512           # LRU max entries

    # ── Observability ─────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    METRICS_ENABLED: bool = True
    SENTRY_DSN: str = ""               # empty = Sentry disabled

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.API_KEYS)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton — safe to call anywhere."""
    return Settings()


settings = get_settings()
