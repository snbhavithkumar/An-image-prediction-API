# Image Prediction API

> Production-grade image classification REST API built with **FastAPI**, **PyTorch**, and **Vision Transformer (ViT-B/16)**.

[![CI](https://github.com/your-username/image-prediction-api/actions/workflows/deploy.yml/badge.svg)](https://github.com/your-username/image-prediction-api/actions)
[![Coverage](https://codecov.io/gh/your-username/image-prediction-api/branch/main/graph/badge.svg)](https://codecov.io/gh/your-username/image-prediction-api)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

---

## What makes this production-grade

| Concern | Implementation |
|---|---|
| **ML Inference** | ViT-B/16 with async micro-batch queue; requests within 20 ms window share one GPU forward-pass |
| **Caching** | SHA-256 content-addressed LRU result cache; identical images return in < 1 ms |
| **Auth** | API-key authentication via `X-API-Key` header; zero config = auth disabled (dev) |
| **Rate limiting** | Per-identity sliding-window limiter with standard `X-RateLimit-*` headers |
| **Observability** | Prometheus metrics, structured JSON logging with request-ID correlation |
| **Error tracking** | Sentry integration (optional; set `SENTRY_DSN`) |
| **Testing** | 40+ tests; mocked engine so tests need no GPU and run in seconds |
| **Docker** | Multi-stage build, non-root user, HEALTHCHECK, Trivy vulnerability scan in CI |
| **CI/CD** | lint → type-check → test (multi-Python matrix) → docker build → deploy |

---

## API endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/health` | ✗ | Liveness + readiness probe |
| `GET` | `/api/v1/metrics` | ✗ | Prometheus scrape endpoint |
| `GET` | `/api/v1/model/info` | ✓ | Architecture, device, class count |
| `POST` | `/api/v1/predict` | ✓ | Classify one image |
| `POST` | `/api/v1/predict/batch` | ✓ | Classify up to 8 images in one request |

Interactive docs → `http://localhost:8000/docs`

---

## Quick start

```bash
git clone https://github.com/your-username/image-prediction-api.git
cd image-prediction-api

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
uvicorn app.main:app --reload
```

---

## Usage examples

### Classify an image

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: your-key" \
  -F "file=@dog.jpg"
```

```json
{
  "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "top_prediction": "golden_retriever",
  "confidence": 0.9423,
  "top_k": [
    { "label": "golden_retriever",       "confidence": 0.9423 },
    { "label": "labrador_retriever",     "confidence": 0.0341 },
    { "label": "cocker_spaniel",         "confidence": 0.0098 },
    { "label": "flat_coated_retriever",  "confidence": 0.0072 },
    { "label": "curly_coated_retriever", "confidence": 0.0051 }
  ],
  "cached": false,
  "inference_ms": 38.4
}
```

### Classify a batch

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "X-API-Key: your-key" \
  -F "files=@dog.jpg" \
  -F "files=@cat.jpg"
```

### Python client

```python
import httpx

client = httpx.Client(
    base_url="http://localhost:8000",
    headers={"X-API-Key": "your-key"},
)

with open("dog.jpg", "rb") as f:
    r = client.post("/api/v1/predict", files={"file": ("dog.jpg", f, "image/jpeg")})

result = r.json()
print(f"{result['top_prediction']} ({result['confidence']:.1%})")
```

---

## Architecture deep-dive

### Async batch inference queue

Instead of running one model forward-pass per HTTP request, incoming requests
are enqueued. A background worker drains the queue in micro-batches:

```
Request A ──┐
Request B ──┼──► [queue] ──► batch worker ──► GPU forward-pass (A+B+C) ──► resolve futures
Request C ──┘
```

Configurable via `BATCH_MAX_SIZE` and `BATCH_WAIT_MS`. Single-instance
throughput increases proportionally with batch size.

### Result cache

Every image is hashed (SHA-256) before inference. Cache lookups are O(1) and
return sub-millisecond. The cache is swap-friendly: replace
`prediction_cache` in `app/core/cache.py` with a Redis backend for
multi-instance deployments.

### Request-ID correlation

Every request gets a UUID assigned in middleware. It is propagated through
every log line via a `contextvars.ContextVar`, returned in `X-Request-ID`,
and echoed in the response body — making incident investigation trivial.

### Model versioning

On startup, the engine records the SHA-256 of the weights file. The
`GET /model/info` endpoint exposes this, making it easy to verify which
exact weights are serving traffic.

---

## Docker

```bash
# Build
docker build -t image-prediction-api .

# Run
docker run -p 8000:8000 --env-file .env image-prediction-api

# Full stack (API + Prometheus + Grafana)
docker compose up
```

Grafana → `http://localhost:3000` (admin / admin)
Prometheus → `http://localhost:9090`

---

## Running tests

```bash
# All tests
pytest

# With coverage report
pytest --cov=app --cov-report=term-missing

# Specific file
pytest tests/test_engine.py -v
```

---

## Using custom model weights

1. Fine-tune a ViT-B/16 model in PyTorch.
2. Save the state dict:
   ```python
   torch.save(model.state_dict(), "app/models/best_model.pth")
   ```
3. Set `MODEL_PATH=app/models/best_model.pth` in `.env`.
4. Restart. The engine detects and loads your weights automatically.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | `development` / `staging` / `production` |
| `API_KEYS` | `[]` | Comma-separated valid API keys; empty = auth off |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window |
| `MODEL_PATH` | `app/models/best_model.pth` | Custom weights path |
| `TOP_K_PREDICTIONS` | `5` | Classes returned per image |
| `BATCH_MAX_SIZE` | `8` | Images per GPU batch |
| `BATCH_WAIT_MS` | `20` | Batch collection window (ms) |
| `CACHE_ENABLED` | `true` | Enable result cache |
| `CACHE_TTL_SECONDS` | `300` | Cache entry lifetime |
| `METRICS_ENABLED` | `true` | Expose Prometheus metrics |
| `SENTRY_DSN` | `` | Sentry DSN; empty = disabled |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Deployment

The API is stateless and containerised. Recommended platforms:

| Platform | Notes |
|---|---|
| **Google Cloud Run** | `gcloud run deploy`; auto-scales to zero |
| **AWS App Runner** | Push to ECR → create service |
| **Render** | Connect GitHub repo; Docker runtime |
| **Railway** | `railway up` |
| **Azure Container Apps** | `az containerapp up` |

For GPU inference in production, use a VM/node with a CUDA-capable GPU and
set `device=cuda` (auto-detected).

---

## Project structure

```
image-prediction-api/
├── app/
│   ├── main.py                    # App factory + lifespan
│   ├── api/
│   │   ├── deps.py                # Dependency injection (model, auth, rate-limit)
│   │   └── v1/
│   │       ├── router.py
│   │       └── endpoints.py       # /health /metrics /model/info /predict /predict/batch
│   ├── core/
│   │   ├── config.py              # Pydantic settings
│   │   ├── logging.py             # Structured JSON logging + request-ID
│   │   ├── metrics.py             # Prometheus counters, histograms, gauges
│   │   ├── cache.py               # SHA-256 LRU result cache
│   │   └── security.py            # API-key auth + sliding-window rate limiter
│   ├── middleware/
│   │   └── request_context.py     # Request-ID injection, access log, metrics
│   ├── schemas/
│   │   └── prediction.py          # Pydantic v2 request/response models
│   └── services/
│       └── ml_engine.py           # ViT-B/16 async batch inference engine
├── tests/
│   ├── conftest.py                # Fixtures, mock engine, image helpers
│   ├── test_api.py                # Endpoint integration tests (auth, cache, batch)
│   └── test_engine.py             # Unit tests (preprocess, cache, rate-limiter)
├── deploy/
│   ├── prometheus.yml
│   └── grafana/dashboards/
├── .github/workflows/deploy.yml   # lint → typecheck → test → docker → deploy
├── docker-compose.yml             # API + Prometheus + Grafana
├── Dockerfile                     # Multi-stage, non-root, Trivy-scanned
├── pyproject.toml                 # ruff, mypy, pytest config
├── requirements.txt
└── .env.example
```

---

## License

MIT
