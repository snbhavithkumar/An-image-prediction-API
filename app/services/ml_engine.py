"""
MLEngine — production-grade Vision Transformer inference service.

Key features
────────────
• Async batch queue: requests arriving within BATCH_WAIT_MS are grouped
  into a single GPU forward-pass, dramatically improving throughput.
• Model versioning: tracks name + SHA-256 of the weights file.
• Graceful warm-up: runs one dummy forward-pass after load to prime
  PyTorch's CUDA kernels and eliminate first-request latency spikes.
• Device auto-selection: CUDA → MPS (Apple Silicon) → CPU.
• Structured logging and Prometheus metrics on every inference call.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ViT_B_16_Weights

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import INFERENCE_BATCH_SIZE, MODEL_LOAD_DURATION, track_inference_time
from app.schemas.prediction import PredictionResult

logger = get_logger("ml_engine")

# ── Pre-processing pipeline ───────────────────────────────────────────────────
# Matches the normalisation used during ViT-B/16 ImageNet pre-training.
_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(settings.MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ── Batch queue item ──────────────────────────────────────────────────────────

@dataclass
class _InferenceRequest:
    tensor: torch.Tensor                            # (1, 3, 224, 224)
    future: asyncio.Future                          # resolved with List[PredictionResult]


# ── Model versioning ──────────────────────────────────────────────────────────

@dataclass
class ModelVersion:
    name: str
    architecture: str
    weights_sha256: Optional[str]
    custom_weights: bool
    num_classes: int
    device: str


# ── Engine ────────────────────────────────────────────────────────────────────

class MLEngine:
    """
    ML Inference Engine - Handles all image classification
    
    This is the core of the image prediction system:
    • Loads the Vision Transformer model once at startup
    • Accepts image classification requests
    • Batches requests arriving within 20ms together for GPU efficiency
    • Returns predictions with confidence scores
    
    Design pattern: Singleton (only one instance used app-wide)
    Thread safety: Uses asyncio (event loop is single-threaded)
    Background worker: Runs as a continuous asyncio Task
    """

    def __init__(self) -> None:
        self._model: Optional[torch.nn.Module] = None
        self._labels: List[str] = []
        self._version: Optional[ModelVersion] = None
        self._is_loaded = False
        self._device = self._select_device()

        # Async batch queue
        self._queue: asyncio.Queue[_InferenceRequest] = asyncio.Queue()
        self._batch_worker_task: Optional[asyncio.Task] = None

    # ── Device selection ──────────────────────────────────────────────────────

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info("Selected inference device", extra={"device": str(device)})
        return device

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def version(self) -> Optional[ModelVersion]:
        return self._version

    @property
    def labels(self) -> List[str]:
        return self._labels

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load weights and class labels. Called once at startup by the lifespan hook.
        Blocks intentionally — the API must not accept traffic before the model is ready.
        """
        t0 = time.perf_counter()
        logger.info("Loading ML model", extra={"device": str(self._device)})

        weights_sha256: Optional[str] = None
        custom_weights = False
        custom_path = Path(settings.MODEL_PATH)

        if custom_path.exists():
            logger.info("Loading custom weights", extra={"path": str(custom_path)})
            weights_sha256 = self._sha256_file(custom_path)
            self._model = models.vit_b_16()
            state_dict = torch.load(custom_path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
            custom_weights = True
        else:
            logger.info("No custom weights found — using pre-trained ViT-B/16 (ImageNet)")
            try:
                pretrained_weights = ViT_B_16_Weights.IMAGENET1K_V1
                self._model = models.vit_b_16(weights=pretrained_weights)
            except Exception as e:
                logger.warning(
                    f"Could not load pre-trained weights: {e}. Loading model without pre-training.",
                    extra={"error": str(e)}
                )
                # Load model without pre-training for development/testing
                self._model = models.vit_b_16(weights=None)

        self._model.to(self._device)
        self._model.eval()

        # Load class labels from torchvision metadata
        self._labels = ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"]

        self._version = ModelVersion(
            name=settings.MODEL_NAME,
            architecture="Vision Transformer (ViT-B/16)",
            weights_sha256=weights_sha256,
            custom_weights=custom_weights,
            num_classes=len(self._labels),
            device=str(self._device),
        )

        elapsed = time.perf_counter() - t0
        MODEL_LOAD_DURATION.set(elapsed)  # type: ignore[union-attr]
        logger.info(
            "Model loaded",
            extra={
                "elapsed_s": round(elapsed, 2),
                "classes": len(self._labels),
                "custom_weights": custom_weights,
            },
        )

        self._warm_up()
        self._is_loaded = True

    def _warm_up(self) -> None:
        """
        Run one dummy forward-pass to prime CUDA kernels.
        Prevents the first real request from being anomalously slow.
        """
        logger.info("Running model warm-up pass")
        dummy = torch.zeros(1, 3, settings.MODEL_INPUT_SIZE, settings.MODEL_INPUT_SIZE, device=self._device)
        with torch.no_grad():
            _ = self._model(dummy)  # type: ignore[misc]
        logger.info("Model warm-up complete")

    def start_batch_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        """Launch the background batch-inference worker. Call from lifespan."""
        self._batch_worker_task = loop.create_task(self._batch_worker())
        logger.info("Batch inference worker started")

    async def shutdown(self) -> None:
        """Cancel the batch worker and release model memory."""
        if self._batch_worker_task:
            self._batch_worker_task.cancel()
            try:
                await self._batch_worker_task
            except asyncio.CancelledError:
                pass
        self._model = None
        self._is_loaded = False
        logger.info("ML engine shut down")

    # ── Async batch worker ────────────────────────────────────────────────────

    async def _batch_worker(self) -> None:
        """
        Continuously drain the queue in micro-batches.

        Waits up to BATCH_WAIT_MS for a first item, then collects any
        additional items that arrived during that window (up to BATCH_MAX_SIZE),
        runs them as a single batched forward-pass, and resolves all futures.
        """
        logger.info("Batch worker running", extra={
            "max_batch": settings.BATCH_MAX_SIZE,
            "wait_ms": settings.BATCH_WAIT_MS,
        })

        while True:
            batch: List[_InferenceRequest] = []

            # Block until at least one item arrives
            try:
                first = await self._queue.get()
                batch.append(first)
            except asyncio.CancelledError:
                return

            # Collect additional items up to the window
            deadline = time.monotonic() + settings.BATCH_WAIT_MS / 1000
            while len(batch) < settings.BATCH_MAX_SIZE:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            await self._process_batch(batch)

    async def _process_batch(self, batch: List[_InferenceRequest]) -> None:
        """Run the batch through the model and resolve each future."""
        INFERENCE_BATCH_SIZE.observe(len(batch))  # type: ignore[union-attr]

        try:
            # Stack tensors → (B, 3, 224, 224)
            stacked = torch.cat([req.tensor for req in batch], dim=0).to(self._device)

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._forward, stacked)

            for req, top_k in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(top_k)

        except Exception as exc:
            logger.exception("Batch inference failed", exc_info=exc)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)

    def _forward(self, batch_tensor: torch.Tensor) -> List[List[PredictionResult]]:
        """
        Synchronous forward pass — runs in a thread-pool executor so it
        doesn't block the asyncio event loop.
        """
        with track_inference_time():
            with torch.no_grad():
                logits = self._model(batch_tensor)          # (B, C)
                probs = F.softmax(logits, dim=1)            # (B, C)

        k = settings.TOP_K_PREDICTIONS
        top_k_probs, top_k_idx = torch.topk(probs, k=k, dim=1)

        all_results: List[List[PredictionResult]] = []
        for probs_row, idx_row in zip(top_k_probs.cpu().tolist(), top_k_idx.cpu().tolist()):
            all_results.append([
                PredictionResult(label=self._labels[i], confidence=round(p, 6))
                for p, i in zip(probs_row, idx_row)
            ])
        return all_results

    # ── Public inference API ──────────────────────────────────────────────────

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """
        Decode raw bytes → normalised float tensor (1, 3, 224, 224).
        Raises ValueError for corrupt/non-image data.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Could not decode image: {exc}") from exc
        return _TRANSFORM(image).unsqueeze(0)

    async def predict(self, image_bytes: bytes) -> List[PredictionResult]:
        """
        Submit one image for async batched inference. Returns top-K results.

        Raises:
            RuntimeError  — engine not loaded
            ValueError    — corrupt image
            asyncio.TimeoutError — inference didn't complete in time
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call MLEngine.load() first.")

        tensor = self.preprocess(image_bytes)

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(_InferenceRequest(tensor=tensor, future=future))

        return await asyncio.wait_for(future, timeout=settings.INFERENCE_TIMEOUT_SECONDS)

    async def predict_batch(self, images: List[bytes]) -> List[List[PredictionResult]]:
        """Submit multiple images concurrently; results preserve input order."""
        tasks = [self.predict(img) for img in images]
        return await asyncio.gather(*tasks, return_exceptions=False)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# ── Module-level singleton ────────────────────────────────────────────────────
ml_engine = MLEngine()
