from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


def _prepare_device(preference: str = "auto") -> torch.device:
    normalized = (preference or "auto").lower()
    if normalized not in {"auto", "cpu", "gpu"}:
        logger.warning("Unknown device preference '%s', defaulting to auto", preference)
        normalized = "auto"
    if normalized == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("GPU requested but not available, falling back to CPU")
    if normalized == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _select_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():  # pragma: no cover - depends on hardware
            return torch.bfloat16
    return torch.float16


class RerankModel:
    """Cross-encoder rerank model wrapper."""

    def __init__(
        self,
        model_name: str,
        *,
        device_preference: str = "auto",
        mixed_precision: bool = True,
        enable_quantization: bool = False,
    ):
        self.model_name = model_name
        self.device = _prepare_device(device_preference)
        self.use_mixed_precision = mixed_precision and self.device.type == "cuda"
        self.enable_quantization = enable_quantization
        self.autocast_dtype = _select_autocast_dtype(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if self.enable_quantization and self.device.type == "cpu":
            self._maybe_quantize()
        self.model.eval()
        self.model.to(self.device)

    def _maybe_quantize(self) -> None:
        try:
            from transformers import BitsAndBytesConfig

            config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, quantization_config=config, device_map="cpu"
            )
            logger.info("Loaded %s with bitsandbytes 8-bit quantization on CPU", self.model_name)
            return
        except Exception as exc:  # pragma: no cover - optional dependency path
            logger.warning("bitsandbytes quantization unavailable, falling back to CPU-safe quantization: %s", exc)

        try:
            # Torch dynamic quantization works without extra dependencies and is CPU-friendly.
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            logger.info("Applied torch dynamic quantization for %s on CPU", self.model_name)
        except Exception as exc:  # pragma: no cover - configuration dependent
            logger.warning("Quantization requested but failed, continuing without quantization: %s", exc)

    @torch.inference_mode()
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        pairs = [(query, doc) for doc in documents]
        scores = self.score_pairs(pairs)
        if isinstance(scores, Iterable):
            return [float(score) for score in scores]
        return [float(scores)]

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        inputs = self.tokenizer(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        ctx = (
            torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
            if self.use_mixed_precision
            else nullcontext()
        )
        with ctx:
            logits = self.model(**inputs).logits
        if logits.ndim == 1:
            scores_tensor = logits
        elif logits.shape[-1] == 1:
            scores_tensor = logits.squeeze(-1)
        else:
            scores_tensor = logits[:, 0]
        scores = scores_tensor.detach().cpu().tolist()
        if isinstance(scores, float):
            return [float(scores)]
        return [float(score) for score in scores]

    def warmup(self) -> None:
        self.score("warmup", ["warmup document"])
