from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _prepare_device() -> torch.device:
    return torch.device("cpu")


class RerankModel:
    """Cross-encoder rerank model wrapper."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = _prepare_device()
        self.model.to(self.device)

    @torch.inference_mode()
    def score(self, query: str, documents: list[str]) -> list[float]:
        pairs = [(query, doc) for doc in documents]
        inputs = self.tokenizer(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
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
