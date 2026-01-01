from __future__ import annotations

from .model import RerankModel


def rerank_documents(model: RerankModel, query: str, documents: list[str], top_k: int) -> list[dict]:
    scores = model.score(query, documents)
    scored = [
        {"index": idx, "document": document, "relevance_score": float(score)}
        for idx, (document, score) in enumerate(zip(documents, scores))
    ]
    scored.sort(key=lambda item: item["relevance_score"], reverse=True)
    return scored[:top_k]
