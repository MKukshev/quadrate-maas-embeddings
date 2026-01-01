from __future__ import annotations

import inspect
from typing import Awaitable, Callable

ScoreProvider = Callable[[str, list[str]], Awaitable[list[float]] | list[float]]


async def rerank_documents(score_provider: ScoreProvider, query: str, documents: list[str], top_k: int) -> list[dict]:
    scores_or_awaitable = score_provider(query, documents)
    scores = await scores_or_awaitable if inspect.isawaitable(scores_or_awaitable) else scores_or_awaitable
    scored = [
        {"index": idx, "document": document, "relevance_score": float(score)}
        for idx, (document, score) in enumerate(zip(documents, scores))
    ]
    scored.sort(key=lambda item: item["relevance_score"], reverse=True)
    return scored[:top_k]
