from __future__ import annotations

from typing import Callable

import pytest
from fastapi.testclient import TestClient

from services.rerank.src.main import create_app


class FakeModel:
    def __init__(self, scores: list[float] | Callable[[list[str]], list[float]] = None):
        self.scores = scores or []

    def warmup(self) -> None:  # pragma: no cover - no-op for tests
        return None

    def score(self, query: str, documents: list[str]) -> list[float]:
        if callable(self.scores):
            return self.scores(documents)
        if len(self.scores) >= len(documents):
            return self.scores[: len(documents)]
        return list(range(len(documents)))


def override_app(fake: FakeModel) -> TestClient:
    app = create_app(model_loader=lambda settings: fake)
    return TestClient(app)


def test_document_limits(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_MAX_DOCUMENTS", "1")
    monkeypatch.setenv("RERANK_MAX_DOCUMENT_LENGTH", "5")

    client = override_app(FakeModel())
    response = client.post(
        "/v1/rerank",
        json={"model": "demo", "query": "hello", "documents": ["short", "extra"]},
    )
    assert response.status_code == 400

    response = client.post(
        "/v1/rerank",
        json={"model": "demo", "query": "hello", "documents": ["toolong"]},
    )
    assert response.status_code == 400


def test_empty_documents_validation(monkeypatch: pytest.MonkeyPatch):
    client = override_app(FakeModel())
    response = client.post(
        "/v1/rerank",
        json={"model": "demo", "query": "hello", "documents": []},
    )
    assert response.status_code == 400


def test_top_k_sorting(monkeypatch: pytest.MonkeyPatch):
    fake = FakeModel(scores=[0.1, 0.9, 0.5])
    client = override_app(fake)
    response = client.post(
        "/v1/rerank",
        json={"model": "demo", "query": "hello", "documents": ["a", "b", "c"], "top_k": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["document"] == "b"
    assert body["results"][0]["relevance_score"] == pytest.approx(0.9)
    assert len(body["results"]) == 2


def test_top_k_defaults(monkeypatch: pytest.MonkeyPatch):
    fake = FakeModel(scores=[0.2, 0.1])
    client = override_app(fake)
    response = client.post(
        "/v1/rerank",
        json={"model": "demo", "query": "hello", "documents": ["first", "second"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["document"] == "first"
