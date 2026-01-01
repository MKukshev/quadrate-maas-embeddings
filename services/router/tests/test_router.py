from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from services.router.src.main import create_app
from services.router.src.routing import UpstreamConfig, load_routing_config
from services.router.src.tokenization import TokenizerProvider


def write_yaml(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content))


def setup_configs(
    tmp_path: Path,
    enabled: bool = True,
    max_top_k: int = 5,
    served_name: str | None = None,
) -> dict[str, Path]:
    routing = {
        "embeddings": [
            {
                "model": "test-embedding",
                "enabled": enabled,
                **({"served_name": served_name} if served_name else {}),
                "upstream": {
                    "type": "infinity",
                    "url": "http://embed-upstream",
                    "api_key": "embed-key",
                    "timeout_seconds": 5,
                },
            }
        ],
        "rerank": [
            {
                "model": "test-rerank",
                "enabled": enabled,
                "max_top_k": max_top_k,
                "upstream": {
                    "type": "rerank",
                    "url": "http://rerank-upstream",
                    "api_key": "rerank-key",
                    "timeout_seconds": 5,
                },
            }
        ],
    }
    auth = {"api_keys": ["test-key"]}
    rate_limits = {"default": {"capacity": 100, "refill_rate": 100}}

    routing_path = tmp_path / "routing.json"
    auth_path = tmp_path / "auth.json"
    rate_path = tmp_path / "rate.json"

    write_yaml(routing_path, routing)
    write_yaml(auth_path, auth)
    write_yaml(rate_path, rate_limits)
    return {"routing": routing_path, "auth": auth_path, "rate": rate_path}


def with_env(monkeypatch: pytest.MonkeyPatch, paths: dict[str, Path]) -> None:
    monkeypatch.setenv("ROUTER_ROUTING_PATH", str(paths["routing"]))
    monkeypatch.setenv("ROUTER_AUTH_PATH", str(paths["auth"]))
    monkeypatch.setenv("ROUTER_RATE_LIMITS_PATH", str(paths["rate"]))
    monkeypatch.setenv("ROUTER_ALLOW_DISABLED_MODELS", "true")
    monkeypatch.setenv("ROUTER_TOKENIZER_NAME", "dummy-tokenizer")


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_length=True, is_split_into_words=False, **kwargs):
        tokens = text.split()
        return {"length": [len(tokens)], "input_ids": [list(range(len(tokens)))]}

    def decode(self, input_ids, skip_special_tokens=True):
        return " ".join(["token"] * len(input_ids))


@pytest.fixture(autouse=True)
def mock_tokenizer_provider(monkeypatch: pytest.MonkeyPatch):
    dummy = DummyTokenizer()

    def _get(self, name: str):
        return dummy

    monkeypatch.setattr(TokenizerProvider, "get", _get)


def test_load_routing_config(tmp_path: Path):
    paths = setup_configs(tmp_path)
    config = load_routing_config(paths["routing"])
    assert "test-embedding" in config.embeddings
    assert "test-rerank" in config.rerank
    assert config.rerank["test-rerank"].max_top_k == 5


def test_unknown_model_returns_400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = setup_configs(tmp_path)
    with_env(monkeypatch, paths)
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "unknown", "input": "hello"},
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["message"]


def test_top_k_exceeds_max_returns_400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = setup_configs(tmp_path, max_top_k=3)
    with_env(monkeypatch, paths)
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/rerank",
            json={"model": "test-rerank", "query": "q", "documents": ["a", "b", "c", "d"], "top_k": 5},
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 400


def test_disabled_model_returns_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = setup_configs(tmp_path, enabled=False)
    with_env(monkeypatch, paths)
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "test-embedding", "input": "hello"},
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 503


def test_successful_embedding_proxy(httpx_mock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = setup_configs(tmp_path)
    with_env(monkeypatch, paths)
    httpx_mock.add_response(
        method="POST",
        url="http://embed-upstream/v1/embeddings",
        json={"data": [{"embedding": [0.1, 0.2, 0.3]}]},
    )

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "test-embedding", "input": "hello"},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_safe_limit_exceeded_returns_400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = setup_configs(tmp_path)
    with_env(monkeypatch, paths)
    monkeypatch.setenv("ROUTER_SAFE_REQUEST_TOKEN_LIMIT", "1")

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "test-embedding", "input": "too many tokens here"},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 400
    assert "exceeds safe limit" in response.json()["error"]["message"]


def test_served_name_forwarded(httpx_mock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = setup_configs(tmp_path, served_name="upstream-model")
    with_env(monkeypatch, paths)
    captured = {}

    def _callback(request: httpx.Request):
        captured["payload"] = request.json()
        return httpx.Response(200, json={"data": [{"embedding": [0.4, 0.5]}]})

    httpx_mock.add_callback(_callback, method="POST", url="http://embed-upstream/v1/embeddings")

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "test-embedding", "input": "hello"},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    assert captured["payload"]["model"] == "upstream-model"


def test_document_truncation_by_tokens(httpx_mock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = setup_configs(tmp_path)
    with_env(monkeypatch, paths)
    monkeypatch.setenv("ROUTER_DOCUMENT_TOKEN_LIMIT", "2")

    captured = {}

    def _callback(request: httpx.Request):
        captured["payload"] = request.json()
        return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

    httpx_mock.add_callback(_callback, method="POST", url="http://embed-upstream/v1/embeddings")

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "test-embedding", "input": "one two three four"},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200
    assert captured["payload"]["input"] == "token token"


def test_timeout_ms_precedence():
    cfg = UpstreamConfig(type="infinity", url="http://example.com", timeout_ms=1500, timeout_seconds=10)
    assert cfg.get_timeout(default_timeout=30) == 1.5
