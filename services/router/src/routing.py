from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import httpx
import yaml
from pydantic import BaseModel, Field, HttpUrl, field_validator


class UpstreamType(str):
    """Supported upstream providers."""

    INFINITY = "infinity"
    RERANK = "rerank"
    QWEN3 = "qwen3"


class UpstreamConfig(BaseModel):
    """Common upstream configuration."""

    type: str = Field(description="Upstream provider type")
    url: HttpUrl = Field(description="Base URL for upstream service")
    api_key: Optional[str] = None
    timeout_seconds: Optional[float] = Field(default=None, gt=0)
    timeout_ms: Optional[int] = Field(default=None, alias="timeout_ms", gt=0)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        allowed = {UpstreamType.INFINITY, UpstreamType.RERANK, UpstreamType.QWEN3}
        if value not in allowed:
            raise ValueError(f"Unsupported upstream type: {value}")
        return value

    def get_timeout(self, default_timeout: float | httpx.Timeout | None = None) -> float | httpx.Timeout | None:
        """Return httpx-compatible timeout value honoring ms or seconds config."""

        if self.timeout_ms is not None:
            return self.timeout_ms / 1000.0
        if self.timeout_seconds is not None:
            return self.timeout_seconds
        return default_timeout


class EmbeddingRoute(BaseModel):
    """Routing entry for embeddings."""

    model: str
    served_name: Optional[str] = Field(default=None, description="Upstream served model name")
    enabled: bool = True
    upstream: UpstreamConfig


class RerankRoute(BaseModel):
    """Routing entry for rerank."""

    model: str
    enabled: bool = True
    max_top_k: int = Field(default=200, ge=1)
    upstream: UpstreamConfig


class RoutingConfig(BaseModel):
    """Runtime routing configuration for embeddings and rerank."""

    embeddings: Dict[str, EmbeddingRoute] = Field(default_factory=dict)
    rerank: Dict[str, RerankRoute] = Field(default_factory=dict)

    def list_models(self, include_disabled: bool = False) -> List[dict]:
        """Return a flat list of models with their metadata."""

        models: List[dict] = []
        for route in self.embeddings.values():
            if include_disabled or route.enabled:
                models.append({"id": route.model, "object": "embedding", "enabled": route.enabled})
        for route in self.rerank.values():
            if include_disabled or route.enabled:
                models.append(
                    {
                        "id": route.model,
                        "object": "rerank",
                        "enabled": route.enabled,
                        "max_top_k": route.max_top_k,
                    }
                )
        return models


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected YAML structure in {path}")
    return data


def load_routing_config(path: Path) -> RoutingConfig:
    """Load routing YAML into validated models."""

    raw = _load_yaml(path)
    embeddings_raw = raw.get("embeddings", []) or []
    rerank_raw = raw.get("rerank", []) or []

    embeddings: Dict[str, EmbeddingRoute] = {}
    for item in embeddings_raw:
        route = EmbeddingRoute.model_validate(item)
        embeddings[route.model] = route

    rerank_entries: Dict[str, RerankRoute] = {}
    for item in rerank_raw:
        route = RerankRoute.model_validate(item)
        rerank_entries[route.model] = route

    return RoutingConfig(embeddings=embeddings, rerank=rerank_entries)
