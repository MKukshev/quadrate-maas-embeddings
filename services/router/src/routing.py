from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

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
    model_version: Optional[str] = Field(default=None, description="Optional version identifier for the route")
    served_name: Optional[str] = Field(default=None, description="Upstream served model name")
    enabled: bool = True
    feature_flag: Optional[str] = Field(default=None, description="Feature flag controlling availability")
    upstream: UpstreamConfig


class RerankRoute(BaseModel):
    """Routing entry for rerank."""

    model: str
    model_version: Optional[str] = Field(default=None, description="Optional version identifier for the route")
    enabled: bool = True
    max_top_k: int = Field(default=200, ge=1)
    feature_flag: Optional[str] = Field(default=None, description="Feature flag controlling availability")
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
                models.append(
                    {
                        "id": route.model,
                        "object": "embedding",
                        "enabled": route.enabled,
                        "model_version": route.model_version,
                    }
                )
        for route in self.rerank.values():
            if include_disabled or route.enabled:
                models.append(
                    {
                        "id": route.model,
                        "object": "rerank",
                        "enabled": route.enabled,
                        "model_version": route.model_version,
                        "max_top_k": route.max_top_k,
                    }
                )
        return models

    def apply_feature_flags(self, flags: Set[str]) -> None:
        """Soft-disable routes gated by feature flags."""

        for route in self.embeddings.values():
            if route.feature_flag:
                route.enabled = route.enabled and (route.feature_flag in flags)
        for route in self.rerank.values():
            if route.feature_flag:
                route.enabled = route.enabled and (route.feature_flag in flags)

    def describe(self) -> dict:
        """Return routing configuration with secrets masked."""

        def _mask_upstream(upstream: UpstreamConfig) -> dict:
            upstream_dict = upstream.model_dump()
            if upstream_dict.get("api_key"):
                upstream_dict["api_key"] = "***redacted***"
            return upstream_dict

        return {
            "embeddings": [
                {
                    "model": route.model,
                    "model_version": route.model_version,
                    "served_name": route.served_name,
                    "enabled": route.enabled,
                    "feature_flag": route.feature_flag,
                    "upstream": _mask_upstream(route.upstream),
                }
                for route in self.embeddings.values()
            ],
            "rerank": [
                {
                    "model": route.model,
                    "model_version": route.model_version,
                    "enabled": route.enabled,
                    "feature_flag": route.feature_flag,
                    "max_top_k": route.max_top_k,
                    "upstream": _mask_upstream(route.upstream),
                }
                for route in self.rerank.values()
            ],
        }


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

    routing = RoutingConfig(embeddings=embeddings, rerank=rerank_entries)
    qwen3_enabled = os.getenv("QWEN3_ENABLED")
    qwen3_api_key = os.getenv("QWEN3_API_KEY")
    if qwen3_enabled is not None or qwen3_api_key is not None:
        for route in routing.embeddings.values():
            if route.upstream.type == UpstreamType.QWEN3:
                if qwen3_enabled is not None:
                    route.enabled = qwen3_enabled.lower() in {"1", "true", "yes", "on"}
                if qwen3_api_key:
                    route.upstream.api_key = qwen3_api_key

    return routing
