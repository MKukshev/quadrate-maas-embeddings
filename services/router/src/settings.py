from __future__ import annotations

import yaml
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseModel):
    """API key configuration loaded from YAML."""

    api_keys: List[str] = Field(default_factory=list)


class RateLimitConfig(BaseModel):
    """Token bucket settings."""

    capacity: int = Field(default=60, ge=1)
    refill_rate: float = Field(default=30.0, gt=0)


class RateLimitSettings(BaseModel):
    """Rate limit settings loaded from YAML."""

    default: RateLimitConfig = Field(default_factory=RateLimitConfig)
    per_api_key: dict[str, RateLimitConfig] = Field(default_factory=dict)

    @field_validator("per_api_key", mode="before")
    @classmethod
    def _ensure_mapping(cls, value: Optional[dict[str, dict]]) -> dict[str, RateLimitConfig]:
        return value or {}


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables."""

    routing_path: Path = Field(default=Path("configs/routing.yaml"))
    auth_path: Path = Field(default=Path("configs/auth.yaml"))
    rate_limits_path: Path = Field(default=Path("configs/rate_limits.yaml"))
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    allow_disabled_models: bool = Field(
        default=False,
        description="Expose disabled models in /v1/models when True, otherwise hide them.",
    )
    environment: str = Field(default="development")

    model_config = SettingsConfigDict(env_prefix="ROUTER_", case_sensitive=False)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected YAML structure in {path}")
    return data


def load_auth_settings(path: Path) -> AuthSettings:
    raw = _load_yaml(path)
    api_keys = raw.get("api_keys", []) or []
    return AuthSettings(api_keys=api_keys)


def load_rate_limit_settings(path: Path) -> RateLimitSettings:
    raw = _load_yaml(path)
    default = RateLimitConfig(**raw.get("default", {})) if isinstance(raw.get("default", {}), dict) else RateLimitConfig()
    per_key_raw: dict[str, Union[dict, RateLimitConfig]] = raw.get("per_api_key", {}) or {}
    per_api_key: dict[str, RateLimitConfig] = {}
    for key, cfg in per_key_raw.items():
        per_api_key[key] = RateLimitConfig(**cfg) if isinstance(cfg, dict) else cfg  # type: ignore[arg-type]
    return RateLimitSettings(default=default, per_api_key=per_api_key)
