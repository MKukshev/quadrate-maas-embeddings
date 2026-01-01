from __future__ import annotations

from fastapi import Header, HTTPException, status

from .settings import AuthSettings


class Authenticator:
    """Simple API-key authenticator."""

    def __init__(self, settings: AuthSettings) -> None:
        self._keys = set(settings.api_keys)

    def validate_key(self, api_key: str | None) -> str:
        if not self._keys:
            return api_key or ""
        if api_key is None or api_key not in self._keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return api_key


def api_key_dependency(
    authenticator: Authenticator,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """FastAPI dependency to enforce API-key authentication."""

    return authenticator.validate_key(x_api_key)
