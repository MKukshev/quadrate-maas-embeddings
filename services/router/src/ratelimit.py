from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict

from fastapi import HTTPException, status

from .metrics import RATE_LIMIT_DROPS
from .settings import RateLimitConfig, RateLimitSettings

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimitConfig) -> None:
        self.capacity = config.capacity
        self.refill_rate = config.refill_rate  # tokens per minute
        self.tokens = float(self.capacity)
        self.updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.updated_at
        refill_amount = (self.refill_rate / 60.0) * elapsed
        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.updated_at = now

    async def consume(self, amount: float = 1.0) -> float:
        async with self._lock:
            self._refill()
            if self.tokens >= amount:
                self.tokens -= amount
                return 0.0
            deficit = amount - self.tokens
            wait_seconds = deficit / (self.refill_rate / 60.0)
            return max(wait_seconds, 0.0)


class RateLimiter:
    """Rate limiter with per-key buckets."""

    def __init__(self, settings: RateLimitSettings) -> None:
        self._settings = settings
        self._buckets: Dict[str, TokenBucket] = {}

    def _get_bucket(self, api_key: str) -> TokenBucket:
        key_label = api_key or "anonymous"
        if key_label not in self._buckets:
            if api_key:
                config = self._settings.per_api_key.get(api_key, self._settings.default)
            else:
                config = self._settings.anonymous or self._settings.default
            self._buckets[key_label] = TokenBucket(config)
        return self._buckets[key_label]

    async def check(self, api_key: str) -> None:
        key = api_key or "anonymous"
        bucket = self._get_bucket(api_key)
        wait_time = await bucket.consume()
        if wait_time > 0:
            logger.warning("Rate limit exceeded for key='%s', retry after %.2fs", key, wait_time)
            RATE_LIMIT_DROPS.labels(api_key=key).inc()
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded, retry after {wait_time:.2f} seconds",
            )
