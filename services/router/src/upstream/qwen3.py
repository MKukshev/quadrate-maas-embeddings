from __future__ import annotations

import asyncio
import logging

import httpx

from ..routing import UpstreamConfig

logger = logging.getLogger(__name__)


async def embeddings(
    client: httpx.AsyncClient,
    config: UpstreamConfig,
    payload: dict,
    request_id: str,
) -> dict:
    """Proxy embeddings request to Qwen3 upstream."""

    headers = {"X-Request-Id": request_id}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    base_url = str(config.url).rstrip("/")
    try:
        response = await client.post(
            f"{base_url}/v1/embeddings",
            json=payload,
            headers=headers,
            timeout=config.get_timeout(client.timeout),
        )
    except asyncio.CancelledError:
        logger.info(
            "Qwen3 upstream request canceled (request_id=%s, upstream=%s)",
            request_id,
            base_url,
            extra={"event": "upstream_cancelled", "upstream": base_url},
        )
        raise
    response.raise_for_status()
    data = response.json()
    embeddings_data = data.get("data") or data.get("embeddings") or []
    normalized = []
    for idx, item in enumerate(embeddings_data):
        vector = item.get("embedding") if isinstance(item, dict) else item
        normalized.append(
            {
                "object": "embedding",
                "embedding": vector,
                "index": item.get("index", idx) if isinstance(item, dict) else idx,
            }
        )

    return {
        "object": "list",
        "data": normalized,
        "model": payload.get("model"),
    }
