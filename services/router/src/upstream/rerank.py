from __future__ import annotations

import httpx

from ..routing import UpstreamConfig


async def rerank(
    client: httpx.AsyncClient,
    config: UpstreamConfig,
    payload: dict,
    request_id: str,
) -> dict:
    """Proxy rerank request to rerank upstream."""

    headers = {"X-Request-Id": request_id}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    base_url = str(config.url).rstrip("/")
    response = await client.post(
        f"{base_url}/v1/rerank",
        json=payload,
        headers=headers,
        timeout=config.get_timeout(client.timeout),
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("data") or data.get("results") or []
    normalized = []
    for idx, item in enumerate(results):
        normalized.append(
            {
                "index": item.get("index", idx),
                "relevance_score": item.get("relevance_score") or item.get("score"),
                "document": item.get("document")
                or (payload.get("documents", [None])[idx] if idx < len(payload.get("documents", [])) else None),
            }
        )
    return {
        "object": "rerank",
        "model": payload.get("model"),
        "data": normalized,
    }
