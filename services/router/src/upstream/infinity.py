from __future__ import annotations

import httpx

from ..routing import UpstreamConfig


async def embeddings(
    client: httpx.AsyncClient,
    config: UpstreamConfig,
    payload: dict,
    request_id: str,
    served_model: str | None = None,
) -> dict:
    """Proxy embeddings request to Infinity upstream."""

    headers = {"X-Request-Id": request_id}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    upstream_payload = {
        **payload,
        "model": served_model or payload.get("model"),
    }

    response = await client.post(
        f"{config.url}/v1/embeddings",
        json=upstream_payload,
        headers=headers,
        timeout=config.get_timeout(client.timeout),
    )
    response.raise_for_status()
    data = response.json()

    # Normalize into OpenAI-like response
    embeddings_data = data.get("data") or data.get("embeddings") or []
    normalized = []
    for idx, item in enumerate(embeddings_data):
        vector = item.get("embedding") if isinstance(item, dict) else item
        normalized.append({"object": "embedding", "embedding": vector, "index": idx})

    return {
        "object": "list",
        "data": normalized,
        "model": payload.get("model"),
    }
