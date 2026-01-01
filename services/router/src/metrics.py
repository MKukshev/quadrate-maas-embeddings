from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_COUNTER = Counter(
    "router_requests_total",
    "Total number of processed requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "router_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint", "method"],
)
