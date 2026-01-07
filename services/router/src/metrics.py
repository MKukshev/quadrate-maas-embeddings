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

UPSTREAM_LATENCY = Histogram(
    "router_upstream_latency_seconds",
    "Latency of requests sent to upstream services",
    ["upstream", "operation"],
)

UPSTREAM_ERRORS = Counter(
    "router_upstream_errors_total",
    "Number of upstream errors grouped by upstream and operation",
    ["upstream", "operation", "status"],
)

RATE_LIMIT_DROPS = Counter(
    "router_rate_limit_drops_total",
    "Requests rejected due to rate limiting",
    ["api_key"],
)

RERANK_DOCUMENTS_COUNTER = Counter(
    "router_rerank_documents_total",
    "Number of documents processed through the router rerank endpoint",
    ["model"],
)

READINESS_PARTIAL_FAILURES = Counter(
    "router_readiness_degraded_total",
    "Number of degraded readiness responses grouped by upstream",
    ["upstream"],
)

READINESS_DEGRADED_EVENTS = Counter(
    "router_readiness_degraded_events_total",
    "Number of degraded readiness responses grouped by failing upstream count",
    ["failing_count"],
)

CLIENT_DISCONNECTS = Counter(
    "router_client_disconnects_total",
    "Number of requests canceled because the client disconnected",
    ["endpoint", "stage"],
)
