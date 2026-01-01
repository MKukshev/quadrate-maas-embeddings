from prometheus_client import Counter, Histogram

REQUEST_COUNTER = Counter(
    "rerank_requests_total",
    "Number of rerank service requests",
    labelnames=("endpoint", "method", "status"),
)

REQUEST_LATENCY = Histogram(
    "rerank_request_latency_seconds",
    "Latency of rerank service requests",
    labelnames=("endpoint", "method"),
)

ERROR_COUNTER = Counter(
    "rerank_errors_total",
    "Number of rerank service errors",
    labelnames=("endpoint", "reason"),
)

DOCUMENT_COUNTER = Counter(
    "rerank_documents_total",
    "Number of documents processed by the rerank endpoint",
    labelnames=("model",),
)
