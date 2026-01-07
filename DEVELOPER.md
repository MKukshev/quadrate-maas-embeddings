# Developer Guide

## Canceling in-flight embeddings requests

You can cancel an in-flight `/v1/embeddings` request by using its `X-Request-Id` header value with the cancel endpoint.

### Example: cancel by request id

```bash
curl -sS -X POST "http://localhost:8080/v1/requests/req-cancel-1/cancel" \
  -H "X-API-Key: test-key"
```

Example response:

```json
{"status":"canceled"}
```

### Example: fetch request id and cancel

```bash
REQUEST_ID=$(
  curl -sS -D - -o /dev/null \
    -H "X-API-Key: test-key" \
    -H "Content-Type: application/json" \
    -d '{"model":"test-embedding","input":"hello"}' \
    "http://localhost:8080/v1/embeddings" \
  | awk -F': ' '/^X-Request-Id:/ {print $2}' \
  | tr -d '\r'
)

curl -sS -X POST "http://localhost:8080/v1/requests/${REQUEST_ID}/cancel" \
  -H "X-API-Key: test-key"
```
