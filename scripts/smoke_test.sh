#!/usr/bin/env bash
set -euo pipefail

ROUTER_URL=${ROUTER_URL:-"http://localhost:8085"}

models=("bge-m3" "multilingual-e5-large" "frida")

JQ_BIN=$(command -v jq || true)

pretty_print() {
  if [[ -n "${JQ_BIN}" ]]; then
    "${JQ_BIN}" .
  else
    cat
  fi
}

echo "==> Checking router readiness at ${ROUTER_URL}"
curl -sf "${ROUTER_URL}/health/ready" | pretty_print

call_embedding() {
  local model=$1
  echo "==> Embeddings for model: ${model}"
  curl -sf \
    -H "Content-Type: application/json" \
    -X POST \
    -d "{\"model\":\"${model}\",\"input\":\"hello from smoke test\"}" \
    "${ROUTER_URL}/v1/embeddings" | pretty_print
}

for model in "${models[@]}"; do
  call_embedding "${model}"
done

echo "==> Rerank smoke check with 20 documents"
docs=()
for i in $(seq 1 20); do
  docs+=("\"doc${i}\"")
done
docs_joined=$(IFS=,; echo "${docs[*]}")

curl -sf \
  -H "Content-Type: application/json" \
  -X POST \
  -d "{\"model\":\"rerank-base\",\"query\":\"test\",\"documents\":[${docs_joined}],\"top_k\":5}" \
  "${ROUTER_URL}/v1/rerank" | pretty_print

echo "Smoke tests completed successfully."
