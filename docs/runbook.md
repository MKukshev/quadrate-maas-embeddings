# Runbook

## Запуск сервисов

Сервисы Router и Rerank поднимаются как обычные FastAPI-приложения.

```bash
# Router
export ROUTING_PATH=configs/routing.yaml
export AUTH_PATH=configs/auth.yaml
export RATE_LIMITS_PATH=configs/rate_limits.yaml
uvicorn services.router.src.main:app --host 0.0.0.0 --port 8000

# Rerank
export RERANK_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-2-v2
uvicorn services.rerank.src.main:app --host 0.0.0.0 --port 9002
```

## Наблюдение и метрики

Оба сервиса публикуют Prometheus-метрики по пути `/metrics`.

- Router:
  - `router_requests_total`, `router_request_latency_seconds`
  - `router_upstream_latency_seconds`, `router_upstream_errors_total`
  - `router_rate_limit_drops_total`, `router_rerank_documents_total`
- Rerank:
  - `rerank_requests_total`, `rerank_request_latency_seconds`
  - `rerank_documents_total`

## Smoke-тест

Скрипт `scripts/smoke_test.sh` проверяет готовность и выполняет запросы:

```bash
ROUTER_URL=http://localhost:8000 ROUTER_API_KEY=test-key bash scripts/smoke_test.sh
```

Что делает скрипт:
- Проверяет `/health/ready`
- Запрашивает `/v1/embeddings` для трёх моделей (bge-m3, multilingual-e5-large, frida)
- Запускает `/v1/rerank` с 20 документами

## Нагрузочное тестирование

Сценарий k6 расположен в `scripts/bench/k6.js`. Он генерирует смешанный трафик (embeddings + rerank) с постоянной скоростью 5 RPS и выводит p50/p95 и error rate.

```bash
ROUTER_URL=http://localhost:8000 ROUTER_API_KEY=test-key k6 run scripts/bench/k6.js
```
