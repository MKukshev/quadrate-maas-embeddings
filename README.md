# Quadrate MAAS Embeddings

Набор сервисов для проксирования запросов к моделям эмбеддингов и ранжирования. Репозиторий включает готовые образы для FastAPI‑сервисов, конфигурации маршрутизации, примеры локального запуска и манифесты для контейнеризации.

## Структура репозитория

- `services/router` — роутер входящих запросов на эмбеддинги и rerank, проверка ключей и rate limit. Точка входа: `services.router.main`.
- `services/rerank` — сервис ранжирования документов. Точка входа: `services.rerank.main`.
- `configs/` — пример конфигураций маршрутизации, авторизации и rate limit по умолчанию.
- `deploy/compose/` — Docker Compose для локальной сборки и запуска, включая вариант с Qwen3.
- `deploy/k8s/` — Helm chart и готовые манифесты для Kubernetes.
- `docs/` — архитектура, API и runbook.

## Конфигурация

### Роутер
- Пути до конфигураций задаются переменными окружения:
  - `ROUTER_ROUTING_PATH` — файл с моделями и апстримами (по умолчанию `configs/routing.yaml`).
  - `ROUTER_AUTH_PATH` — список API‑ключей и настройка анонимного доступа (по умолчанию `configs/auth.yaml`).
  - `ROUTER_RATE_LIMITS_PATH` — лимиты по ключам и для анонимных запросов (по умолчанию `configs/rate_limits.yaml`).
- Дополнительно:
  - `ROUTER_REQUEST_TIMEOUT_SECONDS` — таймаут прокси‑запросов (дефолт 30).
  - `ROUTER_ALLOW_DISABLED_MODELS` — если `true`, разрешает использовать модели с `enabled: false`.
  - `QWEN3_ENABLED` и `QWEN3_API_KEY` — включение и ключ для Qwen3 upstream.

Пример маршрутизации и лимитов смотрите в `configs/routing.yaml`, `configs/auth.yaml`, `configs/rate_limits.yaml`.

### Rerank
- Основные переменные:
  - `RERANK_MODEL_NAME` — базовая модель (по умолчанию `BAAI/bge-reranker-v2-m3`).
  - `RERANK_MODEL_MAPPING` — JSON с алиасами моделей.
  - `RERANK_MAX_DOCUMENTS` и `RERANK_MAX_DOCUMENT_LENGTH` — ограничения входных данных.
  - `RERANK_MAX_BATCH_SIZE`, `RERANK_BATCH_DELAY_MS`, `RERANK_QUEUE_TIMEOUT_MS` — параметры батчинга.
  - `RERANK_DEVICE`, `RERANK_MIXED_PRECISION`, `RERANK_QUANTIZATION` — выбор устройства и режимов инференса.

## Локальный запуск (Python)

1. Создайте окружение и установите зависимости сервисов:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r services/router/requirements.txt
   pip install -r services/rerank/requirements.txt
   ```
2. Поднимите сервис rerank (порт 9002 для совместимости с конфигами):
   ```bash
   uvicorn services.rerank.src.main:app --host 0.0.0.0 --port 9002
   ```
3. В другом терминале запустите роутер, указав пути к конфигам при необходимости:
   ```bash
   ROUTER_ROUTING_PATH=configs/routing.yaml \
   ROUTER_AUTH_PATH=configs/auth.yaml \
   ROUTER_RATE_LIMITS_PATH=configs/rate_limits.yaml \
   uvicorn services.router.src.main:app --host 0.0.0.0 --port 8000
   ```
4. Отправляйте запросы на `http://localhost:8000/v1/embeddings` или `http://localhost:8000/v1/rerank` с заголовком `X-API-Key`.

## Запуск через Docker Compose

- Базовый стек (router + Infinity + rerank):
  ```bash
  docker compose -f deploy/compose/docker-compose.yml up --build
  ```
  Роутер будет доступен на `http://localhost:${ROUTER_PORT:-8085}`.

- Вариант с Qwen3:
  ```bash
  QWEN3_API_KEY=<ключ> QWEN3_ENABLED=true \
  docker compose -f deploy/compose/docker-compose.yml -f deploy/compose/docker-compose.qwen3.yml up --build
  ```

## Запуск в Kubernetes

- Готовые манифесты находятся в `deploy/k8s/manifests`. Пример применения:
  ```bash
  kubectl apply -f deploy/k8s/manifests/
  ```
- Для кастомизации используйте Helm чарты в `deploy/k8s/helm` и передавайте значения через `values.yaml` или `--set`.

## Тестирование

- Юнит‑тесты роутера:
  ```bash
  pytest services/router/tests
  ```
- Юнит‑тесты rerank (при наличии моделей и ресурсов):
  ```bash
  pytest services/rerank/tests
  ```
- Метрики доступны на `/metrics`, здоровье — `/health/live` и `/health/ready` у каждого сервиса.
