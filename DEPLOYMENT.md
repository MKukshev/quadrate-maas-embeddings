# Инструкция по развёртыванию

Руководство по развёртыванию Quadrate MAAS Embeddings в различных окружениях.

## Содержание

- [Требования](#требования)
- [Быстрый старт (Docker Compose)](#быстрый-старт-docker-compose)
- [Конфигурация](#конфигурация)
- [Развёртывание с GPU](#развёртывание-с-gpu)
- [Production развёртывание](#production-развёртывание)
- [Kubernetes](#kubernetes)
- [Переменные окружения](#переменные-окружения)

---

## Требования

### Минимальные требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32+ GB |
| Disk | 20 GB | 50+ GB (для моделей) |
| GPU | - | NVIDIA с 8+ GB VRAM |

### Программное обеспечение

- Docker 24.0+
- Docker Compose v2.20+
- NVIDIA Container Toolkit (для GPU)
- Python 3.11+ (для локальной разработки)

### Проверка окружения

```bash
# Docker
docker --version
docker compose version

# NVIDIA (опционально)
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Быстрый старт (Docker Compose)

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd quadrate-maas-embeddings
```

### 2. Запуск (CPU режим)

```bash
docker compose -f deploy/compose/docker-compose.yml up -d
```

### 3. Проверка

```bash
# Health check
curl http://localhost:8085/health/ready

# Список моделей
curl http://localhost:8085/v1/models

# Тестовый запрос
curl -X POST http://localhost:8085/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": "Hello world"}'
```

### 4. Остановка

```bash
docker compose -f deploy/compose/docker-compose.yml down
```

---

## Конфигурация

### Структура конфигурационных файлов

```
configs/
├── routing.yaml      # Маршрутизация моделей
├── auth.yaml         # API ключи
└── rate_limits.yaml  # Лимиты запросов
```

### routing.yaml

Определяет доступные модели и upstream сервисы:

```yaml
embeddings:
  - model: "bge-m3"                          # Алиас для клиентов
    served_name: "BAAI/bge-m3"               # Имя модели в upstream
    enabled: true
    upstream:
      type: "infinity"
      url: "http://infinity-embed:7997"
      timeout_ms: 3000

  - model: "multilingual-e5-large"
    served_name: "intfloat/multilingual-e5-large"
    enabled: true
    upstream:
      type: "infinity"
      url: "http://infinity-embed:7997"
      timeout_ms: 3000

  - model: "frida"
    served_name: "ai-forever/FRIDA"
    enabled: true
    upstream:
      type: "infinity"
      url: "http://infinity-embed:7997"
      timeout_ms: 3000

rerank:
  - model: "rerank-base"
    served_name: "cross-encoder/ms-marco-MiniLM-L-2-v2"
    enabled: true
    max_top_k: 50
    upstream:
      type: "rerank"
      url: "http://rerank:9002"
      timeout_seconds: 10
```

### auth.yaml

Настройка API ключей:

```yaml
api_keys:
  - "production-key-1"
  - "production-key-2"
  - "admin-key"
allow_anonymous_without_api_keys: false
```

### rate_limits.yaml

Настройка rate limiting (token bucket):

```yaml
default:
  capacity: 60        # Максимум запросов в bucket
  refill_rate: 30     # Запросов в секунду

anonymous:
  capacity: 30
  refill_rate: 15

per_api_key:
  admin-key:
    capacity: 120
    refill_rate: 60
  production-key-1:
    capacity: 100
    refill_rate: 50
```

---

## Развёртывание с GPU

### Требования

- NVIDIA Driver 525+
- CUDA 12.0+
- NVIDIA Container Toolkit

### Установка NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Docker Compose с GPU

Текущий `docker-compose.yml` уже настроен для GPU. Ключевая секция:

```yaml
infinity-embed:
  image: michaelf34/infinity:latest
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ["0"]  # или MIG UUID
            capabilities: [gpu]
```

### Использование MIG (Multi-Instance GPU)

Для NVIDIA A100/H100/RTX PRO с поддержкой MIG:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"]
          capabilities: [gpu]
```

Получить список MIG устройств:

```bash
nvidia-smi -L
```

### Запуск

```bash
docker compose -f deploy/compose/docker-compose.yml up -d
```

---

## Production развёртывание

### Рекомендуемая архитектура

```
                    ┌─────────────┐
                    │   Nginx/    │
                    │   Traefik   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
        │  Router   │ │ Router  │ │  Router   │
        │  (pod 1)  │ │ (pod 2) │ │  (pod N)  │
        └─────┬─────┘ └────┬────┘ └─────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
   ┌─────▼─────┐    ┌──────▼──────┐   ┌──────▼──────┐
   │ Infinity  │    │  Infinity   │   │   Rerank    │
   │  (GPU 0)  │    │   (GPU 1)   │   │   (CPU)     │
   └───────────┘    └─────────────┘   └─────────────┘
```

### docker-compose.production.yml

```yaml
version: "3.9"

networks:
  maas:
    driver: bridge

services:
  router:
    image: your-registry/maas-router:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 2G
    environment:
      ROUTER_ROUTING_PATH: /app/configs/routing.yaml
      ROUTER_AUTH_PATH: /app/configs/auth.yaml
      ROUTER_RATE_LIMITS_PATH: /app/configs/rate_limits.yaml
      ROUTER_REQUEST_TIMEOUT_SECONDS: 60
    ports:
      - "8085:8000"
    volumes:
      - ./configs:/app/configs:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/live"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - maas

  infinity-embed:
    image: michaelf34/infinity:latest
    command:
      - "v2"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "7997"
      - "--url-prefix"
      - "/v1"
      - "--model-id"
      - "BAAI/bge-m3"
      - "--model-id"
      - "intfloat/multilingual-e5-large"
      - "--model-id"
      - "ai-forever/FRIDA"
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - model-cache:/root/.cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7997/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - maas

  rerank:
    image: your-registry/maas-rerank:latest
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '4'
          memory: 8G
    environment:
      RERANK_MODEL_NAME: cross-encoder/ms-marco-MiniLM-L-2-v2
      RERANK_MODEL_MAPPING: '{"rerank-base": "cross-encoder/ms-marco-MiniLM-L-2-v2"}'
      RERANK_MAX_BATCH_SIZE: 16
      RERANK_DEVICE: cpu
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9002/health/ready"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - maas

volumes:
  model-cache:
```

### Сборка образов

```bash
# Router
docker build -t your-registry/maas-router:latest \
  -f services/router/Dockerfile .

# Rerank
docker build -t your-registry/maas-rerank:latest \
  -f services/rerank/Dockerfile .

# Push
docker push your-registry/maas-router:latest
docker push your-registry/maas-rerank:latest
```

### Nginx конфигурация

```nginx
upstream maas_router {
    least_conn;
    server router1:8000;
    server router2:8000;
    server router3:8000;
}

server {
    listen 443 ssl http2;
    server_name embeddings.example.com;

    ssl_certificate /etc/ssl/certs/embeddings.crt;
    ssl_certificate_key /etc/ssl/private/embeddings.key;

    location / {
        proxy_pass http://maas_router;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Request-ID $request_id;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    location /health {
        proxy_pass http://maas_router;
        access_log off;
    }
}
```

---

## Kubernetes

### Helm Charts

Готовые Helm charts находятся в `deploy/k8s/helm/`.

### Установка

```bash
# Добавить namespace
kubectl create namespace maas

# Установить Infinity
helm upgrade --install infinity-embed \
  deploy/k8s/helm/infinity-embed \
  -n maas \
  --set image.tag=latest \
  --set resources.limits.nvidia.com/gpu=1

# Установить Rerank
helm upgrade --install rerank \
  deploy/k8s/helm/rerank \
  -n maas

# Установить Router
helm upgrade --install router \
  deploy/k8s/helm/router \
  -n maas \
  --set replicaCount=3
```

### Применение манифестов напрямую

```bash
kubectl apply -f deploy/k8s/manifests/ -n maas
```

### Проверка

```bash
kubectl get pods -n maas
kubectl logs -f deployment/router -n maas
```

---

## Переменные окружения

### Router

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `ROUTER_ROUTING_PATH` | `configs/routing.yaml` | Путь к конфигу маршрутизации |
| `ROUTER_AUTH_PATH` | `configs/auth.yaml` | Путь к конфигу авторизации |
| `ROUTER_RATE_LIMITS_PATH` | `configs/rate_limits.yaml` | Путь к конфигу лимитов |
| `ROUTER_REQUEST_TIMEOUT_SECONDS` | `30` | Таймаут запросов к upstream |
| `ROUTER_ALLOW_DISABLED_MODELS` | `false` | Показывать отключённые модели в /v1/models |
| `ROUTER_ROUTING_RELOAD_INTERVAL_SECONDS` | `5` | Интервал перезагрузки конфигов |
| `ROUTER_SAFE_REQUEST_TOKEN_LIMIT` | - | Макс. токенов в запросе |
| `ROUTER_DOCUMENT_TOKEN_LIMIT` | - | Макс. токенов в документе |
| `ROUTER_TOKENIZER_NAME` | - | Имя токенизатора (HuggingFace) |
| `ROUTER_FEATURE_FLAGS` | - | Comma-separated feature flags |
| `QWEN3_ENABLED` | `false` | Включить Qwen3 upstream |
| `QWEN3_API_KEY` | - | API ключ для Qwen3 |

### Rerank

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `RERANK_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-2-v2` | Модель для ранжирования |
| `RERANK_MODEL_MAPPING` | `{}` | JSON с алиасами моделей |
| `RERANK_MAX_DOCUMENTS` | `50` | Макс. документов в запросе |
| `RERANK_MAX_DOCUMENT_LENGTH` | `4096` | Макс. длина документа (chars) |
| `RERANK_MAX_BATCH_SIZE` | `8` | Размер батча для инференса |
| `RERANK_BATCH_DELAY_MS` | `20` | Задержка сбора батча |
| `RERANK_QUEUE_TIMEOUT_MS` | `1000` | Таймаут очереди |
| `RERANK_DEVICE` | `auto` | Устройство: `auto`, `cpu`, `gpu` |
| `RERANK_MIXED_PRECISION` | `true` | Использовать FP16 на GPU |
| `RERANK_QUANTIZATION` | `false` | Включить квантизацию |

### Infinity

| Переменная | Описание |
|------------|----------|
| `HF_HOME` | Директория кэша HuggingFace |
| `TRANSFORMERS_CACHE` | Директория кэша моделей |

---

## Проверка развёртывания

### Smoke test

```bash
./scripts/smoke_test.sh
```

### Ручная проверка

```bash
# 1. Health check
curl http://localhost:8085/health/ready
# Ожидаемый ответ: {"status":"ok"}

# 2. Список моделей
curl http://localhost:8085/v1/models | jq '.data[].id'

# 3. Embeddings
curl -X POST http://localhost:8085/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": "test"}' | jq '.data[0].embedding | length'
# Ожидаемый ответ: 1024

# 4. Rerank
curl -X POST http://localhost:8085/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "rerank-base", "query": "test", "documents": ["a", "b"], "top_k": 1}' | jq
```

---

## Troubleshooting

### Infinity не запускается

```bash
# Проверить логи
docker logs compose-infinity-embed-1

# Частые проблемы:
# - Недостаточно VRAM: уменьшите количество моделей
# - Нет доступа к HuggingFace: проверьте сеть
# - GPU не обнаружен: проверьте nvidia-container-toolkit
```

### Router возвращает 502

```bash
# Проверить доступность upstream
docker exec compose-router-1 curl http://infinity-embed:7997/health
docker exec compose-router-1 curl http://rerank:9002/health/ready
```

### Модели не загружаются

```bash
# Проверить свободное место
df -h

# Очистить кэш
docker volume rm compose_infinity-cache
```

---

## Дополнительные ресурсы

- [Руководство разработчика](DEVELOPER.md)
- [Инструкция по эксплуатации](OPERATIONS.md)
- [Архитектура системы](docs/architecture.md)

