# Инструкция по эксплуатации

Руководство по мониторингу, диагностике и решению проблем Quadrate MAAS Embeddings.

## Содержание

- [Обзор системы](#обзор-системы)
- [Мониторинг](#мониторинг)
- [Логирование](#логирование)
- [Health Checks](#health-checks)
- [Метрики](#метрики)
- [Диагностика проблем](#диагностика-проблем)
- [Типичные проблемы и решения](#типичные-проблемы-и-решения)
- [Обслуживание](#обслуживание)
- [Резервное копирование](#резервное-копирование)

---

## Обзор системы

### Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                         Клиенты                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Router (:8085)                               │
│  • Аутентификация (опционально)                                 │
│  • Rate Limiting (token bucket)                                  │
│  • Маршрутизация запросов                                       │
│  • Prometheus метрики                                            │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
             ▼                                    ▼
┌────────────────────────┐          ┌────────────────────────────┐
│   Infinity (:7997)     │          │      Rerank (:9002)        │
│   • GPU инференс       │          │   • CPU/GPU инференс       │
│   • 3 embedding модели │          │   • Cross-encoder          │
│   • Batching           │          │   • Batching               │
└────────────────────────┘          └────────────────────────────┘
```

### Порты и сервисы

| Сервис | Внутренний порт | Внешний порт | Описание |
|--------|-----------------|--------------|----------|
| Router | 8000 | 8085 | API Gateway |
| Infinity | 7997 | 7997 | Embedding сервис |
| Rerank | 9002 | 9002 | Rerank сервис |

---

## Мониторинг

### Проверка статуса контейнеров

```bash
# Все контейнеры
docker ps --filter "name=compose" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Ресурсы
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
```

### GPU мониторинг

```bash
# Статус GPU
nvidia-smi

# Непрерывный мониторинг
watch -n 1 nvidia-smi

# Процессы на GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

# MIG устройства
nvidia-smi -L
```

### Health endpoints

```bash
# Router
curl -s http://localhost:8085/health/ready | jq
curl -s http://localhost:8085/health/live | jq

# Infinity
curl -s http://localhost:7997/health | jq

# Rerank
curl -s http://localhost:9002/health/ready | jq
curl -s http://localhost:9002/health/live | jq
```

### Скрипт мониторинга

```bash
#!/bin/bash
# monitor.sh - Быстрая проверка состояния системы

echo "=== Контейнеры ==="
docker ps --filter "name=compose" --format "table {{.Names}}\t{{.Status}}"
echo ""

echo "=== Health Checks ==="
echo -n "Router:   "; curl -s http://localhost:8085/health/ready | jq -r '.status'
echo -n "Infinity: "; curl -s http://localhost:7997/health > /dev/null && echo "ok" || echo "fail"
echo -n "Rerank:   "; curl -s http://localhost:9002/health/ready | jq -r '.status'
echo ""

echo "=== Ресурсы ==="
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep compose
echo ""

echo "=== GPU ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

---

## Логирование

### Просмотр логов

```bash
# Router (последние 100 строк)
docker logs --tail 100 compose-router-1

# Infinity
docker logs --tail 100 compose-infinity-embed-1

# Rerank
docker logs --tail 100 compose-rerank-1

# Follow mode (непрерывный вывод)
docker logs -f compose-router-1

# Все сервисы вместе
docker compose -f deploy/compose/docker-compose.yml logs -f
```

### Фильтрация логов

```bash
# Только ошибки
docker logs compose-router-1 2>&1 | grep -i error

# Только warnings
docker logs compose-router-1 2>&1 | grep -i warn

# Конкретный request_id
docker logs compose-router-1 2>&1 | grep "abc123-request-id"

# Запросы за последний час
docker logs --since 1h compose-router-1

# Запросы за определённый период
docker logs --since "2026-01-02T09:00:00" --until "2026-01-02T10:00:00" compose-router-1
```

### Структура логов

#### Router

```
INFO:     172.26.0.1:54321 - "POST /v1/embeddings HTTP/1.1" 200 OK
WARNING:  Validation error for POST /v1/embeddings (request_id=abc123): ...
ERROR:    Unhandled exception for POST /v1/rerank (request_id=def456): ...
```

#### Infinity

```
INFO     2026-01-02 09:00:00,000 infinity_emb INFO: model warmed up, between 24.01-1231.37 embeddings/sec
INFO     2026-01-02 09:00:00,000 infinity_emb INFO: ready to batch requests.
```

#### Rerank

```
INFO:     Application startup complete.
INFO:     172.26.0.4:55094 - "POST /v1/rerank HTTP/1.1" 200 OK
```

### Экспорт логов

```bash
# В файл
docker logs compose-router-1 > router.log 2>&1

# Ротация логов (Docker daemon)
# /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "5"
  }
}

# Применить изменения
sudo systemctl restart docker
```

---

## Health Checks

### Интерпретация статусов

| Статус | Значение | Действие |
|--------|----------|----------|
| `ok` | Все upstream доступны | Нормальная работа |
| `degraded` | Часть upstream недоступна | Проверить логи, перезапустить проблемный сервис |
| `unavailable` | Сервис не готов | Проверить логи запуска |

### Автоматические проверки

```bash
#!/bin/bash
# healthcheck.sh - Проверка с алертингом

ROUTER_URL="http://localhost:8085"
ALERT_WEBHOOK="https://hooks.slack.com/services/xxx"

check_health() {
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$ROUTER_URL/health/ready")
    if [ "$STATUS" != "200" ]; then
        BODY=$(curl -s "$ROUTER_URL/health/ready")
        curl -X POST "$ALERT_WEBHOOK" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"⚠️ MAAS Embeddings health check failed: $BODY\"}"
        return 1
    fi
    return 0
}

check_health
```

### Kubernetes probes

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 5
```

---

## Метрики

### Prometheus endpoints

```bash
# Router метрики
curl -s http://localhost:8085/metrics

# Rerank метрики
curl -s http://localhost:9002/metrics
```

### Ключевые метрики Router

| Метрика | Тип | Описание |
|---------|-----|----------|
| `router_requests_total` | Counter | Общее количество запросов |
| `router_request_duration_seconds` | Histogram | Время обработки запросов |
| `router_upstream_latency_seconds` | Histogram | Латентность upstream |
| `router_upstream_errors_total` | Counter | Ошибки upstream |
| `router_rate_limit_drops_total` | Counter | Отклонённые запросы (rate limit) |
| `router_readiness_degraded_total` | Counter | События degraded статуса |

### Ключевые метрики Rerank

| Метрика | Тип | Описание |
|---------|-----|----------|
| `rerank_requests_total` | Counter | Общее количество запросов |
| `rerank_request_duration_seconds` | Histogram | Время обработки |
| `rerank_documents_total` | Counter | Обработано документов |
| `rerank_errors_total` | Counter | Ошибки по типам |

### Примеры запросов PromQL

```promql
# RPS по endpoint
rate(router_requests_total[5m])

# Средняя латентность
rate(router_request_duration_seconds_sum[5m]) / rate(router_request_duration_seconds_count[5m])

# Процент ошибок
sum(rate(router_requests_total{status=~"5.."}[5m])) / sum(rate(router_requests_total[5m])) * 100

# Rate limit drops
rate(router_rate_limit_drops_total[5m])

# Upstream ошибки
rate(router_upstream_errors_total[5m])
```

### Grafana Dashboard

Пример JSON dashboard:

```json
{
  "panels": [
    {
      "title": "Requests per Second",
      "targets": [
        {"expr": "sum(rate(router_requests_total[1m]))"}
      ]
    },
    {
      "title": "Latency P99",
      "targets": [
        {"expr": "histogram_quantile(0.99, rate(router_request_duration_seconds_bucket[5m]))"}
      ]
    },
    {
      "title": "Error Rate",
      "targets": [
        {"expr": "sum(rate(router_requests_total{status=~\"5..\"}[5m])) / sum(rate(router_requests_total[5m])) * 100"}
      ]
    }
  ]
}
```

---

## Диагностика проблем

### Общий алгоритм

```
1. Проверить статус контейнеров
   └─> docker ps --filter "name=compose"

2. Проверить health endpoints
   └─> curl http://localhost:8085/health/ready

3. Проверить логи проблемного сервиса
   └─> docker logs --tail 200 <container>

4. Проверить ресурсы (CPU/RAM/GPU)
   └─> docker stats / nvidia-smi

5. Проверить сетевую связность между контейнерами
   └─> docker exec compose-router-1 curl http://infinity-embed:7997/health

6. Проверить метрики
   └─> curl http://localhost:8085/metrics | grep error
```

### Диагностические команды

```bash
# Проверить все сервисы
docker compose -f deploy/compose/docker-compose.yml ps

# Сетевая связность
docker exec compose-router-1 ping -c 2 infinity-embed
docker exec compose-router-1 ping -c 2 rerank

# HTTP проверка внутри сети
docker exec compose-router-1 curl -s http://infinity-embed:7997/v1/models
docker exec compose-router-1 curl -s http://rerank:9002/health/ready

# Проверка DNS
docker exec compose-router-1 nslookup infinity-embed

# Информация о контейнере
docker inspect compose-router-1

# События Docker
docker events --filter container=compose-router-1 --since 1h
```

### Тестовые запросы

```bash
# Минимальный embedding запрос
curl -w "\nTime: %{time_total}s\n" -X POST http://localhost:8085/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": "test"}'

# Минимальный rerank запрос
curl -w "\nTime: %{time_total}s\n" -X POST http://localhost:8085/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "rerank-base", "query": "q", "documents": ["a", "b"]}'

# Прямой запрос к Infinity (минуя router)
curl -X POST http://localhost:7997/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-m3", "input": "test"}'

# Прямой запрос к Rerank
curl -X POST http://localhost:9002/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "cross-encoder/ms-marco-MiniLM-L-2-v2", "query": "q", "documents": ["a"]}'
```

---

## Типичные проблемы и решения

### 1. Router возвращает 502 Bad Gateway

**Симптомы:** Запросы к `/v1/embeddings` или `/v1/rerank` возвращают 502.

**Диагностика:**
```bash
docker logs compose-router-1 2>&1 | grep -i error | tail -10
docker exec compose-router-1 curl http://infinity-embed:7997/health
```

**Решения:**
- Upstream не запущен → перезапустить: `docker compose up -d infinity-embed`
- Таймаут → увеличить `timeout_ms` в `routing.yaml`
- Сетевая проблема → проверить docker network

### 2. Статус "degraded"

**Симптомы:** `/health/ready` возвращает `{"status":"degraded"}`.

**Диагностика:**
```bash
docker logs compose-router-1 2>&1 | grep -i degraded
```

**Решения:**
- Один из upstream не отвечает на health check
- Проверить каждый upstream отдельно
- Перезапустить проблемный сервис

### 3. Infinity не запускается

**Симптомы:** Контейнер перезапускается или не стартует.

**Диагностика:**
```bash
docker logs compose-infinity-embed-1 2>&1 | tail -50
```

**Типичные причины:**
- **OOM (Out of Memory):** Недостаточно RAM/VRAM
  ```bash
  # Уменьшить количество моделей в docker-compose.yml
  # Или увеличить память
  ```

- **GPU недоступен:**
  ```bash
  nvidia-smi
  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
  ```

- **Модели не скачиваются:**
  ```bash
  # Проверить сеть
  docker exec compose-infinity-embed-1 curl -I https://huggingface.co
  ```

### 4. Rate Limit (429)

**Симптомы:** Запросы возвращают 429 Too Many Requests.

**Диагностика:**
```bash
curl http://localhost:8085/metrics | grep rate_limit
```

**Решения:**
- Увеличить лимиты в `configs/rate_limits.yaml`
- Использовать exponential backoff на клиенте
- Добавить выделенный API ключ с большими лимитами

### 5. Медленные запросы

**Симптомы:** Высокая латентность.

**Диагностика:**
```bash
# Проверить метрики
curl -s http://localhost:8085/metrics | grep duration

# Проверить GPU утилизацию
nvidia-smi

# Тестовый замер
time curl -X POST http://localhost:8085/v1/embeddings \
  -d '{"model": "bge-m3", "input": "test"}'
```

**Решения:**
- Большие батчи → разбить на меньшие
- GPU перегружен → добавить реплики
- CPU bottleneck → увеличить ресурсы

### 6. Ошибка токенизатора

**Симптомы:** `OSError: xxx is not a valid model identifier`

**Диагностика:**
```bash
docker logs compose-router-1 2>&1 | grep -i tokenizer
```

**Решения:**
- Проверить `served_name` в `routing.yaml`
- Убедиться что имя модели соответствует HuggingFace
- Установить `ROUTER_TOKENIZER_NAME` для явного указания

### 7. Контейнер OOMKilled

**Симптомы:** Контейнер перезапускается, в `docker inspect` видно `OOMKilled: true`.

**Диагностика:**
```bash
docker inspect compose-infinity-embed-1 | grep OOMKilled
dmesg | grep -i "out of memory"
```

**Решения:**
```yaml
# docker-compose.yml
services:
  infinity-embed:
    deploy:
      resources:
        limits:
          memory: 32G  # Увеличить лимит
```

---

## Обслуживание

### Обновление моделей

```bash
# 1. Остановить сервис
docker compose -f deploy/compose/docker-compose.yml stop infinity-embed

# 2. Очистить кэш (опционально)
docker volume rm compose_infinity-cache

# 3. Обновить конфигурацию
vim deploy/compose/docker-compose.yml

# 4. Перезапустить
docker compose -f deploy/compose/docker-compose.yml up -d infinity-embed
```

### Обновление API ключей

```bash
# 1. Редактировать конфиг
vim configs/auth.yaml

# 2. Router перезагрузит автоматически (ROUTER_ROUTING_RELOAD_INTERVAL_SECONDS)
# Или принудительно:
docker restart compose-router-1
```

### Обновление образов

```bash
# 1. Скачать новые образы
docker compose -f deploy/compose/docker-compose.yml pull

# 2. Пересобрать локальные
docker compose -f deploy/compose/docker-compose.yml build

# 3. Перезапустить с новыми образами
docker compose -f deploy/compose/docker-compose.yml up -d
```

### Очистка

```bash
# Удалить остановленные контейнеры
docker container prune -f

# Удалить неиспользуемые образы
docker image prune -f

# Удалить неиспользуемые volumes (ОСТОРОЖНО!)
docker volume prune -f

# Полная очистка (ОСТОРОЖНО!)
docker system prune -a --volumes
```

---

## Резервное копирование

### Что нужно бэкапить

| Компонент | Путь | Важность |
|-----------|------|----------|
| Конфигурация | `configs/` | Критично |
| Docker Compose | `deploy/compose/` | Критично |
| Кэш моделей | Volume `infinity-cache` | Можно восстановить |

### Скрипт резервного копирования

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/maas-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Конфигурация
cp -r configs "$BACKUP_DIR/"
cp -r deploy "$BACKUP_DIR/"

# Docker volumes (опционально)
docker run --rm \
  -v compose_infinity-cache:/data:ro \
  -v "$BACKUP_DIR":/backup \
  alpine tar czf /backup/infinity-cache.tar.gz -C /data .

echo "Backup saved to: $BACKUP_DIR"
```

### Восстановление

```bash
#!/bin/bash
# restore.sh

BACKUP_DIR=$1

# Конфигурация
cp -r "$BACKUP_DIR/configs" ./
cp -r "$BACKUP_DIR/deploy" ./

# Volumes (если нужно)
docker volume create compose_infinity-cache
docker run --rm \
  -v compose_infinity-cache:/data \
  -v "$BACKUP_DIR":/backup:ro \
  alpine tar xzf /backup/infinity-cache.tar.gz -C /data

# Перезапуск
docker compose -f deploy/compose/docker-compose.yml up -d
```

---

## Контакты и эскалация

### Уровни критичности

| Уровень | Описание | SLA | Эскалация |
|---------|----------|-----|-----------|
| P1 | Полная недоступность | 15 мин | Немедленно |
| P2 | Деградация (часть моделей) | 1 час | В рабочее время |
| P3 | Повышенная латентность | 4 часа | В рабочее время |
| P4 | Косметические проблемы | 1 день | По возможности |

### Runbook ссылки

- [Полный runbook](docs/runbook.md)
- [Архитектура](docs/architecture.md)
- [API документация](docs/api.md)

---

## Дополнительные ресурсы

- [Руководство разработчика](DEVELOPER.md)
- [Инструкция по развёртыванию](DEPLOYMENT.md)

