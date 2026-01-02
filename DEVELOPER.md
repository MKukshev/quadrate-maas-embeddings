# Руководство разработчика

Это руководство описывает API интерфейсы сервиса Quadrate MAAS Embeddings и примеры интеграции в ваши приложения.

## Содержание

- [Обзор API](#обзор-api)
- [Аутентификация](#аутентификация)
- [Embeddings API](#embeddings-api)
- [Rerank API](#rerank-api)
- [Модели](#модели)
- [Примеры интеграции](#примеры-интеграции)
- [Rate Limits](#rate-limits)
- [Обработка ошибок](#обработка-ошибок)

---

## Обзор API

Сервис предоставляет OpenAI-совместимый REST API для работы с эмбеддингами и ранжированием документов.

**Base URL:** `http://localhost:8085` (или ваш production endpoint)

### Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/health/live` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe |
| `/v1/models` | GET | Список доступных моделей |
| `/v1/embeddings` | POST | Создание эмбеддингов |
| `/v1/rerank` | POST | Ранжирование документов |
| `/metrics` | GET | Prometheus метрики |

---

## Аутентификация

По умолчанию сервис работает в **анонимном режиме** — `X-API-Key` не требуется.

```bash
curl http://localhost:8085/v1/models
```

### Настройка API ключей (опционально)

Для включения аутентификации измените `configs/auth.yaml`:

```yaml
api_keys:
  - "your-api-key"
  - "another-key"
allow_anonymous_without_api_keys: false
```

При включённой аутентификации добавляйте заголовок:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8085/v1/models
```

---

## Embeddings API

### Создание эмбеддингов

**Endpoint:** `POST /v1/embeddings`

**Request:**

```json
{
  "model": "bge-m3",
  "input": "Текст для векторизации"
}
```

Или массив текстов:

```json
{
  "model": "bge-m3",
  "input": ["Первый текст", "Второй текст", "Третий текст"]
}
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, 0.789, ...],
      "index": 0
    }
  ],
  "model": "bge-m3"
}
```

### Параметры

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `model` | string | Да | Имя модели (см. [Модели](#модели)) |
| `input` | string \| array | Да | Текст или массив текстов для векторизации |

### Пример на Python

```python
import httpx

API_URL = "http://localhost:8085"

def get_embeddings(texts: list[str], model: str = "bge-m3") -> list[list[float]]:
    """Получить эмбеддинги для списка текстов."""
    response = httpx.post(
        f"{API_URL}/v1/embeddings",
        headers={"Content-Type": "application/json"},
        json={"model": model, "input": texts},
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]

# Использование
embeddings = get_embeddings(["Привет мир", "Hello world"])
print(f"Dimension: {len(embeddings[0])}")  # 1024 для bge-m3
```

### Пример на JavaScript/TypeScript

```typescript
const API_URL = "http://localhost:8085";

async function getEmbeddings(texts: string[], model = "bge-m3"): Promise<number[][]> {
  const response = await fetch(`${API_URL}/v1/embeddings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model, input: texts }),
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  const data = await response.json();
  return data.data.map((item: any) => item.embedding);
}

// Использование
const embeddings = await getEmbeddings(["Привет мир", "Hello world"]);
console.log(`Dimension: ${embeddings[0].length}`);
```

---

## Rerank API

### Ранжирование документов

**Endpoint:** `POST /v1/rerank`

Переранжирует документы по релевантности к запросу.

**Request:**

```json
{
  "model": "rerank-base",
  "query": "машинное обучение",
  "documents": [
    "Python для анализа данных",
    "Нейросети и глубокое обучение",
    "Рецепты пирогов"
  ],
  "top_k": 2
}
```

**Response:**

```json
{
  "object": "rerank",
  "model": "rerank-base",
  "data": [
    {
      "index": 1,
      "relevance_score": 9.756,
      "document": "Нейросети и глубокое обучение"
    },
    {
      "index": 0,
      "relevance_score": 5.234,
      "document": "Python для анализа данных"
    }
  ]
}
```

### Параметры

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `model` | string | Да | Имя модели (`rerank-base`) |
| `query` | string | Да | Поисковый запрос |
| `documents` | array | Да | Массив документов для ранжирования |
| `top_k` | int | Нет | Количество топ-результатов (по умолчанию = len(documents)) |

### Пример на Python

```python
import httpx

def rerank_documents(
    query: str,
    documents: list[str],
    top_k: int = 5,
    model: str = "rerank-base"
) -> list[dict]:
    """Переранжировать документы по релевантности к запросу."""
    response = httpx.post(
        f"{API_URL}/v1/rerank",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "query": query,
            "documents": documents,
            "top_k": top_k,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["data"]

# Использование
results = rerank_documents(
    query="искусственный интеллект",
    documents=[
        "Машинное обучение в Python",
        "Кулинарные рецепты",
        "Нейронные сети для NLP",
        "Садоводство для начинающих",
    ],
    top_k=2,
)

for r in results:
    print(f"[{r['index']}] Score: {r['relevance_score']:.2f} - {r['document']}")
```

---

## Модели

### Embedding модели

| Алиас | Полное имя | Размерность | Описание |
|-------|------------|-------------|----------|
| `bge-m3` | BAAI/bge-m3 | 1024 | Универсальная мультиязычная модель |
| `multilingual-e5-large` | intfloat/multilingual-e5-large | 1024 | Мультиязычная E5 |
| `frida` | ai-forever/FRIDA | 1536 | Русскоязычная модель от Сбера |

### Rerank модели

| Алиас | Полное имя | Описание |
|-------|------------|----------|
| `rerank-base` | cross-encoder/ms-marco-MiniLM-L-2-v2 | Быстрая модель ранжирования |

### Получение списка моделей

```bash
curl -s http://localhost:8085/v1/models | jq
```

```json
{
  "data": [
    {"id": "bge-m3", "object": "embedding", "enabled": true},
    {"id": "multilingual-e5-large", "object": "embedding", "enabled": true},
    {"id": "frida", "object": "embedding", "enabled": true},
    {"id": "rerank-base", "object": "rerank", "enabled": true, "max_top_k": 50}
  ]
}
```

---

## Примеры интеграции

### RAG Pipeline

```python
import httpx
import numpy as np

class EmbeddingService:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def embed(self, texts: list[str], model: str = "bge-m3") -> np.ndarray:
        response = httpx.post(
            f"{self.base_url}/v1/embeddings",
            headers=self.headers,
            json={"model": model, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        embeddings = [item["embedding"] for item in response.json()["data"]]
        return np.array(embeddings)
    
    def rerank(self, query: str, documents: list[str], top_k: int = 10) -> list[dict]:
        response = httpx.post(
            f"{self.base_url}/v1/rerank",
            headers=self.headers,
            json={
                "model": "rerank-base",
                "query": query,
                "documents": documents,
                "top_k": top_k,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["data"]


# RAG с двухэтапным поиском
class RAGPipeline:
    def __init__(self, embedding_service: EmbeddingService, vector_store):
        self.embeddings = embedding_service
        self.vector_store = vector_store
    
    def search(self, query: str, top_k: int = 5, rerank_candidates: int = 20):
        # 1. Векторный поиск
        query_embedding = self.embeddings.embed([query])[0]
        candidates = self.vector_store.search(query_embedding, k=rerank_candidates)
        
        # 2. Переранжирование
        documents = [doc.text for doc in candidates]
        reranked = self.embeddings.rerank(query, documents, top_k=top_k)
        
        # 3. Возврат топ результатов
        return [candidates[r["index"]] for r in reranked]
```

### LangChain Integration

```python
from langchain_core.embeddings import Embeddings
from typing import List
import httpx

class QuadrateEmbeddings(Embeddings):
    def __init__(self, base_url: str, model: str = "bge-m3", api_key: str | None = None):
        self.base_url = base_url
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = httpx.post(
            f"{self.base_url}/v1/embeddings",
            headers=self.headers,
            json={"model": self.model, "input": texts},
            timeout=120.0,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Использование с FAISS
from langchain_community.vectorstores import FAISS

embeddings = QuadrateEmbeddings(
    base_url="http://localhost:8085",
    model="bge-m3",
)

vectorstore = FAISS.from_texts(
    texts=["Document 1", "Document 2"],
    embedding=embeddings,
)
```

### Async Python Client

```python
import httpx
import asyncio

class AsyncEmbeddingClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
        self._client = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=60.0)
        return self
    
    async def __aexit__(self, *args):
        await self._client.aclose()
    
    async def embed(self, texts: list[str], model: str = "bge-m3") -> list[list[float]]:
        response = await self._client.post(
            f"{self.base_url}/v1/embeddings",
            headers=self.headers,
            json={"model": model, "input": texts},
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
    
    async def embed_batch(
        self, 
        all_texts: list[str], 
        batch_size: int = 32,
        model: str = "bge-m3"
    ) -> list[list[float]]:
        """Обработка больших объёмов данных батчами."""
        results = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            embeddings = await self.embed(batch, model)
            results.extend(embeddings)
        return results

# Использование
async def main():
    async with AsyncEmbeddingClient("http://localhost:8085") as client:
        texts = ["Text 1", "Text 2", "Text 3"] * 100  # 300 текстов
        embeddings = await client.embed_batch(texts, batch_size=32)
        print(f"Processed {len(embeddings)} embeddings")

asyncio.run(main())
```

---

## Rate Limits

Сервис использует token bucket алгоритм для ограничения запросов.

### Лимиты по умолчанию

| Тип | Capacity | Refill Rate |
|-----|----------|-------------|
| По умолчанию | 60 | 30 req/sec |
| Anonymous | 30 | 15 req/sec |
| admin-key | 120 | 60 req/sec |

### Обработка 429 Too Many Requests

```python
import time
import httpx

def embed_with_retry(texts: list[str], max_retries: int = 3) -> list[list[float]]:
    for attempt in range(max_retries):
        response = httpx.post(
            f"{API_URL}/v1/embeddings",
            json={"model": "bge-m3", "input": texts},
        )
        
        if response.status_code == 429:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Rate limited, waiting {wait_time}s...")
            time.sleep(wait_time)
            continue
        
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
    
    raise Exception("Max retries exceeded")
```

---

## Обработка ошибок

### Формат ошибок

```json
{
  "error": {
    "message": "Описание ошибки",
    "type": "invalid_request_error",
    "code": null
  }
}
```

### Коды ошибок

| HTTP код | Описание | Рекомендация |
|----------|----------|--------------|
| 400 | Невалидный запрос | Проверьте параметры |
| 401 | Не авторизован | Проверьте API ключ |
| 429 | Rate limit | Повторите с backoff |
| 500 | Внутренняя ошибка | Проверьте логи сервиса |
| 502 | Upstream недоступен | Проверьте статус upstream |
| 503 | Модель отключена | Используйте другую модель |
| 504 | Timeout | Уменьшите размер batch |

### Пример обработки

```python
import httpx

def safe_embed(texts: list[str]) -> list[list[float]] | None:
    try:
        response = httpx.post(
            f"{API_URL}/v1/embeddings",
            json={"model": "bge-m3", "input": texts},
            timeout=30.0,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
    
    except httpx.HTTPStatusError as e:
        error_body = e.response.json()
        print(f"API Error: {error_body['error']['message']}")
        return None
    
    except httpx.TimeoutException:
        print("Request timed out")
        return None
```

---

## Дополнительные ресурсы

- [OpenAPI Specification](api/openapi.yaml)
- [Архитектура системы](docs/architecture.md)
- [Инструкция по развёртыванию](DEPLOYMENT.md)
- [Инструкция по эксплуатации](OPERATIONS.md)

