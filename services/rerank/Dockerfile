FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RERANK_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-2-v2 \
    RERANK_MODEL_MAPPING={} \
    RERANK_MAX_DOCUMENTS=50 \
    RERANK_MAX_DOCUMENT_LENGTH=4096

COPY services/rerank/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/rerank /app/services/rerank

EXPOSE 9002

CMD ["uvicorn", "services.rerank.src.main:app", "--host", "0.0.0.0", "--port", "9002"]
