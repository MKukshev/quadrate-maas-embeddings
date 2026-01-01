FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ROUTER_ROUTING_PATH=/app/configs/routing.yaml \
    ROUTER_AUTH_PATH=/app/configs/auth.yaml \
    ROUTER_RATE_LIMITS_PATH=/app/configs/rate_limits.yaml \
    ROUTER_REQUEST_TIMEOUT_SECONDS=30 \
    ROUTER_ALLOW_DISABLED_MODELS=false

COPY services/router/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/router /app/services/router
COPY configs /app/configs

EXPOSE 8000

CMD ["uvicorn", "services.router.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
