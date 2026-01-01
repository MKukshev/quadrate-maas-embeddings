from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, field_validator

from .auth import Authenticator
from .metrics import (
    READINESS_PARTIAL_FAILURES,
    RERANK_DOCUMENTS_COUNTER,
    REQUEST_COUNTER,
    REQUEST_LATENCY,
    UPSTREAM_ERRORS,
    UPSTREAM_LATENCY,
)
from .ratelimit import RateLimiter
from .routing import RoutingConfig, UpstreamType, load_routing_config
from .settings import AppSettings, AuthSettings, RateLimitSettings, load_auth_settings, load_rate_limit_settings
from .tokenization import TokenizerProvider, count_text_tokens, truncate_text_to_tokens
from .upstream import infinity, qwen3, rerank

logger = logging.getLogger(__name__)


class ErrorContent(BaseModel):
    """Standard error body."""

    message: str
    type: str = "invalid_request_error"
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorContent


class EmbeddingsRequest(BaseModel):
    model: str
    input: list[str] | str

    @field_validator("input")
    @classmethod
    def ensure_input(cls, value: Any) -> list[str] | str:
        if isinstance(value, list) and not value:
            raise ValueError("input must not be empty")
        if isinstance(value, list) and not all(isinstance(item, str) for item in value):
            raise ValueError("input items must be strings")
        if isinstance(value, str) and not value:
            raise ValueError("input must not be empty")
        return value


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_k: Optional[int] = Field(default=None, ge=1)

    @field_validator("documents")
    @classmethod
    def ensure_documents(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("documents must not be empty")
        return value


@dataclass
class ApplicationState:
    settings: AppSettings
    routing: RoutingConfig
    auth: AuthSettings
    rate_limits: RateLimitSettings
    authenticator: Authenticator
    rate_limiter: RateLimiter
    http_client: httpx.AsyncClient
    tokenizer_provider: TokenizerProvider


def error_response(message: str, status_code: int, code: str | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorContent(message=message, code=code)).model_dump()
    return JSONResponse(content=body, status_code=status_code)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = AppSettings()
    routing = load_routing_config(settings.routing_path)
    auth = load_auth_settings(settings.auth_path)
    rate_limits = load_rate_limit_settings(settings.rate_limits_path)
    authenticator = Authenticator(auth)
    rate_limiter = RateLimiter(rate_limits)
    tokenizer_provider = TokenizerProvider()
    http_client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)

    app.state.state = ApplicationState(
        settings=settings,
        routing=routing,
        auth=auth,
        rate_limits=rate_limits,
        authenticator=authenticator,
        rate_limiter=rate_limiter,
        tokenizer_provider=tokenizer_provider,
        http_client=http_client,
    )

    yield

    await http_client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="MAAS Router", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = request_id
        timer = REQUEST_LATENCY.labels(endpoint=request.url.path, method=request.method).time()
        response: Response
        try:
            response = await call_next(request)
        except HTTPException as exc:
            response = await http_exception_handler(request, exc)
        except RequestValidationError as exc:
            response = await request_validation_exception_handler(request, exc)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception(
                "Unhandled exception for %s %s (request_id=%s)", request.method, request.url.path, request_id, exc_info=exc
            )
            response = error_response("Internal Server Error", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            timer.observe_duration()
        response.headers["X-Request-Id"] = request_id
        REQUEST_COUNTER.labels(endpoint=request.url.path, method=request.method, status=str(response.status_code)).inc()
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return error_response(str(exc.detail), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(
            "Validation error for %s %s (request_id=%s): %s",
            request.method,
            request.url.path,
            getattr(request.state, "request_id", "unknown"),
            exc,
        )
        return error_response(str(exc), status_code=status.HTTP_400_BAD_REQUEST)

    def get_state(request: Request) -> ApplicationState:
        return request.app.state.state

    def get_authenticator(state: ApplicationState = Depends(get_state)) -> Authenticator:
        return state.authenticator

    def get_rate_limiter(state: ApplicationState = Depends(get_state)) -> RateLimiter:
        return state.rate_limiter

    def get_tokenizer_provider(state: ApplicationState = Depends(get_state)) -> TokenizerProvider:
        return state.tokenizer_provider

    def api_key_dependency(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authenticator: Authenticator = Depends(get_authenticator),
    ) -> str:
        return authenticator.validate_key(x_api_key)

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    async def _check_upstream(client: httpx.AsyncClient, base_url: str) -> bool:
        try:
            response = await client.get(f"{base_url}/health/ready", timeout=2.0)
            return response.status_code < 400
        except Exception:
            return False

    @app.get("/health/ready")
    async def ready(request: Request, state: ApplicationState = Depends(get_state)) -> Response:
        routing = state.routing
        required = [route.upstream for route in routing.embeddings.values() if route.enabled]
        required += [route.upstream for route in routing.rerank.values() if route.enabled]
        if not required:
            return JSONResponse({"status": "ok"})

        checks = []
        for upstream_config in required:
            checks.append(_check_upstream(state.http_client, str(upstream_config.url)))
        results = await asyncio.gather(*checks)
        if not all(results):
            failing = [upstream_config.url.host for upstream_config, ok in zip(required, results) if not ok]
            for upstream in failing:
                READINESS_PARTIAL_FAILURES.labels(upstream=upstream).inc()
            logger.warning(
                "Readiness degraded for upstreams %s (request_id=%s)",
                ", ".join(failing),
                getattr(request.state, "request_id", "unknown"),
            )
            return JSONResponse({"status": "degraded"}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return JSONResponse({"status": "ok"})

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def list_models(state: ApplicationState = Depends(get_state)) -> dict[str, list[dict]]:
        include_disabled = state.settings.allow_disabled_models
        return {"data": state.routing.list_models(include_disabled=include_disabled)}

    @app.post("/v1/embeddings")
    async def create_embeddings(
        payload: EmbeddingsRequest,
        request: Request,
        api_key: str = Depends(api_key_dependency),
        rate_limiter: RateLimiter = Depends(get_rate_limiter),
        tokenizer_provider: TokenizerProvider = Depends(get_tokenizer_provider),
        state: ApplicationState = Depends(get_state),
    ):
        await rate_limiter.check(api_key)

        route = state.routing.embeddings.get(payload.model)
        if route is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown model")
        if not route.enabled:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model disabled")

        request_id = request.state.request_id
        tokenizer_name = state.settings.tokenizer_name or route.served_name or payload.model
        tokenizer = tokenizer_provider.get(tokenizer_name)

        input_texts = payload.input if isinstance(payload.input, list) else [payload.input]
        input_token_counts = [count_text_tokens(tokenizer, text) for text in input_texts]
        total_tokens = sum(input_token_counts)
        safe_limit = state.settings.safe_request_token_limit
        if safe_limit is not None and total_tokens > safe_limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Request token count {total_tokens} exceeds safe limit {safe_limit}",
            )

        doc_limit = state.settings.document_token_limit
        if doc_limit is not None:
            truncated_inputs = [truncate_text_to_tokens(tokenizer, text, doc_limit) for text in input_texts]
        else:
            truncated_inputs = input_texts

        upstream_payload = {"input": truncated_inputs if isinstance(payload.input, list) else truncated_inputs[0], "model": payload.model}
        try:
            if route.upstream.type == UpstreamType.INFINITY:
                served_name = getattr(route, "served_name", None)
                with UPSTREAM_LATENCY.labels(upstream=route.upstream.url.host, operation="embeddings").time():
                    result = await infinity.embeddings(
                        state.http_client,
                        route.upstream,
                        upstream_payload,
                        request_id,
                        served_model=served_name,
                    )
            elif route.upstream.type == UpstreamType.QWEN3:
                with UPSTREAM_LATENCY.labels(upstream=route.upstream.url.host, operation="embeddings").time():
                    result = await qwen3.embeddings(state.http_client, route.upstream, upstream_payload, request_id)
            else:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Unsupported upstream for embeddings: {route.upstream.type}",
                )
        except httpx.HTTPStatusError as exc:
            UPSTREAM_ERRORS.labels(
                upstream=route.upstream.url.host, operation="embeddings", status=str(exc.response.status_code)
            ).inc()
            raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc
        except httpx.RequestError as exc:
            UPSTREAM_ERRORS.labels(upstream=route.upstream.url.host, operation="embeddings", status="request_error").inc()
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        return result

    @app.post("/v1/rerank")
    async def rerank_endpoint(
        payload: RerankRequest,
        request: Request,
        api_key: str = Depends(api_key_dependency),
        rate_limiter: RateLimiter = Depends(get_rate_limiter),
        tokenizer_provider: TokenizerProvider = Depends(get_tokenizer_provider),
        state: ApplicationState = Depends(get_state),
    ):
        await rate_limiter.check(api_key)

        route = state.routing.rerank.get(payload.model)
        if route is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown model")
        if not route.enabled:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model disabled")

        requested_top_k = payload.top_k or len(payload.documents)
        if requested_top_k > route.max_top_k:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Requested top_k exceeds max_top_k")
        top_k = min(requested_top_k, len(payload.documents))

        tokenizer_name = state.settings.tokenizer_name or route.model
        tokenizer = tokenizer_provider.get(tokenizer_name)

        query_tokens = count_text_tokens(tokenizer, payload.query)
        doc_token_counts = [count_text_tokens(tokenizer, doc) for doc in payload.documents]
        total_tokens = query_tokens + sum(doc_token_counts)
        safe_limit = state.settings.safe_request_token_limit
        if safe_limit is not None and total_tokens > safe_limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Request token count {total_tokens} exceeds safe limit {safe_limit}",
            )

        doc_limit = state.settings.document_token_limit
        if doc_limit is not None:
            documents = [truncate_text_to_tokens(tokenizer, doc, doc_limit) for doc in payload.documents]
        else:
            documents = payload.documents

        upstream_payload = {
            "model": payload.model,
            "query": payload.query,
            "documents": documents,
            "top_k": top_k,
        }
        request_id = request.state.request_id
        try:
            if route.upstream.type == UpstreamType.RERANK:
                with UPSTREAM_LATENCY.labels(upstream=route.upstream.url.host, operation="rerank").time():
                    result = await rerank.rerank(state.http_client, route.upstream, upstream_payload, request_id)
            else:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Unsupported upstream for rerank: {route.upstream.type}",
                )
        except httpx.HTTPStatusError as exc:
            UPSTREAM_ERRORS.labels(
                upstream=route.upstream.url.host, operation="rerank", status=str(exc.response.status_code)
            ).inc()
            raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc
        except httpx.RequestError as exc:
            UPSTREAM_ERRORS.labels(upstream=route.upstream.url.host, operation="rerank", status="request_error").inc()
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        RERANK_DOCUMENTS_COUNTER.labels(model=payload.model).inc(len(payload.documents))
        return result

    return app


app = create_app()
