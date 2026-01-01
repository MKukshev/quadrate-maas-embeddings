from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .batching import RerankBatcher
from .inference import rerank_documents
from .metrics import DOCUMENT_COUNTER, ERROR_COUNTER, REQUEST_COUNTER, REQUEST_LATENCY
from .model import RerankModel

logger = logging.getLogger(__name__)


class ErrorContent(BaseModel):
    message: str
    type: str = "invalid_request_error"


class ErrorResponse(BaseModel):
    error: ErrorContent


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_k: int | None = Field(default=None, ge=1)

    @field_validator("query")
    @classmethod
    def ensure_query(cls, value: str) -> str:
        if not value:
            raise ValueError("query must not be empty")
        return value

    @field_validator("documents")
    @classmethod
    def ensure_documents(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("documents must not be empty")
        if not all(isinstance(item, str) and item for item in value):
            raise ValueError("all documents must be non-empty strings")
        return value


class ServiceSettings(BaseModel):
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-2-v2")
    model_mapping: dict[str, str] = Field(default_factory=dict)
    max_documents: int = Field(default=50, ge=1)
    max_document_length: int = Field(default=4096, ge=1)
    device_preference: str = Field(default="auto")
    mixed_precision: bool = Field(default=True)
    enable_quantization: bool = Field(default=False)
    max_batch_size: int = Field(default=8, ge=1)
    batch_delay_ms: int = Field(default=20, ge=1)
    queue_timeout_ms: int = Field(default=1000, ge=1)

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        model_name = os.getenv("RERANK_MODEL_NAME", cls.model_fields["model_name"].default)
        mapping_value = os.getenv("RERANK_MODEL_MAPPING", "").strip()
        model_mapping: dict[str, str] = {}
        if mapping_value:
            try:
                parsed_mapping = json.loads(mapping_value)
            except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
                raise ValueError("RERANK_MODEL_MAPPING must be valid JSON") from exc
            if not isinstance(parsed_mapping, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in parsed_mapping.items()
            ):
                raise ValueError("RERANK_MODEL_MAPPING must be a JSON object with string keys and values")
            model_mapping = parsed_mapping

        settings = cls(
            model_name=model_name,
            model_mapping=model_mapping,
            max_documents=int(os.getenv("RERANK_MAX_DOCUMENTS", cls.model_fields["max_documents"].default)),
            max_document_length=int(
                os.getenv("RERANK_MAX_DOCUMENT_LENGTH", cls.model_fields["max_document_length"].default)
            ),
            device_preference=os.getenv("RERANK_DEVICE", cls.model_fields["device_preference"].default),
            mixed_precision=_parse_bool(
                os.getenv("RERANK_MIXED_PRECISION", str(cls.model_fields["mixed_precision"].default))
            ),
            enable_quantization=_parse_bool(
                os.getenv("RERANK_QUANTIZATION", str(cls.model_fields["enable_quantization"].default))
            ),
            max_batch_size=int(os.getenv("RERANK_MAX_BATCH_SIZE", cls.model_fields["max_batch_size"].default)),
            batch_delay_ms=int(os.getenv("RERANK_BATCH_DELAY_MS", cls.model_fields["batch_delay_ms"].default)),
            queue_timeout_ms=int(os.getenv("RERANK_QUEUE_TIMEOUT_MS", cls.model_fields["queue_timeout_ms"].default)),
        )
        settings._validate_model_mapping()
        return settings

    def _validate_model_mapping(self) -> None:
        invalid_targets = {alias: target for alias, target in self.model_mapping.items() if target != self.model_name}
        if invalid_targets:
            targets = ", ".join(sorted(set(invalid_targets.values())))
            raise ValueError(
                f"Model mapping targets must match RERANK_MODEL_NAME ({self.model_name}); found: {targets}"
            )

    def resolve_model(self, requested_model: str) -> str | None:
        if requested_model == self.model_name:
            return self.model_name
        return self.model_mapping.get(requested_model)


@dataclass
class ApplicationState:
    settings: ServiceSettings
    model: RerankModel
    batcher: RerankBatcher


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def error_response(message: str, status_code: int) -> JSONResponse:
    body = ErrorResponse(error=ErrorContent(message=message)).model_dump()
    return JSONResponse(content=body, status_code=status_code)


@asynccontextmanager
async def lifespan(app: FastAPI, model_loader: Callable[[ServiceSettings], RerankModel]):
    settings = ServiceSettings.from_env()
    model = model_loader(settings)
    model.warmup()
    batcher = RerankBatcher(
        model.score_pairs,
        max_batch_size=settings.max_batch_size,
        max_batch_delay_ms=settings.batch_delay_ms,
        queue_timeout_ms=settings.queue_timeout_ms,
    )
    batcher.start()
    app.state.state = ApplicationState(settings=settings, model=model, batcher=batcher)
    yield
    await batcher.close()


def create_app(model_loader: Callable[[ServiceSettings], RerankModel] | None = None) -> FastAPI:
    loader = model_loader or (
        lambda settings: RerankModel(
            settings.model_name,
            device_preference=settings.device_preference,
            mixed_precision=settings.mixed_precision,
            enable_quantization=settings.enable_quantization,
        )
    )
    app = FastAPI(title="Rerank Service", lifespan=lambda app: lifespan(app, loader))

    @app.middleware("http")
    async def request_metrics_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = request_id
        timer = REQUEST_LATENCY.labels(endpoint=request.url.path, method=request.method).time()
        try:
            response = await call_next(request)
        finally:
            timer.observe_duration()
        response.headers["X-Request-Id"] = request_id
        REQUEST_COUNTER.labels(
            endpoint=request.url.path, method=request.method, status=str(response.status_code)
        ).inc()
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return error_response(message, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
        return error_response(str(exc), status_code=status.HTTP_400_BAD_REQUEST)

    def get_state(request: Request) -> ApplicationState:
        return request.app.state.state

    def get_model(state: ApplicationState = Depends(get_state)) -> RerankModel:
        return state.model

    def get_settings(state: ApplicationState = Depends(get_state)) -> ServiceSettings:
        return state.settings

    def get_batcher(state: ApplicationState = Depends(get_state)) -> RerankBatcher:
        return state.batcher

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready(state: ApplicationState = Depends(get_state)) -> Response:
        if state.model is None:
            return JSONResponse({"status": "unavailable"}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return JSONResponse({"status": "ok"})

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/rerank")
    async def rerank_endpoint(
        payload: RerankRequest,
        model: RerankModel = Depends(get_model),
        batcher: RerankBatcher = Depends(get_batcher),
        settings: ServiceSettings = Depends(get_settings),
    ):
        canonical_model = settings.resolve_model(payload.model)
        if canonical_model is None:
            ERROR_COUNTER.labels(endpoint="/v1/rerank", reason="invalid_model").inc()
            allowed_models = ", ".join(sorted({settings.model_name, *settings.model_mapping.keys()}))
            logger.warning("Unsupported rerank model requested: %s", payload.model)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported model '{payload.model}'. Allowed models: {allowed_models}",
            )
        if payload.model != canonical_model:
            logger.info("Mapping rerank model '%s' to served model '%s'", payload.model, canonical_model)

        if len(payload.documents) > settings.max_documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many documents: received {len(payload.documents)}, max is {settings.max_documents}",
            )
        too_long = [doc for doc in payload.documents if len(doc) > settings.max_document_length]
        if too_long:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Document length exceeds limit: max {settings.max_document_length} characters, "
                    f"offending count={len(too_long)}"
                ),
            )

        top_k = payload.top_k or len(payload.documents)
        top_k = min(top_k, len(payload.documents))
        try:
            results = await rerank_documents(batcher.score, payload.query, payload.documents, top_k)
        except asyncio.TimeoutError:
            ERROR_COUNTER.labels(endpoint="/v1/rerank", reason="timeout").inc()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Rerank request timed out while in queue"
            )
        except Exception as exc:
            logger.exception("Rerank inference failed: %s", exc)
            ERROR_COUNTER.labels(endpoint="/v1/rerank", reason="inference_error").inc()
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference failed")
        DOCUMENT_COUNTER.labels(model=canonical_model).inc(len(payload.documents))
        return {"object": "rerank", "model": canonical_model, "results": results}

    return app


app = create_app()
