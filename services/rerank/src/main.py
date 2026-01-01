from __future__ import annotations

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

from .inference import rerank_documents
from .metrics import DOCUMENT_COUNTER, REQUEST_COUNTER, REQUEST_LATENCY
from .model import RerankModel


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
    max_documents: int = Field(default=50, ge=1)
    max_document_length: int = Field(default=4096, ge=1)

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        return cls(
            model_name=os.getenv("RERANK_MODEL_NAME", cls.model_fields["model_name"].default),
            max_documents=int(os.getenv("RERANK_MAX_DOCUMENTS", cls.model_fields["max_documents"].default)),
            max_document_length=int(
                os.getenv("RERANK_MAX_DOCUMENT_LENGTH", cls.model_fields["max_document_length"].default)
            ),
        )


@dataclass
class ApplicationState:
    settings: ServiceSettings
    model: RerankModel


def error_response(message: str, status_code: int) -> JSONResponse:
    body = ErrorResponse(error=ErrorContent(message=message)).model_dump()
    return JSONResponse(content=body, status_code=status_code)


@asynccontextmanager
async def lifespan(app: FastAPI, model_loader: Callable[[ServiceSettings], RerankModel]):
    settings = ServiceSettings.from_env()
    model = model_loader(settings)
    model.warmup()
    app.state.state = ApplicationState(settings=settings, model=model)
    yield


def create_app(model_loader: Callable[[ServiceSettings], RerankModel] | None = None) -> FastAPI:
    loader = model_loader or (lambda settings: RerankModel(settings.model_name))
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
        settings: ServiceSettings = Depends(get_settings),
    ):
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
        results = rerank_documents(model, payload.query, payload.documents, top_k)
        DOCUMENT_COUNTER.labels(model=payload.model).inc(len(payload.documents))
        return {"object": "rerank", "model": payload.model, "results": results}

    return app


app = create_app()
