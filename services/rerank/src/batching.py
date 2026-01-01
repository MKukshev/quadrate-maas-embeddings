from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, List

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    query: str
    documents: list[str]
    future: "asyncio.Future[list[float]]"


class RerankBatcher:
    def __init__(
        self,
        score_fn: Callable[[list[tuple[str, str]]], list[float]],
        *,
        max_batch_size: int = 8,
        max_batch_delay_ms: int = 20,
        queue_timeout_ms: int = 1000,
    ):
        self._score_fn = score_fn
        self._queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self._max_batch_size = max_batch_size
        self._max_batch_delay = max_batch_delay_ms / 1000
        self._queue_timeout = queue_timeout_ms / 1000
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker(), name="rerank-batcher")

    async def close(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    async def score(self, query: str, documents: list[str]) -> list[float]:
        future: "asyncio.Future[list[float]]" = asyncio.get_running_loop().create_future()
        await self._queue.put(BatchRequest(query=query, documents=documents, future=future))
        try:
            return await asyncio.wait_for(future, timeout=self._queue_timeout)
        except asyncio.TimeoutError:
            if not future.done():
                future.cancel()
            raise

    async def _worker(self) -> None:
        while True:
            request = await self._queue.get()
            batch = [request]
            deadline = time.perf_counter() + self._max_batch_delay
            while len(batch) < self._max_batch_size:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    another = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    batch.append(another)
                except asyncio.TimeoutError:
                    break
            await self._process_batch(batch)

    async def _process_batch(self, batch: List[BatchRequest]) -> None:
        pairs: list[tuple[str, str]] = []
        counts: list[int] = []
        for request in batch:
            counts.append(len(request.documents))
            pairs.extend([(request.query, doc) for doc in request.documents])

        try:
            scores = await asyncio.to_thread(self._score_fn, pairs)
        except Exception as exc:
            logger.exception("Failed to score rerank batch: %s", exc)
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)
            return

        offset = 0
        for request, count in zip(batch, counts):
            slice_scores = scores[offset : offset + count]
            offset += count
            if not request.future.done():
                request.future.set_result(slice_scores)
