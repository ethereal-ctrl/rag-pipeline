"""Stage timings and content-free OpenTelemetry spans for a Pipeline Variant."""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Thread

import httpx
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, StatusCode, Tracer

from rag_pipeline.index import RetrievedChunk

DEFAULT_ENDPOINT = "http://127.0.0.1:6006/v1/traces"
PROBE_TIMEOUT = 0.5


def _collector_available(endpoint: str) -> bool:
    available = Event()

    def probe() -> None:
        try:
            # An empty protobuf request checks the actual OTLP route without sending spans.
            with httpx.stream(
                "POST",
                endpoint,
                content=b"",
                headers={"Content-Type": "application/x-protobuf"},
                timeout=PROBE_TIMEOUT,
            ) as response:
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
                if response.is_success and content_type == "application/x-protobuf":
                    available.set()
        except (httpx.HTTPError, httpx.InvalidURL):
            pass

    worker = Thread(target=probe, daemon=True)
    worker.start()
    # Bound startup even if the system resolver stalls.
    worker.join(timeout=PROBE_TIMEOUT)
    return available.is_set()


@contextmanager
def cli_tracer() -> Generator[Tracer | None]:
    """Probe once, then export in batches for the duration of the CLI command."""
    endpoint = os.environ.get("RAG_TRACING_ENDPOINT", DEFAULT_ENDPOINT)
    if os.environ.get("RAG_TRACING") == "0" or not _collector_available(endpoint):
        yield None
        return
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": "rag-pipeline", "openinference.project.name": "rag-pipeline"}
        ),
        shutdown_on_exit=False,
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, timeout=1)))
    try:
        yield provider.get_tracer("rag_pipeline")
    finally:
        provider.shutdown()


@dataclass(frozen=True)
class StageTrace:
    name: str
    seconds: float
    input_tokens: int | None = 0
    output_tokens: int = 0
    model: str | None = None


@dataclass
class StageUsage:
    input_tokens: int | None = 0
    output_tokens: int = 0
    chunks: tuple[RetrievedChunk, ...] = ()

    def set_input_tokens(self, count: int | None) -> None:
        self.input_tokens = count


def set_chunk_ids(span: Span, chunks: tuple[RetrievedChunk, ...]) -> None:
    if span.is_recording():
        span.set_attribute("rag.chunk_ids", tuple(chunk.chunk_id for chunk in chunks))


class AnswerTrace:
    def __init__(self, tracer: Tracer, variant: str) -> None:
        self._tracer = tracer
        self._variant = variant
        self.stages: list[StageTrace] = []

    @contextmanager
    def span(self, name: str, kind: str) -> Generator[Span]:
        with self._tracer.start_as_current_span(
            name, record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("rag.variant", self._variant)
            span.set_attribute("openinference.span.kind", kind)
            try:
                yield span
            except Exception:
                span.set_status(StatusCode.ERROR)
                raise

    @contextmanager
    def stage(
        self,
        name: str,
        kind: str,
        *,
        model: str | None = None,
        operation: str | None = None,
    ) -> Generator[StageUsage]:
        with self.span(name, kind) as span:
            if operation is not None:
                span.set_attribute("gen_ai.operation.name", operation)
            if model is not None:
                span.set_attribute("gen_ai.request.model", model)
            usage = StageUsage()
            started = time.perf_counter()
            yield usage
            self.stages.append(
                StageTrace(
                    name,
                    time.perf_counter() - started,
                    usage.input_tokens,
                    usage.output_tokens,
                    model,
                )
            )
            if model is not None:
                if usage.input_tokens is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
            set_chunk_ids(span, usage.chunks)
