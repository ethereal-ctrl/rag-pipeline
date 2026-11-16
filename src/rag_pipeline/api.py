"""Local HTTP demo for the Answer seam."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from threading import Lock
from typing import Protocol

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from opentelemetry.trace import Tracer
from pydantic import BaseModel, ConfigDict, Field, field_validator
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_pipeline.agentic import StopReason
from rag_pipeline.citations import RemovedStatement
from rag_pipeline.demo import (
    DEFAULT_QDRANT_URL,
    VARIANT_NAME_PATTERN,
    VARIANTS_DIR,
    IndexBuild,
    answer_variants,
    demo_variant,
    load_variants,
)
from rag_pipeline.index import ChunkIndex
from rag_pipeline.models import (
    Embedder,
    EmbedderSpec,
    LocalModels,
    ModelProvider,
    configured_ollama_url,
    missing_remote_keys,
)
from rag_pipeline.tracing import DEFAULT_ENDPOINT, StageTrace, collector_available, exporting_tracer
from rag_pipeline.variants import Answer, PipelineVariant, VariantConfig, build_variant

MAX_QUESTION_LENGTH = 500
DEFAULT_PHOENIX_URL = "http://localhost:6006"


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    variant: str = Field(pattern=VARIANT_NAME_PATTERN, max_length=80)

    @field_validator("question")
    @classmethod
    def validate_question(cls, question: str) -> str:
        if not (stripped := question.strip()):
            raise ValueError("Question must contain text.")
        return stripped


@dataclass(frozen=True)
class CitationChunk:
    id: str
    title: str
    source: str
    published_at: str
    text: str


@dataclass(frozen=True)
class CitedStatement:
    statement: str
    chunks: tuple[CitationChunk, ...]


@dataclass(frozen=True)
class AnswerResponse:
    answer: str
    abstained: bool
    citations: tuple[CitedStatement, ...]
    removed_statements: tuple[RemovedStatement, ...]
    stages: tuple[StageTrace, ...]
    stop_reason: StopReason | None
    search_steps: int | None
    variant: str
    chosen_variant: str | None
    index_version: str | None
    trace_id: str
    phoenix_url: str


class DemoModels(ModelProvider, Protocol):
    def embedder_for(self, spec: EmbedderSpec) -> Embedder: ...


def _description(config: VariantConfig) -> str:
    rerank = config.retrieval.rerank
    reranker = rerank.reranker.model if rerank else "off"
    citations = "on" if config.generation.citations.enabled else "off"
    agentic = "on" if config.agentic is not None else "off"
    router = "on" if config.router is not None else "off"
    return (
        f"Retrieval {config.retrieval.mode} · Reranker {reranker} · Citations {citations} "
        f"· Agentic Search {agentic} · Router {router}"
    )


class _Runtime:
    def __init__(
        self,
        configs: dict[str, VariantConfig],
        qdrant: QdrantClient,
        models: DemoModels,
        tracer: Tracer,
    ) -> None:
        self.configs = configs
        self.qdrant = qdrant
        self.models = models
        self.tracer = tracer
        self.variants: dict[str, PipelineVariant] = {}
        # A Reranker or Embedder can be shared by different variants in this process.
        self.lock = Lock()

    def _index(self, config: VariantConfig) -> ChunkIndex | None:
        index = config.index
        if index is None or not config.uses_retrieval:
            return None
        embedder = self.models.embedder_for(index.embedder) if index.embedder is not None else None
        return ChunkIndex(self.qdrant, index.collection, embedder)

    def answer(self, request: AnswerRequest) -> tuple[Answer, str | None]:
        config = self.configs.get(request.variant)
        if config is None:
            raise HTTPException(404, f"Unknown variant {request.variant!r}. See GET /variants.")

        candidates = answer_variants(config)
        keys = missing_remote_keys(
            spec for candidate in candidates for spec in candidate.remote_generators()
        )
        if keys:
            raise HTTPException(503, f"Set {', '.join(keys)} to use this remote variant.")

        with self.lock:
            for candidate in candidates:
                if (
                    candidate.uses_retrieval
                    and candidate.index
                    and not self.qdrant.collection_exists(candidate.index.collection)
                ):
                    index = IndexBuild.for_variant(candidate)
                    raise HTTPException(
                        409,
                        f"Collection {index.collection!r} is missing. "
                        f"Run `{index.ingest_command()}`.",
                    )
            variant = self.variants.get(config.name)
            if variant is None:
                router_embedder = None
                if config.router is not None:
                    assert config.index is not None and config.index.embedder is not None
                    router_embedder = self.models.embedder_for(config.index.embedder)
                variant = build_variant(
                    config,
                    index=self._index(config) if config.router is None else None,
                    models=self.models,
                    tracer=self.tracer,
                    router_embedder=router_embedder,
                    index_for=self._index,
                )
                self.variants[config.name] = variant
            index_version = variant.index_version()
            return variant.answer(request.question), index_version


def _response(
    answer: Answer, variant: str, index_version: str | None, phoenix: str
) -> AnswerResponse:
    assert answer.trace_id is not None
    chunks = {chunk.chunk_id: chunk for chunk in answer.retrieved}
    citations = tuple(
        CitedStatement(
            citation.statement,
            tuple(
                CitationChunk(
                    chunk.chunk_id, chunk.title, chunk.source, chunk.published_at, chunk.text
                )
                for chunk in (chunks[id_] for id_ in citation.chunk_ids)
            ),
        )
        for citation in answer.citations
    )
    return AnswerResponse(
        answer.text,
        answer.abstained,
        citations,
        answer.removed_statements,
        answer.trace,
        answer.stop_reason,
        answer.search_steps,
        variant,
        answer.chosen_variant,
        index_version,
        answer.trace_id,
        f"{phoenix}/redirects/traces/{answer.trace_id}",
    )


def _reachable(client: httpx.Client, url: str) -> bool:
    try:
        return client.get(url).is_success
    except (httpx.HTTPError, httpx.InvalidURL):
        return False


def create_app(
    *,
    variants_dir: Path = VARIANTS_DIR,
    qdrant: QdrantClient | None = None,
    models: DemoModels | None = None,
    tracer: Tracer | None = None,
    transport: httpx.BaseTransport | None = None,
) -> FastAPI:
    runtime: _Runtime
    ollama_url: str
    qdrant_url: str
    tracing_endpoint: str
    phoenix_url: str
    default_variant: str

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        nonlocal runtime, ollama_url, qdrant_url, tracing_endpoint, phoenix_url, default_variant
        configs = load_variants(variants_dir)
        if not configs:
            raise ValueError(f"No variant configs in {variants_dir}.")
        selected = demo_variant()
        if selected not in configs and "RAG_DEMO_VARIANT" in os.environ:
            raise ValueError(f"Unknown RAG_DEMO_VARIANT {selected!r}. Check {variants_dir}.")
        default_variant = selected if selected in configs else next(iter(configs))
        ollama_url = configured_ollama_url().rstrip("/")
        qdrant_url = os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL).rstrip("/")
        tracing_endpoint = os.environ.get("RAG_TRACING_ENDPOINT", DEFAULT_ENDPOINT)
        phoenix_url = os.environ.get("PHOENIX_URL", DEFAULT_PHOENIX_URL).rstrip("/")
        with ExitStack() as resources:
            qdrant_client = qdrant
            if qdrant_client is None:
                qdrant_client = QdrantClient(url=qdrant_url, check_compatibility=False)
                resources.callback(qdrant_client.close)
            active_tracer = tracer
            if active_tracer is None:
                endpoint = None if os.environ.get("RAG_TRACING") == "0" else tracing_endpoint
                active_tracer = resources.enter_context(exporting_tracer(endpoint))
            model_provider = models if models is not None else LocalModels(ollama_url=ollama_url)
            runtime = _Runtime(configs, qdrant_client, model_provider, active_tracer)
            yield

    app = FastAPI(title="RAG Pipeline Demo", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return files("rag_pipeline").joinpath("demo.html").read_text("utf-8")

    @app.get("/variants")
    def variants() -> dict[str, str | list[dict[str, str]]]:
        return {
            "default": default_variant,
            "variants": [
                {"name": config.name, "description": _description(config)}
                for config in runtime.configs.values()
            ],
        }

    @app.get("/health")
    def health() -> dict[str, bool]:
        with httpx.Client(timeout=0.5, transport=transport) as client:
            return {
                "qdrant": _reachable(client, f"{qdrant_url}/readyz"),
                "ollama": _reachable(client, f"{ollama_url}/api/tags"),
                "trace_exporter": collector_available(tracing_endpoint, transport=transport),
            }

    @app.post("/answer")
    def answer(request: AnswerRequest) -> AnswerResponse:
        try:
            result, index_version = runtime.answer(request)
        except (ResponseHandlingException, UnexpectedResponse) as error:
            raise HTTPException(503, "Qdrant is unavailable. Check QDRANT_URL.") from error
        except httpx.HTTPError as error:
            raise HTTPException(
                503, "The model request failed. Check the model and provider."
            ) from error
        except (ValueError, OSError) as error:
            raise HTTPException(409, str(error)) from error
        return _response(result, request.variant, index_version, phoenix_url)

    return app


app = create_app()
