"""LightRAG comparator outside the Ablation Ladder."""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import time
from collections.abc import Callable, Coroutine, Sequence
from collections.abc import Generator as Yield
from contextlib import closing, contextmanager
from contextvars import copy_context
from dataclasses import dataclass, replace
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast

import yaml
from opentelemetry.trace import NoOpTracer, Tracer
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    TypeAdapter,
    model_validator,
)

from rag_pipeline.cache import ModelResponse, ResponseCache, cache_key, measure_call
from rag_pipeline.corpus import DATASET_REVISION, Article, article_hash
from rag_pipeline.cost import load_prices, notional_cost
from rag_pipeline.evaluation import (
    EvaluationScope,
    MetricDelta,
    RunResult,
    compare_runs,
    write_text,
)
from rag_pipeline.index import RetrievedChunk, corpus_hash
from rag_pipeline.models import (
    Completion,
    DocumentEmbedderWithUsage,
    Embedder,
    EmbedderSpec,
    Generator,
    GeneratorSpec,
    StructuredGenerator,
    estimate_tokens,
    generation_key,
    model_version,
    query_embedding,
)
from rag_pipeline.rate_limit import quota_hours, quota_wait_seconds, remote_limits
from rag_pipeline.tracing import AnswerTrace, span_trace_id
from rag_pipeline.variants import (
    ABSTENTION_TEXT,
    SYSTEM_PROMPT,
    Answer,
    GenerationConfig,
    RetrievalConfig,
    VariantConfig,
    is_abstention,
)

LIGHTRAG_VERSION = "1.5.7"
GRAPH_DIR = Path("data/graphrag")
BUILD_COMPLETE = "build-complete"
type GraphMode = Literal["mix", "hybrid"]
type GraphQueryType = Literal["inference_query", "comparison_query"]
_COMPLETION = TypeAdapter(Completion)
_DOCUMENTS = TypeAdapter(tuple[list[list[float]], int | None])
_JSON_OBJECT = {"type": "object"}


class GraphConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    extraction: GeneratorSpec
    answer: GeneratorSpec
    embedder: EmbedderSpec
    embedding_dimensions: PositiveInt = 1024
    mode: GraphMode = "mix"
    query_types: tuple[GraphQueryType, ...] = Field(min_length=1)
    context_tokens: PositiveInt = 4000
    chunk_tokens: PositiveInt = 1200
    chunk_overlap: NonNegativeInt = 100
    gleanings: Literal[0, 1] = 1
    prompt_tokens: PositiveInt = 3000

    @model_validator(mode="after")
    def _check_settings(self) -> Self:
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("chunk_overlap must be less than chunk_tokens.")
        if self.answer.provider != "ollama":
            raise ValueError("GraphRAG answers and retrieval keywords use the local ladder model.")
        if len(set(self.query_types)) != len(self.query_types):
            raise ValueError("GraphRAG query_types must be unique.")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> GraphConfig:
        return cls.model_validate(yaml.safe_load(path.read_text("utf-8")))

    def build_id(self, articles: list[Article]) -> str:
        settings = self.model_dump(
            mode="json",
            include={
                "extraction",
                "embedder",
                "embedding_dimensions",
                "chunk_tokens",
                "chunk_overlap",
                "gleanings",
            },
        )
        digest = corpus_hash({article.id: article_hash(article) for article in articles})
        return cache_key("lightrag-build-v1", LIGHTRAG_VERSION, DATASET_REVISION, digest, settings)[
            :16
        ]


def _article_text(article: Article) -> str:
    return f"{article.title}\n\n{article.body}".strip()


@dataclass(frozen=True)
class BuildEstimate:
    articles: int
    chunks: int
    llm_calls: int
    estimated_tokens: int
    quota_hours: float | None
    quota_wait_seconds: float = 0

    @property
    def quota_days(self) -> int | None:
        return math.ceil(self.quota_hours / 24) if self.quota_hours is not None else None


def estimate_build(config: GraphConfig, articles: list[Article]) -> BuildEstimate:
    """Estimate extraction and gleaning from a fresh quota before creating any models."""
    stride = config.chunk_tokens - config.chunk_overlap
    chunks = sum(
        max(
            1,
            math.ceil(
                (estimate_tokens("", _article_text(article), 0) - config.chunk_overlap) / stride
            ),
        )
        for article in articles
    )
    calls = chunks * (1 + config.gleanings)
    # Assume full chunks and allow for the extraction instructions and previous turns.
    tokens = calls * (
        config.chunk_tokens + config.prompt_tokens + config.extraction.max_output_tokens
    )
    spec = config.extraction
    limits = remote_limits(spec.remote_provider, spec.model) if spec.remote_provider else None
    return BuildEstimate(
        len(articles),
        chunks,
        calls,
        tokens,
        quota_hours(limits, calls, tokens) if limits else None,
        quota_wait_seconds(limits, calls, tokens) if limits else 0,
    )


class GraphModels(Protocol):
    def generator_for(self, spec: GeneratorSpec) -> Generator: ...
    def embedder_for(self, spec: EmbedderSpec) -> Embedder: ...


class GraphModelCalls:
    """Route LightRAG's model callbacks through the project's model clients and cache."""

    def __init__(
        self,
        spec: GeneratorSpec,
        generator: Generator,
        embedder_spec: EmbedderSpec,
        embedder: Embedder,
        cache: ResponseCache,
        trace: AnswerTrace,
    ) -> None:
        self.spec = spec
        self.generator = generator
        self._embedder_spec = embedder_spec
        self._embedder = embedder
        self._cache = cache
        self.trace = trace
        self.latency_adjustment = 0.0

    def _count_replay[T](self, response: ModelResponse[T], started: float) -> None:
        if response.cache_hit:
            self.latency_adjustment += response.seconds - (time.perf_counter() - started)

    def _completion(
        self,
        system: str,
        prompt: str,
        json_object: bool,
    ) -> ModelResponse[Completion]:
        generator = self.generator
        key = generation_key(self.spec, generator.name, model_version(generator), system, prompt)
        call = partial(generator.complete, system, prompt)
        if json_object and isinstance(generator, StructuredGenerator):
            key = cache_key(key, _JSON_OBJECT)
            call = partial(generator.complete_structured, system, prompt, _JSON_OBJECT)

        def complete() -> ModelResponse[Completion]:
            response = measure_call(call)
            seconds = response.value.call_seconds
            return response if seconds is None else ModelResponse(response.value, seconds)

        return self._cache.get_or_call(key, _COMPLETION, complete)

    async def llm(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Sequence[dict[str, Any]] | None = None,
        *,
        response_format: dict[str, Any] | None = None,
        stage_name: str = "graph_llm",
        **_: Any,
    ) -> str:
        if history_messages:
            # The generator interface has two messages. Preserve turn order and roles as data.
            history = json.dumps(list(history_messages), ensure_ascii=False)
            history = history.replace("<", "\\u003c").replace(">", "\\u003e")
            prompt = f"Previous turns (JSON):\n{history}\n\n{prompt}"
        json_object = response_format is not None and response_format.get("type") == "json_object"
        with self.trace.stage(stage_name, "LLM", model=self.spec.model, operation="chat") as stage:
            started = time.perf_counter()
            response = await asyncio.to_thread(
                self._completion, system_prompt or "", prompt, json_object
            )
            self._count_replay(response, started)
            stage.seconds = response.seconds
            stage.cache_hit = response.cache_hit
            stage.set_completion(response.value)
            return response.value.text

    def _documents(self, texts: list[str]) -> ModelResponse[tuple[list[list[float]], int | None]]:
        embedder = self._embedder
        key = cache_key(
            "graph_embed_documents",
            self._embedder_spec.model_dump(),
            embedder.name,
            model_version(embedder),
            texts,
        )

        def embed() -> tuple[list[list[float]], int | None]:
            if isinstance(embedder, DocumentEmbedderWithUsage):
                return embedder.embed_documents_with_usage(texts)
            return embedder.embed_documents(texts), None

        return self._cache.get_or_call(key, _DOCUMENTS, lambda: measure_call(embed))

    async def embed(self, texts: list[str], *, context: str = "document") -> list[list[float]]:
        vectors: list[list[float]] = []
        stage_name = "graph_embed_query" if context == "query" else "graph_embed_documents"
        model = self._embedder_spec.model
        if not texts:
            return vectors
        if context == "query":
            for text in texts:
                with self.trace.stage(
                    stage_name, "EMBEDDING", model=model, operation="embeddings"
                ) as stage:
                    started = time.perf_counter()
                    response = await asyncio.to_thread(
                        query_embedding,
                        self._embedder,
                        self._embedder_spec,
                        text,
                        self._cache,
                    )
                    self._count_replay(response, started)
                    vector, stage.input_tokens = response.value
                    stage.seconds = response.seconds
                    stage.cache_hit = response.cache_hit
                    vectors.append(vector)
            return vectors
        with self.trace.stage(
            stage_name, "EMBEDDING", model=model, operation="embeddings"
        ) as stage:
            docs = await asyncio.to_thread(self._documents, texts)
            vectors, stage.input_tokens = docs.value
            stage.seconds = docs.seconds
            stage.cache_hit = docs.cache_hit
        return vectors


class GraphBackend(Protocol):
    async def initialize(self) -> None: ...
    async def finalize(self) -> None: ...
    async def insert(self, articles: list[str], ids: list[str], urls: list[str]) -> int: ...
    async def context(self, question: str) -> str: ...


type GraphFactory = Callable[[Path, GraphConfig, GraphModelCalls], GraphBackend]


class _LightRAG(Protocol):
    async def initialize_storages(self) -> None: ...
    async def finalize_storages(self) -> None: ...
    async def ainsert(self, input: list[str], *, ids: list[str], file_paths: list[str]) -> str: ...
    async def get_processing_status(self) -> dict[str, int]: ...
    async def aquery_llm(
        self,
        query: str,
        *,
        param: object,
        system_prompt: str,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _LightRagBackend:
    rag: _LightRAG
    config: GraphConfig
    query_param: Callable[..., object]

    async def initialize(self) -> None:
        await self.rag.initialize_storages()

    async def finalize(self) -> None:
        await self.rag.finalize_storages()

    async def insert(self, articles: list[str], ids: list[str], urls: list[str]) -> int:
        await self.rag.ainsert(articles, ids=ids, file_paths=urls)
        status = await self.rag.get_processing_status()
        return status.get("processed", 0)

    async def context(self, question: str) -> str:
        param = self.query_param(
            mode=self.config.mode,
            only_need_context=True,
            stream=False,
            max_total_tokens=self.config.context_tokens,
            enable_rerank=False,
        )
        answer = await self.rag.aquery_llm(question, param=param, system_prompt=SYSTEM_PROMPT)
        if answer.get("status") == "failure":
            if answer.get("metadata", {}).get("failure_reason") == "no_results":
                return ""
            raise RuntimeError(f"LightRAG retrieval failed. {answer.get('message', '')}")
        response = answer.get("llm_response", {})
        context = response.get("content")
        if response.get("llm_generated") or not isinstance(context, str):
            raise RuntimeError("LightRAG did not return retrieval-only context.")
        return context


def load_light_rag(*, workers: int = 1) -> GraphFactory:
    """Load the optional package only when opening a real graph."""
    try:
        package = import_module("lightrag")
        utils = import_module("lightrag.utils")
    except ModuleNotFoundError as error:
        if error.name != "lightrag":
            raise
        raise RuntimeError("Install LightRAG with `uv sync --locked --extra graphrag`.") from error

    create = cast("Callable[..., _LightRAG]", package.LightRAG)
    embedding_func = cast("Callable[..., object]", utils.EmbeddingFunc)
    query_param = cast("Callable[..., object]", package.QueryParam)

    def factory(path: Path, config: GraphConfig, calls: GraphModelCalls) -> GraphBackend:
        import numpy as np

        # LightRAG deep-copies its configuration. A function keeps our clients out of that copy.
        async def llm(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: Sequence[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            return await calls.llm(prompt, system_prompt, history_messages, **kwargs)

        async def embed(texts: list[str], context: str = "document"):
            return np.asarray(await calls.embed(texts, context=context), dtype=np.float32)

        rag = create(
            working_dir=str(path),
            workspace=path.name,
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
            doc_status_storage="JsonDocStatusStorage",
            llm_model_func=llm,
            llm_model_name=calls.spec.model,
            embedding_func=embedding_func(
                embedding_dim=config.embedding_dimensions,
                func=embed,
                supports_asymmetric=True,
            ),
            chunk_token_size=config.chunk_tokens,
            chunk_overlap_token_size=config.chunk_overlap,
            entity_extract_max_gleaning=config.gleanings,
            entity_extraction_use_json=False,
            enable_llm_cache=False,
            # Keep LightRAG's extraction checkpoints so it can recover interrupted merges.
            enable_llm_cache_for_entity_extract=True,
            llm_model_max_async=workers,
            embedding_func_max_async=1,
            max_parallel_insert=workers,
            vlm_process_enable=False,
        )
        return _LightRagBackend(rag, config, query_param)

    return factory


@dataclass(frozen=True)
class GraphSession:
    runner: asyncio.Runner
    backend: GraphBackend
    calls: GraphModelCalls
    tracer: Tracer

    def run[T](self, work: Coroutine[Any, Any, T]) -> T:
        return self.runner.run(work, context=copy_context())


@contextmanager
def graph_session(
    config: GraphConfig,
    path: Path,
    spec: GeneratorSpec,
    models: GraphModels,
    cache: ResponseCache,
    tracer: Tracer | None,
    factory: GraphFactory,
) -> Yield[GraphSession]:
    tracer = tracer if tracer is not None else NoOpTracer()
    calls = GraphModelCalls(
        spec,
        models.generator_for(spec),
        config.embedder,
        models.embedder_for(config.embedder),
        cache,
        AnswerTrace(tracer, "graphrag"),
    )
    with asyncio.Runner() as runner:
        backend = factory(path, config, calls)
        runner.run(backend.initialize())
        try:
            yield GraphSession(runner, backend, calls, tracer)
        finally:
            runner.run(backend.finalize())


@contextmanager
def lock_graph(directory: Path) -> Yield[None]:
    """Keep two CLI processes from writing the same local graph at once."""
    directory.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(directory / "operation.sqlite3", timeout=0)) as db:
        try:
            db.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode != sqlite3.SQLITE_BUSY:
                raise
            raise RuntimeError(f"Graph {directory.name} is in use by another process.") from error
        yield


class BuildReport(BaseModel):
    build_id: str
    corpus_revision: str = DATASET_REVISION
    extraction: GeneratorSpec
    articles: int
    processed: int = 0
    complete: bool = False
    seconds: float = 0
    llm_calls: int = 0
    embedding_calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    notional_cost_usd: float | None = 0.0

    @classmethod
    def load(cls, folder: Path, build_id: str) -> BuildReport | None:
        path = folder / f"{build_id}.json"
        return cls.model_validate_json(path.read_text("utf-8")) if path.exists() else None


def build_graph(
    config: GraphConfig,
    articles: list[Article],
    graph_root: Path,
    report_root: Path,
    models: GraphModels,
    cache: ResponseCache,
    tracer: Tracer | None,
    factory: GraphFactory,
) -> BuildReport:
    if not articles:
        raise ValueError("The Corpus has no Articles.")
    build_id = config.build_id(articles)
    directory = graph_root / build_id
    with lock_graph(directory):
        report = BuildReport.load(report_root, build_id) or BuildReport(
            build_id=build_id,
            articles=len(articles),
            extraction=config.extraction,
        )
        marker = directory / BUILD_COMPLETE
        if report.complete and marker.is_file():
            return report
        report.complete = False
        trace = AnswerTrace(tracer if tracer is not None else NoOpTracer(), "graphrag")
        started = time.perf_counter()
        try:
            with (
                trace.span("graphrag.build", "CHAIN"),
                graph_session(
                    config,
                    directory,
                    config.extraction,
                    models,
                    cache,
                    tracer,
                    factory,
                ) as session,
            ):
                session.calls.trace = trace
                report.processed = session.run(
                    session.backend.insert(
                        [_article_text(article) for article in articles],
                        [article.id for article in articles],
                        [article.url for article in articles],
                    )
                )
            if report.processed != len(articles):
                raise RuntimeError(
                    f"LightRAG processed {report.processed}/{len(articles)} Articles. "
                    "Rerun the same build to resume."
                )
            report.complete = True
        finally:
            report.seconds += time.perf_counter() - started
            report.llm_calls += sum(stage.name == "graph_llm" for stage in trace.stages)
            report.embedding_calls += sum(
                stage.name.startswith("graph_embed") for stage in trace.stages
            )
            report.cache_hits += sum(stage.cache_hit for stage in trace.stages)
            report.input_tokens += sum(stage.input_tokens or 0 for stage in trace.stages)
            report.output_tokens += sum(stage.output_tokens for stage in trace.stages)
            uncached = tuple(stage for stage in trace.stages if not stage.cache_hit)
            cost = notional_cost(uncached, load_prices()) if uncached else 0.0
            report.notional_cost_usd = (
                report.notional_cost_usd + cost
                if report.notional_cost_usd is not None and cost is not None
                else None
            )
            write_text(report_root / f"{build_id}.json", report.model_dump_json(indent=2) + "\n")
        marker.touch()
        return report


class GraphVariant:
    cache_enabled = True

    def __init__(self, graph: GraphConfig, build_id: str, session: GraphSession) -> None:
        self._graph = graph
        self._build_id = build_id
        self._session = session
        self.config = VariantConfig(
            name="graphrag",
            retrieval=RetrievalConfig(mode="none"),
            generation=GenerationConfig(generator=graph.answer),
        )
        self.generator = session.calls.generator

    def index_version(self) -> str:
        return f"graphrag-{self._build_id}"

    def evaluation_scope(self) -> EvaluationScope:
        return EvaluationScope(
            "dev",
            self._graph.query_types,
            {
                "graphrag": {
                    "build_id": self._build_id,
                    "lightrag_version": LIGHTRAG_VERSION,
                    "mode": self._graph.mode,
                    "context_tokens": self._graph.context_tokens,
                    "extraction": self._graph.extraction.model_dump(mode="json"),
                    "embedder": self._graph.embedder.model_dump(mode="json"),
                }
            },
        )

    def retrieve(
        self,
        question: str,
        stages: dict[str, float] | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        raise ValueError("GraphRAG returns unranked context and only supports answer evaluation.")

    def answer(self, question: str) -> Answer:
        session = self._session
        trace = AnswerTrace(session.tracer, self.config.name)
        session.calls.trace = trace
        session.calls.latency_adjustment = 0.0
        started = time.perf_counter()
        text = ""
        with trace.span("answer", "CHAIN") as root:
            with trace.stage("graph_retrieve", "RETRIEVER"):
                context = session.run(session.backend.context(question)).strip()
            if context:
                context = context.replace("<", "&lt;").replace(">", "&gt;")
                prompt = f"<context>\n{context}\n</context>\n\nQuestion: {question}"
                text = session.run(
                    session.calls.llm(prompt, SYSTEM_PROMPT, stage_name="generate")
                ).strip()
        abstained = not text or is_abstention(text)
        return Answer(
            ABSTENTION_TEXT if abstained else text,
            abstained,
            (),
            sum(stage.input_tokens or 0 for stage in trace.stages),
            sum(stage.output_tokens for stage in trace.stages),
            max(0.0, time.perf_counter() - started + session.calls.latency_adjustment),
            stage_seconds={stage.name: stage.seconds for stage in trace.stages},
            trace=tuple(trace.stages),
            trace_id=span_trace_id(root),
        )


@dataclass(frozen=True)
class GraphComparison:
    build_id: str
    shared: int
    deltas: tuple[MetricDelta, ...]


def paired_comparison(graph: RunResult, ladder: RunResult) -> GraphComparison:
    if (
        "graphrag" not in graph.meta
        or "graphrag" in ladder.meta
        or any(
            run.meta["split"] != "dev" or run.meta.get("retrieval_only") for run in (graph, ladder)
        )
    ):
        raise ValueError("Compare a GraphRAG answer run with a ladder answer run on the Dev Split.")
    allowed = set(graph.meta["query_types"])
    ladder_ids = {
        (row.query_id, row.query_type) for row in ladder.records if row.query_type in allowed
    }
    graph_ids = {(row.query_id, row.query_type) for row in graph.records}
    shared = ladder_ids & graph_ids
    if not shared:
        raise ValueError("The GraphRAG and ladder runs share no matching Dev queries.")
    base = replace(
        ladder,
        records=tuple(row for row in ladder.records if (row.query_id, row.query_type) in shared),
    )
    candidate = replace(
        graph,
        records=tuple(row for row in graph.records if (row.query_id, row.query_type) in shared),
    )
    metrics = {"accuracy", "latency_p50", "latency_p95", "cost_per_question"}
    deltas = tuple(delta for delta in compare_runs(base, candidate) if delta.metric in metrics)
    return GraphComparison(graph.meta["graphrag"]["build_id"], len(shared), deltas)
