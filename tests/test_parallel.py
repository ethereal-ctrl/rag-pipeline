"""Tests through Evaluate and the shared SQLite quota without starting local services."""

import json
import multiprocessing
import sqlite3
import time
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, Lock
from typing import Protocol, cast

import pytest
from conftest import FTX_QUERY
from fakes import GroundedFakeGenerator, ScriptedVariant
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from rag_pipeline import cli
from rag_pipeline.cache import ResponseCache
from rag_pipeline.corpus import BenchmarkQuery
from rag_pipeline.evaluation import QueryRecord, RunResult, RunStore, evaluate
from rag_pipeline.index import ChunkIndex, RetrievedChunk, TokenCount
from rag_pipeline.judge_evaluation import judge_run
from rag_pipeline.judge_model import JudgeRequest, JudgeVerdict
from rag_pipeline.ladder import LadderRung, run_ladder
from rag_pipeline.models import EmbedderSpec, GeneratorSpec, RerankerSpec
from rag_pipeline.rate_limit import DAY, RateLimiter, RateLimits
from rag_pipeline.variants import (
    Answer,
    GenerationConfig,
    IndexConfig,
    RerankConfig,
    RetrievalConfig,
    VariantConfig,
    build_variant,
)

MODEL = "gemma-4-31b-it"
CONFIG = Path("configs/variants/closed-book-gemma.yaml")
SPEC = GeneratorSpec(provider="gemini", model=MODEL)
QUERIES = [
    replace(FTX_QUERY, id=f"q{i}", question=f"question {i}?", answer="Yes") for i in range(6)
]


class RemoteFixture(ScriptedVariant):
    def __init__(self) -> None:
        super().__init__(
            "remote", {q.question: ("Yes", []) for q in QUERIES}, retrieval_mode="none"
        )
        self.config = self.config.model_copy(
            update={"generation": GenerationConfig(generator=SPEC)}
        )
        self.barrier: Barrier | None = None

    def answer(self, question: str) -> Answer:
        if self.barrier is not None:
            self.barrier.wait(timeout=3)
            time.sleep(0.005)
        return super().answer(question)


def test_parallel_matches_ordered_sequential_scoring(tmp_path: Path):
    store = RunStore(tmp_path)
    variant = RemoteFixture()
    serial = evaluate(variant, QUERIES, "dev", store)
    variant.barrier = Barrier(3)
    parallel = evaluate(variant, QUERIES, "dev", store, workers=3)
    assert serial.records == parallel.records == store.load(parallel.run_id).records
    assert serial.metrics == parallel.metrics
    assert serial.intervals == parallel.intervals
    assert serial.meta["workers"] == 1 and parallel.meta["workers"] == 3
    assert parallel.meta["wall_seconds"] > 0


def test_out_of_order_completion_and_torn_tail_resume(tmp_path: Path):
    later_saved = Event()

    class SignallingStore(RunStore):
        def append(self, run_id: str, record: QueryRecord) -> None:
            super().append(run_id, record)
            if record.query_id == QUERIES[1].id:
                later_saved.set()

    class CrashAfterLater(RemoteFixture):
        def answer(self, question: str) -> Answer:
            if question == QUERIES[0].question:
                assert later_saved.wait(timeout=3)
                raise RuntimeError("simulated crash")
            return super().answer(question)

    store = SignallingStore(tmp_path)
    with pytest.raises(RuntimeError, match="simulated crash"):
        evaluate(CrashAfterLater(), QUERIES[:2], "dev", store, workers=2)
    (pending,) = store.runs()
    assert [record.query_id for record in store.records_so_far(pending.run_id)] == ["q1"]

    file = tmp_path / pending.run_id / "records.jsonl"
    with file.open("ab") as out:
        out.write(b'{"query_id": "q0", "answer": "\\xff')

    class InspectBeforeFinish(SignallingStore):
        def finish(self, result: RunResult) -> None:
            assert {row.query_id for row in self.records_so_far(result.run_id)} == {"q0", "q1"}
            super().finish(result)

    resumed_variant = RemoteFixture()
    resumed = evaluate(resumed_variant, QUERIES[:2], "dev", InspectBeforeFinish(tmp_path))
    assert resumed.run_id == pending.run_id
    assert [record.query_id for record in resumed.records] == ["q0", "q1"]
    assert resumed_variant.asked == [QUERIES[0].question]


def test_eval_cli_refuses_local_and_caps_provider_workers():
    runner = CliRunner()
    local = runner.invoke(cli.app, ["eval", "configs/variants/closed-book.yaml", "--workers", "2"])
    assert local.exit_code == 1 and "remote generators" in local.output
    capped = runner.invoke(cli.app, ["eval", str(CONFIG), "--workers", "31"])
    assert capped.exit_code == 1 and "30 requests/minute" in capped.output


def test_remote_dry_run_reports_parallel_wall_time(monkeypatch: pytest.MonkeyPatch):
    def queries(_split: str) -> list[BenchmarkQuery]:
        return QUERIES

    monkeypatch.setattr(cli, "_split_queries", queries)
    result = CliRunner().invoke(
        cli.app,
        ["eval", str(CONFIG), "--dry-run", "--no-cache", "--workers", "3", "--call-latency", "20"],
    )
    assert result.exit_code == 0, result.output
    assert "40.0 s (3 workers, 20 s/call)" in result.output
    assert "--workers 8" in result.output


class ClockCounter(Protocol):
    value: float

    def get_lock(self) -> AbstractContextManager[object]: ...


class SharedClock:
    def __init__(self, counter: ClockCounter):
        self.counter = counter

    def time(self) -> float:
        with self.counter.get_lock():
            return self.counter.value

    def sleep(self, seconds: float, /) -> None:
        assert seconds > 0
        with self.counter.get_lock():
            self.counter.value += seconds


def reserve_many(limiter: RateLimiter, count: int) -> None:
    for _ in range(count):
        limiter.reserve(MODEL, 3)


class StartBarrier(Protocol):
    def wait(self, timeout: float | None = None) -> int: ...


def reserve_in_process(
    path: Path,
    counter: ClockCounter,
    limits: RateLimits,
    barrier: StartBarrier,
) -> None:
    limiter = RateLimiter(path, limits, clock=SharedClock(counter))
    barrier.wait(timeout=8)
    reserve_many(limiter, 20)


@pytest.mark.parametrize("processes", [False, True])
@pytest.mark.parametrize("tokens_per_minute", [8, 30])
def test_concurrent_quota_reservations_keep_rolling_windows(
    tmp_path: Path,
    processes: bool,
    tokens_per_minute: int,
):
    ctx = multiprocessing.get_context("spawn")
    counter = cast("ClockCounter", ctx.Value("d", 0.0))
    path = tmp_path / "quota.sqlite3"
    limits = RateLimits(
        requests_per_minute=3,
        tokens_per_minute=tokens_per_minute,
        requests_per_day=9,
        requests_per_month=1000,
    )
    limiter = RateLimiter(path, limits, clock=SharedClock(counter))
    if processes:
        barrier = ctx.Barrier(2)
        children = [
            ctx.Process(target=reserve_in_process, args=(path, counter, limits, barrier))
            for _ in range(2)
        ]
        for child in children:
            child.start()
        for child in children:
            child.join(timeout=12)
            if child.is_alive():
                child.terminate()
                child.join()
            assert child.exitcode == 0
    else:
        with ThreadPoolExecutor(max_workers=8) as executor:
            tasks = [executor.submit(reserve_many, limiter, 5) for _ in range(8)]
            for task in tasks:
                task.result()
    with closing(sqlite3.connect(path)) as db:
        rows: list[tuple[float, int]] = db.execute(
            "SELECT at, tokens FROM remote_requests ORDER BY at"
        ).fetchall()
    assert len(rows) == 40
    for at, _ in rows:
        recent = [tokens for when, tokens in rows if at - 60 < when <= at]
        assert len(recent) <= 3 and sum(recent) <= tokens_per_minute
        assert sum(at - DAY < when <= at for when, _ in rows) <= 9


class Activity:
    def __init__(self) -> None:
        self.lock = Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    @contextmanager
    def entered(self) -> Generator[None]:
        with self.lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1


@pytest.mark.parametrize("cached", [False, True])
def test_parallel_queries_serialize_local_embedding_and_reranking(tmp_path: Path, cached: bool):
    embeddings, reranks, remote = Activity(), Activity(), Activity()
    simultaneous_answers = Barrier(2)

    class Embedder:
        name = "fixture-embedder"

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            with embeddings.entered():
                time.sleep(0.005)
                return [1.0]

    class Reranker:
        name = "fixture-reranker"

        def score(self, query: str, texts: list[str]) -> list[float]:
            with reranks.entered():
                time.sleep(0.005)
                return [1.0 for _ in texts]

    class Generator(GroundedFakeGenerator):
        def complete(self, system: str, prompt: str):
            with remote.entered():
                simultaneous_answers.wait(timeout=3)
                return super().complete(system, prompt)

    class Models:
        def generator_for(self, spec: GeneratorSpec) -> Generator:
            return Generator({"fixture evidence": "Yes"})

        def reranker_for(self, spec: RerankerSpec) -> Reranker:
            return Reranker()

    class Index(ChunkIndex):
        def version(self) -> str:
            return "fixture@fixed"

        def search_hybrid(
            self,
            query: str,
            limit: int,
            candidates: int,
            *,
            on_input_tokens: Callable[[TokenCount], None] | None = None,
            embedding: list[float] | None = None,
            dense_query: str | None = None,
            source: str | None = None,
        ) -> list[RetrievedChunk]:
            assert embedding == [1.0]
            return [RetrievedChunk("c", "a", "fixture", "news", "2023", "fixture evidence", 1.0)]

    config = VariantConfig(
        name="hybrid-remote",
        index=IndexConfig(
            collection="fixture", embedder=EmbedderSpec(provider="ollama", model="local-embed")
        ),
        retrieval=RetrievalConfig(
            mode="hybrid",
            rerank=RerankConfig(
                reranker=RerankerSpec(provider="sentence-transformers", model="local-rerank")
            ),
        ),
        generation=GenerationConfig(generator=SPEC),
    )
    client = QdrantClient(":memory:")
    index = Index(client, "fixture", Embedder())
    cache = ResponseCache(tmp_path / "responses.sqlite3") if cached else None
    try:
        variant = build_variant(config, index=index, models=Models(), cache=cache)
        result = evaluate(variant, QUERIES, "dev", RunStore(tmp_path / "runs"), workers=2)
    finally:
        client.close()
    assert embeddings.peak == reranks.peak == 1 and remote.peak == 2
    assert embeddings.calls == reranks.calls == remote.calls == len(QUERIES)
    assert result.metrics["accuracy"] == 1


def test_judge_grades_in_parallel_and_keeps_output_order(tmp_path: Path):
    store = RunStore(tmp_path)
    local = ScriptedVariant(
        "judge-input", {q.question: ("No", []) for q in QUERIES}, retrieval_mode="none"
    )
    run = evaluate(local, QUERIES, "dev", store)
    sample = tuple(q.id for q in QUERIES)
    barrier = Barrier(2)

    class Judge:
        spec = SPEC

        def judge(self, request: JudgeRequest) -> JudgeVerdict:
            barrier.wait(timeout=3)
            return JudgeVerdict(correct=request.question == QUERIES[0].question)

    report = judge_run(store, run.run_id, sample, Judge(), workers=2)
    assert report.requests == len(QUERIES)
    assert report.metrics["judge_accuracy"] == pytest.approx(1 / len(QUERIES))
    rows = (tmp_path / run.run_id / "judge.jsonl").read_text().splitlines()
    assert [json.loads(row)["query_id"] for row in rows] == list(sample)


def test_ladder_uses_workers_only_for_remote_rungs(tmp_path: Path):
    remote = RemoteFixture()
    local = ScriptedVariant(
        "local", {q.question: ("Yes", []) for q in QUERIES}, retrieval_mode="none"
    )
    rungs = tuple(
        LadderRung(variant.config.name, Path("fixture.yaml"), variant.config)
        for variant in (local, remote)
    )

    def build(rung: LadderRung) -> ScriptedVariant:
        return remote if rung.config == remote.config else local

    results = list(run_ladder(rungs, "test-400", RunStore(tmp_path), QUERIES, build, workers=2))
    assert [run.meta["workers"] for run in results] == [1, 2]
