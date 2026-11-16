"""The Evaluate seam: run a Pipeline Variant on a split, score it, and store the run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any, Protocol, runtime_checkable

from rag_pipeline.agentic import StopReason
from rag_pipeline.citations import Citation, RemovedStatement
from rag_pipeline.corpus import NULL_QUERY_TYPE, BenchmarkQuery
from rag_pipeline.cost import load_prices, notional_cost
from rag_pipeline.index import RetrievedChunk
from rag_pipeline.models import Generator
from rag_pipeline.parallel import completed_work
from rag_pipeline.splits import TEST_SPLITS
from rag_pipeline.tracing import StageTrace
from rag_pipeline.variants import Answer, VariantConfig

RANK_CUTOFF = 10
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 0


class Evaluable(Protocol):
    """Anything that answers like a Pipeline Variant."""

    @property
    def config(self) -> VariantConfig: ...

    @property
    def generator(self) -> Generator | None: ...

    def answer(self, question: str) -> Answer: ...

    def retrieve(
        self, question: str, stages: dict[str, float] | None = None
    ) -> tuple[RetrievedChunk, ...]: ...

    def index_version(self) -> str | None: ...


@dataclass(frozen=True)
class EvaluationScope:
    split: str
    query_types: tuple[str, ...]
    provenance: dict[str, object]


@runtime_checkable
class _ScopedEvaluation(Protocol):
    def evaluation_scope(self) -> EvaluationScope: ...


@runtime_checkable
class _RetrievalWithTrace(Protocol):
    def retrieve_answer(self, question: str) -> Answer: ...


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class CitationCounts:
    attempted: int
    shown: int
    with_evidence: int | None


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    query_type: str
    question: str
    reference_answer: str
    answer: str
    abstained: bool
    correct: bool | None  # None in retrieval-only runs
    hit_at_4: bool | None
    hit_at_10: bool | None
    reciprocal_rank: float | None
    average_precision: float | None
    evidence_recall: float | None
    retrieved_chunk_ids: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    seconds: float
    stage_seconds: dict[str, float] = field(default_factory=dict[str, float])
    trace: tuple[StageTrace, ...] = ()
    notional_cost_usd: float | None = None
    citations: tuple[Citation, ...] = ()
    removed_statements: tuple[RemovedStatement, ...] = ()
    shown_chunk_ids: tuple[str, ...] = ()
    citation_counts: CitationCounts | None = None
    search_steps: int | None = None
    stop_reason: StopReason | None = None
    chosen_variant: str | None = None
    route_candidates: tuple[str, ...] = ()
    cited_chunk_texts: dict[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class RunResult:
    run_id: str
    meta: dict[str, Any]
    metrics: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    records: tuple[QueryRecord, ...]


@dataclass(frozen=True)
class StoredRun:
    run_id: str
    meta: dict[str, Any]
    complete: bool


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    baseline: float
    candidate: float
    delta: float
    low: float
    high: float

    @property
    def significant(self) -> bool:
        """The 95% interval of the paired difference excludes zero."""
        return self.low > 0 or self.high < 0


def _squash(text: str) -> str:
    """Whitespace-insensitive form used by the upstream MultiHop-RAG retrieval scoring."""
    return text.replace(" ", "").replace("\n", "")


def _matching_facts(chunk: RetrievedChunk, facts: list[str]) -> list[str]:
    text = _squash(chunk.text)
    return [fact for fact in facts if fact in text]


@dataclass(frozen=True)
class _RankScores:
    hit_at_4: bool
    hit_at_10: bool
    reciprocal_rank: float
    average_precision: float
    evidence_recall: float


def _rank_scores(query: BenchmarkQuery, retrieved: tuple[RetrievedChunk, ...]) -> _RankScores:
    """Retrieval scores for one query, following the upstream MultiHop-RAG scoring.

    A retrieved Chunk is relevant if it contains any Evidence fact (whitespace-insensitive).
    Hits@k is per query: at least one relevant Chunk in the top k.  MAP@10 adds, at each
    relevant rank, the number of not-yet-found facts in that Chunk divided by the rank, and
    divides by min(#facts, 10).  Evidence Recall is the fraction of facts found in the top 10.
    """
    facts = [_squash(e.fact) for e in query.evidence]
    found: set[str] = set()
    first_rank: int | None = None
    precision_sum = 0.0
    for rank, chunk in enumerate(retrieved[:RANK_CUTOFF], start=1):
        in_chunk = _matching_facts(chunk, facts)
        if not in_chunk:
            continue
        first_rank = first_rank or rank
        new = [f for f in in_chunk if f not in found]
        found.update(new)
        precision_sum += len(new) / rank
    return _RankScores(
        hit_at_4=first_rank is not None and first_rank <= 4,
        hit_at_10=first_rank is not None,
        reciprocal_rank=1 / first_rank if first_rank else 0.0,
        average_precision=precision_sum / min(len(facts), RANK_CUTOFF),
        evidence_recall=len(found) / len(set(facts)),
    )


def _normalise(text: str) -> str:
    return " ".join(_NON_ALNUM.sub(" ", text.lower()).split())


def answer_is_correct(query: BenchmarkQuery, answer: str, abstained: bool) -> bool:
    if query.is_null:
        return abstained
    if abstained:
        return False
    return f" {_normalise(query.answer)} " in f" {_normalise(answer)} "


def _retrieve_only(variant: Evaluable, question: str) -> Answer:
    if isinstance(variant, _RetrievalWithTrace):
        return variant.retrieve_answer(question)
    stages: dict[str, float] = {}
    started = time.perf_counter()
    retrieved = variant.retrieve(question, stages)
    return Answer("", False, retrieved, 0, 0, time.perf_counter() - started, stages)


def _citation_counts(query: BenchmarkQuery, answer: Answer) -> CitationCounts | None:
    ids = [
        id_
        for statement in (*answer.citations, *answer.removed_statements)
        for id_ in statement.chunk_ids
    ]
    if not ids:
        return None
    shown = set(answer.shown_chunk_ids)
    evidence_count = None
    if not query.is_null:
        facts = [_squash(e.fact) for e in query.evidence]
        evidence_ids = {
            chunk.chunk_id
            for chunk in answer.retrieved
            if chunk.chunk_id in shown and _matching_facts(chunk, facts)
        }
        evidence_count = sum(id_ in evidence_ids for id_ in ids)
    return CitationCounts(len(ids), sum(id_ in shown for id_ in ids), evidence_count)


def _score(query: BenchmarkQuery, variant: Evaluable, retrieval_only: bool) -> QueryRecord:
    if retrieval_only:
        answer = _retrieve_only(variant, query.question)
    else:
        answer = variant.answer(query.question)
    uses_retrieval = (
        variant.config.uses_retrieval if answer.used_retrieval is None else answer.used_retrieval
    )
    scored = uses_retrieval and not query.is_null
    ranks = _rank_scores(query, answer.retrieved) if scored else None
    cited_ids = {id_ for citation in answer.citations for id_ in citation.chunk_ids}
    return QueryRecord(
        query_id=query.id,
        query_type=query.query_type,
        question=query.question,
        reference_answer=query.answer,
        answer=answer.text,
        abstained=answer.abstained,
        correct=None if retrieval_only else answer_is_correct(query, answer.text, answer.abstained),
        hit_at_4=ranks.hit_at_4 if ranks else None,
        hit_at_10=ranks.hit_at_10 if ranks else None,
        reciprocal_rank=ranks.reciprocal_rank if ranks else None,
        average_precision=ranks.average_precision if ranks else None,
        evidence_recall=ranks.evidence_recall if ranks else None,
        retrieved_chunk_ids=tuple(c.chunk_id for c in answer.retrieved),
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        seconds=answer.seconds,
        stage_seconds=answer.stage_seconds,
        trace=answer.trace,
        citations=answer.citations,
        removed_statements=answer.removed_statements,
        shown_chunk_ids=answer.shown_chunk_ids,
        search_steps=answer.search_steps,
        stop_reason=answer.stop_reason,
        chosen_variant=answer.chosen_variant,
        route_candidates=answer.route_candidates,
        cited_chunk_texts={
            chunk.chunk_id: chunk.text for chunk in answer.retrieved if chunk.chunk_id in cited_ids
        },
        citation_counts=(
            _citation_counts(query, answer)
            if not retrieval_only
            and (variant.config.generation.citations.enabled or answer.chosen_variant is not None)
            else None
        ),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


_RETRIEVAL_METRICS = {
    "hits@4": "hit_at_4",
    "hits@10": "hit_at_10",
    "mrr@10": "reciprocal_rank",
    "map@10": "average_precision",
    "evidence_recall@10": "evidence_recall",
}


class Retrieving(Protocol):
    """Anything that ranks Chunks for a question like a Pipeline Variant."""

    @property
    def config(self) -> VariantConfig: ...

    def retrieve(self, question: str) -> tuple[RetrievedChunk, ...]: ...


def retrieval_metrics(variant: Retrieving, queries: list[BenchmarkQuery]) -> dict[str, float]:
    """Retrieval metrics alone, with no generation (Null Queries excluded)."""
    scored = [_rank_scores(q, variant.retrieve(q.question)) for q in queries if not q.is_null]
    metrics: dict[str, float] = {"queries": len(scored)}
    for name, attribute in _RETRIEVAL_METRICS.items():
        metrics[name] = _mean([float(getattr(s, attribute)) for s in scored])
    return metrics


def aggregate(records: list[QueryRecord]) -> dict[str, float]:
    """Corpus-level metrics from per-query records (Null Queries excluded from retrieval)."""
    metrics: dict[str, float] = {"queries": len(records)}
    for name, attribute in _RETRIEVAL_METRICS.items():
        values = [float(v) for r in records if (v := getattr(r, attribute)) is not None]
        if values:  # absent for Closed-Book runs and Null-Query-only samples
            metrics[name] = _mean(values)
    judged = [r for r in records if r.correct is not None]
    if judged:  # absent for retrieval-only runs
        metrics["accuracy"] = _mean([float(bool(r.correct)) for r in judged])
        for query_type in sorted({r.query_type for r in judged}):
            typed = [float(bool(r.correct)) for r in judged if r.query_type == query_type]
            metrics[f"accuracy[{query_type}]"] = _mean(typed)
        null = [float(r.abstained) for r in judged if r.query_type == NULL_QUERY_TYPE]
        answerable = [float(r.abstained) for r in judged if r.query_type != NULL_QUERY_TYPE]
        if null:
            metrics["abstention_rate[null]"] = _mean(null)
        if answerable:
            metrics["false_abstention_rate"] = _mean(answerable)
    citations = [r.citation_counts for r in judged if r.citation_counts is not None]
    if citations:
        metrics["citation_validity"] = sum(c.shown for c in citations) / sum(
            c.attempted for c in citations
        )
        evidence = [c for c in citations if c.with_evidence is not None]
        if evidence:
            metrics["citation_evidence_precision"] = sum(
                c.with_evidence or 0 for c in evidence
            ) / sum(c.attempted for c in evidence)
    steps = [float(r.search_steps) for r in judged if r.search_steps is not None]
    if steps:
        metrics["search_steps"] = _mean(steps)
    reasons = [r.stop_reason for r in judged if r.stop_reason is not None]
    if reasons:
        for reason in ("sufficient", "max_steps", "max_tokens"):
            metrics[f"stop_reason[{reason}]"] = reasons.count(reason) / len(reasons)
    routed = [r.chosen_variant for r in records if r.chosen_variant is not None]
    if routed:
        for name in dict.fromkeys(name for r in records for name in r.route_candidates):
            metrics[f"routed_share[{name}]"] = routed.count(name) / len(routed)
    seconds = sorted(r.seconds for r in records)
    if seconds:
        metrics["latency_p50"] = _percentile(seconds, 0.50)
        metrics["latency_p95"] = _percentile(seconds, 0.95)
    costs = [r.notional_cost_usd for r in records if r.notional_cost_usd is not None]
    if costs and len(costs) == len(records):
        metrics["cost_per_question"] = _mean(costs)
    if records and all(r.trace for r in records):
        metrics["malformed_responses"] = sum(stage.malformed for r in records for stage in r.trace)
        metrics["output_tokens_per_question"] = sum(
            stage.output_tokens for r in records for stage in r.trace
        ) / len(records)
        if all(stage.input_tokens is not None for r in records for stage in r.trace):
            metrics["input_tokens_per_question"] = sum(
                stage.input_tokens or 0 for r in records for stage in r.trace
            ) / len(records)
        generator_tokens = [
            stage.input_tokens
            for record in records
            for stage in record.trace
            if stage.name == "generate" and stage.input_tokens is not None
        ]
        if len(generator_tokens) == len(records):
            metrics["generator_context_tokens_per_question"] = sum(generator_tokens) / len(records)
    return metrics


def _percentile(sorted_values: list[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _interval(samples: list[float]) -> tuple[float, float]:
    finite = sorted(v for v in samples if not math.isnan(v))
    if not finite:
        return (math.nan, math.nan)
    return (_percentile(finite, 0.025), _percentile(finite, 0.975))


def bootstrap_intervals(
    records: list[QueryRecord],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    *,
    metric_fn: Callable[[list[QueryRecord]], dict[str, float]] = aggregate,
) -> dict[str, tuple[float, float]]:
    return bootstrap_metric_intervals(records, metric_fn, samples, seed)


def bootstrap_metric_intervals[T](
    records: list[T],
    metric_fn: Callable[[list[T]], dict[str, float]],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap 95% intervals, resampling queries with replacement."""
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(samples):
        resample = [records[rng.randrange(len(records))] for _ in records]
        for name, value in metric_fn(resample).items():
            draws.setdefault(name, []).append(value)
    reported = metric_fn(records).keys() - {"queries"}
    return {name: _interval(values) for name, values in draws.items() if name in reported}


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[MetricDelta]:
    """Paired bootstrap over the queries both runs share."""
    base_by_id = {r.query_id: r for r in baseline.records}
    cand_by_id = {r.query_id: r for r in candidate.records}
    shared = sorted(base_by_id.keys() & cand_by_id.keys())
    if not shared:
        raise ValueError("the two runs share no queries")
    base_point = aggregate([base_by_id[i] for i in shared])
    cand_point = aggregate([cand_by_id[i] for i in shared])
    rng = random.Random(seed)
    diffs: dict[str, list[float]] = {}
    for _ in range(samples):
        ids = [shared[rng.randrange(len(shared))] for _ in shared]
        base = aggregate([base_by_id[i] for i in ids])
        cand = aggregate([cand_by_id[i] for i in ids])
        for name in base.keys() & cand.keys():
            diffs.setdefault(name, []).append(cand[name] - base[name])
    deltas: list[MetricDelta] = []
    for name in sorted(base_point.keys() & cand_point.keys() - {"queries"}):
        low, high = _interval(diffs.get(name, []))
        deltas.append(
            MetricDelta(
                metric=name,
                baseline=base_point[name],
                candidate=cand_point[name],
                delta=cand_point[name] - base_point[name],
                low=low,
                high=high,
            )
        )
    return deltas


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class TestSplitAlreadyScored(Exception):
    """Raised when a Pipeline Variant would be scored on the Test Split a second time."""

    __test__ = False  # not a pytest test class


def _record_from_row(row: dict[str, Any]) -> QueryRecord:
    row["retrieved_chunk_ids"] = tuple(row["retrieved_chunk_ids"])
    row["trace"] = tuple(StageTrace(**stage) for stage in row.get("trace", ()))
    row["shown_chunk_ids"] = tuple(row.get("shown_chunk_ids", ()))
    row["route_candidates"] = tuple(row.get("route_candidates", ()))
    row["citations"] = tuple(
        Citation(cite["statement"], tuple(cite["chunk_ids"])) for cite in row.get("citations", ())
    )
    row["removed_statements"] = tuple(
        RemovedStatement(item["statement"], tuple(item["chunk_ids"]), item["reason"])
        for item in row.get("removed_statements", ())
    )
    if row.get("citation_counts") is not None:
        row["citation_counts"] = CitationCounts(**row["citation_counts"])
    return QueryRecord(**row)


def write_text(path: Path, content: str) -> None:
    """Replace generated output atomically, including when a previous run was interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8", delete=False
        ) as temporary:
            destination = Path(temporary.name)
            temporary.write(content)
        destination.replace(path)
    finally:
        if destination is not None:
            destination.unlink(missing_ok=True)


class RunStore:
    """Append-only store of evaluation runs, one directory per run.

    A run directory holds `meta.json` and a `records.jsonl` written one query at a time, so
    an interrupted run can resume.  `metrics.json` and `intervals.json` appear only when the
    run completes; their presence marks a run as finished.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._append_lock = Lock()

    def _run_dirs(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(d for d in self.root.iterdir() if (d / "meta.json").exists())

    def load_meta(self, run_id: str) -> dict[str, Any]:
        return self._meta(self.root / run_id)

    def runs(self) -> list[StoredRun]:
        return [StoredRun(d.name, self._meta(d), self._is_complete(d)) for d in self._run_dirs()]

    @staticmethod
    def _meta(run_dir: Path) -> dict[str, Any]:
        meta: dict[str, Any] = json.loads((run_dir / "meta.json").read_text("utf-8"))
        return meta

    @staticmethod
    def _is_complete(run_dir: Path) -> bool:
        return (run_dir / "metrics.json").exists()

    def completed(self, split: str) -> list[dict[str, Any]]:
        return [meta for _, meta in self._completed_runs(split)]

    def completed_answer_ids(self, split: str) -> list[str]:
        return [
            run_id for run_id, meta in self._completed_runs(split) if not meta.get("retrieval_only")
        ]

    def _completed_runs(self, split: str):
        for run_dir in self._run_dirs():
            if self._is_complete(run_dir):
                meta = self._meta(run_dir)
                if meta["split"] == split:
                    yield run_dir.name, meta

    def find_incomplete(self, fingerprint: str) -> str | None:
        for run_dir in reversed(self._run_dirs()):
            if not self._is_complete(run_dir) and self._meta(run_dir)["fingerprint"] == fingerprint:
                return run_dir.name
        return None

    def begin(self, run_id: str, meta: dict[str, Any]) -> None:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "records.jsonl").touch()
        write_text(run_dir / "meta.json", json.dumps(meta, indent=1))

    def records_so_far(self, run_id: str, *, repair: bool = False) -> list[QueryRecord]:
        records: list[QueryRecord] = []
        path = self.root / run_id / "records.jsonl"
        with self._append_lock, path.open("r+b" if repair else "rb") as data:
            offset = 0
            for line in data:
                try:
                    if not line.endswith(b"\n"):
                        break
                    records.append(_record_from_row(json.loads(line)))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    break  # A crash can interrupt the final line, including a UTF-8 character.
                offset += len(line)
            if repair:
                data.truncate(offset)
        return records

    def append(self, run_id: str, record: QueryRecord) -> None:
        line = json.dumps(asdict(record)) + "\n"
        path = self.root / run_id / "records.jsonl"
        with self._append_lock, path.open("a", encoding="utf-8") as out:
            out.write(line)
            out.flush()

    def finish(self, result: RunResult) -> None:
        run_dir = self.root / result.run_id
        write_text(
            run_dir / "records.jsonl", "".join(json.dumps(asdict(r)) + "\n" for r in result.records)
        )
        write_text(run_dir / "meta.json", json.dumps(result.meta, indent=1))
        write_text(run_dir / "intervals.json", json.dumps(result.intervals, indent=1))
        write_text(run_dir / "metrics.json", json.dumps(result.metrics, indent=1))

    def load(self, run_id: str) -> RunResult:
        run_dir = self.root / run_id
        intervals = json.loads((run_dir / "intervals.json").read_text("utf-8"))
        return RunResult(
            run_id=run_id,
            meta=self._meta(run_dir),
            metrics=json.loads((run_dir / "metrics.json").read_text("utf-8")),
            intervals={name: (low, high) for name, (low, high) in intervals.items()},
            records=tuple(self.records_so_far(run_id)),
        )


def _fingerprint(
    config: VariantConfig,
    index_version: str | None,
    split: str,
    queries: list[BenchmarkQuery],
    retrieval_only: bool,
    cache_enabled: bool,
    scope: EvaluationScope | None = None,
) -> str:
    variant = config.run_config()
    generator = variant["generation"]["generator"]
    if generator["provider"] == "ollama":
        # Keep the fingerprints of older, uncached Ollama runs with the same options.
        for name, default in (("seed", 0), ("context_size", 8192), ("reasoning_effort", "low")):
            if generator[name] == default:
                del generator[name]
    identity: dict[str, object] = {
        "variant": variant,
        "index_version": index_version,
        "retrieval_only": retrieval_only,
        "split": split,
        "queries": [q.id for q in queries],
    }
    if cache_enabled:
        identity["cache_enabled"] = True
    if scope is not None:
        identity["scope"] = asdict(scope)
    if config.generation.citations.enabled:
        # Resuming a cited run must not mix statement-based and short-answer scores.
        identity["citation_schema_version"] = 2
    payload = json.dumps(identity, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _guard_test_split(store: RunStore, config: VariantConfig, allow_rescore: bool) -> None:
    if allow_rescore:
        return
    for meta in (meta for split in TEST_SPLITS for meta in store.completed(split)):
        if meta["variant"]["name"] == config.name:
            raise TestSplitAlreadyScored(
                f"Pipeline Variant {config.name!r} was already scored on the Test Split "
                f"(run started {meta['started_at']}). The Test Split is scored once per "
                "variant (ADR 0005); pass allow_rescore=True only for a documented reason."
            )


def evaluate(
    variant: Evaluable,
    queries: list[BenchmarkQuery],
    split: str,
    store: RunStore,
    *,
    allow_rescore: bool = False,
    retrieval_only: bool = False,
    workers: int = 1,
) -> RunResult:
    """Score a Pipeline Variant on a split, resuming an interrupted identical run if present.

    `retrieval_only` skips generation and scores retrieval alone (for bake-offs on the Dev
    Split); it is refused on the Test Split, which is scored once with the whole pipeline.
    """
    config = variant.config
    config.validate_workers(workers, retrieval_only=retrieval_only)
    scope = variant.evaluation_scope() if isinstance(variant, _ScopedEvaluation) else None
    if scope is not None:
        if split != scope.split or retrieval_only:
            raise ValueError(f"{config.name} only supports answer evaluation on the Dev Split.")
        queries = [query for query in queries if query.query_type in scope.query_types]
        if not queries:
            raise ValueError(f"No {scope.split} queries match {', '.join(scope.query_types)}.")
    if retrieval_only and split in TEST_SPLITS:
        raise ValueError("retrieval-only runs are not allowed on the Test Split (ADR 0005)")
    if split in TEST_SPLITS:
        _guard_test_split(store, config, allow_rescore)
    index_version = variant.index_version()
    cache_enabled = getattr(variant, "cache_enabled", False) is True
    fingerprint = _fingerprint(
        config, index_version, split, queries, retrieval_only, cache_enabled, scope
    )
    run_id = store.find_incomplete(fingerprint)
    if run_id is None:
        started = datetime.now(UTC)
        suffix = "-retrieval" if retrieval_only else ""
        run_id = f"{started:%Y%m%dT%H%M%S%f}-{config.name}-{split}{suffix}"
        generator = variant.generator
        version = getattr(generator, "version", None)
        generator_name = generator.name if generator is not None else None
        store.begin(
            run_id,
            {
                "variant": config.run_config(),
                "split": split,
                "index_version": index_version,
                "fingerprint": fingerprint,
                "generator": generator_name,
                "generator_version": version() if callable(version) else generator_name,
                "commit": _git_commit(),
                "started_at": started.isoformat(),
                "rescored": allow_rescore and split in TEST_SPLITS,
                "retrieval_only": retrieval_only,
                "cache_enabled": cache_enabled,
                "workers": workers,
                **(scope.provenance if scope is not None else {}),
                **({"query_types": list(scope.query_types)} if scope is not None else {}),
            },
        )
    meta = store.load_meta(run_id)
    wall_start = time.perf_counter()
    done = {r.query_id: r for r in store.records_so_far(run_id, repair=True)}
    remaining: dict[str, BenchmarkQuery] = {}
    for query in queries:
        if query.id not in done:
            remaining.setdefault(query.id, query)

    def score(query: BenchmarkQuery) -> QueryRecord:
        record = _score(query, variant, retrieval_only)
        store.append(run_id, record)
        return record

    try:
        for record in completed_work(remaining.values(), score, workers):
            done[record.query_id] = record
    finally:
        meta["workers"] = workers
        meta["wall_seconds"] = meta.get("wall_seconds", 0.0) + (time.perf_counter() - wall_start)
        write_text(store.root / run_id / "meta.json", json.dumps(meta, indent=1))
    prices = load_prices()
    records = [done[q.id] for q in queries]
    records = [
        replace(record, notional_cost_usd=notional_cost(record.trace, prices)) for record in records
    ]
    meta["finished_at"] = datetime.now(UTC).isoformat()
    priced_models = {
        stage.model
        for record in records
        for stage in record.trace
        if stage.model is not None and stage.model in prices
    }
    meta["notional_cost_reference_prices"] = {
        model: prices[model].model_dump() for model in sorted(priced_models)
    }
    result = RunResult(
        run_id=run_id,
        meta=meta,
        metrics=aggregate(records),
        intervals=bootstrap_intervals(records),
        records=tuple(records),
    )
    store.finish(result)
    return result


def export_records(result: RunResult, path: Path) -> Path:
    """Write one CSV row per query, for error analysis in a spreadsheet or notebook."""
    rows = [asdict(r) for r in result.records]
    for row in rows:
        row["retrieved_chunk_ids"] = " ".join(row["retrieved_chunk_ids"])
        row["stage_seconds"] = json.dumps(row["stage_seconds"])
        row["trace"] = json.dumps(row["trace"])
        for name in (
            "citations",
            "removed_statements",
            "shown_chunk_ids",
            "citation_counts",
            "route_candidates",
            "cited_chunk_texts",
        ):
            row[name] = json.dumps(row[name])
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]) if rows else ["query_id"])
        writer.writeheader()
        writer.writerows(rows)
    return path
