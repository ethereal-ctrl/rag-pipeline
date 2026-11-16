"""Grade saved Evaluate records without rebuilding a Pipeline Variant."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import TypeAdapter

from rag_pipeline.cache import ResponseCache
from rag_pipeline.chunking import ChunkingSettings, chunk_article
from rag_pipeline.corpus import NULL_QUERY_TYPE, Article, article_hash, load_articles
from rag_pipeline.evaluation import QueryRecord, RunResult, RunStore, bootstrap_metric_intervals
from rag_pipeline.index import corpus_hash
from rag_pipeline.judge_model import (
    CitedStatement,
    Judge,
    JudgeRequest,
    JudgeVerdict,
    check_judge_family,
)
from rag_pipeline.models import GeneratorSpec
from rag_pipeline.rate_limit import RateLimits, quota_days, remote_limits
from rag_pipeline.variants import VariantConfig


def load_answer_run(store: RunStore, run_id: str, *, dev_only: bool = False) -> RunResult:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError(f"Invalid run ID {run_id!r}")
    meta = store.load_meta(run_id)
    if dev_only and meta["split"] != "dev":
        raise ValueError(f"Calibration requires a Dev Split run. {run_id} is from {meta['split']}.")
    if meta.get("retrieval_only"):
        raise ValueError(f"Run {run_id} is retrieval-only. The Judge needs stored answers.")
    result = store.load(run_id)
    if len(result.records) != result.metrics.get("queries") or any(
        record.correct is None for record in result.records
    ):
        raise ValueError(f"Run {run_id} needs complete scored answers.")
    return result


def generator_models(run: RunResult) -> tuple[str, ...]:
    variant = VariantConfig.model_validate(run.meta["variant"])
    traced = {
        stage.model
        for record in run.records
        for stage in record.trace
        if stage.name == "generate" and stage.model is not None
    }
    if variant.router is None:
        traced.add(variant.generation.generator.model)
    return tuple(sorted(traced))


class CitedChunkLookup:
    """Recover original Chunk text for legacy runs whose Corpus version can be verified."""

    def __init__(self, raw_dir: Path) -> None:
        self._raw_dir = raw_dir
        self._chunk_texts: dict[ChunkingSettings, dict[str, str]] = {}

    @cached_property
    def _corpus(self) -> tuple[list[Article], str]:
        articles = load_articles(self._raw_dir)
        return articles, corpus_hash({article.id: article_hash(article) for article in articles})

    def for_version(self, index_version: str) -> dict[str, str]:
        match = re.fullmatch(
            r"chunks-sent-w([1-9]\d*)-o(\d*)(?:-[^@]+)?@([0-9a-f]{12})", index_version
        )
        if match is None:
            raise ValueError(f"Cannot recover cited Chunk text for stored index {index_version!r}.")
        articles, digest = self._corpus
        if digest != match[3]:
            raise ValueError(f"Local Corpus does not match stored index {index_version!r}.")
        settings = ChunkingSettings(int(match[1]), int(match[2]))
        if settings not in self._chunk_texts:
            self._chunk_texts[settings] = {
                chunk.id: chunk.text
                for article in articles
                for chunk in chunk_article(article, settings)
            }
        return self._chunk_texts[settings]


@dataclass(frozen=True)
class JudgeCase:
    record: QueryRecord
    request: JudgeRequest | None


def judge_cases(
    run: RunResult, sample: tuple[str, ...] | None, chunks: CitedChunkLookup
) -> list[JudgeCase]:
    config = VariantConfig.model_validate(run.meta["variant"])
    by_id = {record.query_id: record for record in run.records}
    records = run.records if sample is None else tuple(by_id[id_] for id_ in sample if id_ in by_id)
    if not records:
        raise ValueError(f"Run {run.run_id} has no stored answers in the requested Judge sample.")

    cases: list[JudgeCase] = []
    for record in records:
        eligible = not record.abstained and record.query_type != NULL_QUERY_TYPE
        correctness = eligible and record.correct is False
        cited_variant = config.generation.citations.enabled or record.chosen_variant is not None
        citations = record.citations if eligible and cited_variant else ()
        if not correctness and not citations:
            cases.append(JudgeCase(record, None))
            continue

        cited: dict[str, str] = {}
        for id_ in dict.fromkeys(id_ for citation in citations for id_ in citation.chunk_ids):
            text = record.cited_chunk_texts.get(id_)
            if text is None:
                index_version: str = run.meta.get("index_version") or ""
                text = chunks.for_version(index_version).get(id_)
            if text is None:
                raise ValueError(
                    f"Run {run.run_id}, query {record.query_id} is missing Chunk {id_}."
                )
            cited[id_] = text
        request = JudgeRequest(
            question=record.question,
            reference_answer=record.reference_answer,
            answer=record.answer,
            judge_correctness=correctness,
            statements=tuple(
                CitedStatement(statement=citation.statement, chunk_ids=citation.chunk_ids)
                for citation in citations
            ),
            chunks=cited,
        )
        cases.append(JudgeCase(record, request))
    return cases


@dataclass(frozen=True)
class JudgePlan:
    sampled: int
    calls: int
    cached: int
    estimated_tokens: int
    cached_tokens: int
    days: int
    shared_days: int
    tokens_per_day: int | None
    tokens_per_month: int | None


def plan_judging(
    cases: list[JudgeCase],
    cache: ResponseCache | None,
    spec: GeneratorSpec,
    *,
    shared_tokens: int = 0,
    shared_calls: int = 0,
) -> JudgePlan:
    requests = {
        case.request.cache_key(spec): case.request for case in cases if case.request is not None
    }
    estimates = {key: request.estimated_tokens(spec) for key, request in requests.items()}
    hits = [key for key in estimates if cache is not None and cache.contains(key)]
    tokens = sum(estimates.values())
    cached_tokens = sum(estimates[key] for key in hits)
    remote = spec.remote_provider
    limits = remote_limits(remote, spec.model) if remote is not None else RateLimits()
    calls_left, tokens_left = len(requests) - len(hits), tokens - cached_tokens
    return JudgePlan(
        len(cases),
        len(requests),
        len(hits),
        tokens,
        cached_tokens,
        quota_days(limits, calls_left, tokens_left),
        quota_days(limits, calls_left + shared_calls, tokens_left + shared_tokens),
        limits.tokens_per_day,
        limits.tokens_per_month,
    )


@dataclass(frozen=True)
class JudgedRecord:
    query_id: str
    containment_correct: bool
    requested: bool
    correct: bool | None
    supported: tuple[bool | None, ...]
    malformed: bool


@dataclass(frozen=True)
class JudgeReport:
    model: str
    sampled: int
    requests: int
    malformed: int
    statements: int
    metrics: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    provider: str = "groq"


def _metrics(rows: list[JudgedRecord]) -> dict[str, float]:
    if not rows:
        return {}
    accepted = sum(row.correct is True for row in rows)
    incorrect = sum(not row.containment_correct for row in rows)
    metrics = {"judge_accuracy": (len(rows) - incorrect + accepted) / len(rows)}
    if incorrect:
        metrics["judge_flipped"] = accepted / incorrect
    statements = [supported for row in rows for supported in row.supported]
    if statements:
        metrics["citation_support_rate"] = sum(value is True for value in statements) / len(
            statements
        )
    return metrics


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


def judge_run(
    store: RunStore,
    run_id: str,
    sample: tuple[str, ...],
    judge: Judge,
    *,
    chunks: CitedChunkLookup | None = None,
) -> JudgeReport:
    run = load_answer_run(store, run_id)
    check_judge_family(judge.spec, generator_models(run), run_id)
    rows: list[JudgedRecord] = []
    lookup = chunks if chunks is not None else CitedChunkLookup(Path("data/raw"))
    for case in judge_cases(run, sample, lookup):
        verdict = JudgeVerdict(None) if case.request is None else judge.judge(case.request)
        rows.append(
            JudgedRecord(
                case.record.query_id,
                case.record.correct is True,
                case.request is not None,
                verdict.correct,
                verdict.supported,
                verdict.malformed,
            )
        )

    report = JudgeReport(
        model=judge.spec.model,
        sampled=len(rows),
        requests=sum(row.requested for row in rows),
        malformed=sum(row.malformed for row in rows),
        statements=sum(len(row.supported) for row in rows),
        metrics=_metrics(rows),
        intervals=bootstrap_metric_intervals(rows, _metrics),
        provider=judge.spec.provider,
    )
    run_dir = store.root / run_id
    write_text(run_dir / "judge.jsonl", "".join(json.dumps(asdict(row)) + "\n" for row in rows))
    write_text(run_dir / "judge-metrics.json", json.dumps(asdict(report), indent=2) + "\n")
    return report


_JUDGE_REPORT = TypeAdapter(JudgeReport)


def with_judge_metrics(run: RunResult, store: RunStore) -> RunResult:
    path = store.root / run.run_id / "judge-metrics.json"
    if not path.is_file():
        return run
    report = _JUDGE_REPORT.validate_json(path.read_bytes())
    return replace(
        run, metrics=run.metrics | report.metrics, intervals=run.intervals | report.intervals
    )
