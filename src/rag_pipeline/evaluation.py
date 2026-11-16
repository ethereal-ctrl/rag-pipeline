"""The Evaluate seam: run a Pipeline Variant on a split, score it, and store the run."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_pipeline.corpus import BenchmarkQuery
from rag_pipeline.index import RetrievedChunk
from rag_pipeline.variants import PipelineVariant

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    query_type: str
    question: str
    reference_answer: str
    answer: str
    abstained: bool
    correct: bool
    hit_at_4: bool | None
    hit_at_10: bool | None
    retrieved_chunk_ids: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    seconds: float


@dataclass(frozen=True)
class RunResult:
    run_id: str
    meta: dict[str, Any]
    metrics: dict[str, float]
    records: tuple[QueryRecord, ...]


def _squash(text: str) -> str:
    """Whitespace-insensitive form used by the upstream MultiHop-RAG retrieval scoring."""
    return text.replace(" ", "").replace("\n", "")


def _evidence_hit(query: BenchmarkQuery, retrieved: tuple[RetrievedChunk, ...], k: int) -> bool:
    facts = [_squash(e.fact) for e in query.evidence]
    return any(fact in _squash(chunk.text) for chunk in retrieved[:k] for fact in facts)


def _normalise(text: str) -> str:
    return " ".join(_NON_ALNUM.sub(" ", text.lower()).split())


def _is_correct(query: BenchmarkQuery, answer: str, abstained: bool) -> bool:
    if query.is_null:
        return abstained
    if abstained:
        return False
    return f" {_normalise(query.answer)} " in f" {_normalise(answer)} "


def _score(query: BenchmarkQuery, variant: PipelineVariant) -> QueryRecord:
    answer = variant.answer(query.question)
    answerable = not query.is_null
    return QueryRecord(
        query_id=query.id,
        query_type=query.query_type,
        question=query.question,
        reference_answer=query.answer,
        answer=answer.text,
        abstained=answer.abstained,
        correct=_is_correct(query, answer.text, answer.abstained),
        hit_at_4=_evidence_hit(query, answer.retrieved, 4) if answerable else None,
        hit_at_10=_evidence_hit(query, answer.retrieved, 10) if answerable else None,
        retrieved_chunk_ids=tuple(c.chunk_id for c in answer.retrieved),
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        seconds=answer.seconds,
    )


def _mean(values: list[bool]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _metrics(records: list[QueryRecord]) -> dict[str, float]:
    answerable = [r for r in records if r.hit_at_4 is not None]
    return {
        "queries": len(records),
        "hits@4": _mean([bool(r.hit_at_4) for r in answerable]),
        "hits@10": _mean([bool(r.hit_at_10) for r in answerable]),
        "accuracy": _mean([r.correct for r in records]),
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class RunStore:
    """Append-only store of evaluation runs, one directory per run."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, result: RunResult) -> None:
        run_dir = self.root / result.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "meta.json").write_text(json.dumps(result.meta, indent=1), "utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=1), "utf-8")
        with (run_dir / "records.jsonl").open("w", encoding="utf-8") as out:
            for record in result.records:
                out.write(json.dumps(asdict(record)) + "\n")

    def load(self, run_id: str) -> RunResult:
        run_dir = self.root / run_id
        lines = (run_dir / "records.jsonl").read_text("utf-8").splitlines()
        records: list[QueryRecord] = []
        for line in lines:
            row = json.loads(line)
            row["retrieved_chunk_ids"] = tuple(row["retrieved_chunk_ids"])
            records.append(QueryRecord(**row))
        return RunResult(
            run_id=run_id,
            meta=json.loads((run_dir / "meta.json").read_text("utf-8")),
            metrics=json.loads((run_dir / "metrics.json").read_text("utf-8")),
            records=tuple(records),
        )


def evaluate(
    variant: PipelineVariant, queries: list[BenchmarkQuery], split: str, store: RunStore
) -> RunResult:
    started = datetime.now(UTC)
    records = [_score(query, variant) for query in queries]
    generator = variant.generator
    version = getattr(generator, "version", None)
    meta: dict[str, Any] = {
        "variant": variant.config.model_dump(),
        "split": split,
        "generator": generator.name,
        "generator_version": version() if callable(version) else generator.name,
        "commit": _git_commit(),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    run_id = f"{started:%Y%m%dT%H%M%S%f}-{variant.config.name}-{split}"
    result = RunResult(run_id=run_id, meta=meta, metrics=_metrics(records), records=tuple(records))
    store.save(result)
    return result
