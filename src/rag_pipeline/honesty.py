"""Honest metrics: what Accuracy hides about Abstention, wrong answers and hedging (Ticket 21).

Accuracy rewards Abstention on Null Queries but treats a wrong answer and a False Abstention
alike. These metrics, computed from stored records, separate the two.
"""

from __future__ import annotations

import re
from dataclasses import replace

from rag_pipeline.corpus import NULL_QUERY_TYPE, BenchmarkQuery
from rag_pipeline.evaluation import (
    QueryRecord,
    RunResult,
    aggregate,
    answer_is_correct,
    bootstrap_intervals,
)

_WORDS = re.compile(r"[a-z0-9]+")


def _says_yes_and_no(answer: str) -> bool:
    words = set(_WORDS.findall(answer.lower()))
    return {"yes", "no"} <= words


def honesty_metrics(records: list[QueryRecord]) -> dict[str, float]:
    """Coverage, Selective Accuracy, penalised scores and hedging over judged records."""
    judged = [r for r in records if r.correct is not None]
    if not judged:
        return {}
    answerable = [r for r in judged if r.query_type != NULL_QUERY_TYPE]
    answered = [r for r in answerable if not r.abstained]
    right = sum(bool(r.correct) for r in judged)
    abstained_answerable = len(answerable) - len(answered)
    wrong = len(judged) - right - abstained_answerable
    hedged = [r for r in answered if _says_yes_and_no(r.answer)]
    n = len(judged)
    metrics = {
        "score(wrong=-1)": (right - wrong) / n,
        "score(wrong=-2)": (right - 2 * wrong) / n,
        "strict_accuracy": (right - sum(bool(r.correct) for r in hedged)) / n,
    }
    if answerable:
        metrics["coverage"] = len(answered) / len(answerable)
    if answered:
        metrics["selective_accuracy"] = sum(bool(r.correct) for r in answered) / len(answered)
        metrics["hedged_rate"] = len(hedged) / len(answered)
    return metrics


def _all_metrics(records: list[QueryRecord]) -> dict[str, float]:
    return aggregate(records) | honesty_metrics(records)


def with_honesty_metrics(run: RunResult) -> RunResult:
    """The run with honest metrics and their 95% intervals, recomputed from its records."""
    records = list(run.records)
    return replace(
        run,
        metrics=run.metrics | honesty_metrics(records),
        intervals=run.intervals | bootstrap_intervals(records, metric_fn=honesty_metrics),
    )


def always_yes(run: RunResult) -> RunResult:
    """The trivial baseline on the run's queries: answer "Yes" to everything, never abstain."""
    records = [
        replace(
            record,
            answer="Yes",
            abstained=False,
            correct=answer_is_correct(
                BenchmarkQuery(
                    record.query_id,
                    record.question,
                    record.reference_answer,
                    record.query_type,
                    (),
                ),
                "Yes",
                abstained=False,
            ),
        )
        for record in run.records
    ]
    return RunResult(
        run_id=f"{run.run_id}-always-yes",
        meta=run.meta | {"variant": {"name": "always-yes"}},
        metrics=_all_metrics(records),
        intervals=bootstrap_intervals(records, metric_fn=_all_metrics),
        records=tuple(records),
    )
