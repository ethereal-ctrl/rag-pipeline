"""Honest metrics: what Accuracy hides about Abstention, wrong answers and hedging (Ticket 21)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import ScriptedVariant
from typer.testing import CliRunner

from rag_pipeline import cli
from rag_pipeline.corpus import BenchmarkQuery, Evidence
from rag_pipeline.evaluation import RunResult, RunStore, evaluate
from rag_pipeline.honesty import always_yes, honesty_metrics

ABSTAIN = "Insufficient information."


def _query(qid: str, qtype: str, answer: str) -> BenchmarkQuery:
    evidence = () if qtype == "null_query" else (Evidence("u", "f"),)
    return BenchmarkQuery(qid, f"question {qid}?", answer, qtype, evidence)


# Four answerable queries (gold Yes) and two Null Queries, one behaviour each.
QUERIES = [
    _query("right", "comparison_query", "Yes"),
    _query("wrong", "comparison_query", "Yes"),
    _query("abstained", "comparison_query", "Yes"),
    _query("hedged", "comparison_query", "Yes"),
    _query("null-abstained", "null_query", "Insufficient information."),
    _query("null-answered", "null_query", "Insufficient information."),
]
ANSWERS = {
    "right": "Yes",
    "wrong": "No",
    "abstained": ABSTAIN,
    "hedged": "Yes for the first, no for the second",
    "null-abstained": ABSTAIN,
    "null-answered": "Yes",
}
SCRIPT = {q.question: (ANSWERS[q.id], ["f"]) for q in QUERIES}


@pytest.fixture
def run(tmp_path: Path) -> RunResult:
    return evaluate(ScriptedVariant("v", SCRIPT), QUERIES, "dev", RunStore(tmp_path))


def test_honest_metrics_separate_abstaining_from_answering_wrongly(run: RunResult):
    metrics = honesty_metrics(list(run.records))

    assert run.metrics["accuracy"] == pytest.approx(3 / 6)
    assert metrics["coverage"] == pytest.approx(3 / 4)
    assert metrics["selective_accuracy"] == pytest.approx(2 / 3)
    assert metrics["score(wrong=-1)"] == pytest.approx((3 - 2) / 6)
    assert metrics["score(wrong=-2)"] == pytest.approx((3 - 4) / 6)


def test_an_answer_saying_yes_and_no_counts_as_wrong_only_in_strict_accuracy(run: RunResult):
    metrics = honesty_metrics(list(run.records))

    assert metrics["hedged_rate"] == pytest.approx(1 / 3)
    assert metrics["strict_accuracy"] == pytest.approx(2 / 6)


def test_the_always_yes_baseline_answers_yes_to_every_query_of_the_run(run: RunResult):
    baseline = always_yes(run)

    assert baseline.meta["variant"]["name"] == "always-yes"
    assert baseline.meta["split"] == "dev"
    assert baseline.metrics["accuracy"] == pytest.approx(4 / 6)
    assert baseline.metrics["abstention_rate[null]"] == 0.0
    assert baseline.metrics["score(wrong=-2)"] == pytest.approx((4 - 2 * 2) / 6)
    assert baseline.intervals["accuracy"][0] <= 4 / 6 <= baseline.intervals["accuracy"][1]


def test_rag_table_shows_honest_metrics_and_the_always_yes_row(
    run: RunResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)

    table = CliRunner().invoke(
        cli.app,
        ["table", run.run_id, "--metrics", "accuracy,coverage,score(wrong=-2)", "--always-yes"],
    )

    assert table.exit_code == 0, table.output
    header, _, row, baseline = table.output.splitlines()
    assert header == "| Variant | Split | n | accuracy | coverage | score(wrong=-2) |"
    assert row.startswith("| v | dev | 6 | 0.500 [")
    assert "| 0.750 [" in row and "| -0.167 [" in row
    assert baseline.startswith("| always-yes | dev | 6 | 0.667 [")
