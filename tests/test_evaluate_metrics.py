"""Evaluate-seam tests with scripted rankings, so every expected value is hand-computed."""

from pathlib import Path

import pytest
from fakes import ScriptedVariant

from rag_pipeline.corpus import BenchmarkQuery, Evidence
from rag_pipeline.evaluation import RunResult, RunStore, compare_runs, evaluate


def _query(qid: str, qtype: str, answer: str, facts: list[str]) -> BenchmarkQuery:
    return BenchmarkQuery(
        id=qid,
        question=f"question {qid}?",
        answer=answer,
        query_type=qtype,
        evidence=tuple(Evidence("u", f) for f in facts),
    )


# A: two facts, found at ranks 2 and 4.  B: one fact at rank 1.  C: nothing found.
A = _query("A", "inference_query", "Alice", ["fact one", "fact two"])
B = _query("B", "comparison_query", "Yes", ["fact three"])
C = _query("C", "comparison_query", "No", ["fact four", "fact five"])
N = _query("N", "null_query", "Insufficient information.", [])

SCRIPT = {
    A.question: ("Alice did it.", ["noise", "has fact two here", "noise", "and fact one"]),
    B.question: ("Yes", ["fact three"]),
    C.question: ("Yes", ["noise"]),
    N.question: ("Insufficient information.", ["noise"]),
}


@pytest.fixture
def result(tmp_path: Path) -> RunResult:
    return evaluate(ScriptedVariant("scripted", SCRIPT), [A, B, C, N], "dev", RunStore(tmp_path))


def test_rank_metrics_match_hand_computed_values(result: RunResult):
    # MRR: A first hit at rank 2, B at rank 1, C none -> (1/2 + 1 + 0) / 3
    assert result.metrics["mrr@10"] == pytest.approx(0.5)
    # MAP as in the upstream scoring: sum of (new facts found / rank) over hit ranks,
    # divided by min(#facts, 10).  A: (1/2 + 1/4) / 2, B: 1, C: 0.
    assert result.metrics["map@10"] == pytest.approx((0.375 + 1 + 0) / 3)
    # Evidence Recall: fraction of each query's facts found in the top 10.
    assert result.metrics["evidence_recall@10"] == pytest.approx((1 + 1 + 0) / 3)
    assert result.metrics["hits@10"] == pytest.approx(2 / 3)


def test_accuracy_is_reported_overall_and_per_query_type(result: RunResult):
    # A correct, B correct, C wrong, N correctly abstained.
    assert result.metrics["accuracy"] == pytest.approx(3 / 4)
    assert result.metrics["accuracy[inference_query]"] == 1.0
    assert result.metrics["accuracy[comparison_query]"] == 0.5
    assert result.metrics["accuracy[null_query]"] == 1.0


def _many(n: int, correct: bool) -> tuple[list[BenchmarkQuery], dict[str, tuple[str, list[str]]]]:
    queries = [_query(f"q{i}", "comparison_query", "Yes", [f"fact {i}"]) for i in range(n)]
    script = {
        q.question: ("Yes" if correct else "No", [f"fact {i}"] if correct else ["noise"])
        for i, q in enumerate(queries)
    }
    return queries, script


def test_every_metric_has_a_reproducible_95_percent_interval(tmp_path: Path):
    queries, script = _many(30, correct=True)
    script[queries[0].question] = ("No", ["noise"])  # one miss, so intervals are not degenerate

    first = evaluate(ScriptedVariant("v", script), queries, "dev", RunStore(tmp_path / "1"))
    second = evaluate(ScriptedVariant("v", script), queries, "dev", RunStore(tmp_path / "2"))

    low, high = first.intervals["accuracy"]
    assert first.metrics["accuracy"] == pytest.approx(29 / 30)
    assert low < first.metrics["accuracy"] <= high
    assert set(first.intervals) >= {"hits@4", "mrr@10", "map@10", "evidence_recall@10"}
    assert first.intervals == second.intervals


def test_comparing_runs_marks_only_real_changes_as_significant(tmp_path: Path):
    queries, bad_script = _many(40, correct=False)
    _, good_script = _many(40, correct=True)
    store = RunStore(tmp_path)
    bad = evaluate(ScriptedVariant("bad", bad_script), queries, "dev", store)
    good = evaluate(ScriptedVariant("good", good_script), queries, "dev", store)
    good_again = evaluate(ScriptedVariant("good2", good_script), queries, "dev", store)

    better = {d.metric: d for d in compare_runs(bad, good)}
    same = {d.metric: d for d in compare_runs(good, good_again)}

    assert better["accuracy"].delta == pytest.approx(1.0)
    assert better["accuracy"].significant
    assert better["hits@4"].significant
    assert not same["accuracy"].significant
    assert same["accuracy"].delta == 0
