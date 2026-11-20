"""Evaluate-seam tests for run bookkeeping: resuming and Test Split discipline."""

import csv
from pathlib import Path

import pytest
from fakes import ScriptedVariant

from rag_pipeline.corpus import BenchmarkQuery, Evidence
from rag_pipeline.evaluation import RunStore, TestSplitAlreadyScored, evaluate, export_records

QUERIES = [
    BenchmarkQuery(f"q{i}", f"question {i}?", "Yes", "comparison_query", (Evidence("u", "f"),))
    for i in range(6)
]
SCRIPT = {q.question: ("Yes", ["f"]) for q in QUERIES}


def test_interrupted_run_resumes_without_losing_or_duplicating_records(tmp_path: Path):
    store = RunStore(tmp_path)
    with pytest.raises(RuntimeError, match="simulated crash"):
        evaluate(ScriptedVariant("v", SCRIPT, crash_after=4), QUERIES, "dev", store)

    resumed_variant = ScriptedVariant("v", SCRIPT)
    result = evaluate(resumed_variant, QUERIES, "dev", store)

    assert [r.query_id for r in result.records] == [q.id for q in QUERIES]
    assert resumed_variant.asked == [q.question for q in QUERIES[4:]]
    assert result.metrics["queries"] == 6
    assert store.load(result.run_id).records == result.records


def test_a_changed_config_starts_a_fresh_run_instead_of_resuming(tmp_path: Path):
    store = RunStore(tmp_path)
    with pytest.raises(RuntimeError):
        evaluate(ScriptedVariant("v", SCRIPT, crash_after=4), QUERIES, "dev", store)

    changed = ScriptedVariant("v", SCRIPT, config_tag="v2")
    evaluate(changed, QUERIES, "dev", store)

    assert len(changed.asked) == 6


def test_test_split_is_scored_only_once_per_variant(tmp_path: Path):
    store = RunStore(tmp_path)
    evaluate(ScriptedVariant("hybrid", SCRIPT), QUERIES, "test", store)

    with pytest.raises(TestSplitAlreadyScored):
        evaluate(ScriptedVariant("hybrid", SCRIPT), QUERIES, "test", store)
    with pytest.raises(TestSplitAlreadyScored):  # same name, tweaked config
        evaluate(ScriptedVariant("hybrid", SCRIPT, config_tag="v2"), QUERIES, "test", store)


def test_test_split_rescoring_needs_an_explicit_override(tmp_path: Path):
    store = RunStore(tmp_path)
    evaluate(ScriptedVariant("hybrid", SCRIPT), QUERIES, "test", store)

    again = evaluate(ScriptedVariant("hybrid", SCRIPT), QUERIES, "test", store, allow_rescore=True)

    assert again.meta["rescored"] is True


@pytest.mark.parametrize("first", ["test", "test-400"])
def test_the_test_subset_shares_the_test_split_score_once_guard(tmp_path: Path, first: str):
    store = RunStore(tmp_path)
    evaluate(ScriptedVariant("hybrid", SCRIPT), QUERIES, first, store)

    for split in ("test", "test-400"):
        with pytest.raises(TestSplitAlreadyScored):
            evaluate(ScriptedVariant("hybrid", SCRIPT), QUERIES, split, store)


def test_the_untouched_600_can_be_scored_once_after_the_400_subset(tmp_path: Path):
    store = RunStore(tmp_path)
    evaluate(ScriptedVariant("hybrid", SCRIPT), QUERIES, "test-400", store)

    final = evaluate(ScriptedVariant("hybrid", SCRIPT), QUERIES, "test-600", store)

    assert final.meta["split"] == "test-600"
    for split in ("test-600", "test"):
        with pytest.raises(TestSplitAlreadyScored):
            evaluate(ScriptedVariant("hybrid", SCRIPT), QUERIES, split, store)


def test_a_full_test_split_score_blocks_the_untouched_600(tmp_path: Path):
    store = RunStore(tmp_path)
    evaluate(ScriptedVariant("hybrid", SCRIPT), QUERIES, "test", store)

    with pytest.raises(TestSplitAlreadyScored):
        evaluate(ScriptedVariant("hybrid", SCRIPT), QUERIES, "test-600", store)


def test_test_subset_rejects_retrieval_only_evaluation(tmp_path: Path):
    with pytest.raises(ValueError, match="Test Split"):
        evaluate(
            ScriptedVariant("bm25", SCRIPT),
            QUERIES,
            "test-400",
            RunStore(tmp_path),
            retrieval_only=True,
        )


def test_dev_split_can_be_scored_repeatedly(tmp_path: Path):
    store = RunStore(tmp_path)
    evaluate(ScriptedVariant("hybrid", SCRIPT), QUERIES, "dev", store)

    second = evaluate(ScriptedVariant("hybrid", SCRIPT), QUERIES, "dev", store)

    assert second.metrics["queries"] == 6


def test_per_query_results_export_as_csv_for_error_analysis(tmp_path: Path):
    store = RunStore(tmp_path)
    result = evaluate(ScriptedVariant("v", SCRIPT), QUERIES, "dev", store)

    path = export_records(store.load(result.run_id), tmp_path / "errors.csv")

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["query_id"] for r in rows] == [q.id for q in QUERIES]
    assert rows[0]["answer"] == "Yes"
    assert rows[0]["correct"] == "True"


def test_closed_book_runs_report_accuracy_but_no_retrieval_metrics(tmp_path: Path):
    variant = ScriptedVariant("closed-book", SCRIPT, retrieval_mode="none")

    result = evaluate(variant, QUERIES, "dev", RunStore(tmp_path))

    assert result.metrics["accuracy"] == 1.0
    assert not any(name.startswith(("hits", "mrr", "map", "evidence")) for name in result.metrics)
    assert all(r.hit_at_10 is None for r in result.records)
    assert "hits@10" not in result.intervals


def test_a_retrieval_only_run_scores_retrieval_without_generating(tmp_path: Path):
    variant = ScriptedVariant("bm25", SCRIPT)
    store = RunStore(tmp_path)

    result = evaluate(variant, QUERIES, "dev", store, retrieval_only=True)

    assert variant.asked == []  # the generator path was never taken
    assert result.metrics["hits@10"] == 1.0
    assert "accuracy" not in result.metrics
    assert all(r.correct is None for r in result.records)
    assert result.meta["retrieval_only"] is True
    # A later full run of the same variant is a different run, not a resumed one.
    assert evaluate(variant, QUERIES, "dev", store).run_id != result.run_id


def test_retrieval_only_runs_are_refused_on_the_test_split(tmp_path: Path):
    with pytest.raises(ValueError, match="Test Split"):
        evaluate(
            ScriptedVariant("bm25", SCRIPT),
            QUERIES,
            "test",
            RunStore(tmp_path),
            retrieval_only=True,
        )


def test_runs_report_median_and_95th_percentile_latency(tmp_path: Path):
    latencies = {q.question: float(n) for n, q in enumerate(QUERIES, start=1)}  # 1 s .. 6 s
    variant = ScriptedVariant("bm25", SCRIPT, latencies=latencies)

    result = evaluate(variant, QUERIES, "dev", RunStore(tmp_path))

    assert result.metrics["latency_p50"] == pytest.approx(3.5)
    assert result.metrics["latency_p95"] == pytest.approx(5.75)  # linear interpolation
    assert "latency_p50" in result.intervals
