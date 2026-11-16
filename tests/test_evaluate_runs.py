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
