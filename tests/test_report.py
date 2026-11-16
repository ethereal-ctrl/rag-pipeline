"""Results tables for the report: one row per run, every value with its 95% interval."""

from __future__ import annotations

from pathlib import Path

from fakes import ScriptedVariant
from test_evaluate_runs import QUERIES, SCRIPT

from rag_pipeline.evaluation import RunStore, evaluate
from rag_pipeline.report import results_table


def test_a_results_table_shows_each_run_with_intervals_and_marks_missing_metrics(
    tmp_path: Path,
):
    store = RunStore(tmp_path)
    full = evaluate(ScriptedVariant("full", SCRIPT), QUERIES, "dev", store)
    retrieval = evaluate(
        ScriptedVariant("retrieval", SCRIPT), QUERIES, "dev", store, retrieval_only=True
    )

    table = results_table([full, retrieval], ["hits@10", "accuracy"])

    assert table.splitlines() == [
        "| Variant | Split | n | hits@10 | accuracy |",
        "|---|---|---:|---:|---:|",
        "| full | dev | 6 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |",
        "| retrieval | dev | 6 | 1.000 [1.000, 1.000] | n/a |",
    ]
