from dataclasses import replace
from pathlib import Path

import pytest
from fakes import ScriptedVariant
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from rag_pipeline import cli
from rag_pipeline.corpus import NULL_QUERY_TYPE, BenchmarkQuery, Evidence
from rag_pipeline.evaluation import RunStore, evaluate
from rag_pipeline.ladder import (
    Ladder,
    LadderRung,
    LargeModelPair,
    estimate_ladder,
    ladder_status,
    run_ladder,
    validate_rungs,
)
from rag_pipeline.ladder_report import majority_baseline, write_ladder_report
from rag_pipeline.models import GeneratorSpec
from rag_pipeline.variants import ABSTENTION_TEXT

QUERY_TYPE = "comparison_query"
QUERIES = [
    BenchmarkQuery(f"q{i}", f"Question {i}?", "Yes", QUERY_TYPE, (Evidence("u", "fact"),))
    for i in range(2)
]


def rung(name: str, *, indexed: bool = True) -> LadderRung:
    variant = ScriptedVariant(name, {}, retrieval_mode="bm25" if indexed else "none")
    return LadderRung(name, Path(f"{name}.yaml"), variant.config)


def scripted(
    item: LadderRung,
    queries: list[BenchmarkQuery],
    answer: str,
    *,
    seconds: float = 1.0,
    crash_after: int | None = None,
) -> ScriptedVariant:
    variant = ScriptedVariant(
        item.config.name,
        {query.question: (answer, ["fact"]) for query in queries},
        retrieval_mode="bm25" if item.config.uses_retrieval else "none",
        latencies={query.question: seconds for query in queries},
        crash_after=crash_after,
    )
    variant.config = item.config
    return variant


def test_committed_ladder_and_bad_two_group_step():
    ladder = Ladder.load(Path("configs/ladder.yaml"))
    assert len(ladder.rungs) == 9 and len(ladder.large_model) == 3
    assert ladder.rungs[-1].exception
    assert [item.config.name for item in ladder.for_split("test-400")] == [
        "closed-book-groq",
        "hybrid-rerank-groq",
        "agentic-groq",
    ]
    previous, candidate = ladder.rungs[2:4]
    changed = candidate.config.model_copy(
        update={
            "generation": candidate.config.generation.model_copy(
                update={"generator": GeneratorSpec(provider="ollama", model="other")}
            ),
        }
    )
    with pytest.raises(ValueError, match=r"generation.*retrieval"):
        validate_rungs((previous, replace(candidate, config=changed)))


def test_status_distinguishes_complete_in_progress_and_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    first, second, third = (rung("first", indexed=False), rung("second"), rung("third"))
    store = RunStore(tmp_path)
    finished = evaluate(scripted(first, QUERIES, "Yes"), QUERIES, "test", store)
    with pytest.raises(RuntimeError, match="simulated crash"):
        evaluate(scripted(second, QUERIES, "Yes", crash_after=1), QUERIES, "test", store)
    status = ladder_status(
        (first, second, third),
        "test",
        store,
        missing=lambda item: ("missing collection fixture",) if item == third else (),
    )
    assert [row.state for row in status] == ["complete", "in progress", "missing"]
    assert status[0].run is not None and status[0].run.run_id == finished.run_id
    assert status[0].run.meta["commit"] and status[0].run.meta["finished_at"]
    assert status[1].run is not None and status[1].run.meta["index_version"] == "s-v1@fixed"
    assert status[2].missing == ("missing collection fixture",)

    def fixture_ladder(_: Path) -> Ladder:
        return Ladder((first, second, third))

    def memory_index(**_: object) -> QdrantClient:
        return QdrantClient(":memory:")

    monkeypatch.setattr(cli.Ladder, "load", fixture_ladder)
    monkeypatch.setattr(cli, "RUNS_DIR", store.root)
    monkeypatch.setattr(cli, "QdrantClient", memory_index)
    result = CliRunner().invoke(cli.app, ["ladder", "status", "--split", "test"])
    assert result.exit_code == 0, result.output
    assert f"| first | complete | {finished.run_id} |" in result.output
    assert "| second | in progress |" in result.output
    assert "| third | missing |" in result.output and "missing collection s-v1" in result.output


def test_runner_skips_complete_resumes_and_stops_then_continues(tmp_path: Path):
    rungs = tuple(rung(name) for name in ("first", "second", "third", "fourth"))
    store = RunStore(tmp_path)
    evaluate(scripted(rungs[0], QUERIES, "Yes"), QUERIES, "test", store)
    with pytest.raises(RuntimeError, match="simulated crash"):
        evaluate(scripted(rungs[1], QUERIES, "Yes", crash_after=1), QUERIES, "test", store)
    partial = next(run.run_id for run in store.runs() if not run.complete)
    asked: dict[str, list[str]] = {}
    built: list[str] = []

    def build(item: LadderRung) -> ScriptedVariant:
        built.append(item.label)
        if item.label == "third":
            raise ValueError("collection unavailable")
        variant = scripted(item, QUERIES, "Yes")
        asked[item.label] = variant.asked
        return variant

    runner = run_ladder(rungs, "test", store, QUERIES, build)
    assert next(runner).run_id == partial
    with pytest.raises(RuntimeError, match=r"third.*collection unavailable"):
        next(runner)
    assert built == ["second", "third"]
    assert asked["second"] == [QUERIES[1].question]
    remaining = list(
        run_ladder(rungs, "test", store, QUERIES, lambda r: scripted(r, QUERIES, "Yes"))
    )
    assert [run.meta["variant"]["name"] for run in remaining] == ["third", "fourth"]
    assert list(run_ladder(rungs, "test", store, QUERIES, build)) == []
    with pytest.raises(RuntimeError, match=r"already scored.*test"):
        list(run_ladder(rungs[:1], "test-400", store, QUERIES, build))


def test_dry_run_uses_latest_dev_p50_and_flags_missing_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    first, second = (rung("first"), rung("second"))
    store = RunStore(tmp_path)
    evaluate(scripted(first, QUERIES, "Yes", seconds=1.5), QUERIES, "dev", store)
    latest = evaluate(scripted(first, QUERIES, "Yes", seconds=2.5), QUERIES, "dev", store)
    plans = estimate_ladder((first, second), "test", store, 1000)
    assert plans[0].seconds == 2500
    assert plans[0].dev_run_id == latest.run_id
    assert plans[1].seconds is None and plans[1].dev_run_id is None

    def fixture_ladder(_: Path) -> Ladder:
        return Ladder((first, second))

    def fixture_split(_: str) -> tuple[str, ...]:
        return tuple(f"test-{index}" for index in range(1000))

    monkeypatch.setattr(cli.Ladder, "load", fixture_ladder)
    monkeypatch.setattr(cli, "RUNS_DIR", store.root)
    monkeypatch.setattr(cli, "_split_ids", fixture_split)
    result = CliRunner().invoke(cli.app, ["ladder", "run", "--split", "test", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert f"first  2500.0 s  Dev run {latest.run_id}" in result.output
    assert "second  no completed Dev latency estimate" in result.output
    assert "Estimated wall time  2500.0 s" in result.output


def test_majority_predictions_use_dev_and_abstain_on_null(tmp_path: Path):
    item = rung("first", indexed=False)
    store = RunStore(tmp_path)
    dev_queries = [replace(q, id=f"dev-{q.id}", answer="No") for q in QUERIES]
    dev = evaluate(scripted(item, dev_queries, "Yes"), dev_queries, "dev", store)
    test_queries = [*QUERIES, BenchmarkQuery("null", "Unknown?", "", NULL_QUERY_TYPE, ())]
    test = evaluate(scripted(item, test_queries, "Yes"), test_queries, "test", store)
    base = majority_baseline(dev.records, test.records)
    assert base.metrics["accuracy"] == pytest.approx(1 / 3)
    assert base.metrics["accuracy[comparison_query]"] == 0
    assert base.metrics["abstention_rate[null]"] == 1
    assert base.records[-1].answer == ABSTENTION_TEXT
    changed = tuple(replace(record, reference_answer="No") for record in test.records)
    repeated = majority_baseline(dev.records, changed)
    assert repeated.metrics["accuracy"] == 1
    assert [r.answer for r in base.records] == [r.answer for r in repeated.records]


def test_report_pairs_steps_marks_pending_and_writes_deterministic_svg(tmp_path: Path):
    first, second, pending, remote = (
        rung(name) for name in ("first", "second", "pending", "remote")
    )
    ladder = Ladder((first, second, pending), (LargeModelPair(second, remote),))
    store = RunStore(tmp_path / "runs")
    baseline = evaluate(scripted(first, QUERIES, "No", seconds=1), QUERIES, "test", store)
    candidate = evaluate(scripted(second, QUERIES, "Yes", seconds=2), QUERIES, "test", store)
    large = evaluate(scripted(remote, QUERIES[:1], "No", seconds=3), QUERIES[:1], "test-400", store)
    dev_queries = [replace(q, id=f"dev-{q.id}") for q in QUERIES]
    dev = evaluate(scripted(first, dev_queries, "No"), dev_queries, "dev", store)
    args = dict(
        dev_ids=tuple(q.id for q in dev_queries),
        test_ids=tuple(q.id for q in QUERIES),
        subset_ids=(QUERIES[0].id,),
    )
    left = tmp_path / "report-a"
    right = tmp_path / "report-b"
    report = write_ladder_report(ladder, store, "test", left, **args)
    text = report.read_text()
    assert "| pending | pending |" in text
    assert "Accuracy +1.000 [+1.000, +1.000]*" in text
    assert all(value.run_id in text for value in (baseline, candidate, large, dev))
    assert "Shared n" in text and "Dev majority" in text
    assert "Hits@4" in text and "Evidence Recall@10" in text
    write_ladder_report(ladder, store, "test", right, **args)
    charts = sorted(left.glob("*.svg"))
    assert len(charts) == 3
    for chart in charts:
        assert "<svg" in chart.read_text()
        assert chart.read_bytes() == (right / chart.name).read_bytes()
