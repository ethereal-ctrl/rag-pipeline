import ast
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest
from conftest import FTX, FTX_QUERY
from fakes import ScriptedVariant
from test_groq import FakeClock
from typer.testing import CliRunner

from rag_pipeline import cli
from rag_pipeline.cache import ResponseCache
from rag_pipeline.chunking import ChunkingSettings, chunk_article
from rag_pipeline.citations import Citation
from rag_pipeline.corpus import NULL_QUERY_TYPE, BenchmarkQuery, Evidence, article_hash
from rag_pipeline.evaluation import RunStore, evaluate
from rag_pipeline.index import corpus_hash
from rag_pipeline.judge_calibration import calibration_report, export_calibration, load_labels
from rag_pipeline.judge_evaluation import CitedChunkLookup, judge_run
from rag_pipeline.judge_model import JudgeRequest, JudgeVerdict
from rag_pipeline.models import LocalModels
from rag_pipeline.rate_limit import RateLimiter, groq_limits
from rag_pipeline.splits import Splits, load_judge_sample, load_test_subset, make_judge_sample
from rag_pipeline.variants import Answer, CitationConfig

MODEL = "openai/gpt-oss-120b"


class CitedVariant(ScriptedVariant):
    def answer(self, question: str) -> Answer:
        answer = super().answer(question)
        if question in {"Supported?", "Broken?"}:
            return replace(
                answer,
                citations=(
                    Citation("First statement", ("c0",)),
                    Citation("Second statement", ("c0",)),
                ),
                shown_chunk_ids=("c0", "c1"),
            )
        return answer


def store_run(store: RunStore, split: str, specification: list[tuple[str, str, str, str]]) -> str:
    queries = [
        BenchmarkQuery(f"q{i}", question, reference, kind, (Evidence("u", "Cited text"),))
        for i, (question, reference, _, kind) in enumerate(specification)
    ]
    variant = CitedVariant(
        "cited",
        {
            question: (answer, ["Cited text", "Other text"])
            for question, _, answer, _ in specification
        },
    )
    generation = variant.config.generation.model_copy(
        update={"citations": CitationConfig(enabled=True)}
    )
    variant.config = variant.config.model_copy(update={"generation": generation})
    return evaluate(variant, queries, split, store).run_id


def test_judge_grades_only_eligible_answers_and_saves_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    kind = "comparison_query"
    run_id = store_run(
        RunStore(tmp_path / "runs"),
        "test",
        [
            ("Correct?", "Yes", "Yes", kind),
            ("Abstained?", "Yes", "Insufficient information.", kind),
            ("Null abstained?", "", "Insufficient information.", NULL_QUERY_TYPE),
            ("Null guessed?", "", "Guess", NULL_QUERY_TYPE),
            ("Supported?", "New York City", "NYC", kind),
            ("Broken?", "Yes", "Yes", kind),
            ("Outside the sample?", "New York City", "NYC", kind),
        ],
    )
    store = RunStore(tmp_path / "runs")
    sample = tuple(f"q{i}" for i in range(6))
    calls: list[dict[str, object]] = []
    interrupted = False

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.host == "api.groq.com"
        assert body["model"] == MODEL
        assert body["response_format"]["json_schema"]["strict"] is True
        assert "untrusted" in body["messages"][0]["content"]
        prompt: dict[str, object] = json.loads(body["messages"][1]["content"])
        if prompt["question"] == "Broken?" and not interrupted:
            return httpx.Response(503)
        calls.append(prompt)
        assert prompt["statements"] == [
            {"statement": "First statement", "chunk_ids": ["c0"]},
            {"statement": "Second statement", "chunk_ids": ["c0"]},
        ]
        assert prompt["chunks"] == {"c0": "Cited text"}
        content = (
            '{"correct":true,"supported":[true,false]}'
            if prompt["question"] == "Supported?"
            else "malformed JSON"
        )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 800, "completion_tokens": 20},
            },
        )

    monkeypatch.setenv("GROQ_API_KEY", "fixture")
    cache = ResponseCache(tmp_path / "cache.sqlite3")
    limiter = RateLimiter(cache.path, groq_limits(MODEL), clock=FakeClock())
    models = LocalModels(groq_transport=httpx.MockTransport(respond), groq_limiter=limiter)
    judge = models.judge_for(cache)
    with pytest.raises(httpx.HTTPStatusError):
        judge_run(store, run_id, sample, judge)

    monkeypatch.setattr(cli, "RUNS_DIR", store.root)
    monkeypatch.setattr(cli, "_judge_sample", lambda: sample)
    monkeypatch.setattr(cli, "CACHE_FILE", cache.path)
    runner = CliRunner()
    dry_run = runner.invoke(cli.app, ["judge", run_id, "--dry-run", "--shared-tokens", "200001"])
    assert dry_run.exit_code == 0, dry_run.output
    plan = {
        label.strip(): int(value)
        for label, _, value in (line.rpartition(" ") for line in dry_run.output.splitlines())
        if value.isdecimal()
    }
    assert plan["Calls before cache"] == 2
    assert plan["Already cached"] == plan["Calls needed"] == 1
    assert plan["Estimated tokens needed"] > 512
    assert plan["Estimated tokens before cache"] == (
        plan["Estimated tokens needed"] + plan["Estimated cached tokens"]
    )
    assert plan["Estimated Judge quota days"] == 1
    assert plan["Estimated shared quota days"] == 2

    interrupted = True
    report = judge_run(store, run_id, sample, judge)
    assert [call["question"] for call in calls] == ["Supported?", "Broken?"]
    assert [call["judge_correctness"] for call in calls] == [True, False]
    assert report.metrics == {
        "judge_accuracy": 4 / 6,
        "judge_flipped": 1 / 3,
        "citation_support_rate": 1 / 4,
    }
    assert set(report.intervals) == report.metrics.keys()
    assert report.malformed == 1
    rows = [
        json.loads(line) for line in (store.root / run_id / "judge.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["correct"] is None and rows[-1]["supported"] == [None, None]
    assert rows[-2]["supported"] == [True, False]

    monkeypatch.delenv("GROQ_API_KEY")
    assert judge_run(store, run_id, sample, LocalModels().judge_for(cache)) == report
    table = runner.invoke(
        cli.app, ["table", run_id, "--metrics", "judge_accuracy,citation_support_rate"]
    )
    assert table.exit_code == 0
    assert "0.667 [" in table.output and "0.250 [" in table.output


def test_judge_recovers_cited_text_for_a_legacy_run_with_a_matching_corpus(tmp_path: Path):
    settings = ChunkingSettings()
    chunk = chunk_article(FTX, settings)[0]
    digest = corpus_hash({FTX.id: article_hash(FTX)})
    store = RunStore(tmp_path / "runs")
    raw = tmp_path / "fixture-corpus"
    raw.mkdir()
    corpus_file = raw / "corpus.json"
    corpus_file.write_text(json.dumps([asdict(FTX)]))
    variant = CitedVariant("cited", {FTX_QUERY.question: (FTX_QUERY.answer, [chunk.text])})
    variant.config = variant.config.model_copy(
        update={
            "generation": variant.config.generation.model_copy(
                update={"citations": CitationConfig(enabled=True)}
            )
        }
    )
    evaluated = evaluate(variant, [FTX_QUERY], "test", store)
    record = replace(evaluated.records[0], citations=(Citation("On trial", (chunk.id,)),))
    store.finish(
        replace(
            evaluated,
            meta={**evaluated.meta, "index_version": f"chunks-{settings.label}@{digest}"},
            records=(record,),
        )
    )

    class InspectingJudge:
        def judge(self, request: JudgeRequest) -> JudgeVerdict:
            assert request.chunks == {chunk.id: chunk.text}
            return JudgeVerdict(correct=None, supported=(True,))

    sample = (FTX_QUERY.id,)
    report = judge_run(
        store, evaluated.run_id, sample, InspectingJudge(), chunks=CitedChunkLookup(raw)
    )
    assert report.metrics["citation_support_rate"] == 1

    corpus_file.write_text(json.dumps([asdict(replace(FTX, body="Different text."))]))
    with pytest.raises(ValueError, match="Corpus does not match"):
        judge_run(store, evaluated.run_id, sample, InspectingJudge(), chunks=CitedChunkLookup(raw))


def test_committed_judge_sample_is_inside_test_subset_and_selection_is_stratified():
    splits = Splits.load(Path("data/splits.json"))
    subset = load_test_subset(Path("data/test-400.json"), splits)
    sample = load_judge_sample(Path("data/judge-300.json"), subset)
    assert len(set(sample)) == 300 and set(sample) <= set(subset)
    queries = [
        BenchmarkQuery(id_, "", "", "one" if i < 300 else "two", ()) for i, id_ in enumerate(subset)
    ]
    selected = make_judge_sample(queries, subset)
    by_id = {q.id: q.query_type for q in queries}
    assert Counter(by_id[id_] for id_ in selected) == {"one": 225, "two": 75}
    assert make_judge_sample(queries[::-1], subset[::-1]) == selected


@pytest.mark.parametrize("split", ["test", "test-400"])
def test_calibration_export_refuses_test_runs(tmp_path: Path, split: str):
    store = RunStore(tmp_path / "runs")
    run_id = store_run(store, split, [("Example?", "Yes", "No", "comparison_query")])
    with pytest.raises(ValueError, match="Dev Split"):
        export_calibration(store, [run_id], tmp_path / "calibration")


def test_calibration_validates_dual_labels_and_reports_agreement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    store = RunStore(tmp_path / "runs")
    kind = "comparison_query"
    run_id = store_run(
        store,
        "dev",
        [
            ("Supported?", "New York City", "NYC", kind),
            *[(f"Paraphrase {i}?", "New York City", "NYC", kind) for i in range(49)],
        ],
    )
    folder = tmp_path / "calibration"
    cases = export_calibration(store, [run_id], folder)
    assert len(cases) == 52
    assert Counter(case.kind for case in cases) == {"correctness": 50, "support": 2}
    assert export_calibration(store, [run_id], folder) == cases
    monkeypatch.setattr(cli, "RUNS_DIR", store.root)
    monkeypatch.setattr(cli, "CALIBRATION_DIR", folder)
    export = CliRunner().invoke(cli.app, ["judge", "calibration", "export"])
    assert export.exit_code == 0, export.output
    assert "Exported 52 Dev cases" in export.output
    assert "labelled_by" in (folder / "LABELLING.md").read_text()
    case_file = folder / "labels.jsonl"
    labels: list[dict[str, str | bool]] = []
    for i, case in enumerate(cases):
        label = (
            int(case.request.question.split()[1].rstrip("?")) % 2 == 0
            if case.request.question.startswith("Paraphrase")
            else True
        )
        labels.append({"case_id": case.case_id, "label": label, "labelled_by": "subscription"})
        if i < 48 or case.kind == "support":
            labels.append({"case_id": case.case_id, "label": label, "labelled_by": "author"})

    def write_labels(rows: list[dict[str, str | bool]]) -> None:
        case_file.write_text("".join(json.dumps(row) + "\n" for row in rows))

    write_labels([row for row in labels if row["labelled_by"] == "subscription"])
    with pytest.raises(ValueError, match="50 author"):
        load_labels(case_file, cases)
    write_labels([*labels, {**labels[0], "label": "true"}])
    with pytest.raises(ValueError, match=r"labels\.jsonl"):
        load_labels(case_file, cases)
    write_labels(labels)

    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(json.loads(request.content)["messages"][1]["content"])
        question: str = prompt["question"]
        calls.append(question)
        correct = question == "Supported?" or int(question.split()[1].rstrip("?")) % 4 < 2
        verdict = {"correct": correct, "supported": [True, False] if prompt["statements"] else []}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(verdict)}}],
                "usage": {"prompt_tokens": 800, "completion_tokens": 20},
            },
        )

    cache = ResponseCache(tmp_path / "cache.sqlite3")
    limiter = RateLimiter(cache.path, groq_limits(MODEL), clock=FakeClock())
    monkeypatch.setenv("GROQ_API_KEY", "fixture")
    judge = LocalModels(
        groq_transport=httpx.MockTransport(respond), groq_limiter=limiter
    ).judge_for(cache)
    report = calibration_report(folder, store.root / "judge", judge)
    assert len(calls) == 50 and calls.count("Supported?") == 1
    correctness = report.agreement["correctness"]["judge_vs_subscription"]
    # 26 true and 24 false labels on each side, with 26 matching decisions.
    assert correctness.metrics["accuracy"] == pytest.approx(26 / 50)
    chance = (26 * 26 + 24 * 24) / 50**2
    assert correctness.metrics["kappa"] == pytest.approx((26 / 50 - chance) / (1 - chance))
    assert "kappa" in correctness.intervals
    support = report.agreement["support"]["judge_vs_subscription"]
    assert support.metrics == {"accuracy": 0.5, "kappa": 0.0}
    assert report.agreement["correctness"]["subscription_vs_author"].metrics["kappa"] == 1.0
    assert list((store.root / "judge").glob("*.json"))
    assert list((store.root / "judge").glob("*.md"))


def test_application_has_no_subscription_clients_or_credentials():
    providers = {
        "openai",
        "anthropic",
        "chatgpt",
        "revchatgpt",
        "claude",
        "claude_code_sdk",
        "anthropic_sdk",
    }
    clients = {"openai", "asyncopenai", "anthropic", "asyncanthropic", "chatgpt"}
    credential_markers = ("key", "token", "session", "cookie", "credential", "password")
    for path in Path("src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                assert all(
                    name.name.split(".")[0].lower() not in providers for name in node.names
                ), path
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0].lower() not in providers, path
            elif isinstance(node, ast.Call):
                function = node.func
                name = (
                    function.id
                    if isinstance(function, ast.Name)
                    else (function.attr if isinstance(function, ast.Attribute) else "")
                )
                assert name.lower() not in clients, path
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.lower()
                assert value.split(".")[0] not in providers, path
                assert not any(
                    host in value
                    for host in ("api.openai.com", "api.anthropic.com", "chatgpt.com/backend-api")
                ), path
                assert not any(
                    value.startswith(provider + "_")
                    and any(marker in value for marker in credential_markers)
                    for provider in providers
                ), path
