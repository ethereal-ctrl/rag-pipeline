import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from conftest import FTX_QUERY
from fakes import ScriptedVariant
from test_groq import FakeClock
from typer.testing import CliRunner

from rag_pipeline import cli
from rag_pipeline.cache import ResponseCache
from rag_pipeline.corpus import BenchmarkQuery
from rag_pipeline.evaluation import RunStore, evaluate
from rag_pipeline.judge_evaluation import judge_run
from rag_pipeline.judge_model import load_judge_spec
from rag_pipeline.ladder import Ladder
from rag_pipeline.models import LocalModels
from rag_pipeline.rate_limit import RateLimiter, RateLimits, remote_limits
from rag_pipeline.variants import VariantConfig, build_variant

GEMINI = Path("configs/variants/closed-book-gemma.yaml")
NVIDIA = Path("configs/variants/closed-book-gemma-nvidia.yaml")
BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "google/gemma-4-31b-it"


def reply(content: str, *, reasoning: bool = False) -> httpx.Response:
    message = {"content": content}
    usage: dict[str, object] = {"prompt_tokens": 10, "completion_tokens": 4}
    if reasoning:
        message["reasoning_content"] = "Reasoning text stays outside the answer."
        usage.update(completion_tokens=598, completion_tokens_details={"reasoning_tokens": 590})
    return httpx.Response(200, json={"choices": [{"message": message}], "usage": usage})


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nim-fixture")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-fixture")


def test_nvidia_evaluation_scores_clean_answer_and_keeps_run_provenance(
    keys: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host or "")
        if request.url.host == "integrate.api.nvidia.com":
            assert str(request.url) == BASE + "/chat/completions"
            assert request.headers["Authorization"] == "Bearer nim-fixture"
            payload = json.loads(request.content)
            assert payload["stream"] is False
            assert payload["chat_template_kwargs"] == {"thinking": False, "enable_thinking": False}
            if payload["model"] == "moonshotai/kimi-k3.1-pro":
                return reply('{"correct": false, "supported": []}')
            return reply("<thought>Ignore Yes</thought>Insufficient information.")
        return reply("Yes")

    cache = ResponseCache(tmp_path / "models.sqlite3")
    store = RunStore(tmp_path / "runs")
    limiter = RateLimiter(cache.path, RateLimits(), clock=FakeClock())
    models = LocalModels(
        gemini_transport=httpx.MockTransport(respond),
        gemini_limiter=limiter,
        nvidia_transport=httpx.MockTransport(respond),
        nvidia_limiter=limiter,
    )
    query = BenchmarkQuery("yes", "Did it happen?", "Yes", "comparison_query", ())
    before, after = [
        evaluate(
            build_variant(VariantConfig.from_yaml(path), index=None, models=models, cache=cache),
            [query],
            "smoke",
            store,
        )
        for path in (GEMINI, NVIDIA)
    ]
    assert calls == ["generativelanguage.googleapis.com", "integrate.api.nvidia.com"]
    assert before.records[0].correct
    (record,) = after.records
    assert record.abstained and record.correct is False and record.trace[-1].reasoning_tokens > 0
    assert record.notional_cost_usd is not None
    assert (after.meta["provider"], after.meta["base_url"]) == ("nvidia", BASE)
    monkeypatch.setattr(cli, "RUNS_DIR", store.root)
    result = CliRunner().invoke(cli.app, ["compare", before.run_id, after.run_id])
    assert result.exit_code == 0 and "gemini" in result.output and "nvidia" in result.output

    for judge_path, run in [
        (Path("configs/judge-nvidia.yaml"), before),
        (Path("configs/judge.yaml"), after),
    ]:
        with pytest.raises(ValueError, match="same model family"):
            judge_run(
                store, run.run_id, (query.id,), models.judge_for(cache, load_judge_spec(judge_path))
            )
    assert len(calls) == 2
    kimi = models.judge_for(cache, load_judge_spec(Path("configs/judge-kimi.yaml")))
    ambiguous = ScriptedVariant("ambiguous", {query.question: ("Probably", [])})
    ambiguous.config = VariantConfig.from_yaml(NVIDIA)
    judge_input = evaluate(ambiguous, [query], "smoke", store)
    graded = judge_run(store, judge_input.run_id, (query.id,), kimi)
    assert graded.requests == 1 and graded.malformed == 0
    assert (graded.provider, graded.base_url) == ("nvidia", BASE)
    assert len(calls) == 3


@pytest.mark.parametrize("thinking", [None, "low", "off"])
def test_kimi_thinking_payload_and_inclusive_reasoning(
    keys: None, tmp_path: Path, thinking: str | None
):
    def respond(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        assert data["model"] == "moonshotai/kimi-k3.1-pro"
        assert data.get("reasoning_effort") == ("low" if thinking == "low" else None)
        assert data.get("chat_template_kwargs") == (
            {"thinking": False, "enable_thinking": False} if thinking == "off" else None
        )
        return reply("Answer.", reasoning=thinking != "off")

    limiter = RateLimiter(tmp_path / "quota.sqlite3", RateLimits(), clock=FakeClock())
    models = LocalModels(nvidia_transport=httpx.MockTransport(respond), nvidia_limiter=limiter)
    spec = load_judge_spec(Path("configs/judge-kimi.yaml")).model_copy(
        update={"thinking": thinking}
    )
    result = models.generator_for(spec).complete("", "Question?")
    assert result.text == "Answer."
    assert (result.output_tokens, result.reasoning_tokens) == (
        (4, 0) if thinking == "off" else (598, 590)
    )


def test_nvidia_retries_reserve_each_request(keys: None, tmp_path: Path):
    clock = FakeClock()
    calls: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        calls.append(clock.now)
        return httpx.Response(503) if len(calls) < 3 else reply("Yes")

    limiter = RateLimiter(
        tmp_path / "quota.sqlite3", RateLimits(requests_per_minute=1), clock=clock
    )
    models = LocalModels(nvidia_transport=httpx.MockTransport(respond), nvidia_limiter=limiter)
    result = build_variant(VariantConfig.from_yaml(NVIDIA), index=None, models=models).answer(
        "Question?"
    )
    assert result.text == "Yes" and calls == [0, 60, 120]


def test_nvidia_rpm_only_under_concurrent_reservations(tmp_path: Path):
    limits = remote_limits("nvidia", MODEL)
    assert limits.requests_per_minute == 40
    assert limits.tokens_per_minute is None and limits.requests_per_day is None
    path = tmp_path / "quota.sqlite3"
    limiter = RateLimiter(path, limits, clock=FakeClock())
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE attempts (at REAL, tokens INTEGER);
            CREATE TRIGGER log_attempt AFTER INSERT ON remote_requests BEGIN
                INSERT INTO attempts VALUES (new.at, new.tokens);
            END;
        """)

    def reserve(_: int) -> int:
        return limiter.reserve(MODEL, 500_000)

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert len(list(pool.map(reserve, range(81)))) == 81
    with closing(sqlite3.connect(path)) as db:
        rows: list[tuple[float, int]] = db.execute("SELECT at, tokens FROM attempts").fetchall()
    assert len(rows) == 81
    for at, tokens in rows:
        assert tokens == 500_000
        assert sum(at - 60 < seen <= at for seen, _ in rows) <= 40


def test_nvidia_dry_run_and_ladder_selection(monkeypatch: pytest.MonkeyPatch):
    def queries(split: str) -> list[BenchmarkQuery]:
        assert split == "test-400"
        return [FTX_QUERY] * 4

    monkeypatch.setattr(cli, "_split_queries", queries)
    result = CliRunner().invoke(
        cli.app,
        [
            "eval",
            str(NVIDIA),
            "--split",
            "test-400",
            "--dry-run",
            "--no-cache",
            "--workers",
            "2",
            "--call-latency",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Remote calls needed       4" in result.output
    assert "4.0 s (2 workers, 2 s/call)" in result.output
    pairs = Ladder.load(Path("configs/ladder.yaml")).large_pairs("nvidia")
    assert [pair.remote.config.name for pair in pairs] == [
        "closed-book-gemma-nvidia",
        "hybrid-rerank-gemma-nvidia",
        "agentic-gemma-nvidia",
    ]


def test_evaluation_refuses_mixed_remote_providers(tmp_path: Path):
    config = VariantConfig.from_yaml(Path("configs/variants/hybrid-rerank-gemma-nvidia.yaml"))
    serialized = config.model_dump()
    serialized["retrieval"]["hyde"] = {
        "generator": {"provider": "gemini", "model": "gemma-4-31b-it"}
    }
    variant = ScriptedVariant("mixed", {})
    variant.config = VariantConfig.model_validate(serialized)
    store = RunStore(tmp_path)
    with pytest.raises(ValueError, match="one remote provider"):
        evaluate(variant, [FTX_QUERY], "smoke", store)
    assert not store.runs()
