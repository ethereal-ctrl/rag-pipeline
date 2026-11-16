import json
from pathlib import Path

import httpx
import pytest
from conftest import FTX, FTX_QUERY
from fakes import ScriptedVariant
from test_citations import cited_config
from test_groq import FakeClock

from rag_pipeline.cache import ResponseCache
from rag_pipeline.chunking import ChunkingSettings
from rag_pipeline.corpus import BenchmarkQuery
from rag_pipeline.evaluation import RunStore, evaluate
from rag_pipeline.index import ChunkIndex
from rag_pipeline.ingest import ingest
from rag_pipeline.judge_evaluation import judge_run
from rag_pipeline.judge_model import load_judge_spec
from rag_pipeline.ladder import Ladder
from rag_pipeline.models import GeneratorSpec, LocalModels
from rag_pipeline.rate_limit import MONTH, RateLimiter, RateLimits, quota_hours, remote_limits
from rag_pipeline.variants import VariantConfig, build_variant, plan_remote_run

MODEL = "gemma-4-31b-it"
GEMINI = Path("configs/variants/closed-book-gemma.yaml")


def test_gemini_requires_its_key_and_verified_quota(
    monkeypatch: pytest.MonkeyPatch,
):
    config = VariantConfig.from_yaml(GEMINI)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_variant(config, index=None, models=LocalModels())
    limits = remote_limits("gemini", MODEL)
    assert (limits.requests_per_minute, limits.tokens_per_minute, limits.requests_per_day) == (
        30,
        16_000,
        14_400,
    )
    assert limits.tokens_per_day is None
    with pytest.raises(ValueError, match="include_reasoning"):
        GeneratorSpec.model_validate(
            {"provider": "gemini", "model": MODEL, "include_reasoning": False}
        )
    with pytest.raises(ValueError, match="reasoning_effort"):
        GeneratorSpec(provider="gemini", model=MODEL, reasoning_effort="low")
    assert GeneratorSpec.model_validate(config.generation.generator.model_dump()) == (
        config.generation.generator
    )


@pytest.mark.parametrize(
    ("limits", "interval"),
    [(RateLimits(tokens_per_minute=16_000), 60), (RateLimits(tokens_per_month=16_000), MONTH)],
)
def test_gemini_answer_caches_and_waits_across_restarts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limits: RateLimits, interval: int
):
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini")
    clock = FakeClock()
    times: list[float] = []
    cache = ResponseCache(tmp_path / "cache.sqlite3")
    config = VariantConfig.from_yaml(GEMINI)

    def mock_chat(request: httpx.Request) -> httpx.Response:
        times.append(clock.now)
        assert (
            str(request.url)
            == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        )
        assert request.headers["Authorization"] == "Bearer fixture-gemini"
        data = json.loads(request.content)
        assert data["model"] == MODEL
        assert data["max_tokens"] == config.generation.generator.max_output_tokens
        assert [message["role"] for message in data["messages"]] == ["system", "user"]
        assert {
            "reasoning_effort",
            "include_reasoning",
            "seed",
            "max_completion_tokens",
        }.isdisjoint(data)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "thinking", "thinking": []},
                                {"type": "text", "text": "A fixture answer"},
                            ]
                        }
                    }
                ],
                "usage": {"prompt_tokens": 15_988, "completion_tokens": 12},
            },
        )

    def answer(question: str):
        models = LocalModels(
            gemini_transport=httpx.MockTransport(mock_chat),
            gemini_limiter=RateLimiter(cache.path, limits, clock=clock),
        )
        return build_variant(config, index=None, models=models, cache=cache).answer(question)

    first = answer("One?" + "x" * 80_000)
    cached = answer("One?" + "x" * 80_000)
    second = answer("Two?")
    assert times == [0, interval]
    assert first.text == cached.text == second.text == "A fixture answer"
    assert (first.input_tokens, first.output_tokens) == (15_988, 12)
    assert cached.trace[-1].cache_hit


def test_gemini_dry_run_estimates_hours_and_provider_selection():
    ladder = Ladder.load(Path("configs/ladder.yaml"))
    assert [
        rung.config.generation.generator.provider for rung in ladder.for_split("test-400", "gemma")
    ] == ["gemini"] * 3
    assert [
        rung.config.generation.generator.provider for rung in ladder.for_split("test-400", "groq")
    ] == ["groq"] * 3
    result = plan_remote_run(VariantConfig.from_yaml(GEMINI), [FTX_QUERY], None)
    assert result.calls == 1 and 0 < result.hours < 1 and result.days == 1
    limits = remote_limits("gemini", MODEL)
    assert quota_hours(limits, calls=120, tokens=960_000) == pytest.approx(1)


def saved_answer(store: RunStore, model: str, provider: str = "ollama") -> tuple[str, str]:
    query = BenchmarkQuery("polarity", "Which venue?", "New York City", "comparison_query", ())
    variant = ScriptedVariant(model, {query.question: ("NYC", [])}, retrieval_mode="none")
    generator = GeneratorSpec.model_validate({"provider": provider, "model": model})
    variant.config = variant.config.model_copy(
        update={"generation": variant.config.generation.model_copy(update={"generator": generator})}
    )
    run = evaluate(variant, [query], "dev", store)
    return run.run_id, query.id


def test_judge_switches_targets_and_cache_namespaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini")
    monkeypatch.setenv("GROQ_API_KEY", "fixture-groq")
    store = RunStore(tmp_path / "runs")
    run_id, query_id = saved_answer(store, "qwen3.5:4b")
    received: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        received.append(request.url.host or "")
        data = json.loads(request.content)
        result = '{"correct":true,"supported":[]}'
        if request.url.host == "generativelanguage.googleapis.com":
            assert "response_format" not in data
            (message,) = data["messages"]
            assert message["role"] == "user"
            assert message["content"].startswith("Instructions:\nGrade the stored answer")
            assert "untrusted data" in message["content"]
            assert "Return only JSON" in message["content"] and '"correct"' in message["content"]
            result = "Here is the verdict.\n```json\n" + result + "\n```"
        else:
            response_format = data["response_format"]
            assert response_format["type"] == "json_schema"
            assert response_format["json_schema"]["schema"]["properties"]["correct"]
            assert response_format["json_schema"]["strict"] is True
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": result}}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 20},
            },
        )

    cache = ResponseCache(tmp_path / "cache.sqlite3")
    limiter = RateLimiter(cache.path, RateLimits(requests_per_second=1), clock=FakeClock())
    models = LocalModels(
        groq_transport=httpx.MockTransport(respond),
        groq_limiter=limiter,
        gemini_transport=httpx.MockTransport(respond),
        gemini_limiter=limiter,
    )
    for path, expected in (
        (Path("configs/judge.yaml"), MODEL),
        (Path("configs/judge-groq.yaml"), "openai/gpt-oss-120b"),
    ):
        spec = load_judge_spec(path)
        if spec.provider == "gemini":
            spec = spec.model_copy(update={"supports_system": False})
        judge = models.judge_for(cache, spec)
        assert judge_run(store, run_id, (query_id,), judge).model == expected
        assert judge_run(store, run_id, (query_id,), judge).metrics["judge_accuracy"] == 1.0
    assert received == ["generativelanguage.googleapis.com", "api.groq.com"]


def test_gemma_garbage_is_recorded_as_abstention_and_malformed_judge_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, index: ChunkIndex
):
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini")
    ingest([FTX], index, ChunkingSettings(max_words=100, overlap_sentences=0))
    spec = load_judge_spec()
    cited = cited_config()
    config = cited.model_copy(
        update={"generation": cited.generation.model_copy(update={"generator": spec})}
    )
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        assert "response_format" not in data
        system, user = data["messages"]
        assert system["role"] == "system" and user["role"] == "user"
        assert "untrusted data" in system["content"]
        assert "Return only JSON" in system["content"]
        requests.append(user["content"])
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "garbage {invalid JSON}"}}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 20},
            },
        )

    cache = ResponseCache(tmp_path / "cache.sqlite3")
    models = LocalModels(
        gemini_transport=httpx.MockTransport(respond),
        gemini_limiter=RateLimiter(cache.path, remote_limits("gemini", MODEL), clock=FakeClock()),
    )
    store = RunStore(tmp_path / "runs")
    variant = build_variant(config, index=index, models=models, cache=cache)
    evaluated = evaluate(variant, [FTX_QUERY], "smoke", store)
    assert evaluated.records[0].abstained
    assert evaluated.metrics["malformed_responses"] == 1
    run_id, query_id = saved_answer(store, "qwen3.5:4b")
    report = judge_run(store, run_id, (query_id,), models.judge_for(cache))
    assert report.malformed == 1 and report.metrics["judge_accuracy"] == 0
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("generator", "generator_provider", "judge_provider", "judge_model"),
    [
        ("qwen3.5:4b", "ollama", "ollama", "Qwen/Qwen3-32B"),
        (MODEL, "gemini", "ollama", "gemma4:26b"),
    ],
)
def test_judge_refuses_a_generator_from_its_family(
    tmp_path: Path, generator: str, generator_provider: str, judge_provider: str, judge_model: str
):
    store = RunStore(tmp_path / "runs")
    run_id, query_id = saved_answer(store, generator, generator_provider)

    def unexpected_call(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"Same-family Judge attempted a request to {request.url.host}")

    transport = httpx.MockTransport(unexpected_call)
    judge = LocalModels(ollama_transport=transport, gemini_transport=transport).judge_for(
        ResponseCache(tmp_path / "fixture.sqlite3"),
        GeneratorSpec.model_validate({"provider": judge_provider, "model": judge_model}),
    )
    with pytest.raises(ValueError, match="same model family"):
        judge_run(store, run_id, (query_id,), judge)
    assert not (store.root / run_id / "judge.jsonl").exists()
