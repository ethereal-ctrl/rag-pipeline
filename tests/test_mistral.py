import json
from pathlib import Path

import httpx
import pytest
from conftest import FTX_QUERY
from fakes import ScriptedVariant
from test_groq import FakeClock

from rag_pipeline.cache import ResponseCache
from rag_pipeline.corpus import BenchmarkQuery
from rag_pipeline.evaluation import RunStore, evaluate
from rag_pipeline.judge_evaluation import judge_run
from rag_pipeline.judge_model import load_judge_spec
from rag_pipeline.ladder import Ladder
from rag_pipeline.models import GeneratorSpec, LocalModels
from rag_pipeline.rate_limit import MONTH, RateLimiter, RateLimits, remote_limits
from rag_pipeline.variants import VariantConfig, build_variant, plan_remote_run

MODEL = "mistral-large-4-0"
MISTRAL = Path("configs/variants/closed-book-mistral.yaml")


def test_mistral_requires_its_key_and_verified_quota(
    monkeypatch: pytest.MonkeyPatch,
):
    config = VariantConfig.from_yaml(MISTRAL)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_REQUESTS_PER_SECOND", raising=False)
    monkeypatch.delenv("MISTRAL_TOKENS_PER_MINUTE", raising=False)
    monkeypatch.delenv("MISTRAL_TOKENS_PER_MONTH", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        build_variant(config, index=None, models=LocalModels())
    with pytest.raises(ValueError, match="verified Mistral"):
        remote_limits("mistral", MODEL)
    with pytest.raises(ValueError, match="reasoning_effort"):
        GeneratorSpec(provider="mistral", model=MODEL, reasoning_effort="low")
    assert GeneratorSpec.model_validate(config.generation.generator.model_dump()) == (
        config.generation.generator
    )


@pytest.mark.parametrize(
    ("limits", "interval"),
    [(RateLimits(requests_per_second=1), 1), (RateLimits(tokens_per_month=20), MONTH)],
)
def test_mistral_answer_caches_and_waits_across_restarts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limits: RateLimits, interval: int
):
    monkeypatch.setenv("MISTRAL_API_KEY", "fixture-mistral")
    clock = FakeClock()
    times: list[float] = []
    cache = ResponseCache(tmp_path / "cache.sqlite3")
    config = VariantConfig.from_yaml(MISTRAL)

    def mock_chat(request: httpx.Request) -> httpx.Response:
        times.append(clock.now)
        assert str(request.url) == "https://api.mistral.ai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer fixture-mistral"
        data = json.loads(request.content)
        assert data["model"] == MODEL
        assert (
            data["random_seed"] == 0
            and data["max_tokens"] == config.generation.generator.max_output_tokens
        )
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
                "usage": {"prompt_tokens": 8, "completion_tokens": 12},
            },
        )

    def answer(question: str):
        models = LocalModels(
            mistral_transport=httpx.MockTransport(mock_chat),
            mistral_limiter=RateLimiter(cache.path, limits, clock=clock),
        )
        return build_variant(config, index=None, models=models, cache=cache).answer(question)

    first = answer("One?")
    cached = answer("One?")
    second = answer("Two?")
    assert times == [0, interval]
    assert first.text == cached.text == second.text == "A fixture answer"
    assert (first.input_tokens, first.output_tokens) == (8, 12)
    assert cached.trace[-1].cache_hit


def test_mistral_dry_run_estimates_limited_month_and_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MISTRAL_REQUESTS_PER_SECOND", "1")
    monkeypatch.setenv("MISTRAL_TOKENS_PER_MINUTE", "1000")
    monkeypatch.setenv("MISTRAL_TOKENS_PER_MONTH", "20")
    ladder = Ladder.load(Path("configs/ladder.yaml"))
    assert [rung.config.generation.generator.provider for rung in ladder.for_split("test-400")] == [
        "mistral"
    ] * 3
    assert [
        rung.config.generation.generator.provider for rung in ladder.for_split("test-400", "groq")
    ] == ["groq"] * 3
    result = plan_remote_run(VariantConfig.from_yaml(MISTRAL), [FTX_QUERY], None)
    assert result.calls == 1 and result.days > MONTH // 86400


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
    monkeypatch.setenv("MISTRAL_API_KEY", "fixture-mistral")
    monkeypatch.setenv("GROQ_API_KEY", "fixture-groq")
    store = RunStore(tmp_path / "runs")
    run_id, query_id = saved_answer(store, "qwen3.5:4b")
    received: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        received.append(request.url.host or "")
        data = json.loads(request.content)
        response_format = data["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["schema"]["properties"]["correct"]
        assert response_format["json_schema"]["strict"] is True
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"correct":true,"supported":[]}'}}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 20},
            },
        )

    cache = ResponseCache(tmp_path / "cache.sqlite3")
    limiter = RateLimiter(cache.path, RateLimits(requests_per_second=1), clock=FakeClock())
    models = LocalModels(
        groq_transport=httpx.MockTransport(respond),
        groq_limiter=limiter,
        mistral_transport=httpx.MockTransport(respond),
        mistral_limiter=limiter,
    )
    for path, expected in (
        (Path("configs/judge.yaml"), MODEL),
        (Path("configs/judge-groq.yaml"), "openai/gpt-oss-120b"),
    ):
        judge = models.judge_for(cache, load_judge_spec(path))
        assert judge_run(store, run_id, (query_id,), judge).model == expected
        assert judge_run(store, run_id, (query_id,), judge).metrics["judge_accuracy"] == 1.0
    assert received == ["api.mistral.ai", "api.groq.com"]


@pytest.mark.parametrize(
    ("generator", "generator_provider", "judge_provider", "judge_model"),
    [
        ("qwen3.5:4b", "ollama", "ollama", "Qwen/Qwen3-32B"),
        (MODEL, "mistral", "mistral", MODEL),
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
    judge = LocalModels(ollama_transport=transport, mistral_transport=transport).judge_for(
        ResponseCache(tmp_path / "fixture.sqlite3"),
        GeneratorSpec.model_validate({"provider": judge_provider, "model": judge_model}),
    )
    with pytest.raises(ValueError, match="same model family"):
        judge_run(store, run_id, (query_id,), judge)
    assert not (store.root / run_id / "judge.jsonl").exists()
