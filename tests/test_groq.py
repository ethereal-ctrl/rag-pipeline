import json
from pathlib import Path

import httpx
import pytest

from rag_pipeline.cache import ResponseCache
from rag_pipeline.corpus import BenchmarkQuery
from rag_pipeline.evaluation import RunStore, evaluate
from rag_pipeline.models import GeneratorSpec, LocalModels
from rag_pipeline.rate_limit import RateLimiter, RateLimits
from rag_pipeline.variants import (
    GenerationConfig,
    RetrievalConfig,
    VariantConfig,
    build_variant,
    plan_remote_run,
)

CONFIG = VariantConfig(
    name="groq-closed-book",
    retrieval=RetrievalConfig(mode="none"),
    generation=GenerationConfig(
        generator=GeneratorSpec(provider="groq", model="openai/gpt-oss-120b", max_output_tokens=8)
    ),
)


class FakeClock:
    now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds > 0
        self.now += seconds


def test_groq_generator_requires_a_key_at_variant_build(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        build_variant(CONFIG, index=None, models=LocalModels())


def test_groq_retries_and_preserves_usage_across_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    clock = FakeClock()
    calls: list[float] = []

    def chat(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/openai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["max_completion_tokens"] == 8
        calls.append(clock.now)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "5"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "An answer"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    def answer(question: str):
        limiter = RateLimiter(
            tmp_path / "models.sqlite3",
            RateLimits(requests_per_minute=2, tokens_per_minute=1000, requests_per_day=2),
            clock=clock,
        )
        models = LocalModels(groq_transport=httpx.MockTransport(chat), groq_limiter=limiter)
        return build_variant(CONFIG, index=None, models=models).answer(question)

    first = answer("First?")
    second = answer("Second?")

    assert calls == [0, 5, 86400]
    assert first.text == second.text == "An answer"
    assert (first.input_tokens, first.output_tokens) == (7, 3)
    assert (first.trace[-1].input_tokens, first.trace[-1].output_tokens) == (7, 3)


@pytest.mark.parametrize(
    ("rpm", "tpm"), [(1, 1000), (10, 200)], ids=["request-budget", "token-budget"]
)
def test_groq_waits_for_minute_budgets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rpm: int, tpm: int
):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    clock = FakeClock()
    calls: list[float] = []

    def respond(_request: httpx.Request) -> httpx.Response:
        calls.append(clock.now)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "An answer"}}],
                "usage": {"prompt_tokens": 150, "completion_tokens": 1},
            },
        )

    limiter = RateLimiter(
        tmp_path / "models.sqlite3",
        RateLimits(requests_per_minute=rpm, tokens_per_minute=tpm, requests_per_day=10),
        clock=clock,
    )
    models = LocalModels(groq_transport=httpx.MockTransport(respond), groq_limiter=limiter)
    variant = build_variant(CONFIG, index=None, models=models)

    variant.answer("First?")
    variant.answer("Second?")

    assert calls == [0, 60]


def test_dry_run_reports_cached_remote_calls_without_using_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    calls: list[str] = []

    def chat(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Yes"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            },
        )

    cache = ResponseCache(tmp_path / "responses.sqlite3")
    queries = [
        BenchmarkQuery("q1", "First?", "Yes", "comparison_query", ()),
        BenchmarkQuery("q2", "Second?", "Yes", "comparison_query", ()),
    ]
    limiter = RateLimiter(
        cache.path,
        RateLimits(requests_per_minute=2, tokens_per_minute=1000, requests_per_day=2),
        clock=FakeClock(),
    )
    models = LocalModels(groq_transport=httpx.MockTransport(chat), groq_limiter=limiter)
    variant = build_variant(CONFIG, index=None, models=models, cache=cache)
    evaluate(variant, queries[:1], "smoke", RunStore(tmp_path / "runs"))

    monkeypatch.delenv("GROQ_API_KEY")
    plan = plan_remote_run(CONFIG, queries, cache)

    assert (plan.calls, plan.cached, plan.days) == (2, 1, 1)
    assert len(calls) == 1
