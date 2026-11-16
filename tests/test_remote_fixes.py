import json
import re
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from conftest import FTX, FTX_QUERY
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_citations import cited_config
from test_groq import FakeClock

from rag_pipeline.chunking import ChunkingSettings
from rag_pipeline.corpus import BenchmarkQuery
from rag_pipeline.evaluation import RunStore, evaluate
from rag_pipeline.index import ChunkIndex
from rag_pipeline.ingest import ingest
from rag_pipeline.models import LocalModels, estimate_tokens
from rag_pipeline.rate_limit import RateLimiter, RateLimits
from rag_pipeline.variants import ABSTENTION_TEXT, VariantConfig, build_variant

CONFIG = Path("configs/variants/closed-book-gemma.yaml")


def completion(
    text: str, usage: dict[str, object] | None = None, **extra: object
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 2, **(usage or {})},
            **extra,
        },
    )


def gemini(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clock: FakeClock,
    response: httpx.MockTransport,
    limits: RateLimits | None = None,
) -> LocalModels:
    monkeypatch.setenv("GEMINI_API_KEY", "fixture")
    limiter = RateLimiter(tmp_path / "quota.sqlite3", limits or RateLimits(), clock=clock)
    return LocalModels(gemini_transport=response, gemini_limiter=limiter)


def test_live_shape_abstains_and_charges_unreported_reasoning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    thought = "The answer could be yes based on the question alone. " * 27
    raw = f"<thought>{thought}</thought>Insufficient information"
    calls: list[float] = []
    clock = FakeClock()

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(clock.now)
        body = json.loads(request.content)
        assert body["extra_body"]["google"]["thinking_config"]["thinking_level"] == "minimal"
        return completion(raw)

    models = gemini(
        monkeypatch,
        tmp_path,
        clock,
        httpx.MockTransport(respond),
        RateLimits(tokens_per_minute=1000),
    )
    config = VariantConfig.from_yaml(CONFIG)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    variant = build_variant(config, index=None, models=models, tracer=provider.get_tracer("test"))
    query = BenchmarkQuery("q_yes", "Does the article say this?", "Yes", "comparison_query", ())
    run = evaluate(variant, [query, replace(query, id="q_again")], "smoke", RunStore(tmp_path))
    record = run.records[0]
    assert record.abstained and record.correct is False
    assert record.answer.rstrip(".") == ABSTENTION_TEXT.rstrip(".")
    expected = estimate_tokens("", thought, 0) - estimate_tokens("", "", 0)
    stage = next(stage for stage in record.trace if stage.name == "generate")
    assert stage.reasoning_tokens == expected
    assert stage.raw_content_length == len(raw)
    assert record.output_tokens == 2 + expected
    assert record.notional_cost_usd == pytest.approx((40 * 0.09 + (2 + expected) * 0.34) / 1e6)
    assert calls == [0, 60]
    generate = next(span for span in exporter.get_finished_spans() if span.name == "generate")
    assert generate.attributes is not None
    assert generate.attributes["rag.reasoning_tokens"] == expected
    assert thought.strip() not in str(generate.attributes)


@pytest.mark.parametrize(
    ("usage", "metadata", "expected"),
    [
        ({"completion_tokens_details": {"reasoning_tokens": 9}}, {}, 9),
        ({}, {"usageMetadata": {"thoughtsTokenCount": 10}}, 10),
        ({"total_tokens": 53}, {}, 11),
    ],
)
def test_tagged_citations_use_reported_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    index: ChunkIndex,
    usage: dict[str, object],
    metadata: dict[str, object],
    expected: int,
):
    ingest([FTX], index, ChunkingSettings(max_words=100, overlap_sentences=0))

    def respond(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        (shown,) = re.findall(r'<chunk id="([^"]+)"', data["messages"][-1]["content"])
        answer = json.dumps(
            {
                "answer": FTX_QUERY.answer,
                "statements": [{"statement": FTX_QUERY.answer, "chunk_ids": [shown]}],
            }
        )
        return completion(
            "<THOUGHT>Think yes</Thought>\n<ThInK>Think no</tHiNk>\n```json\n" + answer + "\n```",
            usage,
            **metadata,
        )

    models = gemini(monkeypatch, tmp_path, FakeClock(), httpx.MockTransport(respond))
    cited = cited_config()
    spec = VariantConfig.from_yaml(CONFIG).generation.generator
    config = cited.model_copy(
        update={"generation": cited.generation.model_copy(update={"generator": spec})}
    )
    answer = build_variant(config, index=index, models=models).answer(FTX_QUERY.question)
    assert answer.text == FTX_QUERY.answer and len(answer.citations) == 1
    assert answer.trace[-1].reasoning_tokens == expected
    assert answer.output_tokens == expected + 2


def test_unterminated_thought_discards_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, index: ChunkIndex
):
    ingest([FTX], index, ChunkingSettings(max_words=100, overlap_sentences=0))
    raw = '<thought>{"answer":"Yes","statements":[]}'
    models = gemini(
        monkeypatch,
        tmp_path,
        FakeClock(),
        httpx.MockTransport(lambda _: completion(raw)),
    )
    cited = cited_config()
    spec = VariantConfig.from_yaml(CONFIG).generation.generator
    config = cited.model_copy(
        update={"generation": cited.generation.model_copy(update={"generator": spec})}
    )
    answer = build_variant(config, index=index, models=models).answer(FTX_QUERY.question)
    assert answer.abstained and answer.text == ABSTENTION_TEXT
    assert answer.trace[-1].malformed and answer.trace[-1].reasoning_tokens > 0


def test_groq_removes_reasoning_and_preserves_inclusive_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    config = VariantConfig.from_yaml(Path("configs/variants/closed-book-groq.yaml"))
    monkeypatch.setenv("GROQ_API_KEY", "fixture")

    def respond(_: httpx.Request) -> httpx.Response:
        return completion(
            "<think>Yes was considered.</think>No",
            {
                "completion_tokens": 12,
                "total_tokens": 52,
                "completion_tokens_details": {"reasoning_tokens": 9},
            },
        )

    models = LocalModels(
        groq_transport=httpx.MockTransport(respond),
        groq_limiter=RateLimiter(tmp_path / "quota.sqlite3", RateLimits(), clock=FakeClock()),
    )
    answer = build_variant(config, index=None, models=models).answer("Question?")
    assert answer.text == "No"
    assert answer.output_tokens == 12 and answer.trace[-1].reasoning_tokens == 9


@pytest.mark.parametrize(
    "limits",
    [RateLimits(requests_per_minute=1), RateLimits(tokens_per_minute=600)],
    ids=["requests", "tokens"],
)
def test_503_retries_consume_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    limits: RateLimits,
):
    clock = FakeClock()
    calls: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        calls.append(clock.now)
        if len(calls) < 3:
            return httpx.Response(503, headers={"retry-after": "5"})
        return completion("Yes")

    models = gemini(monkeypatch, tmp_path, clock, httpx.MockTransport(respond), limits)
    config = VariantConfig.from_yaml(CONFIG)
    assert (
        build_variant(config, index=None, models=models).answer("private fixture question").text
        == "Yes"
    )
    assert calls == [0, 60, 120]
    assert len(caplog.records) == 2 and "private fixture" not in caplog.text


@pytest.mark.parametrize("failures", [503, 400, httpx.ReadTimeout, httpx.ConnectTimeout])
def test_remote_retry_bounds_and_permanent_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failures: int | type[httpx.TimeoutException]
):
    clock = FakeClock()
    calls: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(clock.now)
        if isinstance(failures, int):
            return httpx.Response(failures)
        if len(calls) == 1:
            raise failures("temporary", request=request)
        return completion("Yes")

    models = gemini(monkeypatch, tmp_path, clock, httpx.MockTransport(respond))
    config = VariantConfig.from_yaml(CONFIG)
    variant = build_variant(config, index=None, models=models)
    if failures == 503:
        with pytest.raises(httpx.HTTPError, match=r"Gemini .*6 attempts.*HTTP 503"):
            variant.answer("Question?")
        assert len(calls) == 6 and 0 < calls[-1] < 300
    elif failures == 400:
        with pytest.raises(httpx.HTTPStatusError, match="400"):
            variant.answer("Question?")
        assert calls == [0]
    else:
        assert variant.answer("Question?").text == "Yes"
        assert len(calls) == 2 and calls[-1] > 0


def test_retry_after_http_date_and_time_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    clock = FakeClock()
    calls: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        calls.append(clock.now)
        if len(calls) == 1:
            return httpx.Response(502, headers={"retry-after": "Thu, 01 Jan 1970 00:00:15 GMT"})
        return httpx.Response(504, headers={"retry-after": "300"})

    models = gemini(monkeypatch, tmp_path, clock, httpx.MockTransport(respond))
    variant = build_variant(VariantConfig.from_yaml(CONFIG), index=None, models=models)
    with pytest.raises(httpx.HTTPError, match=r"2 attempts.*HTTP 504"):
        variant.answer("Question?")
    assert calls == [0, 15] and clock.now == 15


def test_retry_does_not_wait_past_budget_for_quota(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    clock = FakeClock()
    calls: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        calls.append(clock.now)
        return httpx.Response(500)

    models = gemini(
        monkeypatch, tmp_path, clock, httpx.MockTransport(respond), RateLimits(requests_per_day=1)
    )
    variant = build_variant(VariantConfig.from_yaml(CONFIG), index=None, models=models)
    with pytest.raises(httpx.HTTPError, match=r"1 attempt.*HTTP 500.*quota"):
        variant.answer("Question?")
    assert calls == [0] and clock.now < 300
