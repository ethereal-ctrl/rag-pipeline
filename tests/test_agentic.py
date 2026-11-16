import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from conftest import ARTICLES, FTX, FTX_QUERY, PIXEL
from fakes import FakeModels, GroundedFakeGenerator, PhraseFakeReranker
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_citations import CitationGenerator, cited_config

from rag_pipeline.cache import ResponseCache
from rag_pipeline.corpus import BenchmarkQuery, Evidence
from rag_pipeline.evaluation import RunStore, evaluate
from rag_pipeline.index import ChunkIndex
from rag_pipeline.ingest import ChunkingSettings, ingest
from rag_pipeline.models import Completion
from rag_pipeline.variants import VariantConfig, build_variant, plan_remote_run

FIRST = replace(FTX, title="Amber cell inventor", body="Lena Vale invented the amber battery.")
SECOND = replace(
    PIXEL,
    title="Lena Vale museum",
    body="Lena Vale founded the museum in the Northbank district.",
)
QUESTION = BenchmarkQuery(
    "q_museum",
    "In which district did the inventor of the amber battery found the museum?",
    "Northbank",
    "inference_query",
    (Evidence(FIRST.url, FIRST.body), Evidence(SECOND.url, SECOND.body)),
)
FIRST_QUERY = "amber battery inventor"
SECOND_QUERY = "Lena Vale museum district"


def agentic_config(
    *,
    max_steps: int = 3,
    max_tokens: int = 8000,
    max_context_chunks: int = 2,
    support_check: str = "llm",
) -> VariantConfig:
    data = cited_config(support_check=support_check).model_dump()
    data["retrieval"]["top_k"] = 5
    data["agentic"] = {
        "max_steps": max_steps,
        "max_tokens": max_tokens,
        "max_context_chunks": max_context_chunks,
    }
    return VariantConfig.model_validate(data)


class SearchGenerator(GroundedFakeGenerator):
    def __init__(
        self,
        *,
        plan: str | None = None,
        decisions: tuple[str, ...] = (),
        usage: dict[str, tuple[int, int]] | None = None,
        required_chunks: int = 2,
    ) -> None:
        super().__init__({})
        self.plan = plan if plan is not None else json.dumps({"sub_queries": [FIRST_QUERY]})
        self.decisions = iter(
            decisions
            or (
                json.dumps({"status": "insufficient", "next_query": SECOND_QUERY}),
                '{"status":"sufficient","next_query":null}',
            )
        )
        self.calls: list[str] = []
        self.inputs: dict[str, list[str]] = {}
        self.usage = usage or {}
        self.required_chunks = required_chunks

    def complete_structured(self, system: str, prompt: str, schema: dict[str, Any]) -> Completion:
        properties = schema["properties"]
        if "sub_queries" in properties:
            kind, text = "plan", self.plan
        elif "status" in properties:
            kind, text = "decide", next(self.decisions)
        elif "verdicts" in properties:
            kind, text = "check", '{"verdicts":[{"statement_index":0,"supported":true}]}'
        else:
            kind = "generate"
            ids = re.findall(r'<chunk id="([^"]+)"', prompt)
            statements = (
                [{"statement": "The inventor founded the museum in Northbank.", "chunk_ids": ids}]
                if len(ids) == self.required_chunks
                else []
            )
            text = json.dumps({"answer": "Northbank", "statements": statements})
        self.calls.append(kind)
        self.inputs.setdefault(kind, []).append(prompt)
        return Completion(text, *self.usage.get(kind, (40, 10)))


def test_two_hops_add_evidence_and_answer_with_citations(index: ChunkIndex, tmp_path: Path):
    ingest([FIRST, SECOND], index, ChunkingSettings(max_words=100, overlap_sentences=0))
    config = agentic_config()
    model = SearchGenerator()
    tracer = TracerProvider()
    exporter = InMemorySpanExporter()
    tracer.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        variant = build_variant(
            config, index=index, models=FakeModels(model), tracer=tracer.get_tracer(__name__)
        )
        result = evaluate(variant, [QUESTION], "smoke", RunStore(tmp_path))
        spans = {span.name: span for span in exporter.get_finished_spans()}
    finally:
        tracer.shutdown()

    (record,) = result.records
    assert record.answer == "Northbank" and record.correct
    assert record.search_steps == 2
    assert record.stop_reason == "sufficient"
    (citation,) = record.citations
    assert citation.chunk_ids == record.retrieved_chunk_ids
    assert citation.chunk_ids == tuple(f"{article.id}#000" for article in (FIRST, SECOND))
    assert result.metrics["evidence_recall@10"] == 1
    assert result.metrics["search_steps"] == 2
    assert result.metrics["stop_reason[sufficient]"] == 1
    assert result.intervals["search_steps"] == (2, 2)
    assert result.intervals["stop_reason[sufficient]"] == (1, 1)
    assert RunStore(tmp_path).load(result.run_id).records == result.records
    assert model.calls == ["plan", "decide", "decide", "generate", "check"]
    assert [stage.name for stage in record.trace] == [
        "plan",
        "retrieve.1",
        "decide.1",
        "retrieve.2",
        "decide.2",
        "generate",
        "check",
    ]
    assert list(record.stage_seconds) == [stage.name for stage in record.trace]
    assert (record.input_tokens, record.output_tokens) == (200, 50)
    assert result.metrics["cost_per_question"] == (200 * 0.10 + 50 * 0.20) / 1e6
    for stage in record.trace:
        expected = (0, 0) if stage.name.startswith("retrieve") else (40, 10)
        assert (stage.input_tokens, stage.output_tokens) == expected
    for step, query in [(1, FIRST_QUERY), (2, SECOND_QUERY)]:
        for kind in ("retrieve", "decide"):
            attrs = spans[f"{kind}.{step}"].attributes
            assert attrs is not None and attrs["rag.sub_query"] == query
    assert FIRST.body in model.inputs["decide"][0]
    assert SECOND.body not in model.inputs["decide"][0]
    assert SECOND.body in model.inputs["decide"][1]


def test_later_hop_reaches_the_answer_after_a_full_first_retrieval(index: ChunkIndex):
    distractors = [
        replace(
            FIRST,
            id=f"a_amber_archive_{i}",
            body="Amber battery archives. " + "The records describe earlier designs. " * 8,
        )
        for i in range(4)
    ]
    ingest([FIRST, SECOND, *distractors], index, ChunkingSettings(max_words=100))
    model = SearchGenerator()
    variant = build_variant(agentic_config(), index=index, models=FakeModels(model))

    answer = variant.answer(QUESTION.question)

    first_id, second_id = (f"{article.id}#000" for article in (FIRST, SECOND))
    assert {chunk.article_id for chunk in answer.retrieved[:5]} == {
        FIRST.id,
        *(article.id for article in distractors),
    }
    assert answer.retrieved[5].chunk_id == second_id
    assert answer.shown_chunk_ids == (first_id, second_id)
    assert answer.citations[0].chunk_ids == (first_id, second_id)
    assert answer.text == "Northbank"
    decision_ids = [re.findall(r'<chunk id="([^"]+)"', prompt) for prompt in model.inputs["decide"]]
    assert len(decision_ids[0]) == 3  # the default per-step contribution
    assert decision_ids[1] == [*decision_ids[0], second_id]


def test_cited_defaults_preserve_the_single_search_and_fingerprint(
    index: ChunkIndex, tmp_path: Path
):
    ingest(ARTICLES, index, ChunkingSettings(max_words=100, overlap_sentences=0))
    generator = CitationGenerator([("Sam Bankman-Fried stood trial.", ["shown"])])
    variant = build_variant(cited_config(), index=index, models=FakeModels(generator))

    result = evaluate(variant, [FTX_QUERY], "smoke", RunStore(tmp_path))

    (record,) = result.records
    assert record.answer == "Sam Bankman-Fried."
    assert [stage.name for stage in record.trace] == ["retrieve", "generate"]
    assert len(generator.prompts) == 1
    assert result.meta["fingerprint"] == "d930a1da6a8aa414"
    assert "agentic" not in result.meta["variant"]
    assert record.search_steps is None and record.stop_reason is None
    assert "search_steps" not in result.metrics


def test_step_limit_deduplicates_the_evidence_pool(index: ChunkIndex):
    ingest([FIRST, SECOND], index, ChunkingSettings(max_words=100, overlap_sentences=0))
    insufficient = json.dumps({"status": "insufficient", "next_query": FIRST_QUERY})
    model = SearchGenerator(decisions=(insufficient,) * 2, required_chunks=1)
    variant = build_variant(agentic_config(max_steps=2), index=index, models=FakeModels(model))

    answer = variant.answer(QUESTION.question)

    assert answer.search_steps == 2
    assert answer.stop_reason == "max_steps"
    assert [chunk.article_id for chunk in answer.retrieved] == [FIRST.id]
    assert model.calls == ["plan", "decide", "decide", "generate", "check"]


def test_final_context_cap_keeps_full_evidence_pool_for_evaluation(index: ChunkIndex):
    ingest([FIRST, SECOND], index, ChunkingSettings(max_words=100, overlap_sentences=0))
    model = SearchGenerator(required_chunks=1)
    variant = build_variant(
        agentic_config(max_context_chunks=1), index=index, models=FakeModels(model)
    )

    answer = variant.answer(QUESTION.question)

    assert len(answer.retrieved) == 2
    assert answer.shown_chunk_ids == (answer.retrieved[0].chunk_id,)
    assert answer.citations[0].chunk_ids == answer.shown_chunk_ids
    assert SECOND.body not in model.inputs["generate"][0]


def test_search_budget_can_stop_a_decision_and_still_leave_room_for_the_answer(
    index: ChunkIndex,
):
    long_article = replace(SECOND, body=SECOND.body + " More background" * 100 + ".")
    ingest([FIRST, long_article], index, ChunkingSettings(max_words=500, overlap_sentences=0))
    model = SearchGenerator(required_chunks=1)
    variant = build_variant(
        agentic_config(max_tokens=2000, max_context_chunks=1),
        index=index,
        models=FakeModels(model),
    )

    answer = variant.answer(QUESTION.question)

    assert answer.search_steps == 2 and answer.stop_reason == "max_tokens"
    assert tuple(chunk.article_id for chunk in answer.retrieved) == (FIRST.id, SECOND.id)
    assert not answer.abstained
    assert model.calls == ["plan", "decide", "generate", "check"]


@pytest.mark.parametrize("final_cap", [None, 4], ids=["example-config", "truncated"])
def test_example_budget_and_rank_interleaving_cover_three_hops(
    index: ChunkIndex, final_cap: int | None
):
    class RankedReranker(PhraseFakeReranker):
        def score(self, query: str, texts: list[str]) -> list[float]:
            self.calls.append((query, len(texts)))
            return [-float(text.split()[0]) for text in texts]

    class MeasuredGenerator(SearchGenerator):
        def complete_structured(
            self, system: str, prompt: str, schema: dict[str, Any]
        ) -> Completion:
            completion = super().complete_structured(system, prompt, schema)
            tokens = (len((system + prompt).encode("utf-8")) + 2) // 3 + 32
            return replace(completion, input_tokens=tokens)

    queries = ("ambercell", "museumledger", "districtmap")
    articles = [
        replace(
            FIRST,
            id=f"a_agentic_{query}_{rank}",
            title=f"{query} article {rank}",
            body=f"{rank} {query}. " + "The inventor Lena Vale founded the Northbank museum. " * 19,
        )
        for query in queries
        for rank in range(10)
    ]
    ingest(articles, index, ChunkingSettings(max_words=200, overlap_sentences=0))
    settings = VariantConfig.from_yaml(Path("configs/variants/agentic.yaml"))
    data = settings.model_dump()
    data["index"] = {"collection": "fixture"}
    data["retrieval"]["mode"] = "bm25"
    data["retrieval"]["rerank"]["candidates"] = 10
    if final_cap is not None:
        data["agentic"]["max_context_chunks"] = final_cap
    config = VariantConfig.model_validate(data)
    model = MeasuredGenerator(
        plan=json.dumps({"sub_queries": queries}),
        decisions=(
            json.dumps({"status": "insufficient", "next_query": queries[1]}),
            json.dumps({"status": "insufficient", "next_query": queries[2]}),
            '{"status":"sufficient","next_query":null}',
        ),
        required_chunks=final_cap if final_cap is not None else 6,
    )
    reranker = RankedReranker("Northbank")
    variant = build_variant(config, index=index, models=FakeModels(model, reranker))

    answer = variant.answer(QUESTION.question)

    assert model.calls == ["plan", "decide", "decide", "decide", "generate", "check"]
    assert answer.search_steps == 3 and answer.stop_reason == "sufficient"
    assert answer.input_tokens + answer.output_tokens <= 16000
    assert all(
        stage.input_tokens + stage.output_tokens <= 8192
        for stage in answer.trace
        if stage.input_tokens is not None
    )
    assert tuple(chunk.article_id for chunk in answer.retrieved) == tuple(
        article.id for article in articles
    )
    expected = tuple(f"a_agentic_{query}_{rank}#000" for rank in range(2) for query in queries)
    assert answer.shown_chunk_ids == expected[:final_cap]
    assert answer.citations[0].chunk_ids == expected[:final_cap]
    for step, prompt in enumerate(model.inputs["decide"], 1):
        assert re.findall(r'<chunk id="([^"]+)"', prompt) == [
            f"a_agentic_{query}_{rank}#000" for query in queries[:step] for rank in range(2)
        ]
    assert reranker.calls == [(query, 10) for query in queries]
    assert [stage.name for stage in answer.trace if stage.name.startswith("rerank")] == [
        "rerank.1",
        "rerank.2",
        "rerank.3",
    ]


def test_malformed_plan_falls_back_and_malformed_decision_ends_search(index: ChunkIndex):
    ingest([FIRST, SECOND], index, ChunkingSettings(max_words=100, overlap_sentences=0))
    model = SearchGenerator(plan="not JSON", decisions=("not JSON",), required_chunks=1)
    variant = build_variant(agentic_config(), index=index, models=FakeModels(model))

    answer = variant.answer(FIRST_QUERY)

    assert answer.search_steps == 1
    assert answer.stop_reason == "sufficient"
    assert [chunk.article_id for chunk in answer.retrieved] == [FIRST.id]
    assert model.calls == ["plan", "decide", "generate", "check"]
    assert [stage.name for stage in answer.trace if stage.malformed] == ["plan", "decide.1"]


@pytest.mark.parametrize(
    ("max_tokens", "usage", "calls", "steps"),
    [
        (50, {}, [], 0),
        (3000, {"decide": (2000, 100)}, ["plan", "decide"], 1),
        (4000, {"generate": (3200, 100)}, ["plan", "decide", "decide", "generate"], 2),
    ],
    ids=["cannot-plan", "cannot-continue-search", "cannot-check"],
)
def test_token_budget_checks_every_generator_call(
    index: ChunkIndex,
    max_tokens: int,
    usage: dict[str, tuple[int, int]],
    calls: list[str],
    steps: int,
):
    ingest([FIRST, SECOND], index, ChunkingSettings(max_words=100, overlap_sentences=0))
    model = SearchGenerator(usage=usage)
    variant = build_variant(
        agentic_config(max_tokens=max_tokens), index=index, models=FakeModels(model)
    )

    answer = variant.answer(QUESTION.question)

    assert answer.stop_reason == "max_tokens"
    assert answer.search_steps == steps
    assert model.calls == calls
    assert answer.input_tokens + answer.output_tokens <= max_tokens
    assert answer.abstained
    assert answer.text == "Insufficient information."
    if calls and calls[-1] == "generate":
        assert answer.removed_statements[0].reason == "max_tokens"


def test_agentic_config_requires_citations():
    data = agentic_config().model_dump()
    data["generation"]["citations"]["enabled"] = False
    with pytest.raises(ValueError, match="Agentic Search requires Citations"):
        build_variant(
            VariantConfig.model_validate(data), index=None, models=FakeModels(SearchGenerator())
        )


def test_remote_plan_budgets_max_steps_and_counts_known_cached_calls(
    index: ChunkIndex, tmp_path: Path
):
    ingest([FIRST, SECOND], index, ChunkingSettings(max_words=100, overlap_sentences=0))
    data = agentic_config().model_dump()
    data["generation"]["generator"].update(provider="groq", model="openai/gpt-oss-120b")
    config = VariantConfig.model_validate(data)
    cache = ResponseCache(tmp_path / "responses.sqlite3")
    model = SearchGenerator()
    variant = build_variant(config, index=index, models=FakeModels(model), cache=cache)

    variant.answer(QUESTION.question)
    cached = variant.answer(QUESTION.question)
    plan = plan_remote_run(config, [QUESTION], cache, index_version=variant.index_version())

    assert model.calls == ["plan", "decide", "decide", "generate", "check"]
    assert all(stage.cache_hit for stage in cached.trace if stage.model == "openai/gpt-oss-120b")
    assert (plan.calls, plan.cached, plan.days, plan.upper_bound) == (6, 5, 1, True)
