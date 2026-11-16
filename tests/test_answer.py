import time
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import ARTICLES, FTX, FTX_QUERY, NULL_QUERY
from fakes import FakeModels, GroundedFakeGenerator, PhraseFakeReranker
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_pipeline.index import ChunkIndex
from rag_pipeline.ingest import ChunkingSettings, ingest
from rag_pipeline.models import EmbedderSpec, GeneratorSpec, RerankerSpec
from rag_pipeline.tracing import cli_tracer
from rag_pipeline.variants import (
    Answer,
    GenerationConfig,
    IndexConfig,
    PipelineVariant,
    RerankConfig,
    RetrievalConfig,
    VariantConfig,
    build_variant,
)

BM25_YAML = """
name: bm25
index:
  collection: fixture
retrieval:
  mode: bm25
  top_k: 3
generation:
  generator:
    provider: ollama
    model: qwen3.5:4b
"""


@pytest.fixture
def models() -> FakeModels:
    return FakeModels(GroundedFakeGenerator({"Sam Bankman-Fried started": "Sam Bankman-Fried"}))


@pytest.fixture
def bm25_config(tmp_path: Path) -> VariantConfig:
    path = tmp_path / "bm25.yaml"
    path.write_text(BM25_YAML)
    return VariantConfig.from_yaml(path)


def test_bm25_variant_answers_from_retrieved_chunks(
    index: ChunkIndex, models: FakeModels, bm25_config: VariantConfig
):
    ingest(ARTICLES, index, ChunkingSettings(max_words=20, overlap_sentences=0))
    variant = build_variant(bm25_config, index=index, models=models)

    answer = variant.answer(FTX_QUERY.question)

    assert answer.text == "Sam Bankman-Fried"
    assert not answer.abstained
    assert answer.retrieved[0].article_id == FTX.id
    assert len(answer.retrieved) <= 3


def test_variant_abstains_when_retrieved_chunks_do_not_support_an_answer(
    index: ChunkIndex, models: FakeModels, bm25_config: VariantConfig
):
    ingest(ARTICLES, index, ChunkingSettings(max_words=20, overlap_sentences=0))
    variant = build_variant(bm25_config, index=index, models=models)

    answer = variant.answer(NULL_QUERY.question)

    assert answer.abstained


CLOSED_BOOK_YAML = """
name: closed-book
retrieval:
  mode: none
generation:
  generator:
    provider: ollama
    model: qwen3.5:4b
"""


def test_closed_book_variant_answers_without_any_retrieval(tmp_path: Path):
    path = tmp_path / "closed-book.yaml"
    path.write_text(CLOSED_BOOK_YAML)
    generator = GroundedFakeGenerator({"FTX": "Sam Bankman-Fried"})
    variant = build_variant(VariantConfig.from_yaml(path), index=None, models=FakeModels(generator))

    answer = variant.answer(FTX_QUERY.question)

    assert answer.retrieved == ()
    assert answer.text == "Sam Bankman-Fried"
    assert "<chunk" not in generator.prompts[0]
    assert FTX.body.split(".")[0] not in generator.prompts[0]


def test_a_retrieval_variant_without_an_index_is_rejected(tmp_path: Path):
    path = tmp_path / "bm25.yaml"
    path.write_text(BM25_YAML.replace("index:\n  collection: fixture\n", ""))

    with pytest.raises(ValueError, match="index"):
        VariantConfig.from_yaml(path)


def _retrieval_variant(mode: str, index: ChunkIndex) -> PipelineVariant:
    config = VariantConfig(
        name=mode,
        index=IndexConfig(
            collection=index.collection, embedder=EmbedderSpec(provider="ollama", model="fake")
        ),
        retrieval=RetrievalConfig(mode=mode, top_k=3),  # type: ignore[arg-type]
        generation=GenerationConfig(generator=GeneratorSpec(provider="ollama", model="m")),
    )
    return build_variant(config, index=index, models=FakeModels(GroundedFakeGenerator({})))


# Shares no words with the Pixel camera Chunk, only synonyms of its words.
SYNONYM_QUESTION = "Does the smartphone take good pictures in the dark?"
PIXEL_CAMERA_SENTENCE = "Its camera uses a new sensor for low light photos."


def test_dense_retrieval_finds_chunks_that_share_no_words_with_the_question(
    dense_index: ChunkIndex,
):
    ingest(ARTICLES, dense_index, ChunkingSettings(max_words=20, overlap_sentences=0))

    dense = _retrieval_variant("dense", dense_index).answer(SYNONYM_QUESTION)
    lexical = _retrieval_variant("bm25", dense_index).answer(SYNONYM_QUESTION)

    assert PIXEL_CAMERA_SENTENCE in dense.retrieved[0].text
    assert not any(PIXEL_CAMERA_SENTENCE in c.text for c in lexical.retrieved)


def test_hybrid_retrieval_fuses_lexical_and_dense_matches(dense_index: ChunkIndex):
    ingest(ARTICLES, dense_index, ChunkingSettings(max_words=20, overlap_sentences=0))
    # "Bankman-Fried" is only found lexically; the camera Chunk only via synonyms.
    question = "Bankman-Fried jury, and smartphone pictures in the dark?"

    hybrid = _retrieval_variant("hybrid", dense_index).answer(question)
    lexical = _retrieval_variant("bm25", dense_index).answer(question)

    assert lexical.retrieved[0].article_id == FTX.id
    assert not any(PIXEL_CAMERA_SENTENCE in c.text for c in lexical.retrieved)
    assert any(c.article_id == FTX.id for c in hybrid.retrieved)
    assert any(PIXEL_CAMERA_SENTENCE in c.text for c in hybrid.retrieved)


def test_dense_variant_without_an_index_embedder_is_rejected():
    with pytest.raises(ValueError, match="embedder"):
        VariantConfig(
            name="dense",
            index=IndexConfig(collection="c"),
            retrieval=RetrievalConfig(mode="dense"),
            generation=GenerationConfig(generator=GeneratorSpec(provider="ollama", model="m")),
        )


def test_reranker_reorders_first_stage_candidates(dense_index: ChunkIndex):
    ingest(ARTICLES, dense_index, ChunkingSettings(max_words=20, overlap_sentences=0))
    question = "What did prosecutors say about the FTX trial and the jury?"
    reranker = PhraseFakeReranker("The jury will hear")
    config = VariantConfig(
        name="hybrid-rerank",
        index=IndexConfig(
            collection=dense_index.collection,
            embedder=EmbedderSpec(provider="ollama", model="fake"),
        ),
        retrieval=RetrievalConfig(
            mode="hybrid",
            top_k=2,
            rerank=RerankConfig(
                reranker=RerankerSpec(provider="sentence-transformers", model="fake"),
                candidates=6,
            ),
        ),
        generation=GenerationConfig(generator=GeneratorSpec(provider="ollama", model="m")),
    )
    first_stage = _retrieval_variant("hybrid", dense_index).answer(question)
    variant = build_variant(
        config, index=dense_index, models=FakeModels(GroundedFakeGenerator({}), reranker)
    )

    answer = variant.answer(question)

    assert "The jury will hear" not in first_stage.retrieved[0].text
    assert "The jury will hear" in answer.retrieved[0].text
    assert len(answer.retrieved) == 2
    assert reranker.calls == [(question, 6)]
    assert answer.stage_seconds.keys() == {"retrieve", "rerank", "generate"}


def test_answer_trace_keeps_stage_timings_and_generation_tokens(
    index: ChunkIndex, models: FakeModels, bm25_config: VariantConfig
):
    ingest(ARTICLES, index, ChunkingSettings(max_words=20, overlap_sentences=0))
    variant = build_variant(bm25_config, index=index, models=models)

    answer = variant.answer(FTX_QUERY.question)

    retrieve, generate = answer.trace
    assert (retrieve.name, retrieve.model, retrieve.input_tokens, retrieve.output_tokens) == (
        "retrieve",
        None,
        0,
        0,
    )
    assert (generate.name, generate.model) == ("generate", "qwen3.5:4b")
    assert (generate.input_tokens, generate.output_tokens) == (
        answer.input_tokens,
        answer.output_tokens,
    )
    assert answer.input_tokens > 0
    assert answer.output_tokens > 0
    assert answer.stage_seconds == {stage.name: stage.seconds for stage in answer.trace}
    assert all(stage.seconds > 0 for stage in answer.trace)


@pytest.mark.parametrize("rerank", [False, True])
def test_answer_exports_stage_spans_without_changing_results(
    index: ChunkIndex, bm25_config: VariantConfig, rerank: bool
):
    ingest(ARTICLES, index, ChunkingSettings(max_words=20, overlap_sentences=0))
    config = bm25_config
    if rerank:
        config = VariantConfig.model_validate(
            {
                **config.model_dump(),
                "retrieval": {
                    **config.retrieval.model_dump(),
                    "rerank": {"reranker": {"provider": "sentence-transformers", "model": "fake"}},
                },
            }
        )
    models = FakeModels(
        GroundedFakeGenerator({"Sam Bankman-Fried started": "Sam Bankman-Fried"}),
        PhraseFakeReranker("The jury will hear"),
    )
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        plain = build_variant(config, index=index, models=models).answer(FTX_QUERY.question)
        traced = build_variant(
            config, index=index, models=models, tracer=provider.get_tracer(__name__)
        ).answer(FTX_QUERY.question)

        def without_timings(answer: Answer) -> Answer:
            return replace(
                answer,
                seconds=0,
                stage_seconds={},
                trace=tuple(replace(stage, seconds=0) for stage in answer.trace),
            )

        assert without_timings(plain) == without_timings(traced)
        spans = {span.name: span for span in exporter.get_finished_spans()}
        kinds = {"answer": "CHAIN", "retrieve": "RETRIEVER", "generate": "LLM"}
        if rerank:
            kinds["rerank"] = "RERANKER"
        assert spans.keys() == kinds.keys()
        root = spans["answer"]
        assert root.parent is None
        for name, span in spans.items():
            if name != "answer":
                assert span.parent == root.context
            assert span.attributes is not None
            assert span.attributes["openinference.span.kind"] == kinds[name]
            assert span.attributes["rag.variant"] == config.name
            assert span.attributes["rag.chunk_ids"]
            assert FTX_QUERY.question not in repr(span.attributes)
            assert all(chunk.text not in repr(span.attributes) for chunk in traced.retrieved)
        generate = spans["generate"].attributes
        assert generate is not None
        assert generate["gen_ai.operation.name"] == "chat"
        assert generate["gen_ai.request.model"] == "qwen3.5:4b"
        assert generate["gen_ai.usage.input_tokens"] == traced.input_tokens
        assert generate["gen_ai.usage.output_tokens"] == traced.output_tokens
        assert root.attributes is not None
        assert root.attributes["rag.chunk_ids"] == tuple(c.chunk_id for c in traced.retrieved)
    finally:
        provider.shutdown()


def test_answer_trace_counts_dense_query_and_reranker_input_tokens(dense_index: ChunkIndex):
    ingest(ARTICLES, dense_index, ChunkingSettings(max_words=20, overlap_sentences=0))
    config = VariantConfig(
        name="hybrid-rerank",
        index=IndexConfig(
            collection=dense_index.collection,
            embedder=EmbedderSpec(provider="ollama", model="fake-embedder"),
        ),
        retrieval=RetrievalConfig(
            mode="hybrid",
            top_k=2,
            rerank=RerankConfig(
                reranker=RerankerSpec(provider="sentence-transformers", model="fake-reranker"),
                candidates=2,
            ),
        ),
        generation=GenerationConfig(generator=GeneratorSpec(provider="ollama", model="m")),
    )
    answer = build_variant(
        config,
        index=dense_index,
        models=FakeModels(GroundedFakeGenerator({}), PhraseFakeReranker("The jury will hear")),
    ).answer(FTX_QUERY.question)

    retrieve, rerank, generate = answer.trace
    assert (retrieve.model, retrieve.input_tokens, retrieve.output_tokens) == (
        "fake-embedder",
        len(FTX_QUERY.question.split()),
        0,
    )
    assert (rerank.model, rerank.input_tokens, rerank.output_tokens) == (
        "fake-reranker",
        sum(
            len(FTX_QUERY.question.split()) + len(chunk.text.split()) for chunk in answer.retrieved
        ),
        0,
    )
    assert generate.input_tokens == answer.input_tokens


@pytest.mark.parametrize("flag", ["0", "1"])
def test_closed_book_answer_still_works_without_a_collector(
    monkeypatch: pytest.MonkeyPatch, flag: str
):
    monkeypatch.setenv("RAG_TRACING", flag)
    monkeypatch.setenv("RAG_TRACING_ENDPOINT", "http://127.0.0.1:0/v1/traces")
    config = VariantConfig.model_validate(
        {
            "name": "closed-book",
            "retrieval": {"mode": "none"},
            "generation": {"generator": {"provider": "ollama", "model": "qwen3.5:4b"}},
        }
    )
    models = FakeModels(GroundedFakeGenerator({"FTX": "Sam Bankman-Fried"}))

    with cli_tracer() as tracer:
        answer = build_variant(config, index=None, models=models, tracer=tracer).answer(
            FTX_QUERY.question
        )

    assert tracer is None
    assert answer.text == "Sam Bankman-Fried"
    assert answer.trace[-1].name == "generate"


def test_rerank_latency_excludes_counting_tokens(index: ChunkIndex, bm25_config: VariantConfig):
    ingest(ARTICLES, index, ChunkingSettings(max_words=20, overlap_sentences=0))
    config = VariantConfig.model_validate(
        {
            **bm25_config.model_dump(),
            "retrieval": {
                **bm25_config.retrieval.model_dump(),
                "rerank": {"reranker": {"provider": "sentence-transformers", "model": "fake"}},
            },
        }
    )

    class SlowTokenCounter(PhraseFakeReranker):
        counting_seconds = 0.0

        def count_input_tokens(self, query: str, texts: list[str]) -> int:
            started = time.perf_counter()
            time.sleep(0.1)
            self.counting_seconds = time.perf_counter() - started
            return sum(len(query.split()) + len(text.split()) for text in texts)

    reranker = SlowTokenCounter("The jury will hear")
    variant = build_variant(
        config, index=index, models=FakeModels(GroundedFakeGenerator({}), reranker)
    )

    answer = variant.answer(FTX_QUERY.question)

    assert reranker.counting_seconds >= 0.1
    assert answer.stage_seconds["rerank"] < reranker.counting_seconds
    rerank = next(stage for stage in answer.trace if stage.name == "rerank")
    assert rerank.input_tokens is not None and rerank.input_tokens > 0
