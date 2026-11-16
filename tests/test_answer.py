from pathlib import Path

import pytest
from conftest import ARTICLES, FTX, FTX_QUERY, NULL_QUERY
from fakes import FakeModels, GroundedFakeGenerator, PhraseFakeReranker

from rag_pipeline.index import ChunkIndex
from rag_pipeline.ingest import ChunkingSettings, ingest
from rag_pipeline.models import EmbedderSpec, GeneratorSpec, RerankerSpec
from rag_pipeline.variants import (
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
