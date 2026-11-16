from pathlib import Path

import pytest
from conftest import ARTICLES, FTX, FTX_QUERY, NULL_QUERY
from fakes import FakeModels, GroundedFakeGenerator

from rag_pipeline.index import ChunkIndex
from rag_pipeline.ingest import ChunkingSettings, ingest
from rag_pipeline.variants import VariantConfig, build_variant

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
