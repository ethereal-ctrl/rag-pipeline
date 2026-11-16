from pathlib import Path

import pytest
from conftest import ARTICLES, PIXEL, QUERIES
from fakes import FakeModels, GroundedFakeGenerator

from rag_pipeline.corpus import BenchmarkQuery, Evidence
from rag_pipeline.evaluation import RunStore, evaluate
from rag_pipeline.index import ChunkIndex
from rag_pipeline.ingest import ChunkingSettings, ingest
from rag_pipeline.models import GeneratorSpec
from rag_pipeline.variants import (
    GenerationConfig,
    IndexConfig,
    PipelineVariant,
    RetrievalConfig,
    VariantConfig,
    build_variant,
)


def _bm25_variant(index: ChunkIndex, top_k: int = 10) -> PipelineVariant:
    config = VariantConfig(
        name="bm25",
        index=IndexConfig(collection=index.collection),
        retrieval=RetrievalConfig(mode="bm25", top_k=top_k),
        generation=GenerationConfig(generator=GeneratorSpec(provider="ollama", model="m")),
    )
    fake = GroundedFakeGenerator({"Sam Bankman-Fried started": "It is Sam Bankman-Fried."})
    return build_variant(config, index=index, models=FakeModels(fake))


@pytest.fixture
def ingested(index: ChunkIndex) -> ChunkIndex:
    ingest(ARTICLES, index, ChunkingSettings(max_words=20, overlap_sentences=0))
    return index


def test_scores_retrieval_hits_and_answer_containment(ingested: ChunkIndex, tmp_path: Path):
    result = evaluate(_bm25_variant(ingested), QUERIES, split="smoke", store=RunStore(tmp_path))

    # Both answerable queries retrieve their Evidence; the Null Query is excluded.
    assert result.metrics["hits@10"] == 1.0
    assert result.metrics["hits@4"] == 1.0
    # FTX answered correctly, Null Query correctly abstained, Pixel wrongly abstained.
    assert result.metrics["accuracy"] == pytest.approx(2 / 3)
    assert result.metrics["queries"] == 3


UPDATES_QUERY = BenchmarkQuery(
    id="q_updates",
    question="How many years of software updates does the new phone get?",
    answer="seven",
    query_type="comparison_query",
    # Spacing and line breaks differ from the Article text on purpose.
    evidence=(Evidence(PIXEL.url, "The new phone   ships with seven years\nof software"),),
)


def test_evidence_matching_ignores_whitespace_differences(ingested: ChunkIndex, tmp_path: Path):
    result = evaluate(_bm25_variant(ingested), [UPDATES_QUERY], "smoke", RunStore(tmp_path))

    assert result.records[0].hit_at_4


def test_evidence_outside_the_retrieved_chunks_is_not_a_hit(ingested: ChunkIndex, tmp_path: Path):
    # The question matches the Pixel launch Chunk, but its Evidence sits in the next Chunk.
    misleading = BenchmarkQuery(
        id="q_launch",
        question="When did Google announce the Pixel 8 at its October event?",
        answer="October",
        query_type="temporal_query",
        evidence=UPDATES_QUERY.evidence,
    )

    result = evaluate(
        _bm25_variant(ingested, top_k=1), [UPDATES_QUERY, misleading], "smoke", RunStore(tmp_path)
    )

    records = {r.query_id: r for r in result.records}
    assert records["q_updates"].hit_at_4
    assert not records["q_launch"].hit_at_4
    assert result.metrics["hits@4"] == 0.5


def test_run_is_stored_with_its_metadata_and_records(ingested: ChunkIndex, tmp_path: Path):
    store = RunStore(tmp_path)

    result = evaluate(_bm25_variant(ingested), QUERIES, split="smoke", store=store)

    loaded = store.load(result.run_id)
    assert loaded.metrics == result.metrics
    assert loaded.records == result.records
    assert loaded.meta["variant"]["name"] == "bm25"
    assert loaded.meta["split"] == "smoke"
    assert loaded.meta["generator"] == "fake-grounded"
    assert "commit" in loaded.meta


def test_every_run_records_the_index_version_it_was_scored_on(ingested: ChunkIndex, tmp_path: Path):
    version = ingest(ARTICLES, ingested, ChunkingSettings(20, 0)).index_version

    result = evaluate(_bm25_variant(ingested), QUERIES, "smoke", RunStore(tmp_path))

    assert result.meta["index_version"] == version
    assert version.startswith("fixture@")
