"""Oracle retrieval: gold Evidence Chunks in the context, tested at the Pipeline Variant seam."""

import pytest
from conftest import FTX
from fakes import FakeModels, GroundedFakeGenerator
from qdrant_client import QdrantClient
from test_source_aware import VERGE, VERGE_FACT

from rag_pipeline.chunking import ChunkingSettings
from rag_pipeline.corpus import Evidence
from rag_pipeline.index import ChunkIndex
from rag_pipeline.ingest import ingest
from rag_pipeline.models import GeneratorSpec
from rag_pipeline.variants import (
    GenerationConfig,
    IndexConfig,
    RetrievalConfig,
    VariantConfig,
    build_variant,
)

QUESTION = "Did a witness describe the founder to the jury during the FTX trial?"


def _index() -> ChunkIndex:
    index = ChunkIndex(QdrantClient(":memory:"), collection="sources")
    ingest([FTX, VERGE], index, ChunkingSettings())
    return index


def _oracle() -> VariantConfig:
    return VariantConfig(
        name="oracle",
        index=IndexConfig(collection="sources"),
        retrieval=RetrievalConfig(mode="oracle", top_k=5),
        generation=GenerationConfig(
            generator=GeneratorSpec(provider="ollama", model="fake"), context_chunks=5
        ),
    )


def test_the_oracle_puts_the_gold_evidence_chunk_into_the_context():
    gold = {QUESTION: (Evidence(VERGE.url, VERGE_FACT),)}
    generator = GroundedFakeGenerator({VERGE_FACT: "Yes"})
    variant = build_variant(_oracle(), index=_index(), models=FakeModels(generator), gold=gold)
    answer = variant.answer(QUESTION)
    assert not answer.abstained and answer.text == "Yes"
    assert [c.article_id for c in answer.retrieved] == [VERGE.id]


def test_a_question_without_gold_evidence_gets_an_empty_context_and_abstains():
    generator = GroundedFakeGenerator({VERGE_FACT: "Yes"})
    variant = build_variant(
        _oracle(), index=_index(), models=FakeModels(generator), gold={QUESTION: ()}
    )
    answer = variant.answer(QUESTION)
    assert answer.abstained and answer.retrieved == ()


def test_the_oracle_refuses_to_run_without_gold_evidence():
    with pytest.raises(ValueError, match="gold Evidence"):
        build_variant(_oracle(), index=_index(), models=FakeModels(GroundedFakeGenerator({})))
