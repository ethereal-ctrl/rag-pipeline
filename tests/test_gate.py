"""The retrieval gate: Smoke Set retrieval metrics checked against committed thresholds."""

from __future__ import annotations

import pytest
from conftest import ARTICLES, QUERIES
from fakes import FakeModels, GroundedFakeGenerator

from rag_pipeline.gate import retrieval_gate
from rag_pipeline.index import ChunkIndex
from rag_pipeline.ingest import ChunkingSettings, ingest
from rag_pipeline.models import Completion, GeneratorSpec
from rag_pipeline.variants import (
    GenerationConfig,
    IndexConfig,
    PipelineVariant,
    RetrievalConfig,
    VariantConfig,
    build_variant,
)


class ExplodingGenerator(GroundedFakeGenerator):
    """The gate must never call the LLM."""

    def __init__(self) -> None:
        super().__init__({})

    def complete(self, system: str, prompt: str) -> Completion:
        raise AssertionError("the retrieval gate called the generator")


@pytest.fixture
def bm25_variant(index: ChunkIndex) -> PipelineVariant:
    ingest(ARTICLES, index, ChunkingSettings(max_words=20, overlap_sentences=0))
    config = VariantConfig(
        name="bm25",
        index=IndexConfig(collection=index.collection),
        retrieval=RetrievalConfig(mode="bm25"),
        generation=GenerationConfig(generator=GeneratorSpec(provider="ollama", model="m")),
    )
    return build_variant(config, index=index, models=FakeModels(ExplodingGenerator()))


def test_gate_scores_retrieval_without_the_llm(bm25_variant: PipelineVariant):
    report = retrieval_gate([bm25_variant], QUERIES, {"bm25": {"hits@10": 1.0}})

    assert report.passed
    # The Null Query has no Evidence and is left out of retrieval scoring.
    assert report.metrics["bm25"]["queries"] == 2
    assert report.metrics["bm25"]["hits@10"] == 1.0


def test_a_drop_below_a_threshold_fails_with_a_readable_message(bm25_variant: PipelineVariant):
    report = retrieval_gate(
        [bm25_variant], QUERIES, {"bm25": {"hits@10": 1.0, "evidence_recall@10": 1.01}}
    )

    assert not report.passed
    assert report.failures == ("bm25 evidence_recall@10 = 1.000 is below the threshold 1.010",)


def test_a_threshold_for_a_missing_variant_or_metric_fails(bm25_variant: PipelineVariant):
    report = retrieval_gate(
        [bm25_variant], QUERIES, {"bm25": {"ndcg@10": 0.5}, "hybrid": {"hits@10": 0.9}}
    )

    assert report.failures == (
        "bm25 ndcg@10 has no value (threshold 0.500)",
        "hybrid hits@10 has no value (threshold 0.900)",
    )
