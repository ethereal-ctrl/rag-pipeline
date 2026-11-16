from dataclasses import replace

import pytest
from conftest import ARTICLES, FTX, LEAGUE, PIXEL
from fakes import ConceptFakeEmbedder, FakeModels, GroundedFakeGenerator
from qdrant_client import QdrantClient

from rag_pipeline.index import ChunkIndex, IndexMismatch, index_collection
from rag_pipeline.ingest import ChunkingSettings, ingest
from rag_pipeline.models import GeneratorSpec
from rag_pipeline.variants import (
    GenerationConfig,
    IndexConfig,
    RetrievalConfig,
    VariantConfig,
    build_variant,
)

SMALL_CHUNKS = ChunkingSettings(max_words=20, overlap_sentences=0)


def test_ingested_article_is_found_by_its_keywords(index: ChunkIndex):
    ingest(ARTICLES, index, SMALL_CHUNKS)

    hits = index.search_bm25("Bankman-Fried fraud trial", limit=1)

    assert hits[0].article_id == FTX.id
    assert "Sam Bankman-Fried" in hits[0].text


def test_reingesting_the_same_articles_keeps_chunk_ids_and_does_not_duplicate(index: ChunkIndex):
    first = ingest(ARTICLES, index, SMALL_CHUNKS)
    ids_before = {h.chunk_id for h in index.search_bm25("phone camera derby trial", limit=50)}

    ingest(ARTICLES, index, SMALL_CHUNKS)

    assert index.count() == first.chunks_written
    assert {
        h.chunk_id for h in index.search_bm25("phone camera derby trial", limit=50)
    } == ids_before


def test_chunks_stay_within_the_word_limit_and_keep_sentences_whole(index: ChunkIndex):
    ingest([PIXEL], index, SMALL_CHUNKS)

    chunks = index.search_bm25("Pixel phone software camera Google", limit=50)

    assert len(chunks) == 2
    assert all(len(c.text.split()) <= 20 for c in chunks)
    assert any(
        "The new phone ships with seven years of software updates." in c.text for c in chunks
    )


# --- Incremental ingestion and index versions ---------------------------------------------

PIXEL_EDITED = replace(PIXEL, body="Google announced the Pixel 8 phone at an event in October.")


def test_reingesting_unchanged_articles_skips_them(index: ChunkIndex):
    ingest(ARTICLES, index, SMALL_CHUNKS)

    report = ingest(ARTICLES, index, SMALL_CHUNKS)

    assert set(report.unchanged) == {a.id for a in ARTICLES}
    assert (report.added, report.changed, report.deleted) == ((), (), ())
    assert report.chunks_written == 0


def test_a_changed_article_replaces_its_old_chunks(index: ChunkIndex):
    ingest(ARTICLES, index, SMALL_CHUNKS)

    report = ingest([FTX, PIXEL_EDITED, LEAGUE], index, SMALL_CHUNKS)

    assert report.changed == (PIXEL.id,)
    pixel_chunks = [
        c
        for c in index.search_bm25("Pixel phone software camera", limit=50)
        if c.article_id == PIXEL.id
    ]
    assert [c.text for c in pixel_chunks] == [PIXEL_EDITED.body]


def test_a_deleted_article_leaves_no_chunks_behind(index: ChunkIndex):
    ingest(ARTICLES, index, SMALL_CHUNKS)

    report = ingest([FTX, PIXEL], index, SMALL_CHUNKS)

    assert report.deleted == (LEAGUE.id,)
    assert all(c.article_id != LEAGUE.id for c in index.search_bm25("Arsenal derby", limit=50))


def test_new_articles_are_reported_as_added(index: ChunkIndex):
    ingest([FTX], index, SMALL_CHUNKS)

    report = ingest(ARTICLES, index, SMALL_CHUNKS)

    assert set(report.added) == {PIXEL.id, LEAGUE.id}
    assert report.unchanged == (FTX.id,)


def test_index_version_names_settings_and_embedder_and_follows_the_corpus():
    client = QdrantClient(":memory:")
    small = ChunkIndex(client, index_collection(SMALL_CHUNKS, None))
    dense = ChunkIndex(
        client, index_collection(SMALL_CHUNKS, "concept-fake"), embedder=ConceptFakeEmbedder()
    )

    v_small = ingest(ARTICLES, small, SMALL_CHUNKS).index_version
    v_dense = ingest(ARTICLES, dense, SMALL_CHUNKS).index_version
    v_edited = ingest([FTX, PIXEL_EDITED, LEAGUE], small, SMALL_CHUNKS).index_version

    assert v_small.startswith("chunks-sent-w20-o0@")
    assert v_dense.startswith("chunks-sent-w20-o0-concept-fake@")
    assert v_small.split("@")[1] == v_dense.split("@")[1]  # same Corpus, same hash
    assert v_edited != v_small
    assert small.version() == v_edited


def test_ingesting_with_other_settings_into_an_existing_index_is_refused(index: ChunkIndex):
    ingest(ARTICLES, index, SMALL_CHUNKS)

    with pytest.raises(IndexMismatch, match="sent-w20-o0"):
        ingest(ARTICLES, index, ChunkingSettings(max_words=40, overlap_sentences=0))


def test_two_index_versions_coexist_and_a_variant_config_picks_one():
    client = QdrantClient(":memory:")
    wide = ChunkingSettings(max_words=60, overlap_sentences=0)
    for settings in (SMALL_CHUNKS, wide):
        ingest(ARTICLES, ChunkIndex(client, index_collection(settings, None)), settings)

    def top_chunk(settings: ChunkingSettings) -> str:
        config = VariantConfig(
            name=settings.label,
            index=IndexConfig(collection=index_collection(settings, None)),
            retrieval=RetrievalConfig(mode="bm25", top_k=1),
            generation=GenerationConfig(generator=GeneratorSpec(provider="ollama", model="m")),
        )
        assert config.index is not None
        index = ChunkIndex(client, config.index.collection)
        variant = build_variant(config, index=index, models=FakeModels(GroundedFakeGenerator({})))
        return variant.retrieve("Pixel software updates")[0].text

    assert top_chunk(SMALL_CHUNKS) != top_chunk(wide)
    assert top_chunk(wide) == PIXEL.body  # the whole Article fits one wide Chunk
