from conftest import ARTICLES, FTX, PIXEL

from rag_pipeline.index import ChunkIndex
from rag_pipeline.ingest import ChunkingSettings, ingest

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
