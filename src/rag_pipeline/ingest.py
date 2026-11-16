"""The Ingest seam: Articles in, an indexed set of Chunks out."""

from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.chunking import ChunkingSettings, chunk_article
from rag_pipeline.corpus import Article
from rag_pipeline.index import ChunkIndex

__all__ = ["ChunkingSettings", "IngestReport", "ingest"]


@dataclass(frozen=True)
class IngestReport:
    articles: int
    chunks_written: int


def ingest(articles: list[Article], index: ChunkIndex, settings: ChunkingSettings) -> IngestReport:
    chunks = [chunk for article in articles for chunk in chunk_article(article, settings)]
    written = index.upsert(chunks)
    return IngestReport(articles=len(articles), chunks_written=written)
