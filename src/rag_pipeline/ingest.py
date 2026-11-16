"""The Ingest seam: Articles in, an indexed set of Chunks out, updated incrementally."""

from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.chunking import ChunkingSettings, chunk_article
from rag_pipeline.corpus import Article, article_hash
from rag_pipeline.index import ChunkIndex

__all__ = ["ChunkingSettings", "IngestReport", "ingest"]


@dataclass(frozen=True)
class IngestReport:
    articles: int
    chunks_written: int
    added: tuple[str, ...]
    changed: tuple[str, ...]
    unchanged: tuple[str, ...]
    deleted: tuple[str, ...]
    index_version: str


def ingest(articles: list[Article], index: ChunkIndex, settings: ChunkingSettings) -> IngestReport:
    """Bring the index in line with `articles`, re-chunking only what changed.

    Unchanged Articles (same content hash) are skipped; changed and deleted ones lose their
    old Chunks first, so no stale Chunk survives a shorter rewrite.
    """
    index.check_build(settings.label)
    stored = index.stored_articles()
    current = {a.id: a for a in articles}
    added = tuple(i for i in current if i not in stored)
    changed = tuple(i for i in current if i in stored and stored[i] != article_hash(current[i]))
    unchanged = tuple(i for i in current if i in stored and i not in changed)
    deleted = tuple(sorted(i for i in stored if i not in current))
    index.delete_articles(changed + deleted)
    chunks = [c for i in added + changed for c in chunk_article(current[i], settings)]
    written = index.upsert(chunks)
    return IngestReport(
        articles=len(articles),
        chunks_written=written,
        added=added,
        changed=changed,
        unchanged=unchanged,
        deleted=deleted,
        index_version=index.version(),
    )
