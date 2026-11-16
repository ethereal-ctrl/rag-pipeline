"""The Ingest seam: Articles in, an indexed set of Chunks out, updated incrementally."""

from __future__ import annotations

from dataclasses import dataclass, replace

from rag_pipeline.cache import ModelResponse
from rag_pipeline.chunk_context import Contextualiser
from rag_pipeline.chunking import Chunk, ChunkingSettings, chunk_article
from rag_pipeline.corpus import Article, article_hash
from rag_pipeline.cost import load_prices
from rag_pipeline.index import ChunkIndex
from rag_pipeline.models import Completion

__all__ = ["ChunkingSettings", "IngestReport", "ingest"]


@dataclass(frozen=True)
class ChunkContextReport:
    model: str
    chunks_contextualised: int
    cache_hits: int
    generation_seconds: float
    input_tokens: int
    output_tokens: int
    truncated_articles: tuple[str, ...]
    notional_cost_usd: float | None
    empty_contexts: int = 0


@dataclass(frozen=True)
class IngestReport:
    articles: int
    chunks_written: int
    added: tuple[str, ...]
    changed: tuple[str, ...]
    unchanged: tuple[str, ...]
    deleted: tuple[str, ...]
    index_version: str
    context: ChunkContextReport | None = None


def _contextualise(
    articles: list[Article], settings: ChunkingSettings, contextualiser: Contextualiser
) -> tuple[list[Chunk], ChunkContextReport]:
    chunks: list[Chunk] = []
    responses: list[ModelResponse[Completion]] = []
    truncated: list[str] = []
    for article in articles:
        original = chunk_article(article, settings)
        contexts = contextualiser.contextualise(article, original)
        if contexts.article_truncated:
            truncated.append(article.id)
        chunks.extend(
            replace(chunk, context=response.value.text.strip())
            for chunk, response in zip(original, contexts.responses, strict=True)
        )
        responses.extend(contexts.responses)
    input_tokens = sum(item.value.input_tokens for item in responses)
    output_tokens = sum(item.value.output_tokens for item in responses)
    model = contextualiser.build.model
    price = load_prices().get(model)
    cost = (
        None
        if price is None
        else (
            input_tokens * price.usd_per_million_input_tokens
            + output_tokens * price.usd_per_million_output_tokens
        )
        / 1_000_000
    )
    return chunks, ChunkContextReport(
        model=model,
        chunks_contextualised=len(chunks),
        cache_hits=sum(item.cache_hit for item in responses),
        generation_seconds=sum(item.seconds for item in responses),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        truncated_articles=tuple(truncated),
        notional_cost_usd=cost,
        empty_contexts=sum(not chunk.context for chunk in chunks),
    )


def ingest(
    articles: list[Article],
    index: ChunkIndex,
    settings: ChunkingSettings,
    *,
    contextualiser: Contextualiser | None = None,
) -> IngestReport:
    """Bring the index in line with `articles`, re-chunking only what changed.

    Unchanged Articles (same content hash) are skipped; changed and deleted ones lose their
    old Chunks first, so no stale Chunk survives a shorter rewrite.
    """
    context_build = contextualiser.build if contextualiser is not None else None
    index.check_build(settings.label, context_build)
    stored = index.stored_articles()
    current = {a.id: a for a in articles}
    added = tuple(i for i in current if i not in stored)
    changed = tuple(i for i in current if i in stored and stored[i] != article_hash(current[i]))
    unchanged = tuple(i for i in current if i in stored and i not in changed)
    deleted = tuple(sorted(i for i in stored if i not in current))
    context_report = None
    if contextualiser is None:
        index.delete_articles(changed + deleted)
        chunks = [c for i in added + changed for c in chunk_article(current[i], settings)]
    else:
        # Cache each response before changing the index, so a killed run can reuse it.
        chunks, context_report = _contextualise(
            [current[i] for i in added + changed], settings, contextualiser
        )
        # Context generation can take hours. Recheck the collection before writing to it.
        index.check_build(settings.label, context_build)
        index.delete_articles(changed + deleted)
    written = index.upsert(chunks, context=context_build)
    return IngestReport(
        articles=len(articles),
        chunks_written=written,
        added=added,
        changed=changed,
        unchanged=unchanged,
        deleted=deleted,
        index_version=index.version(),
        context=context_report,
    )
