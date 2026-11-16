"""The Chunk index in Qdrant (ADR 0003)."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any

from qdrant_client import QdrantClient, models

from rag_pipeline import bm25
from rag_pipeline.cache import ModelResponse, ResponseCache
from rag_pipeline.chunk_context import ContextBuild
from rag_pipeline.chunking import Chunk, ChunkingSettings
from rag_pipeline.corpus import article_hash
from rag_pipeline.models import (
    Embedder,
    EmbedderSpec,
    EmbedderWithUsage,
    QueryTokenCounter,
    query_embedding,
)

type TokenCount = int | Callable[[], int | None] | None

BM25_VECTOR = "bm25"
DENSE_VECTOR = "dense"
EMBED_BATCH = 64
_POINT_NAMESPACE = uuid.UUID("6f1c4a52-1b7e-4f3e-9a51-3f4e2b7c9d10")
_NO_EMBEDDER = "none"


class IndexMismatch(Exception):
    """Raised when Chunks built with other settings or another embedder would be mixed in."""


def index_collection(
    settings: ChunkingSettings, embedder_model: str | None, *, context_model: str | None = None
) -> str:
    """The collection for one chunking, Chunk Context model and embedder combination.

    Different combinations live side by side; the Corpus hash is not part of the name, so
    re-ingesting a changed Corpus updates the same collection incrementally.
    """

    def slug(name: str) -> str:
        model = name.rsplit("/", 1)[-1].lower()
        return re.sub(r"[^a-z0-9.]+", "-", model).strip("-")

    name = f"chunks-{settings.label}"
    if context_model:
        name += f"-ctx-{slug(context_model)}"
    if embedder_model:
        name += f"-{slug(embedder_model)}"
    return name


def corpus_hash(article_hashes: dict[str, str]) -> str:
    lines = "\n".join(f"{a}:{h}" for a, h in sorted(article_hashes.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    article_id: str
    title: str
    source: str
    published_at: str
    text: str
    score: float
    context: str | None = None


class ChunkIndex:
    """Chunks with a BM25 sparse vector and, when an embedder is given, a dense vector."""

    def __init__(
        self, client: QdrantClient, collection: str, embedder: Embedder | None = None
    ) -> None:
        self._client = client
        self.collection = collection
        self._embedder = embedder

    def _ensure_collection(self, dense_size: int | None) -> None:
        if self._client.collection_exists(self.collection):
            return
        dense = (
            {DENSE_VECTOR: models.VectorParams(size=dense_size, distance=models.Distance.COSINE)}
            if dense_size
            else {}
        )
        self._client.create_collection(
            self.collection,
            vectors_config=dense,
            sparse_vectors_config={
                BM25_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    def _dense_vectors(self, chunks: list[Chunk]) -> list[list[float]] | None:
        if self._embedder is None:
            return None
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            vectors += self._embedder.embed_documents([c.index_text for c in batch])
        return vectors

    @property
    def _embedder_name(self) -> str:
        return self._embedder.name if self._embedder else _NO_EMBEDDER

    def _stored_payloads(self) -> list[dict[str, Any]]:
        if not self._client.collection_exists(self.collection):
            return []
        payloads: list[dict[str, Any]] = []
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                self.collection,
                limit=2048,
                offset=offset,
                with_payload=[
                    "article_id",
                    "article_hash",
                    "chunking",
                    "embedder",
                    "context_model",
                    "context_build",
                    "article_chunk_count",
                ],
                with_vectors=False,
            )
            payloads += [p.payload or {} for p in points]
            if offset is None:
                return payloads

    def stored_articles(self) -> dict[str, str | None]:
        """Article hashes, with None for legacy or incomplete sets of Chunks."""
        payloads = self._stored_payloads()
        counts: Counter[str] = Counter(
            p["article_id"] for p in payloads if "article_chunk_count" in p
        )
        stored: dict[str, str | None] = {}
        for payload in payloads:
            article_id: str = payload["article_id"]
            expected = payload.get("article_chunk_count")
            # An interrupted batch must not make a partly indexed Article look unchanged.
            stored[article_id] = (
                None
                if expected is not None and counts[article_id] != expected
                else payload.get("article_hash")
            )
        return stored

    def check_build(self, chunking: str, context: ContextBuild | None = None) -> None:
        """Refuse to mix incompatible chunking, Chunk Context or embedding settings."""
        for payload in self._stored_payloads():
            for key, expected in (("chunking", chunking), ("embedder", self._embedder_name)):
                stored = payload.get(key)
                if stored is not None and stored != expected:
                    raise IndexMismatch(
                        f"{self.collection!r} holds Chunks with {key} {stored!r}, not "
                        f"{expected!r}; ingest into its own collection instead"
                    )
            for key, expected in (
                ("context_model", context.model if context else None),
                ("context_build", context.key if context else None),
            ):
                stored = payload.get(key)
                if stored != expected:
                    raise IndexMismatch(
                        f"{self.collection!r} holds Chunks with {key} {stored!r}, not "
                        f"{expected!r}; ingest into its own collection instead"
                    )

    def delete_articles(self, article_ids: Iterable[str]) -> None:
        ids = list(article_ids)
        if not ids or not self._client.collection_exists(self.collection):
            return
        self._client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="article_id", match=models.MatchAny(any=ids))]
                )
            ),
        )

    def version(self) -> str:
        """Collection plus Corpus hash, e.g. chunks-sent-w200-o1-bge-m3@3f9a1c0b2d4e."""
        stored = self.stored_articles()
        return f"{self.collection}@{corpus_hash({a: h or '' for a, h in stored.items()})}"

    def upsert(self, chunks: Iterable[Chunk], *, context: ContextBuild | None = None) -> int:
        chunks = list(chunks)
        if not chunks:
            return 0
        if any((chunk.context is None) != (context is None) for chunk in chunks):
            raise IndexMismatch("Chunk Context payloads require matching build metadata")
        dense = self._dense_vectors(chunks)
        counts = Counter(chunk.article.id for chunk in chunks) if context else Counter[str]()
        self._ensure_collection(len(dense[0]) if dense else None)
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_POINT_NAMESPACE, chunk.id)),
                vector={
                    BM25_VECTOR: bm25.document_vector(chunk.index_text),
                    **({DENSE_VECTOR: dense[n]} if dense else {}),
                },
                payload={
                    "chunk_id": chunk.id,
                    "article_id": chunk.article.id,
                    "position": chunk.position,
                    "title": chunk.article.title,
                    "source": chunk.article.source,
                    "published_at": chunk.article.published_at,
                    "text": chunk.text,
                    "article_hash": article_hash(chunk.article),
                    "chunking": chunk.chunking,
                    "embedder": self._embedder_name,
                    **(
                        {
                            "context": chunk.context,
                            "context_model": context.model,
                            "context_build": context.key,
                            "article_chunk_count": counts[chunk.article.id],
                        }
                        if context
                        else {}
                    ),
                },
            )
            for n, chunk in enumerate(chunks)
        ]
        for start in range(0, len(points), 256):
            self._client.upsert(self.collection, points[start : start + 256])
        return len(points)

    def count(self) -> int:
        if not self._client.collection_exists(self.collection):
            return 0
        return self._client.count(self.collection, exact=True).count

    def search_bm25(self, query: str, limit: int) -> list[RetrievedChunk]:
        response = self._client.query_points(
            self.collection,
            query=bm25.query_vector(query),
            using=BM25_VECTOR,
            limit=limit,
            with_payload=True,
        )
        return [_to_retrieved(p.payload or {}, p.score) for p in response.points]

    def _query_embedding(
        self, query: str, on_input_tokens: Callable[[TokenCount], None] | None
    ) -> list[float]:
        if self._embedder is None:
            raise ValueError(f"index {self.collection!r} has no embedder for dense search")
        if on_input_tokens is not None:
            if isinstance(self._embedder, EmbedderWithUsage):
                vector, input_tokens = self._embedder.embed_query_with_usage(query)
                on_input_tokens(input_tokens)
                return vector
            counter = self._embedder
            on_input_tokens(
                partial(counter.count_query_tokens, query)
                if isinstance(counter, QueryTokenCounter)
                else None
            )
        return self._embedder.embed_query(query)

    def cached_embedding(
        self, query: str, spec: EmbedderSpec, cache: ResponseCache
    ) -> ModelResponse[tuple[list[float], int | None]]:
        embedder = self._embedder
        if embedder is None:
            raise ValueError(f"index {self.collection!r} has no embedder for dense search")
        return query_embedding(embedder, spec, query, cache)

    def search_hybrid(
        self,
        query: str,
        limit: int,
        candidates: int,
        *,
        on_input_tokens: Callable[[TokenCount], None] | None = None,
        embedding: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """BM25 and dense candidates fused with Reciprocal Rank Fusion inside one Qdrant query."""
        response = self._client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(
                    query=bm25.query_vector(query), using=BM25_VECTOR, limit=candidates
                ),
                models.Prefetch(
                    query=embedding
                    if embedding is not None
                    else self._query_embedding(query, on_input_tokens),
                    using=DENSE_VECTOR,
                    limit=candidates,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [_to_retrieved(p.payload or {}, p.score) for p in response.points]

    def search_dense(
        self,
        query: str,
        limit: int,
        *,
        on_input_tokens: Callable[[TokenCount], None] | None = None,
        embedding: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        response = self._client.query_points(
            self.collection,
            query=embedding
            if embedding is not None
            else self._query_embedding(query, on_input_tokens),
            using=DENSE_VECTOR,
            limit=limit,
            with_payload=True,
        )
        return [_to_retrieved(p.payload or {}, p.score) for p in response.points]


def _to_retrieved(payload: dict[str, Any], score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=payload["chunk_id"],
        article_id=payload["article_id"],
        title=payload["title"],
        source=payload["source"],
        published_at=payload["published_at"],
        text=payload["text"],
        score=score,
        context=payload.get("context"),
    )
