"""The Chunk index in Qdrant (ADR 0003)."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from rag_pipeline import bm25
from rag_pipeline.chunking import Chunk
from rag_pipeline.models import Embedder

BM25_VECTOR = "bm25"
DENSE_VECTOR = "dense"
EMBED_BATCH = 64
_POINT_NAMESPACE = uuid.UUID("6f1c4a52-1b7e-4f3e-9a51-3f4e2b7c9d10")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    article_id: str
    title: str
    source: str
    published_at: str
    text: str
    score: float


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
            vectors += self._embedder.embed_documents([c.text for c in batch])
        return vectors

    def upsert(self, chunks: Iterable[Chunk]) -> int:
        chunks = list(chunks)
        dense = self._dense_vectors(chunks)
        self._ensure_collection(len(dense[0]) if dense else None)
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_POINT_NAMESPACE, chunk.id)),
                vector={
                    BM25_VECTOR: bm25.document_vector(chunk.text),
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

    def _query_embedding(self) -> Embedder:
        if self._embedder is None:
            raise ValueError(f"index {self.collection!r} has no embedder for dense search")
        return self._embedder

    def search_hybrid(self, query: str, limit: int, candidates: int) -> list[RetrievedChunk]:
        """BM25 and dense candidates fused with Reciprocal Rank Fusion inside one Qdrant query."""
        response = self._client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(
                    query=bm25.query_vector(query), using=BM25_VECTOR, limit=candidates
                ),
                models.Prefetch(
                    query=self._query_embedding().embed_query(query),
                    using=DENSE_VECTOR,
                    limit=candidates,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [_to_retrieved(p.payload or {}, p.score) for p in response.points]

    def search_dense(self, query: str, limit: int) -> list[RetrievedChunk]:
        response = self._client.query_points(
            self.collection,
            query=self._query_embedding().embed_query(query),
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
    )
