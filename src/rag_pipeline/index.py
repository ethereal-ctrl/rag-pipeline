"""The Chunk index in Qdrant (ADR 0003)."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from rag_pipeline import bm25
from rag_pipeline.chunking import Chunk

BM25_VECTOR = "bm25"
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
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self.collection = collection

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(self.collection):
            return
        self._client.create_collection(
            self.collection,
            vectors_config={},
            sparse_vectors_config={
                BM25_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    def upsert(self, chunks: Iterable[Chunk]) -> int:
        self._ensure_collection()
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_POINT_NAMESPACE, chunk.id)),
                vector={BM25_VECTOR: bm25.document_vector(chunk.text)},
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
            for chunk in chunks
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
