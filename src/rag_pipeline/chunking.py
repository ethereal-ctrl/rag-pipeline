"""Splitting Articles into Chunks along sentence boundaries."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from rag_pipeline.corpus import Article

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkingSettings:
    max_words: int = 200
    overlap_sentences: int = 1
    strategy: Literal["sentences", "semantic"] = "sentences"
    semantic_percentile: int = 90

    def __post_init__(self) -> None:
        if self.max_words < 1 or self.overlap_sentences < 0:
            raise ValueError("Chunk size must be positive and overlap must be nonnegative.")
        if not 0 <= self.semantic_percentile <= 100:
            raise ValueError("The semantic percentile must be between 0 and 100.")

    @property
    def label(self) -> str:
        if self.strategy == "semantic":
            overlap = f"-o{self.overlap_sentences}" if self.overlap_sentences else ""
            return f"sem-p{self.semantic_percentile}-w{self.max_words}{overlap}"
        return f"sent-w{self.max_words}-o{self.overlap_sentences}"


@dataclass(frozen=True)
class Chunk:
    id: str
    article: Article
    position: int
    text: str
    chunking: str = ""  # label of the settings that produced it
    context: str | None = None

    @property
    def index_text(self) -> str:
        return f"{self.context}\n\n{self.text}" if self.context else self.text


def _sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENTENCE_END.split(text)) if s]


def _cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(x * y for x, y in zip(left, right, strict=True))
    left_norm, right_norm = math.hypot(*left), math.hypot(*right)
    norm_product = left_norm * right_norm
    if norm_product == 0:
        return float(left_norm != right_norm)
    return 1 - min(1.0, max(-1.0, dot / norm_product))


def _semantic_breaks(vectors: list[list[float]], percentile: int) -> set[int]:
    distances = [_cosine_distance(left, right) for left, right in pairwise(vectors)]
    ordered = sorted(distances)
    position = percentile * (len(ordered) - 1) / 100
    lower, upper = math.floor(position), math.ceil(position)
    threshold = ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
    return {i for i, distance in enumerate(distances, start=1) if distance > threshold}


def chunk_article(
    article: Article,
    settings: ChunkingSettings,
    *,
    embed_sentences: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[Chunk]:
    """Pack whole sentences into Chunks of at most `max_words` words.

    A single sentence longer than the limit becomes its own Chunk rather than being cut,
    so Evidence facts (which are sentences) are never split across Chunks.
    """
    sentences = _sentences(article.body)
    breaks: set[int] = set()
    if settings.strategy == "semantic":
        if embed_sentences is None:
            raise ValueError("Semantic chunking requires the index embedder.")
        if len(sentences) > 1:
            vectors = embed_sentences(sentences)
            if len(vectors) != len(sentences):
                raise ValueError("Semantic chunking requires one embedding per sentence.")
            breaks = _semantic_breaks(vectors, settings.semantic_percentile)

    groups: list[list[str]] = []
    current: list[str] = []
    for i, sentence in enumerate(sentences):
        sentence_words = len(sentence.split())
        current_words = sum(len(s.split()) for s in current)
        if current and (i in breaks or current_words + sentence_words > settings.max_words):
            groups.append(current)
            overlap = settings.overlap_sentences if i not in breaks else 0
            current = current[-overlap:] if overlap else []
            if settings.strategy == "semantic":
                # Keep the carried whole sentences only when they fit with the new one.
                while (
                    current
                    and sum(len(s.split()) for s in current) + sentence_words > settings.max_words
                ):
                    current.pop(0)
        current.append(sentence)
    if current:
        groups.append(current)
    return [
        Chunk(
            id=f"{article.id}#{n:03d}",
            article=article,
            position=n,
            text=" ".join(group),
            chunking=settings.label,
        )
        for n, group in enumerate(groups)
    ]
