"""Splitting Articles into Chunks along sentence boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_pipeline.corpus import Article

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkingSettings:
    max_words: int = 200
    overlap_sentences: int = 1

    @property
    def label(self) -> str:
        return f"sent-w{self.max_words}-o{self.overlap_sentences}"


@dataclass(frozen=True)
class Chunk:
    id: str
    article: Article
    position: int
    text: str
    chunking: str = ""  # label of the settings that produced it


def _sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENTENCE_END.split(text)) if s]


def chunk_article(article: Article, settings: ChunkingSettings) -> list[Chunk]:
    """Pack whole sentences into Chunks of at most `max_words` words.

    A single sentence longer than the limit becomes its own Chunk rather than being cut,
    so Evidence facts (which are sentences) are never split across Chunks.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    for sentence in _sentences(article.body):
        words = sum(len(s.split()) for s in current) + len(sentence.split())
        if current and words > settings.max_words:
            groups.append(current)
            current = current[-settings.overlap_sentences :] if settings.overlap_sentences else []
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
