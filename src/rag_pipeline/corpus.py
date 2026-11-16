"""The MultiHop-RAG Corpus and Benchmark Queries, pinned to an exact upstream revision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DATASET_REPO = "yixuantt/MultiHopRAG"
DATASET_REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
DATASET_LICENSE = "ODC-BY 1.0 (Tang & Yang, 2024, arXiv:2401.15391)"
DATASET_FILES = {
    "corpus.json": "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
    "MultiHopRAG.json": "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
}

NULL_QUERY_TYPE = "null_query"


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    body: str
    source: str
    published_at: str
    category: str
    url: str


@dataclass(frozen=True)
class Evidence:
    article_url: str
    fact: str


@dataclass(frozen=True)
class BenchmarkQuery:
    id: str
    question: str
    answer: str
    query_type: str
    evidence: tuple[Evidence, ...]

    @property
    def is_null(self) -> bool:
        return self.query_type == NULL_QUERY_TYPE


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def article_id_for(url: str) -> str:
    return "a_" + _short_hash(url)


def query_id_for(question: str) -> str:
    return "q_" + _short_hash(question)


def fetch_dataset(raw_dir: Path) -> None:
    """Download the pinned dataset files and verify their checksums."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, sha256 in DATASET_FILES.items():
        target = raw_dir / name
        if not target.exists() or _sha256(target) != sha256:
            url = (
                f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{name}"
            )
            response = httpx.get(url, follow_redirects=True, timeout=120)
            response.raise_for_status()
            target.write_bytes(response.content)
        actual = _sha256(target)
        if actual != sha256:
            raise ValueError(f"{name}: checksum {actual} does not match pinned {sha256}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_articles(raw_dir: Path) -> list[Article]:
    rows: list[dict[str, Any]] = json.loads((raw_dir / "corpus.json").read_text("utf-8"))
    return [
        Article(
            id=article_id_for(row["url"]),
            title=row["title"],
            body=row["body"],
            source=row["source"],
            published_at=row["published_at"],
            category=row["category"],
            url=row["url"],
        )
        for row in rows
    ]


def load_queries(raw_dir: Path) -> list[BenchmarkQuery]:
    rows: list[dict[str, Any]] = json.loads((raw_dir / "MultiHopRAG.json").read_text("utf-8"))
    return [
        BenchmarkQuery(
            id=query_id_for(row["query"]),
            question=row["query"],
            answer=row["answer"],
            query_type=row["question_type"],
            evidence=tuple(Evidence(e["url"], e["fact"]) for e in row["evidence_list"]),
        )
        for row in rows
    ]
