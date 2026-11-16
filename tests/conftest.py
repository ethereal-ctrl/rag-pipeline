"""A tiny fixture Corpus with known Evidence, shared by all seam tests."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from rag_pipeline.corpus import Article, BenchmarkQuery, Evidence, article_id_for
from rag_pipeline.index import ChunkIndex


def _article(url: str, title: str, body: str) -> Article:
    return Article(
        id=article_id_for(url),
        title=title,
        body=body,
        source="Fixture News",
        published_at="2023-10-01T00:00:00+00:00",
        category="technology",
        url=url,
    )


FTX = _article(
    "https://fixture.test/ftx",
    "The FTX trial begins",
    "The criminal trial of Sam Bankman-Fried started on Tuesday in New York. "
    "Prosecutors say the former FTX chief committed fraud for personal gain. "
    "The jury will hear from former colleagues over several weeks.",
)
PIXEL = _article(
    "https://fixture.test/pixel",
    "Google unveils the Pixel 8",
    "Google announced the Pixel 8 phone at an event in October. "
    "The new phone ships with seven years of software updates. "
    "Its camera uses a new sensor for low light photos.",
)
LEAGUE = _article(
    "https://fixture.test/league",
    "Arsenal win the derby",
    "Arsenal beat Tottenham two goals to one in the north London derby. "
    "The winning goal came in the final minute from a corner kick.",
)
ARTICLES = [FTX, PIXEL, LEAGUE]

FTX_QUERY = BenchmarkQuery(
    id="q_ftx",
    question="Who is on trial for fraud after leading the crypto exchange FTX?",
    answer="Sam Bankman-Fried",
    query_type="inference_query",
    evidence=(Evidence(FTX.url, "The criminal trial of Sam Bankman-Fried started on Tuesday"),),
)
PIXEL_QUERY = BenchmarkQuery(
    id="q_pixel",
    question="How many years of software updates does the Pixel 8 phone get?",
    answer="seven",
    query_type="comparison_query",
    evidence=(Evidence(PIXEL.url, "The new phone ships with seven years of software updates."),),
)
NULL_QUERY = BenchmarkQuery(
    id="q_null",
    question="What did the Fixture News editor eat for breakfast on Monday?",
    answer="Insufficient information.",
    query_type="null_query",
    evidence=(),
)
QUERIES = [FTX_QUERY, PIXEL_QUERY, NULL_QUERY]


@pytest.fixture
def index() -> ChunkIndex:
    return ChunkIndex(QdrantClient(":memory:"), collection="fixture")
