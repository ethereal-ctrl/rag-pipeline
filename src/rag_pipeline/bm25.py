"""BM25 as sparse vectors.

Term-frequency weights live on the document side; Qdrant applies IDF at query time.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

from qdrant_client import models

K1 = 1.2
B = 0.75
AVG_DOC_WORDS = 150.0  # fixed so incremental ingestion does not shift existing weights

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    (  # noqa: SIM905
        "a an and are as at be but by for from has have he her his i in is it "
        "its of on or she that the their them they this to was were which who will with you your"
    ).split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _term_id(term: str) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest(), "big")


def document_vector(text: str) -> models.SparseVector:
    terms = tokenize(text)
    length_norm = 1 - B + B * len(terms) / AVG_DOC_WORDS
    counts = Counter(terms)
    return models.SparseVector(
        indices=[_term_id(t) for t in counts],
        values=[tf * (K1 + 1) / (tf + K1 * length_norm) for tf in counts.values()],
    )


def query_vector(text: str) -> models.SparseVector:
    terms = sorted(set(tokenize(text)))
    return models.SparseVector(indices=[_term_id(t) for t in terms], values=[1.0] * len(terms))
