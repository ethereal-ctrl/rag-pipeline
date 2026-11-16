"""Deterministic stand-ins for the model roles."""

from __future__ import annotations

import hashlib
import math
from typing import ClassVar, Literal

from rag_pipeline.bm25 import tokenize
from rag_pipeline.index import RetrievedChunk
from rag_pipeline.models import Completion, GeneratorSpec, RerankerSpec
from rag_pipeline.variants import (
    Answer,
    GenerationConfig,
    IndexConfig,
    RetrievalConfig,
    VariantConfig,
    is_abstention,
)

ABSTAIN = "Insufficient information."


class GroundedFakeGenerator:
    """Answers only when a known supporting phrase appears in the prompt, else abstains.

    This mimics a well-behaved grounded generator: retrieval quality decides the answer.
    """

    def __init__(self, answers_by_phrase: dict[str, str]) -> None:
        self._answers = answers_by_phrase
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return "fake-grounded"

    def complete(self, system: str, prompt: str) -> Completion:
        self.prompts.append(prompt)
        for phrase, answer in self._answers.items():
            if phrase in prompt:
                return Completion(text=answer, input_tokens=len(prompt.split()), output_tokens=3)
        return Completion(text=ABSTAIN, input_tokens=len(prompt.split()), output_tokens=2)


class FakeModels:
    """A model provider that hands out the same fakes for every spec."""

    def __init__(
        self, generator: GroundedFakeGenerator, reranker: PhraseFakeReranker | None = None
    ) -> None:
        self.generator = generator
        self.reranker = reranker
        self.requested: list[GeneratorSpec] = []

    def generator_for(self, spec: GeneratorSpec) -> GroundedFakeGenerator:
        self.requested.append(spec)
        return self.generator

    def reranker_for(self, spec: RerankerSpec) -> PhraseFakeReranker:
        if self.reranker is None:
            raise AssertionError("this test did not expect a reranker")
        return self.reranker


class ScriptedVariant:
    """A Pipeline Variant whose answer and retrieved Chunk texts are fixed per question.

    Lets Evaluate-seam tests pin exact rankings and hand-compute the expected metrics.
    """

    def __init__(
        self,
        name: str,
        script: dict[str, tuple[str, list[str]]],
        config_tag: str = "v1",
        crash_after: int | None = None,
        retrieval_mode: Literal["none", "bm25"] = "bm25",
        latencies: dict[str, float] | None = None,
    ) -> None:
        self.config = VariantConfig(
            name=name,
            index=None if retrieval_mode == "none" else IndexConfig(collection=f"s-{config_tag}"),
            retrieval=RetrievalConfig(mode=retrieval_mode, top_k=10),
            generation=GenerationConfig(generator=GeneratorSpec(provider="ollama", model="m")),
        )
        self.generator = GroundedFakeGenerator({})
        self._script = script
        self._crash_after = crash_after
        self._latencies = latencies or {}
        self.asked: list[str] = []

    def index_version(self) -> str | None:
        return None if self.config.index is None else f"{self.config.index.collection}@fixed"

    def retrieve(
        self, question: str, stages: dict[str, float] | None = None
    ) -> tuple[RetrievedChunk, ...]:
        _, chunk_texts = self._script[question]
        if not self.config.uses_retrieval:
            return ()
        return tuple(
            RetrievedChunk(f"c{i}", "a", "t", "s", "2023-10-01", body, 1.0 / (i + 1))
            for i, body in enumerate(chunk_texts)
        )

    def answer(self, question: str) -> Answer:
        if self._crash_after is not None and len(self.asked) >= self._crash_after:
            raise RuntimeError("simulated crash")
        self.asked.append(question)
        text, _ = self._script[question]
        seconds = self._latencies.get(question, 0.5)
        return Answer(text, is_abstention(text), self.retrieve(question), 10, 2, seconds)


class ConceptFakeEmbedder:
    """Bag-of-concepts embeddings: a few synonyms share a concept, so dense search can find
    Chunks that share no words with the question, which BM25 cannot."""

    SYNONYMS: ClassVar[dict[str, str]] = {
        "smartphone": "phone",
        "handset": "phone",
        "pictures": "photos",
        "dark": "light",
        "crypto": "ftx",
    }
    dimensions = 64

    @property
    def name(self) -> str:
        return "fake-concepts"

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            concept = self.SYNONYMS.get(token, token)
            bucket = int(hashlib.md5(concept.encode()).hexdigest(), 16) % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def count_query_tokens(self, text: str) -> int:
        return len(text.split())


class PhraseFakeReranker:
    """Scores a Chunk 1.0 if it contains the preferred phrase, else 0.0."""

    def __init__(self, preferred_phrase: str) -> None:
        self._phrase = preferred_phrase
        self.calls: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return "fake-phrase-reranker"

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, len(texts)))
        return [1.0 if self._phrase in t else 0.0 for t in texts]

    def count_input_tokens(self, query: str, texts: list[str]) -> int:
        query_tokens = len(query.split())
        return sum(query_tokens + len(text.split()) for text in texts)
