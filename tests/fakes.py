"""Deterministic stand-ins for the model roles."""

from __future__ import annotations

from typing import Literal

from rag_pipeline.index import RetrievedChunk
from rag_pipeline.models import Completion, GeneratorSpec
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
    """A model provider that hands out the same fake generator for every spec."""

    def __init__(self, generator: GroundedFakeGenerator) -> None:
        self.generator = generator
        self.requested: list[GeneratorSpec] = []

    def generator_for(self, spec: GeneratorSpec) -> GroundedFakeGenerator:
        self.requested.append(spec)
        return self.generator


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
        self.asked: list[str] = []

    def answer(self, question: str) -> Answer:
        if self._crash_after is not None and len(self.asked) >= self._crash_after:
            raise RuntimeError("simulated crash")
        self.asked.append(question)
        text, chunk_texts = self._script[question]
        if not self.config.uses_retrieval:
            chunk_texts = []
        retrieved = tuple(
            RetrievedChunk(f"c{i}", "a", "t", "s", "2023-10-01", body, 1.0 / (i + 1))
            for i, body in enumerate(chunk_texts)
        )
        return Answer(text, is_abstention(text), retrieved, 10, 2, 0.5)
