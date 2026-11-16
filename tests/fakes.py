"""Deterministic stand-ins for the model roles."""

from __future__ import annotations

from rag_pipeline.models import Completion, GeneratorSpec

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
