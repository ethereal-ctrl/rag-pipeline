"""Model roles behind replaceable interfaces: local (Ollama), free-tier APIs, or test fakes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict


class GeneratorSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["ollama"]
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 256


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


class Generator(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, system: str, prompt: str) -> Completion: ...


class ModelProvider(Protocol):
    def generator_for(self, spec: GeneratorSpec) -> Generator: ...


class OllamaGenerator:
    def __init__(self, spec: GeneratorSpec, base_url: str = "http://localhost:11434") -> None:
        self._spec = spec
        self._http = httpx.Client(base_url=base_url, timeout=600)

    @property
    def name(self) -> str:
        return f"ollama:{self._spec.model}"

    def complete(self, system: str, prompt: str) -> Completion:
        response = self._http.post(
            "/api/chat",
            json={
                "model": self._spec.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": self._spec.temperature,
                    "seed": 0,
                    "num_predict": self._spec.max_output_tokens,
                    "num_ctx": 8192,
                },
            },
        )
        response.raise_for_status()
        body = response.json()
        return Completion(
            text=body["message"]["content"].strip(),
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
        )

    def version(self) -> str:
        response = self._http.post("/api/show", json={"model": self._spec.model})
        response.raise_for_status()
        details = response.json().get("details", {})
        return (
            f"{self._spec.model} ({details.get('parameter_size', '?')}, "
            f"{details.get('quantization_level', '?')})"
        )


class LocalModels:
    """The runtime model provider: everything served locally (ADR 0001)."""

    def __init__(self, ollama_url: str = "http://localhost:11434") -> None:
        self._ollama_url = ollama_url

    def generator_for(self, spec: GeneratorSpec) -> OllamaGenerator:
        return OllamaGenerator(spec, self._ollama_url)
