"""Model roles behind replaceable interfaces: local (Ollama), free-tier APIs, or test fakes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

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


class EmbedderSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["ollama", "sentence-transformers"]
    model: str
    query_prefix: str = ""  # some embedders (e.g. Qwen3-Embedding) want an instruction


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class EmbedderWithUsage(Protocol):
    def embed_query_with_usage(self, text: str) -> tuple[list[float], int | None]: ...


@runtime_checkable
class QueryTokenCounter(Protocol):
    def count_query_tokens(self, text: str) -> int | None: ...


class RerankerSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["sentence-transformers"]
    model: str
    max_length: int = 512
    batch_size: int = 16


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, texts: list[str]) -> list[float]: ...


@runtime_checkable
class RerankTokenCounter(Protocol):
    def count_input_tokens(self, query: str, texts: list[str]) -> int | None: ...


class Generator(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, system: str, prompt: str) -> Completion: ...


class ModelProvider(Protocol):
    def generator_for(self, spec: GeneratorSpec) -> Generator: ...

    def reranker_for(self, spec: RerankerSpec) -> Reranker: ...


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


class OllamaEmbedder:
    def __init__(self, spec: EmbedderSpec, base_url: str = "http://localhost:11434") -> None:
        self._spec = spec
        self._http = httpx.Client(base_url=base_url, timeout=600)

    @property
    def name(self) -> str:
        return f"ollama:{self._spec.model}"

    def _embed(self, texts: list[str]) -> tuple[list[list[float]], int | None]:
        response = self._http.post(
            "/api/embed", json={"model": self._spec.model, "input": texts, "truncate": True}
        )
        response.raise_for_status()
        body = response.json()
        return body["embeddings"], body.get("prompt_eval_count")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)[0]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_query_with_usage(text)[0]

    def embed_query_with_usage(self, text: str) -> tuple[list[float], int | None]:
        vectors, input_tokens = self._embed([self._spec.query_prefix + text])
        return vectors[0], input_tokens


class SentenceTransformersEmbedder:
    """A bi-encoder run in-process; small models make the CI gate possible on a CPU runner."""

    def __init__(self, spec: EmbedderSpec) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, only when used

        self._spec = spec
        self._model = SentenceTransformer(spec.model)

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._spec.model}"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(  # pyright: ignore[reportUnknownMemberType]
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(x) for x in row] for row in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._spec.query_prefix + text])[0]

    def count_query_tokens(self, text: str) -> int | None:
        tokenizer = self._model.tokenizer
        if tokenizer is None:
            return None
        prompt = self._model.prompts.get(self._model.default_prompt_name or "") or ""
        encoded = tokenizer(
            prompt + self._spec.query_prefix + text,
            truncation=True,
            max_length=self._model.max_seq_length,
        )
        return len(encoded["input_ids"])


class SentenceTransformersReranker:
    """A cross-encoder run locally (Apple GPU via MPS when available)."""

    def __init__(self, spec: RerankerSpec) -> None:
        from sentence_transformers import CrossEncoder  # heavy import, only when used

        self._spec = spec
        self._model = CrossEncoder(spec.model, max_length=spec.max_length)

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._spec.model}"

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._model.predict(  # pyright: ignore[reportUnknownMemberType]
            [(query, t) for t in texts], batch_size=self._spec.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]

    def count_input_tokens(self, query: str, texts: list[str]) -> int | None:
        if not texts:
            return 0
        tokenizer = self._model.tokenizer
        if tokenizer is None:
            return None
        encoded = tokenizer(
            [query] * len(texts), texts, truncation=True, max_length=self._spec.max_length
        )
        pairs: list[list[int]] = encoded["input_ids"]
        return sum(map(len, pairs))


class LocalModels:
    """The runtime model provider: everything served locally (ADR 0001)."""

    def __init__(self, ollama_url: str = "http://localhost:11434") -> None:
        self._ollama_url = ollama_url
        self._rerankers: dict[RerankerSpec, SentenceTransformersReranker] = {}
        self._embedders: dict[EmbedderSpec, Embedder] = {}

    def reranker_for(self, spec: RerankerSpec) -> SentenceTransformersReranker:
        if spec not in self._rerankers:
            self._rerankers[spec] = SentenceTransformersReranker(spec)
        return self._rerankers[spec]

    def generator_for(self, spec: GeneratorSpec) -> OllamaGenerator:
        return OllamaGenerator(spec, self._ollama_url)

    def embedder_for(self, spec: EmbedderSpec) -> Embedder:
        if spec not in self._embedders:
            self._embedders[spec] = (
                SentenceTransformersEmbedder(spec)
                if spec.provider == "sentence-transformers"
                else OllamaEmbedder(spec, self._ollama_url)
            )
        return self._embedders[spec]
