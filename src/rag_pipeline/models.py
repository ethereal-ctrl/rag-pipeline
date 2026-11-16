"""Model roles behind replaceable interfaces: local (Ollama), free-tier APIs, or test fakes."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    model_serializer,
    model_validator,
)

from rag_pipeline.cache import CACHE_FILE, ModelResponse, ResponseCache, cache_key, measure_call
from rag_pipeline.rate_limit import RateLimiter, RemoteProvider, remote_limits

if TYPE_CHECKING:
    from rag_pipeline.judge_model import Judge

DEFAULT_OLLAMA_URL = "http://localhost:11434"


def configured_ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)


class GeneratorSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["ollama", "groq", "mistral"]
    model: str
    temperature: float = 0.0
    max_output_tokens: PositiveInt = 256
    seed: int = 0
    context_size: PositiveInt = 8192
    reasoning_effort: Literal["low", "medium", "high"] = "low"

    @model_validator(mode="after")
    def _validate_provider_options(self) -> Self:
        if self.provider == "mistral" and "reasoning_effort" in self.model_fields_set:
            raise ValueError("reasoning_effort is a Groq option in GeneratorSpec")
        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        payload: dict[str, Any] = handler(self)
        if self.provider == "mistral":
            payload.pop("reasoning_effort", None)
        return payload

    @property
    def remote_provider(self) -> RemoteProvider | None:
        return self.provider if self.provider != "ollama" else None


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    call_seconds: float | None = None


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


@runtime_checkable
class StructuredGenerator(Protocol):
    def complete_structured(
        self, system: str, prompt: str, schema: dict[str, Any]
    ) -> Completion: ...


class ModelProvider(Protocol):
    def generator_for(self, spec: GeneratorSpec) -> Generator: ...

    def reranker_for(self, spec: RerankerSpec) -> Reranker: ...


def model_version(model: Generator | Embedder | Reranker) -> str | None:
    get_version = getattr(model, "cache_version", None) or getattr(model, "version", None)
    version = get_version() if callable(get_version) else None
    return version if isinstance(version, str) else None


_QUERY_EMBEDDING = TypeAdapter(tuple[list[float], int | None])


def query_embedding(
    embedder: Embedder, spec: EmbedderSpec, query: str, cache: ResponseCache | None
) -> ModelResponse[tuple[list[float], int | None]]:
    def embed() -> ModelResponse[tuple[list[float], int | None]]:
        if isinstance(embedder, EmbedderWithUsage):
            return measure_call(partial(embedder.embed_query_with_usage, query))
        response = measure_call(lambda: embedder.embed_query(query))
        tokens = (
            embedder.count_query_tokens(query) if isinstance(embedder, QueryTokenCounter) else None
        )
        return ModelResponse((response.value, tokens), response.seconds)

    if cache is None:
        return embed()
    key = cache_key("embed_query", spec.model_dump(), embedder.name, model_version(embedder), query)
    return cache.get_or_call(key, _QUERY_EMBEDDING, embed)


def _ollama_digest(client: httpx.Client, name: str) -> str | None:
    response = client.get("/api/tags")
    response.raise_for_status()
    tag = name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"
    return next(
        (model["digest"] for model in response.json()["models"] if model["name"] == tag),
        None,
    )


class OllamaGenerator:
    def __init__(
        self,
        spec: GeneratorSpec,
        base_url: str = DEFAULT_OLLAMA_URL,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._spec = spec
        self._http = httpx.Client(base_url=base_url, timeout=600, transport=transport)

    @property
    def name(self) -> str:
        return f"ollama:{self._spec.model}"

    def complete_structured(self, system: str, prompt: str, schema: dict[str, Any]) -> Completion:
        return self.complete(system, prompt, schema=schema)

    def complete(
        self, system: str, prompt: str, *, schema: dict[str, Any] | None = None
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self._spec.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": self._spec.temperature,
                "seed": self._spec.seed,
                "num_predict": self._spec.max_output_tokens,
                "num_ctx": self._spec.context_size,
            },
        }
        if schema is not None:
            payload["format"] = schema
        response = self._http.post("/api/chat", json=payload)
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

    @cached_property
    def _digest(self) -> str | None:
        return _ollama_digest(self._http, self._spec.model)

    def cache_version(self) -> str | None:
        return self._digest


ESTIMATED_BYTES_PER_TOKEN = 3


def estimate_tokens(
    system: str, prompt: str, max_output_tokens: int, *, conservative: bool = False
) -> int:
    size = len(system.encode("utf-8")) + len(prompt.encode("utf-8"))
    # Agentic Search reserves one token per byte and extra room for chat framing.
    input_tokens = size + 128 if conservative else math.ceil(size / ESTIMATED_BYTES_PER_TOKEN) + 32
    return input_tokens + max_output_tokens


@dataclass(frozen=True)
class _RemoteChat:
    base_url: str
    path: str
    key_variable: str
    max_tokens: str
    seed: str


_REMOTE_CHATS = {
    "groq": _RemoteChat(
        "https://api.groq.com",
        "/openai/v1/chat/completions",
        "GROQ_API_KEY",
        "max_completion_tokens",
        "seed",
    ),
    "mistral": _RemoteChat(
        "https://api.mistral.ai",
        "/v1/chat/completions",
        "MISTRAL_API_KEY",
        "max_tokens",
        "random_seed",
    ),
}


def remote_key_variable(provider: RemoteProvider) -> str:
    return _REMOTE_CHATS[provider].key_variable


def missing_remote_keys(specs: Iterable[GeneratorSpec]) -> tuple[str, ...]:
    variables = {
        remote_key_variable(provider)
        for spec in specs
        if (provider := spec.remote_provider) is not None
    }
    return tuple(sorted(variable for variable in variables if not os.getenv(variable, "").strip()))


class RemoteChatGenerator:
    def __init__(
        self,
        spec: GeneratorSpec,
        *,
        transport: httpx.BaseTransport | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        provider = spec.remote_provider
        if provider is None:
            raise ValueError("Remote chat requires a remote provider.")
        self._chat = _REMOTE_CHATS[provider]
        key = os.environ.get(self._chat.key_variable, "").strip()
        if not key:
            raise ValueError(f"set {self._chat.key_variable} to use {provider.title()}")
        self._spec = spec
        self._limiter = (
            limiter
            if limiter is not None
            else RateLimiter(CACHE_FILE, remote_limits(provider, spec.model))
        )
        # Keep Groq's existing rows. Mistral limits apply across the organization.
        self._quota_key = spec.model if provider == "groq" else "mistral"
        self._http = httpx.Client(
            base_url=self._chat.base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=60,
            transport=transport,
        )

    @property
    def name(self) -> str:
        return f"{self._spec.provider}:{self._spec.model}"

    def complete_structured(self, system: str, prompt: str, schema: dict[str, Any]) -> Completion:
        return self.complete(system, prompt, schema=schema)

    def complete(
        self, system: str, prompt: str, *, schema: dict[str, Any] | None = None
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self._spec.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._spec.temperature,
            self._chat.max_tokens: self._spec.max_output_tokens,
            self._chat.seed: self._spec.seed,
        }
        if self._spec.provider == "groq":
            payload.update(reasoning_effort=self._spec.reasoning_effort, include_reasoning=False)
        if schema is not None:
            # Mistral chat schema format: https://docs.mistral.ai/api/endpoint/chat#chat-completion
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "rag_response", "strict": True, "schema": schema},
            }
        tokens = estimate_tokens(system, prompt, self._spec.max_output_tokens)
        backoff = 1.0
        while True:
            request = self._limiter.reserve(self._quota_key, tokens)
            started = time.perf_counter()
            response = self._http.post(self._chat.path, json=payload)
            seconds = time.perf_counter() - started
            if response.status_code != 429:
                response.raise_for_status()
                body = response.json()
                usage = body["usage"]
                self._limiter.record_usage(
                    request, usage["prompt_tokens"] + usage["completion_tokens"]
                )
                content: str | list[dict[str, Any]] | None = body["choices"][0]["message"][
                    "content"
                ]
                if isinstance(content, list):
                    content = "".join(chunk["text"] for chunk in content if chunk["type"] == "text")
                return Completion(
                    (content or "").strip(),
                    usage["prompt_tokens"],
                    usage["completion_tokens"],
                    seconds,
                )
            self._limiter.record_usage(request, 0)
            try:
                retry_after = float(response.headers.get("retry-after", "0"))
            except ValueError:
                retry_after = 0.0
            self._limiter.clock.sleep(
                max(backoff, retry_after) if math.isfinite(retry_after) else backoff
            )
            backoff = min(backoff * 2, 60)


class OllamaEmbedder:
    def __init__(
        self,
        spec: EmbedderSpec,
        base_url: str = DEFAULT_OLLAMA_URL,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._spec = spec
        self._http = httpx.Client(base_url=base_url, timeout=600, transport=transport)

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

    @cached_property
    def _digest(self) -> str | None:
        return _ollama_digest(self._http, self._spec.model)

    def cache_version(self) -> str | None:
        return self._digest


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

    def cache_version(self) -> str | None:
        return _hf_revision(getattr(self._model[0], "auto_model", None))


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

    def cache_version(self) -> str | None:
        return _hf_revision(self._model.model)


def _hf_revision(model: object) -> str | None:
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    return revision if isinstance(revision, str) else None


class LocalModels:
    """The runtime model provider for local and remote models."""

    def __init__(
        self,
        ollama_url: str | None = None,
        *,
        ollama_transport: httpx.BaseTransport | None = None,
        groq_transport: httpx.BaseTransport | None = None,
        groq_limiter: RateLimiter | None = None,
        mistral_transport: httpx.BaseTransport | None = None,
        mistral_limiter: RateLimiter | None = None,
    ) -> None:
        self._ollama_url = configured_ollama_url() if ollama_url is None else ollama_url
        self._ollama_transport = ollama_transport
        self._remote_transports = {"groq": groq_transport, "mistral": mistral_transport}
        self._remote_limiters = {"groq": groq_limiter, "mistral": mistral_limiter}
        self._rerankers: dict[RerankerSpec, SentenceTransformersReranker] = {}
        self._embedders: dict[EmbedderSpec, Embedder] = {}

    def reranker_for(self, spec: RerankerSpec) -> SentenceTransformersReranker:
        if spec not in self._rerankers:
            self._rerankers[spec] = SentenceTransformersReranker(spec)
        return self._rerankers[spec]

    def generator_for(self, spec: GeneratorSpec) -> RemoteChatGenerator | OllamaGenerator:
        if (provider := spec.remote_provider) is not None:
            return RemoteChatGenerator(
                spec,
                transport=self._remote_transports[provider],
                limiter=self._remote_limiters[provider],
            )
        return OllamaGenerator(spec, self._ollama_url, transport=self._ollama_transport)

    def judge_for(self, cache: ResponseCache, spec: GeneratorSpec | None = None) -> Judge:
        from rag_pipeline.judge_model import ModelJudge, load_judge_spec

        spec = spec if spec is not None else load_judge_spec()
        return ModelJudge(spec, self.generator_for, cache)

    def embedder_for(self, spec: EmbedderSpec) -> Embedder:
        if spec not in self._embedders:
            self._embedders[spec] = (
                SentenceTransformersEmbedder(spec)
                if spec.provider == "sentence-transformers"
                else OllamaEmbedder(spec, self._ollama_url, transport=self._ollama_transport)
            )
        return self._embedders[spec]
