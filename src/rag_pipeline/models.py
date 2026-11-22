"""Model roles behind replaceable interfaces: local (Ollama), free-tier APIs, or test fakes."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC
from email.utils import parsedate_to_datetime
from functools import cached_property, partial
from threading import Lock
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
_LOGGER = logging.getLogger(__name__)
_LOCAL_MODEL_LOCKS: dict[tuple[str, str], Lock] = {}
_LOCAL_MODEL_LOCKS_GUARD = Lock()


def local_model_lock(provider: str, model: str) -> Lock:
    with _LOCAL_MODEL_LOCKS_GUARD:
        return _LOCAL_MODEL_LOCKS.setdefault((provider, model), Lock())


def configured_ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)


type GeneratorProvider = Literal["ollama", "groq", "gemini", "nvidia"]


class GeneratorSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: GeneratorProvider
    model: str
    temperature: float = 0.0
    max_output_tokens: PositiveInt = 256
    seed: int = 0
    context_size: PositiveInt = 8192
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    supports_system: bool = True
    structured_output: Literal["json_schema", "prompt"] = "json_schema"
    thinking: Literal["off", "low"] | None = None

    @model_validator(mode="after")
    def _validate_provider_options(self) -> Self:
        if self.provider in ("gemini", "nvidia") and "reasoning_effort" in self.model_fields_set:
            raise ValueError("reasoning_effort is a Groq option in GeneratorSpec")
        if self.thinking == "off" and self.provider not in ("gemini", "nvidia"):
            raise ValueError("thinking: off requires Gemini or NVIDIA in GeneratorSpec")
        if self.thinking == "low" and self.provider != "nvidia":
            raise ValueError("thinking: low requires NVIDIA in GeneratorSpec")
        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        payload: dict[str, Any] = handler(self)
        if self.provider in ("gemini", "nvidia"):
            payload.pop("reasoning_effort", None)
        if self.thinking is None:
            payload.pop("thinking", None)
        # Keep existing Groq and Ollama cache keys and saved configurations compatible.
        if self.supports_system and "supports_system" not in self.model_fields_set:
            payload.pop("supports_system", None)
        if (
            self.structured_output == "json_schema"
            and "structured_output" not in self.model_fields_set
        ):
            payload.pop("structured_output", None)
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
    reasoning_tokens: int = 0
    raw_content_length: int | None = None


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
class DocumentEmbedderWithUsage(Protocol):
    def embed_documents_with_usage(
        self, texts: list[str]
    ) -> tuple[list[list[float]], int | None]: ...


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


def generation_key(
    spec: GeneratorSpec, name: str, version: str | None, system: str, prompt: str
) -> str:
    # Older Gemini cache entries may contain unfiltered thoughts and understated usage.
    namespace = {
        "gemini": "generate-gemini-v2",
        "nvidia": "generate-nvidia-v1",
    }.get(spec.provider, "generate-v1")
    return cache_key(namespace, spec.model_dump(), name, version, system, prompt)


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

    with local_model_lock(spec.provider, spec.model):
        if cache is None:
            return embed()
        key = cache_key(
            "embed_query", spec.model_dump(), embedder.name, model_version(embedder), query
        )
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


_REMOTE_CHATS = {
    "groq": _RemoteChat(
        "https://api.groq.com",
        "/openai/v1/chat/completions",
        "GROQ_API_KEY",
        "max_completion_tokens",
    ),
    # Gemini's OpenAI-compatible endpoint: https://ai.google.dev/gemini-api/docs/openai
    "gemini": _RemoteChat(
        "https://generativelanguage.googleapis.com",
        "/v1beta/openai/chat/completions",
        "GEMINI_API_KEY",
        "max_tokens",
    ),
    "nvidia": _RemoteChat(
        "https://integrate.api.nvidia.com/v1",
        "/chat/completions",
        "NVIDIA_API_KEY",
        "max_tokens",
    ),
}

_THOUGHT_BLOCK = re.compile(
    r"<(thought|think)\s*>(.*?)(?:</\1\s*>|\Z)", flags=re.IGNORECASE | re.DOTALL
)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 6
_RETRY_SECONDS = 300.0
_REQUEST_TIMEOUT = float(os.environ.get("RAG_REMOTE_TIMEOUT", "60"))


def _content_parts(content: str | list[dict[str, Any]] | None) -> tuple[str, str]:
    if isinstance(content, list):
        text: list[str] = []
        thoughts: list[str] = []
        for chunk in content:
            if chunk.get("type") == "text":
                text.append(chunk["text"])
            elif chunk.get("type") in {"thinking", "reasoning"}:
                value: Any = chunk.get("text", chunk.get("thinking"))
                visible, nested_thoughts = _content_parts(value)
                thoughts.extend((visible, nested_thoughts))
        return "".join(text), "".join(thoughts)
    return content or "", ""


def _remote_completion(
    provider: RemoteProvider,
    body: dict[str, Any],
    seconds: float,
    prompted_json: bool,
) -> Completion:
    message: dict[str, Any] = body["choices"][0].get("message") or {}
    content, separate = _content_parts(message.get("content"))
    text = content.strip()
    raw_length = len(content) + len(separate)
    thought_parts = [separate, message.get("reasoning") or message.get("reasoning_content") or ""]
    while match := _THOUGHT_BLOCK.match(text):
        thought_parts.append(match[2])
        text = text[match.end() :].lstrip()

    usage: dict[str, Any] = body["usage"]
    input_tokens, visible_tokens = usage["prompt_tokens"], usage["completion_tokens"]
    total_output = (usage.get("total_tokens") or 0) - input_tokens
    details: dict[str, Any] = usage.get("completion_tokens_details") or {}
    detailed_reasoning = details.get("reasoning_tokens") or 0
    google: dict[str, Any] = body.get("usageMetadata") or body.get("usage_metadata") or {}
    # Gemini's total can include thoughts even when completion_tokens only counts the answer.
    reasoning_tokens = max(
        detailed_reasoning,
        usage.get("reasoning_tokens") or 0,
        *(
            fields.get(key) or 0
            for fields in (usage, google)
            for key in ("thoughtsTokenCount", "thoughts_token_count")
        ),
        total_output - visible_tokens,
        0,
    )
    if reasoning_tokens == 0 and (thoughts := "".join(thought_parts)):
        reasoning_tokens = estimate_tokens("", thoughts, 0) - estimate_tokens("", "", 0)
    # Kimi's completion count includes its reasoning detail, even when total_tokens is absent.
    inclusive = (provider == "nvidia" and 0 < detailed_reasoning <= visible_tokens) or (
        provider == "groq" and total_output == visible_tokens >= reasoning_tokens
    )
    output_tokens = (
        max(visible_tokens, total_output) if inclusive else visible_tokens + reasoning_tokens
    )
    return Completion(
        _first_json_object(text) if prompted_json else text,
        input_tokens,
        output_tokens,
        seconds,
        reasoning_tokens,
        raw_length,
    )


def _retry_after(header: str, now: float) -> float:
    try:
        delay = float(header)
    except ValueError:
        try:
            date = parsedate_to_datetime(header)
            delay = (date if date.tzinfo else date.replace(tzinfo=UTC)).timestamp() - now
        except (ValueError, OverflowError):
            return 0.0
    return delay if math.isfinite(delay) and delay > 0 else 0.0


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    quoted = escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text


def remote_instructions(spec: GeneratorSpec, system: str, schema: dict[str, Any] | None) -> str:
    if schema is not None and spec.structured_output == "prompt":
        schema_text = json.dumps(schema, separators=(",", ":"))
        system += f"\n\nReturn only JSON matching this schema:\n{schema_text}"
    return system


def remote_key_variable(provider: RemoteProvider) -> str:
    return _REMOTE_CHATS[provider].key_variable


def generator_base_url(provider: GeneratorProvider) -> str:
    return configured_ollama_url() if provider == "ollama" else _REMOTE_CHATS[provider].base_url


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
        self._provider: RemoteProvider = provider
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
        self._http = httpx.Client(
            base_url=self._chat.base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=_REQUEST_TIMEOUT,
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
        system = remote_instructions(self._spec, system, schema)
        if not self._spec.supports_system:
            prompt = f"Instructions:\n{system}\n\n{prompt}"
            system = ""
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        payload: dict[str, Any] = {
            "model": self._spec.model,
            "messages": messages,
            "temperature": self._spec.temperature,
            self._chat.max_tokens: self._spec.max_output_tokens,
        }
        if self._spec.provider == "groq":
            payload.update(
                reasoning_effort=self._spec.reasoning_effort,
                include_reasoning=False,
                seed=self._spec.seed,
            )
        elif self._provider == "nvidia":
            payload["stream"] = False
            # Request switches verified in the author's 2026-11-18 NIM probe.
            # https://docs.api.nvidia.com/nim/reference/google-gemma-4-31b-it
            # https://docs.api.nvidia.com/nim/reference/moonshotai-kimi-k3_1-pro-infer
            if self._spec.thinking == "off":
                payload["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
            elif self._spec.thinking == "low":
                payload["reasoning_effort"] = "low"
        elif self._spec.thinking == "off":
            # Gemma 4 maps minimal to off: https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api
            # Chat REST envelope: https://ai.google.dev/gemini-api/docs/openai#extra-body
            payload["extra_body"] = {
                "google": {
                    "thinking_config": {"thinking_level": "minimal", "include_thoughts": False}
                }
            }
        prompted_json = schema is not None and self._spec.structured_output == "prompt"
        if schema is not None and not prompted_json:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "rag_response", "strict": True, "schema": schema},
            }
        tokens = estimate_tokens(system, prompt, self._spec.max_output_tokens)
        return self._complete_with_retries(payload, tokens, prompted_json)

    def _complete_with_retries(
        self, payload: dict[str, Any], tokens: int, prompted_json: bool
    ) -> Completion:
        clock = self._limiter.clock
        deadline: float | None = None
        failure = ""
        last_error: httpx.HTTPError | None = None
        attempts = 0
        for _ in range(_MAX_ATTEMPTS):
            try:
                request = self._limiter.reserve(self._spec.model, tokens, deadline=deadline)
            except TimeoutError:
                failure += ", quota wait exceeds the retry budget"
                break
            attempts += 1
            now = clock.time()
            if deadline is None:
                # A fresh call may wait for a quota window before the retry budget starts.
                deadline = now + _RETRY_SECONDS
            retry_after = 0.0
            try:
                started = time.perf_counter()
                response = self._http.post(
                    self._chat.path,
                    json=payload,
                    timeout=min(_REQUEST_TIMEOUT, max(0, deadline - now)),
                )
                seconds = time.perf_counter() - started
                response.raise_for_status()
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as error:
                failure, last_error = type(error).__name__, error
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if status not in _RETRYABLE_STATUS:
                    raise
                failure, last_error = f"HTTP {status}", error
                retry_after = _retry_after(
                    error.response.headers.get("retry-after", ""), clock.time()
                )
            else:
                result = _remote_completion(self._provider, response.json(), seconds, prompted_json)
                self._limiter.record_usage(request, result.input_tokens + result.output_tokens)
                return result
            # Keep the reservation on failure so each retry counts against RPM and TPM.
            backoff = 2.0**attempts
            delay = max(random.uniform(backoff / 2, backoff), retry_after)
            if attempts == _MAX_ATTEMPTS or clock.time() + delay >= deadline:
                break
            _LOGGER.warning(
                "%s %s retry %d after %s (%.1f s)",
                self._spec.provider.title(),
                self._spec.model,
                attempts,
                failure,
                delay,
            )
            clock.sleep(delay)
        raise httpx.HTTPError(
            f"{self._spec.provider.title()} model {self._spec.model!r} exhausted retries after "
            f"{attempts} attempts ({failure}). Rerun the command to resume."
        ) from last_error


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
        return self.embed_documents_with_usage(texts)[0]

    def embed_documents_with_usage(self, texts: list[str]) -> tuple[list[list[float]], int | None]:
        return self._embed(texts)

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
        gemini_transport: httpx.BaseTransport | None = None,
        gemini_limiter: RateLimiter | None = None,
        nvidia_transport: httpx.BaseTransport | None = None,
        nvidia_limiter: RateLimiter | None = None,
    ) -> None:
        self._ollama_url = configured_ollama_url() if ollama_url is None else ollama_url
        self._ollama_transport = ollama_transport
        self._remote_transports = {
            "groq": groq_transport,
            "gemini": gemini_transport,
            "nvidia": nvidia_transport,
        }
        self._remote_limiters = {
            "groq": groq_limiter,
            "gemini": gemini_limiter,
            "nvidia": nvidia_limiter,
        }
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
