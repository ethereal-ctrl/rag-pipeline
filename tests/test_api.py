import json
import re
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from conftest import FTX, FTX_QUERY
from fakes import FakeModels, GroundedFakeGenerator
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from rag_pipeline import cli
from rag_pipeline.api import create_app
from rag_pipeline.chunk_context import ContextBuild
from rag_pipeline.chunking import ChunkingSettings, chunk_article
from rag_pipeline.corpus import Article
from rag_pipeline.index import ChunkIndex
from rag_pipeline.models import Completion, Embedder, EmbedderSpec, LocalModels
from rag_pipeline.variants import VariantConfig

DEMO_VARIANT = "hybrid-rerank-cited"
OLLAMA_HOST = "custom-ollama.test"


class ApiFakeModels(FakeModels):
    def embedder_for(self, spec: EmbedderSpec) -> Embedder:
        raise AssertionError(f"Fixture uses BM25 and needs no embedder for {spec.model}.")


def save_variants(folder: Path) -> None:
    config = VariantConfig.from_yaml(Path("configs/variants/hybrid-rerank-cited.yaml"))
    payload = config.run_config()
    payload["retrieval"] = {"mode": "bm25"}
    payload["index"] = {"collection": "fixture"}
    payload["generation"]["citations"]["support_check"] = "none"
    (folder / f"{DEMO_VARIANT}.yaml").write_text(json.dumps(payload))
    source = Path("configs/variants/closed-book-groq.yaml")
    (folder / source.name).write_text(source.read_text())
    (folder / "ignored.yml").write_text("{}")


class CitedGenerator(GroundedFakeGenerator):
    def __init__(self, chunk_id: str) -> None:
        super().__init__({})
        self.chunk_id = chunk_id

    def complete_structured(
        self, system: str, prompt: str, schema: dict[str, object]
    ) -> Completion:
        content: dict[str, object] = (
            {"answer": "", "statements": []}
            if "Unknown question?" in prompt
            else {
                "answer": FTX_QUERY.answer,
                "statements": [
                    {"statement": "FTX's chief is on trial.", "chunk_ids": [self.chunk_id]},
                    {"statement": "An invalid Citation", "chunk_ids": ["not-shown"]},
                ],
            }
        )
        return Completion(json.dumps(content), 23, 11)


def test_http_answer_returns_original_citation_and_root_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    save_variants(tmp_path)
    monkeypatch.setenv("PHOENIX_URL", "https://phoenix-ui.test/")
    article = replace(FTX, body=FTX.body + ' <img src=x onerror="alert(1)">')
    chunk = chunk_article(article, ChunkingSettings())[0]
    qdrant = QdrantClient(":memory:")
    index = ChunkIndex(qdrant, "fixture")
    index.upsert(
        [replace(chunk, context="Added Chunk Context")],
        context=ContextBuild("fixture-context", "fixture-build"),
    )
    models = ApiFakeModels(CitedGenerator(chunk.id))
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with TestClient(
            create_app(
                variants_dir=tmp_path,
                qdrant=qdrant,
                models=models,
                tracer=provider.get_tracer("test"),
                transport=httpx.MockTransport(service_response),
            )
        ) as client:
            response = client.post(
                "/answer", json={"question": FTX_QUERY.question, "variant": DEMO_VARIANT}
            )
            abstained = client.post(
                "/answer", json={"question": "Unknown question?", "variant": DEMO_VARIANT}
            )
    finally:
        provider.shutdown()

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["answer"] == FTX_QUERY.answer and data["abstained"] is False
    assert data["variant"] == DEMO_VARIANT and data["index_version"] == index.version()
    assert data["citations"] == [
        {
            "statement": "FTX's chief is on trial.",
            "chunks": [
                {
                    "id": chunk.id,
                    "title": article.title,
                    "source": article.source,
                    "published_at": article.published_at,
                    "text": chunk.text,
                }
            ],
        }
    ]
    assert data["removed_statements"][0]["reason"] == "unshown_chunk"
    assert data["stages"][-1]["input_tokens"] == 23 and data["stages"][-1]["output_tokens"] == 11
    assert data["stages"][-1]["seconds"] >= 0
    assert re.fullmatch("[0-9a-f]{32}", data["trace_id"])
    roots = [span for span in exporter.get_finished_spans() if span.parent is None]
    assert roots[0].name == "answer" and roots[0].context is not None
    assert data["trace_id"] == f"{roots[0].context.trace_id:032x}"
    assert data["phoenix_url"] == (f"https://phoenix-ui.test/redirects/traces/{data['trace_id']}")
    assert abstained.json()["abstained"] is True
    assert len(models.requested) == 1


def service_response(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("unreachable", request=request)


def test_http_validation_health_and_allow_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    save_variants(tmp_path)
    outside = tmp_path.parent / "outside.yaml"
    outside.write_text((tmp_path / f"{DEMO_VARIANT}.yaml").read_text())
    (tmp_path / "linked.yaml").symlink_to(outside)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("RAG_DEMO_VARIANT", "closed-book-groq")
    with TestClient(
        create_app(
            variants_dir=tmp_path,
            qdrant=QdrantClient(":memory:"),
            models=ApiFakeModels(GroundedFakeGenerator({})),
            transport=httpx.MockTransport(service_response),
        )
    ) as client:
        variants = client.get("/variants").json()
        assert variants["default"] == "closed-book-groq"
        assert [variant["name"] for variant in variants["variants"]] == [
            "closed-book-groq",
            DEMO_VARIANT,
        ]
        assert "bm25" in variants["variants"][1]["description"]
        for variant, question, status in (
            ("unknown", "Question?", 404),
            ("../x", "Question?", 422),
            (DEMO_VARIANT, "   ", 422),
            (DEMO_VARIANT, "x" * 501, 422),
        ):
            assert (
                client.post("/answer", json={"variant": variant, "question": question}).status_code
                == status
            )
        missing = client.post("/answer", json={"variant": DEMO_VARIANT, "question": "Question?"})
        assert missing.status_code == 409
        assert "rag ingest --collection fixture" in missing.json()["detail"]
        groq = client.post("/answer", json={"variant": "closed-book-groq", "question": "Question?"})
        assert groq.status_code == 503 and "GROQ_API_KEY" in groq.json()["detail"]
        assert client.get("/health").json() == {
            "qdrant": False,
            "ollama": False,
            "trace_exporter": False,
        }
        page = client.get("/")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert "textContent" in page.text and "chunk.text" in page.text
        assert not any(
            sink in page.text for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML")
        )


def test_api_and_cli_use_configured_ollama_for_generation_and_embeddings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("OLLAMA_URL", f"http://{OLLAMA_HOST}:11223")
    monkeypatch.setenv("RAG_TRACING", "0")
    paths: list[str] = []

    def ollama(request: httpx.Request) -> httpx.Response:
        assert request.url.host == OLLAMA_HOST and request.url.port == 11223
        paths.append(request.url.path)
        if request.url.path == "/api/embed":
            count = len(json.loads(request.content)["input"])
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]] * count})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"content": "Fixture answer"}})

    transport = httpx.MockTransport(ollama)
    qdrant = QdrantClient(":memory:")

    def qdrant_fixture(*, url: str, check_compatibility: bool) -> QdrantClient:
        assert url == "http://localhost:6333" and not check_compatibility
        return qdrant

    def articles_fixture(_: Path) -> list[Article]:
        return [FTX]

    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setattr(cli, "LocalModels", lambda: LocalModels(ollama_transport=transport))
    monkeypatch.setattr(cli, "QdrantClient", qdrant_fixture)
    fetched: list[Path] = []
    monkeypatch.setattr(cli, "fetch_dataset", fetched.append)
    monkeypatch.setattr(cli, "load_articles", articles_fixture)
    monkeypatch.setattr(cli, "RAW_DIR", tmp_path)
    monkeypatch.delenv("RAG_DEMO_VARIANT", raising=False)
    run = CliRunner().invoke(cli.app, ["bootstrap"])
    assert run.exit_code == 0, run.output
    assert paths == ["/api/embed"] and fetched == [tmp_path]
    assert "Chunks written" in run.output

    variants = tmp_path / "variants"
    variants.mkdir()
    source = Path("configs/variants/closed-book.yaml")
    (variants / source.name).write_text(source.read_text())
    with TestClient(
        create_app(
            variants_dir=variants,
            qdrant=QdrantClient(":memory:"),
            models=LocalModels(ollama_transport=transport),
            transport=httpx.MockTransport(
                lambda request: (
                    ollama(request)
                    if request.url.host == OLLAMA_HOST
                    else service_response(request)
                )
            ),
        )
    ) as client:
        response = client.post("/answer", json={"variant": "closed-book", "question": "Question?"})
        assert response.json()["answer"] == "Fixture answer"
        assert client.get("/health").json()["ollama"] is True
    assert paths[-2:] == ["/api/chat", "/api/tags"]
