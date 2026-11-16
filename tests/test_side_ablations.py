from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from conftest import FTX, FTX_QUERY, PIXEL
from fakes import (
    ConceptFakeEmbedder,
    FakeModels,
    GroundedFakeGenerator,
    PhraseFakeReranker,
    ScriptedVariant,
)
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from rag_pipeline import cli
from rag_pipeline.cache import ResponseCache
from rag_pipeline.chunking import ChunkingSettings
from rag_pipeline.corpus import BenchmarkQuery, Evidence
from rag_pipeline.demo import IndexBuild
from rag_pipeline.evaluation import RunStore, evaluate
from rag_pipeline.index import ChunkIndex, IndexMismatch, index_collection
from rag_pipeline.ingest import ingest
from rag_pipeline.models import Completion, GeneratorSpec, LocalModels
from rag_pipeline.rate_limit import RateLimiter, groq_limits
from rag_pipeline.variants import VariantConfig, build_variant, plan_remote_run

CONFIGS = Path("configs/side-ablations")


def test_chunk_sizes_coexist_and_the_yaml_variant_selects_its_index():
    article = replace(
        PIXEL,
        body=" ".join(f"Fact {i} on Pixel " + "software " * 15 + "updates." for i in range(20)),
    )
    qdrant = QdrantClient(":memory:")
    models = FakeModels(GroundedFakeGenerator({}), PhraseFakeReranker("Fact"))
    version_names: list[str] = []
    chunk_counts: list[int] = []
    max_seen: list[int] = []
    for size in (100, 200, 400):
        config = VariantConfig.from_yaml(CONFIGS / f"chunk-{size}.yaml")
        build = IndexBuild.for_variant(config)
        assert build.embedder is not None and build.settings.max_words == size
        assert build.collection == index_collection(build.settings, build.embedder.model)
        index = ChunkIndex(qdrant, build.collection, ConceptFakeEmbedder())
        report = ingest([article], index, build.settings)
        variant = build_variant(config, index=index, models=models)
        chunks = variant.retrieve("Pixel software updates")
        assert chunks and all(chunk.article_id == article.id for chunk in chunks)
        version_names.append(report.index_version)
        chunk_counts.append(report.chunks_written)
        max_seen.append(max(len(chunk.text.split()) for chunk in chunks))
        assert variant.index_version() == report.index_version
    assert len(set(version_names)) == len(qdrant.get_collections().collections) == 3
    assert chunk_counts[0] > chunk_counts[1] > chunk_counts[2]
    assert max_seen[0] <= 100 < max_seen[1] <= 200 < max_seen[2] <= 400
    reference = VariantConfig.from_yaml(CONFIGS / "chunk-200.yaml").run_config()
    for name, field_group in (
        ("chunk-100", "index"),
        ("chunk-400", "index"),
        ("semantic", "index"),
        ("hyde", "retrieval"),
    ):
        configured = VariantConfig.from_yaml(CONFIGS / f"{name}.yaml").run_config()
        assert {
            key for key in reference if key != "name" and configured[key] != reference[key]
        } == {field_group}
    reference_build = IndexBuild.for_variant(VariantConfig.from_yaml(CONFIGS / "chunk-200.yaml"))
    semantic_build = IndexBuild.for_variant(VariantConfig.from_yaml(CONFIGS / "semantic.yaml"))
    assert (
        replace(
            semantic_build.settings,
            strategy=reference_build.settings.strategy,
            semantic_percentile=reference_build.settings.semantic_percentile,
        )
        == reference_build.settings
    )
    assert semantic_build.embedder is not None
    assert semantic_build.collection == index_collection(
        semantic_build.settings, semantic_build.embedder.model
    )


class TopicEmbedder(ConceptFakeEmbedder):
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(texts)
        return [
            [1.0, 0.0] if text.startswith(("Granite", "Basalt")) else [0.0, 1.0] for text in texts
        ]


TOPIC_SENTENCES = [
    "Granite forms slowly.",
    "Basalt cools quickly.",
    "Markets opened higher.",
    "The index gained.",
    "Bond yields fell.",
    "Investors bought equities.",
    "Income rose steadily.",
]


def test_semantic_ingest_splits_at_topic_and_cap_and_reuses_sentence_embeddings(tmp_path: Path):
    sentences = TOPIC_SENTENCES
    article = replace(FTX, body=" ".join(sentences))
    long_article = replace(PIXEL, body="Word " * 10 + "sentence.")
    settings = ChunkingSettings(
        max_words=9, overlap_sentences=0, strategy="semantic", semantic_percentile=90
    )
    qdrant = QdrantClient(":memory:")
    cache = ResponseCache(tmp_path / "semantic.sqlite3")
    embedder = TopicEmbedder()
    collection = index_collection(settings, "concept-fake")
    index = ChunkIndex(qdrant, collection, embedder)
    first = ingest([article, long_article], index, settings, cache=cache)

    hits = index.search_bm25(
        "granite basalt markets index bond investors income sentence", limit=10
    )
    actual = {hit.chunk_id: hit.text for hit in hits}
    expected = {
        f"{article.id}#000": " ".join(sentences[:2]),
        f"{article.id}#001": " ".join(sentences[2:5]),
        f"{article.id}#002": " ".join(sentences[5:]),
        f"{long_article.id}#000": long_article.body,
    }
    assert first.index_version.startswith("chunks-sem-p90-w9-concept-fake@")
    assert actual == expected
    assert sentences[3] in actual[f"{article.id}#001"]
    assert all(len(text.split()) <= 9 for id_, text in actual.items() if id_.startswith(article.id))
    assert embedder.batches[0] == sentences

    second = ingest([article, long_article], index, settings, cache=cache)
    assert second.unchanged == (article.id, long_article.id) and second.chunks_written == 0
    assert second.index_version == first.index_version
    assert len(embedder.batches) == 2
    with pytest.raises(IndexMismatch, match="chunking"):
        ingest([article], index, ChunkingSettings(max_words=9, overlap_sentences=0))

    rebuilt = ChunkIndex(qdrant, "semantic-copy", embedder)
    ingest([article, long_article], rebuilt, settings, cache=cache)
    assert len(embedder.batches) == 3
    assert embedder.batches[-1] == list(expected.values())
    plain = ChunkIndex(qdrant, "sentence-only", embedder)
    ingest([article], plain, ChunkingSettings(max_words=9, overlap_sentences=0))
    with pytest.raises(IndexMismatch, match="chunking"):
        ingest([article], plain, settings, cache=cache)


@pytest.mark.parametrize("max_words", [9, 5], ids=["overlap-fits", "overlap-exceeds-cap"])
def test_semantic_ingest_carries_whole_sentences_that_fit_at_both_boundaries(max_words: int):
    sentences = TOPIC_SENTENCES
    article = replace(FTX, body=" ".join(sentences))
    settings = ChunkingSettings(max_words=max_words, overlap_sentences=1, strategy="semantic")
    collection = index_collection(settings, "concept-fake")
    index = ChunkIndex(QdrantClient(":memory:"), collection, TopicEmbedder())

    report = ingest([article], index, settings)

    hits = index.search_bm25("granite basalt markets index bond investors income", limit=10)
    texts = [hit.text for hit in sorted(hits, key=lambda hit: hit.chunk_id)]
    if max_words == 9:
        # The first break is the topic shift. The other breaks follow the word cap.
        assert texts == [
            " ".join(sentences[:2]),
            " ".join(sentences[1:4]),
            " ".join(sentences[3:6]),
            " ".join(sentences[5:]),
        ]
    else:
        assert texts == sentences
    assert all(len(text.split()) <= max_words for text in texts)
    assert report.index_version.startswith(f"chunks-sem-p90-w{max_words}-o1-concept-fake@")


PASSAGE = "HYPOTHETICAL description of a fraud trial outside Manhattan."
QUESTION = "Where did it happen?"
EVIDENCE = "The fraud trial began outside Manhattan."


class HydeEmbedder(ConceptFakeEmbedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if EVIDENCE in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if text == PASSAGE else [0.0, 1.0]


class PassageGenerator(GroundedFakeGenerator):
    def complete(self, system: str, prompt: str) -> Completion:
        self.prompts.append(prompt)
        assert "untrusted" in system
        assert QUESTION in prompt
        return Completion(PASSAGE, 15, 10, call_seconds=0.7)


class HydeModels(FakeModels):
    def __init__(self) -> None:
        super().__init__(GroundedFakeGenerator({EVIDENCE: "Manhattan"}))
        self.passage = PassageGenerator({})

    def generator_for(self, spec: GeneratorSpec) -> GroundedFakeGenerator:
        if spec.model == "qwen3.5:4b":
            return self.passage
        return super().generator_for(spec)


def test_hyde_changes_dense_retrieval_and_charges_only_its_own_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    config = VariantConfig.model_validate(
        {
            "name": "hyde-fixture",
            "index": {
                "collection": "fixture",
                "embedder": {"provider": "ollama", "model": "qwen3-embedding:0.6b"},
            },
            "retrieval": {"mode": "hybrid", "top_k": 1, "candidates": 1},
            "generation": {"generator": {"provider": "ollama", "model": "qwen3.5:9b"}},
        }
    )
    changed = config.model_dump()
    changed["retrieval"]["hyde"] = {
        "enabled": True,
        "generator": {"provider": "ollama", "model": "qwen3.5:4b"},
    }
    hyde = VariantConfig.model_validate(changed)
    index = ChunkIndex(QdrantClient(":memory:"), "fixture", HydeEmbedder())
    ingest(
        [replace(FTX, body=EVIDENCE), replace(PIXEL, body="Cash was transferred.")],
        index,
        ChunkingSettings(),
    )
    models = HydeModels()
    cache = ResponseCache(tmp_path / "models.sqlite3")
    store = RunStore(tmp_path / "runs")
    query = BenchmarkQuery(
        "q_hyde", QUESTION, "Manhattan", "inference_query", (Evidence(FTX.url, EVIDENCE),)
    )
    vanilla = build_variant(config, index=index, models=models, cache=cache)
    baseline = evaluate(vanilla, [query], "dev", store)
    assert baseline.records[0].hit_at_4 is False and baseline.records[0].abstained
    assert models.passage.prompts == []

    pipeline = build_variant(hyde, index=index, models=models, cache=cache)
    result = evaluate(pipeline, [query], "dev", store)
    answer = result.records[0]
    assert answer.hit_at_4 is True and answer.answer == "Manhattan"
    assert result.meta["fingerprint"] != baseline.meta["fingerprint"]
    assert PASSAGE not in models.generator.prompts[-1]
    assert [stage.name for stage in answer.trace] == ["hyde", "retrieve", "generate"]
    expansion, retrieve, generate = answer.trace
    assert (expansion.input_tokens, expansion.output_tokens, expansion.seconds) == (15, 10, 0.7)
    assert answer.seconds >= 0.7
    assert answer.notional_cost_usd == pytest.approx(
        (
            15 * 0.10
            + 10 * 0.20
            + (retrieve.input_tokens or 0) * 0.010
            + (generate.input_tokens or 0) * 0.10
            + generate.output_tokens * 0.15
        )
        / 1_000_000
    )
    metric = "generator_context_tokens_per_question"
    assert result.metrics[metric] == generate.input_tokens
    assert result.intervals[metric] == (generate.input_tokens, generate.input_tokens)
    assert pipeline.answer(QUESTION).trace[0].cache_hit
    assert len(models.passage.prompts) == 1
    with patch.object(index, "search_hybrid", wraps=index.search_hybrid) as search:
        pipeline.answer(QUESTION)
        assert search.call_args is not None and search.call_args.args[0] == QUESTION
        changed["retrieval"]["hyde"]["bm25"] = "both"
        expanded = build_variant(
            VariantConfig.model_validate(changed), index=index, models=models, cache=cache
        )
        expanded.answer(QUESTION)
        assert search.call_args is not None and search.call_args.args[0] == f"{QUESTION} {PASSAGE}"
    assert len(models.passage.prompts) == 1
    monkeypatch.setattr(cli, "RUNS_DIR", store.root)
    table = CliRunner().invoke(cli.app, ["table", result.run_id])
    assert table.exit_code == 0 and metric in table.output


def test_absent_and_disabled_hyde_keep_the_previous_run_fingerprint(tmp_path: Path):
    script: dict[str, tuple[str, list[str]]] = {FTX_QUERY.question: (FTX_QUERY.answer, [])}
    store = RunStore(tmp_path)
    variant = ScriptedVariant("fingerprint", script, retrieval_mode="none")
    first = evaluate(variant, [FTX_QUERY], "dev", store)
    config = variant.config.model_dump()
    config["retrieval"]["hyde"] = {"enabled": False}
    variant.config = VariantConfig.model_validate(config)
    second = evaluate(variant, [FTX_QUERY], "dev", store)
    assert first.records[0].answer == second.records[0].answer
    assert first.meta["fingerprint"] == second.meta["fingerprint"] == "59be0664c6ddb8be"


def test_retrieval_only_remote_plan_accounts_for_the_cached_hyde_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    model = "openai/gpt-oss-120b"
    config = VariantConfig.model_validate(
        {
            "name": "remote-expansion",
            "index": {"collection": "fixture", "embedder": {"provider": "ollama", "model": "fake"}},
            "retrieval": {
                "mode": "dense",
                "hyde": {"generator": {"provider": "groq", "model": model}},
            },
            "generation": {"generator": {"provider": "ollama", "model": "fake"}},
        }
    )
    index = ChunkIndex(QdrantClient(":memory:"), "fixture", HydeEmbedder())
    ingest([replace(FTX, body=EVIDENCE)], index, ChunkingSettings())
    query = BenchmarkQuery(
        "q_hyde", QUESTION, "Manhattan", "inference_query", (Evidence(FTX.url, EVIDENCE),)
    )
    cache = ResponseCache(tmp_path / "cache.sqlite3")
    calls: list[str] = []

    def chat(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.groq.com"
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": PASSAGE}}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 10},
            },
        )

    def ollama_metadata(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        return httpx.Response(200, json={"details": {}})

    monkeypatch.setenv("GROQ_API_KEY", "fixture")
    models = LocalModels(
        ollama_transport=httpx.MockTransport(ollama_metadata),
        groq_transport=httpx.MockTransport(chat),
        groq_limiter=RateLimiter(cache.path, groq_limits(model)),
    )
    variant = build_variant(config, index=index, models=models, cache=cache)
    before = plan_remote_run(config, [query], cache, retrieval_only=True)
    assert (before.calls, before.cached, before.days) == (1, 0, 1)
    result = evaluate(variant, [query], "dev", RunStore(tmp_path / "runs"), retrieval_only=True)
    assert [stage.name for stage in result.records[0].trace] == ["hyde", "retrieve"]
    assert "generator_context_tokens_per_question" not in result.metrics
    after = plan_remote_run(config, [query], cache, retrieval_only=True)
    assert (after.calls, after.cached, after.days) == (1, 1, 0)
    assert len(calls) == 1
