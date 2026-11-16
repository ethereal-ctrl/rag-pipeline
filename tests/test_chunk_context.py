from dataclasses import replace
from pathlib import Path

import pytest
from conftest import FTX, FTX_QUERY, PIXEL
from fakes import ConceptFakeEmbedder, FakeModels, GroundedFakeGenerator
from qdrant_client import QdrantClient, models

from rag_pipeline.cache import ResponseCache
from rag_pipeline.chunk_context import ModelContextualiser
from rag_pipeline.evaluation import RunStore, evaluate
from rag_pipeline.index import ChunkIndex, IndexMismatch, index_collection
from rag_pipeline.ingest import ChunkingSettings, ingest
from rag_pipeline.models import Completion, GeneratorSpec, estimate_tokens
from rag_pipeline.variants import VariantConfig, build_variant

SMALL = ChunkingSettings(max_words=20, overlap_sentences=0)
MODEL = "qwen3.5:4b"


class ContextGenerator(GroundedFakeGenerator):
    def __init__(self, *, crash_after: int | None = None, revision: str = "revision-one") -> None:
        super().__init__({})
        self.crash_after = crash_after
        self.revision = revision
        self.systems: list[str] = []

    def version(self) -> str:
        return self.revision

    def complete(self, system: str, prompt: str) -> Completion:
        if len(self.prompts) == self.crash_after:
            raise RuntimeError("interrupted Chunk Context generation")
        self.prompts.append(prompt)
        self.systems.append(system)
        context = (
            "Quasar identifies the October FTX trial." if FTX.title in prompt else "Phone news."
        )
        return Completion(context, 100, 10)


def contextualiser(
    path: Path,
    generator: ContextGenerator,
    *,
    model: str = MODEL,
    context_size: int = 8192,
    article_tokens: int = 4096,
) -> ModelContextualiser:
    spec = GeneratorSpec(
        provider="ollama",
        model=model,
        context_size=context_size,
        max_output_tokens=128,
    )
    return ModelContextualiser(
        spec, lambda _: generator, ResponseCache(path), article_tokens=article_tokens
    )


def test_context_generation_resumes_by_chunk_and_keeps_each_article_prefix(
    index: ChunkIndex, tmp_path: Path
):
    path = tmp_path / "responses.sqlite3"
    interrupted = ContextGenerator(crash_after=1)
    with pytest.raises(RuntimeError, match="interrupted Chunk Context generation"):
        ingest([FTX, PIXEL], index, SMALL, contextualiser=contextualiser(path, interrupted))

    resumed = ContextGenerator()
    report = ingest([FTX, PIXEL], index, SMALL, contextualiser=contextualiser(path, resumed))

    stats = report.context
    assert stats is not None
    assert stats.chunks_contextualised == report.chunks_written
    assert stats.cache_hits == 1
    assert len(resumed.prompts) == report.chunks_written - 1
    assert stats.input_tokens == 100 * report.chunks_written
    assert stats.output_tokens == 10 * report.chunks_written
    assert stats.generation_seconds > 0
    assert stats.notional_cost_usd == pytest.approx(
        report.chunks_written * (100 * 0.10 + 10 * 0.20) / 1e6
    )
    prompts = interrupted.prompts + resumed.prompts
    prefixes = [prompt.split("<chunk>")[0] for prompt in prompts]
    ftx_prefixes = [prefix for prefix in prefixes if FTX.title in prefix]
    assert len(ftx_prefixes) >= 2 and len(set(ftx_prefixes)) == 1
    assert prefixes[: len(ftx_prefixes)] == ftx_prefixes
    assert FTX.source in prefixes[0] and FTX.published_at in prefixes[0]
    assert all(prompt.endswith("</chunk>") for prompt in prompts)
    assert len(set(interrupted.systems + resumed.systems)) == 1

    unchanged = ingest([FTX, PIXEL], index, SMALL, contextualiser=contextualiser(path, resumed))
    assert unchanged.unchanged == (FTX.id, PIXEL.id)
    assert unchanged.chunks_written == 0
    assert len(resumed.prompts) == report.chunks_written - 1


def test_article_truncation_is_reported_and_prefix_stays_shared(index: ChunkIndex, tmp_path: Path):
    generator = ContextGenerator()
    context_size = 1024
    report = ingest(
        [FTX],
        index,
        SMALL,
        contextualiser=contextualiser(
            tmp_path / "responses.sqlite3", generator, article_tokens=45, context_size=context_size
        ),
    )

    assert report.context is not None and report.context.truncated_articles == (FTX.id,)
    prefixes = [prompt.split("<chunk>")[0] for prompt in generator.prompts]
    assert len(prefixes) >= 2 and len(set(prefixes)) == 1
    assert FTX.title in prefixes[0]
    assert FTX.body not in prefixes[0]
    assert "The jury will hear" in generator.prompts[-1].split("<chunk>")[1]
    assert all(
        estimate_tokens(system, prompt, 128) <= context_size
        for system, prompt in zip(generator.systems, generator.prompts, strict=True)
    )


@pytest.mark.parametrize(("repetitions", "truncated"), [(129, False), (260, True)])
def test_article_prefix_uses_token_budget_with_default_window(
    index: ChunkIndex, tmp_path: Path, repetitions: int, truncated: bool
):
    sentence = "The local council reviewed plans for the region's rail network. "
    last = "At year's end, the council approved the Northbank station."
    article = replace(FTX, body=sentence * repetitions + last)
    generator = ContextGenerator()

    report = ingest(
        [article],
        index,
        ChunkingSettings(),
        contextualiser=contextualiser(tmp_path / "responses.sqlite3", generator),
    )

    assert report.context is not None
    assert report.context.truncated_articles == ((article.id,) if truncated else ())
    prefix, _, chunk = generator.prompts[-1].partition("<chunk>")
    assert (article.body in prefix) is not truncated
    assert (last in prefix) is not truncated
    assert last in chunk
    assert all(
        estimate_tokens(system, prompt, 128) <= 8192
        for system, prompt in zip(generator.systems, generator.prompts, strict=True)
    )


class ContextEmbedder(ConceptFakeEmbedder):
    def __init__(self) -> None:
        self.indexed_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.indexed_texts.extend(texts)
        return [[1.0, 0.0] if "Quasar" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


@pytest.mark.parametrize("retry_text", [" ", "Retry context."], ids=["fallback", "recovered"])
def test_empty_chunk_context_retries_and_is_cached(tmp_path: Path, retry_text: str):
    calls: list[tuple[int, str]] = []

    class SeededGenerator(ContextGenerator):
        def __init__(self, seed: int) -> None:
            super().__init__()
            self.seed = seed

        def complete(self, system: str, prompt: str) -> Completion:
            chunk = prompt.split("<chunk>")[1]
            calls.append((self.seed, chunk))
            if "Sam Bankman-Fried" in chunk:
                text = "" if self.seed == 7 else retry_text
                return Completion(text, 100, 2)
            return Completion("Phone news.", 100, 10)

    client = QdrantClient(":memory:")
    embedder = ContextEmbedder()
    first = ChunkIndex(client, "first", embedder)
    second = ChunkIndex(client, "second", embedder)
    spec = GeneratorSpec(provider="ollama", model=MODEL, seed=7)
    cache_path = tmp_path / "responses.sqlite3"

    def generator_for(options: GeneratorSpec) -> SeededGenerator:
        return SeededGenerator(options.seed)

    writer = ModelContextualiser(spec, generator_for, ResponseCache(cache_path))
    settings = ChunkingSettings(max_words=100)

    report = ingest([FTX, PIXEL], first, settings, contextualiser=writer)
    restarted = ingest(
        [FTX, PIXEL],
        second,
        settings,
        contextualiser=ModelContextualiser(spec, generator_for, ResponseCache(cache_path)),
    )

    assert report.chunks_written == restarted.chunks_written == 2
    stats = report.context
    assert stats is not None and restarted.context is not None
    assert stats.empty_contexts == restarted.context.empty_contexts == int(not retry_text.strip())
    assert stats.input_tokens == restarted.context.input_tokens == 300
    assert stats.output_tokens == restarted.context.output_tokens == 14
    assert stats.generation_seconds == restarted.context.generation_seconds
    assert stats.cache_hits == 0 and restarted.context.cache_hits == 2
    assert [seed for seed, _ in calls] == [7, 8, 7]
    assert calls[0][1] == calls[1][1]
    original = first.search_bm25("Sam Bankman-Fried", limit=1)[0]
    assert original.text == FTX.body and original.context == retry_text.strip()
    indexed_text = f"{retry_text}\n\n{FTX.body}" if retry_text.strip() else FTX.body
    assert embedder.indexed_texts[0] == embedder.indexed_texts[2] == indexed_text


@pytest.mark.parametrize("mode", ["bm25", "dense"])
def test_context_improves_retrieval_without_changing_chunk_text_or_evidence(
    tmp_path: Path, mode: str
):
    settings = ChunkingSettings(max_words=100)
    client = QdrantClient(":memory:")
    embedder = ContextEmbedder()
    collection = index_collection(settings, "qwen3-embedding:0.6b", context_model=MODEL)
    index = ChunkIndex(client, collection, embedder)
    ingest(
        [FTX, PIXEL],
        index,
        settings,
        contextualiser=contextualiser(tmp_path / "responses.sqlite3", ContextGenerator()),
    )
    config = VariantConfig.model_validate(
        {
            "name": mode,
            "index": {
                "collection": collection,
                "embedder": {"provider": "ollama", "model": "qwen3-embedding:0.6b"},
            },
            "retrieval": {"mode": mode, "top_k": 1},
            "generation": {"generator": {"provider": "ollama", "model": MODEL}},
        }
    )
    generator = GroundedFakeGenerator({"Sam Bankman-Fried started": "Sam Bankman-Fried"})
    variant = build_variant(config, index=index, models=FakeModels(generator))

    answer = variant.answer("Quasar")
    result = evaluate(
        variant, [replace(FTX_QUERY, question="Quasar")], "smoke", RunStore(tmp_path / "runs")
    )

    (chunk,) = answer.retrieved
    assert chunk.chunk_id == f"{FTX.id}#000"
    assert chunk.text == FTX.body
    assert chunk.context == "Quasar identifies the October FTX trial."
    assert embedder.indexed_texts[0] == f"{chunk.context}\n\n{FTX.body}"
    assert FTX.body in generator.prompts[0] and chunk.context not in generator.prompts[0]
    assert result.metrics["hits@4"] == result.metrics["evidence_recall@10"] == 1.0


def test_contextual_and_plain_index_versions_coexist_and_reject_mixing(tmp_path: Path):
    client = QdrantClient(":memory:")
    settings = ChunkingSettings(max_words=100)
    plain = ChunkIndex(client, index_collection(settings, None))
    context = ChunkIndex(client, index_collection(settings, None, context_model=MODEL))
    plain_report = ingest([FTX], plain, settings)
    model = ContextGenerator()
    writer = contextualiser(tmp_path / "responses.sqlite3", model)
    report = ingest([FTX], context, settings, contextualiser=writer)

    assert plain_report.context is None
    assert plain_report.chunks_written == 1
    assert plain.search_bm25("FTX trial", limit=1)[0].text == FTX.body
    assert plain_report.index_version.startswith("chunks-sent-w100-o1@")
    assert report.index_version.startswith("chunks-sent-w100-o1-ctx-qwen3.5-4b@")
    assert plain.version() == plain_report.index_version
    assert plain.search_bm25("Quasar", limit=1) == []
    assert context.search_bm25("Quasar", limit=1)[0].context
    with pytest.raises(IndexMismatch, match="context_model"):
        ingest([FTX], plain, settings, contextualiser=writer)
    with pytest.raises(IndexMismatch, match="context_model"):
        ingest([FTX], context, settings)
    with pytest.raises(IndexMismatch, match="context_model"):
        ingest(
            [FTX],
            context,
            settings,
            contextualiser=contextualiser(
                tmp_path / "responses.sqlite3", model, model="qwen3.5:9b"
            ),
        )
    assert len(model.prompts) == 1


@pytest.mark.parametrize(
    ("context_size", "revision"), [(12288, "revision-one"), (8192, "revision-two")]
)
def test_cache_separates_context_models_by_options_and_revision(
    tmp_path: Path, context_size: int, revision: str
):
    client = QdrantClient(":memory:")
    path = tmp_path / "responses.sqlite3"
    index = ChunkIndex(client, "first")
    first = ContextGenerator()
    ingest([FTX], index, SMALL, contextualiser=contextualiser(path, first))
    second = ContextGenerator(revision=revision)
    other = contextualiser(path, second, context_size=context_size)

    with pytest.raises(IndexMismatch, match="context_build"):
        ingest([FTX], index, SMALL, contextualiser=other)
    assert second.prompts == []
    result = ingest([FTX], ChunkIndex(client, "second"), SMALL, contextualiser=other)

    assert second.prompts == first.prompts
    assert result.context is not None and result.context.cache_hits == 0


def test_resuming_a_partly_written_article_rebuilds_it_from_cached_context(tmp_path: Path):
    client = QdrantClient(":memory:")
    index = ChunkIndex(client, "context")
    path = tmp_path / "responses.sqlite3"
    first = ingest([FTX], index, SMALL, contextualiser=contextualiser(path, ContextGenerator()))
    points, _ = client.scroll("context", with_payload=False)
    client.delete("context", points_selector=models.PointIdsList(points=[points[0].id]))
    assert index.count() == first.chunks_written - 1
    second = ContextGenerator()

    restored = ingest([FTX], index, SMALL, contextualiser=contextualiser(path, second))

    assert restored.changed == (FTX.id,)
    assert second.prompts == []
    assert restored.context is not None and restored.context.cache_hits == first.chunks_written
    assert restored.index_version == first.index_version
    assert index.count() == first.chunks_written
