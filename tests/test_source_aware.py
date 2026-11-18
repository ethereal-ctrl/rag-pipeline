"""Source-aware retrieval and the decisive answer prompt, tested at the Pipeline Variant seam."""

from dataclasses import replace
from typing import Literal

from conftest import FTX
from fakes import FakeModels, GroundedFakeGenerator
from qdrant_client import QdrantClient

from rag_pipeline.chunking import ChunkingSettings
from rag_pipeline.corpus import article_id_for
from rag_pipeline.index import ChunkIndex
from rag_pipeline.ingest import ingest
from rag_pipeline.models import GeneratorSpec
from rag_pipeline.variants import (
    GenerationConfig,
    IndexConfig,
    RetrievalConfig,
    SourceAwareConfig,
    VariantConfig,
    build_variant,
)

VERGE_FACT = "A former colleague told the jury about the founder"


def _crunch(n: int):
    url = f"https://fixture.test/crunch-{n}"
    return replace(
        FTX,
        id=article_id_for(url),
        url=url,
        source="TechCrunch",
        title=f"FTX trial update {n}",
        body=f"FTX trial FTX trial fraud FTX trial day {n} of the FTX trial.",
    )


VERGE = replace(
    FTX,
    id=article_id_for("https://fixture.test/verge"),
    url="https://fixture.test/verge",
    source="The Verge",
    title="Inside the courtroom",
    body=f"{VERGE_FACT} of FTX on Wednesday afternoon in Manhattan.",
)
QUESTION = "Do TechCrunch and The Verge both report on the FTX trial?"


def _config(
    source_aware: SourceAwareConfig | None,
    prompt: Literal["strict", "decisive"] = "strict",
) -> VariantConfig:
    return VariantConfig(
        name="bm25-sources",
        index=IndexConfig(collection="sources"),
        retrieval=RetrievalConfig(mode="bm25", top_k=3, source_aware=source_aware),
        generation=GenerationConfig(
            generator=GeneratorSpec(provider="ollama", model="fake"),
            context_chunks=2,
            prompt=prompt,
        ),
    )


def _index() -> ChunkIndex:
    index = ChunkIndex(QdrantClient(":memory:"), collection="sources")
    ingest([*(_crunch(n) for n in range(5)), VERGE], index, ChunkingSettings())
    return index


def test_a_source_named_in_the_question_reaches_the_generator():
    index = _index()
    generator = GroundedFakeGenerator({VERGE_FACT: "Yes"})
    plain = build_variant(_config(None), index=index, models=FakeModels(generator))
    assert {c.source for c in plain.retrieve(QUESTION)} == {"TechCrunch"}
    assert plain.answer(QUESTION).abstained

    aware = build_variant(
        _config(SourceAwareConfig(per_source=5)), index=index, models=FakeModels(generator)
    )
    answer = aware.answer(QUESTION)
    assert not answer.abstained and answer.text == "Yes"
    assert {c.source for c in answer.retrieved[:2]} == {"TechCrunch", "The Verge"}


def test_a_question_naming_no_source_retrieves_as_before():
    index = _index()
    models = FakeModels(GroundedFakeGenerator({}))
    question = "What happened on day 3 of the FTX trial?"
    plain = build_variant(_config(None), index=index, models=models).retrieve(question)
    aware = build_variant(_config(SourceAwareConfig()), index=index, models=models).retrieve(
        question
    )
    assert [c.chunk_id for c in aware] == [c.chunk_id for c in plain]


def test_the_decisive_prompt_is_sent_only_when_configured():
    index = _index()
    generator = GroundedFakeGenerator({})
    systems: list[str] = []
    original = generator.complete

    def record(system: str, prompt: str):
        systems.append(system)
        return original(system, prompt)

    generator.complete = record  # type: ignore[method-assign]
    prompts: tuple[Literal["strict", "decisive"], ...] = ("strict", "decisive")
    for prompt in prompts:
        build_variant(_config(None, prompt), index=index, models=FakeModels(generator)).answer(
            QUESTION
        )
    assert "published" not in systems[0]
    assert "published" in systems[1] and "Insufficient information." in systems[1]


def test_default_settings_keep_existing_run_configs_unchanged():
    config = _config(None).run_config()
    assert "source_aware" not in config["retrieval"]
    assert "prompt" not in config["generation"]
    changed = _config(SourceAwareConfig(), "decisive").run_config()
    assert changed["retrieval"]["source_aware"] == {"per_source": 10}
    assert changed["generation"]["prompt"] == "decisive"
