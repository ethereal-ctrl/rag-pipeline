import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from conftest import ARTICLES, FTX_QUERY
from fakes import ConceptFakeEmbedder, FakeModels, GroundedFakeGenerator, ScriptedVariant
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_pipeline.cache import cache_key
from rag_pipeline.corpus import BenchmarkQuery, Evidence
from rag_pipeline.evaluation import RunStore, TestSplitAlreadyScored, evaluate
from rag_pipeline.index import ChunkIndex
from rag_pipeline.ingest import ChunkingSettings, ingest
from rag_pipeline.models import EmbedderSpec
from rag_pipeline.router import FeatureSettings, LogisticModel, RouterArtifact
from rag_pipeline.router_training import RouterTrainingConfig, train_router
from rag_pipeline.tracing import StageTrace
from rag_pipeline.variants import Answer, VariantConfig, build_variant

EMBEDDER = EmbedderSpec(provider="ollama", model="bge-m3")


class RoutingEmbedder(ConceptFakeEmbedder):
    def embed_query_with_usage(self, text: str) -> tuple[list[float], int]:
        return [float(text == FTX_QUERY.question)], 7


def candidate_files(tmp_path: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for name in ("closed-book", "bm25"):
        config = VariantConfig.from_yaml(Path(f"configs/variants/{name}.yaml"))
        path = tmp_path / f"{name}.json"
        path.write_text(config.model_dump_json(), "utf-8")
        paths.append(path)
    return tuple(paths)


def router_config(paths: tuple[Path, ...], artifact: Path) -> VariantConfig:
    data: dict[str, Any] = VariantConfig.from_yaml(
        Path("configs/variants/closed-book.yaml")
    ).model_dump()
    data.update(
        name="router",
        index={"collection": "fixture", "embedder": EMBEDDER.model_dump()},
        router={"artifact": artifact, "candidates": paths},
    )
    return VariantConfig.model_validate(data)


def test_router_routes_escalates_and_scores_mixed_retrieval(index: ChunkIndex, tmp_path: Path):
    paths = candidate_files(tmp_path)
    artifact_path = tmp_path / "trained.json"
    artifact = RouterArtifact(
        feature_settings=FeatureSettings(embedder=EMBEDDER, embedding_dimensions=1),
        candidate_order=("closed-book", "bm25"),
        candidate_config_hashes=tuple(
            cache_key(VariantConfig.from_yaml(path).run_config()) for path in paths
        ),
        classifier=LogisticModel(
            classes=(0, 1),
            coefficients=((4.0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
            intercepts=(0, 0),
        ),
        threshold=0.8,
        training_runs=("dev-closed", "dev-bm25"),
        training_data_hash="fixture-hash",
    )
    artifact.save(artifact_path)
    assert RouterArtifact.load(artifact_path) == artifact
    assert json.loads(artifact_path.read_text())["candidate_order"] == ["closed-book", "bm25"]
    ingest(ARTICLES, index, ChunkingSettings(max_words=100))
    models = FakeModels(GroundedFakeGenerator({"FTX": "Sam Bankman-Fried"}))
    exporter = InMemorySpanExporter()
    tracer = TracerProvider()
    tracer.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        variant = build_variant(
            router_config(paths, artifact_path),
            index=None,
            models=models,
            tracer=tracer.get_tracer(__name__),
            router_embedder=RoutingEmbedder(),
            index_for=lambda _: index,
        )
        assert len(models.requested) == 2
        answer = variant.answer(FTX_QUERY.question)
        escalated = replace(FTX_QUERY, id="q_low", question="Who did FTX prosecutors charge?")
        store = RunStore(tmp_path / "runs")
        result = evaluate(variant, [FTX_QUERY, escalated], "test", store)
        with pytest.raises(TestSplitAlreadyScored):
            evaluate(variant, [FTX_QUERY, escalated], "test", store)
        spans = exporter.get_finished_spans()
    finally:
        tracer.shutdown()

    assert answer.chosen_variant == "closed-book"
    assert answer.text == "Sam Bankman-Fried"
    assert answer.trace[0].name == "route" and answer.trace[0].input_tokens == 7
    assert answer.stage_seconds["route"] >= 0
    assert len(models.requested) == 2
    first, second = result.records
    assert first.chosen_variant == "closed-book" and first.hit_at_4 is None
    assert second.chosen_variant == "bm25" and second.hit_at_4 is True
    assert result.metrics["hits@4"] == 1.0
    assert result.metrics["routed_share[closed-book]"] == 0.5
    assert result.metrics["routed_share[bm25]"] == 0.5
    assert result.intervals["routed_share[bm25]"] == (0.0, 1.0)
    assert RunStore(tmp_path / "runs").load(result.run_id).records == result.records
    attrs = next(span.attributes for span in spans if span.name == "route")
    assert attrs is not None and attrs["rag.chosen_variant"] == "closed-book"


TRAINING_QUERIES = [
    BenchmarkQuery(
        f"q_{i:02}",
        "Cheap question 2023" if i < 8 else f"Expensive question {i}",
        "Yes",
        "comparison_query" if i % 2 else "inference_query",
        (Evidence("fixture", "fixture fact"),),
    )
    for i in range(16)
]


class TrainingEmbedder(ConceptFakeEmbedder):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [6.0 if text.startswith("Cheap") else -6.0]


def store_candidate_run(
    store: RunStore,
    config: VariantConfig,
    queries: list[BenchmarkQuery],
    split: str = "dev",
    *,
    retrieval_only: bool = False,
) -> str:
    cheap = config.name == "closed-book"
    script = {
        q.question: (
            "Yes" if q.id != "q_15" and (not cheap or q.question.startswith("Cheap")) else "No",
            ["fixture fact"],
        )
        for q in queries
    }

    class Candidate(ScriptedVariant):
        def answer(self, question: str) -> Answer:
            result = super().answer(question)
            seconds = 1.0 if cheap else 3.0
            tokens = 100 if cheap else 1000
            return replace(
                result,
                input_tokens=tokens,
                output_tokens=0,
                seconds=seconds,
                trace=(StageTrace("generate", seconds, tokens, 0, "qwen3.5:4b"),),
            )

    candidate = Candidate(config.name, script)
    candidate.config = config
    return evaluate(candidate, queries, split, store, retrieval_only=retrieval_only).run_id


def arrange_training(
    tmp_path: Path,
    store: RunStore,
    queries: list[BenchmarkQuery],
) -> tuple[Path, dict[str, str | None]]:
    paths = candidate_files(tmp_path)
    config = router_config(paths, tmp_path / "artifact.json")
    config_path = tmp_path / "router.json"
    config_path.write_text(config.model_dump_json(), "utf-8")
    candidates = [VariantConfig.from_yaml(path) for path in paths]
    return config_path, {c.name: store_candidate_run(store, c, queries) for c in candidates}


def test_training_reports_hand_computed_dev_outcomes_and_ignores_query_type(tmp_path: Path):
    store = RunStore(tmp_path / "runs")
    path, runs = arrange_training(tmp_path, store, TRAINING_QUERIES)
    embedder = TrainingEmbedder()
    settings = RouterTrainingConfig(variant=path, runs=runs, folds=4)
    result = train_router(settings, store, embedder)

    report = result.report
    assert report.label_distribution == {"closed-book": 8, "bm25": 8}
    assert report.classifier_accuracy == 1
    assert report.confusion_matrix == ((8, 0), (0, 8))
    assert report.outcomes["routed"]["accuracy"].value == 15 / 16
    assert report.outcomes["routed"]["latency_mean"].value == 2
    assert report.outcomes["routed"]["cost_per_question"].value == pytest.approx(0.000055)
    assert report.outcomes["most_expensive"]["latency_mean"].value == 3
    assert report.outcomes["most_expensive"]["accuracy"].value == 15 / 16
    assert report.outcomes["oracle"]["latency_mean"].value == 2
    assert report.outcomes["oracle"]["accuracy"].value == 15 / 16
    assert report.candidates["bm25"]["cost_per_question"].value == pytest.approx(0.0001)
    assert report.outcomes["routed"]["accuracy"].interval[0] <= 15 / 16
    assert embedder.queries == [q.question for q in TRAINING_QUERIES]
    artifact_path, json_path, markdown_path = result.save(tmp_path / "runs" / "router")
    assert RouterArtifact.load(artifact_path) == result.artifact
    assert json.loads(json_path.read_text())["label_distribution"] == report.label_distribution
    assert "## Limits" in markdown_path.read_text()

    altered = [replace(q, query_type="temporal_query") for q in TRAINING_QUERIES]
    _, altered_runs = arrange_training(tmp_path, store, altered)
    altered_result = train_router(
        RouterTrainingConfig(variant=path, runs=altered_runs, folds=4), store, TrainingEmbedder()
    )
    # The two runs differ in their labels' metadata only. Question features, including
    # the duplicate question text, and the fitted model must be exactly the same.
    assert result.artifact.training_data_hash == altered_result.artifact.training_data_hash
    assert result.artifact.classifier == altered_result.artifact.classifier
    assert result.artifact.threshold == altered_result.artifact.threshold


@pytest.mark.parametrize(
    ("kind", "error"),
    [
        ("test", "Dev Split"),
        ("test-400", "Dev Split"),
        ("missing", "bm25.*does-not-exist"),
        ("mismatch", "bm25.*same Benchmark Queries"),
        ("retrieval-only", "bm25.*retrieval-only"),
    ],
)
def test_training_rejects_invalid_candidate_runs(tmp_path: Path, kind: str, error: str):
    store = RunStore(tmp_path / "runs")
    path, runs = arrange_training(tmp_path, store, TRAINING_QUERIES)
    candidate = VariantConfig.from_yaml(tmp_path / "bm25.json")
    if kind == "missing":
        runs["bm25"] = "does-not-exist"
    elif kind == "mismatch":
        runs["bm25"] = store_candidate_run(store, candidate, TRAINING_QUERIES[:-1])
    elif kind == "retrieval-only":
        runs["bm25"] = store_candidate_run(store, candidate, TRAINING_QUERIES, retrieval_only=True)
    else:
        runs["bm25"] = store_candidate_run(store, candidate, TRAINING_QUERIES, kind)
    embedder = TrainingEmbedder()

    with pytest.raises(ValueError, match=error):
        train_router(RouterTrainingConfig(variant=path, runs=runs, folds=4), store, embedder)

    assert embedder.queries == []
