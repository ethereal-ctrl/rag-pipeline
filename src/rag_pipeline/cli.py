"""Command line entry points: `rag data ...`, `rag ingest`, `rag eval`, `rag gate`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from rag_pipeline.cache import CACHE_FILE, ResponseCache
from rag_pipeline.chunking import ChunkingSettings
from rag_pipeline.corpus import (
    DATASET_LICENSE,
    BenchmarkQuery,
    fetch_dataset,
    load_articles,
    load_queries,
)
from rag_pipeline.evaluation import (
    RunStore,
    TestSplitAlreadyScored,
    compare_runs,
    evaluate,
    export_records,
)
from rag_pipeline.gate import GateConfig, GateReport, check_minimums, retrieval_gate
from rag_pipeline.index import ChunkIndex, IndexMismatch, index_collection
from rag_pipeline.ingest import IngestReport, ingest
from rag_pipeline.models import EmbedderSpec, LocalModels
from rag_pipeline.report import metric_label, metric_number, results_table
from rag_pipeline.splits import (
    DEFAULT_SEED,
    DEFAULT_SIZES,
    SPLIT_NAMES,
    TEST_SPLITS,
    Splits,
    load_test_subset,
    make_splits,
)
from rag_pipeline.tracing import cli_tracer
from rag_pipeline.variants import PipelineVariant, VariantConfig, build_variant, plan_remote_run

RAW_DIR = Path("data/raw")
SPLITS_FILE = Path("data/splits.json")
TEST_SUBSET_FILE = Path("data/test-400.json")
RUNS_DIR = Path("runs")
GATE_FILE = Path("configs/gate/gate.yaml")

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data", help="Fetch the pinned dataset and build splits.")

QdrantUrl = Annotated[str, typer.Option(envvar="QDRANT_URL")]


@data_app.command("fetch")
def data_fetch() -> None:
    """Download the pinned MultiHop-RAG files and verify their checksums."""
    fetch_dataset(RAW_DIR)
    typer.echo(f"Dataset ready in {RAW_DIR} ({DATASET_LICENSE}).")


@data_app.command("splits")
def data_splits(force: bool = False) -> None:
    """Create the fixed splits. Refuses to overwrite the committed file unless --force."""
    if SPLITS_FILE.exists() and not force:
        raise typer.BadParameter(f"{SPLITS_FILE} exists; splits are fixed (ADR 0005)")
    make_splits(load_queries(RAW_DIR), DEFAULT_SIZES, DEFAULT_SEED).save(SPLITS_FILE)
    typer.echo(f"Wrote {SPLITS_FILE}.")


def _print_ingest(report: IngestReport) -> None:
    typer.echo(f"Index version {report.index_version}")
    typer.echo(
        f"  {report.articles} Articles: {len(report.added)} added, {len(report.changed)} changed, "
        f"{len(report.unchanged)} unchanged, {len(report.deleted)} deleted; "
        f"{report.chunks_written} Chunks written."
    )


@app.command("ingest")
def ingest_command(
    max_words: int = 200,
    overlap_sentences: int = 1,
    embedder: Annotated[str | None, typer.Option(help="Embedding model for dense vectors.")] = None,
    embedder_provider: Annotated[
        str, typer.Option(help="ollama or sentence-transformers.")
    ] = "ollama",
    collection: Annotated[
        str | None, typer.Option(help="Override the name derived from settings and embedder.")
    ] = None,
    qdrant_url: QdrantUrl = "http://localhost:6333",
) -> None:
    """Chunk and index the Corpus incrementally (BM25 always; dense vectors with --embedder)."""
    settings = ChunkingSettings(max_words, overlap_sentences)
    dense = (
        LocalModels().embedder_for(
            EmbedderSpec.model_validate({"provider": embedder_provider, "model": embedder})
        )
        if embedder
        else None
    )
    name = collection or index_collection(settings, embedder)
    index = ChunkIndex(QdrantClient(url=qdrant_url), name, dense)
    try:
        report = ingest(load_articles(RAW_DIR), index, settings)
    except IndexMismatch as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    _print_ingest(report)


@app.command("eval")
def eval_command(
    ctx: typer.Context,
    variant_file: Path,
    split: str = "smoke",
    limit: Annotated[int | None, typer.Option(help="Only the first N queries (debugging).")] = None,
    allow_rescore: Annotated[
        bool, typer.Option(help="Re-score a variant on the Test Split (ADR 0005).")
    ] = False,
    retrieval_only: Annotated[
        bool, typer.Option(help="Score retrieval alone, without generating (not on test).")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option(help="Bypass stored model responses for this run.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Estimate remote calls without running any models.")
    ] = False,
    qdrant_url: QdrantUrl = "http://localhost:6333",
) -> None:
    """Evaluate a Pipeline Variant on a split and store the run (resumes if interrupted)."""
    splits = (*SPLIT_NAMES, "test-400")
    if split not in splits:
        raise typer.BadParameter(f"split must be one of {splits}")
    if split in TEST_SPLITS and limit is not None:
        raise typer.BadParameter("--limit is not allowed on the Test Split")
    if split in TEST_SPLITS and retrieval_only:
        raise typer.BadParameter("--retrieval-only is not allowed on the Test Split")
    config = VariantConfig.from_yaml(variant_file)
    try:
        if dry_run:
            remote = config.generation.generator.provider == "groq" and not retrieval_only
            queries = _split_queries(split)[:limit] if remote else []
            _print_remote_plan(config, queries, qdrant_url, retrieval_only, no_cache)
            return
        queries = _split_queries(split)[:limit]
        tracer = ctx.with_resource(cli_tracer())
        models = LocalModels()
        variant = build_variant(
            config,
            index=_index_for(config, models, qdrant_url),
            models=models,
            tracer=tracer,
            cache=None if no_cache else ResponseCache(),
        )
        result = evaluate(
            variant,
            queries,
            split,
            RunStore(RUNS_DIR),
            allow_rescore=allow_rescore,
            retrieval_only=retrieval_only,
        )
    except (TestSplitAlreadyScored, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Run {result.run_id}")
    _print_metrics(result.metrics, result.intervals)


def _split_queries(split: str):
    by_id = {q.id: q for q in load_queries(RAW_DIR)}
    splits = Splits.load(SPLITS_FILE)
    ids = load_test_subset(TEST_SUBSET_FILE, splits) if split == "test-400" else splits.ids(split)
    return [by_id[i] for i in ids]


def _print_remote_plan(
    config: VariantConfig,
    queries: list[BenchmarkQuery],
    qdrant_url: str,
    retrieval_only: bool,
    no_cache: bool,
) -> None:
    remote = config.generation.generator.provider == "groq" and not retrieval_only
    cache = ResponseCache() if remote and not no_cache and CACHE_FILE.exists() else None
    version = None
    if cache is not None and config.uses_retrieval and config.index is not None:
        client = QdrantClient(url=qdrant_url, timeout=2, check_compatibility=False)
        try:
            version = ChunkIndex(client, config.index.collection).version()
        except ResponseHandlingException:
            typer.echo("Index version unavailable. Cached retrieval prompts could not be checked.")
        finally:
            client.close()
    plan = plan_remote_run(
        config, queries, cache, retrieval_only=retrieval_only, index_version=version
    )
    typer.echo(f"Remote calls before cache  {plan.calls}")
    typer.echo(f"Already cached             {plan.cached}")
    typer.echo(f"Remote calls needed       {plan.calls - plan.cached}")
    typer.echo(f"Estimated quota days       {plan.days}")


def _index_for(config: VariantConfig, models: LocalModels, qdrant_url: str) -> ChunkIndex | None:
    if config.index is None:
        return None
    embedder = models.embedder_for(config.index.embedder) if config.index.embedder else None
    return ChunkIndex(QdrantClient(url=qdrant_url), config.index.collection, embedder)


def _print_metrics(metrics: dict[str, float], intervals: dict[str, tuple[float, float]]) -> None:
    for name, value in metrics.items():
        if name == "queries":
            typer.echo(f"  {name:<34} {int(value)}")
            continue
        low, high = intervals.get(name, (float("nan"), float("nan")))
        estimate, lower, upper = (metric_number(name, number) for number in (value, low, high))
        typer.echo(f"  {metric_label(name):<34} {estimate}  [{lower}, {upper}]")


@app.command("compare")
def compare_command(baseline_run: str, candidate_run: str) -> None:
    """Paired comparison of two runs; '*' marks a significant change (95% interval)."""
    store = RunStore(RUNS_DIR)
    deltas = compare_runs(store.load(baseline_run), store.load(candidate_run))
    typer.echo(f"  {'metric':<34} {'base':>6} {'cand':>6} {'delta':>7}  95% interval")
    for d in deltas:
        mark = "*" if d.significant else " "
        base, candidate, low, high = (
            metric_number(d.metric, value, signed=signed)
            for value, signed in (
                (d.baseline, False),
                (d.candidate, False),
                (d.low, True),
                (d.high, True),
            )
        )
        delta = metric_number(d.metric, d.delta, signed=True)
        typer.echo(
            f"{mark} {metric_label(d.metric):<34} {base:>6} {candidate:>6} {delta:>7}"
            f"  [{low}, {high}]"
        )


@app.command("table")
def table_command(
    run_ids: list[str],
    metrics: Annotated[str, typer.Option(help="Comma-separated metric names, in column order.")] = (
        "hits@4,hits@10,mrr@10,evidence_recall@10,accuracy,"
        "latency_p50,latency_p95,cost_per_question"
    ),
    output: Annotated[Path | None, typer.Option(help="Also write the table here.")] = None,
) -> None:
    """Markdown table of runs with 95% intervals, for the report."""
    store = RunStore(RUNS_DIR)
    table = results_table([store.load(r) for r in run_ids], metrics.split(","))
    typer.echo(table)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table + "\n", "utf-8")


@app.command("export")
def export_command(run_id: str, output: Path | None = None) -> None:
    """Export a run's per-query records as CSV for error analysis."""
    path = export_records(RunStore(RUNS_DIR).load(run_id), output or RUNS_DIR / f"{run_id}.csv")
    typer.echo(f"Wrote {path}.")


def _report_gate(title: str, report: GateReport) -> None:
    typer.echo(title)
    for variant, metrics in report.metrics.items():
        typer.echo(f"  {variant}")
        for name, value in metrics.items():
            typer.echo(f"    {name:<32} {value:.3f}")
    for failure in report.failures:
        typer.echo(f"  FAIL {failure}", err=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:  # rendered on the GitHub Actions run page
        lines = [f"### {title}", "", "| variant | metric | value |", "|---|---|---:|"]
        lines += [
            f"| {variant} | {name} | {value:.3f} |"
            for variant, metrics in report.metrics.items()
            for name, value in metrics.items()
        ]
        lines += ["", *[f"- :x: {failure}" for failure in report.failures], ""]
        with Path(summary).open("a", encoding="utf-8") as out:
            out.write("\n".join(lines) + "\n")


@app.command("gate")
def gate_command(
    ctx: typer.Context,
    gate_file: Path = GATE_FILE,
    answers: Annotated[
        bool, typer.Option(help="Also run the generator and check answer-level metrics.")
    ] = False,
    qdrant_url: QdrantUrl = "http://localhost:6333",
) -> None:
    """Check Smoke Set metrics against the committed thresholds; exit 1 on any drop."""
    tracer = ctx.with_resource(cli_tracer()) if answers else None
    gate = GateConfig.from_yaml(gate_file)
    configs = [VariantConfig.from_yaml(path) for path in gate.variants]
    models = LocalModels()
    variants: list[PipelineVariant] = []
    # Build each collection once, with the embedder of whichever variant needs dense vectors.
    by_collection: dict[str, VariantConfig] = {}
    for config in configs:
        if config.index is not None and (
            config.index.collection not in by_collection or config.index.embedder
        ):
            by_collection[config.index.collection] = config
    for config in by_collection.values():
        index = _index_for(config, models, qdrant_url)
        if index is not None:  # incremental: a no-op when the Corpus is unchanged
            _print_ingest(ingest(load_articles(RAW_DIR), index, ChunkingSettings()))
    for config in configs:
        # Every variant on a collection queries it with the embedder it was built with.
        built = by_collection[config.index.collection] if config.index else config
        variants.append(
            build_variant(
                config, index=_index_for(built, models, qdrant_url), models=models, tracer=tracer
            )
        )
    queries = _split_queries(gate.split)
    retrieval = retrieval_gate(variants, queries, gate.retrieval)
    _report_gate(f"Retrieval gate ({gate.split}, {len(queries)} queries)", retrieval)
    failed = not retrieval.passed
    if answers:
        store = RunStore(RUNS_DIR)
        metrics = {v.config.name: evaluate(v, queries, gate.split, store).metrics for v in variants}
        answer_report = check_minimums(metrics, gate.answers)
        _report_gate(f"Answer gate ({gate.split})", answer_report)
        failed = failed or not answer_report.passed
    if failed:
        raise typer.Exit(code=1)
    typer.echo("Gate passed.")
