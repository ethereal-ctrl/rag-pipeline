"""Command line entry points: `rag data ...`, `rag ingest`, `rag eval`, `rag gate`."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from httpx import HTTPError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from typer._click.core import Context as ClickContext
from typer.core import TyperGroup

from rag_pipeline.cache import CACHE_FILE, ResponseCache
from rag_pipeline.chunk_context import (
    CONTEXT_OUTPUT_TOKENS,
    DEFAULT_ARTICLE_TOKENS,
    ModelContextualiser,
)
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
from rag_pipeline.judge_calibration import calibration_report, export_calibration
from rag_pipeline.judge_evaluation import (
    CitedChunkLookup,
    judge_cases,
    judge_run,
    load_answer_run,
    plan_judging,
    with_judge_metrics,
)
from rag_pipeline.models import EmbedderSpec, GeneratorSpec, LocalModels
from rag_pipeline.report import metric_label, metric_number, results_table
from rag_pipeline.router_training import RouterTrainingConfig, train_router
from rag_pipeline.splits import (
    DEFAULT_SEED,
    DEFAULT_SIZES,
    SPLIT_NAMES,
    TEST_SPLITS,
    Splits,
    load_judge_sample,
    load_test_subset,
    make_splits,
)
from rag_pipeline.tracing import cli_tracer
from rag_pipeline.variants import (
    PipelineVariant,
    VariantConfig,
    build_variant,
    plan_remote_run,
    router_candidates,
)

RAW_DIR = Path("data/raw")
SPLITS_FILE = Path("data/splits.json")
TEST_SUBSET_FILE = Path("data/test-400.json")
JUDGE_SAMPLE_FILE = Path("data/judge-300.json")
CALIBRATION_DIR = Path("data/judge-calibration")
RUNS_DIR = Path("runs")
GATE_FILE = Path("configs/gate/gate.yaml")

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data", help="Fetch the pinned dataset and build splits.")
router_app = typer.Typer(no_args_is_help=True)
app.add_typer(router_app, name="router", help="Train the Router from saved Dev Split runs.")


class _JudgeGroup(TyperGroup):
    def parse_args(self, ctx: ClickContext, args: list[str]) -> list[str]:
        # Let the user pass a run ID directly while keeping calibration as a subcommand.
        if args and args[0] not in {"calibration", "run", "--help", "-h"}:
            args = ["run", *args]
        return super().parse_args(ctx, args)


judge_app = typer.Typer(cls=_JudgeGroup, no_args_is_help=True)
calibration_app = typer.Typer(no_args_is_help=True)
app.add_typer(judge_app, name="judge", help="Grade a stored run with `rag judge RUN`.")
judge_app.add_typer(calibration_app, name="calibration", help="Calibrate the Judge on Dev cases.")

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
    context = report.context
    if context is not None:
        typer.echo(
            f"  {context.chunks_contextualised} Chunks contextualised "
            f"({context.empty_contexts} empty), "
            f"{context.cache_hits} cache hits, {context.generation_seconds:.1f} generation seconds."
        )
        typer.echo(
            f"  {context.input_tokens} input tokens, {context.output_tokens} output tokens, "
            f"{len(context.truncated_articles)} truncated Articles."
        )
        if context.notional_cost_usd is not None:
            typer.echo(f"  Chunk Context cost (notional USD)  {context.notional_cost_usd:.6g}")


def _save_ingest_report(report: IngestReport, collection: str) -> Path:
    folder = RUNS_DIR / "ingest"
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    path = folder / f"{Path(collection).name}-{timestamp}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


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
    chunk_context: Annotated[
        str | None, typer.Option(help="Local Ollama model that writes Chunk Context.")
    ] = None,
    context_size: Annotated[
        int, typer.Option(min=1, help="Chunk Context model's context window.")
    ] = 8192,
    article_tokens: Annotated[
        int, typer.Option(min=1, help="Maximum estimated tokens for the Article prefix body.")
    ] = DEFAULT_ARTICLE_TOKENS,
) -> None:
    """Chunk and index the Corpus incrementally (BM25 always; dense vectors with --embedder)."""
    settings = ChunkingSettings(max_words, overlap_sentences)
    models = LocalModels()
    dense = (
        models.embedder_for(
            EmbedderSpec.model_validate({"provider": embedder_provider, "model": embedder})
        )
        if embedder
        else None
    )
    name = collection or index_collection(settings, embedder, context_model=chunk_context)
    index = ChunkIndex(QdrantClient(url=qdrant_url), name, dense)
    try:
        writer = None
        if chunk_context is not None:
            spec = GeneratorSpec(
                provider="ollama",
                model=chunk_context,
                context_size=context_size,
                max_output_tokens=CONTEXT_OUTPUT_TOKENS,
            )
            writer = ModelContextualiser(
                spec,
                models.generator_for,
                ResponseCache(),
                article_tokens=article_tokens,
            )
        report = ingest(load_articles(RAW_DIR), index, settings, contextualiser=writer)
    except (IndexMismatch, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    _print_ingest(report)
    if report.context is not None:
        path = _save_ingest_report(report, name)
        typer.echo(f"Wrote {path}.")


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
            candidates = (config,) if config.router is None else router_candidates(config.router)
            remote = not retrieval_only and any(
                candidate.generation.generator.provider == "groq" for candidate in candidates
            )
            queries = _split_queries(split)[:limit] if remote else []
            _print_remote_plan(config, queries, qdrant_url, retrieval_only, no_cache)
            return
        queries = _split_queries(split)[:limit]
        tracer = ctx.with_resource(cli_tracer())
        models = LocalModels()
        router_embedder = None
        if config.router is not None:
            assert config.index is not None and config.index.embedder is not None
            router_embedder = models.embedder_for(config.index.embedder)
        variant = build_variant(
            config,
            index=_index_for(config, models, qdrant_url) if config.router is None else None,
            models=models,
            tracer=tracer,
            cache=None if no_cache else ResponseCache(),
            router_embedder=router_embedder,
            index_for=lambda candidate: _index_for(candidate, models, qdrant_url),
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
    remote = (
        config.router is None
        and config.generation.generator.provider == "groq"
        and not retrieval_only
    )
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
    label = "Remote calls before cache"
    if plan.upper_bound:
        bound = "across Router candidates" if config.router is not None else "at max_steps"
        label += f" (upper bound {bound})"
    typer.echo(f"{label}  {plan.calls}")
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


@router_app.command("train")
def router_train(router_config: Path) -> None:
    """Fit the Router and write its JSON artifact and Dev cross-validation report."""
    try:
        settings = RouterTrainingConfig.from_yaml(router_config)
        variant = VariantConfig.from_yaml(settings.variant)
        if variant.router is None or variant.index is None or variant.index.embedder is None:
            raise ValueError(f"{settings.variant} needs a Router variant with an index embedder")
        embedder = LocalModels().embedder_for(variant.index.embedder)
        result = train_router(settings, RunStore(RUNS_DIR), embedder, cache=ResponseCache())
        paths = result.save(RUNS_DIR / "router")
    except (OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    report = result.report
    typer.echo(
        f"{report.queries} Dev queries, {report.folds} folds, "
        f"classifier accuracy {report.classifier_accuracy:.3f}, threshold {report.threshold:.2f}"
    )
    for name, metrics in report.candidates.items():
        cost = metrics.get("cost_per_question")
        notional = "unavailable" if cost is None else f"${cost.value:.6g}"
        typer.echo(
            f"  {name:<20} {report.label_distribution[name]} labels, "
            f"{metrics['latency_mean'].value:.3f} s mean latency, {notional} notional per question"
        )
    for path in paths:
        typer.echo(f"Wrote {path}")


def _judge_sample() -> tuple[str, ...]:
    test_subset = load_test_subset(TEST_SUBSET_FILE, Splits.load(SPLITS_FILE))
    return load_judge_sample(JUDGE_SAMPLE_FILE, test_subset)


@judge_app.command("run")
def judge_command(
    run_id: str,
    dry_run: Annotated[
        bool, typer.Option(help="Estimate the Groq workload without calls.")
    ] = False,
    shared_tokens: Annotated[
        int, typer.Option(min=0, help="Other planned Groq tokens sharing the daily budget.")
    ] = 0,
    shared_calls: Annotated[
        int, typer.Option(min=0, help="Other planned Groq calls sharing the daily budget.")
    ] = 0,
) -> None:
    """Grade the fixed Judge sample in a completed run."""
    store = RunStore(RUNS_DIR)
    lookup = CitedChunkLookup(RAW_DIR)
    try:
        sample = _judge_sample()
        if dry_run:
            cases = judge_cases(load_answer_run(store, run_id), sample, lookup)
            cache = ResponseCache(CACHE_FILE) if CACHE_FILE.exists() else None
            plan = plan_judging(
                cases, cache, shared_tokens=shared_tokens, shared_calls=shared_calls
            )
            typer.echo(f"Sampled answers              {plan.sampled}")
            typer.echo(f"Calls before cache           {plan.calls}")
            typer.echo(f"Already cached               {plan.cached}")
            typer.echo(f"Calls needed                 {plan.calls - plan.cached}")
            typer.echo(f"Estimated tokens before cache  {plan.estimated_tokens}")
            typer.echo(f"Estimated cached tokens      {plan.cached_tokens}")
            typer.echo(f"Estimated tokens needed      {plan.estimated_tokens - plan.cached_tokens}")
            typer.echo(f"Estimated Judge quota days  {plan.days}")
            typer.echo(f"Estimated shared quota days  {plan.shared_days}")
            typer.echo(
                f"Judge and GPT-OSS-120B test-400 runs share {plan.tokens_per_day:,} tokens/day."
            )
            return
        if shared_tokens or shared_calls:
            raise ValueError("--shared-tokens and --shared-calls are only used with --dry-run.")
        judge = LocalModels().judge_for(ResponseCache(CACHE_FILE))
        report = judge_run(store, run_id, sample, judge, chunks=lookup)
    except (OSError, ValueError, HTTPError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Graded {report.sampled} stored answers. Malformed verdicts {report.malformed}.")
    _print_metrics(report.metrics, report.intervals)


@calibration_app.command("export")
def judge_calibration_export(run_ids: list[str] | None = None) -> None:
    """Export cases from completed Dev Split runs for manual labelling."""
    store = RunStore(RUNS_DIR)
    try:
        cases = export_calibration(
            store,
            run_ids if run_ids is not None else store.completed_answer_ids("dev"),
            CALIBRATION_DIR,
            chunks=CitedChunkLookup(RAW_DIR),
        )
    except (OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Exported {len(cases)} Dev cases and LABELLING.md to {CALIBRATION_DIR}.")


@calibration_app.command("report")
def judge_calibration_report() -> None:
    """Measure Groq Judge agreement against the exported Dev labels."""
    try:
        judge = LocalModels().judge_for(ResponseCache(CACHE_FILE))
        report = calibration_report(CALIBRATION_DIR, RUNS_DIR / "judge", judge)
    except (OSError, ValueError, HTTPError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Reported agreement for {report.cases} Dev cases in {RUNS_DIR / 'judge'}.")


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
    runs = [with_judge_metrics(store.load(run_id), store) for run_id in run_ids]
    table = results_table(runs, metrics.split(","))
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
