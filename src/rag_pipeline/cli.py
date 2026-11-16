"""Command line entry points: `rag data ...`, `rag ingest`, `rag eval`, `rag gate`."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import typer
from httpx import HTTPError
from opentelemetry.trace import Tracer
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
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
    Article,
    BenchmarkQuery,
    fetch_dataset,
    load_articles,
    load_queries,
)
from rag_pipeline.demo import (
    DEFAULT_QDRANT_URL,
    VARIANTS_DIR,
    IndexBuild,
    answer_variants,
    demo_variant,
    load_variants,
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
from rag_pipeline.ladder import (
    HeldOutSplit,
    Ladder,
    LadderRung,
    estimate_ladder,
    ladder_status,
    run_ladder,
)
from rag_pipeline.ladder_report import write_ladder_report
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
    RemoteRunPlan,
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
LADDER_FILE = Path("configs/ladder.yaml")
LADDER_REPORT_DIR = Path("docs/report/ladder")

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data", help="Fetch the pinned dataset and build splits.")
router_app = typer.Typer(no_args_is_help=True)
app.add_typer(router_app, name="router", help="Train the Router from saved Dev Split runs.")
ladder_app = typer.Typer(no_args_is_help=True)
app.add_typer(ladder_app, name="ladder", help="Plan, run and report the Ablation Ladder.")


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


def _context_writer(
    models: LocalModels,
    model: str,
    context_size: int = 8192,
    article_tokens: int = DEFAULT_ARTICLE_TOKENS,
) -> ModelContextualiser:
    spec = GeneratorSpec(
        provider="ollama",
        model=model,
        context_size=context_size,
        max_output_tokens=CONTEXT_OUTPUT_TOKENS,
    )
    return ModelContextualiser(
        spec, models.generator_for, ResponseCache(), article_tokens=article_tokens
    )


def _build_index(
    build: IndexBuild,
    client: QdrantClient,
    models: LocalModels,
    articles: list[Article],
    *,
    context_size: int = 8192,
    article_tokens: int = DEFAULT_ARTICLE_TOKENS,
) -> IngestReport:
    embedder = models.embedder_for(build.embedder) if build.embedder else None
    writer = (
        _context_writer(models, build.context_model, context_size, article_tokens)
        if build.context_model
        else None
    )
    cache = ResponseCache() if build.settings.strategy == "semantic" else None
    return ingest(
        articles,
        ChunkIndex(client, build.collection, embedder),
        build.settings,
        contextualiser=writer,
        cache=cache,
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
    chunk_context: Annotated[
        str | None, typer.Option(help="Local Ollama model that writes Chunk Context.")
    ] = None,
    context_size: Annotated[
        int, typer.Option(min=1, help="Chunk Context model's context window.")
    ] = 8192,
    article_tokens: Annotated[
        int, typer.Option(min=1, help="Maximum estimated tokens for the Article prefix body.")
    ] = DEFAULT_ARTICLE_TOKENS,
    strategy: Literal["sentences", "semantic"] = "sentences",
    semantic_percentile: Annotated[int, typer.Option(min=0, max=100)] = 90,
    variant_file: Annotated[
        Path | None, typer.Option(help="Use the index settings from this Pipeline Variant.")
    ] = None,
) -> None:
    """Chunk and index the Corpus incrementally (BM25 always; dense vectors with --embedder)."""
    try:
        if variant_file is None:
            settings = ChunkingSettings(max_words, overlap_sentences, strategy, semantic_percentile)
            spec = (
                EmbedderSpec.model_validate({"provider": embedder_provider, "model": embedder})
                if embedder
                else None
            )
            name = collection or index_collection(settings, embedder, context_model=chunk_context)
            build = IndexBuild(name, spec, settings, chunk_context)
        else:
            config = VariantConfig.from_yaml(variant_file)
            if config.index is None or config.router is not None:
                raise ValueError("Choose a Pipeline Variant with a direct index to ingest.")
            build = IndexBuild.for_variant(config)
        client = QdrantClient(url=qdrant_url, check_compatibility=False)
        try:
            report = _build_index(
                build,
                client,
                LocalModels(),
                load_articles(RAW_DIR),
                context_size=context_size,
                article_tokens=article_tokens,
            )
        finally:
            client.close()
    except (IndexMismatch, OSError, ValueError, HTTPError, ResponseHandlingException) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    _print_ingest(report)
    if report.context is not None:
        path = _save_ingest_report(report, build.collection)
        typer.echo(f"Wrote {path}.")


@app.command("bootstrap")
def bootstrap_command(qdrant_url: QdrantUrl = DEFAULT_QDRANT_URL) -> None:
    """Fetch the pinned Corpus and incrementally build the configured demo index."""
    name = demo_variant()
    try:
        config = load_variants(VARIANTS_DIR).get(name)
        if config is None:
            raise ValueError(f"Unknown RAG_DEMO_VARIANT {name!r}. Check {VARIANTS_DIR}.")
        builds = {
            IndexBuild.for_variant(candidate)
            for candidate in answer_variants(config)
            if candidate.uses_retrieval and candidate.index is not None
        }
        fetch_dataset(RAW_DIR)
        if not builds:
            typer.echo(f"Dataset ready. Variant {name!r} needs no collection.")
            return
        articles = load_articles(RAW_DIR)
        models = LocalModels()
        client = QdrantClient(url=qdrant_url, check_compatibility=False)
        try:
            for build in sorted(builds, key=lambda item: item.collection):
                typer.echo(f"Ingesting {build.collection} for {name}.")
                _print_ingest(_build_index(build, client, models, articles))
        finally:
            client.close()
    except (IndexMismatch, OSError, ValueError, HTTPError, ResponseHandlingException) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@app.command("serve")
def serve_command(
    host: Annotated[str, typer.Option(help="Bind the API to this address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
) -> None:
    """Serve the demo page and Answer API."""
    import uvicorn

    uvicorn.run("rag_pipeline.api:app", host=host, port=port)


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
        bool, typer.Option(help="Score retrieval without forming a final answer (Dev or Smoke).")
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
            remote = any(
                candidate.uses_groq(retrieval_only=retrieval_only) for candidate in candidates
            )
            queries = _split_queries(split)[:limit] if remote else []
            _print_remote_plan(config, queries, qdrant_url, retrieval_only, no_cache)
            return
        queries = _split_queries(split)[:limit]
        tracer = ctx.with_resource(cli_tracer())
        variant = _evaluation_variant(
            config, LocalModels(), qdrant_url, tracer, None if no_cache else ResponseCache()
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


def _split_ids(split: str) -> tuple[str, ...]:
    splits = Splits.load(SPLITS_FILE)
    return load_test_subset(TEST_SUBSET_FILE, splits) if split == "test-400" else splits.ids(split)


def _split_queries(split: str) -> list[BenchmarkQuery]:
    by_id = {q.id: q for q in load_queries(RAW_DIR)}
    return [by_id[i] for i in _split_ids(split)]


def _remote_plan(
    config: VariantConfig,
    queries: list[BenchmarkQuery],
    qdrant_url: str,
    retrieval_only: bool,
    no_cache: bool,
) -> RemoteRunPlan:
    remote = config.router is None and config.uses_groq(retrieval_only=retrieval_only)
    cache = ResponseCache() if remote and not no_cache and CACHE_FILE.exists() else None
    version = None
    if (
        cache is not None
        and not retrieval_only
        and config.generation.generator.provider == "groq"
        and config.uses_retrieval
        and config.index is not None
    ):
        client = QdrantClient(url=qdrant_url, timeout=2, check_compatibility=False)
        try:
            version = ChunkIndex(client, config.index.collection).version()
        except (ResponseHandlingException, UnexpectedResponse):
            typer.echo("Index version unavailable. Cached retrieval prompts could not be checked.")
        finally:
            client.close()
    return plan_remote_run(
        config, queries, cache, retrieval_only=retrieval_only, index_version=version
    )


def _print_remote_plan(
    config: VariantConfig,
    queries: list[BenchmarkQuery],
    qdrant_url: str,
    retrieval_only: bool,
    no_cache: bool,
) -> None:
    plan = _remote_plan(config, queries, qdrant_url, retrieval_only, no_cache)
    label = "Remote calls before cache"
    if plan.upper_bound:
        bound = "across Router candidates" if config.router is not None else "at max_steps"
        label += f" (upper bound {bound})"
    typer.echo(f"{label}  {plan.calls}")
    typer.echo(f"Already cached             {plan.cached}")
    typer.echo(f"Remote calls needed       {plan.calls - plan.cached}")
    typer.echo(f"Estimated quota days       {plan.days}")


def _index_for(config: VariantConfig, models: LocalModels, qdrant_url: str) -> ChunkIndex | None:
    if config.index is None or not config.uses_retrieval:
        return None
    embedder = models.embedder_for(config.index.embedder) if config.index.embedder else None
    return ChunkIndex(QdrantClient(url=qdrant_url), config.index.collection, embedder)


def _evaluation_variant(
    config: VariantConfig,
    models: LocalModels,
    qdrant_url: str,
    tracer: Tracer | None,
    cache: ResponseCache | None,
) -> PipelineVariant:
    router_embedder = None
    if config.router is not None:
        assert config.index is not None and config.index.embedder is not None
        router_embedder = models.embedder_for(config.index.embedder)
    return build_variant(
        config,
        index=_index_for(config, models, qdrant_url),
        models=models,
        tracer=tracer,
        cache=cache,
        router_embedder=router_embedder,
        index_for=lambda candidate: _index_for(candidate, models, qdrant_url),
    )


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


def _ladder_gaps(
    rung: LadderRung,
    client: QdrantClient,
    collections: dict[str, str | None],
) -> tuple[str, ...]:
    gaps: list[str] = []
    if (router := rung.config.router) is not None and not router.artifact.is_file():
        gaps.append(f"missing Router artifact {router.artifact}")
    candidates = answer_variants(rung.config)
    if (
        any(candidate.uses_groq() for candidate in candidates)
        and not os.getenv("GROQ_API_KEY", "").strip()
    ):
        gaps.append("missing GROQ_API_KEY")
    for candidate in candidates:
        if not candidate.uses_retrieval or candidate.index is None:
            continue
        collection = candidate.index.collection
        if collection not in collections:
            try:
                collections[collection] = (
                    None
                    if client.collection_exists(collection)
                    else f"missing collection {collection}"
                )
            except (ResponseHandlingException, UnexpectedResponse):
                collections[collection] = f"could not check collection {collection}"
        if (gap := collections[collection]) is not None and gap not in gaps:
            gaps.append(gap)
    return tuple(gaps)


@ladder_app.command("status")
def ladder_status_command(
    split: HeldOutSplit = "test", qdrant_url: QdrantUrl = DEFAULT_QDRANT_URL
) -> None:
    """Show saved runs and check collections required by pending rungs."""
    try:
        rungs = Ladder.load(LADDER_FILE).for_split(split)
        client = QdrantClient(url=qdrant_url, timeout=2, check_compatibility=False)
        collections: dict[str, str | None] = {}
        try:
            rows = ladder_status(
                rungs,
                split,
                RunStore(RUNS_DIR),
                missing=lambda rung: _ladder_gaps(rung, client, collections),
            )
        finally:
            client.close()
    except (OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo("| Rung | Status | Run | Commit | Index version | Date (UTC) | Missing |")
    typer.echo("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        run = row.run
        meta = run.meta if run else {}
        date = str(meta.get("finished_at") or meta.get("started_at") or "—")
        cells = [
            row.rung.label,
            row.state,
            run.run_id if run else "—",
            str(meta.get("commit") or "—"),
            str(meta.get("index_version") or "—"),
            date,
            ", ".join(row.missing) or "—",
        ]
        typer.echo("| " + " | ".join(cells) + " |")


def _print_ladder_estimate(
    rungs: tuple[LadderRung, ...], split: HeldOutSplit, store: RunStore, qdrant_url: str
) -> None:
    count = len(_split_ids(split))
    estimates = estimate_ladder(rungs, split, store, count)
    total = 0.0
    pending = unknown = quota_days = 0
    queries: list[BenchmarkQuery] | None = None
    for estimate in estimates:
        status = estimate.status
        name = status.rung.label
        if status.state == "complete":
            assert status.run is not None
            typer.echo(f"{name}  complete  {status.run.run_id}")
            continue
        if status.state == "blocked":
            typer.echo(f"{name}  blocked  {', '.join(status.missing)}")
            continue
        pending += 1
        if estimate.seconds is None:
            unknown += 1
            typer.echo(f"{name}  no completed Dev latency estimate")
        else:
            total += estimate.seconds
            typer.echo(f"{name}  {estimate.seconds:.1f} s  Dev run {estimate.dev_run_id}")
        if split == "test-400":
            if queries is None:
                queries = _split_queries(split)
            plan = _remote_plan(status.rung.config, queries, qdrant_url, False, False)
            quota_days += plan.days
            upper = " upper bound" if plan.upper_bound else ""
            typer.echo(
                f"  Groq  {plan.calls - plan.cached} calls after cache, "
                f"{plan.days} quota day(s){upper}"
            )
    typer.echo(f"Estimated wall time  {total:.1f} s across {pending - unknown} rung(s)")
    if unknown:
        typer.echo(f"Rungs without a Dev estimate  {unknown}")
    if split == "test-400":
        typer.echo(f"Sum of per-rung Groq quota day estimates  {quota_days}")


@ladder_app.command("run")
def ladder_run_command(
    ctx: typer.Context,
    split: HeldOutSplit = "test",
    dry_run: bool = False,
    qdrant_url: QdrantUrl = DEFAULT_QDRANT_URL,
) -> None:
    """Run unfinished rungs in order using the normal held-out score guard."""
    try:
        rungs = Ladder.load(LADDER_FILE).for_split(split)
        store = RunStore(RUNS_DIR)
        if dry_run:
            _print_ladder_estimate(rungs, split, store, qdrant_url)
            return
        statuses = ladder_status(rungs, split, store)
        if all(row.state == "complete" for row in statuses):
            typer.echo(f"All {split} rungs are complete.")
            return
        queries = _split_queries(split)
        client = QdrantClient(url=qdrant_url, timeout=2, check_compatibility=False)
        ctx.call_on_close(client.close)
        tracer = ctx.with_resource(cli_tracer())
        models = LocalModels()
        cache = ResponseCache()
        collections: dict[str, str | None] = {}

        def build(rung: LadderRung) -> PipelineVariant:
            if gaps := _ladder_gaps(rung, client, collections):
                raise ValueError(", ".join(gaps))
            typer.echo(f"Evaluating {rung.label} ({rung.config.name}) on {split}.")
            return _evaluation_variant(rung.config, models, qdrant_url, tracer, cache)

        for result in run_ladder(rungs, split, store, queries, build):
            typer.echo(f"Completed {result.run_id}")
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@ladder_app.command("report")
def ladder_report_command(
    split: HeldOutSplit = "test",
    output: Path = LADDER_REPORT_DIR,
) -> None:
    """Render Markdown and SVGs from saved evaluations."""
    try:
        splits = Splits.load(SPLITS_FILE)
        path = write_ladder_report(
            Ladder.load(LADDER_FILE),
            RunStore(RUNS_DIR),
            split,
            output,
            dev_ids=splits.dev,
            test_ids=splits.test,
            subset_ids=load_test_subset(TEST_SUBSET_FILE, splits),
        )
    except (OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Wrote {path} and its three SVG charts.")


@app.command("table")
def table_command(
    run_ids: list[str],
    metrics: Annotated[str, typer.Option(help="Comma-separated metric names, in column order.")] = (
        "hits@4,hits@10,mrr@10,evidence_recall@10,accuracy,"
        "generator_context_tokens_per_question,latency_p50,latency_p95,cost_per_question"
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
