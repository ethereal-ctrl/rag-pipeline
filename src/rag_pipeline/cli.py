"""Command line entry points: `rag data ...`, `rag ingest`, `rag eval`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from qdrant_client import QdrantClient

from rag_pipeline.chunking import ChunkingSettings
from rag_pipeline.corpus import DATASET_LICENSE, fetch_dataset, load_articles, load_queries
from rag_pipeline.evaluation import (
    RunStore,
    TestSplitAlreadyScored,
    compare_runs,
    evaluate,
    export_records,
)
from rag_pipeline.index import ChunkIndex
from rag_pipeline.ingest import ingest
from rag_pipeline.models import LocalModels
from rag_pipeline.splits import DEFAULT_SEED, DEFAULT_SIZES, SPLIT_NAMES, Splits, make_splits
from rag_pipeline.variants import VariantConfig, build_variant

RAW_DIR = Path("data/raw")
SPLITS_FILE = Path("data/splits.json")
RUNS_DIR = Path("runs")

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


@app.command("ingest")
def ingest_command(
    collection: str = "chunks-sent-w200-o1",
    max_words: int = 200,
    overlap_sentences: int = 1,
    qdrant_url: QdrantUrl = "http://localhost:6333",
) -> None:
    """Chunk and index the Corpus."""
    index = ChunkIndex(QdrantClient(url=qdrant_url), collection)
    report = ingest(load_articles(RAW_DIR), index, ChunkingSettings(max_words, overlap_sentences))
    typer.echo(f"{report.articles} Articles -> {report.chunks_written} Chunks in {collection}.")


@app.command("eval")
def eval_command(
    variant_file: Path,
    split: str = "smoke",
    limit: Annotated[int | None, typer.Option(help="Only the first N queries (debugging).")] = None,
    allow_rescore: Annotated[
        bool, typer.Option(help="Re-score a variant on the Test Split (ADR 0005).")
    ] = False,
    qdrant_url: QdrantUrl = "http://localhost:6333",
) -> None:
    """Evaluate a Pipeline Variant on a split and store the run (resumes if interrupted)."""
    if split not in SPLIT_NAMES:
        raise typer.BadParameter(f"split must be one of {SPLIT_NAMES}")
    if split == "test" and limit is not None:
        raise typer.BadParameter("--limit is not allowed on the Test Split")
    config = VariantConfig.from_yaml(variant_file)
    by_id = {q.id: q for q in load_queries(RAW_DIR)}
    queries = [by_id[i] for i in Splits.load(SPLITS_FILE).ids(split)][:limit]
    index = ChunkIndex(QdrantClient(url=qdrant_url), config.index.collection)
    variant = build_variant(config, index=index, models=LocalModels())
    try:
        result = evaluate(variant, queries, split, RunStore(RUNS_DIR), allow_rescore=allow_rescore)
    except TestSplitAlreadyScored as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Run {result.run_id}")
    _print_metrics(result.metrics, result.intervals)


def _print_metrics(metrics: dict[str, float], intervals: dict[str, tuple[float, float]]) -> None:
    for name, value in metrics.items():
        if name == "queries":
            typer.echo(f"  {name:<34} {int(value)}")
            continue
        low, high = intervals.get(name, (float("nan"), float("nan")))
        typer.echo(f"  {name:<34} {value:.3f}  [{low:.3f}, {high:.3f}]")


@app.command("compare")
def compare_command(baseline_run: str, candidate_run: str) -> None:
    """Paired comparison of two runs; '*' marks a significant change (95% interval)."""
    store = RunStore(RUNS_DIR)
    deltas = compare_runs(store.load(baseline_run), store.load(candidate_run))
    typer.echo(f"  {'metric':<34} {'base':>6} {'cand':>6} {'delta':>7}  95% interval")
    for d in deltas:
        mark = "*" if d.significant else " "
        typer.echo(
            f"{mark} {d.metric:<34} {d.baseline:6.3f} {d.candidate:6.3f} {d.delta:+7.3f}"
            f"  [{d.low:+.3f}, {d.high:+.3f}]"
        )


@app.command("export")
def export_command(run_id: str, output: Path | None = None) -> None:
    """Export a run's per-query records as CSV for error analysis."""
    path = export_records(RunStore(RUNS_DIR).load(run_id), output or RUNS_DIR / f"{run_id}.csv")
    typer.echo(f"Wrote {path}.")
