"""Command line entry points: `rag data ...`, `rag ingest`, `rag eval`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from qdrant_client import QdrantClient

from rag_pipeline.chunking import ChunkingSettings
from rag_pipeline.corpus import DATASET_LICENSE, fetch_dataset, load_articles, load_queries
from rag_pipeline.evaluation import RunStore, evaluate
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
    qdrant_url: QdrantUrl = "http://localhost:6333",
) -> None:
    """Evaluate a Pipeline Variant on a split and store the run."""
    if split not in SPLIT_NAMES:
        raise typer.BadParameter(f"split must be one of {SPLIT_NAMES}")
    config = VariantConfig.from_yaml(variant_file)
    by_id = {q.id: q for q in load_queries(RAW_DIR)}
    queries = [by_id[i] for i in Splits.load(SPLITS_FILE).ids(split)][:limit]
    index = ChunkIndex(QdrantClient(url=qdrant_url), config.index.collection)
    variant = build_variant(config, index=index, models=LocalModels())
    result = evaluate(variant, queries, split, RunStore(RUNS_DIR))
    typer.echo(f"Run {result.run_id}")
    for name, value in result.metrics.items():
        typer.echo(
            f"  {name:>10}: {value:.4f}" if isinstance(value, float) else f"  {name:>10}: {value}"
        )
