# RAG Pipeline: a measured Ablation Ladder on MultiHop-RAG

Work in progress. The goal: measure, with confidence intervals, which advanced retrieval
techniques actually pay off on multi-hop questions, running fully locally.
See `GLOSSARY.md` for the vocabulary and `docs/adr/` for the design decisions.

## Quick start

```bash
uv sync
docker compose up -d          # Qdrant and Phoenix
ollama pull qwen3.5:4b        # local generator
uv run rag data fetch         # pinned MultiHop-RAG files, checksum-verified
uv run rag ingest             # chunk + index the Corpus (incremental)
uv run rag eval configs/variants/bm25.yaml --split smoke
```

Ingestion is incremental: Articles are compared by content hash, so unchanged ones are
skipped and changed or deleted ones lose their old Chunks. Each chunking + embedder
combination gets its own collection (e.g. `chunks-sent-w200-o1-bge-m3`), so index versions
coexist; the index version adds the Corpus hash and is stored with every run.

Every metric comes with a bootstrap 95% confidence interval. Runs are stored under `runs/`
with their config, split, model version and commit, and resume where they stopped if
interrupted. Each Pipeline Variant can be scored on the Test Split only once (ADR 0005).

```bash
uv run rag compare RUN_A RUN_B   # paired bootstrap; '*' marks significant changes
uv run rag export RUN            # per-query CSV for error analysis
```

Retrieval metrics (Hits@4/@10, MRR@10, MAP@10) follow the upstream MultiHop-RAG scoring
exactly and were cross-checked against it; Evidence Recall@10 adds the fraction of all
Evidence found. Answer accuracy requires the whole normalised reference answer to appear in
the answer, which is stricter than the upstream any-shared-word check.

## Tracing and notional cost

Start Phoenix with `docker compose up -d phoenix`. After running an evaluation, open
[Phoenix](http://localhost:6006) and select the `rag-pipeline` project to inspect an
answer's retrieval, optional Reranker, and generation spans. Spans carry the Pipeline
Variant, Chunk IDs and available model token counts. Prompts and Chunk text stay out of
the spans. The CLI checks the collector at startup and exports over OTLP/HTTP when it is
reachable. Set `RAG_TRACING_ENDPOINT` to override
`http://127.0.0.1:6006/v1/traces` or set `RAG_TRACING=0` to disable export.

Evaluation stores timings and token counts by stage. The `cost_per_question` metric sums
each stage's token counts at the [hosted reference list prices](src/rag_pipeline/prices.yaml)
and averages the totals in notional USD. It is available when every model used has a
reference price and the run has complete token counts. The runtime costs nothing in API
fees under [ADR 0001](docs/adr/0001-zero-cost-no-subscription-runtime.md).

## Development

```bash
uv run pytest        # seam tests with a fixture Corpus and fake models, no network
uv run ruff check src tests
uv run pyright
uv run rag gate      # Smoke Set retrieval metrics vs. committed thresholds
uv run rag gate --answers   # plus answer-level checks with the local generator
```

Every pull request runs lint, type checks and tests, then a retrieval gate: CI builds the
full index with a small CPU embedder (`BAAI/bge-small-en-v1.5`) in a Qdrant service
container, scores BM25 and Hybrid retrieval on the Smoke Set with no LLM, and fails if any
metric falls below the thresholds in `configs/gate/gate.yaml`. No secrets, no paid services.

## Data

MultiHop-RAG by Yixuan Tang and Yi Yang (arXiv:2401.15391), used under the
[ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/) licence, pinned to Hugging Face
revision `71ac0d0b` of `yixuantt/MultiHopRAG`.
