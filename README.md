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

## Cached evaluations and the Groq comparison

Evaluations cache completions, query embeddings and Reranker scores in
`data/cache/models.sqlite3`. Stage traces flag cache hits and reuse the original
model-call durations. Retrieval still queries the current index, so index timing
can vary between runs. Pass `--no-cache` to bypass the response cache.

Set `GROQ_API_KEY` with a Groq Free Plan key to run the large-model comparison.
Start with one Smoke Set query, then use the fixed `test-400` subset for the held-out
comparison. The score-once rule applies across `test` and `test-400` for each Pipeline
Variant. The dry run needs no key or model calls.

```bash
uv run rag eval configs/variants/closed-book-groq.yaml --split test-400 --dry-run
uv run rag eval configs/variants/closed-book-groq.yaml --split smoke --limit 1
```

The limiter persists requests and estimated token reservations in SQLite and reconciles
them with Groq's reported usage. Limits use rolling minute and 24-hour windows. The
dry run estimates quota days at a fresh allowance and, for retrieval variants, estimates
the prompt length from the configured number of Chunks. Its cache count includes
known prompts for the current index version.

## Citations and Abstention

The [cited Hybrid + Reranker variant](configs/variants/hybrid-rerank-cited.yaml) asks
for a short answer and separate statements with Chunk IDs. Its support check evaluates
the statements in one call using the cited Chunk text. Evaluation scores the short
answer when at least one statement is supported and the short answer is present and
usable. Per-query records include shown Chunk IDs, the retained Citations, and the
removed statements with rejection reasons.

Evaluation reports Abstention on Null Queries and False Abstention on answerable
queries. Citation validity measures the share of proposed IDs actually shown to the
generator. Citation Evidence precision measures the share that point to a shown
Chunk containing a Benchmark Query's Evidence fact. Evidence precision uses
answerable queries, and both metrics are omitted when their denominator is empty.

```bash
uv run rag eval configs/variants/hybrid-rerank-cited.yaml --split smoke
```

## Agentic Search

Compare [Agentic Search](configs/variants/agentic.yaml) with the
[cited Hybrid + Reranker variant](configs/variants/hybrid-rerank-cited.yaml).
Agentic Search plans sub-queries, reuses retrieval and reranking for each search,
and checks whether the Evidence is sufficient. It caps the final answer's context
and uses the same Citation and support check as the comparison variant.

For Hits, MRR, MAP and Evidence Recall, the ranking is the deduplicated Evidence
pool in order of first appearance across searches. Evaluation also reports mean
search steps and the shares that stop on sufficiency, the step limit or the token
limit, with bootstrap intervals.

The token budget covers planning, decisions, generation and the support check
across the whole answer. Each call reserves estimated input and the configured
output allowance before execution, and debits the reported usage afterwards. The
example uses a separate 8K per-call context limit. A Groq dry run shows the upper
bound on calls at `max_steps`.

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
