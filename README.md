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

## Chunk Context

Build and evaluate the contextual Hybrid + Reranker variant after the model bake-off.

```bash
uv run --no-env-file rag ingest --embedder qwen3-embedding:0.6b --chunk-context qwen3.5:4b --context-size 8192 --article-tokens 4096
uv run --no-env-file rag eval configs/variants/hybrid-rerank-context.yaml --split smoke
```

Ingestion sends each Article's Chunks consecutively with the same Article prefix.
`--context-size` sets the model's context window. `--article-tokens` caps the Article
body in estimated tokens, and the prefix may be shorter to leave room for the Chunk.
An empty response is retried once with the next seed. If it stays empty, ingestion
indexes that Chunk's original text. Restart the same command after an interruption to
reuse cached responses. At one to three seconds per uncached Chunk, the full 6,759-Chunk
Corpus takes roughly two to six hours of generation.

BM25 and dense search index the Chunk Context followed by the original Chunk text.
Answer generation, Citations and Evidence matching use the original text. For the plain
comparison, use the same variant settings with the collection
`chunks-sent-w200-o1-qwen3-embedding-0.6b`. Contextual ingest reports under `runs/ingest/`
record each invocation's Chunk count, empty contexts, cache hits, generation time and
tokens for all attempts, truncated Article IDs, and notional USD where a reference price
is available.

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
Agentic Search plans sub-queries and reuses retrieval and reranking for each search.
Each step contributes up to `chunks_per_step` new Chunks to the Evidence shown to
the sufficiency check. Final generation takes the highest-ranked contribution
from each step, then the next rank from each step, until it reaches
`max_context_chunks`. It uses the same Citation and support check as the
comparison variant.

For Hits, MRR, MAP and Evidence Recall, the ranking uses all retrieved Chunks,
deduplicated in order of first appearance across searches. Evaluation also reports mean
search steps and the shares that stop on sufficiency, the step limit or the token
limit, with bootstrap intervals.

The token budget covers planning, decisions, generation and the support check
across the whole answer. Each call reserves estimated input and the configured
output allowance before execution, and debits the reported usage afterwards. The
example uses a separate 8K per-call context limit. A Groq dry run shows the upper
bound on calls at `max_steps`.

## Router

The [Router variant](configs/variants/router.yaml) orders its candidates from cheapest
to most expensive. For each Dev question, the training label is the first candidate
with a stored correct answer. If none answered correctly, the label is the last candidate.
The classifier uses only the question text, with a Qwen3-Embedding 0.6B query embedding
and features for word count, digit count, years and comparison words.

1. Evaluate each candidate on the full Dev Split with its configured index and models.

   ```bash
   for variant in closed-book bm25 hybrid-rerank agentic; do
     uv run --no-env-file rag eval "configs/variants/${variant}.yaml" --split dev || break
   done
   ```

2. Put the four run IDs in [the training config](configs/router/router.yaml), then train.

   ```bash
   uv run --no-env-file rag router train configs/router/router.yaml
   ```

   The reports in `runs/router/` compare held-out routed answers with always using the
   most expensive candidate and with a cheapest-correct oracle. The Router tries
   candidates in order and takes the first whose cumulative probability clears the threshold.
   Within each outer Dev fold, an inner cross-validation chooses the threshold with the
   lowest mean notional cost among thresholds whose accuracy reaches the lower end of the
   always-most-expensive candidate's 95% bootstrap interval. If pricing is incomplete,
   it minimizes mean latency. The deployment threshold uses out-of-fold predictions across Dev.

3. Commit the trained JSON artifact at `configs/router/trained.json` and evaluate the
   Router once on the Test Split.

   ```bash
   uv run --no-env-file rag eval configs/variants/router.yaml --split test
   ```

The evaluation records each chosen variant and reports `routed_share[<variant>]`.
Its latency and embedding token totals include the `route` stage.

## Judge and calibration

The [fixed Judge sample](data/judge-300.json) contains 300 queries stratified by query
type within `test-400`. Judge completed Test Split runs to compare the same questions
across local and Groq variants. The Judge reads saved answers and uses Groq
`openai/gpt-oss-120b` for ambiguous correctness and Citation support.

```bash
uv run --no-env-file rag judge RUN_ID --dry-run
uv run --no-env-file rag judge RUN_ID --dry-run --shared-tokens 800000 --shared-calls 400
uv run --env-file .env rag judge RUN_ID
uv run --no-env-file rag table RUN_ID --metrics judge_accuracy,judge_flipped,citation_support_rate
```

The dry run needs no key. Use the optional shared token and call estimates for upcoming
GPT-OSS-120B generation runs. They use the same 200,000-token daily budget and persistent
limiter as the Judge. The same Judge command resumes after interruption using the response
cache. Keep the cache when running over several days.

Containment-correct answers keep their primary score. The Judge can accept an ambiguous
answer that containment missed. The flipped share uses all containment-incorrect sampled
answers. For a non-abstained answer to a non-Null Query, Judge support covers the final cited
statements and uses only their cited text. Malformed verdicts are recorded as null and
count as no new correct or supported decisions. The verdicts and metrics with bootstrap
intervals are stored in `judge.jsonl` and `judge-metrics.json` alongside the run.
Older runs need the matching local Corpus when their saved records lack cited text.

Export from completed Dev runs, including a citation variant to cover both tasks.
With no run IDs, the command uses all completed Dev runs with scored answers.

```bash
uv run --no-env-file rag judge calibration export DEV_RUN_ID CITED_DEV_RUN_ID
# Follow data/judge-calibration/LABELLING.md to create labels.jsonl.
uv run --env-file .env rag judge calibration report
```

Copy the exported cases into a subscription chat by hand for one-time labelling, then
independently check at least 50 cases as the author. Keep both labels where they differ.
Commit the cases and validated labels under `data/judge-calibration/`. The calibration
command calls the Groq Judge and writes JSON and Markdown agreement reports under
`runs/judge/`, with accuracy and Cohen's kappa for correctness and support. The reports
also include subscription versus author agreement on the double-checked cases.

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
