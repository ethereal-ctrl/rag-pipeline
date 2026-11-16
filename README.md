# RAG Pipeline: a measured Ablation Ladder on MultiHop-RAG

Work in progress. The goal: measure, with confidence intervals, which advanced retrieval
techniques actually pay off on multi-hop questions, running fully locally.
See `GLOSSARY.md` for the vocabulary and `docs/adr/` for the design decisions.

## Run it yourself

1. Install Docker with Compose and Ollama, then start Ollama on the host. On an Apple M4,
   host Ollama uses the Apple GPU. The container's Reranker runs on CPU and takes longer
   than the host Reranker on MPS.
2. Pull the generator and embedder for the default `hybrid-rerank-cited` variant.

   ```bash
   ollama pull qwen3.5:4b && ollama pull bge-m3
   ```

3. In the clone, start the API, Qdrant and Phoenix, then build the demo index.

   ```bash
   docker compose up -d
   docker compose run --rm api rag bootstrap
   ```

   Bootstrap fetches the pinned MultiHop-RAG dataset, verifies its checksums, and builds the
   selected variant's index. As a reference, embedding the Corpus with Qwen3-Embedding 0.6B
   took about 16 minutes on an M4. Later runs skip unchanged Articles. The first request using
   a Reranker also downloads its weights into the persistent Hugging Face cache.
4. Open [the demo](http://localhost:8000) and ask a question. Expand the Citations to see the
   original Chunk text. Open the answer's trace in [Phoenix](http://localhost:6006).
5. For host inference with MPS, keep Qdrant and Phoenix running, stop the container API if it
   is running, and start the same app with uv.

   ```bash
   docker compose stop api
   uv run rag serve
   ```

The default demo runtime costs nothing. Set `RAG_DEMO_VARIANT` to choose the default in the
page and the index that bootstrap builds. Pull the Ollama models listed in the selected
variant's config. The container reaches host Ollama at `http://host.docker.internal:11434`.
If the container cannot reach Ollama on macOS, quit the desktop app and start the host server
with `OLLAMA_HOST=0.0.0.0:11434 ollama serve` in a separate terminal.

The API reads `OLLAMA_URL`, `QDRANT_URL`, `RAG_TRACING_ENDPOINT` and `PHOENIX_URL`. The first
three configure connections, and `PHOENIX_URL` sets the URL used in browser trace links.
The local demo page, API docs and reachability checks are available at `/`, `/docs` and
`/health` respectively.

## Run evaluations

```bash
uv sync
docker compose up -d qdrant phoenix
ollama pull qwen3.5:4b && ollama pull qwen3-embedding:0.6b
uv run rag data fetch         # pinned MultiHop-RAG files, checksum-verified
uv run rag ingest --variant-file configs/variants/bm25.yaml
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

## Cached evaluations and the large-model comparison

Evaluations cache completions, query embeddings and Reranker scores in
`data/cache/models.sqlite3`. Stage traces flag cache hits and reuse the original
model-call durations. Retrieval still queries the current index, so index timing
can vary between runs. Pass `--no-cache` to bypass the response cache.

The large-model comparison and Judge default to Gemma 4 31B
(`gemma-4-31b-it`) on the Gemini API. In Google AI Studio, open **API keys**
and create a key against a **Free Tier** project. Put the key in the worktree's
`.env` as `GEMINI_API_KEY`. The author's project showed 30 requests per minute,
16,000 tokens per minute, and 14,400 requests per day on 2026-11-16. Check your
project's current limits and regional availability before running.
[AI Studio API keys](https://ai.google.dev/gemini-api/docs/api-key) ·
[Project rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) ·
[Available regions](https://ai.google.dev/gemini-api/docs/available-regions)

Gemma 4 can return reasoning in the answer text. The Gemma configs request
Google's `minimal` thinking level, which Google documents as off for Gemma 4.
The shared remote client removes any returned leading thought blocks before
scoring or parsing JSON. Reasoning tokens count toward output usage, notional cost,
and the persistent limiter. When a response omits the count, the client estimates
it from the removed text. Traces record counts and raw content length.
[Gemma thinking](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api#thinking) ·
[OpenAI-compatible thinking settings](https://ai.google.dev/gemini-api/docs/openai#extra-body)

Google may use free-tier prompts and responses to improve its products. Its terms
provide regional exceptions for the EEA, the UK, and Switzerland. Remote
benchmarking sends only public benchmark questions and Corpus text, along with
answers generated from those inputs for judging. The demo API defaults to the
local `hybrid-rerank-cited` variant. Arbitrary user questions stay on the machine
unless the demo is explicitly configured to use a remote variant.
[Gemini API data terms](https://ai.google.dev/gemini-api/terms)

Set `GROQ_API_KEY` in the same file to use the Groq Free Plan alternative.

Start with one Smoke Set query, then use the fixed `test-400` subset. The score-once
rule applies across `test` and `test-400` for each Pipeline Variant. The dry run
uses configured quotas and needs no key or model calls.

```bash
uv run --env-file .env rag eval configs/variants/closed-book-gemma.yaml --split test-400 --dry-run
uv run --env-file .env rag eval configs/variants/closed-book-gemma.yaml --split smoke --limit 1
uv run --env-file .env rag eval configs/variants/closed-book-groq.yaml --split smoke --limit 1
```

The limiter persists requests and estimated token reservations in SQLite and
reconciles them with reported usage. Both providers use rolling minute and 24-hour
windows. Gemma generation and judging share the same model quota when they use the
same cache database. Dry runs estimate hours and whole quota days at a fresh allowance.
For retrieval variants, the estimate uses the configured number of Chunks and known
cache hits for the current index version.
Remote calls retry transient server errors, timeouts, and rate-limit responses
with backoff. If retries run out, rerun the same evaluation command to resume.

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
[contextualized, cited Hybrid + Reranker variant](configs/variants/hybrid-rerank-context-cited.yaml).
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
across local and remote variants. The Judge reads saved answers and grades ambiguous
correctness and Citation support. The default `configs/judge.yaml` uses Gemma
4 31B. The Judge must be from a different family than every generator used for
the run. Select `configs/judge-groq.yaml` to use Groq `openai/gpt-oss-120b` for Gemma runs.

```bash
uv run --env-file .env rag judge RUN_ID --dry-run
uv run --env-file .env rag judge RUN_ID --dry-run --shared-tokens 800000 --shared-calls 400
uv run --env-file .env rag judge RUN_ID
uv run --env-file .env rag judge GEMMA_RUN_ID --judge-config configs/judge-groq.yaml
uv run --no-env-file rag table RUN_ID --metrics judge_accuracy,judge_flipped,citation_support_rate
```

The dry run needs no key. Use the optional shared token and call estimates for
upcoming workloads that use the same quota as the Judge. Rerun the same Judge command
to resume after interruption using the response cache.

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
command uses the selected Judge config and writes JSON and Markdown agreement reports
under `runs/judge/`. They record the Judge model, accuracy, and Cohen's kappa for
correctness and support, plus subscription versus author agreement on the
double-checked cases. Use `--judge-config` to choose the Judge for calibration.

## Side Ablations

The configs in `configs/side-ablations/` compare against Hybrid + Reranker with
Qwen3-Embedding 0.6B, BGE-Reranker-v2-m3 over 30 candidates, and the provisional
Qwen3.5 4B generator. Build the four collections from their Pipeline Variants and
score all five variants on Dev. HyDE reuses the 200-word reference collection.

```bash
for variant in chunk-100 chunk-200 chunk-400 semantic; do
  uv run --no-env-file rag ingest --variant-file "configs/side-ablations/${variant}.yaml" || break
done

for variant in chunk-100 chunk-200 chunk-400 semantic hyde; do
  uv run --no-env-file rag eval "configs/side-ablations/${variant}.yaml" --split dev || break
done
```

Larger Chunks can raise Evidence Recall@10 by packing more facts into the same number
of retrieved Chunks. They also give the generator more text. The default `rag table`
includes `generator_context_tokens_per_question`, the mean input tokens from the
final `generate` stage including the prompt framing.

Semantic chunking starts a new Chunk where adjacent sentence cosine distance exceeds
the Article's 90th percentile or the next sentence exceeds the Chunk's 200-word budget.
This variant uses the same one-sentence overlap setting as the reference. It carries
the previous Chunk's last sentence across either type of boundary when it fits. A
sentence longer than the budget stays whole in its own Chunk. Ingestion batches and
caches the sentence embeddings and skips unchanged Articles on subsequent builds.

HyDE generates a short hypothetical passage for the dense arm. By default BM25 uses
the question. Set `retrieval.hyde.bm25: both` to search the combined question and
passage with BM25. The answer generator receives the original question and retrieved
Chunks as Evidence. The `hyde` stage records the extra generation time and tokens.
[HyDE paper](https://aclanthology.org/2023.acl-long.99/)

The [Side Ablations report](docs/report/side-ablations.md) has slots for the measured
comparisons and conclusions.

## Final Ablation Ladder

The [ladder definition](configs/ladder.yaml) records the order and checks each
adjacent config change. The early retrieval rungs share the plain dense index.
Chunk Context, Citations, and Agentic Search share the contextualized index.
The Router delegates to existing variants and needs its artifact fitted on Dev.

After locking the bake-off winners, build the plain and contextualized indexes
using their final configs. Enter the matching completed Dev run IDs in
`configs/router/router.yaml` before training the Router.

```bash
uv run --no-env-file rag ingest --variant-file configs/variants/hybrid-rerank.yaml
uv run --no-env-file rag ingest --variant-file configs/variants/hybrid-rerank-context.yaml
uv run --no-env-file rag router train configs/router/router.yaml
```

The status command shows completed runs, interruptions, and missing collections.
The local time estimate uses the latest saved Dev p50 for each rung. The runner
uses the normal Test Split guard and resumes a matching interrupted run.

```bash
uv run --no-env-file rag ladder status --split test
uv run --no-env-file rag ladder run --split test --dry-run
uv run --no-env-file rag ladder run --split test
uv run --env-file .env rag ladder status --split test-400
uv run --env-file .env rag ladder run --split test-400 --dry-run
uv run --env-file .env rag ladder run --split test-400
uv run --env-file .env rag ladder run --split test-400 --provider groq --dry-run
uv run --no-env-file rag ladder report --split test --output docs/report/ladder
```

The large-model comparison uses saved local answers on the same queries as the selected
remote provider. Gemma is selected by default. Use `--provider groq` with
`ladder status`, `ladder run`, and `ladder report` to select the Groq comparison.
Agentic Search is the deepest rung with a direct generator to swap. Estimate its
quota separately since it can call the remote generator multiple times per question.
The report includes pending rows, source run IDs, and the Dev majority baseline
once saved runs cover all fixed Dev and Test queries. After grading completed
runs with the appropriate Judge config, regenerate the report to include their Judge metrics.

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
