# GraphRAG Dev experiment

## Setup

LightRAG retrieves context in the configured `mix` or `hybrid` mode. The configured
build uses Gemma 4 31B for extraction and local Qwen3-Embedding 0.6B for embeddings.
Retrieval keywords and answers use the ladder's local Qwen3.5 4B generator.
Queries are restricted to inference and comparison on the Dev Split.

<!-- table: build ID, Corpus revision, extraction model, answer model, embedder, mode, ladder run ID -->

## One-off graph build

<!-- table: Articles, elapsed seconds, LLM calls, input and output tokens, cache hits, notional USD -->

## Paired Dev comparison

<!-- table: shared query counts, accuracy, Abstention, latency, tokens and notional query cost -->
<!-- table: paired deltas and 95% bootstrap intervals for accuracy, latency and query cost -->

## Limitations

LightRAG's extraction prompts lack the project's prompt-injection hardening.
The final answer uses the ladder's untrusted-context instructions and escapes
context delimiters. Context-only retrieval supplies no ranked Chunks for scoring.

<!-- interpretation: build model, Citation and Agentic Search settings in the ladder run, cache state -->

## Conclusion

<!-- conclusion -->
