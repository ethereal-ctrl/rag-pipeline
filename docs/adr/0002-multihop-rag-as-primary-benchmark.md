# MultiHop-RAG is the primary benchmark

We evaluate on MultiHop-RAG (609 news Articles, about 1.25M tokens, 2,556 Benchmark Queries with Evidence, ODC-BY). It is small enough to index and contextualise on a 16 GB laptop for free, it has a real shared Corpus, its query types (inference, comparison, temporal, null) let us show where each technique helps, and its Null Queries test Abstention directly.

## Considered Options

- **FRAMES**: harder and more recognised, but needs a large Wikipedia Corpus we cannot index for free.
- **MuSiQue / HotpotQA**: ship per-question paragraphs instead of a shared Corpus; HotpotQA is saturated and likely in training data.
- **Own domain corpus**: attractive, but no reference answers; every label would be our own.

## Consequences

- The Articles date from Sept to Dec 2023, so 2026 models may have memorised them. A closed-book (no retrieval) Pipeline Variant is required as a memorisation baseline.
- Retrieval hits are matched by Evidence fact text, not document IDs, and the paper's and the code's definitions of Hits@K differ; we must state which one we report.
