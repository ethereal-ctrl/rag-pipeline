# Model bake-off (Side Ablation)

All runs use the fixed Dev Split (500 Benchmark Queries; retrieval metrics exclude the
Null Queries). Every value has a bootstrap 95% interval. Δ is the paired bootstrap
difference on shared queries, and `*` marks an interval that excludes zero. The roles were
decided one at a time, and each later comparison builds on the earlier winners.

The tables are rendered from stored runs with `rag table`, and the deltas with
`rag compare`.

| Role | Candidates | Winner |
|---|---|---|
| Embedder | bge-m3, Qwen3-Embedding 0.6B, Qwen3-Embedding 4B | **Qwen3-Embedding 0.6B** |
| Reranker | bge-reranker-v2-m3, Qwen3-Reranker 0.6B | **bge-reranker-v2-m3** |
| Generator | qwen3.5:4b, qwen3.5:9b | **qwen3.5:4b** |

## Embedder

Hybrid retrieval (BM25 + dense, Reciprocal Rank Fusion), retrieval only. Latency is per
question and is mostly the query embedding.

| Embedder | Hits@4 | Hits@10 | MRR@10 | MAP@10 | Evidence Recall@10 | Latency p50 (s) | Index build |
|---|---:|---:|---:|---:|---:|---:|---:|
| bge-m3 | 0.737 [0.693, 0.776] | 0.875 [0.843, 0.905] | 0.589 [0.554, 0.627] | 0.299 [0.280, 0.321] | 0.567 [0.536, 0.597] | 0.054 | 15 min |
| Qwen3-Embedding 0.6B | 0.773 [0.731, 0.811] | 0.921 [0.893, 0.944] | 0.628 [0.591, 0.664] | 0.319 [0.298, 0.339] | 0.596 [0.567, 0.624] | 0.070 | 16 min |
| Qwen3-Embedding 4B | 0.785 [0.746, 0.822] | 0.923 [0.899, 0.945] | 0.644 [0.611, 0.681] | 0.329 [0.308, 0.350] | 0.611 [0.582, 0.640] | 0.439 | 2 h 25 min |

- **Qwen3-Embedding 0.6B vs bge-m3:**
  - every retrieval metric improves significantly: Hits@10 Δ +0.045* [+0.021, +0.070], MRR@10 Δ +0.039* [+0.013, +0.066];
  - p50 latency rises by only 15 ms.
- **Qwen3-Embedding 4B vs 0.6B:**
  - only Evidence Recall@10 improves significantly: Δ +0.015* [+0.002, +0.031];
  - Hits@4, Hits@10 and MRR@10 show no significant change;
  - in exchange, query latency rises 6× (+0.37 s at p50) and the index takes 9× as long to build.

**Conclusion.** The 0.6B model gets nearly all of the quality for a fraction of the cost.
The larger embedder buys a marginal recall gain that does not reach the ranking metrics
the generator depends on.

## Reranker

Built on the embedder winner, retrieval only. Each reranker re-scores the first 30
candidates.

| Reranker | Hits@4 | Hits@10 | MRR@10 | MAP@10 | Evidence Recall@10 | Latency p50 (s) | Latency p95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.773 [0.731, 0.811] | 0.921 [0.893, 0.944] | 0.628 [0.591, 0.664] | 0.319 [0.298, 0.339] | 0.596 [0.567, 0.624] | 0.070 | 0.096 |
| bge-reranker-v2-m3 | 0.821 [0.784, 0.853] | 0.925 [0.900, 0.948] | 0.701 [0.667, 0.736] | 0.368 [0.346, 0.390] | 0.651 [0.621, 0.679] | 5.380 | 6.897 |
| Qwen3-Reranker 0.6B | 0.850 [0.819, 0.886] | 0.950 [0.928, 0.969] | 0.699 [0.667, 0.733] | 0.370 [0.348, 0.390] | 0.669 [0.639, 0.696] | 7.567 | 8.994 |

- **bge-reranker vs none.** The reranker does not find more Evidence (Hits@10 unchanged) but ranks it higher:
  - Hits@4 Δ +0.048* [+0.009, +0.087];
  - MRR@10 Δ +0.073* [+0.036, +0.107];
  - Evidence Recall@10 Δ +0.055*.

  That matters because the generator sees only the top 5 Chunks.
- **Qwen3-Reranker vs bge-reranker:**
  - MRR@10 and MAP@10 are equal (Δ −0.002 and +0.001);
  - Hits@10 Δ +0.025* [+0.002, +0.050], while Hits@4 Δ +0.029 [−0.005, +0.064] is not significant;
  - p50 latency is 2.2 s higher, about +40%.

**Conclusion.** bge-reranker-v2-m3 ranks just as well at the top for 40% less latency.

**Latency note.** Reranking costs about 5.4 s per question inside the pipeline but about
1.2 s when measured in isolation. The difference is likely contention between the MPS
cross-encoder and Ollama on the same 16 GB GPU, and it is a target for the tracing
analysis.

## Generator

Built on the embedder and reranker winners, full answers, with the final prompts that
include delimiter escaping.

| Generator | Accuracy | Comparison | Inference | Temporal | Abstention on Null Queries | False Abstention | Latency p50 (s) | Latency p95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3.5:4b | 0.690 [0.650, 0.726] | 0.623 [0.548, 0.696] | 0.887 [0.834, 0.934] | 0.395 [0.305, 0.481] | 0.915 [0.841, 0.982] | 0.100 [0.074, 0.129] | 11.70 | 14.35 |
| qwen3.5:9b | 0.696 [0.658, 0.734] | 0.581 [0.509, 0.655] | 0.900 [0.850, 0.946] | 0.439 [0.352, 0.534] | 0.966 [0.912, 1.000] | 0.147 [0.116, 0.181] | 21.42 | 40.58 |

- **qwen3.5:9b vs 4b:**
  - accuracy is unchanged: Δ +0.006 [−0.026, +0.040], and no per-type difference is significant;
  - the larger model abstains more often on answerable queries: False Abstention Δ +0.048* [+0.020, +0.074];
  - it takes almost twice as long at p50 (+9.7 s) and about three times as long at p95 (+26.2 s).

**Conclusion (negative result).** On this task, with this context, doubling the generator's
size buys no accuracy. It trades a slightly higher correct-Abstention rate for more
over-caution and much higher latency.

## Cost note

The notional cost per question is dominated by the reranker, not the generator.
Re-scoring 30 candidates feeds about 10,400 input tokens to the cross-encoder, against
about 1,900 for the generator. Reports should therefore show cost by stage.

## Reproduce

```bash
uv run rag eval configs/bakeoff/<config>.yaml --split dev [--retrieval-only]
uv run rag table RUN...
uv run rag compare BASELINE_RUN CANDIDATE_RUN
```

The embedder and reranker comparisons are retrieval only (`--retrieval-only`). The run
ids are listed in the Ticket 10 notes.
