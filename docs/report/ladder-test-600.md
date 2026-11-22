# Ablation Ladder on the untouched Test Split (test-600)

Final scores on the 600 Test Split queries that no design decision has seen (ADR 0008).
All variant choices were made on the Dev Split (500 queries). The 400-query subset of the
Test Split was used for error analysis and is development history, not a final score.
Every value has a 95% percentile-bootstrap interval. "Significant" means the 95% interval of
the paired difference excludes zero (`rag compare`).

Generators: Qwen3.5 4B, run locally through Ollama on an M4 with 16 GB, and Gemma 4 31B
through NVIDIA NIM's free tier. Retrieval uses Qwen3-Embedding 0.6B, BM25 and the
bge-reranker-v2-m3 Reranker. Accuracy is a substring match against the reference answer.
A Null Query counts as correct only when the model abstains.

Full table with intervals: [`ladder-test-600-table.md`](ladder-test-600-table.md).

## The ladder (Qwen3.5 4B)

| Rung | Accuracy | Null Queries abstained | False Abstention | Score (wrong = −2) | p50 latency (s) |
|---|---:|---:|---:|---:|---:|
| Always "Yes" (trivial baseline) | 0.325 | 0.000 | 0.000 | −1.025 | — |
| Closed-Book | 0.445 | 0.014 | 0.009 | −0.648 | 0.4 |
| BM25 | 0.665 | 0.930 | 0.129 | 0.222 | 5.6 |
| Dense | 0.613 | 0.901 | 0.166 | 0.133 | 6.1 |
| Hybrid (RRF) | 0.685 | 0.859 | 0.102 | 0.235 | 6.1 |
| + Reranker | 0.722 | 0.887 | 0.100 | 0.342 | 11.4 |
| + Chunk Context | 0.720 | 0.873 | 0.091 | 0.320 | 10.8 |
| + Citations & support check | 0.650 | 0.930 | 0.242 | 0.377 | 20.6 |
| Agentic Search | 0.747 | 0.958 | 0.153 | 0.510 | 43.3 |
| **Hybrid + Reranker + source-aware retrieval (chosen)** | **0.775** | 0.845 | 0.053 | 0.418 | 13.4 |
| Oracle retrieval (gold Evidence in context) | 0.855 | 0.972 | 0.004 | 0.572 | 3.0 |

## Paired comparisons (accuracy, 95% interval of the difference)

| Step | Δ accuracy | Significant |
|---|---:|:---:|
| Closed-Book → BM25 | +0.220 [+0.167, +0.273] | yes |
| BM25 → Dense | −0.052 [−0.090, −0.010] | yes |
| Dense → Hybrid | +0.072 [+0.037, +0.105] | yes |
| BM25 → Hybrid | +0.020 [−0.012, +0.053] | no |
| Hybrid → + Reranker | +0.037 [+0.000, +0.070] | borderline |
| + Reranker → + Chunk Context | −0.002 [−0.018, +0.015] | no |
| + Chunk Context → + Citations | −0.070 [−0.103, −0.035] | yes |
| + Chunk Context → Agentic Search | +0.027 [−0.013, +0.065] | no |
| + Reranker → source-aware retrieval | +0.053 [+0.030, +0.077] | yes |
| Agentic Search → source-aware retrieval | +0.028 [−0.008, +0.067] | no |
| Chosen → Oracle retrieval | +0.080 [+0.050, +0.113] | yes |

## Findings

1. **Retrieval is what makes the model honest.** Without retrieval the 4B model answers
   98.6% of the Null Queries with an invented answer. With any retrieval it abstains on
   85–96% of them.
2. **BM25 beats dense retrieval on this corpus** (−0.052 for Dense, significant). News
   questions hinge on names, outlets and dates, where exact term match is strong. Hybrid
   fusion recovers Dense, but it is not significantly better than BM25 alone.
3. **Chunk Context did not pay off** (−0.002). It is cheaper at answer time, but it gives no
   accuracy gain on test-600.
4. **Citations cost accuracy.** Requiring cited, supported statements lifts False Abstention
   from 9% to 24%. Accuracy drops by 0.070. The answers it does give are more often right
   (Selective Accuracy 0.808 vs 0.769). Traceability has a price, and this measures it.
5. **The cheap lever beat the agent.** Source-aware retrieval lifts the named outlets'
   articles into the context. It gives +0.053 over the Reranker at +2 s per question.
   Agentic Search gives +0.027 at +32 s. The two are not significantly different in
   accuracy. The agent abstains more, so it wins on the score that punishes wrong answers
   (0.510 vs 0.418).
6. **The headroom is in retrieval.** With gold Evidence the same 4B model reaches 0.855. That
   puts the retrieval loss at 8.0 points for the 4B model and 11.2 points for Gemma
   (0.712 → 0.823). The remaining gap to 1.0 is the generator's limit, largest on temporal
   queries (Oracle accuracy 0.752 for the 4B model, 0.555 for Gemma).

## Large model (Gemma 4 31B via NVIDIA NIM)

| Variant | Accuracy | Null Queries abstained | False Abstention | Selective Accuracy | Score (wrong = −2) |
|---|---:|---:|---:|---:|---:|
| Closed-Book | 0.473 | 0.662 | 0.493 | 0.884 | 0.290 |
| Hybrid + Reranker | 0.578 | 1.000 | 0.416 | 0.893 | 0.468 |
| + sources, 10 chunks, decisive prompt (chosen) | 0.712 | 1.000 | 0.240 | 0.886 | 0.558 |
| Oracle retrieval | 0.823 | 1.000 | 0.102 | 0.891 | 0.650 |
| Agentic Search | pending | | | | |

Gemma never answers a Null Query, and its answers are right 89% of the time. It abstains far
too often on answerable ones, though. The three Dev-chosen levers cut False Abstention from
42% to 24% (+0.133 accuracy, significant). The local 4B model still has higher accuracy
(−0.063 for Gemma, significant). Gemma wins on the penalised score (0.558 vs 0.418). Which
model is "better" depends on what a wrong answer costs.

## Judge check of the substring match

Kimi K3.1 Pro re-grades the correctness of every stored answer of the chosen variants
(`rag judge run RUN_ID --all-records`). On correctness it agrees with independent labels at
κ 0.889 ([`judge-calibration.md`](judge-calibration.md)). On support it is not reliable, so
it is used here for correctness only.

| Variant | Substring accuracy | Judge accuracy | Verdicts that differ |
|---|---:|---:|---:|
| 4B chosen (source-aware retrieval) | 0.775 | 0.792 [0.758, 0.822] | 0.074 |
| Gemma chosen | 0.712 | pending | |

The Judge credits a few correct answers that do not contain the reference verbatim. The
intervals overlap, so the substring match is not what drives the reported scores.

## Router headroom (Dev Split)

An oracle router that picks, per query, a variant that got it right gives an upper bound on
routing. Chance inflates this bound, because with more candidates some are right by luck.

| Candidates | Best single variant | Oracle router |
|---|---:|---:|
| 4B chosen and Gemma chosen | 0.770 | 0.858 (+0.088) |
| All 9 Dev answer runs | 0.776 | 0.912 (+0.136) |

A trained Router (logistic regression on the question embedding, 5-fold cross-validation on
Dev) does not reach this headroom. It learns to always pick the local 4B model, so it matches
the 4B model's 0.770. Its threshold rule only accepts routing that stays above the weaker
candidate's interval and then minimises cost. From the question alone it cannot tell which
model will be right, and only 23 of the 115 "Gemma only" questions are recognised. The Router
was therefore not scored on test-600, where it would reproduce the 4B row. Details:
[`router-4b-gemma.md`](router-4b-gemma.md).

## Reproduce

```bash
uv run rag table RUN_ID ... --always-yes
uv run rag compare BASELINE_RUN_ID CANDIDATE_RUN_ID
```

The stored runs live under `runs/`, one folder per run ID, named after the variant and split.
