# Router Dev Split training

Artifact `05e2030a7dccb9bad6365cb656fa78cde1d08f937916eb684076fe56b4fe5d04`

500 queries, 5 outer folds, seed 0.
Held-out classifier accuracy was 0.770.
The fitted threshold is 0.00.

Accept thresholds whose routed accuracy reaches the lower end of the always-most-expensive 95% bootstrap interval. Minimize mean notional cost when all outcomes are priced, or mean latency otherwise. Break ties by higher accuracy, lower latency, then lower threshold.

## Candidate runs

| Candidate | Dev run | Labels | Accuracy | Mean latency (s) | Notional USD/query |
| --- | --- | ---: | ---: | ---: | ---: |
| hybrid-rerank-sources | `20261119T020309625032-hybrid-rerank-sources-dev` | 385 | 0.77 [0.732, 0.8061] | 11.93 [11.76, 12.13] | 0.0004008 [0.0003974, 0.0004043] |
| hybrid-rerank-gemma-nvidia-sources-ctx10-decisive | `20261119T004758456378-hybrid-rerank-gemma-nvidia-sources-ctx10-decisive-dev` | 115 | 0.698 [0.656, 0.736] | 23.13 [22.05, 24.32] | 0.0005308 [0.0005273, 0.0005344] |

## Confusion matrix

Rows are actual cheapest-correct labels and columns are classifier predictions.

| Actual / predicted | hybrid-rerank-sources | hybrid-rerank-gemma-nvidia-sources-ctx10-decisive |
| --- | ---: | ---: |
| hybrid-rerank-sources | 362 | 23 |
| hybrid-rerank-gemma-nvidia-sources-ctx10-decisive | 92 | 23 |

## Held-out simulated outcomes

Estimates include 95% bootstrap intervals.

| Policy | Accuracy | Mean latency (s) | Notional USD/query |
| --- | ---: | ---: | ---: |
| Router | 0.77 [0.734, 0.806] | 11.93 [11.77, 12.14] | 0.0004008 [0.0003972, 0.0004042] |
| Always most expensive | 0.698 [0.656, 0.736] | 23.13 [22.05, 24.32] | 0.0005308 [0.0005273, 0.0005344] |
| Cheapest correct oracle | 0.858 [0.824, 0.8841] | 14.18 [13.61, 14.86] | 0.0004312 [0.0004258, 0.0004371] |

## Limits

- The Dev Split has about 500 queries. Small classes can give unstable held-out estimates.
- Labels can be imbalanced because they depend on the candidate order and stored correctness.
- Dev cross-validation is optimistic until the Router is evaluated on the Test Split.
- The simulation uses stored candidate outcomes. The live Router adds its own embedding latency and tokens.
