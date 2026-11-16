# Mistral for the large-model comparison and the Judge

Groq Free Plan allows 200,000 tokens per day for GPT-OSS 120B. At that cap, a large-model RAG rung is estimated to take about four days and Agentic Search up to 30 days. Grading with the same model draws on the same daily quota. [Groq rate limits](https://console.groq.com/docs/rate-limits)

## Decision

Default the large-model rungs and the Judge to Mistral Large 4 (`mistral-large-4-0`), a versioned API model with JSON Schema support. The standard reference rates on 2026-11-16 are $0.70 per million input tokens and $2.10 per million output tokens. [Model card and rates](https://docs.mistral.ai/models/mistral-large-4-0) · [Versioned ID](https://docs.mistral.ai/vibe/code/cli/configuration) · [Chat API](https://docs.mistral.ai/api/endpoint/chat)

Keep the zero-cost requirement in [ADR 0001](0001-zero-cost-no-subscription-runtime.md). On 2026-11-16 the public plan documentation lists Free mode without API access and gives organization-specific limits. A no-cost API entitlement and the previously reported Experiment quotas remain unverified. The model is pinned as the candidate for an organization with confirmed no-cost access. The quota configuration requires verified values from that organization before the runtime can use Mistral. Recalculate the run estimate once those values are available. [Mistral plan and quota documentation](https://docs.mistral.ai/admin/billing-usage/usage-limits)

## Alternatives

- Groq keeps the existing GPT-OSS 120B benchmark and Judge available, with the daily cap and its scheduling cost.
- Cerebras would require verifying a no-cost entitlement, a reproducible model with structured output, and sufficient quota for both workloads.
- A smaller held-out subset would reduce remote time and cost, widen the uncertainty intervals, and give up the fixed `test-400` comparison.

## Consequences

- The Judge checks generator families before grading. Use Mistral to judge Qwen or GPT-OSS runs. Use the Groq Judge to grade Mistral runs.
- Groq remains a selectable large-model provider and Judge. The Mistral generator and Judge share the persistent quota when they use the same cache database. The local limiter uses a rolling 31-day window for monthly caps.
- Mistral's former Experiment tier was reported to require phone verification. Its Free-mode data policy allows use of request data for training and offers an opt-out setting. Remote benchmarking sends public benchmark inputs and their generated answers. The demo keeps a local default. [Former Experiment help article (archive link)](https://web.archive.org/web/20260601000000/https://help.mistral.ai/en/articles/455206-how-can-i-try-the-api-for-free-with-the-experiment-plan) · [Mistral data policy](https://help.mistral.ai/en/articles/347617-do-you-use-my-user-data-to-train-your-artificial-intelligence-models)
