# Gemma for the large-model comparison and the Judge

Groq Free Plan allows 200,000 tokens per day for GPT-OSS 120B. At that cap, a large-model RAG rung is estimated to take about four days and Agentic Search up to 30 days. Judging with GPT-OSS 120B draws on the same daily quota. [Groq rate limits](https://console.groq.com/docs/rate-limits)

## Decision

Use Gemma 4 31B (`gemma-4-31b-it`) through the Gemini API free tier for the three large-model rungs and as the default Judge. This is the exact ID listed by Google on 2026-11-16. Google documents system instructions for Gemma 4. Its Gemma API guide leaves JSON-schema mode unspecified, so we include the schema in the prompt and validate the extracted JSON. Both capabilities are configurable per model. [Gemma on the Gemini API](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api) · [OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)

## Alternatives

- Groq remains selectable. Its daily token cap stays in effect.
- Mistral offered the author no free API access. [Mistral plans and limits](https://docs.mistral.ai/admin/billing-usage/usage-limits)
- The Gemini Flash models showed 20 requests per day on the author's project on 2026-11-16.
- Cerebras would require confirming its free-tier model, capacity, and structured output.

## Consequences

- The author's Google AI Studio project showed 30 requests/minute, 16,000 tokens/minute, and 14,400 requests/day for both Gemma 4 31B and 26B on 2026-11-16. It showed no daily token cap for these models. Limits are per project and can change. At sustained throughput the benchmark and Judge workloads are estimated at about a day. [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- Google may use free-tier prompts and outputs to improve products, with regional exceptions described in its terms. Check regional availability and project-level access before running. Google's availability list includes Germany and other EU countries. The runtime uses the Gemini API key. [Gemini API data terms](https://ai.google.dev/gemini-api/terms) · [Available regions](https://ai.google.dev/gemini-api/docs/available-regions)
- The consumer Gemini subscription is outside the runtime under ADR 0001. Remote benchmark requests use public benchmark inputs and their generated answers. The demo defaults to local models.
- Use the Groq Judge for Gemma runs and the Gemma Judge for Qwen and GPT-OSS runs. The Judge checks generator families before grading.
