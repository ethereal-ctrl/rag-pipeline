# Zero-cost: no paid APIs, and consumer subscriptions never at runtime

The project has a budget of 0 €. The only paid access is the Claude Pro and ChatGPT/Codex consumer subscriptions, whose terms allow personal, dev-time use but not serving as an application's backend. So every model the running pipeline calls is either local (Apple M4, 16 GB) or on a free API tier. The subscriptions may be used only for one-off, manually started dev-time jobs, such as labelling a sample to calibrate the judge, and never inside the pipeline or the eval harness.

## Consequences

- Model size is capped by 16 GB of unified memory; the system will not reach frontier-model answer quality, and the write-up must compare variants against each other rather than against frontier numbers.
- Free-tier rate limits bound how many Benchmark Queries a remote model can process per day, which shapes the eval split sizes.
