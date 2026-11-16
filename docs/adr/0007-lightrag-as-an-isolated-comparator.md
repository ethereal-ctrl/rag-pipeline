# LightRAG as an isolated comparator

Use LightRAG only for the Dev Split GraphRAG experiment. Its graph extraction and retrieval provide a reference implementation for checking whether graph retrieval helps on inference and comparison Benchmark Queries. The Ablation Ladder and demo keep the project's own pipeline under ADR 0004.

LightRAG is a pinned optional dependency loaded by the graph commands. It owns the experiment's local graph files. The project's model clients, cache, tracing, answer prompt and Evaluate seam supply the measurements. Record the one-off build cost separately from per-query results. [LightRAG](https://github.com/HKUDS/LightRAG)
