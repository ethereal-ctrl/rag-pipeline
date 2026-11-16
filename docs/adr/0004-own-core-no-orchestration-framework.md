# Own pipeline core and agent loop, no orchestration framework

The retrieval pipeline and the Agentic Search loop are plain Python we write ourselves, not LangChain, LlamaIndex or LangGraph. Benchmarks show frameworks do not change answer quality, they add upgrade breakage, and the project exists to show understanding of each stage; a small hand-written loop is fully traceable and easy to explain. Libraries are still used for single concerns (model clients, Qdrant client, sentence-transformers, OpenTelemetry).
