# RAG Pipeline: a measured Ablation Ladder on MultiHop-RAG

Work in progress. The goal: measure, with confidence intervals, which advanced retrieval
techniques actually pay off on multi-hop questions, running fully locally.
See `GLOSSARY.md` for the vocabulary and `docs/adr/` for the design decisions.

## Quick start

```bash
uv sync
docker compose up -d          # Qdrant
ollama pull qwen3.5:4b        # local generator
uv run rag data fetch         # pinned MultiHop-RAG files, checksum-verified
uv run rag ingest             # chunk + index the Corpus
uv run rag eval configs/variants/bm25.yaml --split smoke
```

Runs are stored under `runs/` with their config, split, model version and commit.
The fixed splits are committed in `data/splits.json` and never regenerated (ADR 0005).

## Development

```bash
uv run pytest        # seam tests with a fixture Corpus and fake models, no network
uv run ruff check src tests
uv run pyright
```

## Data

MultiHop-RAG by Yixuan Tang and Yi Yang (arXiv:2401.15391), used under the
[ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/) licence, pinned to Hugging Face
revision `71ac0d0b` of `yixuantt/MultiHopRAG`.
