"""Configuration shared by the demo API and its index bootstrap."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from rag_pipeline.chunking import ChunkingSettings
from rag_pipeline.models import EmbedderSpec
from rag_pipeline.variants import VariantConfig, router_candidates

VARIANTS_DIR = Path("configs/variants")
VARIANT_NAME_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$"
DEFAULT_DEMO_VARIANT = "hybrid-rerank-cited"
DEFAULT_QDRANT_URL = "http://localhost:6333"


def demo_variant() -> str:
    return os.environ.get("RAG_DEMO_VARIANT", DEFAULT_DEMO_VARIANT)


def load_variants(folder: Path) -> dict[str, VariantConfig]:
    variants: dict[str, VariantConfig] = {}
    for file in sorted(folder.glob("*.yaml")):
        if (
            file.is_symlink()
            or not file.is_file()
            or not re.fullmatch(VARIANT_NAME_PATTERN, file.stem)
        ):
            continue
        config = VariantConfig.from_yaml(file)
        if config.name != file.stem:
            raise ValueError(
                f"Variant name {config.name!r} must match the file name {file.name!r}."
            )
        variants[config.name] = config
    return variants


def answer_variants(config: VariantConfig) -> tuple[VariantConfig, ...]:
    return router_candidates(config.router) if config.router is not None else (config,)


@dataclass(frozen=True)
class IndexBuild:
    collection: str
    embedder: EmbedderSpec | None
    settings: ChunkingSettings
    context_model: str | None

    @classmethod
    def for_variant(cls, config: VariantConfig) -> IndexBuild:
        assert config.index is not None
        collection = config.index.collection
        match = re.match(r"chunks-sent-w([1-9]\d*)-o(\d*)(?:-|$)", collection)
        settings = config.index.chunking
        if settings is None:
            settings = (
                ChunkingSettings(int(match[1]), int(match[2])) if match else ChunkingSettings()
            )
        return cls(collection, config.index.embedder, settings, config.index.chunk_context)

    def ingest_command(self) -> str:
        args = ["rag", "ingest", "--collection", self.collection]
        if self.embedder is not None:
            args += [
                "--embedder",
                self.embedder.model,
                "--embedder-provider",
                self.embedder.provider,
            ]
        if self.settings != ChunkingSettings():
            args += [
                "--max-words",
                str(self.settings.max_words),
                "--overlap-sentences",
                str(self.settings.overlap_sentences),
            ]
        if self.settings.strategy == "semantic":
            args += [
                "--strategy",
                "semantic",
                "--semantic-percentile",
                str(self.settings.semantic_percentile),
            ]
        if self.context_model is not None:
            args += ["--chunk-context", self.context_model]
        return shlex.join(args)
