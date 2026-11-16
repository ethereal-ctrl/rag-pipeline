"""Question features and JSON inference for the Router."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PositiveInt, model_validator

from rag_pipeline.cache import cache_key
from rag_pipeline.models import EmbedderSpec

_WORD = re.compile(r"\b\w+\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_COMPARISONS = frozenset({"both", "same", "while", "whereas", "than", "compare", "versus"})
_TEXT_FEATURES = ("log_word_count", "log_digit_count", "has_year", "has_comparison_word")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RouterConfig(_Frozen):
    artifact: Path
    candidates: tuple[Path, ...] = Field(min_length=2)


class FeatureSettings(_Frozen):
    embedder: EmbedderSpec
    embedding_dimensions: PositiveInt
    embedder_revision: str | None = None
    version: Literal[1] = 1
    text_features: tuple[str, ...] = _TEXT_FEATURES

    @model_validator(mode="after")
    def validate_features(self) -> Self:
        if self.text_features != _TEXT_FEATURES:
            raise ValueError("unsupported Router text features")
        return self

    @property
    def dimensions(self) -> int:
        return self.embedding_dimensions + len(self.text_features)


def question_features(question: str, embedding: list[float]) -> tuple[float, ...]:
    """Only the incoming question and its query embedding are visible to the Router."""
    words = _WORD.findall(question.lower())
    return (
        *embedding,
        math.log1p(len(words)) / 4,
        math.log1p(sum(character.isdigit() for character in question)) / 4,
        float(bool(_YEAR.search(question))),
        float(not _COMPARISONS.isdisjoint(words)),
    )


class LogisticModel(_Frozen):
    classes: tuple[int, ...] = Field(min_length=1)
    coefficients: tuple[tuple[FiniteFloat, ...], ...]
    intercepts: tuple[FiniteFloat, ...]

    @model_validator(mode="after")
    def validate_rows(self) -> Self:
        if tuple(sorted(set(self.classes))) != self.classes or self.classes[0] < 0:
            raise ValueError("Router classifier classes must be sorted unique candidate indices")
        if len(self.classes) != len(self.coefficients) or len(self.classes) != len(self.intercepts):
            raise ValueError("Router classifier needs one coefficient row and intercept per class")
        return self

    def probabilities(self, features: tuple[float, ...], candidates: int) -> tuple[float, ...]:
        scores = [
            math.fsum(weight * value for weight, value in zip(row, features, strict=True)) + bias
            for row, bias in zip(self.coefficients, self.intercepts, strict=True)
        ]
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        total = math.fsum(weights)
        probabilities = [0.0] * candidates
        for label, weight in zip(self.classes, weights, strict=True):
            probabilities[label] = weight / total
        return tuple(probabilities)


def choose_candidate(probabilities: tuple[float, ...], threshold: float) -> int:
    cumulative = 0.0
    for index, probability in enumerate(probabilities[:-1]):
        cumulative += probability
        if min(1.0, cumulative) > threshold:
            return index
    return len(probabilities) - 1


class RouterArtifact(_Frozen):
    feature_settings: FeatureSettings
    candidate_order: tuple[str, ...] = Field(min_length=2)
    candidate_config_hashes: tuple[str, ...]
    classifier: LogisticModel
    threshold: float = Field(ge=0, le=1)
    training_runs: tuple[str, ...]
    training_data_hash: str

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        count = len(self.candidate_order)
        if len(set(self.candidate_order)) != count or len(self.candidate_config_hashes) != count:
            raise ValueError("Router artifact candidate names and configs must match")
        if len(self.training_runs) != count or self.classifier.classes[-1] >= count:
            raise ValueError("Router artifact training runs or classes do not match the candidates")
        if any(
            len(row) != self.feature_settings.dimensions for row in self.classifier.coefficients
        ):
            raise ValueError("Router artifact coefficients do not match its feature settings")
        return self

    @classmethod
    def load(cls, path: Path) -> RouterArtifact:
        if not path.is_file():
            raise ValueError(f"Router artifact is missing at {path}. Train the Router first.")
        return cls.model_validate_json(path.read_text("utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(self.model_dump_json(indent=2) + "\n", "utf-8")
        temporary.replace(path)

    @property
    def digest(self) -> str:
        return cache_key(self.model_dump(mode="json"))

    def choose(self, question: str, embedding: list[float]) -> int:
        if len(embedding) != self.feature_settings.embedding_dimensions:
            raise ValueError("Router query embedding dimensions do not match the trained artifact")
        features = question_features(question, embedding)
        return choose_candidate(
            self.classifier.probabilities(features, len(self.candidate_order)), self.threshold
        )
