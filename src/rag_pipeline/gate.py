"""The quality gate: Smoke Set metrics must stay at or above committed thresholds.

CI runs the retrieval part (no LLM, no secrets); `rag gate --answers` adds the
answer-level checks locally, where the generator runs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from rag_pipeline.corpus import BenchmarkQuery
from rag_pipeline.evaluation import Retrieving, retrieval_metrics

Minimums = dict[str, dict[str, float]]  # variant name -> metric -> lowest allowed value


class GateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str = "smoke"
    variants: list[Path]  # Pipeline Variant configs, relative to the gate file
    retrieval: Minimums
    answers: Minimums = {}

    @classmethod
    def from_yaml(cls, path: Path) -> GateConfig:
        config = cls.model_validate(yaml.safe_load(path.read_text("utf-8")))
        return config.model_copy(update={"variants": [path.parent / v for v in config.variants]})


@dataclass(frozen=True)
class GateReport:
    metrics: dict[str, dict[str, float]]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def check_minimums(metrics: dict[str, dict[str, float]], minimums: Minimums) -> GateReport:
    failures: list[str] = []
    for variant, floors in minimums.items():
        for metric, floor in floors.items():
            value = metrics.get(variant, {}).get(metric)
            if value is None or math.isnan(value):
                failures.append(f"{variant} {metric} has no value (threshold {floor:.3f})")
            elif value < floor:
                failures.append(
                    f"{variant} {metric} = {value:.3f} is below the threshold {floor:.3f}"
                )
    return GateReport(metrics=metrics, failures=tuple(failures))


def retrieval_gate(
    variants: Sequence[Retrieving], queries: list[BenchmarkQuery], minimums: Minimums
) -> GateReport:
    """Score each variant's retrieval on the queries and check it against the minimums."""
    metrics = {v.config.name: retrieval_metrics(v, queries) for v in variants}
    return check_minimums(metrics, minimums)
