"""Fit and validate the Router against stored Dev Split answers."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Protocol, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rag_pipeline.cache import ResponseCache, cache_key
from rag_pipeline.evaluation import QueryRecord, RunStore, bootstrap_intervals
from rag_pipeline.models import Embedder, model_version, query_embedding
from rag_pipeline.router import (
    FeatureSettings,
    LogisticModel,
    RouterArtifact,
    choose_candidate,
    question_features,
)
from rag_pipeline.variants import VariantConfig, router_candidates

_SEED = 0
_THRESHOLDS = tuple(step / 20 for step in range(21))
_THRESHOLD_RULE = (
    "Accept thresholds whose routed accuracy reaches the lower end of the always-most-expensive "
    "95% bootstrap interval. Minimize mean notional cost when all outcomes are priced, or mean "
    "latency otherwise. Break ties by higher accuracy, lower latency, then lower threshold."
)
_LIMITS = (
    "The Dev Split has about 500 queries. Small classes can give unstable held-out estimates.",
    "Labels can be imbalanced because they depend on the candidate order and stored correctness.",
    "Dev cross-validation is optimistic until the Router is evaluated on the Test Split.",
    "The simulation uses stored candidate outcomes. The live Router adds its own embedding "
    "latency and tokens.",
)


class RouterTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: Path
    runs: dict[str, str | None]
    folds: int = Field(default=5, ge=2)

    @classmethod
    def from_yaml(cls, path: Path) -> RouterTrainingConfig:
        return cls.model_validate(yaml.safe_load(path.read_text("utf-8")))


@dataclass(frozen=True)
class Estimate:
    value: float
    interval: tuple[float, float]


@dataclass(frozen=True)
class TrainingReport:
    artifact_hash: str
    training_runs: dict[str, str]
    queries: int
    seed: int
    folds: int
    threshold: float
    fold_thresholds: tuple[float, ...]
    threshold_rule: str
    label_distribution: dict[str, int]
    classifier_accuracy: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    candidates: dict[str, dict[str, Estimate]]
    outcomes: dict[str, dict[str, Estimate]]
    limits: tuple[str, ...]

    def markdown(self) -> str:
        lines = [
            "# Router Dev Split training",
            "",
            f"Artifact `{self.artifact_hash}`",
            "",
            f"{self.queries} queries, {self.folds} outer folds, seed {self.seed}.",
            f"Held-out classifier accuracy was {self.classifier_accuracy:.3f}.",
            f"The fitted threshold is {self.threshold:.2f}.",
            "",
            self.threshold_rule,
            "",
            "## Candidate runs",
            "",
            "| Candidate | Dev run | Labels | Accuracy | Mean latency (s) | Notional USD/query |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for name, metrics in self.candidates.items():
            lines.append(
                f"| {name} | `{self.training_runs[name]}` | {self.label_distribution[name]} | "
                + " | ".join(_format_estimate(metrics.get(metric)) for metric in _METRIC_NAMES)
                + " |"
            )
        lines += [
            "",
            "## Confusion matrix",
            "",
            "Rows are actual cheapest-correct labels and columns are classifier predictions.",
            "",
            "| Actual / predicted | " + " | ".join(self.candidates) + " |",
            "| --- | " + " | ".join("---:" for _ in self.candidates) + " |",
        ]
        for name, row in zip(self.candidates, self.confusion_matrix, strict=True):
            lines.append(f"| {name} | " + " | ".join(str(count) for count in row) + " |")
        lines += [
            "",
            "## Held-out simulated outcomes",
            "",
            "Estimates include 95% bootstrap intervals.",
            "",
            "| Policy | Accuracy | Mean latency (s) | Notional USD/query |",
            "| --- | ---: | ---: | ---: |",
        ]
        names = {
            "routed": "Router",
            "most_expensive": "Always most expensive",
            "oracle": "Cheapest correct oracle",
        }
        for policy, metrics in self.outcomes.items():
            lines.append(
                f"| {names[policy]} | "
                + " | ".join(_format_estimate(metrics.get(metric)) for metric in _METRIC_NAMES)
                + " |"
            )
        lines += ["", "## Limits", "", *(f"- {limit}" for limit in self.limits)]
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class TrainingResult:
    artifact: RouterArtifact
    report: TrainingReport
    artifact_path: Path

    def save(self, report_dir: Path) -> tuple[Path, Path, Path]:
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{self.artifact.digest}.json"
        markdown_path = json_path.with_suffix(".md")
        payload = json.dumps(asdict(self.report), indent=2, allow_nan=False)
        json_path.write_text(payload + "\n", "utf-8")
        markdown_path.write_text(self.report.markdown(), "utf-8")
        self.artifact.save(self.artifact_path)
        return self.artifact_path, json_path, markdown_path


@dataclass(frozen=True)
class _Example:
    features: tuple[float, ...]
    outcomes: tuple[QueryRecord, ...]
    label: int


def _candidate_records(
    store: RunStore, candidate: VariantConfig, run_id: str | None
) -> dict[str, QueryRecord]:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError(f"Candidate {candidate.name} needs a valid stored Dev run ID")
    description = f"Candidate {candidate.name} run {run_id}"
    try:
        meta = store.load_meta(run_id)
    except FileNotFoundError as error:
        raise ValueError(f"{description} is missing") from error
    if meta.get("split") != "dev":
        raise ValueError(f"{description} must be from the Dev Split (got {meta.get('split')!r})")
    if meta.get("retrieval_only"):
        raise ValueError(f"{description} is retrieval-only")
    try:
        run = store.load(run_id)
        stored_config = VariantConfig.model_validate(meta.get("variant"))
    except FileNotFoundError as error:
        raise ValueError(f"{description} is incomplete") from error
    except ValueError as error:
        raise ValueError(f"{description} contains invalid data") from error
    if stored_config.run_config() != candidate.run_config():
        raise ValueError(f"{description} does not match its candidate config")
    records = {row.query_id: row for row in run.records}
    if (
        not records
        or len(records) != len(run.records)
        or run.metrics.get("queries") != len(records)
        or any(row.correct is None for row in records.values())
    ):
        raise ValueError(f"{description} needs complete scored Benchmark Queries")
    return records


def _stratified_folds(examples: list[_Example], folds: int) -> list[list[int]]:
    count = min(folds, len(examples))
    if count < 2:
        return []
    strata: dict[int, list[int]] = {}
    for index, row in enumerate(examples):
        strata.setdefault(row.label, []).append(index)
    rng = random.Random(_SEED)
    held_out: list[list[int]] = [[] for _ in range(count)]
    offset = 0
    for label in sorted(strata):
        indices = strata[label]
        rng.shuffle(indices)
        for index in indices:
            held_out[offset % count].append(index)
            offset += 1
    return held_out


class _LogisticEstimator(Protocol):
    @property
    def coef_(self) -> Iterable[Iterable[float]]: ...

    @property
    def intercept_(self) -> Iterable[float]: ...

    def fit(self, features: list[tuple[float, ...]], labels: list[int]) -> object: ...


def _fit(examples: list[_Example]) -> LogisticModel:
    from sklearn.linear_model import LogisticRegression

    classes = tuple(sorted({row.label for row in examples}))
    zero_row = (0.0,) * len(examples[0].features)
    if len(classes) == 1:
        return LogisticModel(classes=classes, coefficients=(zero_row,), intercepts=(0.0,))
    model = cast(_LogisticEstimator, LogisticRegression(max_iter=1000, random_state=_SEED))
    model.fit([row.features for row in examples], [row.label for row in examples])
    weights = tuple(tuple(float(weight) for weight in row) for row in model.coef_)
    biases = tuple(float(bias) for bias in model.intercept_)
    if len(classes) == 2:
        # sklearn stores log odds for the positive class in the binary case.
        weights = (zero_row, *weights)
        biases = (0.0, *biases)
    return LogisticModel(classes=classes, coefficients=weights, intercepts=biases)


def _folds(examples: list[_Example], count: int):
    for indices in _stratified_folds(examples, count):
        held_out = set(indices)
        training = [row for index, row in enumerate(examples) if index not in held_out]
        model = _fit(training)
        yield (
            training,
            [
                (row, model.probabilities(row.features, len(row.outcomes)))
                for row in (examples[index] for index in indices)
            ],
        )


_METRIC_NAMES = ("accuracy", "latency_mean", "cost_per_question")


def _outcome_metrics(records: list[QueryRecord]) -> dict[str, float]:
    metrics = {
        "accuracy": fmean(float(bool(record.correct)) for record in records),
        "latency_mean": fmean(record.seconds for record in records),
    }
    costs = [record.notional_cost_usd for record in records if record.notional_cost_usd is not None]
    if len(costs) == len(records):
        metrics["cost_per_question"] = fmean(costs)
    return metrics


def _estimates(records: list[QueryRecord]) -> dict[str, Estimate]:
    intervals = bootstrap_intervals(records, metric_fn=_outcome_metrics)
    metrics = _outcome_metrics(records)
    return {key: Estimate(value, intervals[key]) for key, value in metrics.items()}


def _format_estimate(estimate: Estimate | None) -> str:
    if estimate is None:
        return "unavailable"
    low, high = estimate.interval
    return f"{estimate.value:.4g} [{low:.4g}, {high:.4g}]"


def _tune(predictions: list[tuple[_Example, tuple[float, ...]]]) -> float:
    if not predictions:
        return 1.0
    baseline = [row.outcomes[-1] for row, _ in predictions]
    floor, _ = bootstrap_intervals(baseline, metric_fn=_outcome_metrics)["accuracy"]
    priced = all(
        outcome.notional_cost_usd is not None for row, _ in predictions for outcome in row.outcomes
    )
    primary = "cost_per_question" if priced else "latency_mean"
    eligible: list[tuple[float, float, float, float]] = []
    for threshold in _THRESHOLDS:
        selected = [row.outcomes[choose_candidate(p, threshold)] for row, p in predictions]
        metrics = _outcome_metrics(selected)
        if metrics["accuracy"] >= floor:
            eligible.append(
                (metrics[primary], -metrics["accuracy"], metrics["latency_mean"], threshold)
            )
    return min(eligible)[-1]


def train_router(
    settings: RouterTrainingConfig,
    store: RunStore,
    embedder: Embedder,
    *,
    cache: ResponseCache | None = None,
) -> TrainingResult:
    variant = VariantConfig.from_yaml(settings.variant)
    routing = variant.router
    if routing is None:
        raise ValueError(f"{settings.variant} needs a Router variant config")
    assert variant.index is not None and variant.index.embedder is not None
    spec = variant.index.embedder
    candidates = router_candidates(routing)
    names = tuple(candidate.name for candidate in candidates)
    if unexpected := settings.runs.keys() - set(names):
        raise ValueError(f"Unknown Router candidates in runs: {', '.join(sorted(unexpected))}")
    runs: dict[str, str] = {}
    records: dict[str, dict[str, QueryRecord]] = {}
    for candidate in candidates:
        name = candidate.name
        run_id = settings.runs.get(name)
        rows = _candidate_records(store, candidate, run_id)
        assert run_id is not None
        if records:
            reference = records[names[0]]
            if rows.keys() != reference.keys() or any(
                row.question != reference[key].question for key, row in rows.items()
            ):
                raise ValueError(
                    f"Candidate {name} run {run_id} must cover the same Benchmark Queries "
                    f"as candidate {names[0]} run {runs[names[0]]}"
                )
        runs[name], records[name] = run_id, rows

    query_ids = sorted(records[names[0]])
    if len(query_ids) < 2:
        raise ValueError("Router training needs at least two Dev Split questions")
    examples: list[_Example] = []
    dimensions = 0
    revision = model_version(embedder)
    for query_id in query_ids:
        outcomes = tuple(records[name][query_id] for name in names)
        question = outcomes[0].question
        embedding, _ = query_embedding(embedder, spec, question, cache).value
        if (
            not embedding
            or (dimensions and len(embedding) != dimensions)
            or not all(math.isfinite(value) for value in embedding)
        ):
            raise ValueError(f"Invalid Router question embedding for {query_id}")
        dimensions = len(embedding)
        label = next((index for index, row in enumerate(outcomes) if row.correct), len(names) - 1)
        examples.append(_Example(question_features(question, embedding), outcomes, label))

    confusion = [[0] * len(names) for _ in names]
    fold_thresholds: list[float] = []
    routed: list[QueryRecord] = []
    held_out: list[tuple[_Example, tuple[float, ...]]] = []
    for training, predictions in _folds(examples, settings.folds):
        # The outer held-out questions and outcomes never enter the threshold search.
        threshold = _tune([item for _, inner in _folds(training, settings.folds) for item in inner])
        fold_thresholds.append(threshold)
        held_out.extend(predictions)
        for row, probabilities in predictions:
            predicted = max(enumerate(probabilities), key=lambda pair: pair[1])[0]
            confusion[row.label][predicted] += 1
            routed.append(row.outcomes[choose_candidate(probabilities, threshold)])

    artifact = RouterArtifact(
        feature_settings=FeatureSettings(
            embedder=spec, embedding_dimensions=dimensions, embedder_revision=revision
        ),
        candidate_order=names,
        candidate_config_hashes=tuple(cache_key(config.run_config()) for config in candidates),
        classifier=_fit(examples),
        threshold=_tune(held_out),
        training_runs=tuple(runs.values()),
        training_data_hash=cache_key(
            [
                (
                    row.outcomes[0].query_id,
                    row.outcomes[0].question,
                    row.features,
                    [(r.correct, r.seconds, r.notional_cost_usd) for r in row.outcomes],
                )
                for row in examples
            ]
        ),
    )
    labels = Counter(row.label for row in examples)
    report = TrainingReport(
        artifact_hash=artifact.digest,
        training_runs=runs,
        queries=len(examples),
        seed=_SEED,
        folds=len(fold_thresholds),
        threshold=artifact.threshold,
        fold_thresholds=tuple(fold_thresholds),
        threshold_rule=_THRESHOLD_RULE,
        label_distribution={name: labels[index] for index, name in enumerate(names)},
        classifier_accuracy=sum(row[i] for i, row in enumerate(confusion)) / len(examples),
        confusion_matrix=tuple(tuple(row) for row in confusion),
        candidates={
            name: _estimates([row.outcomes[index] for row in examples])
            for index, name in enumerate(names)
        },
        outcomes={
            "routed": _estimates(routed),
            "most_expensive": _estimates([row.outcomes[-1] for row in examples]),
            "oracle": _estimates([row.outcomes[row.label] for row in examples]),
        },
        limits=_LIMITS,
    )
    return TrainingResult(artifact, report, routing.artifact)
