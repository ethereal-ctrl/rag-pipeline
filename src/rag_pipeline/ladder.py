"""Validate the Ablation Ladder and schedule its remaining held-out evaluations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rag_pipeline.corpus import BenchmarkQuery
from rag_pipeline.evaluation import Evaluable, RunResult, RunStore, StoredRun, evaluate
from rag_pipeline.rate_limit import RemoteProvider
from rag_pipeline.variants import VariantConfig

type HeldOutSplit = Literal["test", "test-400"]


class _Definition(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _RungDefinition(_Definition):
    label: str = Field(min_length=1)
    config: Path
    exception: str | None = None


class _PairDefinition(_Definition):
    local: Path
    config: Path


class _LargeModelDefinition(_Definition):
    split: Literal["test-400"]
    provider: RemoteProvider = "groq"  # Compatibility with older ladder configs.
    reason: str = Field(min_length=1)
    pairs: tuple[_PairDefinition, ...]
    alternatives: dict[RemoteProvider, tuple[_PairDefinition, ...]] = Field(
        default_factory=dict[RemoteProvider, tuple[_PairDefinition, ...]]
    )


class _LadderDefinition(_Definition):
    rungs: tuple[_RungDefinition, ...] = Field(min_length=1)
    large_model: _LargeModelDefinition


@dataclass(frozen=True)
class LadderRung:
    label: str
    path: Path
    config: VariantConfig
    exception: str | None = None


@dataclass(frozen=True)
class LargeModelPair:
    local: LadderRung
    remote: LadderRung


@dataclass(frozen=True)
class Ladder:
    rungs: tuple[LadderRung, ...]
    large_model: tuple[LargeModelPair, ...] = ()
    large_model_reason: str = ""
    large_model_provider: RemoteProvider = "groq"
    alternatives: dict[RemoteProvider, tuple[LargeModelPair, ...]] = field(
        default_factory=dict[RemoteProvider, tuple[LargeModelPair, ...]]
    )

    @classmethod
    def load(cls, path: Path) -> Ladder:
        data = _LadderDefinition.model_validate(yaml.safe_load(path.read_text("utf-8")))
        rungs = tuple(
            LadderRung(
                item.label, item.config, VariantConfig.from_yaml(item.config), item.exception
            )
            for item in data.rungs
        )
        validate_rungs(rungs)
        by_path = {rung.path: rung for rung in rungs}
        used = {rung.config.name for rung in rungs}

        def load_pairs(
            items: tuple[_PairDefinition, ...],
            provider: RemoteProvider,
        ) -> tuple[LargeModelPair, ...]:
            pairs: list[LargeModelPair] = []
            for item in items:
                if item.local not in by_path:
                    raise ValueError(
                        f"Large-model counterpart {item.local} is missing from the ladder."
                    )
                local = by_path[item.local]
                config = VariantConfig.from_yaml(item.config)
                if config.name in used:
                    raise ValueError(f"Duplicate ladder variant name {config.name!r}.")
                used.add(config.name)
                spec = config.generation.generator
                if (
                    local.config.router is not None
                    or spec.provider != provider
                    or spec.model.endswith("-latest")
                    or config.model_copy(
                        update={
                            "name": local.config.name,
                            "generation": config.generation.model_copy(
                                update={"generator": local.config.generation.generator}
                            ),
                        }
                    )
                    != local.config
                ):
                    raise ValueError(
                        f"{item.config} must swap only {item.local}'s generator "
                        f"for a pinned {provider.title()} model."
                    )
                pairs.append(LargeModelPair(local, LadderRung(local.label, item.config, config)))
            return tuple(pairs)

        large = data.large_model
        if large.provider in large.alternatives:
            raise ValueError("The default large-model provider is also listed as an alternative.")
        return cls(
            rungs,
            load_pairs(large.pairs, large.provider),
            large.reason,
            large.provider,
            {
                provider: load_pairs(items, provider)
                for provider, items in large.alternatives.items()
            },
        )

    def large_pairs(self, provider: RemoteProvider | None = None) -> tuple[LargeModelPair, ...]:
        if provider is None or provider == self.large_model_provider:
            return self.large_model
        if provider not in self.alternatives:
            raise ValueError(f"No large-model rungs configured for {provider.title()}.")
        return self.alternatives[provider]

    def for_split(
        self,
        split: HeldOutSplit,
        provider: RemoteProvider | None = None,
    ) -> tuple[LadderRung, ...]:
        return (
            self.rungs
            if split == "test"
            else tuple(pair.remote for pair in self.large_pairs(provider))
        )


def validate_rungs(rungs: tuple[LadderRung, ...]) -> None:
    if len({rung.config.name for rung in rungs}) != len(rungs):
        raise ValueError("Ladder variant names must be unique.")
    for rung in rungs:
        if rung.exception and (rung != rungs[-1] or rung.config.router is None):
            raise ValueError("Only the final Router rung can declare an exception.")
    for previous, current in pairwise(rungs):
        if current.exception:
            continue
        before = previous.config.model_dump(mode="json", exclude={"name"})
        after = current.config.model_dump(mode="json", exclude={"name"})
        changed = sorted(key for key in before if before[key] != after[key])
        if len(changed) != 1:
            fields = ", ".join(changed) or "none"
            raise ValueError(
                f"{previous.label} → {current.label} must change one field group. Found {fields}."
            )


@dataclass(frozen=True)
class RungStatus:
    rung: LadderRung
    state: Literal["complete", "in progress", "missing", "blocked"]
    run: StoredRun | None
    missing: tuple[str, ...] = ()


def _status(
    rung: LadderRung,
    split: HeldOutSplit,
    runs: list[StoredRun],
    missing: Callable[[LadderRung], tuple[str, ...]] | None,
) -> RungStatus:
    same = [
        run
        for run in runs
        if run.meta["variant"]["name"] == rung.config.name
        and run.meta["split"] in ("test", "test-400")
    ]
    completed = [run for run in same if run.complete and run.meta["split"] == split]
    if completed:
        # Prefer the original completed score if an explicit override was later used.
        selected = min(completed, key=lambda run: (bool(run.meta.get("rescored")), run.run_id))
        return RungStatus(rung, "complete", selected)
    if other := next((run for run in same if run.complete), None):
        return RungStatus(rung, "blocked", other, (f"already scored on {other.meta['split']}",))
    partial = next((run for run in reversed(same) if run.meta["split"] == split), None)
    gaps = missing(rung) if missing is not None else ()
    return RungStatus(rung, "in progress" if partial else "missing", partial, gaps)


def ladder_status(
    rungs: tuple[LadderRung, ...],
    split: HeldOutSplit,
    store: RunStore,
    *,
    missing: Callable[[LadderRung], tuple[str, ...]] | None = None,
) -> tuple[RungStatus, ...]:
    runs = store.runs()
    return tuple(_status(rung, split, runs, missing) for rung in rungs)


def run_ladder(
    rungs: tuple[LadderRung, ...],
    split: HeldOutSplit,
    store: RunStore,
    queries: list[BenchmarkQuery],
    build: Callable[[LadderRung], Evaluable],
) -> Iterator[RunResult]:
    for rung in rungs:
        status = _status(rung, split, store.runs(), None)
        if status.state == "complete":
            continue
        if status.state == "blocked":
            raise RuntimeError(f"{rung.label}: {status.missing[0]}")
        try:
            yield evaluate(build(rung), queries, split, store)
        except Exception as error:
            raise RuntimeError(f"{rung.label} ({rung.config.name}): {error}") from error


@dataclass(frozen=True)
class RungEstimate:
    status: RungStatus
    seconds: float | None = None
    dev_run_id: str | None = None


def estimate_ladder(
    rungs: tuple[LadderRung, ...], split: HeldOutSplit, store: RunStore, count: int
) -> tuple[RungEstimate, ...]:
    dev = {
        run.meta["variant"]["name"]: run.run_id
        for run in store.runs()
        if run.complete and run.meta["split"] == "dev" and not run.meta.get("retrieval_only")
    }
    estimates: list[RungEstimate] = []
    for status in ladder_status(rungs, split, store):
        run_id = dev.get(status.rung.config.name)
        seconds = None
        if status.state in ("missing", "in progress") and run_id is not None:
            p50 = store.load(run_id).metrics.get("latency_p50")
            if p50 is not None:
                seconds = p50 * count
        estimates.append(RungEstimate(status, seconds, run_id))
    return tuple(estimates)
