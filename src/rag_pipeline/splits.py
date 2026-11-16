"""Fixed, stratified Dev / Smoke / Test / reserve splits of the Benchmark Queries (ADR 0005)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rag_pipeline.corpus import BenchmarkQuery

SPLIT_NAMES = ("dev", "smoke", "test", "reserve")
TEST_SPLITS = ("test", "test-400")
TEST_SUBSET_SIZE = 400
JUDGE_SAMPLE_SIZE = 300


@dataclass(frozen=True)
class SplitSizes:
    dev: int
    smoke: int
    test: int


DEFAULT_SIZES = SplitSizes(dev=500, smoke=50, test=1000)
DEFAULT_SEED = 20261115


@dataclass(frozen=True)
class Splits:
    seed: int
    dev: tuple[str, ...]
    smoke: tuple[str, ...]
    test: tuple[str, ...]
    reserve: tuple[str, ...]

    def ids(self, name: str) -> tuple[str, ...]:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
        ids: tuple[str, ...] = getattr(self, name)
        return ids

    def save(self, path: Path) -> None:
        payload = {"seed": self.seed, **{name: list(self.ids(name)) for name in SPLIT_NAMES}}
        path.write_text(json.dumps(payload, indent=1) + "\n", "utf-8")

    @classmethod
    def load(cls, path: Path) -> Splits:
        payload = json.loads(path.read_text("utf-8"))
        return cls(seed=payload["seed"], **{n: tuple(payload[n]) for n in SPLIT_NAMES})


def _allocate(total: int, group_sizes: dict[str, int]) -> dict[str, int]:
    """Split `total` across groups proportionally to their sizes (largest remainder)."""
    population = sum(group_sizes.values())
    exact = {g: total * n / population for g, n in group_sizes.items()}
    counts = {g: int(x) for g, x in exact.items()}
    leftover = total - sum(counts.values())
    for g in sorted(exact, key=lambda g: (counts[g] - exact[g], g))[:leftover]:
        counts[g] += 1
    return counts


def make_test_subset(queries: list[BenchmarkQuery], splits: Splits) -> tuple[str, ...]:
    return _stratified_sample(queries, splits.test, TEST_SUBSET_SIZE)


def make_judge_sample(
    queries: list[BenchmarkQuery], test_subset: tuple[str, ...]
) -> tuple[str, ...]:
    return _stratified_sample(queries, test_subset, JUDGE_SAMPLE_SIZE)


def _stratified_sample(
    queries: list[BenchmarkQuery], population: tuple[str, ...], size: int
) -> tuple[str, ...]:
    by_id = {q.id: q for q in queries}
    by_type: dict[str, list[str]] = defaultdict(list)
    for id_ in sorted(population):
        by_type[by_id[id_].query_type].append(id_)
    counts = _allocate(size, {name: len(ids) for name, ids in by_type.items()})
    rng = random.Random(DEFAULT_SEED)
    return tuple(id_ for name in sorted(by_type) for id_ in rng.sample(by_type[name], counts[name]))


def load_test_subset(path: Path, splits: Splits) -> tuple[str, ...]:
    data = json.loads(path.read_text("utf-8"))
    ids: tuple[str, ...] = tuple(data["test-400"])
    if (
        data["seed"] != DEFAULT_SEED
        or len(ids) != TEST_SUBSET_SIZE
        or len(set(ids)) != TEST_SUBSET_SIZE
        or not set(ids).issubset(splits.test)
    ):
        raise ValueError("invalid committed 400-query Test Split subset")
    return ids


def load_judge_sample(path: Path, test_subset: tuple[str, ...]) -> tuple[str, ...]:
    data = json.loads(path.read_text("utf-8"))
    ids: tuple[str, ...] = tuple(data["judge-300"])
    if (
        data["seed"] != DEFAULT_SEED
        or len(ids) != JUDGE_SAMPLE_SIZE
        or len(set(ids)) != JUDGE_SAMPLE_SIZE
        or not set(ids).issubset(test_subset)
    ):
        raise ValueError("invalid committed Judge sample of the 400-query Test Split subset")
    return ids


def make_splits(queries: list[BenchmarkQuery], sizes: SplitSizes, seed: int) -> Splits:
    if sizes.dev + sizes.test > len(queries) or sizes.smoke > sizes.dev:
        raise ValueError(f"split sizes {sizes} do not fit {len(queries)} queries")
    rng = random.Random(seed)
    by_type: dict[str, list[str]] = defaultdict(list)
    for query in sorted(queries, key=lambda q: q.id):
        by_type[query.query_type].append(query.id)
    for ids in by_type.values():
        rng.shuffle(ids)

    group_sizes = {t: len(ids) for t, ids in by_type.items()}
    dev_counts = _allocate(sizes.dev, group_sizes)
    test_counts = _allocate(sizes.test, group_sizes)
    smoke_counts = _allocate(sizes.smoke, dev_counts)

    dev: list[str] = []
    smoke: list[str] = []
    test: list[str] = []
    reserve: list[str] = []
    for query_type in sorted(by_type):
        ids = by_type[query_type]
        n_dev, n_test = dev_counts[query_type], test_counts[query_type]
        dev += ids[:n_dev]
        smoke += ids[: smoke_counts[query_type]]
        test += ids[n_dev : n_dev + n_test]
        reserve += ids[n_dev + n_test :]
    return Splits(seed, tuple(dev), tuple(smoke), tuple(test), tuple(reserve))
