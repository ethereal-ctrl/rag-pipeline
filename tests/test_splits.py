from pathlib import Path

from rag_pipeline.corpus import BenchmarkQuery
from rag_pipeline.splits import SplitSizes, make_splits, untouched_test_ids


def _queries(per_type: dict[str, int]) -> list[BenchmarkQuery]:
    return [
        BenchmarkQuery(id=f"{t}-{i}", question=f"{t} {i}?", answer="x", query_type=t, evidence=())
        for t, n in per_type.items()
        for i in range(n)
    ]


SIZES = SplitSizes(dev=20, smoke=5, test=40)
QUERIES = _queries({"inference_query": 40, "comparison_query": 30, "null_query": 30})


def test_splits_are_disjoint_and_sized_with_rest_in_reserve():
    splits = make_splits(QUERIES, SIZES, seed=7)

    assert len(splits.dev) == 20
    assert len(splits.test) == 40
    assert len(splits.reserve) == 40
    assert set(splits.dev).isdisjoint(splits.test)
    assert set(splits.dev).isdisjoint(splits.reserve)
    assert set(splits.test).isdisjoint(splits.reserve)


def test_smoke_set_lies_inside_dev_split():
    splits = make_splits(QUERIES, SIZES, seed=7)

    assert len(splits.smoke) == 5
    assert set(splits.smoke) <= set(splits.dev)


def test_splits_are_stratified_by_query_type():
    splits = make_splits(QUERIES, SIZES, seed=7)
    type_of = {q.id: q.query_type for q in QUERIES}

    test_types = [type_of[i] for i in splits.test]
    # 40 % / 30 % / 30 % of 40 queries
    assert test_types.count("inference_query") == 16
    assert test_types.count("comparison_query") == 12
    assert test_types.count("null_query") == 12


def test_same_seed_gives_same_splits_and_other_seed_differs():
    assert make_splits(QUERIES, SIZES, seed=7) == make_splits(QUERIES, SIZES, seed=7)
    assert make_splits(QUERIES, SIZES, seed=7).test != make_splits(QUERIES, SIZES, seed=8).test


def test_splits_round_trip_through_their_file(tmp_path: Path):
    splits = make_splits(QUERIES, SIZES, seed=7)
    path = tmp_path / "splits.json"

    splits.save(path)

    assert type(splits).load(path) == splits


def test_the_untouched_test_ids_are_the_test_split_minus_its_subset():
    splits = make_splits(QUERIES, SIZES, seed=7)
    subset = splits.test[:15]

    untouched = untouched_test_ids(splits, subset)

    assert untouched == splits.test[15:]
    assert set(untouched).isdisjoint(subset)
