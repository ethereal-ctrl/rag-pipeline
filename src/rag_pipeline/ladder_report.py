"""Assemble the ladder report from saved outcomes and Dev reference answers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from rag_pipeline.corpus import NULL_QUERY_TYPE, BenchmarkQuery
from rag_pipeline.evaluation import (
    QueryRecord,
    RunResult,
    RunStore,
    StoredRun,
    aggregate,
    answer_is_correct,
    bootstrap_intervals,
    compare_runs,
)
from rag_pipeline.judge_evaluation import with_judge_metrics, write_text
from rag_pipeline.ladder import HeldOutSplit, Ladder, LadderProvider, LadderRung, ladder_status
from rag_pipeline.report import metric_cell, metric_number
from rag_pipeline.variants import ABSTENTION_TEXT, VariantConfig

METRICS = {
    "hits@4": "Hits@4",
    "mrr@10": "MRR@10",
    "evidence_recall@10": "Evidence Recall@10",
    "accuracy": "Accuracy",
    "abstention_rate[null]": "Null Abstention",
    "false_abstention_rate": "False Abstention",
    "citation_validity": "Citation validity",
    "citation_evidence_precision": "Citation evidence precision",
    "judge_accuracy": "Judge accuracy",
    "judge_flipped": "Judge flipped",
    "citation_support_rate": "Judge citation support",
    "latency_p50": "p50 latency (s)",
    "latency_p95": "p95 latency (s)",
    "generator_context_tokens_per_question": "Generator context tokens/query",
    "cost_per_question": "Notional USD/query",
}
_STEP_METRICS = ("accuracy", "hits@4", "mrr@10", "latency_p50", "latency_p95")
_LARGE_METRICS = (
    "accuracy",
    "hits@4",
    "mrr@10",
    "evidence_recall@10",
    "latency_p50",
    "latency_p95",
    "cost_per_question",
)


def majority_baseline(dev: tuple[QueryRecord, ...], test: tuple[QueryRecord, ...]) -> RunResult:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in dev:
        if record.query_type != NULL_QUERY_TYPE:
            counts[record.query_type][record.reference_answer] += 1
    answers = {
        kind: min(frequencies, key=lambda answer: (-frequencies[answer], answer))
        for kind, frequencies in counts.items()
    }
    rows: list[QueryRecord] = []
    for record in test:
        query = BenchmarkQuery(
            record.query_id, record.question, record.reference_answer, record.query_type, ()
        )
        answer = answers.get(query.query_type, ABSTENTION_TEXT)
        abstained = answer == ABSTENTION_TEXT
        rows.append(
            QueryRecord(
                query.id,
                query.query_type,
                query.question,
                query.answer,
                answer,
                abstained,
                answer_is_correct(query, answer, abstained),
                None,
                None,
                None,
                None,
                None,
                (),
                0,
                0,
                0.0,
            )
        )

    def baseline_metrics(records: list[QueryRecord]) -> dict[str, float]:
        return {
            name: value
            for name, value in aggregate(records).items()
            if name.startswith(("accuracy", "abstention_rate", "false_abstention_rate"))
            or name == "queries"
        }

    return RunResult(
        "dev-majority",
        {},
        baseline_metrics(rows),
        bootstrap_intervals(rows, metric_fn=baseline_metrics),
        tuple(rows),
    )


def _references(
    store: RunStore, runs: list[StoredRun], split: str, ids: tuple[str, ...]
) -> tuple[tuple[QueryRecord, ...], tuple[str, ...]]:
    wanted = set(ids)
    found: dict[str, QueryRecord] = {}
    sources: list[str] = []
    for run in reversed(runs):
        if not run.complete or run.meta["split"] != split or run.meta.get("retrieval_only"):
            continue
        missing = wanted - found.keys()
        if not missing:
            break
        available = {
            record.query_id: record
            for record in store.load(run.run_id).records
            if record.query_id in missing
        }
        if available:
            found.update(available)
            sources.append(run.run_id)
    if found.keys() != wanted:
        return (), ()
    return tuple(found[id_] for id_ in ids), tuple(sources)


def _select_rows(
    run: RunResult,
    ids: set[str],
    *,
    keep_judge_metrics: bool = False,
) -> RunResult | None:
    records = [record for record in run.records if record.query_id in ids]
    if not records:
        return None
    if len(records) == len(run.records):
        return run
    metrics = aggregate(records)
    intervals = bootstrap_intervals(records)
    if keep_judge_metrics and len(records) == len(ids):
        # The fixed Judge sample is contained in the full test-400 subset.
        for metric in ("judge_accuracy", "judge_flipped", "citation_support_rate"):
            if metric in run.metrics:
                metrics[metric] = run.metrics[metric]
                intervals[metric] = run.intervals[metric]
    return replace(
        run,
        records=tuple(records),
        metrics=metrics,
        intervals=intervals,
    )


def _completed(
    store: RunStore, rungs: tuple[LadderRung, ...], split: HeldOutSplit
) -> dict[str, RunResult]:
    return {
        row.rung.config.name: with_judge_metrics(store.load(row.run.run_id), store)
        for row in ladder_status(rungs, split, store)
        if row.state == "complete" and row.run is not None
    }


def _compatible(rung: LadderRung, run: RunResult) -> bool:
    return (
        VariantConfig.model_validate(run.meta["variant"]).run_config() == rung.config.run_config()
    )


def _step(previous: RunResult | None, run: RunResult | None, metrics: tuple[str, ...]) -> str:
    if previous is None or run is None:
        return "—"
    try:
        delta_by_metric = {delta.metric: delta for delta in compare_runs(previous, run)}
    except ValueError:
        return "No shared queries"
    cells: list[str] = []
    for key in metrics:
        if (delta := delta_by_metric.get(key)) is not None:
            point, low, high = (
                metric_number(key, value, signed=True)
                for value in (delta.delta, delta.low, delta.high)
            )
            mark = "*" if delta.significant else ""
            cells.append(f"{METRICS[key]} {point} [{low}, {high}]{mark}")
    return "<br>".join(cells) or "—"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
        for row in [headers, ["---"] * len(headers), *rows]
    )


def _cells(run: RunResult | None, metrics: tuple[str, ...]) -> list[str]:
    return [metric_cell(run, metric) for metric in metrics] if run else ["—"] * len(metrics)


def _run_id(run: RunResult | None) -> str:
    return f"`{run.run_id}`" if run else "pending"


def write_ladder_report(
    ladder: Ladder,
    store: RunStore,
    split: HeldOutSplit,
    output: Path,
    *,
    dev_ids: tuple[str, ...],
    test_ids: tuple[str, ...],
    subset_ids: tuple[str, ...],
    provider: LadderProvider | None = None,
) -> Path:
    from rag_pipeline.ladder_charts import write_charts

    full = _completed(store, ladder.rungs, "test")
    subset = _completed(store, ladder.rungs, "test-400")
    pairs = ladder.large_pairs(provider)
    provider_name = (provider or ladder.large_model_provider).title()
    remote = _completed(store, tuple(pair.remote for pair in pairs), "test-400")
    fixed_subset = set(subset_ids)
    local_runs: dict[str, RunResult | None] = {}
    if split == "test":
        local_runs.update(full)
    else:
        for rung in ladder.rungs:
            name = rung.config.name
            run = subset.get(name) or full.get(name)
            local_runs[name] = (
                _select_rows(run, fixed_subset, keep_judge_metrics=True)
                if run is not None
                else None
            )

    saved = store.runs()
    dev, dev_sources = _references(store, saved, "dev", dev_ids)
    scoring_ids = test_ids if split == "test" else subset_ids
    held_out, held_out_sources = _references(store, saved, "test", scoring_ids)
    if not held_out and split == "test-400":
        held_out, held_out_sources = _references(store, saved, "test-400", scoring_ids)
    baseline = majority_baseline(dev, held_out) if dev and held_out else None

    sections = [
        f"# Ablation Ladder · {split}",
        "Cells show point estimates and 95% bootstrap intervals. A trailing asterisk marks "
        "a paired interval that excludes zero. Step changes use shared queries and compare "
        "adjacent rungs with matching stored configurations. Judge metrics use the fixed "
        "Judge sample when stored for the displayed run.",
    ]
    metric_keys = tuple(METRICS)
    rows: list[list[str]] = []
    previous: RunResult | None = None
    previous_compatible = True
    plotted: list[tuple[str, RunResult | None]] = []
    for rung in ladder.rungs:
        run = local_runs.get(rung.config.name)
        compatible = run is None or _compatible(rung, run)
        delta = (
            _step(previous, run, _STEP_METRICS)
            if compatible and previous_compatible
            else "Config changed"
        )
        rows.append([rung.label, _run_id(run), *_cells(run, metric_keys), delta])
        plotted.append((rung.label, run))
        previous, previous_compatible = run, compatible
    rows.append(
        [
            "Dev majority",
            "calculated" if baseline else "pending",
            *_cells(baseline, metric_keys),
            "—",
        ]
    )
    sections.append(_table(["Rung", "Run", *METRICS.values(), "Step vs previous"], rows))

    kinds = sorted(
        {
            metric
            for _, run in [*plotted, ("Dev majority", baseline)]
            if run is not None
            for metric in run.metrics
            if metric.startswith("accuracy[")
        }
    )
    sections.extend(
        [
            "## Accuracy by query type",
            _table(
                ["Rung", "Run", *(kind[9:-1] for kind in kinds)],
                [
                    [
                        label,
                        "calculated" if run is baseline and baseline else _run_id(run),
                        *_cells(run, tuple(kinds)),
                    ]
                    for label, run in [*plotted, ("Dev majority", baseline)]
                ],
            ),
            "## Large-model comparison · test-400",
            f"Local estimates use the same saved queries as {provider_name}. Completed full Test "
            "Split runs supply the matching local answers. The paired step subtracts the local "
            "estimate from the remote estimate.",
        ]
    )
    large_rows: list[list[str]] = []
    comparison_sources: list[tuple[LadderRung, RunResult | None]] = []
    for pair in pairs:
        local = subset.get(pair.local.config.name) or full.get(pair.local.config.name)
        remote_run = remote.get(pair.remote.config.name)
        if local is not None:
            comparison_sources.append((pair.local, local))
        comparison_sources.append((pair.remote, remote_run))
        shared: set[str] = (
            fixed_subset
            & {row.query_id for row in local.records}
            & {row.query_id for row in remote_run.records}
            if local is not None and remote_run is not None
            else set()
        )
        left = _select_rows(local, shared) if local else None
        right = _select_rows(remote_run, shared) if remote_run else None
        compatible = (local is None or _compatible(pair.local, local)) and (
            remote_run is None or _compatible(pair.remote, remote_run)
        )
        large_rows.append(
            [
                pair.local.label,
                _run_id(local),
                _run_id(remote_run),
                str(len(shared)) if shared else "—",
                *_cells(left, _LARGE_METRICS),
                *_cells(right, _LARGE_METRICS),
                _step(left, right, _LARGE_METRICS) if compatible else "Config changed",
            ]
        )
    sections.append(
        _table(
            [
                "Rung",
                "Local run",
                f"{provider_name} run",
                "Shared n",
                *(f"Local {METRICS[key]}" for key in _LARGE_METRICS),
                *(f"{provider_name} {METRICS[key]}" for key in _LARGE_METRICS),
                "Paired step",
            ],
            large_rows,
        )
    )

    provenance: list[list[str]] = []
    sources: list[tuple[LadderRung, RunResult | None]] = [
        (rung, local_runs.get(rung.config.name)) for rung in ladder.rungs
    ]
    sources += comparison_sources
    recorded: set[str] = set()
    for rung, run in sources:
        if run is None:
            provenance.append([rung.config.name, "pending", *["—"] * 6])
            continue
        if run.run_id in recorded:
            continue
        recorded.add(run.run_id)
        provenance.append(
            [
                rung.config.name,
                _run_id(run),
                *(
                    str(run.meta.get(key) or "—")
                    for key in (
                        "commit",
                        "index_version",
                        "generator_version",
                        "started_at",
                        "finished_at",
                    )
                ),
                "matches" if _compatible(rung, run) else "changed",
            ]
        )
    sections.extend(
        [
            "## Provenance",
            _table(
                [
                    "Pipeline Variant",
                    "Run",
                    "Commit",
                    "Index version",
                    "Generator version",
                    "Started (UTC)",
                    "Finished (UTC)",
                    "Config",
                ],
                provenance,
            ),
            "The majority rule learns reference answers from the full Dev Split and predicts "
            "Abstention for Null Queries. Ties use lexical order.",
            f"Runs used to learn the majority rule · {', '.join(dev_sources) or 'pending'}",
            f"Runs supplying held-out references · {', '.join(held_out_sources) or 'pending'}",
            "## Charts",
            "![Accuracy by rung](accuracy.svg)",
            "![Accuracy and latency](quality-latency.svg)",
            "![Accuracy by query type](accuracy-by-type.svg)",
        ]
    )
    path = output / "report.md"
    output.mkdir(parents=True, exist_ok=True)
    write_charts(output, plotted, baseline, kinds)
    write_text(path, "\n\n".join(sections) + "\n")
    return path
