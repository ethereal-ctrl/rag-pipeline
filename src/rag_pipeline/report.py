"""Markdown results tables for the report, built from stored runs."""

from __future__ import annotations

from rag_pipeline.evaluation import RunResult


def metric_label(metric: str) -> str:
    return "cost_per_question (notional USD)" if metric == "cost_per_question" else metric


def metric_number(metric: str, value: float, *, signed: bool = False) -> str:
    precision = ".6g" if metric == "cost_per_question" else ".3f"
    return format(value, ("+" if signed else "") + precision)


def _cell(run: RunResult, metric: str) -> str:
    if metric not in run.metrics:
        return "n/a"
    low, high = run.intervals.get(metric, (float("nan"), float("nan")))
    estimate, lower, upper = (
        metric_number(metric, value) for value in (run.metrics[metric], low, high)
    )
    return f"{estimate} [{lower}, {upper}]"


def results_table(runs: list[RunResult], metrics: list[str]) -> str:
    """One row per run: variant, split, query count, then each metric with its 95% interval."""
    lines = [
        "| Variant | Split | n | " + " | ".join(map(metric_label, metrics)) + " |",
        "|---|---|---:|" + "---:|" * len(metrics),
    ]
    for run in runs:
        cells = [_cell(run, m) for m in metrics]
        variant = run.meta["variant"]["name"]
        lines.append(
            f"| {variant} | {run.meta['split']} | {int(run.metrics['queries'])} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)
