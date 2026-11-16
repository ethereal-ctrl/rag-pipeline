"""Markdown results tables for the report, built from stored runs."""

from __future__ import annotations

from rag_pipeline.evaluation import RunResult


def _cell(run: RunResult, metric: str) -> str:
    if metric not in run.metrics:
        return "n/a"
    low, high = run.intervals.get(metric, (float("nan"), float("nan")))
    return f"{run.metrics[metric]:.3f} [{low:.3f}, {high:.3f}]"


def results_table(runs: list[RunResult], metrics: list[str]) -> str:
    """One row per run: variant, split, query count, then each metric with its 95% interval."""
    lines = [
        "| Variant | Split | n | " + " | ".join(metrics) + " |",
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
