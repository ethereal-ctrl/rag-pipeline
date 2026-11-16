"""Headless SVG charts for the stored Ablation Ladder report."""

# Matplotlib's plotting stubs leave forwarded artist keyword arguments untyped.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from rag_pipeline.evaluation import RunResult

# Okabe-Ito colours.
_PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
type Series = list[tuple[str, RunResult | None]]


def _figure() -> tuple[Figure, Axes]:
    figure = Figure(figsize=(12, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    return figure, axes


def _rung_ticks(axes: Axes, entries: Series) -> None:
    axes.set_xticks(range(len(entries)), [label for label, _ in entries], rotation=22, ha="right")
    axes.set_xlabel("Ablation Ladder rung")


def _save(figure: Figure, path: Path) -> None:
    figure.savefig(path, format="svg", metadata={"Date": None})


def _accuracy(output: Path, entries: Series, baseline: RunResult | None) -> None:
    figure, axes = _figure()
    for i, (_, run) in enumerate(entries):
        if run is None or "accuracy" not in run.metrics:
            continue
        accuracy = run.metrics["accuracy"]
        low, high = run.intervals["accuracy"]
        axes.errorbar(
            i,
            100 * accuracy,
            yerr=[[100 * max(0, accuracy - low)], [100 * max(0, high - accuracy)]],
            fmt="o",
            color=_PALETTE[0],
            capsize=4,
        )
    if baseline is not None:
        axes.axhline(
            100 * baseline.metrics["accuracy"],
            color=_PALETTE[1],
            linestyle="--",
            label="Dev majority",
        )
        axes.legend()
    _rung_ticks(axes, entries)
    axes.set_title("Answer accuracy by rung")
    axes.set_ylabel("Accuracy (%)")
    axes.set_ylim(0, 100)
    axes.grid(axis="y", alpha=0.25)
    _save(figure, output / "accuracy.svg")


def _quality_latency(output: Path, entries: Series) -> None:
    figure, axes = _figure()
    points = [
        (run.metrics["latency_p50"], 100 * run.metrics["accuracy"], label)
        for label, run in entries
        if run is not None and {"latency_p50", "accuracy"} <= run.metrics.keys()
    ]
    frontier: list[tuple[float, float]] = []
    for latency, accuracy, _ in sorted(points, key=lambda point: (point[0], -point[1])):
        if not frontier or accuracy > frontier[-1][1]:
            frontier.append((latency, accuracy))
    if len(frontier) > 1:
        axes.plot(
            [latency for latency, _ in frontier],
            [accuracy for _, accuracy in frontier],
            color=_PALETTE[2],
            linestyle="--",
            label="Pareto frontier",
        )
        axes.legend(loc="lower right")
    if points:
        axes.scatter(
            [latency for latency, _, _ in points],
            [accuracy for _, accuracy, _ in points],
            color=_PALETTE[0],
        )
        for i, (latency, accuracy, label) in enumerate(points):
            axes.annotate(
                label,
                (latency, accuracy),
                xytext=(5, -12 if i % 2 else 6),
                textcoords="offset points",
                fontsize=8,
            )
    axes.set_title("Answer quality and latency")
    axes.set_xlabel("p50 latency (s)")
    axes.set_ylabel("Accuracy (%)")
    axes.set_ylim(0, 105)
    axes.grid(alpha=0.25)
    _save(figure, output / "quality-latency.svg")


def _accuracy_by_type(output: Path, entries: Series, kinds: list[str]) -> None:
    figure, axes = _figure()
    width = 0.8 / max(len(kinds), 1)
    for n, kind in enumerate(kinds):
        offset = width * (n - (len(kinds) - 1) / 2)
        bars = [
            (i + offset, 100 * run.metrics[kind])
            for i, (_, run) in enumerate(entries)
            if run is not None and kind in run.metrics
        ]
        if bars:
            axes.bar(
                [x for x, _ in bars],
                [y for _, y in bars],
                width=width,
                color=_PALETTE[n % len(_PALETTE)],
                label=kind[9:-1].removesuffix("_query"),
            )
    if kinds:
        axes.legend()
    _rung_ticks(axes, entries)
    axes.set_title("Answer accuracy by query type")
    axes.set_ylabel("Accuracy (%)")
    axes.set_ylim(0, 100)
    axes.grid(axis="y", alpha=0.25)
    _save(figure, output / "accuracy-by-type.svg")


def write_charts(
    output: Path, entries: Series, baseline: RunResult | None, kinds: list[str]
) -> None:
    matplotlib.use("Agg")
    with matplotlib.rc_context({"svg.hashsalt": "rag-ladder", "font.family": "DejaVu Sans"}):
        _accuracy(output, entries, baseline)
        _quality_latency(output, entries)
        _accuracy_by_type(output, entries, kinds)
