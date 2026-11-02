"""Render calculated DataLens chart recommendations as local Plotly figures."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from datalens.models import ChartRecommendation


@dataclass
class RenderedChart:
    """An interactive figure and the exact data used to draw it."""

    recommendation: ChartRecommendation
    figure: go.Figure
    plot_data: pd.DataFrame
    source_rows: int
    plotted_rows: int
    transformation: str

    @property
    def excluded_rows(self) -> int:
        return self.source_rows - self.plotted_rows

    def to_dict(self) -> dict[str, Any]:
        """Return small, JSON-compatible facts about the rendered chart."""
        return {
            "chart_type": self.recommendation.chart_type,
            "title": self.figure.layout.title.text,
            "columns": list(self.recommendation.columns),
            "reason": self.recommendation.reason,
            "source_rows": self.source_rows,
            "plotted_rows": self.plotted_rows,
            "excluded_rows": self.excluded_rows,
            "plotted_points": len(self.plot_data),
            "transformation": self.transformation,
        }

    def save(self, path: str | Path) -> Path:
        """Save a self-contained interactive HTML file without overwriting one."""
        destination = Path(path)
        if destination.suffix.lower() != ".html":
            raise ValueError("Chart output must end in .html")
        if destination.exists():
            raise FileExistsError(f"Chart output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.figure.write_html(
            str(destination), include_plotlyjs=True, full_html=True, auto_open=False
        )
        return destination


def _finite_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return values


def _histogram(frame: pd.DataFrame, recommendation: ChartRecommendation) -> RenderedChart:
    name = recommendation.columns[0]
    numbers = _finite_numeric(frame[name])
    values = numbers[np.isfinite(numbers)]
    if not len(values):
        raise ValueError(f"No finite numeric values remain in {name!r} to plot")
    edges = np.histogram_bin_edges(values, bins="auto")
    if len(edges) - 1 > 40:
        edges = np.linspace(values.min(), values.max(), 41)
    counts, edges = np.histogram(values, bins=edges)
    data = pd.DataFrame({
        "bin_start": edges[:-1], "bin_end": edges[1:], "count": counts,
    })
    figure = go.Figure(go.Bar(
        x=(edges[:-1] + edges[1:]) / 2,
        y=counts,
        width=np.diff(edges),
        marker_color="#b91c1c",
        customdata=np.column_stack((edges[:-1], edges[1:])),
        hovertemplate="From %{customdata[0]:.3g} to %{customdata[1]:.3g}<br>Count: %{y}<extra></extra>",
    ))
    figure.update_layout(
        title=recommendation.title, xaxis_title=name, yaxis_title="Count", bargap=0.03,
        template="plotly_white",
    )
    return RenderedChart(
        recommendation, figure, data, len(frame), len(values),
        "Finite numeric values; bins computed with NumPy (at most 40 bins).",
    )


def _category_key(value: Any) -> tuple[str, str]:
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return ("missing", "")
    except (TypeError, ValueError):
        pass
    return (type(value).__name__, repr(value))


def _bar(
    frame: pd.DataFrame, recommendation: ChartRecommendation, *, max_categories: int
) -> RenderedChart:
    name = recommendation.columns[0]
    counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], Any] = {}
    for value in frame[name].tolist():
        key = _category_key(value)
        counts[key] += 1
        examples.setdefault(key, value)
    ordered = sorted(counts, key=lambda key: (-counts[key], key))
    shown = ordered[: max_categories - 1] if len(ordered) > max_categories else ordered
    labels = ["(Missing)" if key[0] == "missing" else str(examples[key]) for key in shown]
    label_counts = Counter(labels)
    labels = [
        f"{label} [{key[0]}]" if label_counts[label] > 1 else label
        for label, key in zip(labels, shown)
    ]
    values = [counts[key] for key in shown]
    grouped = len(ordered) - len(shown)
    if grouped:
        labels.append(f"Other ({grouped} categories)")
        values.append(sum(counts[key] for key in ordered[len(shown):]))
    data = pd.DataFrame({"category": labels, "count": values})
    figure = go.Figure(go.Bar(
        x=data["category"], y=data["count"], marker_color="#b91c1c",
        hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
    ))
    figure.update_layout(
        title=recommendation.title, xaxis_title=name, yaxis_title="Rows",
        template="plotly_white",
    )
    return RenderedChart(
        recommendation, figure, data, len(frame), len(frame),
        f"Exact row counts, including missing values; {grouped} categories grouped into Other.",
    )


def _line(
    frame: pd.DataFrame, recommendation: ChartRecommendation, *, aggregation: str
) -> RenderedChart:
    date_name, measure_name = recommendation.columns
    dates = pd.to_datetime(frame[date_name], errors="coerce", format="mixed", utc=True)
    numbers = _finite_numeric(frame[measure_name])
    valid = dates.notna().to_numpy() & np.isfinite(numbers)
    if not valid.any():
        raise ValueError("No rows have both a valid date and a finite numeric value")
    data = pd.DataFrame({
        "date": dates[valid].to_numpy(), "value": numbers[valid],
    })
    grouped = data.groupby("date", as_index=False, sort=True)["value"].agg(aggregation)
    grouped = grouped.rename(columns={"value": measure_name})
    figure = go.Figure(go.Scatter(
        x=grouped["date"], y=grouped[measure_name], mode="lines+markers",
        line_color="#b91c1c",
        hovertemplate="%{x}<br>%{y:.3g}<extra></extra>",
    ))
    figure.update_layout(
        title=f"{aggregation.title()} {measure_name} by {date_name}",
        xaxis_title=date_name,
        yaxis_title=f"{aggregation.title()} {measure_name}", template="plotly_white",
    )
    return RenderedChart(
        recommendation, figure, grouped, len(frame), int(valid.sum()),
        f"Rows with valid dates and finite values; {aggregation} by exact date/time (UTC).",
    )


def _scatter(frame: pd.DataFrame, recommendation: ChartRecommendation) -> RenderedChart:
    x_name, y_name = recommendation.columns
    x_values = _finite_numeric(frame[x_name])
    y_values = _finite_numeric(frame[y_name])
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if not valid.any():
        raise ValueError("No rows have finite values for both scatter axes")
    data = pd.DataFrame({x_name: x_values[valid], y_name: y_values[valid]})
    figure = go.Figure(go.Scattergl(
        x=data[x_name], y=data[y_name], mode="markers",
        marker={"color": "#b91c1c", "opacity": 0.6, "size": 7},
        hovertemplate=f"{x_name}: %{{x:.3g}}<br>{y_name}: %{{y:.3g}}<extra></extra>",
    ))
    figure.update_layout(
        title=recommendation.title, xaxis_title=x_name, yaxis_title=y_name,
        template="plotly_white",
    )
    return RenderedChart(
        recommendation, figure, data, len(frame), int(valid.sum()),
        "Rows with finite values on both axes; no aggregation.",
    )


def render_recommendation(
    frame: pd.DataFrame, recommendation: ChartRecommendation, *,
    aggregation: str = "mean", max_categories: int = 20,
) -> RenderedChart:
    """Render one analysis recommendation without changing the source data."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("visualise expects a pandas DataFrame")
    if aggregation not in {"mean", "sum"}:
        raise ValueError("aggregation must be mean or sum")
    if not isinstance(max_categories, int) or max_categories < 2:
        raise ValueError("max_categories must be an integer of at least 2")
    if frame.columns.has_duplicates:
        raise ValueError("Column names must be unique for visualization")
    if any(name not in frame.columns for name in recommendation.columns):
        raise ValueError("A recommended chart column is missing from the DataFrame")
    if recommendation.chart_type == "histogram":
        return _histogram(frame, recommendation)
    if recommendation.chart_type == "bar":
        return _bar(frame, recommendation, max_categories=max_categories)
    if recommendation.chart_type == "line":
        return _line(frame, recommendation, aggregation=aggregation)
    if recommendation.chart_type == "scatter":
        return _scatter(frame, recommendation)
    raise ValueError(f"Unsupported chart type: {recommendation.chart_type}")
