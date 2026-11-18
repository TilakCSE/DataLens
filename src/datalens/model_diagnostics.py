"""Measured class diagnostics from one untouched holdout prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def _json_label(value: Any) -> str | int | float | bool | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_class_diagnostics(
    y_train: Any, y_test: Any, predictions: Any, labels: Any,
) -> dict[str, Any]:
    """Calculate class support and errors with one fixed label order."""
    ordered = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=ordered, zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=ordered)
    serializable_labels = [_json_label(label) for label in ordered]
    return {
        "labels": serializable_labels,
        "train_support": [int((y_train == label).sum()) for label in ordered],
        "per_class": [
            {
                "label": label,
                "precision": float(class_precision),
                "recall": float(class_recall),
                "f1": float(class_f1),
                "support": int(class_support),
            }
            for label, class_precision, class_recall, class_f1, class_support in zip(
                serializable_labels, precision, recall, f1, support, strict=True,
            )
        ],
        "confusion_matrix": matrix.astype(int).tolist(),
        "matrix_rows": "actual",
        "matrix_columns": "predicted",
        "zero_division": 0,
    }


def save_confusion_html(
    diagnostics: dict[str, Any], path: str | Path, *, title: str,
) -> Path:
    """Save an interactive confusion matrix without replacing an existing file."""
    import plotly.graph_objects as go

    destination = Path(path)
    if destination.suffix.lower() != ".html":
        raise ValueError("Confusion matrix output must end in .html")
    if destination.exists():
        raise FileExistsError(f"Confusion matrix output already exists: {destination}")
    if not diagnostics:
        raise ValueError("No class diagnostics are available to plot")
    labels = [str(label) for label in diagnostics["labels"]]
    matrix = diagnostics["confusion_matrix"]
    figure = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels, text=matrix,
        texttemplate="%{text}",
        colorscale=[[0, "#fff5f5"], [1, "#b91c1c"]],
        colorbar={"title": "Rows"},
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Rows: %{z}<extra></extra>",
    ))
    figure.update_layout(
        title=title,
        xaxis_title="Predicted class",
        yaxis_title="Actual class",
        yaxis_autorange="reversed",
        template="plotly_white",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(destination), include_plotlyjs=True, full_html=True, auto_open=False)
    return destination
