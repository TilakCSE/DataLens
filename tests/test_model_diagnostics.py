"""Class diagnostics must describe the exact held-out prediction counts."""

import json

import numpy as np

from datalens.model_diagnostics import build_class_diagnostics


def test_multiclass_matrix_axes_support_and_zero_recall():
    train = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])
    actual = np.array(["A", "B", "C", "C"])
    predicted = np.array(["A", "A", "B", "B"])

    result = build_class_diagnostics(train, actual, predicted, np.array(["A", "B", "C"]))

    assert result["labels"] == ["A", "B", "C"]
    assert result["train_support"] == [3, 3, 3]
    assert result["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 2, 0]]
    assert [item["support"] for item in result["per_class"]] == [1, 1, 2]
    assert [item["recall"] for item in result["per_class"]] == [1.0, 0.0, 0.0]
    assert result["matrix_rows"] == "actual"
    assert result["matrix_columns"] == "predicted"
    json.dumps(result, allow_nan=False)


def test_numpy_numeric_labels_become_json_scalars():
    train = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
    actual = np.array([0, 1], dtype=np.int64)
    predicted = np.array([0, 0], dtype=np.int64)

    result = build_class_diagnostics(train, actual, predicted, np.array([0, 1]))

    assert result["labels"] == [0, 1]
    assert [item["label"] for item in result["per_class"]] == [0, 1]
    json.dumps(result, allow_nan=False)
