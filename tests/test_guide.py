"""Checks for the opt-in guided experience and its applied choices."""

import json

import pandas as pd
from sklearn.preprocessing import RobustScaler

import datalens as dl


def _frame():
    frame = pd.DataFrame({
        "PassengerId": range(1, 41),
        "age": [float(20 + i) for i in range(40)],
        "group": ["A" if i % 2 else "B" for i in range(40)],
        "label": [i % 2 for i in range(40)],
    })
    frame.loc[[3, 7], "age"] = None
    frame.loc[[5, 11], "group"] = None
    return frame


def test_analyse_keeps_chart_ideas_and_adds_traceable_cleaning_suggestions():
    report = dl.analyse(_frame())

    assert report.recommendations
    actions = {item.column: item.recommended_action for item in report.cleaning_suggestions}
    assert actions["age"] == "median"
    assert actions["group"] == "constant"
    assert actions["PassengerId"] == "exclude_column"
    assert "2 missing" in next(
        item.reason for item in report.cleaning_suggestions if item.column == "age"
    )
    json.dumps(report.to_dict(), allow_nan=False)


def test_guide_applies_imputer_scaler_model_and_save_choices(tmp_path):
    frame = _frame()
    original = frame.copy(deep=True)
    answers = iter(["s", "y", "y", "2", "2", "3", "2", "y", "y"])
    output = []

    result = dl.guide(
        frame, target="label", output=tmp_path / "guided.joblib",
        input_fn=lambda _: next(answers), output_fn=output.append,
    )

    assert result is not None
    pd.testing.assert_frame_equal(frame, original)
    details = result.to_dict()
    assert details["model"] == "logistic_regression"
    assert details["selection_method"] == "user_override"
    assert details["preprocessing"]["numeric_imputers"]["age"] == "mean"
    assert details["preprocessing"]["categorical_imputers"]["group"] == "most_frequent"
    assert details["preprocessing"]["scaler_for_logistic_regression"] == "robust"
    numeric = result.pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    assert numeric.named_steps["impute"].strategy == "mean"
    assert isinstance(numeric.named_steps["scale"], RobustScaler)
    assert (tmp_path / "guided.joblib").exists()
    assert (tmp_path / "guided.json").exists()
    assert any("Chart ideas" in line for line in output)


def test_guide_can_stop_after_analysis_without_training():
    frame = _frame()
    original = frame.copy(deep=True)
    answers = iter(["s", "s"])
    output = []

    result = dl.guide(
        frame, input_fn=lambda _: next(answers), output_fn=output.append
    )

    assert result is None
    pd.testing.assert_frame_equal(frame, original)
    assert any("Training skipped" in line for line in output)


def test_guide_skips_a_column_and_removes_duplicate_rows():
    frame = pd.concat([_frame(), _frame().iloc[[0, 1]]], ignore_index=True)
    original = frame.copy(deep=True)
    answers = iter(["s", "y", "y", "y", "s", "1", "3", "y", "s"])

    result = dl.guide(
        frame, target="label", input_fn=lambda _: next(answers),
        output_fn=lambda _: None,
    )

    assert result is not None
    pd.testing.assert_frame_equal(frame, original)
    assert result.dropped_duplicate_rows == 2
    assert "age" in result.excluded_columns
    assert "age" not in result.feature_columns
    assert result.model == "random_forest"
