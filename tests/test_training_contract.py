"""Behavioral checks for the first supervised DataLens training workflow.

These use small, deliberately messy tables rather than an external dataset so
that preprocessing, evaluation, and persistence can be checked repeatably.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from datalens.training import train_classifier


def _passenger_data(size=80):
    """Include missing values and both numeric and categorical predictors."""
    frame = pd.DataFrame(
        {
            "age": [20 + index % 43 for index in range(size)],
            "fare": [10.0 + (index * 7) % 95 for index in range(size)],
            "port": [["C", "Q", "S"][index % 3] for index in range(size)],
            "survived": [index % 2 for index in range(size)],
        }
    )
    frame.loc[3, "age"] = np.nan
    frame.loc[7, "fare"] = np.nan
    frame.loc[11, "port"] = None
    return frame


def test_training_handles_mixed_missing_data_without_mutating_input():
    frame = _passenger_data()
    original = frame.copy(deep=True)

    result = train_classifier(frame, "survived")

    pd.testing.assert_frame_equal(frame, original)
    report = result.to_dict()
    json.dumps(report, allow_nan=False)
    assert report["task"] == "classification"
    assert report["target"] == "survived"
    assert report["train_rows"] + report["test_rows"] == len(frame)
    assert {"accuracy", "balanced_accuracy", "f1_macro"} <= report[
        "test_metrics"
    ].keys()
    assert len(report["candidates"]) >= 3  # including a dummy baseline
    predictions = result.pipeline.predict(frame.drop(columns="survived"))
    assert len(predictions) == len(frame)
    assert set(predictions).issubset({0, 1})


@pytest.mark.parametrize(
    "frame,target",
    [
        (_passenger_data(), "absent"),
        (pd.DataFrame(), "survived"),
        (_passenger_data().assign(survived=None), "survived"),
        (_passenger_data().assign(survived=1), "survived"),
    ],
    ids=["unknown-target", "empty-input", "null-target", "one-class"],
)
def test_training_rejects_invalid_supervised_tasks(frame, target):
    with pytest.raises(ValueError):
        train_classifier(frame, target)


def test_training_has_reproducible_predictions_for_fixed_seed():
    frame = _passenger_data()
    features = frame.drop(columns="survived")

    first = train_classifier(frame, "survived", random_state=27)
    second = train_classifier(frame, "survived", random_state=27)

    np.testing.assert_array_equal(
        first.pipeline.predict(features), second.pipeline.predict(features)
    )


def test_preprocessing_is_not_fitted_on_holdout_only_category():
    frame = _passenger_data()
    _, test_indices = train_test_split(
        frame.index,
        test_size=0.2,
        stratify=frame["survived"],
        random_state=42,
    )
    frame.loc[test_indices[0], "port"] = "HOLDOUT_ONLY_TOKEN"

    result = train_classifier(frame, "survived", test_size=0.2, random_state=42)

    fitted_encoders = list(_encoders_in_fitted_pipeline(result.pipeline))
    assert fitted_encoders, "categorical predictors should be encoded in the fitted pipeline"
    assert all(
        "HOLDOUT_ONLY_TOKEN" not in category_values
        for encoder in fitted_encoders
        for category_values in encoder.categories_
    ), "holdout-only values must not influence the fitted encoder"


def test_saved_full_pipeline_reloads_and_predicts_identically(tmp_path):
    frame = _passenger_data()
    features = frame.drop(columns="survived")
    result = train_classifier(frame, "survived")
    destination = tmp_path / "passenger-model.joblib"

    result.save(destination)

    restored = joblib.load(destination)
    np.testing.assert_array_equal(
        restored.predict(features), result.pipeline.predict(features)
    )
    assert (tmp_path / "passenger-model.json").exists(), (
        "a manifest should accompany the saved model for reproducibility"
    )


def _encoders_in_fitted_pipeline(estimator):
    """Walk fitted objects; `get_params` may expose unfitted constructor clones."""
    if isinstance(estimator, OneHotEncoder):
        if hasattr(estimator, "categories_"):
            yield estimator
    elif isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            yield from _encoders_in_fitted_pipeline(step)
    elif isinstance(estimator, ColumnTransformer):
        for _, transformer, _ in estimator.transformers_:
            if transformer not in ("drop", "passthrough"):
                yield from _encoders_in_fitted_pipeline(transformer)
