"""Focused behaviour checks for the first supervised training workflow."""

import json

import joblib
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import datalens as dl
from datalens.training import train_classifier


def _mixed_frame() -> pd.DataFrame:
    rows = 40
    return pd.DataFrame(
        {
            "record_id": range(1, rows + 1),
            "age": [None if i % 9 == 0 else float(i + 20) for i in range(rows)],
            "group": [None if i % 11 == 0 else ("A" if i % 2 else 2) for i in range(rows)],
            "unique_note": [f"person-{i}" for i in range(rows)],
            "outcome": [i % 2 for i in range(rows)],
        }
    )


def test_mixed_values_and_high_cardinality_are_handled_without_mutating_input():
    frame = _mixed_frame()
    original = frame.copy(deep=True)

    result = train_classifier(frame, "outcome")

    pd.testing.assert_frame_equal(frame, original)
    assert set(result.feature_columns) == {"age", "group"}
    assert set(result.excluded_columns) == {"record_id", "unique_note"}
    assert {item["model"] for item in result.candidates} == {
        "dummy", "logistic_regression", "random_forest"
    }
    assert len(result.pipeline.predict(frame[result.feature_columns])) == len(frame)
    json.dumps(result.to_dict())


def test_public_train_entry_point_and_camel_case_identifier():
    frame = _mixed_frame().rename(columns={"record_id": "PassengerId"})

    result = dl.train(frame, target="outcome")

    assert "PassengerId" in result.excluded_columns
    assert "PassengerId" not in result.feature_columns
    assert result.to_dict()["target"] == "outcome"


def test_train_only_imputation_statistics_and_saved_pipeline(tmp_path):
    frame = _mixed_frame()
    result = train_classifier(frame, "outcome", random_state=7)
    X = frame[result.feature_columns]
    y = frame["outcome"]
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )
    numeric = result.pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    assert numeric.named_steps["impute"].statistics_[0] == X_train["age"].median()

    artifact = result.save(tmp_path / "model.joblib")
    restored = joblib.load(artifact)
    assert restored.predict(X.head(5)).tolist() == result.pipeline.predict(X.head(5)).tolist()
    manifest = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert manifest["model"] == result.model
    assert manifest["versions"]["scikit_learn"]


@pytest.mark.parametrize(
    "frame,target,part",
    [
        (pd.DataFrame({"x": [1, 2]}), "missing", "not found"),
        (pd.DataFrame({"x": range(8), "y": [1] * 8}), "y", "at least two classes"),
        (pd.DataFrame({"x": range(8), "y": [0] * 7 + [1]}), "y", "at least 3 rows"),
        (pd.DataFrame({"y": [0, 1] * 5}), "y", "No supported feature"),
    ],
)
def test_invalid_training_inputs_are_explained(frame, target, part):
    with pytest.raises(ValueError, match=part):
        train_classifier(frame, target)
