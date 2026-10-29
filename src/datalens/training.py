"""Local, reproducible supervised training for labelled tabular data.

The first training workflow is deliberately bounded to classification.  Every
learned preprocessing step lives inside the fitted scikit-learn Pipeline, so
cross-validation and the holdout test set cannot share fitted statistics.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from datalens.cleaning import looks_like_identifier



def _as_text_with_missing(values: Any) -> Any:
    """Normalize mixed categorical values before sklearn imputes and encodes."""
    frame = pd.DataFrame(values).astype("object")
    return frame.map(
        lambda value: np.nan if pd.isna(value) else str(value)
    ).to_numpy(dtype=object)


@dataclass
class TrainingResult:
    """The evaluated result and its complete, fitted prediction pipeline."""

    pipeline: Pipeline
    target: str
    train_rows: int
    test_rows: int
    model: str
    cv_score: float
    test_metrics: dict[str, float]
    candidates: list[dict[str, Any]]
    feature_columns: list[str]
    excluded_columns: list[str]
    warnings: list[str]
    random_state: int
    test_size: float
    dropped_target_rows: int
    dataset_fingerprint: str
    preprocessing: dict[str, Any] = field(default_factory=dict)
    selection_method: str = "cross_validation"
    dropped_duplicate_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible facts about this run, excluding model bytes."""
        return {
            "task": "classification",
            "target": self.target,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "model": self.model,
            "selection_metric": "f1_macro",
            "cv_score": self.cv_score,
            "test_metrics": dict(self.test_metrics),
            "candidates": [dict(item) for item in self.candidates],
            "feature_columns": list(self.feature_columns),
            "excluded_columns": list(self.excluded_columns),
            "warnings": list(self.warnings),
            "random_state": self.random_state,
            "test_size": self.test_size,
            "dropped_target_rows": self.dropped_target_rows,
            "dataset_fingerprint": self.dataset_fingerprint,
            "preprocessing": dict(self.preprocessing),
            "selection_method": self.selection_method,
            "dropped_duplicate_rows": self.dropped_duplicate_rows,
        }

    def save(self, path: str | Path) -> Path:
        """Save a trusted local joblib pipeline and neighbouring JSON manifest.

        Loading pickle/joblib from an untrusted source is unsafe.  Users should
        only load model artifacts produced by a trusted local training run.
        """
        output = Path(path)
        if output.suffix != ".joblib":
            output = output.with_suffix(".joblib")
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, output)
        manifest = self.to_dict()
        manifest["artifact"] = output.name
        manifest["versions"] = {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        }
        output.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return output


def _feature_groups(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    excluded: list[str] = []
    warnings: list[str] = []
    for name in frame.columns:
        series = frame[name]
        non_null = series.dropna()
        if non_null.empty:
            excluded.append(name)
            warnings.append(f"Excluded {name}: every value is missing.")
        elif looks_like_identifier(name):
            excluded.append(name)
            warnings.append(f"Excluded {name}: its name suggests an identifier.")
        elif pd.api.types.is_datetime64_any_dtype(series):
            excluded.append(name)
            warnings.append(f"Excluded {name}: datetime feature engineering is not in this version.")
        elif pd.api.types.is_bool_dtype(series):
            categorical.append(name)
        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(name)
        elif non_null.nunique(dropna=True) > 50 or non_null.nunique(dropna=True) / len(non_null) > 0.8:
            excluded.append(name)
            warnings.append(f"Excluded {name}: high-cardinality text needs explicit feature engineering.")
        else:
            categorical.append(name)
    return numeric, categorical, excluded, warnings


def _preprocessor(
    numeric: list[str], categorical: list[str], *, scale: bool,
    numeric_imputers: dict[str, str], categorical_imputers: dict[str, str],
    scaler: str,
) -> ColumnTransformer:
    transforms: list[tuple[str, Any, list[str]]] = []
    numeric_groups: dict[str, list[str]] = {}
    for name in numeric:
        numeric_groups.setdefault(numeric_imputers.get(name, "median"), []).append(name)
    for strategy, names in numeric_groups.items():
        numeric_steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy=strategy, keep_empty_features=True))
        ]
        if scale and scaler != "none":
            scaler_type = {
                "standard": StandardScaler,
                "minmax": MinMaxScaler,
                "robust": RobustScaler,
            }[scaler]
            numeric_steps.append(("scale", scaler_type()))
        group_name = "numeric" if len(numeric_groups) == 1 else f"numeric_{strategy}"
        transforms.append((group_name, Pipeline(numeric_steps), names))
    categorical_groups: dict[str, list[str]] = {}
    for name in categorical:
        categorical_groups.setdefault(categorical_imputers.get(name, "constant"), []).append(name)
    for strategy, names in categorical_groups.items():
        transforms.append(
            (
                "categorical" if len(categorical_groups) == 1 else f"categorical_{strategy}",
                Pipeline(
                    [
                        ("text", FunctionTransformer(_as_text_with_missing, validate=False)),
                        (
                            "impute",
                            SimpleImputer(
                                strategy=strategy,
                                fill_value="__missing__" if strategy == "constant" else None,
                                keep_empty_features=True,
                            ),
                        ),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                names,
            )
        )
    return ColumnTransformer(transforms, remainder="drop")


def _pipeline(
    estimator: Any, numeric: list[str], categorical: list[str], *, scale: bool,
    numeric_imputers: dict[str, str], categorical_imputers: dict[str, str], scaler: str,
) -> Pipeline:
    return Pipeline([
        ("preprocess", _preprocessor(
            numeric, categorical, scale=scale, numeric_imputers=numeric_imputers,
            categorical_imputers=categorical_imputers, scaler=scaler,
        )),
        ("model", estimator),
    ])


def _fingerprint(frame: pd.DataFrame) -> str:
    try:
        hashes = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    except TypeError:
        hashes = pd.util.hash_pandas_object(frame.astype(str), index=True).values.tobytes()
    return hashlib.sha256(hashes).hexdigest()


def train_classifier(
    df: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    numeric_imputers: dict[str, str] | None = None,
    categorical_imputers: dict[str, str] | None = None,
    exclude_columns: list[str] | None = None,
    scaler: str = "standard",
    model: str = "auto",
    drop_duplicates: bool = False,
) -> TrainingResult:
    """Compare bounded local classifiers and evaluate once on a holdout set.

    The test set is never used for candidate selection or preprocessing fit.
    This version supports independent, labelled tabular rows with a random
    stratified split. Time-ordered or grouped observations need a future split
    strategy and should be reviewed by the user before using this workflow.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("train_classifier expects a pandas DataFrame")
    if df.columns.has_duplicates:
        raise ValueError("Column names must be unique for training")
    if not isinstance(target, str) or target not in df.columns:
        raise ValueError(f"Target column {target!r} was not found")
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be greater than 0 and less than 0.5")
    if scaler not in {"standard", "minmax", "robust", "none"}:
        raise ValueError("scaler must be standard, minmax, robust, or none")
    if model not in {"auto", "logistic_regression", "random_forest"}:
        raise ValueError("model must be auto, logistic_regression, or random_forest")
    numeric_imputers = dict(numeric_imputers or {})
    categorical_imputers = dict(categorical_imputers or {})
    if any(value not in {"median", "mean"} for value in numeric_imputers.values()):
        raise ValueError("numeric imputers must be median or mean")
    if any(value not in {"constant", "most_frequent"} for value in categorical_imputers.values()):
        raise ValueError("categorical imputers must be constant or most_frequent")
    excluded_by_user = list(exclude_columns or [])
    if target in excluded_by_user:
        raise ValueError("The target cannot be excluded as a feature")
    unknown = set(excluded_by_user + list(numeric_imputers) + list(categorical_imputers)) - set(df.columns)
    if unknown:
        raise ValueError(f"Unknown feature column(s): {', '.join(sorted(unknown))}")

    working = df.drop_duplicates().copy() if drop_duplicates else df.copy()
    dropped_duplicate_rows = len(df) - len(working)
    labelled = working.loc[working[target].notna()].copy()
    dropped_target_rows = len(working) - len(labelled)
    if len(labelled) < 8:
        raise ValueError("At least 8 labelled rows are required for classification")
    y = labelled[target]
    if y.nunique(dropna=True) < 2:
        raise ValueError("The target must contain at least two classes")
    if y.nunique(dropna=True) > 20:
        raise ValueError("This first classification workflow supports at most 20 target classes")
    if y.value_counts(dropna=False).min() < 3:
        raise ValueError("Each target class needs at least 3 rows for a stratified split")

    features = labelled.drop(columns=[target, *excluded_by_user])
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError as error:
        raise ValueError(f"Unable to make a stratified train/test split: {error}") from error
    # All data-dependent feature decisions use training rows only.
    numeric, categorical, excluded, warnings = _feature_groups(X_train)
    excluded = excluded_by_user + excluded
    if dropped_duplicate_rows:
        warnings.append(f"Removed {dropped_duplicate_rows} exact duplicate row(s) before splitting.")
    feature_columns = numeric + categorical
    if not feature_columns:
        raise ValueError("No supported feature columns remain after validation")
    X_train = X_train.loc[:, feature_columns]
    X_test = X_test.loc[:, feature_columns]
    if dropped_target_rows:
        warnings.append(f"Excluded {dropped_target_rows} rows with missing target values.")
    if any(pd.api.types.is_datetime64_any_dtype(features[name]) for name in features.columns):
        warnings.append("Random splitting may be unsuitable when rows are time ordered.")
    min_train_class = int(y_train.value_counts().min())
    if min_train_class < 2:
        raise ValueError("Not enough rows per target class remain for cross-validation")
    folds = min(3, min_train_class)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    candidates = [
        ("dummy", DummyClassifier(strategy="most_frequent"), False, {}),
        (
            "logistic_regression",
            LogisticRegression(max_iter=1000, random_state=random_state),
            True,
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=60, random_state=random_state, n_jobs=1),
            False,
            {"model__max_depth": [5, 10, None]},
        ),
    ]
    if model != "auto":
        candidates = [item for item in candidates if item[0] in {"dummy", model}]
    scores: list[dict[str, Any]] = []
    fitted_candidates: dict[str, Pipeline] = {}
    for name, estimator, scale, grid in candidates:
        pipe = _pipeline(
            estimator, numeric, categorical, scale=scale,
            numeric_imputers=numeric_imputers,
            categorical_imputers=categorical_imputers, scaler=scaler,
        )
        if grid:
            search = GridSearchCV(
                pipe, grid, cv=cv, scoring="f1_macro", n_jobs=1, refit=True, error_score="raise"
            )
            search.fit(X_train, y_train)
            fitted_candidates[name] = search.best_estimator_
            score = float(search.best_score_)
            best_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
        else:
            fold_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
            score = float(fold_scores.mean())
            best_params = {}
        scores.append({"model": name, "cv_score": score, "best_params": best_params})

    # Never choose the dummy baseline as the final model: it is a reference.
    ranked = sorted(scores[1:], key=lambda item: (-item["cv_score"], item["model"]))
    winning = ranked[0]
    name = winning["model"]
    fitted = fitted_candidates[name]
    predictions = fitted.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
    }
    warnings.append("Hyperparameter search is deliberately limited to three settings per trained model.")
    return TrainingResult(
        pipeline=fitted,
        target=target,
        train_rows=len(X_train),
        test_rows=len(X_test),
        model=name,
        cv_score=float(winning["cv_score"]),
        test_metrics=metrics,
        candidates=scores,
        feature_columns=feature_columns,
        excluded_columns=excluded,
        warnings=warnings,
        random_state=random_state,
        test_size=test_size,
        dropped_target_rows=dropped_target_rows,
        dataset_fingerprint=_fingerprint(labelled),
        preprocessing={
            "numeric_imputers": {name: numeric_imputers.get(name, "median") for name in numeric},
            "categorical_imputers": {name: categorical_imputers.get(name, "constant") for name in categorical},
            "scaler_for_logistic_regression": scaler,
        },
        selection_method="cross_validation" if model == "auto" else "user_override",
        dropped_duplicate_rows=dropped_duplicate_rows,
    )
