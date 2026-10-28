# DataLens AutoML phase contract (proposal)

## Purpose

Extend the existing deterministic `dl.analyse(df)` foundation into a local,
guided workflow that takes a labelled tabular dataset to a saved, reusable
predictive pipeline without requiring the user to write model-training code.
Keep `dl.analyse(df)` unchanged as the primary profiling interface. The new
workflow must work without a hosted AI provider or vector database.

The first delivered task is **supervised tabular classification**. Regression,
deep learning, graphical/web interfaces, and other data modalities follow only
after the classification workflow is correct and independently evaluated.

## Product boundary

Input: a pandas DataFrame or a CSV/Excel file plus a user-confirmed target
column. Output: a fitted preprocessing-and-estimator pipeline, an evaluation
report, a reproducible configuration, and a local model artifact. Deployment
and serving are outside this phase.

An unlabelled dataset is eligible for `analyse`, but not this supervised
training flow. The system may suggest plausible targets but must never silently
choose one. The user must confirm the target and intended prediction task.

## Proposed public interfaces

Python: a separate `dl.train(df, target=..., ...)` interface returning a
structured `TrainingResult`, to be finalized before implementation. CLI:
`datalens train <path> --target <column>` with an optional guided mode that
lets the user select among valid, explained choices. Both interfaces call the
same training engine and produce the same report.

The CLI must display a recommended choice with its reason. “Recommended” is a
rule-based default for the current dataset and estimator, not a claim that it
will achieve the best possible score. Users can override it. A non-interactive
mode must allow reproducible runs and automated tests.

## First vertical slice: classification

1. Validate the target, feature columns, row count, missing labels, class
   counts, and data types. Flag likely identifiers and potential leakage
   columns for user review.
2. Create an untouched holdout test split before fitting preprocessing.
   Stratify when valid; reject or explain datasets too small for a meaningful
   split. Grouped and time-ordered data require dedicated split strategies
   rather than a random split.
3. Build a scikit-learn `ColumnTransformer` inside a `Pipeline`: numeric
   imputation and optional scaling; categorical imputation and one-hot
   encoding. Fit all learned transformations on training folds only.
4. Train a `DummyClassifier` baseline and a small bounded set of candidates,
   initially logistic regression, random forest, and optionally KNN. Candidate
   availability and preprocessing depend on the dataset. Do not force scaling
   where it is not useful.
5. Select and tune using cross-validation on the training partition only,
   subject to an explicit time/iteration budget. Evaluate the selected
   pipeline exactly once on the untouched holdout set.
6. Report class balance, split method, selected features, preprocessing,
   candidates and search space, selection metric, cross-validation scores,
   holdout metrics, and limitations. Do not invent confidence percentages.
7. Save the **entire fitted pipeline** plus a JSON manifest with package
   versions, random seed, target, feature schema, transformations, metrics,
   dataset fingerprint, and training configuration. Reload the artifact and
   verify predictions on held-out rows match before declaring success.

Titanic is the first end-to-end example (`Survived` as target). Its numeric-
coded categories, missing values, and identifier-like fields make it useful
for exposing mistakes. Add at least one independent classification dataset
before claiming generality.

## Subsequent slices

1. **Regression:** target validation, appropriate metrics (MAE/RMSE/R²),
   baseline, model candidates, and saved full pipeline.
2. **Guided CLI:** arrow-key choices or numbered fallback, with explanations
   for task, split, metric, estimator, imputer, encoder, and scaler. Offer
   advanced controls only when relevant; safe defaults need a one-command run.
3. **GUI/web:** use the same training engine and report. Do not implement
   independent model-selection logic in the interface.
4. **Deep learning:** choose a concrete modality and task first. A tabular MLP
   may be benchmarked as one candidate; image, text, and sequence models need
   separate ingestion, compute, validation, and persistence designs. Do not
   assume deep learning is preferable for every dataset.

## Research question and evaluation

The project cannot claim novelty merely for being free or no-code: AWS
SageMaker Canvas and open-source AutoML tools already cover much of the
workflow. A defensible investigation is whether **profile-informed,
explainable recommendations and validation safeguards** reduce invalid or
misleading models on messy datasets while remaining usable with no code.

Benchmark DataLens against a manual scikit-learn baseline and at least one
free AutoML baseline (for example FLAML or AutoGluon) under comparable time
and hardware budgets. Measure task completion, invalid configurations and
data-leakage cases, model quality on held-out data, runtime, memory use, and
reproducibility. Report where the baseline is better.

## Acceptance criteria for the next implementation milestone

- Existing `analyse` API and CLI behavior remain intact.
- Titanic classification can run locally from API and CLI without an AI key.
- Missing values and mixed numeric/categorical types work in both training
  and prediction.
- No imputer, encoder, or scaler is fitted on the holdout test data.
- A dummy baseline and at least two candidate estimators are compared fairly.
- Results name the actual selection and holdout metrics and explain the choice.
- Saved pipeline reloads and predicts without repeating manual preprocessing.
- Focused tests cover target problems, leakage prevention, serialization,
  and CLI/API report parity.

## Implementation references

- [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: mixed-type ColumnTransformer example](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)
- [scikit-learn: model persistence and pickle/joblib risks](https://scikit-learn.org/stable/model_persistence.html)

Model artifacts are trusted local outputs. Loading an untrusted pickle or
joblib file can execute code; cross-version loading is unsupported. Record
package versions and do not auto-load user-supplied model files.
