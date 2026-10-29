# DataLens

DataLens is a local Python data understanding and training tool. It profiles a
pandas DataFrame, reports quality findings, recommends charts using explicit
rules, and can train a tabular classification pipeline. It does not require an
AI provider, database, or web service.

## Install

Python 3.11 or newer is required. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On macOS or Linux, use `.venv/bin/python -m pip install -e ".[dev]"`.

## Repeat the core demo

The clean sample shows ordinary profiling and chart suggestions:

```powershell
.\.venv\Scripts\datalens.exe analyse examples/sales_clean.csv
```

The messy sample contains missing values and a duplicated row, which should
appear in the quality findings:

```powershell
.\.venv\Scripts\datalens.exe analyse examples/sales_messy.csv
```

The same report is available as machine-readable JSON:

```powershell
.\.venv\Scripts\datalens.exe analyse examples/sales_messy.csv --format json
```

`analyse` reports the data as it is. Its result includes column profiles,
calculated quality issues, cleaning suggestions, and chart ideas; it does not
change the DataFrame or ask questions. For an interactive walkthrough, run:

```powershell
.\.venv\Scripts\datalens.exe guide examples/titanic_train_demo.csv --target Survived --output runs/titanic-guided.joblib
```

The guide shows chart ideas and then asks whether to continue to training. It
offers per-column missing-value choices (recommended imputer, alternative, or
skip the column), duplicate handling when needed, scaler and model choices,
and a final save/skip choice. Enter `x` at any prompt to exit. Fitted imputers,
encoders, and scalers remain inside the training pipeline; reviewing a cleaning
suggestion does not edit the source file.

## Larger public dataset demo

The repository also includes a local Titanic training CSV used for the project
presentation. It has 891 rows and 12 columns. The CLI reports 177 missing Age
values, 687 missing Cabin values, 2 missing Embarked values, and no exact
duplicate rows:

```powershell
.\.venv\Scripts\datalens.exe analyse examples/titanic_train_demo.csv
```

See [the dataset source note](examples/TITANIC_DEMO_SOURCE.md) for its public
mirror, Kaggle competition reference, and checksum. Chart suggestions are
type-based; low-cardinality numeric columns such as `Survived` and `Pclass`
can receive count charts. Use `--verbose` for every column statistic and chart
idea, or `--format json` for the full structured report.

## Train a first classification model

The first training workflow accepts a labelled CSV or Excel file and an explicit
target column. It compares a dummy baseline, logistic regression, and random
forest using training-only cross-validation. Numeric imputation and scaling,
categorical imputation and encoding, and the chosen estimator are saved together
in a scikit-learn pipeline. The selected model is evaluated once on a separate
holdout set.

```powershell
.\.venv\Scripts\datalens.exe train examples/titanic_train_demo.csv --target Survived --output titanic-model.joblib
```

This creates `titanic-model.joblib` and a neighbouring `titanic-model.json`
manifest. The CLI prints the model choice, candidate cross-validation scores,
held-out accuracy, balanced accuracy, and macro F1. Use `--format json` for the
full training report. Choose a new output path for each run; the CLI will not
overwrite an existing artifact.

From Python, the same training engine is available through the public API:

```python
import pandas as pd
import datalens as dl

df = pd.read_csv("examples/titanic_train_demo.csv")
result = dl.train(df, target="Survived")
print(result.to_dict())
result.save("titanic-model.joblib")
predictions = result.pipeline.predict(df.drop(columns="Survived").head())
```

This version handles local tabular classification with random stratified splits.
It automatically chooses among two trained candidates when using `dl.train` or
the non-interactive `train` CLI command. The guided path offers model and
preprocessing choices. Regression, grouped or time-aware splitting, and deep
learning are future work. Review excluded features and any possible target
leakage before using the saved model. Model artifacts are Python pickle-based:
load only artifacts you trust, using compatible package versions.

You can also run the CLI as `python -m datalens.cli analyse ...` from the
installed environment. To verify the core on Windows, run
`.\.venv\Scripts\python.exe -m pytest`.

## Python interface

```python
import pandas as pd
import datalens as dl

df = pd.read_csv("examples/sales_messy.csv")
report = dl.analyse(df)
print(report.summary)
print(report.to_dict())
for suggestion in report.cleaning_suggestions:
    print(suggestion.column, suggestion.recommended_action, suggestion.reason)
```

The CLI calls the same `analyse` function and displays the same report. Chart
recommendations are type-based suggestions with stated reasons. They are not
claims that a chart reveals a causal relationship.

`dl.analyse(df)` is the original non-interactive analysis API. Use
`dl.train(df, target="...")` for reproducible automatic classification, or
`dl.guide(df, target="...")` for the optional question-by-question terminal
workflow. These are separate functions so analysis never unexpectedly starts
model training.

The original DataLens vision extends beyond the current core: rendered and
interactive charts, relationship and anomaly analysis, natural-language
questions, dashboards, and more data sources are still planned. The current
analysis API provides measured profiles, quality findings, and chart
recommendations; it does not claim to deliver those later features yet.

See the [core contract](Docs/core-contract.md) and
[training phase contract](Docs/automl-phase-contract.md) for the deliverables,
the [agent workflow](Docs/agent-workflow.md) for the worktree setup, and the
[project brief](Docs/DataLens%20-%20Major%20Project%20Brief.docx) for the wider vision.
