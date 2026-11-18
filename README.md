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

The guide shows chart ideas, offers to save one as an interactive HTML chart,
and then asks whether to continue to training. It
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

## Inspect class errors

The training result now includes held-out precision, recall, F1, and row count
for every target class, plus an actual-versus-predicted confusion matrix. The
JSON run manifest contains the full data. The ordinary CLI output highlights
the class with the lowest held-out recall; `--diagnostics` prints every class.

Run the Python API demo and save an interactive matrix:

```powershell
.\.venv\Scripts\python.exe examples\class_diagnostics_demo.py --matrix-output runs\charts\python-class-matrix.html
```

Run the same phase through the CLI and save the fitted model, JSON report, and
matrix chart:

```powershell
.\.venv\Scripts\datalens.exe train examples\titanic_train_demo.csv --target Survived --output runs\models\titanic-diagnostics.joblib --diagnostics --matrix-output runs\charts\cli-class-matrix.html
```

The Python demo accepts another CSV and `--target` to inspect a different
classification task. For the saved five-class Heart Disease file, use
`datasets\external\heart_cleveland.csv --target num`. Choose new output names
on later runs. The matrix rows are actual classes and columns are predicted
classes; the diagonal counts correct predictions. See the
[diagnostics method](Docs/class-diagnostics.md) for interpretation and limits.

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

## Render an interactive chart

Try the current visualization phase from Python. This script prints the ranked
ideas, renders the first one, shows the exact plotted data, and saves an HTML
chart you can open in a browser:

```powershell
.\.venv\Scripts\python.exe examples\visualization_ranking_demo.py --output runs\charts\python-demo.html
```

The CSV is optional and defaults to `examples/sales_clean.csv`. Add `--chart N`
to render a specific original chart number. Choose a new output filename when
you run it again; DataLens does not overwrite existing charts.

List the top five chart ideas calculated by `analyse`, show every idea with
`--all`, and save a selected chart by its stable 1-based number:

```powershell
.\.venv\Scripts\datalens.exe visualise examples/sales_clean.csv
.\.venv\Scripts\datalens.exe visualise examples/sales_clean.csv --all
.\.venv\Scripts\datalens.exe visualise examples/sales_clean.csv --chart 5 --output runs/charts/sales-over-time.html
```

The `visualize` spelling also works. The saved HTML is interactive and
self-contained; choose a new output name for each run because DataLens will not
overwrite one. `runs/charts/` is ignored by Git. Histograms count finite values,
bars count categories including missing values, scatter plots use rows with
both axes present, and line charts aggregate valid rows by exact date/time.
The line default is `mean`; use `--aggregation sum` when a total makes more
sense. `--max-categories` controls how many bars are shown before the remaining
categories are grouped into `Other`.

From Python, the same renderer accepts a chart number or a recommendation from
the analysis report:

```python
import pandas as pd
import datalens as dl

df = pd.read_csv("examples/sales_clean.csv")
report = dl.analyse(df)
print(report.summary)
ranked = dl.rank_charts(df)
print(ranked[0].number, ranked[0].ranking_reason)
chart = dl.visualise(df, chart=ranked[0].number)
print(chart.to_dict())  # source rows, plotted rows, and transformation
print(chart.plot_data)  # exact values shown in the figure
chart.save("runs/charts/sales-over-time.html")
```

The first visualization engine renders the four current recommendation types:
histogram, category count bar, time line, and numeric scatter. Recommendations
are ranked by a documented rule: show one high-coverage example of each
available chart type, then the remaining ideas by source-column completeness.
The ranking is a navigation aid, not a learned usefulness or confidence score.
The original `report.recommendations` order and its chart numbers remain stable.
See [the chart ranking method](Docs/chart-ranking.md) for the exact rule and its
limits.

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

The original DataLens vision extends beyond the current core: deeper
relationship and anomaly analysis, natural-language questions, dashboards,
and more data sources are still planned. The current analysis API provides
measured profiles, quality findings, and chart recommendations, and the
visualization API renders the supported chart types locally.

For five public-data checks and reusable local demo files, see the
[validation dataset guide](Docs/validation-datasets.md) and
[measured validation results](Docs/external-validation-results.md). The CSVs,
saved models, and detailed run output live in the Git-ignored
`datasets/external/` folder.

See the [core contract](Docs/core-contract.md) and
[training phase contract](Docs/automl-phase-contract.md) for the deliverables,
the [agent workflow](Docs/agent-workflow.md) for the worktree setup, and the
[project brief](Docs/DataLens%20-%20Major%20Project%20Brief.docx) for the wider vision.
