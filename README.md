# DataLens

DataLens is a local Python data understanding engine. The core profiles a pandas
DataFrame, reports quality findings, and recommends charts using explicit rules.
It does not require an AI provider, database, or web service.

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

## Larger public dataset demo

The repository also includes a local Titanic training CSV used for the project
presentation. It has 891 rows and 12 columns. The CLI reports 177 missing Age
values, 687 missing Cabin values, 2 missing Embarked values, and no exact
duplicate rows:

```powershell
.\.venv\Scripts\datalens.exe analyse examples/titanic_train_demo.csv
```

See [the dataset source note](examples/TITANIC_DEMO_SOURCE.md) for its public
mirror, Kaggle competition reference, and checksum. These are type-based chart
suggestions; numeric-coded categories such as `Survived` and `Pclass` currently
receive generic numeric suggestions.

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
```

The CLI calls the same `analyse` function and displays the same report. Chart
recommendations are type-based suggestions with stated reasons. They are not
claims that a chart reveals a causal relationship.

See the [core contract](Docs/core-contract.md) for the deliverable,
the [agent workflow](Docs/agent-workflow.md) for the worktree setup, and the
[project brief](Docs/DataLens%20-%20Major%20Project%20Brief.docx) for the wider vision.
