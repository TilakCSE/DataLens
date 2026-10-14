# DataLens

DataLens is a local Python data understanding engine. The core profiles a pandas
DataFrame, reports quality findings, and recommends charts using explicit rules.
It does not require an AI provider, database, or web service.

## Install

Python 3.11 or newer is required. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Repeat the core demo

The clean sample shows ordinary profiling and chart suggestions:

```powershell
datalens analyse examples/sales_clean.csv
```

The messy sample contains missing values and a duplicated row, which should
appear in the quality findings:

```powershell
datalens analyse examples/sales_messy.csv
```

The same report is available as machine-readable JSON:

```powershell
datalens analyse examples/sales_messy.csv --format json
```

You can also run the CLI as `python -m datalens.cli analyse ...` from the
installed environment. To verify the core, run `python -m pytest`.

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

See the [core contract](Docs/core-contract.md) for the deliverable and the
[project brief](Docs/DataLens%20-%20Major%20Project%20Brief.docx) for the wider
vision.
