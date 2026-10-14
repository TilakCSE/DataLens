# DataLens core contract

## Objective

Deliver a local, reproducible core demonstration: load a CSV or Excel file, analyse a pandas DataFrame, return a structured profile, identify basic quality issues, and recommend a few suitable chart types. This is the foundation for later web, CLI, package, and editor interfaces.

## Public interface

```python
import datalens as dl

report = dl.analyse(dataframe)
print(report.summary)
print(report.to_dict())
```

`analyse` accepts a pandas DataFrame and does not mutate it. It returns `AnalysisReport` from `datalens.models`. The CLI uses a file loader to obtain a DataFrame, then calls the same function. A future `analyze` spelling may be added as an alias, but `analyse` is the promised interface.

## Report shape

`AnalysisReport` contains row and column counts, duplicate row count, one `ColumnProfile` per column, quality issues, and chart recommendations. `to_dict()` returns JSON-serializable values. Each recommendation names its columns and gives a concrete reason; a rule-based score must not be presented as a probability.

Column kinds are `numeric`, `categorical`, `datetime`, `boolean`, or `unknown`. The source pandas dtype is reported separately. Missing values and unique counts must be calculated consistently, including empty DataFrames.

## Component boundaries

- `datalens.io`: CSV/Excel loading with clear errors for unsupported or unreadable files.
- `datalens.analysis`: column kind inference, deterministic profile, and quality checks.
- `datalens.recommend`: deterministic chart recommendations from the profile.
- `datalens.cli`: text and JSON output over the same report, plus a repeatable demo command or documented sample invocations.
- `datalens.models`: shared report classes, owned by the integration coordinator.

The integration coordinator owns `datalens.api`, `datalens.__init__`, and any necessary wiring across components.

## Core acceptance criteria

1. `dl.analyse(df)` returns correct counts and does not mutate `df`.
2. A CSV and an Excel workbook with equivalent data produce equivalent profiles.
3. Missing values, duplicate rows, numeric summaries, categorical frequencies, and date ranges are shown accurately.
4. A messy sample makes at least two meaningful quality issues visible.
5. Chart suggestions follow transparent type-based rules and include the columns and reason.
6. The CLI demonstrates the same results in readable text and JSON.
7. Focused tests pass, and a new user can repeat the demo from the README.

## Scope for this deliverable

Use in-memory pandas analysis. The web app, AI explanations, accounts, storage, and enterprise data connectors follow in later phases.
