# DataLens

DataLens is a Python data understanding engine. Its first deliverable will accept a pandas DataFrame or a CSV/Excel file, profile the data, report quality issues, and recommend appropriate visualizations using explainable rules.

The planned Python interface is:

```python
import pandas as pd
import datalens as dl

df = pd.read_csv("sales.csv")
report = dl.analyse(df)
print(report.summary)
```

The CLI will use the same engine:

```powershell
datalens analyse sales.csv
```

Implementation is in progress. See [the core contract](Docs/core-contract.md) for the agreed interface and acceptance criteria. The [project brief](Docs/DataLens%20-%20Major%20Project%20Brief.docx) describes the larger product vision.
