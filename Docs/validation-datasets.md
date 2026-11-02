# External validation datasets

Five distinct public classification datasets are kept locally under
`datasets/external/`. That folder is ignored by Git so the CSVs and generated
run artifacts remain available on this computer without entering the repository.

Recreate the files with an internet connection:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_validation_datasets.py
```

Use `--force` only when you intend to replace existing local copies. The script
writes an ignored `manifest.json` containing the exact file hashes, row counts,
target names, and source archive locations. `adult_demo_5000.csv` is a fixed
sample of `adult.csv`; it is a convenience copy, not a sixth source dataset.
`heart_binary.csv` is an additional target variant of the same Heart Disease
source: it maps `num = 0` to absence and `num > 0` to presence, following the
binary task described by UCI. The original five-class file is kept unchanged.

| Dataset | Local demo file | Target | Why it is included |
| --- | --- | --- | --- |
| [Titanic](https://www.kaggle.com/competitions/titanic/data) | `titanic.csv` | `Survived` | Mixed columns, missing values, identifier and high-cardinality text. [Copy provenance](../examples/TITANIC_DEMO_SOURCE.md). |
| [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | `breast_cancer.csv` | `Diagnosis` | Mostly clean numeric features and an ID column. |
| [Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom) | `mushroom.csv` | `poisonous` | Entirely categorical features and a missing-value marker. |
| [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) | `heart_cleveland.csv` | `num` | Small mixed-type table with missing entries and multiple target classes. |
| [Adult](https://archive.ics.uci.edu/dataset/2/adult) | `adult_demo_5000.csv` | `income` | Larger mixed numeric/categorical data with missing entries; the full `adult.csv` is also saved. |

These are engineering checks, not estimates of general model performance.
Medical and income datasets are used only to check software behavior; the
resulting models must not be used for decisions about people.

Run one dataset through the existing CLI:

```powershell
.\.venv\Scripts\datalens.exe analyse datasets\external\mushroom.csv
.\.venv\Scripts\datalens.exe train datasets\external\mushroom.csv --target poisonous --output datasets\external\mushroom-model.joblib
```

Run profiling, training, model saving, and reload verification across all five:

```powershell
.\.venv\Scripts\python.exe scripts\validate_external_datasets.py
```

The ignored `datasets/external/validation_results.json` records the measured
findings. Saved pipelines go to the ignored `datasets/external/models/` folder.
