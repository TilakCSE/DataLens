# DataLens external dataset validation — 2 November 2026

This is an engineering validation of the current Python API and CLI on five
distinct public classification datasets. It checks whether the existing
workflow accepts different table shapes and whether saved pipelines reload.
It is **not** a benchmark proving model quality or product generality.

## Method

- Source CSVs, conversion notes, targets, hashes, and reproduction commands are
  documented in [validation-datasets.md](validation-datasets.md). Dataset bytes
  and run artifacts stay in the Git-ignored `datasets/external/` folder.
- Every file was profiled with `dl.analyse(df)`. The CLI JSON report was parsed
  and compared with the corresponding Python API report.
- Every file was trained with `dl.train(df, target=...)` using default settings:
  random seed 42, a stratified 80/20 split, training-only cross-validation with
  up to three folds, and macro F1 for selecting between logistic regression and
  random forest against a dummy baseline.
- Each fitted pipeline and JSON manifest was saved under
  `datasets/external/models/`. The pipeline was reloaded and its predictions
  on the first five source rows were compared with the in-memory pipeline.
- Environment: Python 3.13.7, pandas 3.0.6, scikit-learn 1.9.1, NumPy 2.5.4,
  joblib 1.6.0. Exact measured output is in the ignored
  `datasets/external/validation_results.json`.

## Results

| Dataset | Rows trained | Missing-value columns | Chart ideas | Selected model | Held-out macro F1 | Software checks |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Titanic | 891 | 3 | 9 | Random forest | 0.778 | Passed |
| Breast Cancer Wisconsin (Diagnostic) | 569 | 0 | 32 | Logistic regression | 0.962 | Passed |
| Mushroom | 8,124 | 1 | 23 | Random forest | 1.000 | Passed |
| Heart Disease (Cleveland, original five-class target) | 303 | 2 | 15 | Random forest | 0.192 | Passed; poor model result |
| Adult (fixed 5,000-row sample) | 5,000 | 3 | 16 | Logistic regression | 0.811 | Passed |

For all five: analysis completed, CLI/API analysis reports matched, training
completed, model artifacts reloaded, and the sampled predictions matched. The
full 32,561-row Adult source was also profiled successfully; the report found
24 exact duplicate rows and took about 1.5 seconds on this machine. Training
used the saved 5,000-row sample to keep the demo quick.

## What the tests exposed

The original Heart Disease target has five classes with counts 164, 55, 36,
35, and 13. Its held-out macro F1 of 0.192 is low despite the software run
succeeding. The [UCI dataset description](https://archive.ics.uci.edu/dataset/45/heart+disease)
also describes a binary presence-versus-absence formulation. On a separately
saved binary variant of the same 303 rows, DataLens selected logistic
regression and measured held-out macro F1 of 0.852. These are different tasks;
the binary score does not fix or supersede the five-class result.

The tool currently reports overall and balanced accuracy and macro F1, but
does not surface per-class recall or confusion matrices. Those would help a
user see why a small class performs poorly. The recommendation engine also
offers many chart ideas on wide numeric tables (32 on Breast Cancer), so chart
ranking and rendering are worthwhile next steps.

One random split per dataset cannot establish stable performance. This run
does not test grouped or time-ordered data, calibration, fairness, or whether
the output models are appropriate for real-world decisions.
