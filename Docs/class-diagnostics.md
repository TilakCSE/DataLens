# Classification diagnostics

DataLens calculates class diagnostics from the same untouched holdout
predictions used for overall accuracy, balanced accuracy, and macro F1. The
target is supplied by the user. Model selection and preprocessing still use
training data only.

`TrainingResult.class_diagnostics` and its `to_dict()` form contain:

- `labels`: fixed class order for every following array and matrix.
- `train_support`: rows of each class in the training split.
- `per_class`: held-out precision, recall, F1, and support for each class.
- `confusion_matrix`: rows are actual classes; columns are predictions. A
  diagonal cell counts correct predictions. Every row totals that class's
  held-out support.
- `zero_division`: zero means an undefined precision or recall is reported as
  0, matching the metric calculation policy.

The CLI shows the lowest observed class recall in its compact summary. Add
`--diagnostics` to see every class and the matrix. Add `--matrix-output FILE.html`
to save a standalone interactive heatmap. The JSON manifest saved beside the
model contains all diagnostic counts and metrics. From Python, inspect
`result.class_diagnostics` or call `result.save_confusion_matrix(path)`.

These values describe one stratified random holdout split. A class with few
held-out rows can have an unstable recall estimate; comparing tasks or models
requires matching data and evaluation procedures. A confusion matrix explains
which classes the fitted model confused, but does not identify a causal reason.

## Five-class Heart Disease check, 18 November 2026

Using the saved Cleveland file with the original `num` target and the default
split, DataLens selected random forest. On 61 held-out rows, accuracy was 0.541
and macro F1 was 0.192. Class 0 recall was 32/33 (0.970); classes 2, 3, and 4
had zero correct predictions out of 7, 7, and 3 held-out rows respectively.
This explains why accuracy alone gives an incomplete picture of this run.

Reproduce the report with the Python demo:

```powershell
.\.venv\Scripts\python.exe examples\class_diagnostics_demo.py datasets\external\heart_cleveland.csv --target num
```

The dataset source and reproduction steps are in
[validation-datasets.md](validation-datasets.md). These numbers describe this
one split and should not be presented as a stable clinical performance estimate.
