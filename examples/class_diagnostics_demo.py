"""Run the public DataLens training API and inspect held-out class errors."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import datalens as dl


def main() -> None:
    parser = argparse.ArgumentParser(description="Try DataLens class diagnostics")
    parser.add_argument(
        "csv", nargs="?", type=Path,
        default=Path(__file__).with_name("titanic_train_demo.csv"),
        help="CSV to train on (default: examples/titanic_train_demo.csv)",
    )
    parser.add_argument("--target", default="Survived", help="Column to predict")
    parser.add_argument("--model-output", type=Path, help="Save fitted .joblib and JSON report")
    parser.add_argument("--matrix-output", type=Path, help="Save interactive matrix .html")
    args = parser.parse_args()

    if args.model_output is not None and args.model_output.suffix.lower() != ".joblib":
        parser.error("model output must end in .joblib")
    if args.matrix_output is not None and args.matrix_output.suffix.lower() != ".html":
        parser.error("matrix output must end in .html")
    if args.model_output is not None and (
        args.model_output.exists() or args.model_output.with_suffix(".json").exists()
    ):
        parser.error(f"model output already exists: {args.model_output}")
    if args.matrix_output is not None and args.matrix_output.exists():
        parser.error(f"matrix output already exists: {args.matrix_output}")

    frame = pd.read_csv(args.csv)
    result = dl.train(frame, target=args.target)
    report = result.to_dict()
    diagnostics = report["class_diagnostics"]
    print(f"Target: {report['target']} | Model: {report['model']}")
    print(f"Training rows: {report['train_rows']} | Held-out rows: {report['test_rows']}")
    print("Held-out metrics: " + ", ".join(
        f"{name}={value:.3f}" for name, value in report["test_metrics"].items()
    ))
    print("Per-class results (train rows, test rows, precision, recall, F1):")
    for train_count, item in zip(
        diagnostics["train_support"], diagnostics["per_class"], strict=True,
    ):
        print(
            f"  {item['label']}: {train_count}, {item['support']}, "
            f"{item['precision']:.3f}, {item['recall']:.3f}, {item['f1']:.3f}"
        )
    print("Confusion matrix (rows actual, columns predicted):")
    print("  Labels: " + ", ".join(map(str, diagnostics["labels"])))
    for label, counts in zip(
        diagnostics["labels"], diagnostics["confusion_matrix"], strict=True,
    ):
        print(f"  {label}: " + " ".join(map(str, counts)))

    if args.model_output is not None:
        print(f"Saved fitted pipeline: {result.save(args.model_output)}")
    if args.matrix_output is not None:
        print(f"Saved interactive matrix: {result.save_confusion_matrix(args.matrix_output)}")


if __name__ == "__main__":
    main()
