"""Command-line interface for the DataLens core."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from datalens.models import AnalysisReport


def format_report(report: AnalysisReport, *, verbose: bool = False) -> str:
    """Render compact highlights or the full report without changing its facts."""
    kinds: dict[str, int] = {}
    for column in report.columns:
        kinds[column.kind] = kinds.get(column.kind, 0) + 1

    lines = [
        "DataLens | analysis",
        f"Rows: {report.row_count}  Columns: {report.column_count}  "
        f"Duplicate rows: {report.duplicate_row_count}",
        "Column types: "
        + (", ".join(f"{kind} {count}" for kind, count in kinds.items()) if kinds else "none"),
    ]

    issues = report.quality_issues
    visible_issues = issues if verbose else tuple(
        issue for issue in issues if issue.severity.lower() == "error"
    ) + tuple(
        issue for issue in issues if issue.severity.lower() != "error"
    )[:4]
    lines.extend(("", f"Quality issues ({len(issues)}):"))
    if not issues:
        lines.append("  None detected")
    else:
        for issue in visible_issues:
            columns = f" [{', '.join(issue.columns)}]" if issue.columns else ""
            lines.append(f"  {issue.severity.upper()}: {issue.message}{columns}")
        if len(visible_issues) < len(issues):
            lines.append(f"  ... {len(issues) - len(visible_issues)} more")

    recommendations = report.recommendations
    visible_recommendations = recommendations if verbose else recommendations[:3]
    lines.extend(("", f"Chart ideas ({len(recommendations)}):"))
    if not recommendations:
        lines.append("  None available for the profiled column types")
    else:
        for recommendation in visible_recommendations:
            lines.append(
                f"  {recommendation.chart_type}: {recommendation.title} "
                f"[{', '.join(recommendation.columns)}]"
            )
            lines.append(f"    Why: {recommendation.reason}")
        if len(visible_recommendations) < len(recommendations):
            lines.append(f"  ... {len(recommendations) - len(visible_recommendations)} more")

    if verbose:
        lines.extend(("", "Column profiles:"))
        if not report.columns:
            lines.append("  (none)")
        for column in report.columns:
            lines.append(
                f"  {column.name}: {column.kind} | dtype={column.source_dtype} | "
                f"non-null={column.non_null_count} | missing={column.missing_count} | "
                f"unique={column.unique_count}"
            )
            for key, value in column.statistics.items():
                lines.append(f"    {key}: {_format_value(value)}")
    suggestions = report.cleaning_suggestions
    if suggestions:
        visible = suggestions if verbose else suggestions[:3]
        lines.extend(("", f"Cleaning suggestions ({len(suggestions)}):"))
        for suggestion in visible:
            label = suggestion.column or "dataset"
            lines.append(f"  {label}: {suggestion.recommended_action} - {suggestion.reason}")
        if len(visible) < len(suggestions):
            lines.append(f"  ... {len(suggestions) - len(visible)} more")
    if not verbose:
        lines.extend(("", "Use --verbose for all details, --format json for the full report, or guide for choices."))

    return "\n".join(lines)


def _format_value(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}: {_format_value(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def format_training_result(result: object, *, saved_path: Path | None = None) -> str:
    """Summarize measured training results without calling scores probabilities."""
    data = result.to_dict()
    metrics = data["test_metrics"]
    lines = [
        "DataLens | training",
        f"Task: {data['task']}  Target: {data['target']}",
        f"Data: {data['train_rows']} train / {data['test_rows']} test rows  "
        f"Features: {len(data['feature_columns'])} used, {len(data['excluded_columns'])} excluded",
        f"Selected model: {data['model']}",
        (
            f"  Why: user selected; training cross-validation {data['selection_metric']} "
            f"was {data['cv_score']:.3f}"
            if data.get("selection_method") == "user_override"
            else f"  Why: highest training cross-validation {data['selection_metric']} "
                 f"({data['cv_score']:.3f}) among the tested candidates"
        ),
        "Held-out test: " + ", ".join(f"{name} {score:.3f}" for name, score in metrics.items()),
        "",
        "Candidates (training cross-validation):",
    ]
    for candidate in data["candidates"]:
        lines.append(
            f"  {candidate['model']}: {data['selection_metric']} {candidate['cv_score']:.3f}"
        )
    preprocessing = data.get("preprocessing")
    if preprocessing:
        numeric_methods = ", ".join(sorted(set(preprocessing["numeric_imputers"].values()))) or "none"
        categorical_methods = ", ".join(sorted(set(preprocessing["categorical_imputers"].values()))) or "none"
        scaling = (
            preprocessing["scaler_for_logistic_regression"]
            if data["model"] == "logistic_regression" else "not used by selected model"
        )
        lines.append(
            f"Preprocessing: numeric {numeric_methods}; categorical {categorical_methods}; scaling {scaling}"
        )
    if data["dropped_target_rows"]:
        lines.append(f"Rows without a target excluded: {data['dropped_target_rows']}")
    if data.get("dropped_duplicate_rows"):
        lines.append(f"Exact duplicate rows removed: {data['dropped_duplicate_rows']}")
    if data["excluded_columns"]:
        lines.append("Excluded features: " + ", ".join(data["excluded_columns"]))
    if data["warnings"]:
        lines.extend(("", "Review:"))
        lines.extend(f"  - {warning}" for warning in data["warnings"])
    if saved_path is not None:
        lines.extend(("", f"Saved pipeline: {saved_path}", f"Run manifest: {saved_path.with_suffix('.json')}"))
    return "\n".join(lines)


def _training_output_path(dataset_path: str, target: str, output: str | None) -> Path:
    if output is not None:
        return Path(output)
    dataset = Path(dataset_path)
    safe_target = "".join(character if character.isalnum() or character in "-_" else "_" for character in target)
    return dataset.with_name(f"{dataset.stem}-{safe_target}.joblib")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="datalens", description="Analyse or train on a local dataset")
    commands = parser.add_subparsers(dest="command", required=True)
    analyse_parser = commands.add_parser("analyse", help="Analyse a CSV or Excel file")
    analyse_parser.add_argument("path", help="Path to a CSV or Excel file")
    analyse_parser.add_argument("--format", choices=("text", "json"), default="text")
    analyse_parser.add_argument("--verbose", action="store_true", help="Show every column and suggestion")
    train_parser = commands.add_parser("train", help="Train a tabular classification model")
    train_parser.add_argument("path", help="Path to a CSV or Excel file")
    train_parser.add_argument("--target", required=True, help="Column to predict")
    train_parser.add_argument("--output", help="Model artifact path (.joblib)")
    train_parser.add_argument("--format", choices=("text", "json"), default="text")
    guide_parser = commands.add_parser("guide", help="Walk through analysis, cleaning choices, and optional training")
    guide_parser.add_argument("path", help="Path to a CSV or Excel file")
    guide_parser.add_argument("--target", help="Proposed target column; you will confirm it")
    guide_parser.add_argument("--output", help="Suggested model artifact path (.joblib)")
    args = parser.parse_args(argv)

    # Lazy imports keep report formatting usable without loading data or ML libraries.
    from datalens.io import load_dataset

    try:
        dataset = load_dataset(args.path)
        if args.command == "guide":
            from datalens.guided import guide_dataframe

            guide_dataframe(
                dataset, target=args.target, output=args.output, source_name=args.path
            )
            return 0
        if args.command == "analyse":
            from datalens.api import analyse

            report = analyse(dataset)
        else:
            destination = _training_output_path(args.path, args.target, args.output)
            if destination.suffix.lower() != ".joblib":
                raise ValueError("model output must end in .joblib")
            if destination.exists() or destination.with_suffix(".json").exists():
                raise ValueError(f"model output already exists: {destination}")
            from datalens.training import train_classifier

            report = train_classifier(dataset, args.target, test_size=0.2, random_state=42)
            saved_path = report.save(destination)
    except (OSError, ValueError) as error:
        print(f"DataLens: {error}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    elif args.command == "train":
        print(format_training_result(report, saved_path=saved_path))
    else:
        print(format_report(report, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
