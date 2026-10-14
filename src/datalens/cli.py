"""Command-line interface for the DataLens core."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from datalens.models import AnalysisReport


def format_report(report: AnalysisReport) -> str:
    """Render a readable view of the same report returned by the Python API."""
    lines = [
        "DataLens analysis",
        f"Rows: {report.row_count}",
        f"Columns: {report.column_count}",
        f"Duplicate rows: {report.duplicate_row_count}",
        "",
        "Column profiles:",
    ]

    if not report.columns:
        lines.append("  (none)")
    for column in report.columns:
        lines.append(
            f"  {column.name}: {column.kind} (source dtype: {column.source_dtype}; "
            f"non-null: {column.non_null_count}; missing: {column.missing_count}; "
            f"unique: {column.unique_count})"
        )
        for key, value in column.statistics.items():
            lines.append(f"    {key}: {_format_value(value)}")

    lines.extend(("", "Quality issues:"))
    if report.quality_issues:
        for issue in report.quality_issues:
            columns = f" [{', '.join(issue.columns)}]" if issue.columns else ""
            lines.append(f"  {issue.severity}: {issue.message} ({issue.code}){columns}")
    else:
        lines.append("  None detected")

    lines.extend(("", "Chart recommendations:"))
    if report.recommendations:
        for recommendation in report.recommendations:
            lines.append(
                f"  {recommendation.chart_type}: {recommendation.title} "
                f"[{', '.join(recommendation.columns)}]"
            )
            lines.append(f"    {recommendation.reason}")
    else:
        lines.append("  None available for the profiled column types")

    return "\n".join(lines)


def _format_value(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}: {_format_value(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="datalens", description="Profile a local dataset")
    commands = parser.add_subparsers(dest="command", required=True)
    analyse_parser = commands.add_parser("analyse", help="Analyse a CSV or Excel file")
    analyse_parser.add_argument("path", help="Path to a CSV or Excel file")
    analyse_parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    # Import here so formatting can be tested independently of file ingestion.
    from datalens.api import analyse
    from datalens.io import load_dataset

    try:
        report = analyse(load_dataset(args.path))
    except (OSError, ValueError) as error:
        print(f"DataLens: {error}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
