"""Run the chart ranking and rendering API on a local CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import datalens as dl


def main() -> None:
    parser = argparse.ArgumentParser(description="Try DataLens chart ranking in Python")
    parser.add_argument(
        "csv", nargs="?", type=Path,
        default=Path(__file__).with_name("sales_clean.csv"),
        help="CSV to analyse (default: examples/sales_clean.csv)",
    )
    parser.add_argument(
        "--chart", type=int,
        help="Original chart number to render (default: first ranked idea)",
    )
    parser.add_argument("--output", type=Path, help="Save an interactive .html chart")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    report = dl.analyse(frame)
    ranked = dl.rank_charts(frame)
    print(f"Data: {report.row_count} rows, {report.column_count} columns")
    print(f"Chart ideas: {len(ranked)}; showing up to five ranked quick picks")
    for item in ranked[:5]:
        print(f"  {item.number}) {item.recommendation.title}")
        print(f"     {item.ranking_reason}")

    if not ranked:
        print("No chart is available for these columns.")
        return

    number = args.chart if args.chart is not None else ranked[0].number
    chart = dl.visualise(frame, chart=number)
    facts = chart.to_dict()
    print(f"\nRendered chart {number}: {facts['title']}")
    print(f"Rows plotted: {facts['plotted_rows']}/{facts['source_rows']}")
    print(f"Method: {facts['transformation']}")
    print("Plotted data (first 10 points):")
    print(chart.plot_data.head(10).to_string(index=False))
    if args.output is not None:
        print(f"Saved interactive chart: {chart.save(args.output)}")


if __name__ == "__main__":
    main()
