"""Checks that rendered chart values match their source rows."""

import json

import numpy as np
import pandas as pd
import pytest

import datalens as dl
from datalens.cli import main


def _chart(frame: pd.DataFrame, chart_type: str):
    recommendation = next(
        item for item in dl.analyse(frame).recommendations
        if item.chart_type == chart_type
    )
    return dl.visualise(frame, chart=recommendation)


def test_histogram_counts_only_finite_numeric_values_without_mutating_input():
    frame = pd.DataFrame({"amount": [1.0, 2.0, None, 3.0, np.inf, 2.0]})
    original = frame.copy(deep=True)

    result = _chart(frame, "histogram")

    pd.testing.assert_frame_equal(frame, original)
    assert result.plot_data["count"].sum() == 4
    assert result.plotted_rows == 4
    assert result.excluded_rows == 2
    assert sum(result.figure.data[0].y) == 4
    json.dumps(result.to_dict(), allow_nan=False)


def test_bar_chart_counts_missing_rows_and_groups_overflow_categories():
    frame = pd.DataFrame({"group": ["A", "A", "B", None, "C", "D"]})

    result = dl.visualise(frame, chart=1, max_categories=3)

    assert result.recommendation.chart_type == "bar"
    assert result.plot_data["count"].sum() == len(frame)
    assert result.plotted_rows == len(frame)
    assert any("Other" in label for label in result.plot_data["category"])
    assert result.plot_data.loc[
        result.plot_data["category"].str.contains("Other"), "count"
    ].iloc[0] == 3


def test_bar_chart_keeps_mixed_type_categories_distinct():
    frame = pd.DataFrame({
        "group": pd.Series([1, "1", None, 1], dtype=object),
    })

    result = dl.visualise(frame)

    assert result.plot_data.set_index("category")["count"].to_dict() == {
        "1 [int]": 2,
        "(Missing)": 1,
        "1 [str]": 1,
    }


def test_line_chart_states_and_calculates_aggregation_for_repeated_dates():
    frame = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01", "2026-01-02", None],
        "revenue": [10.0, 20.0, 40.0, 999.0],
    })
    line = next(
        item for item in dl.analyse(frame).recommendations
        if item.chart_type == "line"
    )

    mean_chart = dl.visualise(frame, chart=line)
    sum_chart = dl.visualise(frame, chart=line, aggregation="sum")

    assert mean_chart.plot_data["revenue"].tolist() == [15.0, 40.0]
    assert sum_chart.plot_data["revenue"].tolist() == [30.0, 40.0]
    assert mean_chart.plotted_rows == 3
    assert "Mean" in mean_chart.to_dict()["title"]
    assert "sum by exact date/time" in sum_chart.transformation


def test_scatter_uses_only_complete_numeric_pairs():
    frame = pd.DataFrame({
        "x": [1.0, 2.0, None, 4.0],
        "y": [10.0, None, 30.0, 40.0],
    })

    result = _chart(frame, "scatter")

    assert result.plot_data.to_dict("list") == {"x": [1.0, 4.0], "y": [10.0, 40.0]}
    assert result.plotted_rows == 2
    assert result.excluded_rows == 2


def test_visualise_rejects_unavailable_chart_and_refuses_overwrite(tmp_path):
    frame = pd.DataFrame({"value": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="between 1"):
        dl.visualise(frame, chart=99)
    with pytest.raises(ValueError, match="No chart recommendations"):
        dl.visualise(pd.DataFrame())

    result = dl.visualise(frame)
    destination = tmp_path / "value.html"
    result.save(destination)
    assert "plotly" in destination.read_text(encoding="utf-8").lower()
    with pytest.raises(FileExistsError):
        result.save(destination)


def test_cli_lists_and_saves_selected_chart(tmp_path, capsys):
    dataset = tmp_path / "groups.csv"
    pd.DataFrame({"group": ["A", "A", "B"]}).to_csv(dataset, index=False)
    destination = tmp_path / "groups.html"

    assert main(["visualise", str(dataset)]) == 0
    assert "1) bar" in capsys.readouterr().out
    assert main([
        "visualize", str(dataset), "--chart", "1", "--output", str(destination)
    ]) == 0
    assert destination.exists()
    assert "3 plotted / 3 source" in capsys.readouterr().out
