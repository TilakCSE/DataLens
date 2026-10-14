"""Checks the complete public path used in the DataLens demo."""

import json
from pathlib import Path

import pandas as pd

import datalens as dl
from datalens.cli import main
from datalens.io import load_dataset


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_public_api_profiles_without_mutating_input():
    frame = load_dataset(EXAMPLES / "sales_clean.csv")
    before = frame.copy(deep=True)

    report = dl.analyse(frame)

    pd.testing.assert_frame_equal(frame, before)
    assert report.row_count == len(frame)
    assert report.column_count == len(frame.columns)
    assert any(column.kind == "numeric" for column in report.columns)
    assert any(item.chart_type == "line" for item in report.recommendations)
    json.dumps(report.to_dict(), allow_nan=False)


def test_messy_demo_reveals_quality_issues():
    report = dl.analyse(load_dataset(EXAMPLES / "sales_messy.csv"))
    codes = {issue.code for issue in report.quality_issues}
    assert report.duplicate_row_count > 0
    assert "missing_values" in codes
    assert "duplicate_rows" in codes


def test_csv_and_excel_produce_equivalent_analysis(tmp_path):
    frame = pd.DataFrame({"region": ["east", "west", "east"], "amount": [10, 20, 30]})
    csv_path = tmp_path / "equivalent.csv"
    excel_path = tmp_path / "equivalent.xlsx"
    frame.to_csv(csv_path, index=False)
    frame.to_excel(excel_path, index=False)

    csv_report = dl.analyse(load_dataset(csv_path))
    excel_report = dl.analyse(load_dataset(excel_path))

    assert csv_report.to_dict() == excel_report.to_dict()


def test_real_cli_json_uses_same_report(capsys):
    path = EXAMPLES / "sales_messy.csv"
    expected = dl.analyse(load_dataset(path)).to_dict()

    assert main(["analyse", str(path), "--format", "json"]) == 0

    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(expected))
