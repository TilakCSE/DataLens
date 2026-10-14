"""Tests for local dataset ingestion."""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from datalens.io import load_dataset


def test_csv_and_excel_load_equivalent_data(tmp_path):
    expected = pd.DataFrame(
        {
            "region": ["East", "West", "East"],
            "sales": [10.0, 20.5, None],
            "units": [2, 3, 4],
        }
    )
    csv_path = tmp_path / "sales.csv"
    excel_path = tmp_path / "sales.xlsx"
    expected.to_csv(csv_path, index=False)
    expected.to_excel(excel_path, index=False)

    assert_frame_equal(load_dataset(csv_path), expected)
    assert_frame_equal(load_dataset(str(excel_path)), expected)


def test_load_dataset_reports_missing_path(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset(missing)


def test_load_dataset_rejects_unsupported_format(tmp_path):
    text_path = tmp_path / "data.txt"
    text_path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        load_dataset(text_path)


def test_load_dataset_reports_unreadable_file(tmp_path):
    broken = tmp_path / "broken.xlsx"
    broken.write_text("not an Excel workbook", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read dataset"):
        load_dataset(broken)


def test_load_dataset_rejects_directory(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        load_dataset(tmp_path)
