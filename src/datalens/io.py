"""Load local tabular files for analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV or modern Excel workbook into a new DataFrame.

    The first worksheet is used for Excel files, matching pandas' default.
    Parsing is left to pandas so source values are not silently cleaned or
    changed by DataLens.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    if not source.is_file():
        raise ValueError(f"Dataset path is not a file: {source}")

    extension = source.suffix.lower()
    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xlsm": pd.read_excel,
    }
    reader = readers.get(extension)
    if reader is None:
        raise ValueError(
            f"Unsupported dataset format '{extension or '<none>'}' for {source}. "
            "Use .csv, .xlsx, or .xlsm."
        )

    try:
        return reader(source)
    except (OSError, ValueError, ImportError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read dataset {source}: {exc}") from exc
