"""Public analysis entry point shared by Python and CLI users."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datalens.analysis import profile_dataframe
from datalens.models import AnalysisReport
from datalens.recommend import recommend_charts

if TYPE_CHECKING:
    import pandas as pd


def analyse(dataframe: pd.DataFrame) -> AnalysisReport:
    """Analyse a pandas DataFrame without modifying it."""
    columns, duplicate_count, issues = profile_dataframe(dataframe)
    return AnalysisReport(
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        duplicate_row_count=duplicate_count,
        columns=columns,
        quality_issues=issues,
        recommendations=recommend_charts(columns),
    )
