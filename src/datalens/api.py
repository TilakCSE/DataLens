"""Public analysis entry point shared by Python and CLI users."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datalens.analysis import profile_dataframe
from datalens.models import AnalysisReport
from datalens.recommend import recommend_charts

if TYPE_CHECKING:
    import pandas as pd
    from datalens.training import TrainingResult


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


def train(
    dataframe: pd.DataFrame,
    *,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    """Train and evaluate a local tabular classifier on labelled data."""
    from datalens.training import train_classifier

    return train_classifier(
        dataframe, target, test_size=test_size, random_state=random_state
    )
