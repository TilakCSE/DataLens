"""Public analysis entry point shared by Python and CLI users."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from datalens.analysis import profile_dataframe
from datalens.cleaning import suggest_cleaning
from datalens.models import AnalysisReport, ChartRecommendation
from datalens.recommend import recommend_charts

if TYPE_CHECKING:
    import pandas as pd
    from datalens.training import TrainingResult
    from datalens.rendering import RenderedChart


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
        cleaning_suggestions=suggest_cleaning(columns, duplicate_count),
    )


def visualise(
    dataframe: pd.DataFrame, chart: int | ChartRecommendation = 1, *,
    aggregation: str = "mean", max_categories: int = 20,
) -> RenderedChart:
    """Render one numbered chart idea from this DataFrame's analysis report.

    Chart numbers start at 1, matching the CLI. The returned chart can be
    inspected in Python or saved as a self-contained interactive HTML file.
    """
    from datalens.rendering import render_recommendation

    report = analyse(dataframe)
    if not report.recommendations:
        raise ValueError("No chart recommendations are available for this dataset")
    if isinstance(chart, bool):
        raise ValueError("chart must be a 1-based recommendation number or ChartRecommendation")
    if isinstance(chart, int):
        if not 1 <= chart <= len(report.recommendations):
            raise ValueError(f"chart must be between 1 and {len(report.recommendations)}")
        recommendation = report.recommendations[chart - 1]
    elif isinstance(chart, ChartRecommendation) and chart in report.recommendations:
        recommendation = chart
    else:
        raise ValueError("chart must be a recommendation from dl.analyse(df)")
    return render_recommendation(
        dataframe, recommendation, aggregation=aggregation,
        max_categories=max_categories,
    )


def train(
    dataframe: pd.DataFrame,
    *,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    numeric_imputers: dict[str, str] | None = None,
    categorical_imputers: dict[str, str] | None = None,
    exclude_columns: list[str] | None = None,
    scaler: str = "standard",
    model: str = "auto",
    drop_duplicates: bool = False,
) -> TrainingResult:
    """Train and evaluate a local tabular classifier on labelled data."""
    from datalens.training import train_classifier

    return train_classifier(
        dataframe, target, test_size=test_size, random_state=random_state,
        numeric_imputers=numeric_imputers, categorical_imputers=categorical_imputers,
        exclude_columns=exclude_columns, scaler=scaler, model=model,
        drop_duplicates=drop_duplicates,
    )


def guide(
    dataframe: pd.DataFrame, *, target: str | None = None,
    output: str | Path | None = None,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> TrainingResult | None:
    """Run the optional interactive analysis-to-training walkthrough."""
    from datalens.guided import guide_dataframe

    return guide_dataframe(
        dataframe, target=target, output=output,
        input_fn=input_fn, output_fn=output_fn,
    )
