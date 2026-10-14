"""Explainable chart suggestions based only on profiled column kinds."""

from __future__ import annotations

from datalens.models import ChartRecommendation, ColumnProfile


def recommend_charts(columns: tuple[ColumnProfile, ...]) -> tuple[ChartRecommendation, ...]:
    """Suggest chart types in a stable order from the supplied column profiles.

    These are type-based suggestions, not assessments of statistical confidence.
    The caller can decide which suggestions to display or plot.
    """
    recommendations: list[ChartRecommendation] = []
    numeric = [column for column in columns if column.kind == "numeric"]
    categorical = [column for column in columns if column.kind == "categorical"]
    dates = [column for column in columns if column.kind == "datetime"]

    for column in numeric:
        recommendations.append(
            ChartRecommendation(
                chart_type="histogram",
                title=f"Distribution of {column.name}",
                columns=(column.name,),
                reason=f"{column.name} is numeric, so a histogram can show its distribution.",
            )
        )

    for column in categorical:
        recommendations.append(
            ChartRecommendation(
                chart_type="bar",
                title=f"Counts by {column.name}",
                columns=(column.name,),
                reason=f"{column.name} is categorical, so a bar chart can compare category counts.",
            )
        )

    if dates and numeric:
        date, measure = dates[0], numeric[0]
        recommendations.append(
            ChartRecommendation(
                chart_type="line",
                title=f"{measure.name} over {date.name}",
                columns=(date.name, measure.name),
                reason=(
                    f"{date.name} is datetime and {measure.name} is numeric, "
                    "so a line chart can show values across time."
                ),
            )
        )

    if len(numeric) >= 2:
        first, second = numeric[:2]
        recommendations.append(
            ChartRecommendation(
                chart_type="scatter",
                title=f"{second.name} versus {first.name}",
                columns=(first.name, second.name),
                reason=(
                    f"{first.name} and {second.name} are numeric, "
                    "so a scatter plot can show their relationship."
                ),
            )
        )

    return tuple(recommendations)
