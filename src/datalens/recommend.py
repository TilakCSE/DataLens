"""Explainable chart suggestions based on the deterministic column profiles."""

from __future__ import annotations

from datalens.models import ChartRecommendation, ColumnProfile


def _is_identifier(column: ColumnProfile) -> bool:
    """Skip near-unique fields that read as record identifiers."""
    raw_name = column.name.strip()
    name = raw_name.casefold().replace("_", "").replace(" ", "")
    identifier_name = (
        name in {"id", "identifier"}
        or raw_name.endswith(("Id", "ID", "_id"))
        or name.endswith("identifier")
    )
    return identifier_name and column.non_null_count > 0 and (
        column.unique_count / column.non_null_count >= 0.9
    )


def _is_low_cardinality_number(column: ColumnProfile) -> bool:
    return (
        column.non_null_count >= 20
        and 1 < column.unique_count <= 10
        and column.unique_count / column.non_null_count <= 0.2
    )


def _is_high_cardinality_category(column: ColumnProfile) -> bool:
    return column.unique_count > 30 and column.unique_count / max(column.non_null_count, 1) > 0.5


def recommend_charts(columns: tuple[ColumnProfile, ...]) -> tuple[ChartRecommendation, ...]:
    """Suggest chart types in a stable order from the supplied column profiles.

    These are profile-based suggestions, not assessments of statistical confidence.
    The caller can decide which suggestions to display or plot.
    """
    recommendations: list[ChartRecommendation] = []
    numeric = [
        column for column in columns
        if column.kind == "numeric" and not _is_identifier(column)
    ]
    continuous = [column for column in numeric if not _is_low_cardinality_number(column)]
    discrete = [column for column in numeric if _is_low_cardinality_number(column)]
    categorical = [
        column for column in columns
        if column.kind == "categorical"
        and not _is_identifier(column)
        and not _is_high_cardinality_category(column)
    ]
    dates = [column for column in columns if column.kind == "datetime"]

    for column in continuous:
        recommendations.append(
            ChartRecommendation(
                chart_type="histogram",
                title=f"Distribution of {column.name}",
                columns=(column.name,),
                reason=f"{column.name} is numeric, so a histogram can show its distribution.",
            )
        )

    for column in discrete:
        recommendations.append(
            ChartRecommendation(
                chart_type="bar",
                title=f"Counts by {column.name}",
                columns=(column.name,),
                reason=(
                    f"{column.name} has {column.unique_count} distinct numeric values "
                    f"across {column.non_null_count} non-missing rows, so a bar chart "
                    "can compare their counts."
                ),
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

    if dates and continuous:
        date, measure = dates[0], continuous[0]
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

    if len(continuous) >= 2:
        first, second = continuous[:2]
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
