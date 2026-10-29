"""Transparent data-quality suggestions derived from the analysis report."""

from __future__ import annotations

import re

from datalens.models import CleaningSuggestion, ColumnProfile


_ID_NAME = re.compile(r"(^id$|[_\s-]id$|^id[_\s-]|identifier$|^index$)", re.I)


def looks_like_identifier(name: object) -> bool:
    """Recognize common ID names, including camel-case names like PassengerId."""
    text = str(name).strip()
    return bool(_ID_NAME.search(text) or re.search(r"[a-z0-9](?:Id|ID)$", text))


def _missing_suggestion(column: ColumnProfile) -> CleaningSuggestion:
    count = column.missing_count
    if column.non_null_count == 0:
        return CleaningSuggestion(
            column.name, "missing_values", "exclude_column", (),
            f"All {count} values in {column.name} are missing, so an imputer cannot learn from this column.",
        )
    if column.kind == "numeric":
        return CleaningSuggestion(
            column.name, "missing_values", "median", ("mean", "exclude_column"),
            f"{column.name} has {count} missing value(s); median is less affected by extreme values than mean.",
        )
    return CleaningSuggestion(
        column.name, "missing_values", "constant", ("most_frequent", "exclude_column"),
        f"{column.name} has {count} missing value(s); a separate missing category preserves that information.",
    )


def suggest_cleaning(
    columns: tuple[ColumnProfile, ...], duplicate_row_count: int
) -> tuple[CleaningSuggestion, ...]:
    """Recommend reviewable actions without changing or fitting the input data."""
    imputations: list[CleaningSuggestion] = []
    exclusions: list[CleaningSuggestion] = []
    for column in columns:
        if column.non_null_count == 0:
            if column.missing_count:
                exclusions.append(_missing_suggestion(column))
        elif looks_like_identifier(column.name):
            exclusions.append(CleaningSuggestion(
                column.name, "identifier", "exclude_column", ("review_column",),
                f"{column.name} looks like a row identifier, which usually does not generalize to new rows.",
            ))
        elif column.kind == "categorical" and (
            column.unique_count > 50
            or column.unique_count / column.non_null_count > 0.8
        ):
            exclusions.append(CleaningSuggestion(
                column.name, "high_cardinality", "exclude_column", ("feature_engineering",),
                f"{column.name} has {column.unique_count} distinct values; this first trainer does not encode high-cardinality text.",
            ))
        elif column.missing_count:
            imputations.append(_missing_suggestion(column))
    suggestions = imputations
    if duplicate_row_count:
        suggestions.append(
            CleaningSuggestion(
                None, "duplicate_rows", "drop_duplicates", ("keep_duplicates",),
                f"{duplicate_row_count} row(s) repeat an earlier full row and could cross a random split.",
            )
        )
    return tuple(suggestions + exclusions)
