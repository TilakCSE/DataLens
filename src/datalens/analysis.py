"""Deterministic, in-memory profiling of pandas DataFrames."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import math
import re
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from datalens.models import ColumnProfile, QualityIssue


_DATE_TEXT = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ]\S+)?$")


def _is_missing(value: Any) -> bool:
    """Treat only scalar pandas missing markers as missing."""
    try:
        result = pd.isna(value)
        return isinstance(result, (bool, int)) and bool(result)
    except (TypeError, ValueError):
        return False


def _family(value: Any) -> str:
    if isinstance(value, (bool,)) or type(value).__name__ == "bool_":
        return "boolean"
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return "datetime"
    if isinstance(value, str):
        return "text"
    if isinstance(value, (int, float, complex)) or type(value).__module__.startswith("numpy") and isinstance(value, __import__("numbers").Number):
        return "numeric"
    return "other"


def _kind(series: pd.Series, values: list[Any]) -> tuple[str, bool]:
    if is_bool_dtype(series.dtype):
        return "boolean", False
    if is_datetime64_any_dtype(series.dtype):
        return "datetime", False
    if is_numeric_dtype(series.dtype):
        return "numeric", False
    if not values:
        return "unknown", False

    families = {_family(value) for value in values}
    mixed = len(families) > 1
    if families == {"boolean"}:
        return "boolean", False
    if families == {"numeric"}:
        return "numeric", False
    if families == {"datetime"}:
        return "datetime", False
    if families == {"text"} and all(_DATE_TEXT.match(value) for value in values):
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        if not pd.isna(parsed).any():
            return "datetime", False
    return "categorical", mixed


def _key(value: Any) -> Any:
    """Make even nested object cells hashable, with their types preserved."""
    if isinstance(value, dict):
        return ("dict", tuple(sorted((repr(_key(k)), _key(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_key(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted((_key(item) for item in value), key=repr)))
    if _is_missing(value):
        return ("missing",)
    try:
        hash(value)
    except TypeError:
        return (type(value).__name__, repr(value))
    return (type(value).__name__, value)


def _json_value(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    return repr(value)


def _statistics(kind: str, values: list[Any]) -> dict[str, Any]:
    if not values:
        return {}
    if kind == "numeric":
        numbers = pd.to_numeric(pd.Series(values), errors="coerce")
        numbers = numbers[numbers.notna()]
        if numbers.empty:
            return {}
        return {name: _json_value(getattr(numbers, name)()) for name in ("min", "max", "mean", "median", "std")}
    if kind == "datetime":
        dates = pd.to_datetime(values, errors="coerce", format="mixed")
        dates = dates[~pd.isna(dates)]
        if len(dates) == 0:
            return {}
        return {"min": _json_value(min(dates)), "max": _json_value(max(dates))}
    if kind in {"categorical", "boolean"}:
        counts = Counter(_key(value) for value in values)
        examples = {_key(value): _json_value(value) for value in values}
        ordered = sorted(counts, key=lambda key: (-counts[key], repr(key)))
        return {"top_values": [{"value": examples[key], "count": counts[key]} for key in ordered[:10]]}
    return {}


def profile_dataframe(frame: pd.DataFrame) -> tuple[tuple[ColumnProfile, ...], int, tuple[QualityIssue, ...]]:
    """Return column profiles, duplicate count, and calculated quality findings.

    The input is only read. Duplicate count means rows beyond their first occurrence.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("profile_dataframe expects a pandas DataFrame")

    profiles: list[ColumnProfile] = []
    issues: list[QualityIssue] = []
    for position in range(frame.shape[1]):
        series = frame.iloc[:, position]
        name = str(frame.columns[position])
        values = [value for value in series.tolist() if not _is_missing(value)]
        kind, mixed = _kind(series, values)
        missing_count = len(series) - len(values)
        profile = ColumnProfile(
            name=name,
            kind=kind,
            source_dtype=str(series.dtype),
            non_null_count=len(values),
            missing_count=missing_count,
            unique_count=len({_key(value) for value in values}),
            statistics=_statistics(kind, values),
        )
        profiles.append(profile)
        if missing_count:
            issues.append(QualityIssue("missing_values", f"Column '{name}' has {missing_count} missing value(s).", "warning", (name,)))
        if mixed:
            issues.append(QualityIssue("mixed_types", f"Column '{name}' contains mixed value types.", "warning", (name,)))

    seen: set[Any] = set()
    duplicate_count = 0
    for row in frame.itertuples(index=False, name=None):
        key = tuple(_key(value) for value in row)
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)
    if duplicate_count:
        issues.append(QualityIssue("duplicate_rows", f"Dataset has {duplicate_count} duplicate row(s).", "warning"))
    return tuple(profiles), duplicate_count, tuple(issues)
