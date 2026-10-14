"""Shared result objects used by the Python API and CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    kind: str
    source_dtype: str
    non_null_count: int
    missing_count: int
    unique_count: int
    statistics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityIssue:
    code: str
    message: str
    severity: str
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChartRecommendation:
    chart_type: str
    title: str
    columns: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class AnalysisReport:
    row_count: int
    column_count: int
    duplicate_row_count: int
    columns: tuple[ColumnProfile, ...]
    quality_issues: tuple[QualityIssue, ...] = ()
    recommendations: tuple[ChartRecommendation, ...] = ()

    @property
    def summary(self) -> dict[str, int]:
        return {
            "rows": self.row_count,
            "columns": self.column_count,
            "duplicate_rows": self.duplicate_row_count,
            "quality_issues": len(self.quality_issues),
            "recommendations": len(self.recommendations),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report once statistics are normalized."""
        return asdict(self)
