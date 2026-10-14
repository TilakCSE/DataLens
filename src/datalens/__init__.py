"""DataLens public Python interface."""

from datalens.api import analyse
from datalens.models import AnalysisReport, ChartRecommendation, ColumnProfile, QualityIssue

__all__ = ["analyse", "AnalysisReport", "ChartRecommendation", "ColumnProfile", "QualityIssue"]
