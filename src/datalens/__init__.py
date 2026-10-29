"""DataLens public Python interface."""

from datalens.api import analyse, train
from datalens.models import AnalysisReport, ChartRecommendation, ColumnProfile, QualityIssue

__all__ = ["analyse", "train", "AnalysisReport", "ChartRecommendation", "ColumnProfile", "QualityIssue"]
