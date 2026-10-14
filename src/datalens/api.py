"""Public analysis entry point; completed during component integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datalens.models import AnalysisReport

if TYPE_CHECKING:
    import pandas as pd


def analyse(dataframe: pd.DataFrame) -> AnalysisReport:
    """Analyse a pandas DataFrame without modifying it."""
    raise NotImplementedError("The DataLens core components are being implemented")
