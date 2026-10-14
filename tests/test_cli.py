import json
import sys
import types

from datalens.cli import format_report, main
from datalens.models import AnalysisReport, ChartRecommendation, ColumnProfile, QualityIssue


def sample_report() -> AnalysisReport:
    return AnalysisReport(
        row_count=3,
        column_count=1,
        duplicate_row_count=1,
        columns=(ColumnProfile("amount", "numeric", "float64", 2, 1, 2,
                               {"mean": 12.5}),),
        quality_issues=(QualityIssue("missing_values", "1 value is missing", "warning",
                                     ("amount",)),),
        recommendations=(ChartRecommendation("histogram", "Distribution of amount",
                                              ("amount",), "amount is numeric"),),
    )


def test_text_format_contains_profile_quality_and_recommendations():
    output = format_report(sample_report())
    assert "Rows: 3" in output
    assert "Duplicate rows: 1" in output
    assert "amount: numeric" in output
    assert "mean: 12.5" in output
    assert "1 value is missing" in output
    assert "histogram: Distribution of amount" in output


def test_text_format_handles_empty_report():
    output = format_report(AnalysisReport(0, 0, 0, ()))
    assert "Rows: 0" in output
    assert "Column profiles:\n  (none)" in output
    assert "None detected" in output


def test_cli_uses_loader_and_api_for_json(monkeypatch, capsys):
    loaded = object()
    calls = []
    io_module = types.ModuleType("datalens.io")
    io_module.load_dataset = lambda path: calls.append(path) or loaded
    monkeypatch.setitem(sys.modules, "datalens.io", io_module)
    monkeypatch.setattr("datalens.api.analyse", lambda frame: sample_report() if frame is loaded else None)

    assert main(["analyse", "example.csv", "--format", "json"]) == 0
    assert calls == ["example.csv"]
    result = json.loads(capsys.readouterr().out)
    assert result["row_count"] == 3
    assert result["columns"][0]["name"] == "amount"


def test_cli_reports_loader_error(monkeypatch, capsys):
    io_module = types.ModuleType("datalens.io")
    io_module.load_dataset = lambda path: (_ for _ in ()).throw(ValueError("unsupported file"))
    monkeypatch.setitem(sys.modules, "datalens.io", io_module)

    assert main(["analyse", "bad.txt"]) == 1
    assert "unsupported file" in capsys.readouterr().err
