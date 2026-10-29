import json
import sys
import types
from pathlib import Path

from datalens.cli import format_report, format_training_result, main
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


def test_compact_text_format_shows_highlights_without_every_statistic():
    output = format_report(sample_report())
    assert "Rows: 3" in output
    assert "Duplicate rows: 1" in output
    assert "Column types: numeric 1" in output
    assert "mean: 12.5" not in output
    assert "1 value is missing" in output
    assert "histogram: Distribution of amount" in output
    assert "Why: amount is numeric" in output
    assert "--verbose" in output


def test_verbose_text_format_shows_all_column_statistics():
    output = format_report(sample_report(), verbose=True)
    assert "amount: numeric" in output
    assert "mean: 12.5" in output
    assert "Column profiles:" in output


def test_text_format_handles_empty_report():
    output = format_report(AnalysisReport(0, 0, 0, ()))
    assert "Rows: 0" in output
    assert "Column types: none" in output
    assert "None detected" in output


def test_compact_text_keeps_every_error_and_bounds_other_highlights():
    issues = tuple(
        QualityIssue(f"info_{index}", f"info {index}", "info") for index in range(8)
    ) + (QualityIssue("invalid", "critical problem", "error"),)
    recommendations = tuple(
        ChartRecommendation("bar", f"Chart {index}", ("amount",), "counted rows")
        for index in range(8)
    )
    report = AnalysisReport(3, 1, 0, sample_report().columns, issues, recommendations)
    output = format_report(report)
    assert "critical problem" in output
    assert "info 0" in output and "info 3" in output
    assert "info 4" not in output
    assert "... 4 more" in output
    assert "Chart 2" in output and "Chart 3" not in output
    assert "... 5 more" in output


def sample_training_result():
    class TrainingResult:
        def to_dict(self):
            return {
                "task": "classification",
                "target": "Survived",
                "train_rows": 8,
                "test_rows": 2,
                "model": "LogisticRegression",
                "selection_metric": "f1_macro",
                "cv_score": 0.75,
                "test_metrics": {"accuracy": 0.5, "balanced_accuracy": 0.5, "f1_macro": 0.5},
                "candidates": [
                    {"model": "DummyClassifier", "cv_score": 0.4},
                    {"model": "LogisticRegression", "cv_score": 0.75},
                ],
                "feature_columns": ["Age", "Sex"],
                "excluded_columns": ["PassengerId"],
                "warnings": ["PassengerId looks like an identifier"],
                "random_state": 42,
                "test_size": 0.2,
                "dropped_target_rows": 0,
            }

        def save(self, path):
            saved = Path(path)
            saved.write_text("test artifact", encoding="utf-8")
            saved.with_suffix(".json").write_text("{}", encoding="utf-8")
            return saved

    return TrainingResult()


def test_training_text_summarizes_measured_scores():
    output = format_training_result(sample_training_result(), saved_path=Path("model.joblib"))
    assert "Selected model: LogisticRegression" in output
    assert "cross-validation f1_macro (0.750)" in output
    assert "Held-out test: accuracy 0.500" in output
    assert "PassengerId looks like an identifier" in output
    assert "Saved pipeline: model.joblib" in output


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


def test_cli_train_saves_artifact_and_prints_summary(monkeypatch, capsys, tmp_path):
    loaded = object()
    calls = []
    io_module = types.ModuleType("datalens.io")
    io_module.load_dataset = lambda path: loaded
    training_module = types.ModuleType("datalens.training")
    training_module.train_classifier = lambda df, target, **kwargs: (
        calls.append((df, target, kwargs)) or sample_training_result()
    )
    monkeypatch.setitem(sys.modules, "datalens.io", io_module)
    monkeypatch.setitem(sys.modules, "datalens.training", training_module)
    output = tmp_path / "titanic-model.joblib"

    assert main(["train", "titanic.csv", "--target", "Survived", "--output", str(output)]) == 0
    assert calls == [(loaded, "Survived", {"test_size": 0.2, "random_state": 42})]
    assert output.exists()
    assert "Selected model: LogisticRegression" in capsys.readouterr().out


def test_cli_train_json_is_full_result(monkeypatch, capsys, tmp_path):
    io_module = types.ModuleType("datalens.io")
    io_module.load_dataset = lambda path: object()
    training_module = types.ModuleType("datalens.training")
    training_module.train_classifier = lambda df, target, **kwargs: sample_training_result()
    monkeypatch.setitem(sys.modules, "datalens.io", io_module)
    monkeypatch.setitem(sys.modules, "datalens.training", training_module)

    assert main(["train", "titanic.csv", "--target", "Survived", "--output", str(tmp_path / "model.joblib"), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model"] == "LogisticRegression"
    assert data["candidates"][0]["model"] == "DummyClassifier"


def test_cli_train_will_not_overwrite_existing_artifact(monkeypatch, capsys, tmp_path):
    artifact = tmp_path / "model.joblib"
    artifact.write_text("existing", encoding="utf-8")
    io_module = types.ModuleType("datalens.io")
    io_module.load_dataset = lambda path: object()
    monkeypatch.setitem(sys.modules, "datalens.io", io_module)

    assert main(["train", "titanic.csv", "--target", "Survived", "--output", str(artifact)]) == 1
    assert artifact.read_text(encoding="utf-8") == "existing"
    assert "already exists" in capsys.readouterr().err
