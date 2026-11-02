"""Run DataLens profiling and classification on the five local public datasets.

Results and saved model artifacts stay inside the Git-ignored datasets/external
folder. Run fetch_validation_datasets.py first.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import datalens as dl


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "external"
CASES = [
    ("titanic", "titanic.csv", "Survived"),
    ("breast_cancer", "breast_cancer.csv", "Diagnosis"),
    ("mushroom", "mushroom.csv", "poisonous"),
    ("heart_cleveland", "heart_cleveland.csv", "num"),
    ("adult", "adult_demo_5000.csv", "income"),
]


def main() -> None:
    if not DATA.is_dir():
        raise SystemExit("No local datasets found. Run scripts/fetch_validation_datasets.py first.")
    findings: dict[str, object] = {"cases": []}
    model_dir = DATA / "models"
    model_dir.mkdir(exist_ok=True)
    for name, filename, target in CASES:
        source = DATA / filename
        if not source.is_file():
            raise SystemExit(f"Missing {source}; run scripts/fetch_validation_datasets.py first.")
        frame = pd.read_csv(source)
        case: dict[str, object] = {
            "name": name, "file": filename, "target": target,
            "rows": len(frame), "columns": len(frame.columns),
            "target_counts": {
                str(label): int(count)
                for label, count in frame[target].value_counts(dropna=False).items()
            },
        }
        start = time.perf_counter()
        try:
            analysis = dl.analyse(frame)
            case["analysis"] = {
                "status": "passed",
                "seconds": round(time.perf_counter() - start, 3),
                "summary": analysis.summary,
                "missing_columns": {
                    column.name: column.missing_count
                    for column in analysis.columns if column.missing_count
                },
                "quality_codes": [issue.code for issue in analysis.quality_issues],
                "chart_types": [item.chart_type for item in analysis.recommendations],
                "cleaning_suggestions": [
                    {"column": item.column, "action": item.recommended_action}
                    for item in analysis.cleaning_suggestions
                ],
            }
            cli = subprocess.run(
                [sys.executable, "-m", "datalens.cli", "analyse", str(source), "--format", "json"],
                capture_output=True, text=True, check=True,
            )
            if json.loads(cli.stdout) != json.loads(json.dumps(analysis.to_dict())):
                raise AssertionError("CLI JSON differs from the Python API analysis report")
            case["cli_analysis_matches_api"] = True
        except Exception as error:
            case["analysis"] = {
                "status": "failed", "type": type(error).__name__, "error": str(error),
            }
        start = time.perf_counter()
        try:
            training = dl.train(frame, target=target)
            artifact = training.save(model_dir / f"{name}.joblib")
            restored = joblib.load(artifact)
            sample = frame.drop(columns=[target]).head(5)
            np.testing.assert_array_equal(
                training.pipeline.predict(sample), restored.predict(sample)
            )
            case["training"] = {
                "status": "passed",
                "seconds": round(time.perf_counter() - start, 3),
                "report": training.to_dict(),
                "saved_model": str(artifact.relative_to(ROOT)),
                "reload_predictions_match": True,
            }
        except Exception as error:
            case["training"] = {
                "status": "failed", "seconds": round(time.perf_counter() - start, 3),
                "type": type(error).__name__, "error": str(error),
            }
        findings["cases"].append(case)
        print(
            f"{name}: analyse={case['analysis']['status']} "
            f"train={case['training']['status']} "
            f"cli_parity={case.get('cli_analysis_matches_api', False)}"
        )
    full_adult = pd.read_csv(DATA / "adult.csv")
    start = time.perf_counter()
    full_adult_report = dl.analyse(full_adult)
    findings["adult_full_analysis"] = {
        "rows": len(full_adult),
        "seconds": round(time.perf_counter() - start, 3),
        "summary": full_adult_report.summary,
    }
    heart_binary = pd.read_csv(DATA / "heart_binary.csv")
    start = time.perf_counter()
    binary_result = dl.train(heart_binary, target="heart_disease")
    findings["heart_binary_followup"] = {
        "rows": len(heart_binary),
        "seconds": round(time.perf_counter() - start, 3),
        "target_counts": {
            str(label): int(count)
            for label, count in heart_binary["heart_disease"].value_counts().items()
        },
        "model": binary_result.model,
        "test_metrics": binary_result.test_metrics,
    }
    destination = DATA / "validation_results.json"
    destination.write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {destination}")
    failures = [
        case["name"] for case in findings["cases"]
        if case["analysis"]["status"] != "passed"
        or case["training"]["status"] != "passed"
        or not case.get("cli_analysis_matches_api", False)
    ]
    if failures:
        raise SystemExit(f"Validation failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
