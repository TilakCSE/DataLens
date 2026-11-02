"""Save five public classification datasets as local, ignored demo CSVs.

Run from the repository root with ``python scripts/fetch_validation_datasets.py``.
The UCI source archives are downloaded only when the matching local CSVs are
absent, unless --force is used. No dataset bytes are committed to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "datasets" / "external"
UCI = "https://archive.ics.uci.edu/static/public"

MUSHROOM_COLUMNS = [
    "poisonous", "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
    "gill_attachment", "gill_spacing", "gill_size", "gill_color", "stalk_shape",
    "stalk_root", "stalk_surface_above_ring", "stalk_surface_below_ring",
    "stalk_color_above_ring", "stalk_color_below_ring", "veil_type", "veil_color",
    "ring_number", "ring_type", "spore_print_color", "population", "habitat",
]
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def _uci_frame(dataset_id: int, archive_name: str, member: str, **read_options: object) -> tuple[pd.DataFrame, str]:
    url = f"{UCI}/{dataset_id}/{archive_name}.zip"
    request = urllib.request.Request(url, headers={"User-Agent": "DataLens-validation/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        with archive.open(member) as source:
            frame = pd.read_csv(source, **read_options)
    return frame, url


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, target: str, source: str, note: str) -> dict[str, object]:
    frame = pd.read_csv(path)
    return {
        "file": path.name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "target": target,
        "source": source,
        "note": note,
        "sha256": _sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Download and rewrite existing CSVs")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    specifications = [
        (
            "breast_cancer.csv", "Diagnosis", 17, "breast+cancer+wisconsin+diagnostic",
            "wdbc.data", {"header": None, "names": ["PatientId", "Diagnosis", *load_breast_cancer().feature_names]},
            "Original UCI WDBC rows; PatientId retained to test identifier handling.",
        ),
        (
            "mushroom.csv", "poisonous", 73, "mushroom", "agaricus-lepiota.data",
            {"header": None, "names": MUSHROOM_COLUMNS, "na_values": ["?"]},
            "Original UCI categorical rows; '?' is parsed as a missing value.",
        ),
        (
            "heart_cleveland.csv", "num", 45, "heart+disease", "processed.cleveland.data",
            {"header": None, "names": HEART_COLUMNS, "na_values": ["?"]},
            "Processed Cleveland subset; original 0-4 target classes are retained.",
        ),
        (
            "adult.csv", "income", 2, "adult", "adult.data",
            {"header": None, "names": ADULT_COLUMNS, "na_values": ["?"], "skipinitialspace": True},
            "Original UCI adult.data training portion; adult.test is not combined.",
        ),
    ]
    manifest: dict[str, object] = {"datasets": []}
    for filename, target, dataset_id, archive_name, member, options, note in specifications:
        destination = OUTPUT / filename
        source = f"{UCI}/{dataset_id}/{archive_name}.zip ({member})"
        if args.force or not destination.exists():
            frame, _ = _uci_frame(dataset_id, archive_name, member, **options)
            frame.to_csv(destination, index=False)
            print(f"Saved {destination} ({len(frame)} rows)")
        else:
            print(f"Using existing {destination}")
        manifest["datasets"].append(_record(destination, target, source, note))

    heart_binary = OUTPUT / "heart_binary.csv"
    if args.force or not heart_binary.exists():
        heart = pd.read_csv(OUTPUT / "heart_cleveland.csv")
        heart["heart_disease"] = (heart.pop("num") > 0).astype(int)
        heart.to_csv(heart_binary, index=False)
        print(f"Saved {heart_binary} (binary target variant)")
    manifest["datasets"].append(_record(
        heart_binary, "heart_disease", "heart_cleveland.csv",
        "Derived variant: num > 0 is presence and num = 0 is absence; same source dataset.",
    ))

    adult_demo = OUTPUT / "adult_demo_5000.csv"
    if args.force or not adult_demo.exists():
        adult = pd.read_csv(OUTPUT / "adult.csv")
        adult.sample(n=5000, random_state=42).sort_index().to_csv(adult_demo, index=False)
        print(f"Saved {adult_demo} (5000 rows)")
    manifest["datasets"].append(_record(
        adult_demo, "income", "adult.csv",
        "Deterministic 5000-row sample (pandas sample, random_state=42) for quick training demos.",
    ))

    titanic_source = ROOT / "examples" / "titanic_train_demo.csv"
    titanic_destination = OUTPUT / "titanic.csv"
    if args.force or not titanic_destination.exists():
        shutil.copyfile(titanic_source, titanic_destination)
        print(f"Saved {titanic_destination}")
    manifest["datasets"].append(_record(
        titanic_destination, "Survived", str(titanic_source.relative_to(ROOT)),
        "Existing DataLens public Titanic demo copy; see examples/TITANIC_DEMO_SOURCE.md.",
    ))
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {OUTPUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
