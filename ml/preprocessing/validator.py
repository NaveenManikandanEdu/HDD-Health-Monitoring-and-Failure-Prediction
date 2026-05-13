# ml/preprocessing/validator.py
import json
import pandas as pd

SCHEMA_PATH = "ml/artifacts/feature_columns.json"
LABELS = ["failure", "failure_next_day", "failure_next_7_days"]


def preflight_one(path):
    df = pd.read_parquet(path)
    print(f"\n{path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)["features"]

    missing = [c for c in schema if c not in df.columns]
    print(f"Missing features: {len(missing)}")

    for l in LABELS:
        print(f"{l}: sum={df[l].sum() if l in df else 'MISSING'}")
