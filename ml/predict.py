# ml/predict.py
"""
Chunk-upload prediction script.

Workflow:
 - Accept one or more raw CSV paths
 - For each CSV: load, basic_clean, lazy engineer_features
 - Align features to master schema (ml/artifacts/feature_columns.json)
 - Load saved models and predict probabilities
 - Print report: total rows, % at-risk (threshold default 0.10), top-10 risky devices

Usage:
 python -m ml.predict --csv_paths data/raw/2025-07-01.csv data/raw/2025-07-02.csv --threshold 0.10
"""

import os
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

# preprocessing functions
from ml.data_processing.clean import basic_clean
from ml.data_processing.feature_engineering import engineer_features, load_saved_schema

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")
FEATURE_SCHEMA_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.json")

MODEL_DIR = os.path.join(REPO_ROOT, "model")
MODEL_PATH_DAY = os.path.join(MODEL_DIR, "model_failure_next_day.txt")
MODEL_PATH_7D = os.path.join(MODEL_DIR, "model_failure_next_7_days.txt")

LABEL_DAY = "failure_next_day"
LABEL_7D = "failure_next_7_days"

DEFAULT_THRESHOLD = 0.10  # lowered for rare-event detection

def _load_model(path: str) -> Optional[lgb.Booster]:
    if not os.path.exists(path):
        return None
    try:
        return lgb.Booster(model_file=path)
    except Exception:
        return None

def detect_feature_list_from_schema() -> List[str]:
    schema = load_saved_schema()
    # support dict or list shapes
    if isinstance(schema, dict):
        if "features" in schema:
            return list(schema["features"])
        if "feature_columns" in schema:
            return list(schema["feature_columns"])
    if isinstance(schema, list):
        return schema
    raise RuntimeError("Unexpected schema format for feature columns")

def align_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for f in feature_cols:
        if f not in out.columns:
            out[f] = 0
    out = out[feature_cols]
    # numeric conversion & memory-friendly dtype
    for c in out.columns:
        if not np.issubdtype(out[c].dtype, np.number):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out.astype(np.float32)

def load_and_prepare_csvs(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"CSV not found: {p}")
        df = pd.read_csv(p, low_memory=False)
        df = basic_clean(df)
        df = engineer_features(df, lazy=True)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def predict_batch(df: pd.DataFrame, model_day: Optional[lgb.Booster], model_7d: Optional[lgb.Booster]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["device_id", "prob_next_day", "prob_next_7_days"])
    df = df.copy()
    if "device_id" not in df.columns:
        df["device_id"] = df.index.astype(str)

    feature_cols = detect_feature_list_from_schema()
    X = align_features(df, feature_cols)

    prob_day = model_day.predict(X) if model_day is not None else np.zeros(len(X))
    prob_7d = model_7d.predict(X) if model_7d is not None else np.zeros(len(X))

    out = pd.DataFrame({
        "device_id": df["device_id"].astype(str),
        "prob_next_day": prob_day,
        "prob_next_7_days": prob_7d
    })
    return out

def print_report(preds: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD):
    total = len(preds)
    if total == 0:
        print("No rows to report.")
        return
    preds = preds.copy()
    preds["at_risk"] = ((preds["prob_next_day"] >= threshold) | (preds["prob_next_7_days"] >= threshold)).astype(int)
    n_at_risk = int(preds["at_risk"].sum())
    pct_at_risk = n_at_risk / total * 100.0

    n_day = int((preds["prob_next_day"] >= threshold).sum())
    n_7d = int((preds["prob_next_7_days"] >= threshold).sum())

    print("=== Chunk Upload Prediction Report ===")
    print(f"Total rows/devices       : {total}")
    print(f"At risk (>= {threshold:.2f}) : {n_at_risk} ({pct_at_risk:.2f}%)")
    print(f"Predicted fail tomorrow  : {n_day} ({n_day/total*100:.2f}%)")
    print(f"Predicted fail in 7 days : {n_7d} ({n_7d/total*100:.2f}%)")
    print("\nTop 10 risky devices (by 7-day probability then 1-day):")
    top10 = preds.sort_values(["prob_next_7_days", "prob_next_day"], ascending=False).head(10)
    print(top10[["device_id", "prob_next_day", "prob_next_7_days"]].to_string(index=False))

def main():
    parser = argparse.ArgumentParser(prog="ml.predict")
    parser.add_argument("--csv_paths", nargs="+", required=True, help="One or more raw CSV paths")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    df = load_and_prepare_csvs(args.csv_paths)
    if df.empty:
        print("No data loaded from provided CSVs.")
        return

    model_day = _load_model(MODEL_PATH_DAY)
    model_7d = _load_model(MODEL_PATH_7D)
    if model_day is None and model_7d is None:
        print("No saved models found. Please run training first.")
        return

    preds = predict_batch(df, model_day, model_7d)
    print_report(preds, threshold=args.threshold)
    return preds

if __name__ == "__main__":
    main()
